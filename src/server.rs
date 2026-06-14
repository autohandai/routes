use crate::{
    accounting::{BudgetAccounting, BudgetReservation, BudgetUsageSnapshot},
    classifier::{JudgeMetricsSnapshot, SmartClassifier},
    config::{BudgetConfig, RouterConfig},
    openapi,
    provider::{ProviderClient, ProviderResponse, is_transient_status},
    router::RoutingEngine,
    telemetry::DecisionLogger,
    tokens::estimate_tokens,
    types::{
        ClassifyRequest, ClassifyResponse, ModelCapability, ModelConfig, MultimodelRequest,
        OpenAiAudioMultipartRequest, OpenAiChatRequest, OpenAiEmbeddingsRequest,
        OpenAiImagesRequest, OpenAiMultipartPart, OpenAiResponsesRequest, OpenAiSpeechRequest,
        RouterPolicy,
    },
};
use anyhow::Result;
use axum::{
    Json, Router,
    body::Body,
    extract::{Multipart, Path, State},
    http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::HashMap,
    fmt::Write as _,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{net::TcpListener, sync::oneshot, time::sleep};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};

#[derive(Clone)]
pub struct AppState {
    pub engine: RoutingEngine<SmartClassifier>,
    pub providers: ProviderClient,
    pub metrics: Arc<RouterMetrics>,
    pub accounting: BudgetAccounting,
    pub telemetry: DecisionLogger,
}

#[derive(Default)]
pub struct RouterMetrics {
    route_requests: AtomicU64,
    classify_requests: AtomicU64,
    chat_requests: AtomicU64,
    responses_requests: AtomicU64,
    embeddings_requests: AtomicU64,
    images_requests: AtomicU64,
    speech_requests: AtomicU64,
    audio_transcription_requests: AtomicU64,
    audio_translation_requests: AtomicU64,
    fallback_routes: AtomicU64,
    failover_attempts: AtomicU64,
    failover_successes: AtomicU64,
    auth_failures: AtomicU64,
    upstream_errors: AtomicU64,
    budget_rejections: AtomicU64,
    selected_models: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    total_tokens: AtomicU64,
    estimated_cost_micros: AtomicU64,
    per_model: Mutex<HashMap<String, SelectionMetrics>>,
    per_provider: Mutex<HashMap<String, SelectionMetrics>>,
}

#[derive(Debug, Serialize)]
struct MetricsSnapshot {
    route_requests: u64,
    classify_requests: u64,
    chat_requests: u64,
    responses_requests: u64,
    embeddings_requests: u64,
    images_requests: u64,
    speech_requests: u64,
    audio_transcription_requests: u64,
    audio_translation_requests: u64,
    fallback_routes: u64,
    failover_attempts: u64,
    failover_successes: u64,
    auth_failures: u64,
    upstream_errors: u64,
    budget_rejections: u64,
    selected_models: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    estimated_cost_micros: u64,
    estimated_cost_usd: f64,
    per_model: Vec<SelectionMetricsSnapshot>,
    per_provider: Vec<SelectionMetricsSnapshot>,
    budget: BudgetSnapshot,
    judge: JudgeMetricsSnapshot,
}

#[derive(Debug, Default, Clone)]
struct SelectionMetrics {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    estimated_cost_micros: u64,
}

#[derive(Debug, Serialize)]
struct SelectionMetricsSnapshot {
    id: String,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    estimated_cost_micros: u64,
    estimated_cost_usd: f64,
}

#[derive(Debug, Serialize)]
struct BudgetSnapshot {
    accounting_backend: String,
    max_chat_requests: Option<u64>,
    max_total_tokens: Option<u64>,
    max_estimated_cost_micros: Option<u64>,
    used_chat_requests: u64,
    used_total_tokens: u64,
    used_estimated_cost_micros: u64,
    chat_requests_remaining: Option<u64>,
    total_tokens_remaining: Option<u64>,
    estimated_cost_micros_remaining: Option<u64>,
}

impl RouterMetrics {
    fn snapshot_with_budget(
        &self,
        budget: Option<&BudgetConfig>,
        accounting: &BudgetAccounting,
        judge: JudgeMetricsSnapshot,
    ) -> MetricsSnapshot {
        let estimated_cost_micros = self.estimated_cost_micros.load(Ordering::Relaxed);
        MetricsSnapshot {
            route_requests: self.route_requests.load(Ordering::Relaxed),
            classify_requests: self.classify_requests.load(Ordering::Relaxed),
            chat_requests: self.chat_requests.load(Ordering::Relaxed),
            responses_requests: self.responses_requests.load(Ordering::Relaxed),
            embeddings_requests: self.embeddings_requests.load(Ordering::Relaxed),
            images_requests: self.images_requests.load(Ordering::Relaxed),
            speech_requests: self.speech_requests.load(Ordering::Relaxed),
            audio_transcription_requests: self.audio_transcription_requests.load(Ordering::Relaxed),
            audio_translation_requests: self.audio_translation_requests.load(Ordering::Relaxed),
            fallback_routes: self.fallback_routes.load(Ordering::Relaxed),
            failover_attempts: self.failover_attempts.load(Ordering::Relaxed),
            failover_successes: self.failover_successes.load(Ordering::Relaxed),
            auth_failures: self.auth_failures.load(Ordering::Relaxed),
            upstream_errors: self.upstream_errors.load(Ordering::Relaxed),
            budget_rejections: self.budget_rejections.load(Ordering::Relaxed),
            selected_models: self.selected_models.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
            estimated_cost_micros,
            estimated_cost_usd: estimated_cost_micros as f64 / 1_000_000.0,
            per_model: snapshot_selection_map(&self.per_model),
            per_provider: snapshot_selection_map(&self.per_provider),
            budget: BudgetSnapshot::from_config(self, budget, accounting),
            judge,
        }
    }

    fn record_selection(&self, model: &ModelConfig) {
        self.selected_models.fetch_add(1, Ordering::Relaxed);
        increment_selection(&self.per_model, &model.id, 1, 0, 0, 0, 0);
        increment_selection(&self.per_provider, &model.provider, 1, 0, 0, 0, 0);
    }

    fn record_usage(&self, model: &ModelConfig, usage: UsageAccounting) {
        let cost_micros = usage.estimated_cost_micros(model);
        self.prompt_tokens
            .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(usage.completion_tokens, Ordering::Relaxed);
        self.total_tokens
            .fetch_add(usage.total_tokens, Ordering::Relaxed);
        self.estimated_cost_micros
            .fetch_add(cost_micros, Ordering::Relaxed);
        increment_selection(
            &self.per_model,
            &model.id,
            0,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            cost_micros,
        );
        increment_selection(
            &self.per_provider,
            &model.provider,
            0,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            cost_micros,
        );
    }

    fn model_request_count(&self) -> u64 {
        self.chat_requests
            .load(Ordering::Relaxed)
            .saturating_add(self.responses_requests.load(Ordering::Relaxed))
            .saturating_add(self.embeddings_requests.load(Ordering::Relaxed))
            .saturating_add(self.images_requests.load(Ordering::Relaxed))
            .saturating_add(self.speech_requests.load(Ordering::Relaxed))
            .saturating_add(self.audio_transcription_requests.load(Ordering::Relaxed))
            .saturating_add(self.audio_translation_requests.load(Ordering::Relaxed))
    }
}

impl BudgetSnapshot {
    fn from_config(
        metrics: &RouterMetrics,
        budget: Option<&BudgetConfig>,
        accounting: &BudgetAccounting,
    ) -> Self {
        let Some(budget) = budget else {
            return Self {
                accounting_backend: "disabled".to_string(),
                max_chat_requests: None,
                max_total_tokens: None,
                max_estimated_cost_micros: None,
                used_chat_requests: 0,
                used_total_tokens: 0,
                used_estimated_cost_micros: 0,
                chat_requests_remaining: None,
                total_tokens_remaining: None,
                estimated_cost_micros_remaining: None,
            };
        };
        let (accounting_backend, used) = match accounting {
            BudgetAccounting::Process => (
                "process".to_string(),
                BudgetUsageSnapshot {
                    request_count: metrics.model_request_count(),
                    total_tokens: metrics.total_tokens.load(Ordering::Relaxed),
                    estimated_cost_micros: metrics.estimated_cost_micros.load(Ordering::Relaxed),
                },
            ),
            BudgetAccounting::File(_) => (
                "file".to_string(),
                accounting.snapshot().unwrap_or_default(),
            ),
        };
        Self {
            accounting_backend,
            max_chat_requests: budget.max_chat_requests,
            max_total_tokens: budget.max_total_tokens,
            max_estimated_cost_micros: budget.max_estimated_cost_micros,
            used_chat_requests: used.request_count,
            used_total_tokens: used.total_tokens,
            used_estimated_cost_micros: used.estimated_cost_micros,
            chat_requests_remaining: remaining(budget.max_chat_requests, used.request_count),
            total_tokens_remaining: remaining(budget.max_total_tokens, used.total_tokens),
            estimated_cost_micros_remaining: remaining(
                budget.max_estimated_cost_micros,
                used.estimated_cost_micros,
            ),
        }
    }
}

fn remaining(limit: Option<u64>, used: u64) -> Option<u64> {
    limit.map(|limit| limit.saturating_sub(used))
}

#[derive(Debug, Clone, Copy)]
struct UsageAccounting {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl UsageAccounting {
    fn estimated_cost_micros(self, model: &ModelConfig) -> u64 {
        let input = self.prompt_tokens as f64 * model.cost_per_million_input as f64;
        let output = self.completion_tokens as f64 * model.cost_per_million_output as f64;
        (input + output).round().max(0.0) as u64
    }
}

fn usage_from_value(value: &Value) -> Option<UsageAccounting> {
    let usage = value.get("usage")?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let completion_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(prompt_tokens.saturating_add(completion_tokens));
    Some(UsageAccounting {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

fn increment_selection(
    map: &Mutex<HashMap<String, SelectionMetrics>>,
    id: &str,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost_micros: u64,
) {
    let Ok(mut map) = map.lock() else {
        return;
    };
    let entry = map.entry(id.to_string()).or_default();
    entry.requests = entry.requests.saturating_add(requests);
    entry.prompt_tokens = entry.prompt_tokens.saturating_add(prompt_tokens);
    entry.completion_tokens = entry.completion_tokens.saturating_add(completion_tokens);
    entry.total_tokens = entry.total_tokens.saturating_add(total_tokens);
    entry.estimated_cost_micros = entry.estimated_cost_micros.saturating_add(cost_micros);
}

fn snapshot_selection_map(
    map: &Mutex<HashMap<String, SelectionMetrics>>,
) -> Vec<SelectionMetricsSnapshot> {
    let Ok(map) = map.lock() else {
        return Vec::new();
    };
    let mut snapshots = map
        .iter()
        .map(|(id, metrics)| SelectionMetricsSnapshot {
            id: id.clone(),
            requests: metrics.requests,
            prompt_tokens: metrics.prompt_tokens,
            completion_tokens: metrics.completion_tokens,
            total_tokens: metrics.total_tokens,
            estimated_cost_micros: metrics.estimated_cost_micros,
            estimated_cost_usd: metrics.estimated_cost_micros as f64 / 1_000_000.0,
        })
        .collect::<Vec<_>>();
    snapshots.sort_by(|left, right| left.id.cmp(&right.id));
    snapshots
}

pub fn app(state: AppState) -> Router {
    let state = Arc::new(state);
    Router::new()
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
        .route("/v1/router/raw", post(raw_router))
        .route("/v1/router/classify", post(classify))
        .route("/v1/router/multimodel", post(multimodel))
        .route("/v1/router/providers", get(provider_status))
        .route("/v1/router/:provider", post(provider_router))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/translations", post(audio_translations))
        .route("/metrics", get(metrics))
        .route("/metrics/prometheus", get(prometheus_metrics))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            request_context,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn serve(state: AppState, bind: &str) -> Result<()> {
    let shutdown_timeout = state.engine.config().runtime.graceful_shutdown_timeout();
    let listener = TcpListener::bind(bind).await?;
    info!("listening on http://{}", listener.local_addr()?);
    serve_with_shutdown_timeout(listener, app(state), shutdown_timeout).await?;
    Ok(())
}

async fn serve_with_shutdown_timeout(
    listener: TcpListener,
    app: Router,
    shutdown_timeout: Duration,
) -> std::io::Result<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (signal_seen_tx, signal_seen_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = signal_seen_tx.send(());
        let _ = shutdown_tx.send(());
    });

    let server = async {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    };
    tokio::pin!(server);

    tokio::select! {
        result = &mut server => result,
        _ = async {
            let _ = signal_seen_rx.await;
            sleep(shutdown_timeout).await;
        } => {
            warn!(
                timeout_ms = shutdown_timeout.as_millis(),
                "graceful shutdown timed out; forcing server future to stop"
            );
            Ok(())
        }
    }
}

async fn wait_for_shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        warn!(%error, "failed to listen for shutdown signal");
    }
    info!("shutdown signal received");
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "service": "autohand-router" }))
}

async fn openapi_json() -> Json<Value> {
    Json(openapi::spec())
}

async fn request_context(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    if !is_public_request(&request) && !authorized(&state, request.headers()) {
        state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
        let mut response = (
            StatusCode::UNAUTHORIZED,
            Json(ProviderClient::error_json(
                "missing or invalid bearer token",
            )),
        )
            .into_response();
        insert_request_id(response.headers_mut(), &request_id);
        return response;
    }

    let mut response = next.run(request).await;
    insert_request_id(response.headers_mut(), &request_id);
    response
}

fn is_public_request(request: &Request<Body>) -> bool {
    request.method() == Method::OPTIONS
        || matches!(request.uri().path(), "/health" | "/openapi.json")
}

fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    let tokens = state.engine.config().auth_tokens();
    if tokens.is_empty() {
        return true;
    }
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return false;
    };
    tokens
        .iter()
        .any(|allowed| constant_time_eq(allowed.as_bytes(), token.as_bytes()))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for idx in 0..left.len().max(right.len()) {
        let l = left.get(idx).copied().unwrap_or_default();
        let r = right.get(idx).copied().unwrap_or_default();
        diff |= (l ^ r) as usize;
    }
    diff == 0
}

fn insert_request_id(headers: &mut HeaderMap, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        headers.insert("x-autohand-router-request-id", value);
    }
}

async fn metrics(State(state): State<Arc<AppState>>) -> Json<MetricsSnapshot> {
    let config = state.engine.config();
    let judge = state.engine.classifier().judge_metrics();
    Json(
        state
            .metrics
            .snapshot_with_budget(Some(&config.budget), &state.accounting, judge),
    )
}

async fn prometheus_metrics(State(state): State<Arc<AppState>>) -> Response {
    let config = state.engine.config();
    let judge = state.engine.classifier().judge_metrics();
    let snapshot =
        state
            .metrics
            .snapshot_with_budget(Some(&config.budget), &state.accounting, judge);
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        render_prometheus_metrics(&snapshot),
    )
        .into_response()
}

fn render_prometheus_metrics(snapshot: &MetricsSnapshot) -> String {
    let mut output = String::new();
    push_metric_family(
        &mut output,
        "autohand_router_requests_total",
        "counter",
        "Router request counters by endpoint class.",
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "router")],
        snapshot.route_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "classify")],
        snapshot.classify_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "chat")],
        snapshot.chat_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "responses")],
        snapshot.responses_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "embeddings")],
        snapshot.embeddings_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "images")],
        snapshot.images_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "speech")],
        snapshot.speech_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "audio_transcriptions")],
        snapshot.audio_transcription_requests,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_requests_total",
        &[("endpoint", "audio_translations")],
        snapshot.audio_translation_requests,
    );

    push_metric_family(
        &mut output,
        "autohand_router_events_total",
        "counter",
        "Router operational event counters.",
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "fallback_routes")],
        snapshot.fallback_routes,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "failover_attempts")],
        snapshot.failover_attempts,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "failover_successes")],
        snapshot.failover_successes,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "auth_failures")],
        snapshot.auth_failures,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "upstream_errors")],
        snapshot.upstream_errors,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "budget_rejections")],
        snapshot.budget_rejections,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_events_total",
        &[("event", "selected_models")],
        snapshot.selected_models,
    );

    push_metric_family(
        &mut output,
        "autohand_router_tokens_total",
        "counter",
        "Buffered upstream token usage counters.",
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_tokens_total",
        &[("type", "prompt")],
        snapshot.prompt_tokens,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_tokens_total",
        &[("type", "completion")],
        snapshot.completion_tokens,
    );
    push_labeled_metric(
        &mut output,
        "autohand_router_tokens_total",
        &[("type", "total")],
        snapshot.total_tokens,
    );
    push_metric_family(
        &mut output,
        "autohand_router_estimated_cost_micros_total",
        "counter",
        "Estimated upstream cost in micro-dollars.",
    );
    push_metric(
        &mut output,
        "autohand_router_estimated_cost_micros_total",
        snapshot.estimated_cost_micros,
    );

    push_selection_metrics(&mut output, "model", &snapshot.per_model);
    push_selection_metrics(&mut output, "provider", &snapshot.per_provider);
    push_budget_metrics(&mut output, &snapshot.budget);
    push_judge_metrics(&mut output, &snapshot.judge);
    output
}

fn push_selection_metrics(
    output: &mut String,
    label_name: &'static str,
    selections: &[SelectionMetricsSnapshot],
) {
    let request_name = format!("autohand_router_selection_requests_total_by_{label_name}");
    let token_name = format!("autohand_router_selection_tokens_total_by_{label_name}");
    let cost_name =
        format!("autohand_router_selection_estimated_cost_micros_total_by_{label_name}");
    push_metric_family(
        output,
        &request_name,
        "counter",
        "Selected model/provider request counters.",
    );
    push_metric_family(
        output,
        &token_name,
        "counter",
        "Selected model/provider token counters.",
    );
    push_metric_family(
        output,
        &cost_name,
        "counter",
        "Selected model/provider estimated cost counters.",
    );
    for selection in selections {
        let id = selection.id.as_str();
        push_labeled_metric(
            output,
            &request_name,
            &[(label_name, id)],
            selection.requests,
        );
        push_labeled_metric(
            output,
            &token_name,
            &[(label_name, id), ("type", "prompt")],
            selection.prompt_tokens,
        );
        push_labeled_metric(
            output,
            &token_name,
            &[(label_name, id), ("type", "completion")],
            selection.completion_tokens,
        );
        push_labeled_metric(
            output,
            &token_name,
            &[(label_name, id), ("type", "total")],
            selection.total_tokens,
        );
        push_labeled_metric(
            output,
            &cost_name,
            &[(label_name, id)],
            selection.estimated_cost_micros,
        );
    }
}

fn push_budget_metrics(output: &mut String, budget: &BudgetSnapshot) {
    push_metric_family(
        output,
        "autohand_router_budget_accounting_backend",
        "gauge",
        "Budget accounting backend label with value 1 for the active backend.",
    );
    push_labeled_metric(
        output,
        "autohand_router_budget_accounting_backend",
        &[("backend", &budget.accounting_backend)],
        1,
    );
    push_metric_family(
        output,
        "autohand_router_budget_limit",
        "gauge",
        "Configured budget limits by resource.",
    );
    push_optional_metric(
        output,
        "autohand_router_budget_limit",
        &[("resource", "requests")],
        budget.max_chat_requests,
    );
    push_optional_metric(
        output,
        "autohand_router_budget_limit",
        &[("resource", "tokens")],
        budget.max_total_tokens,
    );
    push_optional_metric(
        output,
        "autohand_router_budget_limit",
        &[("resource", "cost_micros")],
        budget.max_estimated_cost_micros,
    );
    push_metric_family(
        output,
        "autohand_router_budget_used",
        "gauge",
        "Used budget by resource.",
    );
    push_labeled_metric(
        output,
        "autohand_router_budget_used",
        &[("resource", "requests")],
        budget.used_chat_requests,
    );
    push_labeled_metric(
        output,
        "autohand_router_budget_used",
        &[("resource", "tokens")],
        budget.used_total_tokens,
    );
    push_labeled_metric(
        output,
        "autohand_router_budget_used",
        &[("resource", "cost_micros")],
        budget.used_estimated_cost_micros,
    );
    push_metric_family(
        output,
        "autohand_router_budget_remaining",
        "gauge",
        "Remaining budget by resource when a limit is configured.",
    );
    push_optional_metric(
        output,
        "autohand_router_budget_remaining",
        &[("resource", "requests")],
        budget.chat_requests_remaining,
    );
    push_optional_metric(
        output,
        "autohand_router_budget_remaining",
        &[("resource", "tokens")],
        budget.total_tokens_remaining,
    );
    push_optional_metric(
        output,
        "autohand_router_budget_remaining",
        &[("resource", "cost_micros")],
        budget.estimated_cost_micros_remaining,
    );
}

fn push_judge_metrics(output: &mut String, judge: &JudgeMetricsSnapshot) {
    push_metric_family(
        output,
        "autohand_router_judge_events_total",
        "counter",
        "LLM judge routing counters.",
    );
    push_labeled_metric(
        output,
        "autohand_router_judge_events_total",
        &[("event", "requests")],
        judge.requests,
    );
    push_labeled_metric(
        output,
        "autohand_router_judge_events_total",
        &[("event", "successes")],
        judge.successes,
    );
    push_labeled_metric(
        output,
        "autohand_router_judge_events_total",
        &[("event", "fallbacks")],
        judge.fallbacks,
    );
    push_labeled_metric(
        output,
        "autohand_router_judge_events_total",
        &[("event", "invalid_outputs")],
        judge.invalid_outputs,
    );
    push_labeled_metric(
        output,
        "autohand_router_judge_events_total",
        &[("event", "heuristic_routes")],
        judge.heuristic_routes,
    );
}

fn push_metric_family(output: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} {kind}");
}

fn push_metric(output: &mut String, name: &str, value: u64) {
    let _ = writeln!(output, "{name} {value}");
}

fn push_labeled_metric(output: &mut String, name: &str, labels: &[(&str, &str)], value: u64) {
    let _ = write!(output, "{name}{{");
    for (idx, (key, value)) in labels.iter().enumerate() {
        if idx > 0 {
            output.push(',');
        }
        let _ = write!(output, "{key}=\"{}\"", prometheus_escape(value));
    }
    let _ = writeln!(output, "}} {value}");
}

fn push_optional_metric(
    output: &mut String,
    name: &str,
    labels: &[(&str, &str)],
    value: Option<u64>,
) {
    if let Some(value) = value {
        push_labeled_metric(output, name, labels, value);
    }
}

fn prometheus_escape(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

async fn provider_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = state.engine.config();
    let mut providers = Vec::with_capacity(config.providers.len());
    for provider in &config.providers {
        providers.push(state.providers.check_provider(provider).await);
    }
    Json(serde_json::json!({ "providers": providers }))
}

async fn models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let data = state
        .engine
        .config()
        .models
        .iter()
        .map(|model| {
            serde_json::json!({
                "id": model.id,
                "object": "model",
                "owned_by": model.provider,
                "aliases": model.aliases,
                "local": model.local,
                "context_window": model.context_window,
                "capabilities": model.capabilities
            })
        })
        .collect::<Vec<_>>();
    Json(serde_json::json!({ "object": "list", "data": data }))
}

async fn classify(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ClassifyRequest>,
) -> Json<ClassifyResponse> {
    state
        .metrics
        .classify_requests
        .fetch_add(1, Ordering::Relaxed);
    let classifications = state.engine.classify(&request.input).await;
    Json(ClassifyResponse {
        classifications: crate::types::SelectedClassifications::from_heads(
            classifications,
            &request.classes,
        ),
    })
}

async fn raw_router(
    State(state): State<Arc<AppState>>,
    Json(request): Json<crate::types::RawRouterRequest>,
) -> Json<crate::types::RawRouterResponse> {
    state
        .metrics
        .classify_requests
        .fetch_add(1, Ordering::Relaxed);
    let classifications = state.engine.classify(&request.input).await;
    let difficulty = classifications.difficulty;
    let confidence = difficulty.confidence;
    Json(crate::types::RawRouterResponse {
        difficulty: legacy_raw_difficulty(request.mode, difficulty),
        confidence,
    })
}

fn legacy_raw_difficulty(
    mode: crate::types::LegacyRouterMode,
    difficulty: crate::types::Classification<crate::types::DifficultyLabel>,
) -> crate::types::DifficultyLabel {
    if mode != crate::types::LegacyRouterMode::Aggressive || difficulty.confidence >= 0.86 {
        return difficulty.label;
    }

    match difficulty.label {
        crate::types::DifficultyLabel::Hard => crate::types::DifficultyLabel::Medium,
        crate::types::DifficultyLabel::Medium => crate::types::DifficultyLabel::Easy,
        label => label,
    }
}

async fn provider_router(
    State(state): State<Arc<AppState>>,
    Path(provider): Path<String>,
    Json(request): Json<crate::types::ProviderRouterRequest>,
) -> Response {
    let config = state.engine.config();
    if !config
        .providers
        .iter()
        .any(|configured| configured.name == provider)
    {
        return (
            StatusCode::NOT_FOUND,
            Json(ProviderClient::error_json(format!(
                "provider {provider} is not configured"
            ))),
        )
            .into_response();
    }

    state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
    let response = state
        .engine
        .route(MultimodelRequest {
            input: request.input,
            allowed_models: vec![],
            allowed_providers: vec![provider],
            required_capabilities: Vec::new(),
            policy: request.mode.policy(),
            default_model: None,
            max_output_tokens: None,
        })
        .await;

    Json(crate::types::ProviderRouterResponse {
        model: response.model,
        confidence: response.confidence,
    })
    .into_response()
}

async fn multimodel(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MultimodelRequest>,
) -> Json<crate::types::MultimodelResponse> {
    state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
    let input = request.input.clone();
    let response = state.engine.route(request).await;
    if response.fallback {
        state
            .metrics
            .fallback_routes
            .fetch_add(1, Ordering::Relaxed);
    }
    state
        .telemetry
        .record_route("router.multimodel", &input, &response)
        .await;
    Json(response)
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAiChatRequest>,
) -> Response {
    let prompt = request.prompt_text();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens, requested_output_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = prompt.clone();
        let required_capabilities = request.required_capabilities();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities,
                policy,
                default_model: None,
                max_output_tokens: request.max_output_tokens(),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route("chat.auto", &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (
            models,
            route.estimated_input_tokens,
            route.requested_output_tokens,
        )
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (
            vec![model],
            estimate_tokens(&prompt),
            request.max_output_tokens().unwrap_or(1024),
        )
    };

    dispatch_chat(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
        requested_output_tokens,
    )
    .await
}

async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAiResponsesRequest>,
) -> Response {
    let prompt = request.prompt_text();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens, requested_output_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = prompt.clone();
        let required_capabilities = request.required_capabilities();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities,
                policy,
                default_model: None,
                max_output_tokens: request.max_output_tokens(),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route("responses.auto", &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (
            models,
            route.estimated_input_tokens,
            route.requested_output_tokens,
        )
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (
            vec![model],
            estimate_tokens(&prompt),
            request.max_output_tokens().unwrap_or(1024),
        )
    };

    dispatch_responses(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
        requested_output_tokens,
    )
    .await
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAiEmbeddingsRequest>,
) -> Response {
    let prompt = request.prompt_text();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = prompt.clone();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy,
                default_model: None,
                max_output_tokens: Some(0),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route("embeddings.auto", &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (models, route.estimated_input_tokens)
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (vec![model], estimate_tokens(&prompt))
    };

    dispatch_embeddings(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
    )
    .await
}

async fn images_generations(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAiImagesRequest>,
) -> Response {
    let prompt = request.prompt_text();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = prompt.clone();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy,
                default_model: None,
                max_output_tokens: Some(0),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route("images.auto", &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (models, route.estimated_input_tokens)
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (vec![model], estimate_tokens(&prompt))
    };

    dispatch_images(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
    )
    .await
}

async fn audio_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAiSpeechRequest>,
) -> Response {
    let prompt = request.prompt_text();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = prompt.clone();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Audio],
                policy,
                default_model: None,
                max_output_tokens: Some(0),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route("speech.auto", &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (models, route.estimated_input_tokens)
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (vec![model], estimate_tokens(&prompt))
    };

    dispatch_speech(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
    )
    .await
}

async fn audio_transcriptions(
    State(state): State<Arc<AppState>>,
    multipart: Multipart,
) -> Response {
    let request = match parse_audio_multipart(multipart).await {
        Ok(request) => request,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(message)),
            )
                .into_response();
        }
    };
    audio_multipart_endpoint(state, request, AudioMultipartEndpoint::Transcription).await
}

async fn audio_translations(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    let request = match parse_audio_multipart(multipart).await {
        Ok(request) => request,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(message)),
            )
                .into_response();
        }
    };
    audio_multipart_endpoint(state, request, AudioMultipartEndpoint::Translation).await
}

#[derive(Debug, Clone, Copy)]
enum AudioMultipartEndpoint {
    Transcription,
    Translation,
}

impl AudioMultipartEndpoint {
    fn route_label(self) -> &'static str {
        match self {
            Self::Transcription => "audio.transcriptions.auto",
            Self::Translation => "audio.translations.auto",
        }
    }

    fn default_route_text(self) -> &'static str {
        match self {
            Self::Transcription => "audio transcription",
            Self::Translation => "audio translation",
        }
    }
}

async fn audio_multipart_endpoint(
    state: Arc<AppState>,
    mut request: OpenAiAudioMultipartRequest,
    endpoint: AudioMultipartEndpoint,
) -> Response {
    let prompt = request.prompt_text();
    let route_prompt = if prompt.trim().is_empty() {
        endpoint.default_route_text().to_string()
    } else {
        prompt
    };
    request.route_text = route_prompt.clone();
    let requested_model = request.model.clone();
    let config = state.engine.config();
    let automatic = requested_model.starts_with("router-") || requested_model == "auto";
    let (models, estimated_input_tokens) = if automatic {
        let policy = parse_router_model_policy(&requested_model);
        let route_input = route_prompt.clone();
        let route = state
            .engine
            .route(MultimodelRequest {
                input: route_input.clone(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Audio],
                policy,
                default_model: None,
                max_output_tokens: Some(0),
            })
            .await;
        state.metrics.route_requests.fetch_add(1, Ordering::Relaxed);
        if route.fallback {
            state
                .metrics
                .fallback_routes
                .fetch_add(1, Ordering::Relaxed);
        }
        state
            .telemetry
            .record_route(endpoint.route_label(), &route_input, &route)
            .await;
        let Some(model) = config.find_model(&route.model).cloned() else {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ProviderClient::error_json(format!(
                    "routed model {} is not configured",
                    route.model
                ))),
            )
                .into_response();
        };
        let mut models = vec![model];
        for candidate in route.candidates {
            if models.iter().any(|model| model.id == candidate.model) {
                continue;
            }
            if let Some(model) = config.find_model(&candidate.model).cloned() {
                models.push(model);
            }
        }
        (models, route.estimated_input_tokens)
    } else {
        let Some(model) = config.find_model(&requested_model).cloned() else {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProviderClient::error_json(format!(
                    "model {requested_model} is not configured"
                ))),
            )
                .into_response();
        };
        (vec![model], estimate_tokens(&route_prompt))
    };

    dispatch_audio_multipart(
        state,
        config,
        request,
        models,
        automatic,
        estimated_input_tokens,
        endpoint,
    )
    .await
}

async fn parse_audio_multipart(
    mut multipart: Multipart,
) -> std::result::Result<OpenAiAudioMultipartRequest, String> {
    let mut model = None;
    let mut route_text = String::new();
    let mut parts = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| format!("invalid multipart request: {error}"))?
    {
        let Some(name) = field.name().map(str::to_string) else {
            continue;
        };
        let file_name = field.file_name().map(str::to_string);
        let content_type = field.content_type().map(str::to_string);
        let data = field
            .bytes()
            .await
            .map_err(|error| format!("invalid multipart field {name}: {error}"))?;

        if name == "model" {
            let value = std::str::from_utf8(&data)
                .map_err(|_| "multipart model field must be utf-8".to_string())?
                .trim()
                .to_string();
            if value.is_empty() {
                return Err("multipart model field cannot be empty".to_string());
            }
            model = Some(value);
            continue;
        }

        if (name == "prompt" || name == "input") && route_text.is_empty() {
            if let Ok(value) = std::str::from_utf8(&data) {
                route_text = value.to_string();
            }
        } else if name == "file" && route_text.is_empty() {
            if let Some(file_name) = &file_name {
                route_text = file_name.clone();
            }
        }

        parts.push(OpenAiMultipartPart {
            name,
            file_name,
            content_type,
            data,
        });
    }

    let Some(model) = model else {
        return Err("multipart model field is required".to_string());
    };

    Ok(OpenAiAudioMultipartRequest {
        model,
        route_text,
        parts,
    })
}

async fn dispatch_chat(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiChatRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        requested_output_tokens,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    state.metrics.chat_requests.fetch_add(1, Ordering::Relaxed);

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        match state
            .providers
            .send_chat(&config, model, request.clone())
            .await
        {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    request.stream(),
                    failovers,
                    estimated_input_tokens,
                    requested_output_tokens,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

async fn dispatch_responses(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiResponsesRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        requested_output_tokens,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    state
        .metrics
        .responses_requests
        .fetch_add(1, Ordering::Relaxed);

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        match state
            .providers
            .send_responses(&config, model, request.clone())
            .await
        {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    request.stream(),
                    failovers,
                    estimated_input_tokens,
                    requested_output_tokens,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

async fn dispatch_embeddings(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiEmbeddingsRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        0,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    state
        .metrics
        .embeddings_requests
        .fetch_add(1, Ordering::Relaxed);

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        match state
            .providers
            .send_embeddings(&config, model, request.clone())
            .await
        {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    false,
                    failovers,
                    estimated_input_tokens,
                    0,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

async fn dispatch_images(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiImagesRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        0,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    state
        .metrics
        .images_requests
        .fetch_add(1, Ordering::Relaxed);

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        match state
            .providers
            .send_images(&config, model, request.clone())
            .await
        {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    false,
                    failovers,
                    estimated_input_tokens,
                    0,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

async fn dispatch_speech(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiSpeechRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        0,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    state
        .metrics
        .speech_requests
        .fetch_add(1, Ordering::Relaxed);

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        match state
            .providers
            .send_speech(&config, model, request.clone())
            .await
        {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    false,
                    failovers,
                    estimated_input_tokens,
                    0,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

async fn dispatch_audio_multipart(
    state: Arc<AppState>,
    config: Arc<RouterConfig>,
    request: OpenAiAudioMultipartRequest,
    models: Vec<ModelConfig>,
    allow_failover: bool,
    estimated_input_tokens: u32,
    endpoint: AudioMultipartEndpoint,
) -> Response {
    let Some(first_model) = models.first() else {
        state
            .metrics
            .upstream_errors
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_GATEWAY,
            Json(ProviderClient::error_json("no route candidates available")),
        )
            .into_response();
    };
    if let Some(message) = reserve_budget(
        &state,
        &config.budget,
        first_model,
        estimated_input_tokens,
        0,
    ) {
        state
            .metrics
            .budget_rejections
            .fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ProviderClient::error_json(message)),
        )
            .into_response();
    }

    match endpoint {
        AudioMultipartEndpoint::Transcription => {
            state
                .metrics
                .audio_transcription_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        AudioMultipartEndpoint::Translation => {
            state
                .metrics
                .audio_translation_requests
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    let mut last_error = None;
    let mut failovers = 0_u32;

    for (index, model) in models.iter().enumerate() {
        let result = match endpoint {
            AudioMultipartEndpoint::Transcription => {
                state
                    .providers
                    .send_audio_transcription(&config, model, request.clone())
                    .await
            }
            AudioMultipartEndpoint::Translation => {
                state
                    .providers
                    .send_audio_translation(&config, model, request.clone())
                    .await
            }
        };

        match result {
            Ok(response)
                if allow_failover
                    && is_transient_status(response.status())
                    && index + 1 < models.len() =>
            {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(response) => {
                if failovers > 0 {
                    state
                        .metrics
                        .failover_successes
                        .fetch_add(1, Ordering::Relaxed);
                }
                state.metrics.record_selection(model);
                return upstream_response(
                    &state,
                    response,
                    model,
                    false,
                    failovers,
                    estimated_input_tokens,
                    0,
                )
                .await;
            }
            Err(error) if allow_failover && index + 1 < models.len() => {
                failovers += 1;
                state
                    .metrics
                    .failover_attempts
                    .fetch_add(1, Ordering::Relaxed);
                last_error = Some(error.to_string());
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(error.to_string())),
                )
                    .into_response();
            }
        }
    }

    state
        .metrics
        .upstream_errors
        .fetch_add(1, Ordering::Relaxed);
    (
        StatusCode::BAD_GATEWAY,
        Json(ProviderClient::error_json(last_error.unwrap_or_else(
            || "all route candidates failed".to_string(),
        ))),
    )
        .into_response()
}

fn budget_violation(
    budget: &BudgetConfig,
    metrics: &RouterMetrics,
    model: &ModelConfig,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Option<String> {
    let estimated_total_tokens =
        u64::from(estimated_input_tokens).saturating_add(u64::from(requested_output_tokens));
    let estimated_cost_micros = UsageAccounting {
        prompt_tokens: u64::from(estimated_input_tokens),
        completion_tokens: u64::from(requested_output_tokens),
        total_tokens: estimated_total_tokens,
    }
    .estimated_cost_micros(model);

    if let Some(limit) = budget.max_chat_requests {
        let current = metrics
            .chat_requests
            .load(Ordering::Relaxed)
            .saturating_add(metrics.responses_requests.load(Ordering::Relaxed))
            .saturating_add(metrics.embeddings_requests.load(Ordering::Relaxed))
            .saturating_add(metrics.images_requests.load(Ordering::Relaxed))
            .saturating_add(metrics.speech_requests.load(Ordering::Relaxed))
            .saturating_add(metrics.audio_transcription_requests.load(Ordering::Relaxed))
            .saturating_add(metrics.audio_translation_requests.load(Ordering::Relaxed));
        if current.saturating_add(1) > limit {
            return Some(format!(
                "model request budget exceeded: current={current}, limit={limit}"
            ));
        }
    }
    if let Some(limit) = budget.max_total_tokens {
        let current = metrics.total_tokens.load(Ordering::Relaxed);
        if current.saturating_add(estimated_total_tokens) > limit {
            return Some(format!(
                "token budget exceeded: current={current}, requested={estimated_total_tokens}, limit={limit}"
            ));
        }
    }
    if let Some(limit) = budget.max_estimated_cost_micros {
        let current = metrics.estimated_cost_micros.load(Ordering::Relaxed);
        if current.saturating_add(estimated_cost_micros) > limit {
            return Some(format!(
                "cost budget exceeded: current_micros={current}, requested_micros={estimated_cost_micros}, limit_micros={limit}"
            ));
        }
    }
    None
}

fn reserve_budget(
    state: &AppState,
    budget: &BudgetConfig,
    model: &ModelConfig,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Option<String> {
    match &state.accounting {
        BudgetAccounting::Process => budget_violation(
            budget,
            &state.metrics,
            model,
            estimated_input_tokens,
            requested_output_tokens,
        ),
        BudgetAccounting::File(_) => {
            let reservation =
                BudgetReservation::new(model, estimated_input_tokens, requested_output_tokens);
            state
                .accounting
                .reserve(budget, reservation)
                .err()
                .map(|error| error.to_string())
        }
    }
}

async fn upstream_response(
    state: &AppState,
    upstream: ProviderResponse,
    model: &ModelConfig,
    stream: bool,
    failovers: u32,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Response {
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let mut response = if stream {
        match upstream {
            ProviderResponse::Upstream(upstream) => {
                Response::new(Body::from_stream(upstream.bytes_stream()))
            }
            ProviderResponse::Buffered { body, .. } => Response::new(Body::from(body)),
        }
    } else {
        match upstream.bytes().await {
            Ok(bytes) => {
                if status.is_success() {
                    if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                        if let Some(usage) = usage_from_value(&value) {
                            state.metrics.record_usage(model, usage);
                        }
                    }
                }
                Response::new(Body::from(bytes))
            }
            Err(error) => {
                state
                    .metrics
                    .upstream_errors
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ProviderClient::error_json(format!(
                        "failed to read upstream body: {error}"
                    ))),
                )
                    .into_response();
            }
        }
    };
    *response.status_mut() = status;
    for key in [
        header::CONTENT_TYPE,
        header::CACHE_CONTROL,
        header::TRANSFER_ENCODING,
    ] {
        if let Some(value) = headers.get(&key) {
            response.headers_mut().insert(key, value.clone());
        }
    }
    if let Ok(value) = HeaderValue::from_str(&model.id) {
        response
            .headers_mut()
            .insert("x-autohand-router-model", value);
    }
    if let Ok(value) = HeaderValue::from_str(&model.provider) {
        response
            .headers_mut()
            .insert("x-autohand-router-provider", value);
    }
    if let Ok(value) = HeaderValue::from_str(&failovers.to_string()) {
        response
            .headers_mut()
            .insert("x-autohand-router-failovers", value);
    }
    if let Ok(value) = HeaderValue::from_str(&estimated_input_tokens.to_string()) {
        response
            .headers_mut()
            .insert("x-autohand-router-input-tokens", value);
    }
    if let Ok(value) = HeaderValue::from_str(&requested_output_tokens.to_string()) {
        response
            .headers_mut()
            .insert("x-autohand-router-output-tokens", value);
    }
    response
}

fn parse_router_model_policy(model: &str) -> RouterPolicy {
    if model.contains("cost") {
        RouterPolicy::CostEfficient
    } else if model.contains("capability") || model.contains("heavy") {
        RouterPolicy::CapabilityHeavy
    } else if model.contains("domain") {
        RouterPolicy::DomainSkills
    } else {
        RouterPolicy::Balanced
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AppState, RouterMetrics, UsageAccounting, app, budget_violation, constant_time_eq,
        legacy_raw_difficulty, prometheus_escape, usage_from_value,
    };
    use crate::{
        classifier::SmartClassifier,
        config::{
            AuthConfig, BudgetAccountingBackend, BudgetAccountingConfig, BudgetConfig,
            ClassifierConfig, RouterConfig, RuntimeConfig, ScoringConfig, TelemetryConfig,
        },
        provider::ProviderClient,
        router::RoutingEngine,
        telemetry::DecisionLogger,
        types::{
            ChatMessage, Classification, DifficultyLabel, DomainLabel, LegacyRouterMode,
            ModelConfig, OpenAiChatRequest, OpenAiSpeechRequest, ProviderConfig, ProviderKind,
            RouterPolicy,
        },
    };
    use axum::{
        Json, Router, extract::Multipart, http::StatusCode, response::IntoResponse, routing::post,
    };
    use serde_json::Value;
    use std::time::SystemTime;
    use tokio::net::TcpListener;

    #[test]
    fn constant_time_equality_checks_full_input() {
        assert!(constant_time_eq(b"token", b"token"));
        assert!(!constant_time_eq(b"token", b"tokem"));
        assert!(!constant_time_eq(b"token", b"token-extra"));
        assert!(!constant_time_eq(b"", b"token"));
    }

    #[test]
    fn parses_openai_usage_accounting() {
        let usage = usage_from_value(&serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn estimates_cost_in_micros_from_model_prices() {
        let model = ModelConfig {
            id: "priced".to_string(),
            provider: "test".to_string(),
            aliases: vec![],
            capability: 0.5,
            cost_per_million_input: 2.0,
            cost_per_million_output: 10.0,
            domains: vec![],
            context_window: None,
            capabilities: Default::default(),
            local: false,
        };
        let usage = UsageAccounting {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
        };
        assert_eq!(usage.estimated_cost_micros(&model), 400);
    }

    #[test]
    fn rejects_when_request_budget_is_exhausted() {
        let metrics = RouterMetrics::default();
        metrics
            .chat_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let budget = BudgetConfig {
            max_chat_requests: Some(1),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: Default::default(),
        };
        let model = test_model();
        let violation = budget_violation(&budget, &metrics, &model, 10, 10);
        assert!(violation.unwrap().contains("model request budget"));
    }

    #[test]
    fn request_budget_counts_responses_and_embeddings_requests() {
        let metrics = RouterMetrics::default();
        metrics
            .responses_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        metrics
            .embeddings_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let budget = BudgetConfig {
            max_chat_requests: Some(2),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: Default::default(),
        };
        let model = test_model();
        let violation = budget_violation(&budget, &metrics, &model, 10, 0);
        assert!(violation.unwrap().contains("model request budget"));
    }

    #[test]
    fn rejects_when_estimated_token_budget_would_be_exceeded() {
        let metrics = RouterMetrics::default();
        metrics
            .total_tokens
            .store(90, std::sync::atomic::Ordering::Relaxed);
        let budget = BudgetConfig {
            max_chat_requests: None,
            max_total_tokens: Some(100),
            max_estimated_cost_micros: None,
            accounting: Default::default(),
        };
        let model = test_model();
        let violation = budget_violation(&budget, &metrics, &model, 8, 8);
        assert!(violation.unwrap().contains("token budget"));
    }

    #[tokio::test]
    async fn automatic_chat_failover_skips_transient_candidate_and_records_metrics() {
        let failing_base_url =
            spawn_chat_upstream("strong-fail", StatusCode::SERVICE_UNAVAILABLE).await;
        let healthy_base_url = spawn_chat_upstream("strong-ok", StatusCode::OK).await;
        let config = failover_config(failing_base_url, healthy_base_url);
        let router_url = spawn_router(config).await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{router_url}/v1/chat/completions"))
            .json(&OpenAiChatRequest {
                model: "router-capability".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: Value::String(
                        "Design a production Rust router with distributed failover and security"
                            .to_string(),
                    ),
                }],
                extra: serde_json::Map::from_iter([("max_tokens".to_string(), Value::from(64))]),
            })
            .send()
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-model")
                .and_then(|value| value.to_str().ok()),
            Some("strong-ok")
        );
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-provider")
                .and_then(|value| value.to_str().ok()),
            Some("healthy")
        );
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-failovers")
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
        let body = response.json::<Value>().await.unwrap();
        assert_eq!(body["model"], "strong-ok");

        let metrics = client
            .get(format!("{router_url}/metrics"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(metrics["chat_requests"], 1);
        assert_eq!(metrics["failover_attempts"], 1);
        assert_eq!(metrics["failover_successes"], 1);
        assert_eq!(metrics["selected_models"], 1);
        assert_eq!(metrics["per_model"][0]["id"], "strong-ok");

        let prometheus = client
            .get(format!("{router_url}/metrics/prometheus"))
            .send()
            .await
            .unwrap();
        assert_eq!(prometheus.status(), StatusCode::OK);
        assert!(
            prometheus
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .starts_with("text/plain")
        );
        let body = prometheus.text().await.unwrap();
        assert!(body.contains("autohand_router_requests_total{endpoint=\"chat\"} 1"));
        assert!(body.contains("autohand_router_events_total{event=\"failover_attempts\"} 1"));
        assert!(
            body.contains(
                "autohand_router_selection_requests_total_by_model{model=\"strong-ok\"} 1"
            )
        );
    }

    #[test]
    fn prometheus_labels_are_escaped() {
        assert_eq!(
            prometheus_escape("model\"one\\two\nthree"),
            "model\\\"one\\\\two\\nthree"
        );
    }

    #[tokio::test]
    async fn speech_proxy_forwards_to_configured_path_and_records_metrics() {
        let upstream_url = spawn_speech_upstream().await;
        let config = speech_config(upstream_url);
        let router_url = spawn_router(config).await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{router_url}/v1/audio/speech"))
            .json(&OpenAiSpeechRequest {
                model: "speech-alias".to_string(),
                input: "read this".to_string(),
                voice: "alloy".to_string(),
                extra: Default::default(),
            })
            .send()
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-model")
                .and_then(|value| value.to_str().ok()),
            Some("speech-model")
        );
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-provider")
                .and_then(|value| value.to_str().ok()),
            Some("speech-provider")
        );
        let body = response.json::<Value>().await.unwrap();
        assert_eq!(body["model"], "speech-model");
        assert_eq!(body["input"], "read this");
        assert_eq!(body["voice"], "alloy");

        let metrics = client
            .get(format!("{router_url}/metrics"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(metrics["speech_requests"], 1);
        assert_eq!(metrics["selected_models"], 1);
        assert_eq!(metrics["per_model"][0]["id"], "speech-model");
    }

    #[tokio::test]
    async fn automatic_speech_routing_requires_audio_capability() {
        let upstream_url = spawn_speech_upstream().await;
        let config = speech_config(upstream_url);
        let router_url = spawn_router(config).await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{router_url}/v1/audio/speech"))
            .json(&OpenAiSpeechRequest {
                model: "auto".to_string(),
                input: "read this".to_string(),
                voice: "alloy".to_string(),
                extra: Default::default(),
            })
            .send()
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(
            response
                .headers()
                .get("x-autohand-router-model")
                .and_then(|value| value.to_str().ok()),
            Some("speech-model")
        );
    }

    #[tokio::test]
    async fn multipart_audio_proxy_rewrites_model_and_records_metrics() {
        let upstream_url = spawn_speech_upstream().await;
        let config = speech_config(upstream_url);
        let router_url = spawn_router(config).await;
        let client = reqwest::Client::new();

        let transcription = client
            .post(format!("{router_url}/v1/audio/transcriptions"))
            .multipart(audio_test_form("transcribe this"))
            .send()
            .await
            .unwrap();
        assert!(transcription.status().is_success());
        assert_eq!(
            transcription
                .headers()
                .get("x-autohand-router-model")
                .and_then(|value| value.to_str().ok()),
            Some("speech-model")
        );
        let transcription_body = transcription.json::<Value>().await.unwrap();
        assert_eq!(transcription_body["model"], "speech-model");
        assert_eq!(transcription_body["prompt"], "transcribe this");
        assert_eq!(transcription_body["file_name"], "clip.wav");
        assert_eq!(transcription_body["file_bytes"], 5);

        let translation = client
            .post(format!("{router_url}/v1/audio/translations"))
            .multipart(audio_test_form("translate this"))
            .send()
            .await
            .unwrap();
        assert!(translation.status().is_success());
        let translation_body = translation.json::<Value>().await.unwrap();
        assert_eq!(translation_body["model"], "speech-model");
        assert_eq!(translation_body["prompt"], "translate this");
        assert_eq!(translation_body["file_name"], "clip.wav");

        let metrics = client
            .get(format!("{router_url}/metrics"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(metrics["audio_transcription_requests"], 1);
        assert_eq!(metrics["audio_translation_requests"], 1);
        assert_eq!(metrics["selected_models"], 2);
        assert_eq!(metrics["per_model"][0]["id"], "speech-model");
    }

    #[tokio::test]
    async fn file_budget_accounting_is_shared_across_router_instances() {
        let upstream_url = spawn_chat_upstream("shared-model", StatusCode::OK).await;
        let ledger_path = temp_ledger_path("shared-budget");
        let config = shared_budget_config(upstream_url, ledger_path.clone());
        let first_router_url = spawn_router(config.clone()).await;
        let second_router_url = spawn_router(config).await;
        let client = reqwest::Client::new();

        let first = client
            .post(format!("{first_router_url}/v1/chat/completions"))
            .json(&OpenAiChatRequest {
                model: "shared-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("first request".to_string()),
                }],
                extra: serde_json::Map::new(),
            })
            .send()
            .await
            .unwrap();
        assert!(first.status().is_success());

        let second = client
            .post(format!("{second_router_url}/v1/chat/completions"))
            .json(&OpenAiChatRequest {
                model: "shared-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("second request".to_string()),
                }],
                extra: serde_json::Map::new(),
            })
            .send()
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);

        let metrics = client
            .get(format!("{second_router_url}/metrics"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(metrics["budget"]["accounting_backend"], "file");
        assert_eq!(metrics["budget"]["used_chat_requests"], 1);
        assert_eq!(metrics["budget"]["chat_requests_remaining"], 0);

        let _ = std::fs::remove_file(ledger_path.with_extension("json.lock"));
        let _ = std::fs::remove_file(ledger_path);
    }

    #[test]
    fn aggressive_legacy_raw_mode_downgrades_borderline_difficulty() {
        let difficulty = Classification {
            class_id: 1,
            label: DifficultyLabel::Medium,
            confidence: 0.72,
            meets_threshold: true,
        };
        assert_eq!(
            legacy_raw_difficulty(LegacyRouterMode::Aggressive, difficulty),
            DifficultyLabel::Easy
        );

        let confident = Classification {
            class_id: 1,
            label: DifficultyLabel::Medium,
            confidence: 0.90,
            meets_threshold: true,
        };
        assert_eq!(
            legacy_raw_difficulty(LegacyRouterMode::Aggressive, confident),
            DifficultyLabel::Medium
        );
    }

    fn test_model() -> ModelConfig {
        ModelConfig {
            id: "priced".to_string(),
            provider: "test".to_string(),
            aliases: vec![],
            capability: 0.5,
            cost_per_million_input: 2.0,
            cost_per_million_output: 10.0,
            domains: vec![],
            context_window: None,
            capabilities: Default::default(),
            local: false,
        }
    }

    async fn spawn_chat_upstream(model_id: &'static str, status: StatusCode) -> String {
        async fn chat(
            axum::extract::State((model_id, status)): axum::extract::State<(
                &'static str,
                StatusCode,
            )>,
        ) -> axum::response::Response {
            if status != StatusCode::OK {
                return (
                    status,
                    Json(serde_json::json!({
                        "error": {
                            "message": "transient upstream failure"
                        }
                    })),
                )
                    .into_response();
            }
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12
                    }
                })),
            )
                .into_response()
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state((model_id, status));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_speech_upstream() -> String {
        async fn speech(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "input": request["input"],
                "voice": request["voice"],
                "audio": "mock"
            }))
        }
        async fn audio_multipart(mut multipart: Multipart) -> Json<Value> {
            let mut model = String::new();
            let mut prompt = String::new();
            let mut file_name = String::new();
            let mut file_bytes = 0_usize;

            while let Some(field) = multipart.next_field().await.unwrap() {
                let name = field.name().unwrap_or_default().to_string();
                let field_file_name = field.file_name().map(str::to_string);
                let data = field.bytes().await.unwrap();
                match name.as_str() {
                    "model" => model = std::str::from_utf8(&data).unwrap().to_string(),
                    "prompt" => prompt = std::str::from_utf8(&data).unwrap().to_string(),
                    "file" => {
                        file_name = field_file_name.unwrap_or_default();
                        file_bytes = data.len();
                    }
                    _ => {}
                }
            }

            Json(serde_json::json!({
                "model": model,
                "prompt": prompt,
                "file_name": file_name,
                "file_bytes": file_bytes
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/custom/speech", post(speech))
            .route("/custom/transcriptions", post(audio_multipart))
            .route("/custom/translations", post(audio_multipart));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn audio_test_form(prompt: &str) -> reqwest::multipart::Form {
        let file = reqwest::multipart::Part::bytes(b"audio".to_vec())
            .file_name("clip.wav")
            .mime_str("audio/wav")
            .unwrap();
        reqwest::multipart::Form::new()
            .text("model", "speech-alias")
            .text("prompt", prompt.to_string())
            .part("file", file)
    }

    async fn spawn_router(config: RouterConfig) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let classifier = SmartClassifier::new(config.clone()).unwrap();
        let engine = RoutingEngine::new(config.clone(), classifier);
        let state = AppState {
            engine,
            providers: ProviderClient::new(&config).unwrap(),
            metrics: Default::default(),
            accounting: crate::accounting::BudgetAccounting::from_budget_config(&config.budget)
                .unwrap(),
            telemetry: DecisionLogger::new(&config.telemetry),
        };
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn failover_config(failing_base_url: String, healthy_base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "strong-fail".to_string(),
            policy: RouterPolicy::CapabilityHeavy,
            providers: vec![
                ProviderConfig {
                    name: "failing".to_string(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: failing_base_url,
                    api_key_env: None,
                    api_key: None,
                    chat_path: "/v1/chat/completions".to_string(),
                    responses_path: Some("/v1/responses".to_string()),
                    embeddings_path: Some("/v1/embeddings".to_string()),
                    images_path: Some("/v1/images/generations".to_string()),
                    speech_path: Some("/v1/audio/speech".to_string()),
                    audio_transcriptions_path: Some("/v1/audio/transcriptions".to_string()),
                    audio_translations_path: Some("/v1/audio/translations".to_string()),
                    health_path: None,
                    timeout_ms: 1_000,
                    retries: 0,
                    max_concurrency: None,
                    queue_timeout_ms: None,
                    extra_headers: Default::default(),
                },
                ProviderConfig {
                    name: "healthy".to_string(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: healthy_base_url,
                    api_key_env: None,
                    api_key: None,
                    chat_path: "/v1/chat/completions".to_string(),
                    responses_path: Some("/v1/responses".to_string()),
                    embeddings_path: Some("/v1/embeddings".to_string()),
                    images_path: Some("/v1/images/generations".to_string()),
                    speech_path: Some("/v1/audio/speech".to_string()),
                    audio_transcriptions_path: Some("/v1/audio/transcriptions".to_string()),
                    audio_translations_path: Some("/v1/audio/translations".to_string()),
                    health_path: None,
                    timeout_ms: 1_000,
                    retries: 0,
                    max_concurrency: None,
                    queue_timeout_ms: None,
                    extra_headers: Default::default(),
                },
            ],
            models: vec![
                ModelConfig {
                    id: "strong-fail".to_string(),
                    provider: "failing".to_string(),
                    aliases: vec![],
                    capability: 0.95,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::Coding, DomainLabel::Design],
                    context_window: Some(4096),
                    capabilities: Default::default(),
                    local: true,
                },
                ModelConfig {
                    id: "strong-ok".to_string(),
                    provider: "healthy".to_string(),
                    aliases: vec![],
                    capability: 0.90,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::Coding, DomainLabel::Design],
                    context_window: Some(4096),
                    capabilities: Default::default(),
                    local: true,
                },
            ],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }

    fn speech_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "speech-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "speech-provider".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/v1/chat/completions".to_string(),
                responses_path: Some("/v1/responses".to_string()),
                embeddings_path: Some("/v1/embeddings".to_string()),
                images_path: Some("/v1/images/generations".to_string()),
                speech_path: Some("/custom/speech".to_string()),
                audio_transcriptions_path: Some("/custom/transcriptions".to_string()),
                audio_translations_path: Some("/custom/translations".to_string()),
                health_path: None,
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![ModelConfig {
                id: "speech-model".to_string(),
                provider: "speech-provider".to_string(),
                aliases: vec!["speech-alias".to_string()],
                capability: 0.75,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::General],
                context_window: Some(4096),
                capabilities: crate::types::ModelCapabilities {
                    supports_audio: true,
                    ..Default::default()
                },
                local: true,
            }],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }

    fn shared_budget_config(base_url: String, ledger_path: std::path::PathBuf) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "shared-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "shared-provider".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/v1/chat/completions".to_string(),
                responses_path: Some("/v1/responses".to_string()),
                embeddings_path: Some("/v1/embeddings".to_string()),
                images_path: Some("/v1/images/generations".to_string()),
                speech_path: Some("/v1/audio/speech".to_string()),
                audio_transcriptions_path: Some("/v1/audio/transcriptions".to_string()),
                audio_translations_path: Some("/v1/audio/translations".to_string()),
                health_path: None,
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![ModelConfig {
                id: "shared-model".to_string(),
                provider: "shared-provider".to_string(),
                aliases: vec![],
                capability: 0.75,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::General],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig {
                max_chat_requests: Some(1),
                max_total_tokens: None,
                max_estimated_cost_micros: None,
                accounting: BudgetAccountingConfig {
                    backend: BudgetAccountingBackend::File,
                    file_path: Some(ledger_path.to_string_lossy().to_string()),
                    lock_timeout_ms: 1_000,
                },
            },
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }

    fn temp_ledger_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "autohand-router-{name}-{}.json",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
