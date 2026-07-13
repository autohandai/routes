use crate::{
    config::{AuthConfig, BudgetConfig, RouterConfig, SafetyRoutingConfig, TelemetryConfig},
    conformance::config_fingerprint,
    server::{AppState, app},
    types::ModelEndpoint,
};
use anyhow::{Context, Result};
use axum::{
    Router,
    body::Body,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::post,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
    time::{sleep, timeout},
};

#[derive(Debug, Clone)]
pub struct StreamLiveGateConfig {
    pub revision: String,
    pub max_first_chunk_ms: u64,
    pub max_completion_ms: u64,
    pub cancellation_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamLiveGateReport {
    pub schema_version: u32,
    pub generated_unix_seconds: u64,
    pub source_revision: String,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub thresholds: StreamGateThresholds,
    pub profiles: Vec<ProviderStreamProfile>,
    pub controlled_injections: ControlledStreamInjections,
    pub payloads_redacted: bool,
    pub pass: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamGateThresholds {
    pub max_first_chunk_ms: u64,
    pub max_completion_ms: u64,
    pub cancellation_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStreamProfile {
    pub provider: String,
    pub model: String,
    pub adapter: String,
    pub advertised: bool,
    pub skip_reason: Option<String>,
    pub provider_version: Option<String>,
    pub model_version: Option<String>,
    pub short_stream: Option<ShortStreamEvidence>,
    pub cancellation: Option<CancellationEvidence>,
    pub pass: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortStreamEvidence {
    pub status: u16,
    pub content_type: Option<String>,
    pub first_chunk_ms: u64,
    pub completion_ms: u64,
    pub body_bytes: u64,
    pub body_fnv1a_64: String,
    pub proxy_bytes: Option<u64>,
    pub proxy_fnv1a_64: Option<String>,
    pub done_present: bool,
    pub terminal_usage_present: bool,
    pub proxy_terminal_usage_present: bool,
    pub pass: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationEvidence {
    pub first_chunk_ms: u64,
    pub cancellation_observed_ms: u64,
    pub streams_active_after: u64,
    pub cancelled_delta: u64,
    pub post_cancel_status: Option<u16>,
    pub post_cancel_completed: bool,
    pub readiness_status: Option<u16>,
    pub model_viable_after: bool,
    pub pass: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlledStreamInjections {
    pub retry_after: RetryAfterEvidence,
    pub mid_body_close: MidBodyCloseEvidence,
    pub shutdown_during_stream: ShutdownStreamEvidence,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryAfterEvidence {
    pub upstream_calls: u64,
    pub retry_after_header_seconds: u64,
    pub configured_retry_cap_ms: u64,
    pub elapsed_ms: u64,
    pub final_status: u16,
    pub same_provider_retry: bool,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidBodyCloseEvidence {
    pub client_observed_body_error: bool,
    pub forwarded_prefix_bytes: u64,
    pub proxy_body_errors: u64,
    pub streams_active_after: u64,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownStreamEvidence {
    pub first_chunk_received: bool,
    pub shutdown_elapsed_ms: u64,
    pub shutdown_timeout_ms: u64,
    pub pass: bool,
}

pub async fn run_stream_live_gate(
    source: &RouterConfig,
    gate: StreamLiveGateConfig,
) -> Result<StreamLiveGateReport> {
    validate_gate(&gate)?;
    let config = isolated_live_config(source)?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router_task = tokio::spawn(async move {
        axum::serve(listener, app(AppState::from_config(&config)?))
            .await
            .map_err(anyhow::Error::from)
    });
    let base_url = format!("http://{address}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(
            gate.max_completion_ms.saturating_add(5_000),
        ))
        .build()?;
    let mut profiles = Vec::with_capacity(source.models.len());
    for model in &source.models {
        let provider = source
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let adapter = provider.kind.chat_adapter_contract();
        let advertised = provider.supports_endpoint(ModelEndpoint::Chat)
            && model.capabilities.supports_endpoint(ModelEndpoint::Chat)
            && adapter.supports_streaming;
        if !advertised {
            profiles.push(ProviderStreamProfile {
                provider: provider.name.clone(),
                model: model.id.clone(),
                adapter: adapter.name.to_string(),
                advertised: false,
                skip_reason: Some(if !adapter.supports_streaming {
                    "provider adapter contract does not advertise streaming".to_string()
                } else {
                    "chat endpoint is not jointly advertised by provider and model".to_string()
                }),
                provider_version: None,
                model_version: None,
                short_stream: None,
                cancellation: None,
                pass: true,
                error: None,
            });
            continue;
        }
        profiles.push(
            run_provider_stream_profile(
                &client,
                &base_url,
                &provider.name,
                &model.id,
                adapter.name,
                &gate,
            )
            .await,
        );
    }
    router_task.abort();

    let controlled_injections = run_controlled_injections(&gate).await?;
    let mut failures = profiles
        .iter()
        .filter(|profile| !profile.pass)
        .map(|profile| {
            format!(
                "provider {} model {} stream gate failed: {}",
                profile.provider,
                profile.model,
                profile.error.as_deref().unwrap_or("probe assertion failed")
            )
        })
        .collect::<Vec<_>>();
    if !controlled_injections.pass {
        failures.push("one or more controlled stream failure injections failed".to_string());
    }
    Ok(StreamLiveGateReport {
        schema_version: 1,
        generated_unix_seconds: unix_seconds(),
        source_revision: gate.revision,
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: config_fingerprint(source)?,
        thresholds: StreamGateThresholds {
            max_first_chunk_ms: gate.max_first_chunk_ms,
            max_completion_ms: gate.max_completion_ms,
            cancellation_timeout_ms: gate.cancellation_timeout_ms,
            shutdown_timeout_ms: gate.shutdown_timeout_ms,
        },
        profiles,
        controlled_injections,
        payloads_redacted: true,
        pass: failures.is_empty(),
        failures,
    })
}

async fn run_provider_stream_profile(
    client: &reqwest::Client,
    base_url: &str,
    provider: &str,
    model: &str,
    adapter: &str,
    gate: &StreamLiveGateConfig,
) -> ProviderStreamProfile {
    let result = async {
        let (short_stream, provider_version, model_version) =
            run_short_stream(client, base_url, provider, model, gate).await?;
        anyhow::ensure!(short_stream.pass, "short stream evidence failed");
        let cancellation = run_cancellation(client, base_url, model, gate).await?;
        anyhow::ensure!(cancellation.pass, "cancellation evidence failed");
        Ok::<_, anyhow::Error>((short_stream, cancellation, provider_version, model_version))
    }
    .await;
    match result {
        Ok((short_stream, cancellation, provider_version, model_version)) => {
            ProviderStreamProfile {
                provider: provider.to_string(),
                model: model.to_string(),
                adapter: adapter.to_string(),
                advertised: true,
                skip_reason: None,
                provider_version,
                model_version,
                short_stream: Some(short_stream),
                cancellation: Some(cancellation),
                pass: true,
                error: None,
            }
        }
        Err(error) => ProviderStreamProfile {
            provider: provider.to_string(),
            model: model.to_string(),
            adapter: adapter.to_string(),
            advertised: true,
            skip_reason: None,
            provider_version: None,
            model_version: None,
            short_stream: None,
            cancellation: None,
            pass: false,
            error: Some(format!("{error:#}")),
        },
    }
}

async fn run_short_stream(
    client: &reqwest::Client,
    base_url: &str,
    provider: &str,
    model: &str,
    gate: &StreamLiveGateConfig,
) -> Result<(ShortStreamEvidence, Option<String>, Option<String>)> {
    let started = Instant::now();
    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_request(model, 32, "short stream evidence"))
        .send()
        .await?;
    let status = response.status().as_u16();
    let content_type = header_string(response.headers(), header::CONTENT_TYPE.as_str());
    let provider_version = header_string(response.headers(), "x-provider-version");
    let model_version = header_string(response.headers(), "x-model-version");
    let mut body_stream = response.bytes_stream();
    let mut body = Vec::new();
    let mut first_chunk_ms = None;
    while let Some(chunk) = timeout(
        Duration::from_millis(gate.max_completion_ms),
        body_stream.next(),
    )
    .await
    .context("stream body timed out")?
    {
        let chunk = chunk?;
        first_chunk_ms.get_or_insert_with(|| elapsed_ms(started));
        body.extend_from_slice(&chunk);
    }
    let completion_ms = elapsed_ms(started);
    let first_chunk_ms = first_chunk_ms.unwrap_or(completion_ms);
    let body_hash = format!("{:016x}", fnv1a_64(&body));
    let (done_present, terminal_usage_present) = parse_sse_evidence(&body);
    let metrics =
        wait_for_stream_metrics(client, base_url, provider, model, "success", 1_000).await?;
    let proxy_bytes = metrics.get("last_bytes").and_then(Value::as_u64);
    let proxy_fnv = metrics
        .get("last_fnv1a_64")
        .and_then(Value::as_str)
        .map(str::to_string);
    let proxy_usage = metrics
        .get("last_terminal_usage_present")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let pass = (200..300).contains(&status)
        && content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("text/event-stream"))
        && first_chunk_ms <= gate.max_first_chunk_ms
        && completion_ms <= gate.max_completion_ms
        && done_present
        && terminal_usage_present
        && proxy_bytes == Some(body.len() as u64)
        && proxy_fnv.as_deref() == Some(body_hash.as_str())
        && proxy_usage;
    Ok((
        ShortStreamEvidence {
            status,
            content_type,
            first_chunk_ms,
            completion_ms,
            body_bytes: body.len() as u64,
            body_fnv1a_64: body_hash,
            proxy_bytes,
            proxy_fnv1a_64: proxy_fnv,
            done_present,
            terminal_usage_present,
            proxy_terminal_usage_present: proxy_usage,
            pass,
            error: (!pass)
                .then(|| "status/content-type/latency/SSE/usage/hash assertion failed".to_string()),
        },
        provider_version,
        model_version,
    ))
}

async fn run_cancellation(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    gate: &StreamLiveGateConfig,
) -> Result<CancellationEvidence> {
    let before = metrics_json(client, base_url).await?;
    let cancelled_before = before
        .get("streams_cancelled")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let started = Instant::now();
    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_request(
            model,
            512,
            "produce a deliberately cancellable numbered list",
        ))
        .send()
        .await?;
    anyhow::ensure!(
        response.status().is_success(),
        "cancellation stream returned {}",
        response.status()
    );
    let mut stream = response.bytes_stream();
    let first = timeout(
        Duration::from_millis(gate.max_first_chunk_ms),
        stream.next(),
    )
    .await
    .context("cancellation first chunk timed out")?
    .context("cancellation stream ended before first chunk")??;
    anyhow::ensure!(!first.is_empty(), "cancellation first chunk was empty");
    let first_chunk_ms = elapsed_ms(started);
    drop(stream);

    let cancel_started = Instant::now();
    let after = loop {
        let after = metrics_json(client, base_url).await?;
        let active = after
            .get("streams_active")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        let cancelled = after
            .get("streams_cancelled")
            .and_then(Value::as_u64)
            .unwrap_or(cancelled_before);
        if active == 0 && cancelled > cancelled_before {
            break after;
        }
        if elapsed_ms(cancel_started) > gate.cancellation_timeout_ms {
            break after;
        }
        sleep(Duration::from_millis(10)).await;
    };
    let cancellation_observed_ms = elapsed_ms(cancel_started);
    let streams_active_after = after
        .get("streams_active")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let cancelled_after = after
        .get("streams_cancelled")
        .and_then(Value::as_u64)
        .unwrap_or(cancelled_before);
    let cancelled_delta = cancelled_after.saturating_sub(cancelled_before);

    let followup_client = reqwest::Client::new();
    let followup = followup_client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_request(
            model,
            8,
            "post cancellation capacity probe",
        ))
        .send()
        .await;
    let (post_cancel_status, post_cancel_completed) = match followup {
        Ok(response) => {
            let status = response.status().as_u16();
            let completed = response.bytes().await.is_ok();
            (Some(status), completed)
        }
        Err(_) => (None, false),
    };
    let readiness = followup_client
        .get(format!("{base_url}/health/ready"))
        .send()
        .await;
    let (readiness_status, model_viable_after) = match readiness {
        Ok(response) => {
            let status = response.status().as_u16();
            let value = response.json::<Value>().await.ok();
            let viable = value
                .as_ref()
                .and_then(|value| value.get("viable_models"))
                .and_then(Value::as_array)
                .is_some_and(|models| models.iter().any(|item| item.as_str() == Some(model)));
            (Some(status), viable)
        }
        Err(_) => (None, false),
    };
    let pass = first_chunk_ms <= gate.max_first_chunk_ms
        && cancellation_observed_ms <= gate.cancellation_timeout_ms
        && streams_active_after == 0
        && cancelled_delta == 1
        && post_cancel_status.is_some_and(|status| (200..300).contains(&status))
        && post_cancel_completed
        && readiness_status.is_some_and(|status| (200..300).contains(&status))
        && model_viable_after;
    Ok(CancellationEvidence {
        first_chunk_ms,
        cancellation_observed_ms,
        streams_active_after,
        cancelled_delta,
        post_cancel_status,
        post_cancel_completed,
        readiness_status,
        model_viable_after,
        pass,
        error: (!pass)
            .then(|| "cancellation/active-stream/post-capacity assertion failed".to_string()),
    })
}

async fn run_controlled_injections(
    gate: &StreamLiveGateConfig,
) -> Result<ControlledStreamInjections> {
    let retry_after = run_retry_after_injection().await?;
    let mid_body_close = run_mid_body_close_injection().await?;
    let shutdown_during_stream = run_shutdown_injection(gate.shutdown_timeout_ms).await?;
    Ok(ControlledStreamInjections {
        pass: retry_after.pass && mid_body_close.pass && shutdown_during_stream.pass,
        retry_after,
        mid_body_close,
        shutdown_during_stream,
    })
}

async fn run_retry_after_injection() -> Result<RetryAfterEvidence> {
    let calls = Arc::new(AtomicU64::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let calls_for_server = Arc::clone(&calls);
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let calls = Arc::clone(&calls_for_server);
            async move {
                let call = calls.fetch_add(1, Ordering::Relaxed);
                if call == 0 {
                    (
                        StatusCode::TOO_MANY_REQUESTS,
                        [(header::RETRY_AFTER, "1")],
                        Body::from("rate limited"),
                    )
                        .into_response()
                } else {
                    sse_response(false)
                }
            }
        }),
    );
    let upstream_task = tokio::spawn(async move {
        axum::serve(listener, upstream).await.unwrap();
    });
    let mut config = controlled_stream_config(&format!("http://{address}"))?;
    config.providers[0].retries = 1;
    config.providers[0].retry_max_delay_ms = 100;
    let (base_url, router_task) = spawn_test_router(config).await?;
    let started = Instant::now();
    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_request("stream-model", 8, "retry after"))
        .send()
        .await?;
    let status = response.status().as_u16();
    let body_ok = response.bytes().await.is_ok();
    let elapsed_ms = elapsed_ms(started);
    let upstream_calls = calls.load(Ordering::Relaxed);
    router_task.abort();
    upstream_task.abort();
    let pass = upstream_calls == 2
        && (200..300).contains(&status)
        && body_ok
        && (80..1_000).contains(&elapsed_ms);
    Ok(RetryAfterEvidence {
        upstream_calls,
        retry_after_header_seconds: 1,
        configured_retry_cap_ms: 100,
        elapsed_ms,
        final_status: status,
        same_provider_retry: upstream_calls == 2,
        pass,
    })
}

async fn run_mid_body_close_injection() -> Result<MidBodyCloseEvidence> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let upstream_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await.unwrap();
        let prefix = b"data: {\"choices\":[{\"delta\":{\"content\":\"prefix\"}}]}\n\n";
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        socket
            .write_all(format!("{:X}\r\n", prefix.len()).as_bytes())
            .await
            .unwrap();
        socket.write_all(prefix).await.unwrap();
        socket.write_all(b"\r\n").await.unwrap();
        socket.shutdown().await.unwrap();
    });
    let (base_url, router_task) =
        spawn_test_router(controlled_stream_config(&format!("http://{address}"))?).await?;
    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&stream_request("stream-model", 8, "mid body"))
        .send()
        .await?;
    let mut stream = response.bytes_stream();
    let mut forwarded_prefix_bytes = 0_u64;
    let mut client_observed_body_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(bytes) => {
                forwarded_prefix_bytes = forwarded_prefix_bytes.saturating_add(bytes.len() as u64)
            }
            Err(_) => {
                client_observed_body_error = true;
                break;
            }
        }
    }
    let client = reqwest::Client::new();
    let evidence = wait_for_stream_metrics(
        &client,
        &base_url,
        "stream-provider",
        "stream-model",
        "body_error",
        1_000,
    )
    .await?;
    let metrics = metrics_json(&client, &base_url).await?;
    let proxy_body_errors = evidence["body_errors"].as_u64().unwrap_or(0);
    let streams_active_after = metrics["streams_active"].as_u64().unwrap_or(u64::MAX);
    router_task.abort();
    upstream_task.abort();
    let pass = client_observed_body_error
        && forwarded_prefix_bytes > 0
        && proxy_body_errors == 1
        && streams_active_after == 0;
    Ok(MidBodyCloseEvidence {
        client_observed_body_error,
        forwarded_prefix_bytes,
        proxy_body_errors,
        streams_active_after,
        pass,
    })
}

async fn run_shutdown_injection(shutdown_timeout_ms: u64) -> Result<ShutdownStreamEvidence> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let chunks = stream::unfold(0_u8, |step| async move {
                if step > 10 {
                    return None;
                }
                if step > 0 {
                    sleep(Duration::from_millis(50)).await;
                }
                Some((
                    Ok::<Bytes, io::Error>(Bytes::from("data: {}\n\n")),
                    step + 1,
                ))
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(chunks),
            )
                .into_response()
        }),
    );
    let upstream_task = tokio::spawn(async move {
        axum::serve(listener, upstream).await.unwrap();
    });
    let config = controlled_stream_config(&format!("http://{address}"))?;
    let router_listener = TcpListener::bind("127.0.0.1:0").await?;
    let router_address = router_listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let router = app(AppState::from_config(&config)?);
    let router_task = tokio::spawn(async move {
        axum::serve(router_listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    });
    let response = reqwest::Client::new()
        .post(format!("http://{router_address}/v1/chat/completions"))
        .json(&stream_request("stream-model", 128, "shutdown stream"))
        .send()
        .await?;
    let mut body = response.bytes_stream();
    let first_chunk_received = body.next().await.transpose()?.is_some();
    let started = Instant::now();
    let _ = shutdown_tx.send(());
    sleep(Duration::from_millis(20)).await;
    drop(body);
    let completed = timeout(Duration::from_millis(shutdown_timeout_ms), router_task).await;
    let shutdown_elapsed_ms = elapsed_ms(started);
    upstream_task.abort();
    let pass =
        first_chunk_received && completed.is_ok() && shutdown_elapsed_ms <= shutdown_timeout_ms;
    Ok(ShutdownStreamEvidence {
        first_chunk_received,
        shutdown_elapsed_ms,
        shutdown_timeout_ms,
        pass,
    })
}

fn isolated_live_config(source: &RouterConfig) -> Result<RouterConfig> {
    let mut config = source.clone();
    config.bind = "127.0.0.1:0".to_string();
    config.auth = AuthConfig::default();
    config.budget = BudgetConfig::default();
    config.telemetry = TelemetryConfig::default();
    config.cache = Default::default();
    config.shadow_eval = Default::default();
    config.safety = SafetyRoutingConfig::default();
    config.sticky_routing = Default::default();
    config.runtime.provider_conformance_artifact = None;
    config.runtime.provider_health_sampler.enabled = false;
    config.runtime.ingress.per_credential_requests_per_minute = None;
    config.validate()?;
    Ok(config)
}

fn controlled_stream_config(base_url: &str) -> Result<RouterConfig> {
    let yaml = format!(
        r#"
bind: 127.0.0.1:0
default_model: stream-model
providers:
  - name: stream-provider
    kind: open_ai_compatible
    base_url: {base_url}
    retries: 0
    timeout_ms: 1000
    stream_idle_timeout_ms: 1000
    max_concurrency: 1
    queue_timeout_ms: 500
models:
  - id: stream-model
    provider: stream-provider
    capability: 0.5
    cost_per_million_input: 1.0
    cost_per_million_output: 1.0
    context_window: 4096
    capabilities:
      supported_endpoints: [chat]
"#
    );
    let config = serde_yaml::from_str::<RouterConfig>(&yaml)?;
    config.validate()?;
    Ok(config)
}

async fn spawn_test_router(config: RouterConfig) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = app(AppState::from_config(&config)?);
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Ok((format!("http://{address}"), task))
}

fn sse_response(slow: bool) -> axum::response::Response {
    let chunks = stream::unfold(0_u8, move |step| async move {
        if step >= 3 {
            return None;
        }
        if slow && step > 0 {
            sleep(Duration::from_millis(50)).await;
        }
        let body = match step {
            0 => "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            1 => {
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"
            }
            _ => "data: [DONE]\n\n",
        };
        Some((Ok::<Bytes, io::Error>(Bytes::from(body)), step + 1))
    });
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE.as_str(), "text/event-stream"),
            ("x-provider-version", "controlled-stream-provider-1"),
            ("x-model-version", "controlled-stream-model-1"),
        ],
        Body::from_stream(chunks),
    )
        .into_response()
}

fn stream_request(model: &str, max_tokens: u64, prompt: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true}
    })
}

async fn metrics_json(client: &reqwest::Client, base_url: &str) -> Result<Value> {
    Ok(client
        .get(format!("{base_url}/metrics"))
        .send()
        .await?
        .error_for_status()?
        .json::<Value>()
        .await?)
}

async fn wait_for_stream_metrics(
    client: &reqwest::Client,
    base_url: &str,
    provider: &str,
    model: &str,
    outcome: &str,
    max_wait_ms: u64,
) -> Result<Value> {
    let started = Instant::now();
    loop {
        let metrics = metrics_json(client, base_url).await?;
        if let Some(evidence) = metrics
            .get("stream_evidence")
            .and_then(Value::as_array)
            .and_then(|items| {
                items.iter().find(|item| {
                    item.get("provider").and_then(Value::as_str) == Some(provider)
                        && item.get("model").and_then(Value::as_str) == Some(model)
                        && item.get("last_outcome").and_then(Value::as_str) == Some(outcome)
                })
            })
        {
            return Ok(evidence.clone());
        }
        anyhow::ensure!(
            elapsed_ms(started) <= max_wait_ms,
            "stream metrics did not converge"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

fn parse_sse_evidence(body: &[u8]) -> (bool, bool) {
    let Ok(text) = std::str::from_utf8(body) else {
        return (false, false);
    };
    let mut done = false;
    let mut usage = false;
    for line in text.lines() {
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
        } else if let Ok(value) = serde_json::from_str::<Value>(data) {
            usage |= value.get("usage").is_some_and(|usage| {
                usage.get("total_tokens").and_then(Value::as_u64).is_some()
                    || usage.get("input_tokens").and_then(Value::as_u64).is_some()
            });
        }
    }
    (done, usage)
}

fn header_string(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn validate_gate(gate: &StreamLiveGateConfig) -> Result<()> {
    anyhow::ensure!(
        !gate.revision.trim().is_empty(),
        "revision must not be empty"
    );
    for (name, value) in [
        ("max_first_chunk_ms", gate.max_first_chunk_ms),
        ("max_completion_ms", gate.max_completion_ms),
        ("cancellation_timeout_ms", gate.cancellation_timeout_ms),
        ("shutdown_timeout_ms", gate.shutdown_timeout_ms),
    ] {
        anyhow::ensure!(value > 0, "{name} must be non-zero");
    }
    Ok(())
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderKind;

    #[tokio::test]
    async fn stream_gate_proves_passthrough_usage_cancellation_retry_and_shutdown() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(|| async { sse_response(true) }),
        );
        tokio::spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });
        let config = controlled_stream_config(&format!("http://{address}")).unwrap();
        let report = run_stream_live_gate(
            &config,
            StreamLiveGateConfig {
                revision: "stream-test".to_string(),
                max_first_chunk_ms: 1_000,
                max_completion_ms: 2_000,
                cancellation_timeout_ms: 1_000,
                shutdown_timeout_ms: 1_000,
            },
        )
        .await
        .unwrap();

        assert!(
            report.pass,
            "failures={:?} controlled={:?} profiles={:?}",
            report.failures, report.controlled_injections, report.profiles
        );
        let profile = &report.profiles[0];
        assert!(profile.pass);
        let short = profile.short_stream.as_ref().unwrap();
        assert_eq!(
            short.body_fnv1a_64,
            short.proxy_fnv1a_64.as_deref().unwrap()
        );
        assert_eq!(short.proxy_bytes, Some(short.body_bytes));
        assert!(short.terminal_usage_present);
        assert!(profile.cancellation.as_ref().unwrap().pass);
        assert!(report.controlled_injections.retry_after.pass);
        assert!(report.controlled_injections.mid_body_close.pass);
        assert!(report.controlled_injections.shutdown_during_stream.pass);
    }

    #[tokio::test]
    async fn native_non_streaming_adapter_is_explicitly_skipped() {
        let mut config = controlled_stream_config("http://127.0.0.1:1").unwrap();
        config.providers[0].kind = ProviderKind::OllamaNative;
        config.providers[0].chat_path = "/api/chat".to_string();
        config.validate().unwrap();
        let report = run_stream_live_gate(
            &config,
            StreamLiveGateConfig {
                revision: "stream-skip".to_string(),
                max_first_chunk_ms: 1_000,
                max_completion_ms: 2_000,
                cancellation_timeout_ms: 1_000,
                shutdown_timeout_ms: 1_000,
            },
        )
        .await
        .unwrap();

        assert!(
            report.pass,
            "failures={:?} controlled={:?}",
            report.failures, report.controlled_injections
        );
        assert!(!report.profiles[0].advertised);
        assert!(report.profiles[0].skip_reason.is_some());
    }
}
