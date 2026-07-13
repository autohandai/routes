use crate::{
    classifier::{PromptClassifier, SmartClassifier},
    config::{ClassifierBackend, RouterConfig},
    conformance::config_fingerprint,
    eval::{EvalExample, EvalReport, evaluate, load_jsonl, seeded_holdout},
    router::RoutingEngine,
    types::ProviderKind,
};
use anyhow::{Context, Result};
use axum::{Json, Router, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, task::JoinHandle, time::sleep};

#[derive(Debug, Clone)]
pub struct ClassifierLiveGateConfig {
    pub revision: String,
    pub smoke_runs: usize,
    pub min_adapter_success_rate: f64,
    pub max_fallback_rate: f64,
    pub max_smoke_p95_ms: u64,
    pub holdout_ratio: f32,
    pub holdout_seed: u64,
    pub min_holdout_examples: usize,
    pub min_tier_accuracy: f32,
    pub min_domain_accuracy: f32,
    pub min_model_accuracy: f32,
    pub min_provider_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierLiveGateReport {
    pub schema_version: u32,
    pub generated_unix_seconds: u64,
    pub source_revision: String,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub classifier_backend: String,
    pub configured_model: Option<String>,
    pub dataset: RedactedDatasetEvidence,
    pub thresholds: ClassifierGateThresholds,
    pub live: LiveClassifierEvidence,
    pub holdout: HoldoutEvidence,
    pub failure_injections: Vec<ClassifierFailureInjection>,
    pub payloads_redacted: bool,
    pub pass: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedDatasetEvidence {
    pub file_name: String,
    pub fnv1a_64: String,
    pub source_examples: usize,
    pub holdout_examples: usize,
    pub holdout_ratio: f32,
    pub holdout_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierGateThresholds {
    pub min_adapter_success_rate: f64,
    pub max_fallback_rate: f64,
    pub max_smoke_p95_ms: u64,
    pub min_holdout_examples: usize,
    pub min_tier_accuracy: f32,
    pub min_domain_accuracy: f32,
    pub min_model_accuracy: f32,
    pub min_provider_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveClassifierEvidence {
    pub smoke_runs: usize,
    pub total_adapter_requests: u64,
    pub adapter_successes: u64,
    pub adapter_fallbacks: u64,
    pub invalid_outputs: u64,
    pub heuristic_routes: u64,
    pub adapter_success_rate: f64,
    pub fallback_rate: f64,
    pub smoke_latency_ms: Vec<u64>,
    pub smoke_p50_ms: u64,
    pub smoke_p95_ms: u64,
    pub smoke_max_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoldoutEvidence {
    pub examples: usize,
    pub exact_tier_matches: usize,
    pub domain_examples: usize,
    pub domain_matches: usize,
    pub model_examples: usize,
    pub model_matches: usize,
    pub provider_examples: usize,
    pub provider_matches: usize,
    pub tier_accuracy: f32,
    pub domain_accuracy: f32,
    pub model_accuracy: f32,
    pub provider_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierFailureInjection {
    pub scenario: String,
    pub adapter_requests: u64,
    pub adapter_successes: u64,
    pub adapter_fallbacks: u64,
    pub invalid_outputs: u64,
    pub heuristic_routes: u64,
    pub elapsed_ms: u64,
    pub failed_closed_to_heuristic: bool,
    pub pass: bool,
    pub error: Option<String>,
}

pub async fn run_classifier_live_gate(
    config: &RouterConfig,
    dataset_path: &Path,
    gate: ClassifierLiveGateConfig,
) -> Result<ClassifierLiveGateReport> {
    validate_gate_config(&gate)?;
    let examples = load_jsonl(dataset_path)?;
    let holdout = seeded_holdout(&examples, gate.holdout_ratio, gate.holdout_seed)?;
    let backend = config.classifier.active_backend();
    let configured_model = config.classifier.active_adapter().model;
    let mut failures = Vec::new();
    if backend == ClassifierBackend::Heuristic {
        failures.push(
            "classifier-live-gate requires llm_judge or route_llm; heuristic is not live classifier evidence"
                .to_string(),
        );
    }

    let classifier = SmartClassifier::new(config.clone())?;
    let metrics_handle = classifier.clone();
    let smoke_inputs = [
        "Fix a typo in a Rust comment",
        "Design a fault-tolerant event processing architecture",
        "Summarize a short product update",
        "Write a SQL query for monthly active users",
        "Clarify an ambiguous deployment request",
    ];
    let mut smoke_latency_ms = Vec::with_capacity(gate.smoke_runs);
    for index in 0..gate.smoke_runs {
        let started = Instant::now();
        let _ = classifier
            .classify(smoke_inputs[index % smoke_inputs.len()])
            .await;
        smoke_latency_ms.push(elapsed_ms(started));
    }
    let engine = RoutingEngine::new(config.clone(), classifier);
    let eval = evaluate(&engine, &holdout).await;
    let metrics = metrics_handle.judge_metrics();
    let expected_requests = gate.smoke_runs.saturating_add(holdout.len()) as u64;
    let adapter_success_rate = metrics.successes as f64 / metrics.requests.max(1) as f64;
    let fallback_rate = metrics.fallbacks as f64 / metrics.requests.max(1) as f64;
    let smoke_p50_ms = percentile(&smoke_latency_ms, 50);
    let smoke_p95_ms = percentile(&smoke_latency_ms, 95);
    let smoke_max_ms = smoke_latency_ms.iter().copied().max().unwrap_or_default();

    if metrics.requests != expected_requests {
        failures.push(format!(
            "configured classifier made {} adapter requests for {expected_requests} smoke/holdout classifications",
            metrics.requests
        ));
    }
    if adapter_success_rate < gate.min_adapter_success_rate {
        failures.push(format!(
            "adapter success rate {adapter_success_rate} is below minimum {}",
            gate.min_adapter_success_rate
        ));
    }
    if fallback_rate > gate.max_fallback_rate {
        failures.push(format!(
            "classifier fallback rate {fallback_rate} is above maximum {}",
            gate.max_fallback_rate
        ));
    }
    if smoke_p95_ms > gate.max_smoke_p95_ms {
        failures.push(format!(
            "classifier smoke p95 {smoke_p95_ms}ms is above maximum {}ms",
            gate.max_smoke_p95_ms
        ));
    }
    evaluate_holdout_thresholds(&eval, &holdout, &gate, &mut failures);

    let failure_injections = if backend == ClassifierBackend::Heuristic {
        Vec::new()
    } else {
        run_failure_injections(config).await?
    };
    for injection in failure_injections
        .iter()
        .filter(|injection| !injection.pass)
    {
        failures.push(format!(
            "failure injection {} did not fail closed: {}",
            injection.scenario,
            injection
                .error
                .as_deref()
                .unwrap_or("metric assertion failed")
        ));
    }

    Ok(ClassifierLiveGateReport {
        schema_version: 1,
        generated_unix_seconds: unix_seconds(),
        source_revision: gate.revision,
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: config_fingerprint(config)?,
        classifier_backend: backend.config_key().to_string(),
        configured_model,
        dataset: RedactedDatasetEvidence {
            file_name: dataset_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("redacted-dataset")
                .to_string(),
            fnv1a_64: format!("{:016x}", fnv1a_64(&fs::read(dataset_path)?)),
            source_examples: examples.len(),
            holdout_examples: holdout.len(),
            holdout_ratio: gate.holdout_ratio,
            holdout_seed: gate.holdout_seed,
        },
        thresholds: ClassifierGateThresholds {
            min_adapter_success_rate: gate.min_adapter_success_rate,
            max_fallback_rate: gate.max_fallback_rate,
            max_smoke_p95_ms: gate.max_smoke_p95_ms,
            min_holdout_examples: gate.min_holdout_examples,
            min_tier_accuracy: gate.min_tier_accuracy,
            min_domain_accuracy: gate.min_domain_accuracy,
            min_model_accuracy: gate.min_model_accuracy,
            min_provider_accuracy: gate.min_provider_accuracy,
        },
        live: LiveClassifierEvidence {
            smoke_runs: gate.smoke_runs,
            total_adapter_requests: metrics.requests,
            adapter_successes: metrics.successes,
            adapter_fallbacks: metrics.fallbacks,
            invalid_outputs: metrics.invalid_outputs,
            heuristic_routes: metrics.heuristic_routes,
            adapter_success_rate,
            fallback_rate,
            smoke_latency_ms,
            smoke_p50_ms,
            smoke_p95_ms,
            smoke_max_ms,
        },
        holdout: holdout_evidence(&eval),
        failure_injections,
        payloads_redacted: true,
        pass: failures.is_empty(),
        failures,
    })
}

fn validate_gate_config(gate: &ClassifierLiveGateConfig) -> Result<()> {
    anyhow::ensure!(
        !gate.revision.trim().is_empty(),
        "revision must not be empty"
    );
    anyhow::ensure!(gate.smoke_runs > 0, "smoke runs must be non-zero");
    for (name, value) in [
        ("min_adapter_success_rate", gate.min_adapter_success_rate),
        ("max_fallback_rate", gate.max_fallback_rate),
    ] {
        anyhow::ensure!(
            (0.0..=1.0).contains(&value),
            "{name} must be between zero and one"
        );
    }
    for (name, value) in [
        ("min_tier_accuracy", gate.min_tier_accuracy),
        ("min_domain_accuracy", gate.min_domain_accuracy),
        ("min_model_accuracy", gate.min_model_accuracy),
        ("min_provider_accuracy", gate.min_provider_accuracy),
    ] {
        anyhow::ensure!(
            (0.0..=1.0).contains(&value),
            "{name} must be between zero and one"
        );
    }
    Ok(())
}

fn evaluate_holdout_thresholds(
    eval: &EvalReport,
    holdout: &[EvalExample],
    gate: &ClassifierLiveGateConfig,
    failures: &mut Vec<String>,
) {
    if holdout.len() < gate.min_holdout_examples {
        failures.push(format!(
            "holdout has {} examples, below minimum {}",
            holdout.len(),
            gate.min_holdout_examples
        ));
    }
    for (name, actual, minimum) in [
        ("tier", eval.accuracy, gate.min_tier_accuracy),
        ("domain", eval.domain_accuracy, gate.min_domain_accuracy),
        ("model", eval.model_accuracy, gate.min_model_accuracy),
        (
            "provider",
            eval.provider_accuracy,
            gate.min_provider_accuracy,
        ),
    ] {
        if actual < minimum {
            failures.push(format!(
                "holdout {name} accuracy {actual} is below minimum {minimum}"
            ));
        }
    }
}

fn holdout_evidence(eval: &EvalReport) -> HoldoutEvidence {
    HoldoutEvidence {
        examples: eval.total,
        exact_tier_matches: eval.exact_tier_matches,
        domain_examples: eval.domain_examples,
        domain_matches: eval.domain_matches,
        model_examples: eval.model_examples,
        model_matches: eval.model_matches,
        provider_examples: eval.provider_examples,
        provider_matches: eval.provider_matches,
        tier_accuracy: eval.accuracy,
        domain_accuracy: eval.domain_accuracy,
        model_accuracy: eval.model_accuracy,
        provider_accuracy: eval.provider_accuracy,
    }
}

#[derive(Clone, Copy)]
enum InjectionScenario {
    Timeout,
    InvalidJson,
    RateLimited,
}

impl InjectionScenario {
    fn name(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::InvalidJson => "invalid_json",
            Self::RateLimited => "http_429",
        }
    }
}

async fn run_failure_injections(config: &RouterConfig) -> Result<Vec<ClassifierFailureInjection>> {
    let mut reports = Vec::with_capacity(3);
    for scenario in [
        InjectionScenario::Timeout,
        InjectionScenario::InvalidJson,
        InjectionScenario::RateLimited,
    ] {
        reports.push(run_failure_injection(config, scenario).await?);
    }
    Ok(reports)
}

async fn run_failure_injection(
    source: &RouterConfig,
    scenario: InjectionScenario,
) -> Result<ClassifierFailureInjection> {
    let (base_url, task) = spawn_injection_server(scenario).await?;
    let mut config = source.clone();
    config.runtime.provider_conformance_artifact = None;
    let model_id = config
        .classifier
        .active_adapter()
        .model
        .context("configured classifier model is missing")?;
    let provider_name = config
        .find_model(&model_id)
        .with_context(|| format!("classifier model {model_id} is not configured"))?
        .provider
        .clone();
    let provider = config
        .providers
        .iter_mut()
        .find(|provider| provider.name == provider_name)
        .with_context(|| format!("classifier provider {provider_name} is not configured"))?;
    provider.kind = ProviderKind::OpenAiCompatible;
    provider.base_url = base_url;
    provider.chat_path = "/v1/chat/completions".to_string();
    provider.api_key = None;
    provider.api_key_env = None;
    provider.extra_headers.clear();
    provider.retries = 0;
    provider.timeout_ms = 250;
    config.classifier.llm_judge_timeout_ms = 40;
    config.classifier.adapters.llm_judge.timeout_ms = 40;
    config.classifier.adapters.route_llm.timeout_ms = 40;

    let classifier = SmartClassifier::new(config)?;
    let before = classifier.judge_metrics();
    let started = Instant::now();
    let _ = classifier.classify("controlled failure injection").await;
    let elapsed_ms = elapsed_ms(started);
    let after = classifier.judge_metrics();
    task.abort();
    let adapter_requests = after.requests.saturating_sub(before.requests);
    let adapter_successes = after.successes.saturating_sub(before.successes);
    let adapter_fallbacks = after.fallbacks.saturating_sub(before.fallbacks);
    let invalid_outputs = after.invalid_outputs.saturating_sub(before.invalid_outputs);
    let heuristic_routes = after
        .heuristic_routes
        .saturating_sub(before.heuristic_routes);
    let expected_invalid = matches!(scenario, InjectionScenario::InvalidJson) as u64;
    let failed_closed_to_heuristic = adapter_requests == 1
        && adapter_successes == 0
        && adapter_fallbacks == 1
        && heuristic_routes == 1
        && invalid_outputs == expected_invalid;
    Ok(ClassifierFailureInjection {
        scenario: scenario.name().to_string(),
        adapter_requests,
        adapter_successes,
        adapter_fallbacks,
        invalid_outputs,
        heuristic_routes,
        elapsed_ms,
        failed_closed_to_heuristic,
        pass: failed_closed_to_heuristic,
        error: (!failed_closed_to_heuristic).then(|| {
            format!(
                "expected requests=1 successes=0 fallbacks=1 heuristic=1 invalid={expected_invalid}"
            )
        }),
    })
}

async fn spawn_injection_server(scenario: InjectionScenario) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            match scenario {
                InjectionScenario::Timeout => {
                    sleep(Duration::from_millis(150)).await;
                    (StatusCode::OK, Json(valid_classifier_envelope()))
                }
                InjectionScenario::InvalidJson => (
                    StatusCode::OK,
                    Json(json!({
                        "choices": [{"message": {"content": "not-json"}}]
                    })),
                ),
                InjectionScenario::RateLimited => (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({"error": {"message": "controlled rate limit"}})),
                ),
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Ok((format!("http://{address}"), task))
}

fn valid_classifier_envelope() -> Value {
    Json(json!({
        "choices": [{"message": {"content": valid_classifier_content()}}]
    }))
    .0
}

fn valid_classifier_content() -> String {
    json!({
        "difficulty": "medium",
        "ambiguity": "low",
        "domain": "coding",
        "modality": "text",
        "safety": "safe",
        "cacheability": "medium",
        "latency_sensitivity": "medium",
        "reasoning_depth": "moderate",
        "confidence": 0.99,
        "ambiguity_confidence": 0.99,
        "domain_confidence": 0.99,
        "modality_confidence": 0.99,
        "safety_confidence": 0.99,
        "cacheability_confidence": 0.99,
        "latency_sensitivity_confidence": 0.99,
        "reasoning_depth_confidence": 0.99
    })
    .to_string()
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) * percentile).div_ceil(100);
    sorted[index.min(sorted.len() - 1)]
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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
    use crate::types::RouterPolicy;

    #[tokio::test]
    async fn live_gate_passes_valid_adapter_holdout_and_failure_injections() {
        let base_url = spawn_live_classifier_server(true).await;
        let config = live_classifier_config(&base_url, "super-secret-token");
        let dataset = write_eval_dataset("classifier-live-pass");
        let report = run_classifier_live_gate(&config, &dataset, test_gate())
            .await
            .unwrap();
        let encoded = serde_json::to_string(&report).unwrap();

        assert!(report.pass, "{:?}", report.failures);
        assert_eq!(report.live.adapter_fallbacks, 0);
        assert_eq!(report.live.adapter_success_rate, 1.0);
        assert_eq!(report.holdout.tier_accuracy, 1.0);
        assert!(report.failure_injections.iter().all(|item| item.pass));
        assert!(!encoded.contains("super-secret-token"));
        assert!(!encoded.contains("Design a hidden customer prompt"));
        fs::remove_file(dataset).unwrap();
    }

    #[tokio::test]
    async fn live_gate_fails_when_configured_adapter_falls_back() {
        let base_url = spawn_live_classifier_server(false).await;
        let config = live_classifier_config(&base_url, "secret");
        let dataset = write_eval_dataset("classifier-live-fail");
        let report = run_classifier_live_gate(&config, &dataset, test_gate())
            .await
            .unwrap();

        assert!(!report.pass);
        assert_eq!(report.live.adapter_successes, 0);
        assert!(report.live.adapter_fallbacks > 0);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("success rate"))
        );
        assert!(report.failure_injections.iter().all(|item| item.pass));
        fs::remove_file(dataset).unwrap();
    }

    fn test_gate() -> ClassifierLiveGateConfig {
        ClassifierLiveGateConfig {
            revision: "classifier-test".to_string(),
            smoke_runs: 3,
            min_adapter_success_rate: 1.0,
            max_fallback_rate: 0.0,
            max_smoke_p95_ms: 2_000,
            holdout_ratio: 1.0,
            holdout_seed: 7,
            min_holdout_examples: 2,
            min_tier_accuracy: 1.0,
            min_domain_accuracy: 1.0,
            min_model_accuracy: 1.0,
            min_provider_accuracy: 1.0,
        }
    }

    async fn spawn_live_classifier_server(valid: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                if valid {
                    Json(valid_classifier_envelope())
                } else {
                    Json(json!({
                        "choices": [{"message": {"content": "not-json"}}]
                    }))
                }
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{address}")
    }

    fn live_classifier_config(base_url: &str, secret: &str) -> RouterConfig {
        let yaml = format!(
            r#"
bind: 127.0.0.1:0
default_model: judge-model
policy: balanced
classifier:
  backend: llm_judge
  adapters:
    llm_judge:
      model: judge-model
      timeout_ms: 1000
providers:
  - name: judge-provider
    kind: open_ai_compatible
    base_url: {base_url}
    api_key: {secret}
    retries: 0
    timeout_ms: 1000
models:
  - id: judge-model
    provider: judge-provider
    capability: 0.70
    cost_per_million_input: 1.0
    cost_per_million_output: 1.0
    domains: [coding]
    context_window: 8192
    capabilities:
      supported_endpoints: [chat]
"#
        );
        let config = serde_yaml::from_str::<RouterConfig>(&yaml).unwrap();
        config.validate().unwrap();
        config
    }

    fn write_eval_dataset(label: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("{label}-{}.jsonl", std::process::id()));
        let examples = [
            EvalExample {
                input: "Design a hidden customer prompt".to_string(),
                expected_tier: crate::eval::ExpectedTier::Balanced,
                expected_domain: Some(crate::types::DomainLabel::Coding),
                expected_model: Some("judge-model".to_string()),
                expected_provider: Some("judge-provider".to_string()),
                policy: RouterPolicy::Balanced,
                allowed_models: Vec::new(),
                allowed_providers: Vec::new(),
                required_capabilities: Vec::new(),
            },
            EvalExample {
                input: "Another private labeled prompt".to_string(),
                expected_tier: crate::eval::ExpectedTier::Balanced,
                expected_domain: Some(crate::types::DomainLabel::Coding),
                expected_model: Some("judge-model".to_string()),
                expected_provider: Some("judge-provider".to_string()),
                policy: RouterPolicy::Balanced,
                allowed_models: Vec::new(),
                allowed_providers: Vec::new(),
                required_capabilities: Vec::new(),
            },
        ];
        let body = examples
            .iter()
            .map(|example| serde_json::to_string(example).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, body).unwrap();
        path
    }
}
