use crate::{
    config::RouterConfig,
    conformance::run_provider_conformance_matrix,
    load::{LoadSuiteConfig, LoadSuiteReport, default_load_suite_scenarios, run_load_suite},
    server::{AppState, app},
};
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Body,
    extract::Multipart,
    http::{Response, StatusCode, header},
    response::IntoResponse,
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, task::JoinHandle};

pub const BENCHMARK_FILE: &str = "router.controlled-benchmark.json";
pub const CONFORMANCE_FILE: &str = "router.controlled-conformance.json";
pub const MANIFEST_FILE: &str = "router.controlled-evidence-manifest.json";

#[derive(Debug, Clone)]
pub struct ControlledEvidenceConfig {
    pub revision: String,
    pub output_dir: PathBuf,
    pub runs: usize,
    pub requests_per_scenario: u64,
    pub concurrency: usize,
    pub slo_p95_ms: u64,
    pub slo_error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlledBenchmarkArtifact {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub generated_unix_seconds: u64,
    pub source_revision: String,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub environment: BenchmarkEnvironment,
    pub workload: BenchmarkWorkload,
    pub thresholds: BenchmarkThresholds,
    pub variability: VariabilityBoundary,
    pub resource_observation: ResourceObservation,
    pub runs: Vec<LoadSuiteReport>,
    pub aggregates: Vec<ScenarioAggregate>,
    pub pass: bool,
    pub failures: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEnvironment {
    pub os: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub rust_target_family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    pub warmup_requests_per_scenario: u64,
    pub measured_runs: usize,
    pub requests_per_scenario_per_run: u64,
    pub concurrency: usize,
    pub scenarios: Vec<BenchmarkScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkThresholds {
    pub p95_ms: u64,
    pub error_rate: f64,
    pub required_successful_runs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariabilityBoundary {
    pub provider_class: String,
    pub includes_external_provider_variability: bool,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceObservation {
    pub peak_rss_bytes: Option<u64>,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioAggregate {
    pub scenario: String,
    pub path: String,
    pub runs: usize,
    pub p95_ms: RepeatStatistic,
    pub p99_ms: RepeatStatistic,
    pub requests_per_second: RepeatStatistic,
    pub maximum_error_rate: f64,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatStatistic {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
    pub confidence_95_lower: f64,
    pub confidence_95_upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlledEvidenceManifest {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub generated_unix_seconds: u64,
    pub source_revision: String,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub benchmark_file: String,
    pub benchmark_fnv1a_64: String,
    pub conformance_file: String,
    pub conformance_fnv1a_64: String,
    pub provider_class: String,
    pub includes_external_provider_variability: bool,
    pub pass: bool,
    pub replay_command: String,
}

pub async fn write_controlled_evidence(
    evidence_config: ControlledEvidenceConfig,
) -> Result<ControlledEvidenceManifest> {
    validate_controlled_config(&evidence_config)?;
    let (provider_url, provider_task) = spawn_controlled_provider().await?;
    let router_config = controlled_router_config(&provider_url)?;
    let conformance =
        run_provider_conformance_matrix(router_config.clone(), "controlled evidence".to_string())
            .await?;
    anyhow::ensure!(
        conformance.pass,
        "controlled provider conformance failed before benchmark"
    );

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let router_address = listener.local_addr()?;
    let router_task = tokio::spawn(async move {
        axum::serve(listener, app(AppState::from_config(&router_config)?))
            .await
            .map_err(anyhow::Error::from)
    });
    let base_url = format!("http://{router_address}");
    let scenarios = default_load_suite_scenarios();

    let warmup = run_load_suite(LoadSuiteConfig {
        base_url: base_url.clone(),
        requests_per_scenario: 1,
        concurrency: 1,
        slo_p95_ms: evidence_config.slo_p95_ms,
        slo_error_rate: evidence_config.slo_error_rate,
        scenarios: scenarios.clone(),
    })
    .await?;
    anyhow::ensure!(warmup.pass, "controlled evidence warmup failed");

    let mut runs = Vec::with_capacity(evidence_config.runs);
    for _ in 0..evidence_config.runs {
        runs.push(
            run_load_suite(LoadSuiteConfig {
                base_url: base_url.clone(),
                requests_per_scenario: evidence_config.requests_per_scenario,
                concurrency: evidence_config.concurrency,
                slo_p95_ms: evidence_config.slo_p95_ms,
                slo_error_rate: evidence_config.slo_error_rate,
                scenarios: scenarios.clone(),
            })
            .await?,
        );
    }

    router_task.abort();
    provider_task.abort();
    let aggregates = aggregate_runs(&runs, &scenarios, &evidence_config)?;
    let mut failures = aggregates
        .iter()
        .filter(|aggregate| !aggregate.pass)
        .map(|aggregate| format!("scenario {} exceeded a threshold", aggregate.scenario))
        .collect::<Vec<_>>();
    if runs.iter().filter(|run| run.pass).count() != evidence_config.runs {
        failures.push("one or more measured load-suite runs failed".to_string());
    }
    let replay_command = replay_command(&evidence_config);
    let benchmark = ControlledBenchmarkArtifact {
        schema_version: 1,
        artifact_kind: "controlled_router_overhead_benchmark".to_string(),
        generated_unix_seconds: unix_seconds(),
        source_revision: evidence_config.revision.clone(),
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: conformance.config_fnv1a_64.clone(),
        environment: BenchmarkEnvironment {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            rust_target_family: std::env::consts::FAMILY.to_string(),
        },
        workload: BenchmarkWorkload {
            warmup_requests_per_scenario: 1,
            measured_runs: evidence_config.runs,
            requests_per_scenario_per_run: evidence_config.requests_per_scenario,
            concurrency: evidence_config.concurrency,
            scenarios: scenarios
                .iter()
                .map(|scenario| BenchmarkScenario {
                    name: scenario.name.clone(),
                    path: scenario.path.clone(),
                })
                .collect(),
        },
        thresholds: BenchmarkThresholds {
            p95_ms: evidence_config.slo_p95_ms,
            error_rate: evidence_config.slo_error_rate,
            required_successful_runs: evidence_config.runs,
        },
        variability: VariabilityBoundary {
            provider_class: "controlled_local_mock".to_string(),
            includes_external_provider_variability: false,
            interpretation: "Measures router plus loopback mock overhead; it does not substantiate latency claims for external providers.".to_string(),
        },
        resource_observation: peak_rss_observation(),
        runs,
        aggregates,
        pass: failures.is_empty(),
        failures,
        replay_command: replay_command.clone(),
    };

    fs::create_dir_all(&evidence_config.output_dir).with_context(|| {
        format!(
            "failed to create evidence directory {}",
            evidence_config.output_dir.display()
        )
    })?;
    let benchmark_bytes = pretty_json(&benchmark)?;
    let conformance_bytes = pretty_json(&conformance)?;
    fs::write(
        evidence_config.output_dir.join(BENCHMARK_FILE),
        &benchmark_bytes,
    )?;
    fs::write(
        evidence_config.output_dir.join(CONFORMANCE_FILE),
        &conformance_bytes,
    )?;
    let manifest = ControlledEvidenceManifest {
        schema_version: 1,
        artifact_kind: "controlled_release_evidence_manifest".to_string(),
        generated_unix_seconds: unix_seconds(),
        source_revision: evidence_config.revision,
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: conformance.config_fnv1a_64,
        benchmark_file: BENCHMARK_FILE.to_string(),
        benchmark_fnv1a_64: format!("{:016x}", fnv1a_64(&benchmark_bytes)),
        conformance_file: CONFORMANCE_FILE.to_string(),
        conformance_fnv1a_64: format!("{:016x}", fnv1a_64(&conformance_bytes)),
        provider_class: "controlled_local_mock".to_string(),
        includes_external_provider_variability: false,
        pass: benchmark.pass && conformance.pass,
        replay_command,
    };
    fs::write(
        evidence_config.output_dir.join(MANIFEST_FILE),
        pretty_json(&manifest)?,
    )?;
    validate_evidence_bundle(&evidence_config.output_dir, Some(&manifest.source_revision))?;
    Ok(manifest)
}

pub fn validate_evidence_bundle(directory: &Path, expected_revision: Option<&str>) -> Result<()> {
    let benchmark_path = directory.join(BENCHMARK_FILE);
    let conformance_path = directory.join(CONFORMANCE_FILE);
    let manifest_path = directory.join(MANIFEST_FILE);
    let benchmark_bytes = fs::read(&benchmark_path)
        .with_context(|| format!("failed to read {}", benchmark_path.display()))?;
    let conformance_bytes = fs::read(&conformance_path)
        .with_context(|| format!("failed to read {}", conformance_path.display()))?;
    let manifest = serde_json::from_slice::<ControlledEvidenceManifest>(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )?;
    let benchmark = serde_json::from_slice::<ControlledBenchmarkArtifact>(&benchmark_bytes)?;
    let conformance = serde_json::from_slice::<Value>(&conformance_bytes)?;

    anyhow::ensure!(
        manifest.schema_version == 1,
        "unsupported evidence manifest schema"
    );
    anyhow::ensure!(
        benchmark.schema_version == 1,
        "unsupported benchmark schema"
    );
    anyhow::ensure!(
        manifest.pass && benchmark.pass,
        "evidence bundle did not pass"
    );
    anyhow::ensure!(
        !manifest.includes_external_provider_variability
            && !benchmark.variability.includes_external_provider_variability,
        "controlled evidence must not claim external-provider coverage"
    );
    anyhow::ensure!(
        benchmark.runs.len() >= 2,
        "benchmark requires repeated runs"
    );
    anyhow::ensure!(
        benchmark.workload.measured_runs == benchmark.runs.len(),
        "benchmark run count does not match workload metadata"
    );
    anyhow::ensure!(
        benchmark
            .aggregates
            .iter()
            .all(|aggregate| aggregate.pass && aggregate.runs == benchmark.runs.len()),
        "benchmark aggregate failed or omitted a run"
    );
    anyhow::ensure!(
        conformance.get("schema_version").and_then(Value::as_u64) == Some(2),
        "controlled conformance artifact must use schema version 2"
    );
    anyhow::ensure!(
        conformance.get("pass").and_then(Value::as_bool) == Some(true),
        "controlled conformance artifact did not pass"
    );
    anyhow::ensure!(
        conformance
            .get("reports")
            .and_then(Value::as_array)
            .is_some_and(|reports| !reports.is_empty()),
        "controlled conformance artifact has no reports"
    );
    anyhow::ensure!(
        manifest.benchmark_fnv1a_64 == format!("{:016x}", fnv1a_64(&benchmark_bytes)),
        "benchmark fingerprint does not match manifest"
    );
    anyhow::ensure!(
        manifest.conformance_fnv1a_64 == format!("{:016x}", fnv1a_64(&conformance_bytes)),
        "conformance fingerprint does not match manifest"
    );
    anyhow::ensure!(
        manifest.source_revision == benchmark.source_revision,
        "evidence artifacts disagree on source revision"
    );
    anyhow::ensure!(
        manifest.config_fnv1a_64 == benchmark.config_fnv1a_64
            && manifest.config_fnv1a_64
                == conformance
                    .get("config_fnv1a_64")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
        "evidence artifacts disagree on config fingerprint"
    );
    if let Some(expected) = expected_revision {
        anyhow::ensure!(
            manifest.source_revision == expected,
            "evidence revision {} does not match expected {expected}",
            manifest.source_revision
        );
    }
    Ok(())
}

fn validate_controlled_config(config: &ControlledEvidenceConfig) -> Result<()> {
    anyhow::ensure!(
        !config.revision.trim().is_empty(),
        "revision must not be empty"
    );
    anyhow::ensure!(
        config.runs >= 2,
        "controlled evidence requires at least two runs"
    );
    anyhow::ensure!(
        config.requests_per_scenario > 0,
        "requests must be non-zero"
    );
    anyhow::ensure!(config.concurrency > 0, "concurrency must be non-zero");
    anyhow::ensure!(
        (0.0..=1.0).contains(&config.slo_error_rate),
        "error-rate threshold must be between zero and one"
    );
    Ok(())
}

fn aggregate_runs(
    runs: &[LoadSuiteReport],
    scenarios: &[crate::load::LoadSuiteScenario],
    config: &ControlledEvidenceConfig,
) -> Result<Vec<ScenarioAggregate>> {
    scenarios
        .iter()
        .map(|scenario| {
            let reports = runs
                .iter()
                .map(|run| {
                    run.reports
                        .iter()
                        .find(|entry| entry.scenario == scenario.name)
                        .with_context(|| format!("run omitted scenario {}", scenario.name))
                })
                .collect::<Result<Vec<_>>>()?;
            let p95 = reports
                .iter()
                .map(|entry| entry.report.latency.p95_ms as f64)
                .collect::<Vec<_>>();
            let p99 = reports
                .iter()
                .map(|entry| entry.report.latency.p99_ms as f64)
                .collect::<Vec<_>>();
            let throughput = reports
                .iter()
                .map(|entry| entry.report.requests_per_second)
                .collect::<Vec<_>>();
            let maximum_error_rate = reports
                .iter()
                .map(|entry| entry.report.error_rate)
                .fold(0.0_f64, f64::max);
            let p95_stat = repeat_statistic(&p95);
            Ok(ScenarioAggregate {
                scenario: scenario.name.clone(),
                path: scenario.path.clone(),
                runs: reports.len(),
                pass: p95_stat.maximum <= config.slo_p95_ms as f64
                    && maximum_error_rate <= config.slo_error_rate,
                p95_ms: p95_stat,
                p99_ms: repeat_statistic(&p99),
                requests_per_second: repeat_statistic(&throughput),
                maximum_error_rate,
            })
        })
        .collect()
}

fn repeat_statistic(values: &[f64]) -> RepeatStatistic {
    if values.is_empty() {
        return RepeatStatistic {
            minimum: 0.0,
            mean: 0.0,
            maximum: 0.0,
            confidence_95_lower: 0.0,
            confidence_95_upper: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = if values.len() > 1 {
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64
    } else {
        0.0
    };
    let margin = 1.96 * variance.sqrt() / (values.len() as f64).sqrt();
    RepeatStatistic {
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        mean,
        maximum: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        confidence_95_lower: (mean - margin).max(0.0),
        confidence_95_upper: mean + margin,
    }
}

fn peak_rss_observation() -> ResourceObservation {
    #[cfg(target_os = "linux")]
    {
        let peak = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    let value = line.strip_prefix("VmHWM:")?.trim();
                    let kib = value.split_whitespace().next()?.parse::<u64>().ok()?;
                    Some(kib.saturating_mul(1024))
                })
            });
        return ResourceObservation {
            peak_rss_bytes: peak,
            method: "linux_proc_self_status_vmhwm".to_string(),
        };
    }
    #[cfg(not(target_os = "linux"))]
    ResourceObservation {
        peak_rss_bytes: None,
        method: "unavailable_on_this_platform".to_string(),
    }
}

fn replay_command(config: &ControlledEvidenceConfig) -> String {
    format!(
        "cargo run --locked -- controlled-evidence --revision {} --runs {} --requests-per-scenario {} --concurrency {} --slo-p95-ms {} --slo-error-rate {} --output-dir {}",
        shell_word(&config.revision),
        config.runs,
        config.requests_per_scenario,
        config.concurrency,
        config.slo_p95_ms,
        config.slo_error_rate,
        shell_word(&config.output_dir.display().to_string())
    )
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "./_:-".contains(character))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn pretty_json(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
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

async fn spawn_controlled_provider() -> Result<(String, JoinHandle<()>)> {
    async fn chat(Json(request): Json<Value>) -> axum::response::Response {
        if request.get("stream").and_then(Value::as_bool) == Some(true) {
            return Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header("x-provider-version", "controlled-provider-1")
                .header("x-model-version", "controlled-model-1")
                .body(Body::from(concat!(
                    "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
                    "data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
                    "data: [DONE]\n\n"
                )))
                .unwrap();
        }
        let (content, tool_calls) = if request.get("tools").is_some() {
            (
                Value::Null,
                json!([{
                    "id": "controlled_call",
                    "type": "function",
                    "function": {
                        "name": "conformance_echo",
                        "arguments": "{\"text\":\"ok\"}"
                    }
                }]),
            )
        } else if request.get("response_format").is_some() {
            (Value::String("{\"ok\":true}".to_string()), Value::Null)
        } else {
            (Value::String("ok".to_string()), Value::Null)
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !tool_calls.is_null() {
            message["tool_calls"] = tool_calls;
        }
        (
            [
                ("x-provider-version", "controlled-provider-1"),
                ("x-model-version", "controlled-model-1"),
            ],
            Json(json!({
                "id": "chat_controlled",
                "object": "chat.completion",
                "model": request["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })),
        )
            .into_response()
    }

    async fn responses(Json(request): Json<Value>) -> Json<Value> {
        Json(json!({
            "id": "resp_controlled",
            "object": "response",
            "model": request["model"],
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}]
            }],
            "output_text": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        }))
    }

    async fn embeddings(Json(request): Json<Value>) -> Json<Value> {
        Json(json!({
            "object": "list",
            "model": request["model"],
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        }))
    }

    async fn images(Json(_request): Json<Value>) -> Json<Value> {
        Json(json!({
            "created": 1,
            "data": [{"url": "https://example.test/controlled.png"}]
        }))
    }

    async fn speech(Json(_request): Json<Value>) -> axum::response::Response {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/wav")
            .body(Body::from("RIFFcontrolled-audio"))
            .unwrap()
    }

    async fn audio(mut multipart: Multipart) -> Json<Value> {
        while multipart.next_field().await.ok().flatten().is_some() {}
        Json(json!({"text": "controlled audio"}))
    }

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/images/generations", post(images))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/transcriptions", post(audio))
        .route("/v1/audio/translations", post(audio));
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Ok((format!("http://{address}"), task))
}

fn controlled_router_config(provider_url: &str) -> Result<RouterConfig> {
    let yaml = format!(
        r#"
bind: 127.0.0.1:0
default_model: controlled-model
policy: balanced
providers:
  - name: controlled-provider
    kind: open_ai_compatible
    base_url: {provider_url}
    responses_path: /v1/responses
    embeddings_path: /v1/embeddings
    images_path: /v1/images/generations
    speech_path: /v1/audio/speech
    audio_transcriptions_path: /v1/audio/transcriptions
    audio_translations_path: /v1/audio/translations
    retries: 0
    timeout_ms: 2000
models:
  - id: controlled-model
    provider: controlled-provider
    capability: 0.70
    cost_per_million_input: 0.10
    cost_per_million_output: 0.10
    context_window: 8192
    capabilities:
      supports_vision: true
      supports_audio: true
      supports_tools: true
      supports_json: true
      supports_code: true
      supports_web_apps: true
      supports_long_context: true
      supported_endpoints: [chat, responses, embeddings, images, speech, audio_transcriptions, audio_translations]
"#
    );
    let config = serde_yaml::from_str::<RouterConfig>(&yaml)
        .context("failed to build controlled evidence config")?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_statistics_include_repeat_level_confidence_interval() {
        let statistic = repeat_statistic(&[10.0, 20.0, 30.0]);
        assert_eq!(statistic.minimum, 10.0);
        assert_eq!(statistic.mean, 20.0);
        assert_eq!(statistic.maximum, 30.0);
        assert!(statistic.confidence_95_lower < statistic.mean);
        assert!(statistic.confidence_95_upper > statistic.mean);
    }

    #[tokio::test]
    async fn controlled_bundle_is_revision_linked_and_self_validating() {
        let directory =
            std::env::temp_dir().join(format!("router-controlled-evidence-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let manifest = write_controlled_evidence(ControlledEvidenceConfig {
            revision: "test-revision".to_string(),
            output_dir: directory.clone(),
            runs: 2,
            requests_per_scenario: 2,
            concurrency: 2,
            slo_p95_ms: 2_000,
            slo_error_rate: 0.0,
        })
        .await
        .unwrap();

        assert!(manifest.pass);
        assert_eq!(manifest.source_revision, "test-revision");
        assert!(!manifest.includes_external_provider_variability);
        validate_evidence_bundle(&directory, Some("test-revision")).unwrap();

        let benchmark = serde_json::from_slice::<ControlledBenchmarkArtifact>(
            &fs::read(directory.join(BENCHMARK_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(benchmark.runs.len(), 2);
        assert_eq!(benchmark.aggregates.len(), 6);
        assert!(benchmark.aggregates.iter().all(|aggregate| aggregate.pass));
        assert!(
            benchmark
                .runs
                .iter()
                .flat_map(|run| &run.reports)
                .all(|report| report.report.latency.p99_ms >= report.report.latency.p95_ms)
        );

        fs::write(directory.join(BENCHMARK_FILE), b"{}\n").unwrap();
        assert!(validate_evidence_bundle(&directory, Some("test-revision")).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
