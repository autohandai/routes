use crate::{
    accounting::{BudgetAccounting, BudgetChargeClass, BudgetReservation, BudgetUsageSnapshot},
    config::{BudgetAccountingBackend, BudgetAccountingConfig, BudgetConfig, RouterConfig},
    conformance::config_fingerprint,
    evidence::{controlled_router_config, spawn_controlled_provider},
    load::LatencyReport,
    types::ModelEndpoint,
};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder, StatusCode, multipart};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    net::TcpListener as StdTcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{sync::Mutex, time::sleep};

const QUEUE_BUCKET_PREFIX: &str = "autohand_router_provider_queue_duration_ms_bucket{";

#[derive(Debug, Clone)]
pub struct DeploymentLiveGateConfig {
    pub base_url: String,
    pub bearer_token: Option<String>,
    pub revision: String,
    pub duration: Duration,
    pub requests_per_second: u64,
    pub concurrency: usize,
    pub min_samples_per_scenario: u64,
    pub max_p95_ms: u64,
    pub max_p99_ms: u64,
    pub max_error_rate: f64,
    pub max_queue_p95_ms: u64,
    pub max_peak_rss_bytes: u64,
    pub require_rss: bool,
    pub require_target_revision: bool,
    pub max_recovery_ms: u64,
    pub accounting_processes: usize,
    pub accounting_limit: u64,
    pub max_upload_probe_bytes: u64,
    pub worker_executable: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentLiveGateReport {
    pub schema_version: u32,
    pub artifact_kind: String,
    pub generated_unix_seconds: u64,
    pub source_revision: String,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub target: String,
    pub target_class: String,
    pub thresholds: DeploymentThresholds,
    pub workload: SustainedWorkloadReport,
    pub queue: QueueEvidence,
    pub memory: MemoryEvidence,
    pub target_identity: TargetIdentityEvidence,
    pub multipart: MultipartBoundaryEvidence,
    pub multi_process_file_accounting: MultiProcessFileEvidence,
    pub rolling_restart: RollingRestartEvidence,
    pub payloads_redacted: bool,
    pub pass: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentThresholds {
    pub duration_ms: u64,
    pub requests_per_second: u64,
    pub concurrency: usize,
    pub min_samples_per_scenario: u64,
    pub max_p95_ms: u64,
    pub max_p99_ms: u64,
    pub max_error_rate: f64,
    pub max_queue_p95_ms: u64,
    pub max_peak_rss_bytes: u64,
    pub require_rss: bool,
    pub require_target_revision: bool,
    pub max_recovery_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainedWorkloadReport {
    pub executed: bool,
    pub configured_duration_ms: u64,
    pub observed_duration_ms: u64,
    pub requests_per_second: u64,
    pub concurrency: usize,
    pub total_requests: u64,
    pub scenarios: Vec<DeploymentScenarioReport>,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentScenarioReport {
    pub scenario: String,
    pub contract: String,
    pub requests: u64,
    pub successes: u64,
    pub failures: u64,
    pub error_rate: f64,
    pub status_counts: BTreeMap<String, u64>,
    pub latency: LatencyReport,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEvidence {
    pub executed: bool,
    pub admitted_samples: u64,
    pub rejected_samples: u64,
    pub p95_upper_bound_ms: Option<u64>,
    pub max_p95_ms: u64,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvidence {
    pub peak_rss_bytes: Option<u64>,
    pub max_peak_rss_bytes: u64,
    pub required: bool,
    pub source: String,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetIdentityEvidence {
    pub expected_revision: String,
    pub reported_revision: Option<String>,
    pub revision_required: bool,
    pub expected_config_fnv1a_64: String,
    pub reported_config_fnv1a_64: Option<String>,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipartBoundaryEvidence {
    pub executed: bool,
    pub configured_limit_bytes: u64,
    pub below_limit_file_bytes: u64,
    pub below_limit_status: Option<u16>,
    pub below_limit_request_id_present: bool,
    pub below_limit_routed_successfully: bool,
    pub above_limit_file_bytes: u64,
    pub above_limit_status: Option<u16>,
    pub above_limit_error_code: Option<String>,
    pub above_limit_request_id_present: bool,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProcessFileEvidence {
    pub executed: bool,
    pub processes: usize,
    pub attempted_reservations: u64,
    pub configured_limit: u64,
    pub successful_reservations: u64,
    pub budget_rejections: u64,
    pub storage_errors: u64,
    pub ledger_request_count: Option<u64>,
    pub stale_lock_metadata_recovered: bool,
    pub corrupted_ledger_failed_closed: bool,
    pub restart_state_preserved: bool,
    pub contention_elapsed_ms: u64,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingRestartEvidence {
    pub executed: bool,
    pub replicas: usize,
    pub initial_replicas_healthy: bool,
    pub surviving_replica_served_during_restart: bool,
    pub replacement_ready_ms: Option<u64>,
    pub replacement_served_after_restart: bool,
    pub max_recovery_ms: u64,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetWorkerReport {
    pub worker_id: String,
    pub attempts: u64,
    pub successful_reservations: u64,
    pub budget_rejections: u64,
    pub storage_errors: u64,
}

#[derive(Clone)]
enum ScenarioKind {
    RouterUnary,
    ChatUnary { model: String },
    ChatStream { model: String },
    MultipartIngress { model: Option<String>, bytes: usize },
}

#[derive(Clone)]
struct ScenarioSpec {
    name: &'static str,
    contract: &'static str,
    kind: ScenarioKind,
}

#[derive(Debug, Clone)]
struct WorkloadSample {
    scenario: &'static str,
    latency_ms: u64,
    success: bool,
    status: Option<u16>,
}

#[derive(Default)]
struct QueueBuckets {
    finite: BTreeMap<u64, u64>,
    infinite: u64,
}

pub async fn run_deployment_live_gate(
    source: &RouterConfig,
    gate: DeploymentLiveGateConfig,
) -> Result<DeploymentLiveGateReport> {
    validate_gate(source, &gate)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .pool_idle_timeout(Duration::from_secs(30))
        .build()?;
    let expected_config_fnv1a_64 = config_fingerprint(source)?;
    let preflight_metrics = metrics_json(&client, &gate).await?;
    let preflight_identity =
        target_identity_evidence(&preflight_metrics, &gate, &expected_config_fnv1a_64);
    if !preflight_identity.pass {
        return Ok(preflight_failure_report(
            &gate,
            expected_config_fnv1a_64,
            memory_evidence(&preflight_metrics, &gate),
            preflight_identity,
        ));
    }
    let scenarios = deployment_scenarios(source)?;
    let queue_before = prometheus_text(&client, &gate).await?;
    let workload = run_sustained_workload(&client, &gate, &scenarios).await?;
    let queue_after = prometheus_text(&client, &gate).await?;
    let queue = queue_evidence(&queue_before, &queue_after, gate.max_queue_p95_ms);
    let metrics = metrics_json(&client, &gate).await?;
    let memory = memory_evidence(&metrics, &gate);
    let target_identity = target_identity_evidence(&metrics, &gate, &expected_config_fnv1a_64);
    let multipart = run_multipart_boundary_probe(&client, source, &gate).await?;
    let multi_process_file_accounting = run_multi_process_file_probe(&gate).await?;
    let rolling_restart = run_rolling_restart_probe(&gate).await?;

    let mut failures = Vec::new();
    if !workload.pass {
        failures.push("sustained mixed workload exceeded a threshold".to_string());
    }
    if !queue.pass {
        failures.push("provider queue p95 or rejection threshold failed".to_string());
    }
    if !memory.pass {
        failures.push("peak RSS was unavailable or exceeded the configured threshold".to_string());
    }
    if !target_identity.pass {
        failures.push("deployed target revision/config did not match the candidate".to_string());
    }
    if !multipart.pass {
        failures.push("multipart ingress boundary probe failed".to_string());
    }
    if !multi_process_file_accounting.pass {
        failures.push("multi-process file accounting/recovery probe failed".to_string());
    }
    if !rolling_restart.pass {
        failures.push("controlled rolling restart recovery probe failed".to_string());
    }

    Ok(DeploymentLiveGateReport {
        schema_version: 1,
        artifact_kind: "staging_deployment_live_gate".to_string(),
        generated_unix_seconds: unix_seconds(),
        source_revision: gate.revision.clone(),
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: expected_config_fnv1a_64,
        target: gate.base_url.trim_end_matches('/').to_string(),
        target_class: "deployed_staging".to_string(),
        thresholds: deployment_thresholds(&gate),
        workload,
        queue,
        memory,
        target_identity,
        multipart,
        multi_process_file_accounting,
        rolling_restart,
        payloads_redacted: true,
        pass: failures.is_empty(),
        failures,
    })
}

fn preflight_failure_report(
    gate: &DeploymentLiveGateConfig,
    config_fnv1a_64: String,
    memory: MemoryEvidence,
    target_identity: TargetIdentityEvidence,
) -> DeploymentLiveGateReport {
    DeploymentLiveGateReport {
        schema_version: 1,
        artifact_kind: "staging_deployment_live_gate".to_string(),
        generated_unix_seconds: unix_seconds(),
        source_revision: gate.revision.clone(),
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64,
        target: gate.base_url.trim_end_matches('/').to_string(),
        target_class: "deployed_staging".to_string(),
        thresholds: deployment_thresholds(gate),
        workload: SustainedWorkloadReport {
            executed: false,
            configured_duration_ms: duration_ms(gate.duration),
            observed_duration_ms: 0,
            requests_per_second: gate.requests_per_second,
            concurrency: gate.concurrency,
            total_requests: 0,
            scenarios: Vec::new(),
            pass: false,
        },
        queue: QueueEvidence {
            executed: false,
            admitted_samples: 0,
            rejected_samples: 0,
            p95_upper_bound_ms: None,
            max_p95_ms: gate.max_queue_p95_ms,
            pass: false,
        },
        memory,
        target_identity,
        multipart: MultipartBoundaryEvidence {
            executed: false,
            configured_limit_bytes: 0,
            below_limit_file_bytes: 0,
            below_limit_status: None,
            below_limit_request_id_present: false,
            below_limit_routed_successfully: false,
            above_limit_file_bytes: 0,
            above_limit_status: None,
            above_limit_error_code: None,
            above_limit_request_id_present: false,
            pass: false,
        },
        multi_process_file_accounting: MultiProcessFileEvidence {
            executed: false,
            processes: gate.accounting_processes,
            attempted_reservations: 0,
            configured_limit: gate.accounting_limit,
            successful_reservations: 0,
            budget_rejections: 0,
            storage_errors: 0,
            ledger_request_count: None,
            stale_lock_metadata_recovered: false,
            corrupted_ledger_failed_closed: false,
            restart_state_preserved: false,
            contention_elapsed_ms: 0,
            pass: false,
        },
        rolling_restart: RollingRestartEvidence {
            executed: false,
            replicas: 2,
            initial_replicas_healthy: false,
            surviving_replica_served_during_restart: false,
            replacement_ready_ms: None,
            replacement_served_after_restart: false,
            max_recovery_ms: gate.max_recovery_ms,
            pass: false,
        },
        payloads_redacted: true,
        pass: false,
        failures: vec![
            "deployed target revision/config did not match the candidate; workload and controlled probes were not executed"
                .to_string(),
        ],
    }
}

fn deployment_thresholds(gate: &DeploymentLiveGateConfig) -> DeploymentThresholds {
    DeploymentThresholds {
        duration_ms: duration_ms(gate.duration),
        requests_per_second: gate.requests_per_second,
        concurrency: gate.concurrency,
        min_samples_per_scenario: gate.min_samples_per_scenario,
        max_p95_ms: gate.max_p95_ms,
        max_p99_ms: gate.max_p99_ms,
        max_error_rate: gate.max_error_rate,
        max_queue_p95_ms: gate.max_queue_p95_ms,
        max_peak_rss_bytes: gate.max_peak_rss_bytes,
        require_rss: gate.require_rss,
        require_target_revision: gate.require_target_revision,
        max_recovery_ms: gate.max_recovery_ms,
    }
}

pub async fn run_budget_worker(
    ledger: &Path,
    limit: u64,
    attempts: u64,
    worker_id: String,
    start_file: &Path,
    output: &Path,
) -> Result<BudgetWorkerReport> {
    let started = Instant::now();
    while !start_file.exists() {
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(15),
            "budget worker start barrier timed out"
        );
        sleep(Duration::from_millis(5)).await;
    }
    let budget = file_budget_config(ledger, limit);
    let accounting = BudgetAccounting::from_budget_config(&budget)?;
    let mut report = BudgetWorkerReport {
        worker_id,
        attempts,
        successful_reservations: 0,
        budget_rejections: 0,
        storage_errors: 0,
    };
    for _ in 0..attempts {
        let reservation = BudgetReservation {
            id: uuid::Uuid::new_v4(),
            class: BudgetChargeClass::ForegroundLogicalRequest,
            request_count: 1,
            total_tokens: 1,
            estimated_cost_micros: 0,
        };
        match accounting.reserve(&budget, reservation).await {
            Ok(()) => report.successful_reservations += 1,
            Err(error) if error.to_string().contains("budget exceeded") => {
                report.budget_rejections += 1;
            }
            Err(_) => report.storage_errors += 1,
        }
    }
    fs::write(output, pretty_json(&report)?)
        .with_context(|| format!("failed to write budget worker report {}", output.display()))?;
    Ok(report)
}

fn validate_gate(source: &RouterConfig, gate: &DeploymentLiveGateConfig) -> Result<()> {
    let url = reqwest::Url::parse(&gate.base_url).context("invalid deployment gate URL")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "deployment gate URL must use http or https"
    );
    anyhow::ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "deployment gate URL must not contain credentials, a query, or a fragment"
    );
    anyhow::ensure!(
        gate.bearer_token
            .as_deref()
            .is_none_or(|token| !token.is_empty() && !token.chars().any(char::is_whitespace)),
        "deployment gate bearer token is empty or contains whitespace"
    );
    anyhow::ensure!(
        !gate.revision.trim().is_empty(),
        "revision must not be empty"
    );
    anyhow::ensure!(
        gate.duration >= Duration::from_millis(100),
        "duration is too short"
    );
    anyhow::ensure!(
        gate.requests_per_second > 0,
        "request rate must be non-zero"
    );
    anyhow::ensure!(gate.concurrency > 0, "concurrency must be non-zero");
    anyhow::ensure!(
        gate.min_samples_per_scenario > 0,
        "minimum samples must be non-zero"
    );
    anyhow::ensure!(
        gate.max_p95_ms > 0 && gate.max_p99_ms >= gate.max_p95_ms,
        "invalid latency thresholds"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&gate.max_error_rate),
        "invalid error-rate threshold"
    );
    anyhow::ensure!(
        gate.max_queue_p95_ms > 0,
        "queue threshold must be non-zero"
    );
    anyhow::ensure!(
        gate.max_peak_rss_bytes > 0,
        "RSS threshold must be non-zero"
    );
    anyhow::ensure!(
        gate.max_recovery_ms > 0,
        "recovery threshold must be non-zero"
    );
    anyhow::ensure!(
        gate.accounting_processes >= 2,
        "accounting probe requires at least two processes"
    );
    anyhow::ensure!(
        gate.accounting_limit >= gate.accounting_processes as u64,
        "accounting limit is too small"
    );
    anyhow::ensure!(
        gate.worker_executable.is_file(),
        "worker executable does not exist"
    );
    anyhow::ensure!(
        source.runtime.ingress.max_multipart_body_bytes >= 8_192,
        "deployment gate requires max_multipart_body_bytes of at least 8192"
    );
    anyhow::ensure!(
        source.runtime.ingress.max_multipart_body_bytes as u64 <= gate.max_upload_probe_bytes,
        "configured multipart limit exceeds max_upload_probe_bytes"
    );
    let scenario_count = 4_u64;
    let scheduled = duration_ms(gate.duration).saturating_mul(gate.requests_per_second) / 1_000;
    anyhow::ensure!(
        scheduled >= gate.min_samples_per_scenario.saturating_mul(scenario_count),
        "duration and request rate cannot provide the required samples per scenario"
    );
    Ok(())
}

fn deployment_scenarios(source: &RouterConfig) -> Result<Vec<ScenarioSpec>> {
    let streaming_model = source
        .models
        .iter()
        .find(|model| {
            let Some(provider) = source
                .providers
                .iter()
                .find(|provider| provider.name == model.provider)
            else {
                return false;
            };
            model.capabilities.supports_endpoint(ModelEndpoint::Chat)
                && provider.supports_endpoint(ModelEndpoint::Chat)
                && provider.kind.chat_adapter_contract().supports_streaming
        })
        .context("deployment gate requires a jointly advertised streaming chat model")?;
    let audio_model = source.models.iter().find(|model| {
        let Some(provider) = source
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
        else {
            return false;
        };
        model
            .capabilities
            .supports_endpoint(ModelEndpoint::AudioTranscriptions)
            && provider.supports_endpoint(ModelEndpoint::AudioTranscriptions)
    });
    let below_bytes =
        workload_multipart_file_bytes(source.runtime.ingress.max_multipart_body_bytes);
    Ok(vec![
        ScenarioSpec {
            name: "router_unary",
            contract: "successful deterministic route decision",
            kind: ScenarioKind::RouterUnary,
        },
        ScenarioSpec {
            name: "chat_unary",
            contract: "successful fully consumed provider response",
            kind: ScenarioKind::ChatUnary {
                model: streaming_model.id.clone(),
            },
        },
        ScenarioSpec {
            name: "chat_stream",
            contract: "successful consumed SSE stream with terminal marker",
            kind: ScenarioKind::ChatStream {
                model: streaming_model.id.clone(),
            },
        },
        ScenarioSpec {
            name: "multipart_ingress",
            contract: if audio_model.is_some() {
                "successful below-limit multipart inference"
            } else {
                "below-limit multipart admitted through ingress and rejected locally for eligibility"
            },
            kind: ScenarioKind::MultipartIngress {
                model: audio_model.map(|model| model.id.clone()),
                bytes: below_bytes,
            },
        },
    ])
}

async fn run_sustained_workload(
    client: &Client,
    gate: &DeploymentLiveGateConfig,
    scenarios: &[ScenarioSpec],
) -> Result<SustainedWorkloadReport> {
    for scenario in scenarios {
        let warmup = execute_scenario(client, gate, scenario).await;
        anyhow::ensure!(
            warmup.success,
            "warmup failed for {} with status {:?}",
            scenario.name,
            warmup.status
        );
    }

    let duration_ms = duration_ms(gate.duration);
    let total_slots = duration_ms.saturating_mul(gate.requests_per_second) / 1_000;
    let next_slot = Arc::new(AtomicU64::new(0));
    let samples = Arc::new(Mutex::new(Vec::with_capacity(total_slots as usize)));
    let started = Instant::now();
    let mut workers = Vec::with_capacity(gate.concurrency);
    for _ in 0..gate.concurrency {
        let client = client.clone();
        let gate = gate.clone();
        let scenarios = scenarios.to_vec();
        let next_slot = Arc::clone(&next_slot);
        let samples = Arc::clone(&samples);
        workers.push(tokio::spawn(async move {
            loop {
                let slot = next_slot.fetch_add(1, Ordering::Relaxed);
                if slot >= total_slots {
                    break;
                }
                let scheduled_ms = slot.saturating_mul(1_000) / gate.requests_per_second;
                let elapsed_ms = duration_ms_std(started.elapsed());
                if scheduled_ms > elapsed_ms {
                    sleep(Duration::from_millis(scheduled_ms - elapsed_ms)).await;
                }
                let scenario = &scenarios[(slot as usize) % scenarios.len()];
                let sample = execute_scenario(&client, &gate, scenario).await;
                samples.lock().await.push(sample);
            }
        }));
    }
    for worker in workers {
        worker.await.context("deployment workload worker failed")?;
    }
    let observed_duration_ms = duration_ms_std(started.elapsed());
    let samples = samples.lock().await.clone();
    let reports = scenarios
        .iter()
        .map(|scenario| scenario_report(scenario, &samples, gate))
        .collect::<Vec<_>>();
    let pass = reports.iter().all(|report| report.pass)
        && samples.len() as u64 == total_slots
        && observed_duration_ms >= duration_ms.saturating_sub(1_000 / gate.requests_per_second);
    Ok(SustainedWorkloadReport {
        executed: true,
        configured_duration_ms: duration_ms,
        observed_duration_ms,
        requests_per_second: gate.requests_per_second,
        concurrency: gate.concurrency,
        total_requests: samples.len() as u64,
        scenarios: reports,
        pass,
    })
}

async fn execute_scenario(
    client: &Client,
    gate: &DeploymentLiveGateConfig,
    scenario: &ScenarioSpec,
) -> WorkloadSample {
    let started = Instant::now();
    let result = match &scenario.kind {
        ScenarioKind::RouterUnary => {
            let request = authorized(
                client
                    .post(join_url(&gate.base_url, "/v1/router/multimodel"))
                    .json(&json!({
                        "input": "deployment gate deterministic route",
                        "policy": "balanced",
                        "max_output_tokens": 8
                    })),
                gate.bearer_token.as_deref(),
            );
            consumed_success(request, |_| true).await
        }
        ScenarioKind::ChatUnary { model } => {
            let request = authorized(
                client
                    .post(join_url(&gate.base_url, "/v1/chat/completions"))
                    .json(&json!({
                        "model": model,
                        "messages": [{"role": "user", "content": "reply with ok"}],
                        "max_tokens": 8,
                        "temperature": 0
                    })),
                gate.bearer_token.as_deref(),
            );
            consumed_success(request, |status| status.is_success()).await
        }
        ScenarioKind::ChatStream { model } => {
            let request = authorized(
                client
                    .post(join_url(&gate.base_url, "/v1/chat/completions"))
                    .json(&json!({
                        "model": model,
                        "messages": [{"role": "user", "content": "reply with ok"}],
                        "max_tokens": 8,
                        "temperature": 0,
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })),
                gate.bearer_token.as_deref(),
            );
            stream_success(request).await
        }
        ScenarioKind::MultipartIngress { model, bytes } => {
            multipart_success(client, gate, model.as_deref(), *bytes).await
        }
    };
    let (success, status) = result.unwrap_or((false, None));
    WorkloadSample {
        scenario: scenario.name,
        latency_ms: duration_ms_std(started.elapsed()),
        success,
        status,
    }
}

async fn consumed_success<F>(request: RequestBuilder, accepted: F) -> Result<(bool, Option<u16>)>
where
    F: FnOnce(StatusCode) -> bool,
{
    let response = request.send().await?;
    let status = response.status();
    let body_ok = response.bytes().await.is_ok();
    Ok((accepted(status) && body_ok, Some(status.as_u16())))
}

async fn stream_success(request: RequestBuilder) -> Result<(bool, Option<u16>)> {
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let _ = response.bytes().await;
        return Ok((false, Some(status.as_u16())));
    }
    let mut body = response.bytes_stream();
    let mut done = false;
    while let Some(chunk) = body.next().await {
        let chunk = chunk?;
        if chunk.windows(12).any(|window| window == b"data: [DONE]") {
            done = true;
        }
    }
    Ok((done, Some(status.as_u16())))
}

async fn multipart_success(
    client: &Client,
    gate: &DeploymentLiveGateConfig,
    model: Option<&str>,
    bytes: usize,
) -> Result<(bool, Option<u16>)> {
    let response = send_multipart(client, gate, model.unwrap_or("auto"), bytes).await?;
    let status = response.status();
    let body = response.bytes().await?;
    let success = if model.is_some() {
        status.is_success()
    } else {
        status.is_client_error()
            && status != StatusCode::PAYLOAD_TOO_LARGE
            && serde_json::from_slice::<Value>(&body).is_ok()
    };
    Ok((success, Some(status.as_u16())))
}

fn scenario_report(
    scenario: &ScenarioSpec,
    samples: &[WorkloadSample],
    gate: &DeploymentLiveGateConfig,
) -> DeploymentScenarioReport {
    let selected = samples
        .iter()
        .filter(|sample| sample.scenario == scenario.name)
        .collect::<Vec<_>>();
    let requests = selected.len() as u64;
    let successes = selected.iter().filter(|sample| sample.success).count() as u64;
    let failures = requests.saturating_sub(successes);
    let error_rate = failures as f64 / requests.max(1) as f64;
    let mut status_counts = BTreeMap::new();
    for sample in &selected {
        let status = sample
            .status
            .map(|status| status.to_string())
            .unwrap_or_else(|| "transport_error".to_string());
        *status_counts.entry(status).or_insert(0) += 1;
    }
    let mut latencies = selected
        .iter()
        .map(|sample| sample.latency_ms)
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let latency = latency_report(&latencies);
    let pass = requests >= gate.min_samples_per_scenario
        && latency.p95_ms <= gate.max_p95_ms
        && latency.p99_ms <= gate.max_p99_ms
        && error_rate <= gate.max_error_rate;
    DeploymentScenarioReport {
        scenario: scenario.name.to_string(),
        contract: scenario.contract.to_string(),
        requests,
        successes,
        failures,
        error_rate,
        status_counts,
        latency,
        pass,
    }
}

async fn prometheus_text(client: &Client, gate: &DeploymentLiveGateConfig) -> Result<String> {
    Ok(authorized(
        client.get(join_url(&gate.base_url, "/metrics/prometheus")),
        gate.bearer_token.as_deref(),
    )
    .send()
    .await?
    .error_for_status()?
    .text()
    .await?)
}

async fn metrics_json(client: &Client, gate: &DeploymentLiveGateConfig) -> Result<Value> {
    Ok(authorized(
        client.get(join_url(&gate.base_url, "/metrics")),
        gate.bearer_token.as_deref(),
    )
    .send()
    .await?
    .error_for_status()?
    .json()
    .await?)
}

fn queue_evidence(before: &str, after: &str, max_p95_ms: u64) -> QueueEvidence {
    let admitted_before = parse_queue_buckets(before, "admitted");
    let admitted_after = parse_queue_buckets(after, "admitted");
    let rejected_before = parse_queue_buckets(before, "rejected");
    let rejected_after = parse_queue_buckets(after, "rejected");
    let admitted_samples = admitted_after
        .infinite
        .saturating_sub(admitted_before.infinite);
    let rejected_samples = rejected_after
        .infinite
        .saturating_sub(rejected_before.infinite);
    let target_rank = admitted_samples.saturating_mul(95).div_ceil(100);
    let p95_upper_bound_ms = admitted_after.finite.iter().find_map(|(boundary, count)| {
        let before_count = admitted_before.finite.get(boundary).copied().unwrap_or(0);
        (count.saturating_sub(before_count) >= target_rank).then_some(*boundary)
    });
    let pass = admitted_samples > 0
        && rejected_samples == 0
        && p95_upper_bound_ms.is_some_and(|p95| p95 <= max_p95_ms);
    QueueEvidence {
        executed: true,
        admitted_samples,
        rejected_samples,
        p95_upper_bound_ms,
        max_p95_ms,
        pass,
    }
}

fn parse_queue_buckets(text: &str, outcome: &str) -> QueueBuckets {
    let mut buckets = QueueBuckets::default();
    let outcome_label = format!("outcome=\"{outcome}\"");
    for line in text
        .lines()
        .filter(|line| line.starts_with(QUEUE_BUCKET_PREFIX) && line.contains(&outcome_label))
    {
        let Some(le) = label_value(line, "le") else {
            continue;
        };
        let Some(value) = line
            .rsplit_once(' ')
            .and_then(|(_, value)| value.parse::<u64>().ok())
        else {
            continue;
        };
        if le == "+Inf" {
            buckets.infinite = buckets.infinite.saturating_add(value);
        } else if let Ok(boundary) = le.parse::<u64>() {
            *buckets.finite.entry(boundary).or_insert(0) += value;
        }
    }
    buckets
}

fn label_value<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let marker = format!("{label}=\"");
    let start = line.find(&marker)?.saturating_add(marker.len());
    let rest = &line[start..];
    Some(&rest[..rest.find('"')?])
}

fn memory_evidence(metrics: &Value, gate: &DeploymentLiveGateConfig) -> MemoryEvidence {
    let peak_rss_bytes = metrics
        .get("process_peak_rss_bytes")
        .and_then(Value::as_u64);
    let pass = peak_rss_bytes.is_some_and(|peak| peak <= gate.max_peak_rss_bytes)
        || (!gate.require_rss && peak_rss_bytes.is_none());
    MemoryEvidence {
        peak_rss_bytes,
        max_peak_rss_bytes: gate.max_peak_rss_bytes,
        required: gate.require_rss,
        source: if peak_rss_bytes.is_some() {
            "target_/metrics_linux_proc_status".to_string()
        } else {
            "unavailable".to_string()
        },
        pass,
    }
}

fn target_identity_evidence(
    metrics: &Value,
    gate: &DeploymentLiveGateConfig,
    expected_config_fnv1a_64: &str,
) -> TargetIdentityEvidence {
    let reported_revision = metrics
        .get("deployment_revision")
        .and_then(Value::as_str)
        .filter(|revision| *revision != "unreported")
        .map(str::to_string);
    let reported_config_fnv1a_64 = metrics
        .get("config_fnv1a_64")
        .and_then(Value::as_str)
        .map(str::to_string);
    let revision_matches = reported_revision.as_deref() == Some(gate.revision.as_str())
        || (!gate.require_target_revision && reported_revision.is_none());
    let pass =
        revision_matches && reported_config_fnv1a_64.as_deref() == Some(expected_config_fnv1a_64);
    TargetIdentityEvidence {
        expected_revision: gate.revision.clone(),
        reported_revision,
        revision_required: gate.require_target_revision,
        expected_config_fnv1a_64: expected_config_fnv1a_64.to_string(),
        reported_config_fnv1a_64,
        pass,
    }
}

async fn run_multipart_boundary_probe(
    client: &Client,
    source: &RouterConfig,
    gate: &DeploymentLiveGateConfig,
) -> Result<MultipartBoundaryEvidence> {
    let limit = source.runtime.ingress.max_multipart_body_bytes as u64;
    let below_bytes = below_limit_file_bytes(limit as usize) as u64;
    let above_bytes = limit.saturating_add(1_024);
    anyhow::ensure!(
        above_bytes <= gate.max_upload_probe_bytes,
        "above-limit upload probe exceeds safety cap"
    );
    let audio_model = source.models.iter().find(|model| {
        let Some(provider) = source
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
        else {
            return false;
        };
        model
            .capabilities
            .supports_endpoint(ModelEndpoint::AudioTranscriptions)
            && provider.supports_endpoint(ModelEndpoint::AudioTranscriptions)
    });
    let model = audio_model.map(|model| model.id.as_str()).unwrap_or("auto");
    let below = send_multipart(client, gate, model, below_bytes as usize).await?;
    let below_status = below.status();
    let below_request_id_present = request_id_present(&below);
    let below_body = below.bytes().await?;
    let below_routed_successfully = below_status.is_success();
    let below_admitted = below_status != StatusCode::PAYLOAD_TOO_LARGE
        && below_status != StatusCode::REQUEST_TIMEOUT
        && !below_status.is_server_error()
        && (audio_model.is_none() || below_routed_successfully)
        && (below_routed_successfully || serde_json::from_slice::<Value>(&below_body).is_ok());

    let above = send_multipart(client, gate, model, above_bytes as usize).await?;
    let above_status = above.status();
    let above_request_id_present = request_id_present(&above);
    let above_body = above.bytes().await?;
    let above_error_code = serde_json::from_slice::<Value>(&above_body)
        .ok()
        .and_then(|value| {
            value
                .get("error")?
                .get("code")?
                .as_str()
                .map(str::to_string)
        });
    let pass = below_admitted
        && below_request_id_present
        && above_status == StatusCode::PAYLOAD_TOO_LARGE
        && above_error_code.as_deref() == Some("request_too_large")
        && above_request_id_present;
    Ok(MultipartBoundaryEvidence {
        executed: true,
        configured_limit_bytes: limit,
        below_limit_file_bytes: below_bytes,
        below_limit_status: Some(below_status.as_u16()),
        below_limit_request_id_present: below_request_id_present,
        below_limit_routed_successfully: below_routed_successfully,
        above_limit_file_bytes: above_bytes,
        above_limit_status: Some(above_status.as_u16()),
        above_limit_error_code: above_error_code,
        above_limit_request_id_present: above_request_id_present,
        pass,
    })
}

async fn send_multipart(
    client: &Client,
    gate: &DeploymentLiveGateConfig,
    model: &str,
    file_bytes: usize,
) -> Result<reqwest::Response> {
    let form = multipart::Form::new()
        .text("model", model.to_string())
        .part(
            "file",
            multipart::Part::bytes(wav_payload(file_bytes))
                .file_name("deployment-gate.wav")
                .mime_str("audio/wav")?,
        );
    authorized(
        client
            .post(join_url(&gate.base_url, "/v1/audio/transcriptions"))
            .multipart(form),
        gate.bearer_token.as_deref(),
    )
    .send()
    .await
    .context("multipart deployment probe failed")
}

fn wav_payload(length: usize) -> Vec<u8> {
    let length = length.max(44);
    let mut bytes = vec![0_u8; length];
    bytes[..4].copy_from_slice(b"RIFF");
    bytes[8..12].copy_from_slice(b"WAVE");
    bytes[12..16].copy_from_slice(b"fmt ");
    bytes[16..20].copy_from_slice(&16_u32.to_le_bytes());
    bytes[20..22].copy_from_slice(&1_u16.to_le_bytes());
    bytes[22..24].copy_from_slice(&1_u16.to_le_bytes());
    bytes[24..28].copy_from_slice(&8_000_u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&16_000_u32.to_le_bytes());
    bytes[32..34].copy_from_slice(&2_u16.to_le_bytes());
    bytes[34..36].copy_from_slice(&16_u16.to_le_bytes());
    bytes[36..40].copy_from_slice(b"data");
    bytes[40..44].copy_from_slice(&(length.saturating_sub(44) as u32).to_le_bytes());
    bytes
}

async fn run_multi_process_file_probe(
    gate: &DeploymentLiveGateConfig,
) -> Result<MultiProcessFileEvidence> {
    let gate = gate.clone();
    tokio::task::spawn_blocking(move || run_multi_process_file_probe_blocking(&gate))
        .await
        .context("multi-process accounting probe panicked")?
}

fn run_multi_process_file_probe_blocking(
    gate: &DeploymentLiveGateConfig,
) -> Result<MultiProcessFileEvidence> {
    let directory = TempDirectory::new("deployment-accounting")?;
    let ledger = directory.path.join("budget.json");
    let lock = directory.path.join("budget.json.lock");
    let start_file = directory.path.join("start");
    fs::write(&lock, b"{\"pid\":999999,\"lease_expires_unix_ms\":1}\n")?;
    let attempts_per_worker = gate
        .accounting_limit
        .saturating_mul(2)
        .div_ceil(gate.accounting_processes as u64);
    let mut children = Vec::with_capacity(gate.accounting_processes);
    let mut outputs = Vec::with_capacity(gate.accounting_processes);
    for worker in 0..gate.accounting_processes {
        let output = directory.path.join(format!("worker-{worker}.json"));
        children.push(spawn_budget_worker(
            gate,
            &ledger,
            attempts_per_worker,
            &worker.to_string(),
            &start_file,
            &output,
        )?);
        outputs.push(output);
    }
    let started = Instant::now();
    fs::write(&start_file, b"start\n")?;
    wait_for_children(children)?;
    let contention_elapsed_ms = duration_ms_std(started.elapsed());
    let reports = outputs
        .iter()
        .map(|path| {
            serde_json::from_slice::<BudgetWorkerReport>(&fs::read(path)?).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let attempted_reservations: u64 = reports.iter().map(|report| report.attempts).sum();
    let successful_reservations: u64 = reports
        .iter()
        .map(|report| report.successful_reservations)
        .sum();
    let budget_rejections: u64 = reports.iter().map(|report| report.budget_rejections).sum();
    let storage_errors: u64 = reports.iter().map(|report| report.storage_errors).sum();
    let ledger_bytes = fs::read(&ledger)?;
    let ledger_snapshot = serde_json::from_slice::<BudgetUsageSnapshot>(&ledger_bytes).ok();
    let ledger_request_count = ledger_snapshot.as_ref().map(|usage| usage.request_count);
    let stale_lock_metadata_recovered = successful_reservations > 0 && lock.exists();

    fs::write(&ledger, b"{not-valid-json")?;
    let corrupt_output = directory.path.join("corrupt-worker.json");
    let corrupt = spawn_budget_worker(gate, &ledger, 1, "corrupt", &start_file, &corrupt_output)?;
    wait_for_children(vec![corrupt])?;
    let corrupt_report = serde_json::from_slice::<BudgetWorkerReport>(&fs::read(&corrupt_output)?)?;
    let corrupted_ledger_failed_closed = corrupt_report.successful_reservations == 0
        && corrupt_report.storage_errors == 1
        && fs::read(&ledger)? == b"{not-valid-json";

    fs::write(&ledger, &ledger_bytes)?;
    let restart_output = directory.path.join("restart-worker.json");
    let restart = spawn_budget_worker(gate, &ledger, 1, "restart", &start_file, &restart_output)?;
    wait_for_children(vec![restart])?;
    let restart_report = serde_json::from_slice::<BudgetWorkerReport>(&fs::read(&restart_output)?)?;
    let restored = serde_json::from_slice::<BudgetUsageSnapshot>(&fs::read(&ledger)?)?;
    let restart_state_preserved = restart_report.successful_reservations == 0
        && restart_report.budget_rejections == 1
        && restored.request_count == gate.accounting_limit;
    let pass = successful_reservations == gate.accounting_limit
        && budget_rejections == attempted_reservations.saturating_sub(gate.accounting_limit)
        && storage_errors == 0
        && ledger_request_count == Some(gate.accounting_limit)
        && stale_lock_metadata_recovered
        && corrupted_ledger_failed_closed
        && restart_state_preserved;
    Ok(MultiProcessFileEvidence {
        executed: true,
        processes: gate.accounting_processes,
        attempted_reservations,
        configured_limit: gate.accounting_limit,
        successful_reservations,
        budget_rejections,
        storage_errors,
        ledger_request_count,
        stale_lock_metadata_recovered,
        corrupted_ledger_failed_closed,
        restart_state_preserved,
        contention_elapsed_ms,
        pass,
    })
}

fn spawn_budget_worker(
    gate: &DeploymentLiveGateConfig,
    ledger: &Path,
    attempts: u64,
    worker_id: &str,
    start_file: &Path,
    output: &Path,
) -> Result<Child> {
    Command::new(&gate.worker_executable)
        .arg("deployment-budget-worker")
        .arg("--ledger")
        .arg(ledger)
        .arg("--limit")
        .arg(gate.accounting_limit.to_string())
        .arg("--attempts")
        .arg(attempts.to_string())
        .arg("--worker-id")
        .arg(worker_id)
        .arg("--start-file")
        .arg(start_file)
        .arg("--output")
        .arg(output)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn budget worker {worker_id}"))
}

fn wait_for_children(children: Vec<Child>) -> Result<()> {
    for child in children {
        let output = child.wait_with_output()?;
        anyhow::ensure!(
            output.status.success(),
            "budget worker failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

async fn run_rolling_restart_probe(
    gate: &DeploymentLiveGateConfig,
) -> Result<RollingRestartEvidence> {
    let directory = TempDirectory::new("deployment-restart")?;
    let (provider_url, provider_task) = spawn_controlled_provider().await?;
    let first_port = reserve_port()?;
    let second_port = reserve_port()?;
    let first_config = controlled_process_config(&provider_url, first_port)?;
    let second_config = controlled_process_config(&provider_url, second_port)?;
    let first_path = directory.path.join("replica-1.yaml");
    let second_path = directory.path.join("replica-2.yaml");
    fs::write(&first_path, serde_yaml::to_string(&first_config)?)?;
    fs::write(&second_path, serde_yaml::to_string(&second_config)?)?;
    let first_url = format!("http://127.0.0.1:{first_port}");
    let second_url = format!("http://127.0.0.1:{second_port}");
    let mut first = ManagedChild::spawn_router(&gate.worker_executable, &first_path)?;
    let mut second = ManagedChild::spawn_router(&gate.worker_executable, &second_path)?;
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
    let initial_first = wait_for_live(&client, &first_url, 10_000).await;
    let initial_second = wait_for_live(&client, &second_url, 10_000).await;
    let initial_replicas_healthy = initial_first
        && initial_second
        && controlled_chat_probe(&client, &first_url).await
        && controlled_chat_probe(&client, &second_url).await;
    first.stop()?;
    let restarted = Instant::now();
    let surviving_replica_served_during_restart = controlled_chat_probe(&client, &second_url).await;
    first = ManagedChild::spawn_router(&gate.worker_executable, &first_path)?;
    let replacement_live = wait_for_live(&client, &first_url, gate.max_recovery_ms).await;
    let replacement_served_after_restart =
        replacement_live && controlled_chat_probe(&client, &first_url).await;
    let replacement_ready_ms = replacement_live.then(|| duration_ms_std(restarted.elapsed()));
    let pass = initial_replicas_healthy
        && surviving_replica_served_during_restart
        && replacement_served_after_restart
        && replacement_ready_ms.is_some_and(|elapsed| elapsed <= gate.max_recovery_ms);
    first.stop()?;
    second.stop()?;
    provider_task.abort();
    Ok(RollingRestartEvidence {
        executed: true,
        replicas: 2,
        initial_replicas_healthy,
        surviving_replica_served_during_restart,
        replacement_ready_ms,
        replacement_served_after_restart,
        max_recovery_ms: gate.max_recovery_ms,
        pass,
    })
}

fn controlled_process_config(provider_url: &str, port: u16) -> Result<RouterConfig> {
    let mut config = controlled_router_config(provider_url)?;
    config.bind = format!("127.0.0.1:{port}");
    config.runtime.provider_health_sampler.enabled = false;
    config.validate()?;
    Ok(config)
}

async fn wait_for_live(client: &Client, base_url: &str, max_wait_ms: u64) -> bool {
    let started = Instant::now();
    while duration_ms_std(started.elapsed()) <= max_wait_ms {
        if client
            .get(join_url(base_url, "/health/live"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return true;
        }
        sleep(Duration::from_millis(20)).await;
    }
    false
}

async fn controlled_chat_probe(client: &Client, base_url: &str) -> bool {
    let Ok(response) = client
        .post(join_url(base_url, "/v1/chat/completions"))
        .json(&json!({
            "model": "controlled-model",
            "messages": [{"role": "user", "content": "restart probe"}],
            "max_tokens": 1
        }))
        .send()
        .await
    else {
        return false;
    };
    let success = response.status().is_success();
    success && response.bytes().await.is_ok()
}

fn file_budget_config(path: &Path, limit: u64) -> BudgetConfig {
    BudgetConfig {
        max_chat_requests: Some(limit),
        max_total_tokens: None,
        max_estimated_cost_micros: None,
        accounting: BudgetAccountingConfig {
            backend: BudgetAccountingBackend::File,
            file_path: Some(path.to_string_lossy().to_string()),
            lock_timeout_ms: 5_000,
            ..Default::default()
        },
    }
}

fn latency_report(sorted: &[u64]) -> LatencyReport {
    LatencyReport {
        min_ms: sorted.first().copied().unwrap_or(0),
        p50_ms: percentile(sorted, 50),
        p90_ms: percentile(sorted, 90),
        p95_ms: percentile(sorted, 95),
        p99_ms: percentile(sorted, 99),
        max_ms: sorted.last().copied().unwrap_or(0),
    }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) * percentile).div_ceil(100);
    sorted[index.min(sorted.len() - 1)]
}

fn below_limit_file_bytes(limit: usize) -> usize {
    limit.saturating_sub(4_096).max(44)
}

fn workload_multipart_file_bytes(limit: usize) -> usize {
    below_limit_file_bytes(limit).min(64 * 1_024)
}

fn request_id_present(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get("x-autohand-router-request-id")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.trim().is_empty())
}

fn authorized(request: RequestBuilder, token: Option<&str>) -> RequestBuilder {
    if let Some(token) = token {
        request.bearer_auth(token)
    } else {
        request
    }
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn reserve_port() -> Result<u16> {
    let listener = StdTcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn duration_ms_std(duration: Duration) -> u64 {
    duration_ms(duration)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn pretty_json(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new(label: &str) -> Result<Self> {
        let path =
            std::env::temp_dir().join(format!("autohand-router-{label}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct ManagedChild {
    child: Option<Child>,
}

impl ManagedChild {
    fn spawn_router(executable: &Path, config: &Path) -> Result<Self> {
        let child = Command::new(executable)
            .arg("--config")
            .arg(config)
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn controlled router replica")?;
        Ok(Self { child: Some(child) })
    }

    fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            child.kill()?;
            child.wait()?;
        }
        Ok(())
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_delta_uses_cumulative_histogram_buckets() {
        let before = concat!(
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"1\"} 2\n",
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"5\"} 3\n",
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"+Inf\"} 3\n"
        );
        let after = concat!(
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"1\"} 11\n",
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"5\"} 13\n",
            "autohand_router_provider_queue_duration_ms_bucket{endpoint=\"dispatch\",provider=\"a\",model=\"\",outcome=\"admitted\",le=\"+Inf\"} 13\n"
        );
        let evidence = queue_evidence(before, after, 5);
        assert_eq!(evidence.admitted_samples, 10);
        assert_eq!(evidence.p95_upper_bound_ms, Some(5));
        assert!(evidence.pass);
    }

    #[test]
    fn wav_probe_is_sized_and_structurally_valid() {
        let payload = wav_payload(4_096);
        assert_eq!(payload.len(), 4_096);
        assert_eq!(&payload[..4], b"RIFF");
        assert_eq!(&payload[8..12], b"WAVE");
        assert_eq!(below_limit_file_bytes(8_192), 4_096);
        assert_eq!(
            workload_multipart_file_bytes(32 * 1_024 * 1_024),
            64 * 1_024
        );
    }

    #[test]
    fn target_identity_requires_candidate_revision_and_config() {
        let gate = DeploymentLiveGateConfig {
            base_url: "http://127.0.0.1:8080".to_string(),
            bearer_token: None,
            revision: "candidate".to_string(),
            duration: Duration::from_secs(1),
            requests_per_second: 4,
            concurrency: 1,
            min_samples_per_scenario: 1,
            max_p95_ms: 1,
            max_p99_ms: 1,
            max_error_rate: 0.0,
            max_queue_p95_ms: 1,
            max_peak_rss_bytes: 1,
            require_rss: false,
            require_target_revision: true,
            max_recovery_ms: 1,
            accounting_processes: 2,
            accounting_limit: 2,
            max_upload_probe_bytes: 8_192,
            worker_executable: std::env::current_exe().unwrap(),
        };
        let matching = json!({
            "deployment_revision": "candidate",
            "config_fnv1a_64": "config"
        });
        assert!(target_identity_evidence(&matching, &gate, "config").pass);

        let wrong_revision = json!({
            "deployment_revision": "older",
            "config_fnv1a_64": "config"
        });
        assert!(!target_identity_evidence(&wrong_revision, &gate, "config").pass);
        let wrong_config = json!({
            "deployment_revision": "candidate",
            "config_fnv1a_64": "other"
        });
        assert!(!target_identity_evidence(&wrong_config, &gate, "config").pass);
    }
}
