use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub base_url: String,
    pub path: String,
    pub requests: u64,
    pub concurrency: usize,
    pub slo_p95_ms: u64,
    pub slo_error_rate: f64,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub struct LoadSuiteConfig {
    pub base_url: String,
    pub requests_per_scenario: u64,
    pub concurrency: usize,
    pub slo_p95_ms: u64,
    pub slo_error_rate: f64,
    pub scenarios: Vec<LoadSuiteScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSuiteScenario {
    pub name: String,
    pub path: String,
    pub body: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestReport {
    pub target: String,
    pub requests: u64,
    pub concurrency: usize,
    pub successes: u64,
    pub failures: u64,
    pub error_rate: f64,
    pub duration_ms: u128,
    pub requests_per_second: f64,
    pub latency: LatencyReport,
    pub slo: SloReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSuiteReport {
    pub schema_version: u32,
    pub base_url: String,
    pub requests_per_scenario: u64,
    pub concurrency: usize,
    pub total_requests: u64,
    pub pass: bool,
    pub reports: Vec<NamedLoadTestReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedLoadTestReport {
    pub scenario: String,
    pub path: String,
    pub report: LoadTestReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyReport {
    pub min_ms: u64,
    pub p50_ms: u64,
    pub p90_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub max_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloReport {
    pub p95_ms_threshold: u64,
    pub error_rate_threshold: f64,
    pub pass: bool,
}

#[derive(Debug, Clone)]
struct Sample {
    latency_ms: u64,
    success: bool,
}

pub fn default_multimodel_body() -> Value {
    serde_json::json!({
        "input": "Load test route selection for a production Rust service",
        "policy": "balanced",
        "max_output_tokens": 64
    })
}

pub fn default_load_suite_scenarios() -> Vec<LoadSuiteScenario> {
    vec![
        LoadSuiteScenario {
            name: "router_multimodel".to_string(),
            path: "/v1/router/multimodel".to_string(),
            body: default_multimodel_body(),
        },
        LoadSuiteScenario {
            name: "chat_auto".to_string(),
            path: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": "Load test chat routing for a production Rust service"
                    }
                ],
                "max_tokens": 32
            }),
        },
        LoadSuiteScenario {
            name: "responses_auto".to_string(),
            path: "/v1/responses".to_string(),
            body: serde_json::json!({
                "model": "auto",
                "input": "Load test Responses API routing for a production Rust service",
                "max_output_tokens": 32
            }),
        },
        LoadSuiteScenario {
            name: "embeddings_auto".to_string(),
            path: "/v1/embeddings".to_string(),
            body: serde_json::json!({
                "model": "auto",
                "input": "Load test embeddings routing for a production Rust service"
            }),
        },
        LoadSuiteScenario {
            name: "images_auto".to_string(),
            path: "/v1/images/generations".to_string(),
            body: serde_json::json!({
                "model": "auto",
                "prompt": "A production router load-test diagram"
            }),
        },
        LoadSuiteScenario {
            name: "speech_auto".to_string(),
            path: "/v1/audio/speech".to_string(),
            body: serde_json::json!({
                "model": "auto",
                "input": "Load test speech routing",
                "voice": "alloy"
            }),
        },
    ]
}

pub async fn run_load_test(config: LoadTestConfig) -> Result<LoadTestReport> {
    validate_load_config(&config)?;
    let target = join_url(&config.base_url, &config.path);
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .context("failed to build load-test HTTP client")?;
    let samples = Arc::new(Mutex::new(Vec::with_capacity(config.requests as usize)));
    let next_request = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut workers = Vec::with_capacity(config.concurrency);

    for _ in 0..config.concurrency {
        let client = client.clone();
        let target = target.clone();
        let body = config.body.clone();
        let samples = Arc::clone(&samples);
        let next_request = Arc::clone(&next_request);
        let request_count = config.requests;
        workers.push(tokio::spawn(async move {
            loop {
                let index = next_request.fetch_add(1, Ordering::Relaxed);
                if index >= request_count {
                    break;
                }
                let started = Instant::now();
                let success = match client.post(&target).json(&body).send().await {
                    Ok(response) => response.status().is_success(),
                    Err(_) => false,
                };
                let latency_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                samples.lock().await.push(Sample {
                    latency_ms,
                    success,
                });
            }
        }));
    }

    for worker in workers {
        worker.await.context("load-test worker task failed")?;
    }

    let duration = started.elapsed();
    let samples = samples.lock().await.clone();
    Ok(report_from_samples(
        target,
        config.requests,
        config.concurrency,
        config.slo_p95_ms,
        config.slo_error_rate,
        duration,
        &samples,
    ))
}

pub async fn run_load_suite(config: LoadSuiteConfig) -> Result<LoadSuiteReport> {
    anyhow::ensure!(
        !config.scenarios.is_empty(),
        "load-suite requires at least one scenario"
    );
    let mut reports = Vec::with_capacity(config.scenarios.len());
    for scenario in &config.scenarios {
        let report = run_load_test(LoadTestConfig {
            base_url: config.base_url.clone(),
            path: scenario.path.clone(),
            requests: config.requests_per_scenario,
            concurrency: config.concurrency,
            slo_p95_ms: config.slo_p95_ms,
            slo_error_rate: config.slo_error_rate,
            body: scenario.body.clone(),
        })
        .await?;
        reports.push(NamedLoadTestReport {
            scenario: scenario.name.clone(),
            path: scenario.path.clone(),
            report,
        });
    }
    let pass = reports.iter().all(|entry| entry.report.slo.pass);
    Ok(LoadSuiteReport {
        schema_version: 1,
        base_url: config.base_url,
        requests_per_scenario: config.requests_per_scenario,
        concurrency: config.concurrency,
        total_requests: config
            .requests_per_scenario
            .saturating_mul(reports.len() as u64),
        pass,
        reports,
    })
}

fn validate_load_config(config: &LoadTestConfig) -> Result<()> {
    anyhow::ensure!(
        config.requests > 0,
        "load-test requests must be greater than zero"
    );
    anyhow::ensure!(
        config.concurrency > 0,
        "load-test concurrency must be greater than zero"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&config.slo_error_rate),
        "load-test slo_error_rate must be between 0.0 and 1.0"
    );
    anyhow::ensure!(
        config.base_url.starts_with("http://") || config.base_url.starts_with("https://"),
        "load-test base URL must start with http:// or https://"
    );
    anyhow::ensure!(
        config.path.starts_with('/'),
        "load-test path must start with /"
    );
    Ok(())
}

fn report_from_samples(
    target: String,
    requests: u64,
    concurrency: usize,
    slo_p95_ms: u64,
    slo_error_rate: f64,
    duration: Duration,
    samples: &[Sample],
) -> LoadTestReport {
    let successes = samples.iter().filter(|sample| sample.success).count() as u64;
    let failures = requests.saturating_sub(successes);
    let error_rate = failures as f64 / requests.max(1) as f64;
    let latency = latency_report(samples);
    let duration_ms = duration.as_millis();
    let duration_secs = duration.as_secs_f64().max(0.001);
    let requests_per_second = samples.len() as f64 / duration_secs;
    let pass = latency.p95_ms <= slo_p95_ms && error_rate <= slo_error_rate;

    LoadTestReport {
        target,
        requests,
        concurrency,
        successes,
        failures,
        error_rate,
        duration_ms,
        requests_per_second,
        latency,
        slo: SloReport {
            p95_ms_threshold: slo_p95_ms,
            error_rate_threshold: slo_error_rate,
            pass,
        },
    }
}

fn latency_report(samples: &[Sample]) -> LatencyReport {
    let mut latencies = samples
        .iter()
        .map(|sample| sample.latency_ms)
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    LatencyReport {
        min_ms: *latencies.first().unwrap_or(&0),
        p50_ms: percentile(&latencies, 50),
        p90_ms: percentile(&latencies, 90),
        p95_ms: percentile(&latencies, 95),
        p99_ms: percentile(&latencies, 99),
        max_ms: *latencies.last().unwrap_or(&0),
    }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) * percentile).div_ceil(100);
    sorted[index.min(sorted.len() - 1)]
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::post};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn load_test_reports_latency_and_slo_pass() {
        let base_url = spawn_route_server().await;
        let report = run_load_test(LoadTestConfig {
            base_url,
            path: "/v1/router/multimodel".to_string(),
            requests: 8,
            concurrency: 2,
            slo_p95_ms: 1_000,
            slo_error_rate: 0.0,
            body: default_multimodel_body(),
        })
        .await
        .unwrap();

        assert_eq!(report.requests, 8);
        assert_eq!(report.successes, 8);
        assert_eq!(report.failures, 0);
        assert!(report.slo.pass);
    }

    #[tokio::test]
    async fn load_suite_reports_all_default_scenarios() {
        let base_url = spawn_suite_server(false).await;
        let report = run_load_suite(LoadSuiteConfig {
            base_url,
            requests_per_scenario: 3,
            concurrency: 2,
            slo_p95_ms: 1_000,
            slo_error_rate: 0.0,
            scenarios: default_load_suite_scenarios(),
        })
        .await
        .unwrap();

        assert!(report.pass);
        assert_eq!(report.reports.len(), 6);
        assert_eq!(report.total_requests, 18);
        assert!(
            report
                .reports
                .iter()
                .all(|entry| entry.report.successes == 3)
        );
    }

    #[tokio::test]
    async fn load_suite_fails_when_any_scenario_misses_slo() {
        let base_url = spawn_suite_server(true).await;
        let report = run_load_suite(LoadSuiteConfig {
            base_url,
            requests_per_scenario: 3,
            concurrency: 1,
            slo_p95_ms: 1_000,
            slo_error_rate: 0.0,
            scenarios: default_load_suite_scenarios(),
        })
        .await
        .unwrap();

        assert!(!report.pass);
        let failed = report
            .reports
            .iter()
            .find(|entry| entry.scenario == "embeddings_auto")
            .unwrap();
        assert_eq!(failed.report.failures, 3);
        assert!(!failed.report.slo.pass);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = vec![1, 2, 3, 4, 5];
        assert_eq!(percentile(&values, 50), 3);
        assert_eq!(percentile(&values, 95), 5);
    }

    async fn spawn_route_server() -> String {
        async fn route() -> Json<Value> {
            Json(serde_json::json!({
                "model": "cheap",
                "provider": "local",
                "difficulty": "easy",
                "confidence": 0.9,
                "policy": "balanced",
                "reason": "test",
                "fallback": false,
                "estimated_input_tokens": 8,
                "requested_output_tokens": 64
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/router/multimodel", post(route));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_suite_server(fail_embeddings: bool) -> String {
        async fn ok() -> Json<Value> {
            Json(serde_json::json!({ "ok": true }))
        }
        async fn embeddings_ok() -> Json<Value> {
            Json(serde_json::json!({ "data": [{ "embedding": [0.1] }] }))
        }
        async fn embeddings_fail() -> (axum::http::StatusCode, Json<Value>) {
            (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({ "error": "upstream failed" })),
            )
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut app = Router::new()
            .route("/v1/router/multimodel", post(ok))
            .route("/v1/chat/completions", post(ok))
            .route("/v1/responses", post(ok))
            .route("/v1/images/generations", post(ok))
            .route("/v1/audio/speech", post(ok));
        app = if fail_embeddings {
            app.route("/v1/embeddings", post(embeddings_fail))
        } else {
            app.route("/v1/embeddings", post(embeddings_ok))
        };
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }
}
