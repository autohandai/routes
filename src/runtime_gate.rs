use crate::{
    config::RouterConfig,
    server::{AppState, app},
};
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Body,
    http::{Response, StatusCode, header},
    response::IntoResponse,
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::{net::TcpListener, task::JoinHandle};

const TEST_TOKEN: &str = "runtime-gate-token";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeGateReport {
    pub schema_version: u32,
    pub runtime: String,
    pub classifier: String,
    pub scenarios: Vec<RuntimeGateScenario>,
    pub upstream: RuntimeGateUpstreamEvidence,
    pub pass: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeGateScenario {
    pub id: String,
    pub expected_status: u16,
    pub actual_status: u16,
    pub pass: bool,
    pub assertion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeGateUpstreamEvidence {
    pub primary_requests: u64,
    pub backup_requests: u64,
    pub failover_attempts: u64,
    pub failover_successes: u64,
    pub upstream_http_errors: u64,
    pub attempt_http_errors: u64,
}

/// Runs a controlled mock-provider suite through the real Axum HTTP stack.
/// It is intentionally independent of the in-memory classifier eval gates.
pub async fn run_runtime_gate() -> Result<RuntimeGateReport> {
    let primary_requests = Arc::new(AtomicU64::new(0));
    let backup_requests = Arc::new(AtomicU64::new(0));
    let (primary_url, primary_task) = spawn_primary_provider(Arc::clone(&primary_requests)).await?;
    let (backup_url, backup_task) = spawn_backup_provider(Arc::clone(&backup_requests)).await?;
    let config = runtime_gate_config(&primary_url, &backup_url)?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router_task = tokio::spawn(async move {
        axum::serve(listener, app(AppState::from_config(&config)?))
            .await
            .map_err(anyhow::Error::from)
    });
    let base_url = format!("http://{address}");
    let client = reqwest::Client::new();
    let mut scenarios = Vec::new();

    let unauthenticated = client
        .post(format!("{base_url}/v1/router/classify"))
        .json(&json!({"input": "hello"}))
        .send()
        .await?;
    record_status(
        &mut scenarios,
        "auth_rejection",
        StatusCode::UNAUTHORIZED,
        unauthenticated.status(),
        "missing bearer credentials are rejected before routing",
    );

    let tools = authorized(&client, format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "primary-model",
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [{
                "type": "function",
                "function": {"name": "noop", "parameters": {"type": "object"}}
            }]
        }))
        .send()
        .await?;
    record_status(
        &mut scenarios,
        "capability_rejection",
        StatusCode::BAD_REQUEST,
        tools.status(),
        "an explicit model cannot bypass its declared tool capability",
    );

    let oversized_prompt = "token ".repeat(3_000);
    let context = authorized(&client, format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "primary-model",
            "messages": [{"role": "user", "content": oversized_prompt}],
            "max_tokens": 8
        }))
        .send()
        .await?;
    record_status(
        &mut scenarios,
        "context_rejection",
        StatusCode::BAD_REQUEST,
        context.status(),
        "an explicit model cannot exceed its conservative context eligibility",
    );

    let failover = authorized(&client, format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "Fix this typo"}]
        }))
        .send()
        .await?;
    record_status(
        &mut scenarios,
        "transient_upstream_failover",
        StatusCode::OK,
        failover.status(),
        "automatic routing fails over after a retryable upstream response",
    );

    let streaming = authorized(&client, format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "backup-model",
            "messages": [{"role": "user", "content": "stream briefly"}],
            "stream": true
        }))
        .send()
        .await?;
    let stream_status = streaming.status();
    let stream_type = streaming
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let stream_body = streaming.text().await?;
    let stream_shape_ok = stream_type.starts_with("text/event-stream")
        && stream_body.contains("data:")
        && stream_body.contains("[DONE]");
    scenarios.push(RuntimeGateScenario {
        id: "streaming_passthrough".to_string(),
        expected_status: StatusCode::OK.as_u16(),
        actual_status: stream_status.as_u16(),
        pass: stream_status == StatusCode::OK && stream_shape_ok,
        assertion: "SSE content type and terminal marker survive the HTTP proxy".to_string(),
    });

    let metrics = client
        .get(format!("{base_url}/metrics"))
        .bearer_auth(TEST_TOKEN)
        .send()
        .await?
        .error_for_status()?
        .json::<Value>()
        .await?;
    let evidence = RuntimeGateUpstreamEvidence {
        primary_requests: primary_requests.load(Ordering::Relaxed),
        backup_requests: backup_requests.load(Ordering::Relaxed),
        failover_attempts: metric(&metrics, "failover_attempts")?,
        failover_successes: metric(&metrics, "failover_successes")?,
        upstream_http_errors: metric(&metrics, "upstream_http_errors")?,
        attempt_http_errors: metrics
            .get("upstream_outcomes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|entry| {
                entry.get("scope").and_then(Value::as_str) == Some("attempt")
                    && entry
                        .get("outcome")
                        .and_then(Value::as_str)
                        .is_some_and(|outcome| outcome.starts_with("http_"))
            })
            .filter_map(|entry| entry.get("count").and_then(Value::as_u64))
            .sum(),
    };
    if evidence.primary_requests == 0
        || evidence.backup_requests < 2
        || evidence.failover_attempts == 0
        || evidence.failover_successes == 0
        || evidence.attempt_http_errors == 0
        || evidence.upstream_http_errors != 0
    {
        scenarios.push(RuntimeGateScenario {
            id: "upstream_outcome_metrics".to_string(),
            expected_status: StatusCode::OK.as_u16(),
            actual_status: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            pass: false,
            assertion: "failover and upstream outcomes must be observable".to_string(),
        });
    } else {
        scenarios.push(RuntimeGateScenario {
            id: "upstream_outcome_metrics".to_string(),
            expected_status: StatusCode::OK.as_u16(),
            actual_status: StatusCode::OK.as_u16(),
            pass: true,
            assertion: "failover and upstream outcomes must be observable".to_string(),
        });
    }

    router_task.abort();
    primary_task.abort();
    backup_task.abort();
    let failures = scenarios
        .iter()
        .filter(|scenario| !scenario.pass)
        .map(|scenario| format!("{}: {}", scenario.id, scenario.assertion))
        .collect::<Vec<_>>();
    Ok(RuntimeGateReport {
        schema_version: 1,
        runtime: "axum_http_with_mock_openai_providers".to_string(),
        classifier: "heuristic".to_string(),
        pass: failures.is_empty(),
        failures,
        scenarios,
        upstream: evidence,
    })
}

fn authorized(client: &reqwest::Client, url: String) -> reqwest::RequestBuilder {
    client.post(url).bearer_auth(TEST_TOKEN)
}

fn record_status(
    scenarios: &mut Vec<RuntimeGateScenario>,
    id: &str,
    expected: StatusCode,
    actual: reqwest::StatusCode,
    assertion: &str,
) {
    scenarios.push(RuntimeGateScenario {
        id: id.to_string(),
        expected_status: expected.as_u16(),
        actual_status: actual.as_u16(),
        pass: actual.as_u16() == expected.as_u16(),
        assertion: assertion.to_string(),
    });
}

fn metric(metrics: &Value, name: &str) -> Result<u64> {
    metrics
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("runtime metrics omitted integer field {name}"))
}

async fn spawn_primary_provider(requests: Arc<AtomicU64>) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let requests = Arc::clone(&requests);
            async move {
                requests.fetch_add(1, Ordering::Relaxed);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error": {"message": "controlled transient failure"}})),
                )
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Ok((format!("http://{address}"), task))
}

async fn spawn_backup_provider(requests: Arc<AtomicU64>) -> Result<(String, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let requests = Arc::clone(&requests);
            async move {
                requests.fetch_add(1, Ordering::Relaxed);
                if body.get("stream").and_then(Value::as_bool) == Some(true) {
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from(concat!(
                            "data: {\"id\":\"gate\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
                            "data: [DONE]\n\n"
                        )))
                        .unwrap()
                        .into_response()
                } else {
                    Json(json!({
                        "id": "runtime-gate",
                        "object": "chat.completion",
                        "model": "backup-model",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
                    }))
                    .into_response()
                }
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Ok((format!("http://{address}"), task))
}

fn runtime_gate_config(primary_url: &str, backup_url: &str) -> Result<RouterConfig> {
    let yaml = format!(
        r#"
bind: 127.0.0.1:0
default_model: primary-model
policy: cost_efficient
auth:
  bearer_tokens: [{TEST_TOKEN}]
providers:
  - name: primary
    kind: open_ai_compatible
    base_url: {primary_url}
    retries: 0
    timeout_ms: 1000
  - name: backup
    kind: open_ai_compatible
    base_url: {backup_url}
    retries: 0
    timeout_ms: 1000
models:
  - id: primary-model
    provider: primary
    capability: 0.40
    cost_per_million_input: 0.10
    cost_per_million_output: 0.10
    context_window: 2048
    capabilities:
      supports_tools: false
      supported_endpoints: [chat]
  - id: backup-model
    provider: backup
    capability: 0.40
    cost_per_million_input: 0.20
    cost_per_million_output: 0.20
    context_window: 4096
    capabilities:
      supports_tools: true
      supported_endpoints: [chat]
"#
    );
    let config = serde_yaml::from_str::<RouterConfig>(&yaml)
        .context("failed to build controlled runtime-gate config")?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn controlled_runtime_gate_exercises_real_http_boundaries() {
        let report = run_runtime_gate().await.unwrap();
        assert!(report.pass, "{:?}", report.failures);
        assert_eq!(report.runtime, "axum_http_with_mock_openai_providers");
        assert_eq!(report.scenarios.len(), 6);
        assert!(report.scenarios.iter().all(|scenario| scenario.pass));
        assert!(report.upstream.failover_attempts >= 1);
        assert!(report.upstream.failover_successes >= 1);
        assert!(report.upstream.attempt_http_errors >= 1);
        assert_eq!(report.upstream.upstream_http_errors, 0);
    }
}
