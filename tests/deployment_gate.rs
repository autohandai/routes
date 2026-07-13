use autohand_router::{
    RouterConfig,
    deployment_gate::{DeploymentLiveGateConfig, run_deployment_live_gate},
    server::{AppState, app},
};
use axum::{
    Json, Router,
    body::Body,
    extract::Multipart,
    http::{Response, StatusCode, header},
    response::IntoResponse,
    routing::post,
};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
use tokio::net::TcpListener;

#[tokio::test]
async fn deployment_gate_proves_sustained_and_recovery_boundaries() {
    let provider_url = spawn_provider().await;
    let mut config = test_config(&provider_url);
    config.runtime.ingress.max_multipart_body_bytes = 8_192;
    config.validate().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = AppState::from_config(&config).unwrap();
    let router_task = tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });

    let gate = DeploymentLiveGateConfig {
        base_url: format!("http://{address}"),
        bearer_token: None,
        revision: "deployment-gate-test".to_string(),
        duration: Duration::from_millis(400),
        requests_per_second: 40,
        concurrency: 4,
        min_samples_per_scenario: 4,
        max_p95_ms: 2_000,
        max_p99_ms: 2_000,
        max_error_rate: 0.0,
        max_queue_p95_ms: 1_000,
        max_peak_rss_bytes: u64::MAX,
        require_rss: false,
        require_target_revision: false,
        max_recovery_ms: 5_000,
        accounting_processes: 2,
        accounting_limit: 8,
        max_upload_probe_bytes: 16_384,
        worker_executable: PathBuf::from(env!("CARGO_BIN_EXE_routes")),
    };
    let report = run_deployment_live_gate(&config, gate.clone())
        .await
        .unwrap();
    assert!(
        report.pass,
        "failures={:?} report={report:#?}",
        report.failures
    );
    assert_eq!(report.workload.scenarios.len(), 4);
    assert!(
        report
            .workload
            .scenarios
            .iter()
            .all(|scenario| scenario.pass)
    );
    assert!(report.queue.pass);
    assert!(report.target_identity.pass);
    assert_eq!(
        report.target_identity.reported_config_fnv1a_64,
        Some(report.config_fnv1a_64.clone())
    );
    assert!(report.multipart.pass);
    assert!(report.multi_process_file_accounting.pass);
    assert!(report.rolling_restart.pass);

    let mut mismatched = gate;
    mismatched.require_target_revision = true;
    let rejected = run_deployment_live_gate(&config, mismatched).await.unwrap();
    assert!(!rejected.pass);
    assert!(!rejected.target_identity.pass);
    assert!(!rejected.workload.executed);
    assert!(!rejected.multi_process_file_accounting.executed);
    router_task.abort();
}

async fn spawn_provider() -> String {
    async fn chat(Json(request): Json<Value>) -> axum::response::Response {
        if request.get("stream").and_then(Value::as_bool) == Some(true) {
            return Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
                    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
                    "data: [DONE]\n\n"
                )))
                .unwrap();
        }
        Json(json!({
            "id": "deployment-test",
            "object": "chat.completion",
            "model": request["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }))
        .into_response()
    }

    async fn audio(mut multipart: Multipart) -> Json<Value> {
        while multipart.next_field().await.unwrap().is_some() {}
        Json(json!({"text": "ok"}))
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let provider = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/audio/transcriptions", post(audio));
    tokio::spawn(async move {
        axum::serve(listener, provider).await.unwrap();
    });
    format!("http://{address}")
}

fn test_config(provider_url: &str) -> RouterConfig {
    serde_yaml::from_str(&format!(
        r#"
bind: 127.0.0.1:0
default_model: deployment-model
providers:
  - name: deployment-provider
    kind: open_ai_compatible
    base_url: {provider_url}
    chat_path: /v1/chat/completions
    audio_transcriptions_path: /v1/audio/transcriptions
    retries: 0
    timeout_ms: 2000
    max_concurrency: 2
    queue_timeout_ms: 1000
models:
  - id: deployment-model
    provider: deployment-provider
    capability: 0.7
    cost_per_million_input: 0.1
    cost_per_million_output: 0.1
    context_window: 8192
    capabilities:
      supports_audio: true
      supported_endpoints: [chat, audio_transcriptions]
"#
    ))
    .unwrap()
}
