use crate::{
    classifier::{JudgeMetricsSnapshot, PromptClassifier, SmartClassifier},
    config::RouterConfig,
    types::SelectedClassifications,
};
use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct JudgeSmokeReport {
    pub configured_model: String,
    pub input: String,
    pub pass: bool,
    pub classifications: SelectedClassifications,
    pub metrics_before: JudgeMetricsSnapshot,
    pub metrics_after: JudgeMetricsSnapshot,
}

pub async fn run_judge_smoke(config: RouterConfig, input: String) -> Result<JudgeSmokeReport> {
    let configured_model = config
        .classifier
        .llm_judge_model
        .clone()
        .context("classifier.llm_judge_model must be configured for judge-smoke")?;
    let classifier = SmartClassifier::new(config)?;
    let metrics_before = classifier.judge_metrics();
    let classifications = classifier.classify(&input).await;
    let metrics_after = classifier.judge_metrics();
    let pass = judge_succeeded_without_fallback(&metrics_before, &metrics_after);

    Ok(JudgeSmokeReport {
        configured_model,
        input,
        pass,
        classifications: SelectedClassifications::from_heads(classifications, &[]),
        metrics_before,
        metrics_after,
    })
}

fn judge_succeeded_without_fallback(
    before: &JudgeMetricsSnapshot,
    after: &JudgeMetricsSnapshot,
) -> bool {
    after.requests == before.requests.saturating_add(1)
        && after.successes == before.successes.saturating_add(1)
        && after.fallbacks == before.fallbacks
        && after.heuristic_routes == before.heuristic_routes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{
            AuthConfig, BudgetConfig, ClassifierConfig, RuntimeConfig, ScoringConfig,
            TelemetryConfig,
        },
        types::{DomainLabel, ModelConfig, ProviderConfig, ProviderKind, RouterPolicy},
    };
    use axum::{Json, Router, routing::post};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn judge_smoke_passes_when_live_judge_succeeds() {
        let base_url = spawn_judge_server(
            r#"{"difficulty":"hard","ambiguity":"low","domain":"coding","confidence":0.91,"ambiguity_confidence":0.82,"domain_confidence":0.88}"#,
        )
        .await;
        let report = run_judge_smoke(
            judge_config(base_url),
            "Design a Rust provider router".to_string(),
        )
        .await
        .unwrap();

        assert!(report.pass);
        assert_eq!(report.configured_model, "judge-model");
        assert_eq!(report.metrics_after.requests, 1);
        assert_eq!(report.metrics_after.successes, 1);
        assert_eq!(report.metrics_after.fallbacks, 0);
    }

    #[tokio::test]
    async fn judge_smoke_fails_when_judge_falls_back() {
        let base_url = spawn_judge_server("not-json").await;
        let report = run_judge_smoke(
            judge_config(base_url),
            "Design a Rust provider router".to_string(),
        )
        .await
        .unwrap();

        assert!(!report.pass);
        assert_eq!(report.metrics_after.requests, 1);
        assert_eq!(report.metrics_after.successes, 0);
        assert_eq!(report.metrics_after.fallbacks, 1);
    }

    #[tokio::test]
    async fn judge_smoke_requires_configured_judge_model() {
        let mut config = judge_config("http://127.0.0.1:1".to_string());
        config.classifier.llm_judge_model = None;
        let error = run_judge_smoke(config, "hello".to_string())
            .await
            .expect_err("judge smoke requires a configured judge");

        assert!(error.to_string().contains("llm_judge_model"));
    }

    async fn spawn_judge_server(content: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                Json(serde_json::json!({
                    "choices": [
                        {
                            "message": {
                                "content": content
                            }
                        }
                    ]
                }))
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn judge_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "judge-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "judge-provider".to_string(),
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
                id: "judge-model".to_string(),
                provider: "judge-provider".to_string(),
                aliases: vec![],
                capability: 0.7,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::Coding],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
            classifier: ClassifierConfig {
                llm_judge_model: Some("judge-model".to_string()),
                ..Default::default()
            },
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}
