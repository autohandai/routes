use crate::{
    config::RouterConfig,
    provider::{ProviderClient, ProviderHealth},
    types::{
        ChatMessage, ModelConfig, ModelEndpoint, OpenAiAudioMultipartRequest, OpenAiChatRequest,
        OpenAiEmbeddingsRequest, OpenAiImagesRequest, OpenAiMultipartPart, OpenAiResponsesRequest,
        OpenAiSpeechRequest, ProviderConfig, ProviderKind,
    },
};
use anyhow::{Context, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

pub type VerifiedEndpointCatalog = HashMap<(String, String), Vec<crate::types::ModelEndpoint>>;

#[derive(Deserialize)]
struct ConformanceArtifact {
    schema_version: u32,
    reports: Vec<ConformanceArtifactReport>,
}

#[derive(Deserialize)]
struct ConformanceArtifactReport {
    provider: String,
    model: String,
    chat: ConformanceArtifactChat,
    endpoints: Vec<ConformanceArtifactEndpoint>,
}

#[derive(Deserialize)]
struct ConformanceArtifactChat {
    #[serde(default = "default_true")]
    configured: bool,
    status: u16,
    openai_chat_shape: bool,
    response_model_matches: bool,
    assistant_content_present: bool,
}

#[derive(Deserialize)]
struct ConformanceArtifactEndpoint {
    endpoint: String,
    configured: bool,
    pass: bool,
}

pub fn load_verified_endpoint_catalog(path: &Path) -> Result<VerifiedEndpointCatalog> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read conformance artifact {}", path.display()))?;
    let artifact = serde_json::from_str::<ConformanceArtifact>(&raw)
        .with_context(|| format!("failed to parse conformance artifact {}", path.display()))?;
    anyhow::ensure!(
        artifact.schema_version == 1,
        "unsupported conformance artifact schema_version {} in {}",
        artifact.schema_version,
        path.display()
    );
    let mut catalog = HashMap::new();
    for report in artifact.reports {
        let key = (report.provider.clone(), report.model.clone());
        anyhow::ensure!(
            !catalog.contains_key(&key),
            "duplicate conformance report for provider {} model {}",
            report.provider,
            report.model
        );
        let mut endpoints = Vec::new();
        if report.chat.configured
            && (200..300).contains(&report.chat.status)
            && report.chat.openai_chat_shape
            && report.chat.response_model_matches
            && report.chat.assistant_content_present
        {
            endpoints.push(crate::types::ModelEndpoint::Chat);
        }
        for endpoint in report.endpoints {
            if endpoint.configured && endpoint.pass {
                let endpoint =
                    model_endpoint_from_artifact(&endpoint.endpoint).with_context(|| {
                        format!(
                            "unknown endpoint {} in conformance report for provider {} model {}",
                            endpoint.endpoint, report.provider, report.model
                        )
                    })?;
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
        }
        catalog.insert(key, endpoints);
    }
    Ok(catalog)
}

fn default_true() -> bool {
    true
}

fn model_endpoint_from_artifact(value: &str) -> Option<crate::types::ModelEndpoint> {
    match value {
        "chat" => Some(crate::types::ModelEndpoint::Chat),
        "responses" => Some(crate::types::ModelEndpoint::Responses),
        "embeddings" => Some(crate::types::ModelEndpoint::Embeddings),
        "images" => Some(crate::types::ModelEndpoint::Images),
        "speech" => Some(crate::types::ModelEndpoint::Speech),
        "audio_transcriptions" => Some(crate::types::ModelEndpoint::AudioTranscriptions),
        "audio_translations" => Some(crate::types::ModelEndpoint::AudioTranslations),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderConformanceReport {
    pub schema_version: u32,
    pub provider: String,
    pub provider_kind: ProviderKind,
    pub model: String,
    pub input: String,
    pub pass: bool,
    pub health: ProviderHealth,
    pub chat: ChatConformance,
    pub endpoints: Vec<EndpointConformance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderConformanceMatrixReport {
    pub schema_version: u32,
    pub input: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass: bool,
    pub reports: Vec<ProviderConformanceReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatConformance {
    pub configured: bool,
    pub status: u16,
    pub content_type: Option<String>,
    pub openai_chat_shape: bool,
    pub response_model_matches: bool,
    pub assistant_content_present: bool,
    pub usage_present: bool,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EndpointConformance {
    pub endpoint: &'static str,
    pub configured: bool,
    pub status: Option<u16>,
    pub content_type: Option<String>,
    pub pass: bool,
    pub error: Option<String>,
}

pub async fn run_provider_conformance(
    config: RouterConfig,
    model_handle: String,
    input: String,
) -> Result<ProviderConformanceReport> {
    let model = config
        .find_model(&model_handle)
        .cloned()
        .with_context(|| format!("model {model_handle} is not configured"))?;
    let provider = config
        .providers
        .iter()
        .find(|provider| provider.name == model.provider)
        .cloned()
        .with_context(|| format!("provider {} is not configured", model.provider))?;
    let client = ProviderClient::new(&config)?;
    run_provider_conformance_for_model(&config, &client, &provider, &model, input).await
}

pub async fn run_provider_conformance_matrix(
    config: RouterConfig,
    input: String,
) -> Result<ProviderConformanceMatrixReport> {
    let client = ProviderClient::new(&config)?;
    let mut reports = Vec::with_capacity(config.models.len());
    for model in &config.models {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        reports.push(
            run_provider_conformance_for_model(&config, &client, provider, model, input.clone())
                .await?,
        );
    }
    let passed = reports.iter().filter(|report| report.pass).count();
    let total = reports.len();
    Ok(ProviderConformanceMatrixReport {
        schema_version: 1,
        input,
        total,
        passed,
        failed: total.saturating_sub(passed),
        pass: passed == total,
        reports,
    })
}

async fn run_provider_conformance_for_model(
    config: &RouterConfig,
    client: &ProviderClient,
    provider: &ProviderConfig,
    model: &ModelConfig,
    input: String,
) -> Result<ProviderConformanceReport> {
    let health = client.check_provider(provider).await;
    let chat_configured = endpoint_configured(provider, model, ModelEndpoint::Chat);
    let chat = if chat_configured {
        match client
            .send_chat(
                config,
                model,
                OpenAiChatRequest {
                    model: model.id.clone(),
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: Value::String(input.clone()),
                        extra: Default::default(),
                    }],
                    extra: Default::default(),
                },
            )
            .await
        {
            Ok(response) => response_chat_conformance(model, response).await,
            Err(error) => ChatConformance {
                configured: true,
                status: 0,
                content_type: None,
                openai_chat_shape: false,
                response_model_matches: false,
                assistant_content_present: false,
                usage_present: false,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                error: Some(format!("provider chat request failed: {error:#}")),
            },
        }
    } else {
        ChatConformance {
            configured: false,
            status: 0,
            content_type: None,
            openai_chat_shape: false,
            response_model_matches: false,
            assistant_content_present: false,
            usage_present: false,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            error: None,
        }
    };
    let endpoints = run_endpoint_conformance(config, client, provider, model, &input).await;
    let endpoints_pass = endpoints
        .iter()
        .filter(|endpoint| endpoint.configured)
        .all(|endpoint| endpoint.pass);
    let chat_pass = !chat.configured
        || ((200..300).contains(&chat.status)
            && chat.openai_chat_shape
            && chat.response_model_matches
            && chat.assistant_content_present);
    let pass = chat_pass && endpoints_pass;

    Ok(ProviderConformanceReport {
        schema_version: 1,
        provider: provider.name.clone(),
        provider_kind: provider.kind.clone(),
        model: model.id.clone(),
        input,
        pass,
        health,
        chat,
        endpoints,
    })
}

async fn response_chat_conformance(
    model: &ModelConfig,
    response: crate::provider::ProviderResponse,
) -> ChatConformance {
    let status = response.status();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            return ChatConformance {
                configured: true,
                status: status.as_u16(),
                content_type,
                openai_chat_shape: false,
                response_model_matches: false,
                assistant_content_present: false,
                usage_present: false,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                error: Some(format!("failed to read response body: {error}")),
            };
        }
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => chat_conformance(status.as_u16(), content_type, &model.id, &value),
        Err(error) => ChatConformance {
            configured: true,
            status: status.as_u16(),
            content_type,
            openai_chat_shape: false,
            response_model_matches: false,
            assistant_content_present: false,
            usage_present: false,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            error: Some(format!("response body is not JSON: {error}")),
        },
    }
}

async fn run_endpoint_conformance(
    config: &RouterConfig,
    client: &ProviderClient,
    provider: &ProviderConfig,
    model: &ModelConfig,
    input: &str,
) -> Vec<EndpointConformance> {
    let mut endpoints = Vec::with_capacity(6);
    let responses = endpoint_configured(provider, model, ModelEndpoint::Responses);
    endpoints.push(
        endpoint_result(
            "responses",
            responses,
            if responses {
                Some(
                    client
                        .send_responses(
                            config,
                            model,
                            OpenAiResponsesRequest {
                                model: model.id.clone(),
                                input: Value::String(input.to_string()),
                                extra: Default::default(),
                            },
                        )
                        .await,
                )
            } else {
                None
            },
            validate_json_response,
        )
        .await,
    );
    let embeddings = endpoint_configured(provider, model, ModelEndpoint::Embeddings);
    endpoints.push(
        endpoint_result(
            "embeddings",
            embeddings,
            if embeddings {
                Some(
                    client
                        .send_embeddings(
                            config,
                            model,
                            OpenAiEmbeddingsRequest {
                                model: model.id.clone(),
                                input: Value::String(input.to_string()),
                                extra: Default::default(),
                            },
                        )
                        .await,
                )
            } else {
                None
            },
            validate_embeddings_response,
        )
        .await,
    );
    let images = endpoint_configured(provider, model, ModelEndpoint::Images);
    endpoints.push(
        endpoint_result(
            "images",
            images,
            if images {
                Some(
                    client
                        .send_images(
                            config,
                            model,
                            OpenAiImagesRequest {
                                model: model.id.clone(),
                                prompt: input.to_string(),
                                extra: Default::default(),
                            },
                        )
                        .await,
                )
            } else {
                None
            },
            validate_data_array_response,
        )
        .await,
    );
    let speech = endpoint_configured(provider, model, ModelEndpoint::Speech);
    endpoints.push(
        endpoint_result(
            "speech",
            speech,
            if speech {
                Some(
                    client
                        .send_speech(
                            config,
                            model,
                            OpenAiSpeechRequest {
                                model: model.id.clone(),
                                input: input.to_string(),
                                voice: "alloy".to_string(),
                                extra: Default::default(),
                            },
                        )
                        .await,
                )
            } else {
                None
            },
            validate_non_empty_response,
        )
        .await,
    );
    let audio_transcriptions =
        endpoint_configured(provider, model, ModelEndpoint::AudioTranscriptions);
    endpoints.push(
        endpoint_result(
            "audio_transcriptions",
            audio_transcriptions,
            if audio_transcriptions {
                Some(
                    client
                        .send_audio_transcription(
                            config,
                            model,
                            audio_probe_request(model.id.clone(), input.to_string()),
                        )
                        .await,
                )
            } else {
                None
            },
            validate_non_empty_response,
        )
        .await,
    );
    let audio_translations = endpoint_configured(provider, model, ModelEndpoint::AudioTranslations);
    endpoints.push(
        endpoint_result(
            "audio_translations",
            audio_translations,
            if audio_translations {
                Some(
                    client
                        .send_audio_translation(
                            config,
                            model,
                            audio_probe_request(model.id.clone(), input.to_string()),
                        )
                        .await,
                )
            } else {
                None
            },
            validate_non_empty_response,
        )
        .await,
    );
    endpoints
}

fn endpoint_configured(
    provider: &ProviderConfig,
    model: &ModelConfig,
    endpoint: ModelEndpoint,
) -> bool {
    provider.supports_endpoint(endpoint) && model.capabilities.supports_endpoint(endpoint)
}

async fn endpoint_result(
    endpoint: &'static str,
    configured: bool,
    response: Option<Result<crate::provider::ProviderResponse>>,
    validate: fn(&[u8]) -> Result<()>,
) -> EndpointConformance {
    let Some(response) = response else {
        return EndpointConformance {
            endpoint,
            configured,
            status: None,
            content_type: None,
            pass: true,
            error: None,
        };
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return EndpointConformance {
                endpoint,
                configured,
                status: None,
                content_type: None,
                pass: false,
                error: Some(format!("provider endpoint request failed: {error:#}")),
            };
        }
    };
    let status = response.status();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            return EndpointConformance {
                endpoint,
                configured,
                status: Some(status.as_u16()),
                content_type,
                pass: false,
                error: Some(format!("failed to read endpoint response body: {error}")),
            };
        }
    };
    let validation = if status.is_success() {
        validate(&bytes)
    } else {
        Err(anyhow::anyhow!("endpoint returned HTTP status {status}"))
    };
    EndpointConformance {
        endpoint,
        configured,
        status: Some(status.as_u16()),
        content_type,
        pass: validation.is_ok(),
        error: validation.err().map(|error| error.to_string()),
    }
}

fn validate_json_response(bytes: &[u8]) -> Result<()> {
    serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    Ok(())
}

fn validate_embeddings_response(bytes: &[u8]) -> Result<()> {
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value
            .get("data")
            .and_then(Value::as_array)
            .map(|data| !data.is_empty())
            .unwrap_or(false),
        "embeddings response must include non-empty data array"
    );
    anyhow::ensure!(
        value
            .pointer("/data/0/embedding")
            .and_then(Value::as_array)
            .map(|embedding| !embedding.is_empty())
            .unwrap_or(false),
        "embeddings response must include data[0].embedding array"
    );
    Ok(())
}

fn validate_data_array_response(bytes: &[u8]) -> Result<()> {
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value
            .get("data")
            .and_then(Value::as_array)
            .map(|data| !data.is_empty())
            .unwrap_or(false),
        "response must include non-empty data array"
    );
    Ok(())
}

fn validate_non_empty_response(bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(!bytes.is_empty(), "response body must not be empty");
    Ok(())
}

fn audio_probe_request(model: String, route_text: String) -> OpenAiAudioMultipartRequest {
    OpenAiAudioMultipartRequest {
        model,
        route_text,
        parts: vec![OpenAiMultipartPart {
            name: "file".to_string(),
            file_name: Some("autohand-router-conformance.wav".to_string()),
            content_type: Some("audio/wav".to_string()),
            data: Bytes::from_static(b"RIFF....WAVEfmt "),
        }],
    }
}

fn chat_conformance(
    status: u16,
    content_type: Option<String>,
    expected_model: &str,
    value: &Value,
) -> ChatConformance {
    let object_ok = value
        .get("object")
        .and_then(Value::as_str)
        .map(|object| object == "chat.completion")
        .unwrap_or(false);
    let response_model_matches = value
        .get("model")
        .and_then(Value::as_str)
        .map(|model| model == expected_model)
        .unwrap_or(false);
    let assistant_content_present = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(|content| !content.is_empty())
        .unwrap_or(false);
    let usage = value.get("usage");
    let prompt_tokens = usage
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64);
    let completion_tokens = usage
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(Value::as_u64);
    let total_tokens = usage
        .and_then(|usage| usage.get("total_tokens"))
        .and_then(Value::as_u64);
    ChatConformance {
        configured: true,
        status,
        content_type,
        openai_chat_shape: object_ok,
        response_model_matches,
        assistant_content_present,
        usage_present: prompt_tokens.is_some()
            || completion_tokens.is_some()
            || total_tokens.is_some(),
        prompt_tokens,
        completion_tokens,
        total_tokens,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AuthConfig, BudgetConfig, ClassifierConfig, RuntimeConfig, ScoringConfig, TelemetryConfig,
    };
    use crate::types::{DomainLabel, ModelConfig, ProviderConfig, RouterPolicy};
    use axum::{Json, Router, extract::Multipart, routing::post};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn conformance_passes_for_native_ollama_chat_transform() {
        let base_url = spawn_ollama_native_server().await;
        let report = run_provider_conformance(
            native_ollama_config(base_url),
            "llama3.1:8b".to_string(),
            "hello conformance".to_string(),
        )
        .await
        .unwrap();

        assert!(report.pass);
        assert_eq!(report.provider_kind, ProviderKind::OllamaNative);
        assert_eq!(report.chat.status, 200);
        assert!(report.chat.openai_chat_shape);
        assert!(report.chat.response_model_matches);
        assert!(report.chat.assistant_content_present);
        assert_eq!(report.chat.total_tokens, Some(9));
    }

    #[tokio::test]
    async fn conformance_passes_for_native_llama_cpp_completion_transform() {
        let base_url = spawn_llama_cpp_native_server().await;
        let report = run_provider_conformance(
            native_llama_cpp_config(base_url),
            "llama-cpp-q4".to_string(),
            "hello conformance".to_string(),
        )
        .await
        .unwrap();

        assert!(report.pass);
        assert_eq!(report.provider_kind, ProviderKind::LlamaCppNative);
        assert_eq!(report.chat.status, 200);
        assert!(report.chat.openai_chat_shape);
        assert!(report.chat.response_model_matches);
        assert!(report.chat.assistant_content_present);
        assert_eq!(report.chat.total_tokens, Some(8));
    }

    #[tokio::test]
    async fn conformance_fails_for_non_chat_completion_shape() {
        let base_url = spawn_bad_provider_server().await;
        let report = run_provider_conformance(
            openai_config(base_url),
            "bad-model".to_string(),
            "hello conformance".to_string(),
        )
        .await
        .unwrap();

        assert!(!report.pass);
        assert!(!report.chat.openai_chat_shape);
    }

    #[tokio::test]
    async fn conformance_matrix_reports_all_models_and_fails_on_any_failure() {
        let native_base_url = spawn_ollama_native_server().await;
        let bad_base_url = spawn_bad_provider_server().await;
        let mut config = native_ollama_config(native_base_url);
        let bad_config = openai_config(bad_base_url);
        config.providers.extend(bad_config.providers);
        config.models.extend(bad_config.models);

        let report =
            run_provider_conformance_matrix(config, "hello matrix conformance".to_string())
                .await
                .unwrap();

        assert!(!report.pass);
        assert_eq!(report.total, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!(
            report
                .reports
                .iter()
                .any(|entry| entry.model == "llama3.1:8b" && entry.pass)
        );
        assert!(
            report
                .reports
                .iter()
                .any(|entry| entry.model == "bad-model" && !entry.pass)
        );
    }

    #[tokio::test]
    async fn conformance_matrix_checks_all_configured_optional_endpoints() {
        let base_url = spawn_full_provider_server().await;
        let report = run_provider_conformance_matrix(
            full_openai_config(base_url),
            "hello endpoint conformance".to_string(),
        )
        .await
        .unwrap();

        assert!(report.pass);
        assert_eq!(report.total, 1);
        let model_report = &report.reports[0];
        assert!(model_report.pass);
        assert_eq!(model_report.endpoints.len(), 6);
        assert!(
            model_report
                .endpoints
                .iter()
                .all(|endpoint| endpoint.configured && endpoint.pass)
        );
    }

    #[tokio::test]
    async fn conformance_only_probes_model_declared_endpoints() {
        let base_url = spawn_full_provider_server().await;
        let mut config = full_openai_config(base_url);
        config.models[0].capabilities.supported_endpoints =
            Some(vec![ModelEndpoint::Chat, ModelEndpoint::Responses]);

        let report = run_provider_conformance_matrix(
            config,
            "hello declared endpoint conformance".to_string(),
        )
        .await
        .unwrap();

        let model_report = &report.reports[0];
        assert!(model_report.chat.configured);
        for endpoint in &model_report.endpoints {
            assert_eq!(
                endpoint.configured,
                endpoint.endpoint == "responses",
                "provider path alone enabled {}",
                endpoint.endpoint
            );
        }
    }

    async fn spawn_ollama_native_server() -> String {
        async fn chat(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "message": {
                    "role": "assistant",
                    "content": format!(
                        "native:{}",
                        request["messages"][0]["content"].as_str().unwrap_or_default()
                    )
                },
                "done": true,
                "prompt_eval_count": 6,
                "eval_count": 3
            }))
        }
        async fn health() -> Json<Value> {
            Json(serde_json::json!({ "models": [] }))
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/api/chat", post(chat))
            .route("/api/tags", axum::routing::get(health));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_llama_cpp_native_server() -> String {
        async fn completion(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "content": format!(
                    "native:{}",
                    request["prompt"].as_str().unwrap_or_default()
                ),
                "tokens_evaluated": 5,
                "tokens_predicted": 3,
                "stopped_eos": true
            }))
        }
        async fn health() -> Json<Value> {
            Json(serde_json::json!({ "status": "ok" }))
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/completion", post(completion))
            .route("/health", axum::routing::get(health));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_bad_provider_server() -> String {
        async fn chat() -> Json<Value> {
            Json(serde_json::json!({ "ok": true }))
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/chat/completions", post(chat));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_full_provider_server() -> String {
        async fn chat(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "object": "chat.completion",
                "model": request["model"],
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "ok"
                    }
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            }))
        }
        async fn responses(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "object": "response",
                "model": request["model"],
                "output_text": "ok"
            }))
        }
        async fn embeddings(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "object": "list",
                "model": request["model"],
                "data": [{
                    "object": "embedding",
                    "embedding": [0.1, 0.2],
                    "index": 0
                }]
            }))
        }
        async fn images(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "data": [{ "url": "https://example.test/image.png" }]
            }))
        }
        async fn speech(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "audio": "mock"
            }))
        }
        async fn audio_multipart(mut multipart: Multipart) -> Json<Value> {
            let mut saw_model = false;
            let mut saw_file = false;
            while let Some(field) = multipart.next_field().await.unwrap() {
                match field.name().unwrap_or_default() {
                    "model" => saw_model = true,
                    "file" => saw_file = true,
                    _ => {}
                }
            }
            Json(serde_json::json!({
                "text": "ok",
                "saw_model": saw_model,
                "saw_file": saw_file
            }))
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/responses", post(responses))
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/images/generations", post(images))
            .route("/v1/audio/speech", post(speech))
            .route("/v1/audio/transcriptions", post(audio_multipart))
            .route("/v1/audio/translations", post(audio_multipart));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn native_ollama_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "llama3.1:8b".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "ollama-native".to_string(),
                kind: ProviderKind::OllamaNative,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/api/chat".to_string(),
                responses_path: None,
                embeddings_path: None,
                images_path: None,
                speech_path: None,
                audio_transcriptions_path: None,
                audio_translations_path: None,
                health_path: Some("/api/tags".to_string()),
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![test_model("llama3.1:8b", "ollama-native")],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        }
    }

    fn native_llama_cpp_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "llama-cpp-q4".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "llama-cpp-native".to_string(),
                kind: ProviderKind::LlamaCppNative,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/completion".to_string(),
                responses_path: None,
                embeddings_path: None,
                images_path: None,
                speech_path: None,
                audio_transcriptions_path: None,
                audio_translations_path: None,
                health_path: Some("/health".to_string()),
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![test_model("llama-cpp-q4", "llama-cpp-native")],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        }
    }

    fn openai_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "bad-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "bad-provider".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/v1/chat/completions".to_string(),
                responses_path: None,
                embeddings_path: None,
                images_path: None,
                speech_path: None,
                audio_transcriptions_path: None,
                audio_translations_path: None,
                health_path: None,
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![test_model("bad-model", "bad-provider")],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        }
    }

    fn full_openai_config(base_url: String) -> RouterConfig {
        let mut model = test_model("full-model", "full-provider");
        model.capabilities.supported_endpoints = Some(ModelEndpoint::ALL.to_vec());
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "full-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "full-provider".to_string(),
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
            models: vec![model],
            classifier: ClassifierConfig::default(),
            auth: AuthConfig::default(),
            scoring: ScoringConfig::default(),
            budget: BudgetConfig::default(),
            telemetry: TelemetryConfig::default(),
            runtime: RuntimeConfig::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        }
    }

    fn test_model(id: &str, provider: &str) -> ModelConfig {
        ModelConfig {
            id: id.to_string(),
            provider: provider.to_string(),
            aliases: vec![],
            capability: 0.5,
            cost_per_million_input: 1.0,
            cost_per_million_output: 1.0,
            domains: vec![DomainLabel::General],
            context_window: Some(4096),
            capabilities: Default::default(),
            local: true,
        }
    }
}
