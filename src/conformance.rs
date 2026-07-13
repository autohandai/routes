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
use std::{
    collections::HashMap,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

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
    #[serde(default)]
    usage_present: bool,
    #[serde(default)]
    negative_schema_rejected: bool,
}

#[derive(Deserialize)]
struct ConformanceArtifactEndpoint {
    endpoint: String,
    configured: bool,
    pass: bool,
    #[serde(default)]
    positive_schema_valid: bool,
    #[serde(default)]
    negative_schema_rejected: bool,
}

pub fn load_verified_endpoint_catalog(path: &Path) -> Result<VerifiedEndpointCatalog> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read conformance artifact {}", path.display()))?;
    let artifact = serde_json::from_str::<ConformanceArtifact>(&raw)
        .with_context(|| format!("failed to parse conformance artifact {}", path.display()))?;
    anyhow::ensure!(
        matches!(artifact.schema_version, 1 | 2),
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
            && (artifact.schema_version == 1
                || (report.chat.usage_present && report.chat.negative_schema_rejected))
        {
            endpoints.push(crate::types::ModelEndpoint::Chat);
        }
        for endpoint in report.endpoints {
            if endpoint.configured
                && endpoint.pass
                && (artifact.schema_version == 1
                    || (endpoint.positive_schema_valid && endpoint.negative_schema_rejected))
            {
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
    pub generated_unix_seconds: u64,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub provider: String,
    pub provider_kind: ProviderKind,
    pub provider_version: VersionEvidence,
    pub model: String,
    pub model_version: VersionEvidence,
    pub input: String,
    pub pass: bool,
    pub health: ProviderHealth,
    pub chat: ChatConformance,
    pub features: Vec<FeatureConformance>,
    pub endpoints: Vec<EndpointConformance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderConformanceMatrixReport {
    pub schema_version: u32,
    pub generated_unix_seconds: u64,
    pub router_version: String,
    pub config_fnv1a_64: String,
    pub input: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass: bool,
    pub reports: Vec<ProviderConformanceReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VersionEvidence {
    pub value: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatConformance {
    pub configured: bool,
    pub skip_reason: Option<String>,
    pub status: u16,
    pub content_type: Option<String>,
    pub openai_chat_shape: bool,
    pub response_model_matches: bool,
    pub assistant_content_present: bool,
    pub usage_present: bool,
    pub negative_schema_rejected: bool,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureConformance {
    pub feature: &'static str,
    pub declared: bool,
    pub attempted: bool,
    pub status: Option<u16>,
    pub content_type: Option<String>,
    pub pass: bool,
    pub skip_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EndpointConformance {
    pub endpoint: &'static str,
    pub configured: bool,
    pub status: Option<u16>,
    pub content_type: Option<String>,
    pub fixture: &'static str,
    pub positive_schema_valid: bool,
    pub negative_schema_rejected: bool,
    pub pass: bool,
    pub skip_reason: Option<String>,
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
    let generated_unix_seconds = unix_seconds();
    let config_fnv1a_64 = config_fingerprint(&config)?;
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
        schema_version: 2,
        generated_unix_seconds,
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64,
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
    let (chat, provider_version, model_version) = if chat_configured {
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
            Err(error) => (
                failed_chat_conformance(format!("provider chat request failed: {error:#}")),
                None,
                None,
            ),
        }
    } else {
        (
            skipped_chat_conformance(
                endpoint_skip_reason(provider, model, ModelEndpoint::Chat)
                    .unwrap_or_else(|| "chat probe was not configured".to_string()),
            ),
            None,
            None,
        )
    };
    let endpoints = run_endpoint_conformance(config, client, provider, model, &input).await;
    let features =
        run_feature_conformance(config, client, provider, model, &input, &endpoints).await;
    let endpoints_pass = endpoints
        .iter()
        .filter(|endpoint| endpoint.configured)
        .all(|endpoint| endpoint.pass);
    let features_pass = features
        .iter()
        .filter(|feature| feature.attempted)
        .all(|feature| feature.pass);
    let chat_pass = !chat.configured || chat.pass();
    let pass = chat_pass && endpoints_pass && features_pass;

    Ok(ProviderConformanceReport {
        schema_version: 2,
        generated_unix_seconds: unix_seconds(),
        router_version: env!("CARGO_PKG_VERSION").to_string(),
        config_fnv1a_64: config_fingerprint(config)?,
        provider: provider.name.clone(),
        provider_kind: provider.kind.clone(),
        provider_version: version_evidence(provider_version, "x-provider-version"),
        model: model.id.clone(),
        model_version: version_evidence(model_version, "x-model-version"),
        input,
        pass,
        health,
        chat,
        features,
        endpoints,
    })
}

async fn response_chat_conformance(
    model: &ModelConfig,
    response: crate::provider::ProviderResponse,
) -> (ChatConformance, Option<String>, Option<String>) {
    let status = response.status();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let provider_version = response
        .headers()
        .get("x-provider-version")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let model_version = response
        .headers()
        .get("x-model-version")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            let mut failed =
                failed_chat_conformance(format!("failed to read response body: {error}"));
            failed.status = status.as_u16();
            failed.content_type = content_type;
            return (failed, provider_version, model_version);
        }
    };
    let chat = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => chat_conformance(status.as_u16(), content_type, &model.id, &value),
        Err(error) => {
            let mut failed = failed_chat_conformance(format!("response body is not JSON: {error}"));
            failed.status = status.as_u16();
            failed.content_type = content_type;
            failed
        }
    };
    (chat, provider_version, model_version)
}

impl ChatConformance {
    fn pass(&self) -> bool {
        (200..300).contains(&self.status)
            && content_type_is_json(self.content_type.as_deref())
            && self.openai_chat_shape
            && self.response_model_matches
            && self.assistant_content_present
            && self.usage_present
            && self.negative_schema_rejected
    }
}

fn failed_chat_conformance(error: String) -> ChatConformance {
    ChatConformance {
        configured: true,
        skip_reason: None,
        status: 0,
        content_type: None,
        openai_chat_shape: false,
        response_model_matches: false,
        assistant_content_present: false,
        usage_present: false,
        negative_schema_rejected: chat_negative_schema_rejected(),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        error: Some(error),
    }
}

fn skipped_chat_conformance(reason: String) -> ChatConformance {
    let mut chat = failed_chat_conformance(String::new());
    chat.configured = false;
    chat.skip_reason = Some(reason);
    chat.error = None;
    chat
}

async fn run_feature_conformance(
    config: &RouterConfig,
    client: &ProviderClient,
    provider: &ProviderConfig,
    model: &ModelConfig,
    input: &str,
    endpoints: &[EndpointConformance],
) -> Vec<FeatureConformance> {
    let chat_configured = endpoint_configured(provider, model, ModelEndpoint::Chat);
    let contract = provider.kind.chat_adapter_contract();
    let mut features = Vec::with_capacity(5);

    let streaming_declared = contract.supports_streaming;
    features.push(if chat_configured && streaming_declared {
        let mut extra = serde_json::Map::new();
        extra.insert("stream".to_string(), Value::Bool(true));
        extra.insert(
            "stream_options".to_string(),
            serde_json::json!({"include_usage": true}),
        );
        run_chat_feature(
            config,
            client,
            model,
            "streaming",
            true,
            feature_chat_request(model, input, extra),
            validate_streaming_feature,
        )
        .await
    } else {
        skipped_feature(
            "streaming",
            streaming_declared,
            if !chat_configured {
                "chat endpoint is not declared by both provider and model"
            } else {
                "provider adapter contract does not support streaming"
            },
        )
    });

    features.push(
        run_declared_chat_feature(
            config,
            client,
            provider,
            model,
            input,
            "tools",
            model.capabilities.supports_tools,
            contract.supports_tools,
            |model, input| {
                let mut request = feature_chat_request(model, input, Default::default());
                request.extra.insert(
                    "tools".to_string(),
                    serde_json::json!([{
                        "type": "function",
                        "function": {
                            "name": "conformance_echo",
                            "description": "Return the supplied text",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": false
                            }
                        }
                    }]),
                );
                request.extra.insert(
                    "tool_choice".to_string(),
                    serde_json::json!({
                        "type": "function",
                        "function": {"name": "conformance_echo"}
                    }),
                );
                request
            },
            validate_tools_feature,
        )
        .await,
    );

    features.push(
        run_declared_chat_feature(
            config,
            client,
            provider,
            model,
            input,
            "json",
            model.capabilities.supports_json,
            contract.supports_json,
            |model, input| {
                let mut request = feature_chat_request(model, input, Default::default());
                request.extra.insert(
                    "response_format".to_string(),
                    serde_json::json!({"type": "json_object"}),
                );
                request
            },
            validate_json_feature,
        )
        .await,
    );

    features.push(
        run_declared_chat_feature(
            config,
            client,
            provider,
            model,
            input,
            "vision",
            model.capabilities.supports_vision,
            contract.supports_vision,
            |model, input| OpenAiChatRequest {
                model: model.id.clone(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type": "text", "text": input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2xZ0AAAAASUVORK5CYII="
                            }
                        }
                    ]),
                    extra: Default::default(),
                }],
                extra: Default::default(),
            },
            validate_vision_feature,
        )
        .await,
    );

    let audio_endpoints = endpoints
        .iter()
        .filter(|endpoint| {
            matches!(
                endpoint.endpoint,
                "speech" | "audio_transcriptions" | "audio_translations"
            )
        })
        .collect::<Vec<_>>();
    let audio_attempted = audio_endpoints.iter().any(|endpoint| endpoint.configured);
    let audio_declared = model.capabilities.supports_audio || audio_attempted;
    features.push(FeatureConformance {
        feature: "audio",
        declared: audio_declared,
        attempted: audio_attempted,
        status: None,
        content_type: None,
        pass: !audio_attempted || audio_endpoints.iter().all(|endpoint| endpoint.pass),
        skip_reason: (!audio_attempted).then(|| {
            if audio_declared {
                "model declares audio but no audio endpoint is jointly configured".to_string()
            } else {
                "model does not declare audio and has no audio endpoints".to_string()
            }
        }),
        error: None,
    });
    features
}

#[allow(clippy::too_many_arguments)]
async fn run_declared_chat_feature(
    config: &RouterConfig,
    client: &ProviderClient,
    provider: &ProviderConfig,
    model: &ModelConfig,
    input: &str,
    feature: &'static str,
    declared: bool,
    adapter_supported: bool,
    request: fn(&ModelConfig, &str) -> OpenAiChatRequest,
    validate: fn(u16, Option<&str>, &[u8], &str) -> Result<()>,
) -> FeatureConformance {
    if !endpoint_configured(provider, model, ModelEndpoint::Chat) {
        return skipped_feature(
            feature,
            declared,
            "chat endpoint is not declared by both provider and model",
        );
    }
    if !declared {
        return skipped_feature(
            feature,
            false,
            &format!("model capability supports_{feature} is false"),
        );
    }
    if !adapter_supported {
        return skipped_feature(
            feature,
            true,
            &format!("provider adapter contract does not support {feature}"),
        );
    }
    run_chat_feature(
        config,
        client,
        model,
        feature,
        true,
        request(model, input),
        validate,
    )
    .await
}

fn feature_chat_request(
    model: &ModelConfig,
    input: &str,
    extra: serde_json::Map<String, Value>,
) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: model.id.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Value::String(input.to_string()),
            extra: Default::default(),
        }],
        extra,
    }
}

async fn run_chat_feature(
    config: &RouterConfig,
    client: &ProviderClient,
    model: &ModelConfig,
    feature: &'static str,
    declared: bool,
    request: OpenAiChatRequest,
    validate: fn(u16, Option<&str>, &[u8], &str) -> Result<()>,
) -> FeatureConformance {
    let response = match client.send_chat(config, model, request).await {
        Ok(response) => response,
        Err(error) => {
            return FeatureConformance {
                feature,
                declared,
                attempted: true,
                status: None,
                content_type: None,
                pass: false,
                skip_reason: None,
                error: Some(format!("feature request failed: {error:#}")),
            };
        }
    };
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let result = match response.bytes().await {
        Ok(bytes) => validate(status, content_type.as_deref(), &bytes, &model.id),
        Err(error) => Err(anyhow::anyhow!("failed to read feature response: {error}")),
    };
    FeatureConformance {
        feature,
        declared,
        attempted: true,
        status: Some(status),
        content_type,
        pass: result.is_ok(),
        skip_reason: None,
        error: result.err().map(|error| error.to_string()),
    }
}

fn skipped_feature(feature: &'static str, declared: bool, reason: &str) -> FeatureConformance {
    FeatureConformance {
        feature,
        declared,
        attempted: false,
        status: None,
        content_type: None,
        pass: true,
        skip_reason: Some(reason.to_string()),
        error: None,
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
            endpoint_skip_reason(provider, model, ModelEndpoint::Responses),
            "OpenAI Responses text input",
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
            EndpointValidator::responses(),
        )
        .await,
    );
    let embeddings = endpoint_configured(provider, model, ModelEndpoint::Embeddings);
    endpoints.push(
        endpoint_result(
            "embeddings",
            embeddings,
            endpoint_skip_reason(provider, model, ModelEndpoint::Embeddings),
            "single string embedding input",
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
            EndpointValidator::embeddings(),
        )
        .await,
    );
    let images = endpoint_configured(provider, model, ModelEndpoint::Images);
    endpoints.push(
        endpoint_result(
            "images",
            images,
            endpoint_skip_reason(provider, model, ModelEndpoint::Images),
            "single image-generation prompt",
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
            EndpointValidator::images(),
        )
        .await,
    );
    let speech = endpoint_configured(provider, model, ModelEndpoint::Speech);
    endpoints.push(
        endpoint_result(
            "speech",
            speech,
            endpoint_skip_reason(provider, model, ModelEndpoint::Speech),
            "alloy voice audio output",
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
            EndpointValidator::speech(),
        )
        .await,
    );
    let audio_transcriptions =
        endpoint_configured(provider, model, ModelEndpoint::AudioTranscriptions);
    endpoints.push(
        endpoint_result(
            "audio_transcriptions",
            audio_transcriptions,
            endpoint_skip_reason(provider, model, ModelEndpoint::AudioTranscriptions),
            "bounded WAV multipart transcription",
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
            EndpointValidator::audio_text(),
        )
        .await,
    );
    let audio_translations = endpoint_configured(provider, model, ModelEndpoint::AudioTranslations);
    endpoints.push(
        endpoint_result(
            "audio_translations",
            audio_translations,
            endpoint_skip_reason(provider, model, ModelEndpoint::AudioTranslations),
            "bounded WAV multipart translation",
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
            EndpointValidator::audio_text(),
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

fn endpoint_skip_reason(
    provider: &ProviderConfig,
    model: &ModelConfig,
    endpoint: ModelEndpoint,
) -> Option<String> {
    if endpoint_configured(provider, model, endpoint) {
        None
    } else if !provider.supports_endpoint(endpoint) {
        Some(format!(
            "provider {} does not configure the {} path",
            provider.name,
            endpoint.as_str()
        ))
    } else {
        Some(format!(
            "model {} does not declare the {} endpoint",
            model.id,
            endpoint.as_str()
        ))
    }
}

#[derive(Clone, Copy)]
struct EndpointValidator {
    validate: fn(&[u8], Option<&str>) -> Result<()>,
    negative_fixture: &'static [u8],
    negative_content_type: Option<&'static str>,
}

impl EndpointValidator {
    fn responses() -> Self {
        Self {
            validate: validate_responses_response,
            negative_fixture: br#"{"object":"response","output":[]}"#,
            negative_content_type: Some("application/json"),
        }
    }

    fn embeddings() -> Self {
        Self {
            validate: validate_embeddings_response,
            negative_fixture: br#"{"object":"list","data":[{"embedding":[]}] }"#,
            negative_content_type: Some("application/json"),
        }
    }

    fn images() -> Self {
        Self {
            validate: validate_images_response,
            negative_fixture: br#"{"created":1,"data":[{}]}"#,
            negative_content_type: Some("application/json"),
        }
    }

    fn speech() -> Self {
        Self {
            validate: validate_speech_response,
            negative_fixture: b"",
            negative_content_type: Some("application/json"),
        }
    }

    fn audio_text() -> Self {
        Self {
            validate: validate_audio_text_response,
            negative_fixture: br#"{"text":""}"#,
            negative_content_type: Some("application/json"),
        }
    }
}

async fn endpoint_result(
    endpoint: &'static str,
    configured: bool,
    skip_reason: Option<String>,
    fixture: &'static str,
    response: Option<Result<crate::provider::ProviderResponse>>,
    validator: EndpointValidator,
) -> EndpointConformance {
    let negative_schema_rejected =
        (validator.validate)(validator.negative_fixture, validator.negative_content_type).is_err();
    let Some(response) = response else {
        return EndpointConformance {
            endpoint,
            configured,
            status: None,
            content_type: None,
            fixture,
            positive_schema_valid: false,
            negative_schema_rejected,
            pass: true,
            skip_reason,
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
                fixture,
                positive_schema_valid: false,
                negative_schema_rejected,
                pass: false,
                skip_reason: None,
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
                fixture,
                positive_schema_valid: false,
                negative_schema_rejected,
                pass: false,
                skip_reason: None,
                error: Some(format!("failed to read endpoint response body: {error}")),
            };
        }
    };
    let validation = if status.is_success() {
        (validator.validate)(&bytes, content_type.as_deref())
    } else {
        Err(anyhow::anyhow!("endpoint returned HTTP status {status}"))
    };
    let positive_schema_valid = validation.is_ok();
    let error = match validation {
        Ok(()) if !negative_schema_rejected => {
            Some("endpoint validator accepted its malformed negative fixture".to_string())
        }
        Ok(()) => None,
        Err(error) => Some(error.to_string()),
    };
    EndpointConformance {
        endpoint,
        configured,
        status: Some(status.as_u16()),
        content_type,
        fixture,
        positive_schema_valid,
        negative_schema_rejected,
        pass: positive_schema_valid && negative_schema_rejected,
        skip_reason: None,
        error,
    }
}

fn validate_responses_response(bytes: &[u8], content_type: Option<&str>) -> Result<()> {
    ensure_json_content_type(content_type)?;
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value.get("object").and_then(Value::as_str) == Some("response"),
        "Responses response object must equal response"
    );
    anyhow::ensure!(
        value
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty()),
        "Responses response must include a non-empty id"
    );
    anyhow::ensure!(
        value
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(|model| !model.is_empty()),
        "Responses response must include a non-empty model"
    );
    anyhow::ensure!(
        value
            .get("output")
            .and_then(Value::as_array)
            .is_some_and(|output| !output.is_empty())
            || value
                .get("output_text")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty()),
        "Responses response must include non-empty output or output_text"
    );
    Ok(())
}

fn validate_embeddings_response(bytes: &[u8], content_type: Option<&str>) -> Result<()> {
    ensure_json_content_type(content_type)?;
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value.get("object").and_then(Value::as_str) == Some("list"),
        "embeddings response object must equal list"
    );
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
            .map(|embedding| {
                !embedding.is_empty()
                    && embedding
                        .iter()
                        .all(|component| component.as_f64().is_some())
            })
            .unwrap_or(false),
        "embeddings response must include data[0].embedding array"
    );
    anyhow::ensure!(
        value
            .pointer("/data/0/index")
            .and_then(Value::as_u64)
            .is_some(),
        "embeddings response must include numeric data[0].index"
    );
    Ok(())
}

fn validate_images_response(bytes: &[u8], content_type: Option<&str>) -> Result<()> {
    ensure_json_content_type(content_type)?;
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value.get("created").and_then(Value::as_u64).is_some(),
        "image response must include numeric created"
    );
    let first = value
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first());
    anyhow::ensure!(
        first.is_some(),
        "image response must include non-empty data array"
    );
    anyhow::ensure!(
        first.is_some_and(|image| {
            image
                .get("url")
                .and_then(Value::as_str)
                .is_some_and(|url| !url.is_empty())
                || image
                    .get("b64_json")
                    .and_then(Value::as_str)
                    .is_some_and(|data| !data.is_empty())
        }),
        "image response data item must include non-empty url or b64_json"
    );
    Ok(())
}

fn validate_speech_response(bytes: &[u8], content_type: Option<&str>) -> Result<()> {
    anyhow::ensure!(
        content_type.is_some_and(|value| value.to_ascii_lowercase().starts_with("audio/")),
        "speech response content-type must be audio/*"
    );
    anyhow::ensure!(!bytes.is_empty(), "speech response body must not be empty");
    Ok(())
}

fn validate_audio_text_response(bytes: &[u8], content_type: Option<&str>) -> Result<()> {
    ensure_json_content_type(content_type)?;
    let value = serde_json::from_slice::<Value>(bytes).context("response body is not JSON")?;
    anyhow::ensure!(
        value
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty()),
        "audio transcription/translation response must include non-empty text"
    );
    Ok(())
}

fn ensure_json_content_type(content_type: Option<&str>) -> Result<()> {
    anyhow::ensure!(
        content_type_is_json(content_type),
        "response content-type must be application/json"
    );
    Ok(())
}

fn content_type_is_json(content_type: Option<&str>) -> bool {
    content_type.is_some_and(|value| {
        let value = value.to_ascii_lowercase();
        value.starts_with("application/json") || value.contains("+json")
    })
}

fn audio_probe_request(model: String, route_text: String) -> OpenAiAudioMultipartRequest {
    OpenAiAudioMultipartRequest {
        model,
        route_text,
        parts: vec![OpenAiMultipartPart {
            name: "file".to_string(),
            file_name: Some("autohand-router-conformance.wav".to_string()),
            content_type: Some("audio/wav".to_string()),
            data: Bytes::from_static(
                b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00",
            ),
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
        skip_reason: None,
        status,
        content_type,
        openai_chat_shape: object_ok,
        response_model_matches,
        assistant_content_present,
        usage_present: usage.is_some_and(validate_usage_shape),
        negative_schema_rejected: chat_negative_schema_rejected(),
        prompt_tokens,
        completion_tokens,
        total_tokens,
        error: None,
    }
}

fn chat_negative_schema_rejected() -> bool {
    let malformed = serde_json::json!({
        "object": "chat.completion",
        "model": "fixture",
        "choices": []
    });
    !valid_chat_shape("fixture", &malformed)
}

fn valid_chat_shape(expected_model: &str, value: &Value) -> bool {
    value.get("object").and_then(Value::as_str) == Some("chat.completion")
        && value.get("model").and_then(Value::as_str) == Some(expected_model)
        && value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.is_empty())
        && value.get("usage").is_some_and(validate_usage_shape)
}

fn validate_usage_shape(usage: &Value) -> bool {
    let prompt = usage.get("prompt_tokens").and_then(Value::as_u64);
    let completion = usage.get("completion_tokens").and_then(Value::as_u64);
    let total = usage.get("total_tokens").and_then(Value::as_u64);
    matches!((prompt, completion, total), (Some(prompt), Some(completion), Some(total)) if total >= prompt.saturating_add(completion))
}

fn validate_streaming_feature(
    status: u16,
    content_type: Option<&str>,
    bytes: &[u8],
    _expected_model: &str,
) -> Result<()> {
    anyhow::ensure!(
        (200..300).contains(&status),
        "stream returned HTTP {status}"
    );
    anyhow::ensure!(
        content_type
            .is_some_and(|value| { value.to_ascii_lowercase().starts_with("text/event-stream") }),
        "stream content-type must be text/event-stream"
    );
    let text = std::str::from_utf8(bytes).context("stream response is not UTF-8 SSE")?;
    let mut saw_delta = false;
    let mut saw_usage = false;
    let mut saw_done = false;
    for line in text.lines().map(str::trim) {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let value = serde_json::from_str::<Value>(data).context("SSE data is not JSON")?;
        saw_delta |= value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.is_empty());
        saw_usage |= value.get("usage").is_some_and(validate_usage_shape);
    }
    anyhow::ensure!(saw_delta, "stream omitted a non-empty assistant delta");
    anyhow::ensure!(saw_usage, "stream omitted terminal usage");
    anyhow::ensure!(saw_done, "stream omitted the [DONE] terminal marker");
    Ok(())
}

fn validate_tools_feature(
    status: u16,
    content_type: Option<&str>,
    bytes: &[u8],
    expected_model: &str,
) -> Result<()> {
    let value = feature_json(status, content_type, bytes, expected_model)?;
    let call = value
        .pointer("/choices/0/message/tool_calls/0")
        .context("tool response omitted choices[0].message.tool_calls[0]")?;
    anyhow::ensure!(
        call.get("type").and_then(Value::as_str) == Some("function"),
        "tool call type must equal function"
    );
    anyhow::ensure!(
        call.pointer("/function/name").and_then(Value::as_str) == Some("conformance_echo"),
        "tool response did not call conformance_echo"
    );
    let arguments = call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .context("tool call arguments must be a JSON string")?;
    serde_json::from_str::<Value>(arguments).context("tool call arguments are not valid JSON")?;
    Ok(())
}

fn validate_json_feature(
    status: u16,
    content_type: Option<&str>,
    bytes: &[u8],
    expected_model: &str,
) -> Result<()> {
    let value = feature_json(status, content_type, bytes, expected_model)?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .context("JSON-mode response omitted assistant content")?;
    let parsed = serde_json::from_str::<Value>(content)
        .context("JSON-mode assistant content is not valid JSON")?;
    anyhow::ensure!(parsed.is_object(), "JSON-mode content must be an object");
    Ok(())
}

fn validate_vision_feature(
    status: u16,
    content_type: Option<&str>,
    bytes: &[u8],
    expected_model: &str,
) -> Result<()> {
    let value = feature_json(status, content_type, bytes, expected_model)?;
    anyhow::ensure!(
        value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.is_empty()),
        "vision response omitted assistant content"
    );
    Ok(())
}

fn feature_json(
    status: u16,
    content_type: Option<&str>,
    bytes: &[u8],
    expected_model: &str,
) -> Result<Value> {
    anyhow::ensure!(
        (200..300).contains(&status),
        "feature returned HTTP {status}"
    );
    ensure_json_content_type(content_type)?;
    let value = serde_json::from_slice::<Value>(bytes).context("feature response is not JSON")?;
    anyhow::ensure!(
        value.get("object").and_then(Value::as_str) == Some("chat.completion"),
        "feature response object must equal chat.completion"
    );
    anyhow::ensure!(
        value.get("model").and_then(Value::as_str) == Some(expected_model),
        "feature response model does not match the requested model"
    );
    anyhow::ensure!(
        value.get("usage").is_some_and(validate_usage_shape),
        "feature response omitted a valid usage object"
    );
    Ok(value)
}

fn version_evidence(value: Option<String>, header: &str) -> VersionEvidence {
    VersionEvidence {
        source: if value.is_some() {
            format!("response_header:{header}")
        } else {
            "not_reported".to_string()
        },
        value,
    }
}

fn config_fingerprint(config: &RouterConfig) -> Result<String> {
    let mut redacted = config.clone();
    redacted.auth.bearer_tokens.clear();
    for provider in &mut redacted.providers {
        provider.api_key = None;
        for value in provider.extra_headers.values_mut() {
            *value = "<redacted>".to_string();
        }
    }
    let bytes = serde_json::to_vec(&redacted)?;
    Ok(format!("{:016x}", fnv1a_64(&bytes)))
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
    use crate::config::{
        AuthConfig, BudgetConfig, ClassifierConfig, RuntimeConfig, ScoringConfig, TelemetryConfig,
    };
    use crate::types::{DomainLabel, ModelConfig, ProviderConfig, RouterPolicy};
    use axum::{
        Json, Router,
        body::Body,
        extract::Multipart,
        http::{Response, header},
        response::IntoResponse,
        routing::post,
    };
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
        assert!(report.chat.usage_present);
        assert!(report.chat.negative_schema_rejected);
        assert_eq!(report.chat.total_tokens, Some(9));
        assert!(report.features.iter().all(|feature| feature.pass));
        assert!(
            report
                .features
                .iter()
                .filter(|feature| !feature.attempted)
                .all(|feature| feature.skip_reason.is_some())
        );
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
        assert!(report.chat.usage_present);
        assert!(report.chat.negative_schema_rejected);
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
                .all(|endpoint| endpoint.configured
                    && endpoint.pass
                    && endpoint.positive_schema_valid
                    && endpoint.negative_schema_rejected
                    && endpoint.skip_reason.is_none())
        );
        assert!(
            model_report
                .features
                .iter()
                .all(|feature| feature.declared && feature.attempted && feature.pass)
        );
        assert_eq!(
            model_report.provider_version.value.as_deref(),
            Some("mock-provider-1.2.3")
        );
        assert_eq!(
            model_report.model_version.value.as_deref(),
            Some("mock-model-2026-07")
        );
        assert_eq!(model_report.router_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(model_report.config_fnv1a_64.len(), 16);

        let artifact_path =
            std::env::temp_dir().join(format!("router-conformance-v2-{}.json", std::process::id()));
        std::fs::write(&artifact_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        let catalog = load_verified_endpoint_catalog(&artifact_path).unwrap();
        let verified = catalog
            .get(&("full-provider".to_string(), "full-model".to_string()))
            .unwrap();
        assert_eq!(verified.len(), ModelEndpoint::ALL.len());
        std::fs::remove_file(artifact_path).unwrap();
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
            if !endpoint.configured {
                assert!(endpoint.skip_reason.is_some());
                assert!(endpoint.negative_schema_rejected);
            }
        }
    }

    #[tokio::test]
    async fn conformance_justifies_skipped_chat_and_feature_probes() {
        let base_url = spawn_full_provider_server().await;
        let mut config = full_openai_config(base_url);
        config.models[0].capabilities.supported_endpoints = Some(vec![ModelEndpoint::Responses]);
        config.models[0].capabilities.supports_tools = false;
        config.models[0].capabilities.supports_json = false;
        config.models[0].capabilities.supports_vision = false;
        config.models[0].capabilities.supports_audio = false;

        let report = run_provider_conformance_matrix(config, "skip evidence".to_string())
            .await
            .unwrap();
        let model_report = &report.reports[0];

        assert!(!model_report.chat.configured);
        assert!(model_report.chat.skip_reason.is_some());
        assert!(
            model_report
                .features
                .iter()
                .all(|feature| !feature.attempted && feature.skip_reason.is_some())
        );
        assert!(
            model_report
                .endpoints
                .iter()
                .filter(|endpoint| endpoint.endpoint != "responses")
                .all(|endpoint| !endpoint.configured && endpoint.skip_reason.is_some())
        );
        assert!(model_report.pass);
    }

    #[tokio::test]
    async fn conformance_artifact_fingerprint_does_not_expose_configured_secrets() {
        let base_url = spawn_full_provider_server().await;
        let mut config = full_openai_config(base_url);
        config.providers[0].api_key = Some("provider-secret-value".to_string());
        config.providers[0].extra_headers.insert(
            "x-private-header".to_string(),
            "header-secret-value".to_string(),
        );
        config.auth.bearer_tokens = vec!["router-secret-value".to_string()];

        let report = run_provider_conformance_matrix(config, "secret-safe artifact".to_string())
            .await
            .unwrap();
        let encoded = serde_json::to_string(&report).unwrap();

        assert!(!encoded.contains("provider-secret-value"));
        assert!(!encoded.contains("header-secret-value"));
        assert!(!encoded.contains("router-secret-value"));
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.config_fnv1a_64.len(), 16);
    }

    #[test]
    fn every_endpoint_validator_rejects_its_negative_fixture() {
        for validator in [
            EndpointValidator::responses(),
            EndpointValidator::embeddings(),
            EndpointValidator::images(),
            EndpointValidator::speech(),
            EndpointValidator::audio_text(),
        ] {
            assert!(
                (validator.validate)(validator.negative_fixture, validator.negative_content_type)
                    .is_err()
            );
        }
        assert!(chat_negative_schema_rejected());
    }

    #[test]
    fn schema_v2_catalog_does_not_trust_endpoint_pass_without_schema_evidence() {
        let artifact_path = std::env::temp_dir().join(format!(
            "router-conformance-v2-missing-evidence-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 2,
                "reports": [{
                    "provider": "provider",
                    "model": "model",
                    "chat": {
                        "configured": true,
                        "status": 200,
                        "openai_chat_shape": true,
                        "response_model_matches": true,
                        "assistant_content_present": true,
                        "usage_present": true,
                        "negative_schema_rejected": true
                    },
                    "endpoints": [{
                        "endpoint": "responses",
                        "configured": true,
                        "pass": true
                    }]
                }]
            }))
            .unwrap(),
        )
        .unwrap();

        let catalog = load_verified_endpoint_catalog(&artifact_path).unwrap();
        let verified = catalog
            .get(&("provider".to_string(), "model".to_string()))
            .unwrap();
        assert_eq!(verified, &[ModelEndpoint::Chat]);
        std::fs::remove_file(artifact_path).unwrap();
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
        async fn chat(Json(request): Json<Value>) -> axum::response::Response {
            if request.get("stream").and_then(Value::as_bool) == Some(true) {
                return Response::builder()
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header("x-provider-version", "mock-provider-1.2.3")
                    .header("x-model-version", "mock-model-2026-07")
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
                    serde_json::json!([{
                        "id": "call_conformance",
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
            let mut message = serde_json::json!({
                "role": "assistant",
                "content": content
            });
            if !tool_calls.is_null() {
                message["tool_calls"] = tool_calls;
            }
            (
                [
                    ("x-provider-version", "mock-provider-1.2.3"),
                    ("x-model-version", "mock-model-2026-07"),
                ],
                Json(serde_json::json!({
                "object": "chat.completion",
                "model": request["model"],
                "choices": [{
                    "message": message
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
                })),
            )
                .into_response()
        }
        async fn responses(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "id": "resp_conformance",
                "object": "response",
                "model": request["model"],
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
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
                "created": 1,
                "data": [{ "url": "https://example.test/image.png" }]
            }))
        }
        async fn speech(Json(_request): Json<Value>) -> axum::response::Response {
            Response::builder()
                .header(header::CONTENT_TYPE, "audio/wav")
                .body(Body::from("RIFFmock-audio"))
                .unwrap()
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
                connect_timeout_ms: 5_000,
                stream_idle_timeout_ms: 30_000,
                retry_max_delay_ms: 30_000,
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
                connect_timeout_ms: 5_000,
                stream_idle_timeout_ms: 30_000,
                retry_max_delay_ms: 30_000,
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
                connect_timeout_ms: 5_000,
                stream_idle_timeout_ms: 30_000,
                retry_max_delay_ms: 30_000,
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
        model.capabilities.supports_tools = true;
        model.capabilities.supports_json = true;
        model.capabilities.supports_vision = true;
        model.capabilities.supports_audio = true;
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
                connect_timeout_ms: 5_000,
                stream_idle_timeout_ms: 30_000,
                retry_max_delay_ms: 30_000,
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
