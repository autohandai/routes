use crate::{
    config::RouterConfig,
    types::{
        ModelConfig, OpenAiAudioMultipartRequest, OpenAiChatRequest, OpenAiEmbeddingsRequest,
        OpenAiImagesRequest, OpenAiResponsesRequest, OpenAiSpeechRequest, ProviderConfig,
        ProviderKind,
    },
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use reqwest::{Client, Response, StatusCode, multipart};
use serde::Serialize;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    time::{sleep, timeout},
};

pub enum ProviderResponse {
    Upstream(Response),
    Buffered {
        status: StatusCode,
        headers: reqwest::header::HeaderMap,
        body: Bytes,
    },
}

impl ProviderResponse {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Upstream(response) => response.status(),
            Self::Buffered { status, .. } => *status,
        }
    }

    pub fn headers(&self) -> &reqwest::header::HeaderMap {
        match self {
            Self::Upstream(response) => response.headers(),
            Self::Buffered { headers, .. } => headers,
        }
    }

    pub async fn bytes(self) -> Result<Bytes, reqwest::Error> {
        match self {
            Self::Upstream(response) => response.bytes().await,
            Self::Buffered { body, .. } => Ok(body),
        }
    }

    pub fn into_upstream(self) -> Option<Response> {
        match self {
            Self::Upstream(response) => Some(response),
            Self::Buffered { .. } => None,
        }
    }
}

#[derive(Clone)]
pub struct ProviderClient {
    http: Client,
    concurrency: Arc<HashMap<String, Arc<Semaphore>>>,
    adapters: Arc<HashMap<ProviderKind, Arc<dyn ProviderAdapter>>>,
}

impl ProviderClient {
    pub fn new(config: &RouterConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .context("failed to build provider HTTP client")?;
        let concurrency = config
            .providers
            .iter()
            .filter_map(|provider| {
                provider
                    .max_concurrency
                    .map(|limit| (provider.name.clone(), Arc::new(Semaphore::new(limit))))
            })
            .collect::<HashMap<_, _>>();
        Ok(Self {
            http,
            concurrency: Arc::new(concurrency),
            adapters: Arc::new(provider_adapters()),
        })
    }

    pub async fn send_chat(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiChatRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_chat(&self.http, provider, model, request)
            .await
    }

    pub async fn send_responses(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiResponsesRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_responses(&self.http, provider, model, request)
            .await
    }

    pub async fn send_embeddings(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiEmbeddingsRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_embeddings(&self.http, provider, model, request)
            .await
    }

    pub async fn send_images(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiImagesRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_images(&self.http, provider, model, request)
            .await
    }

    pub async fn send_speech(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiSpeechRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_speech(&self.http, provider, model, request)
            .await
    }

    pub async fn send_audio_transcription(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_audio_transcription(&self.http, provider, model, request)
            .await
    }

    pub async fn send_audio_translation(
        &self,
        config: &RouterConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let _permit = self.acquire_permit(provider).await?;
        self.adapter_for(provider)?
            .send_audio_translation(&self.http, provider, model, request)
            .await
    }

    pub async fn check_provider(&self, provider: &ProviderConfig) -> ProviderHealth {
        match self.adapter_for(provider) {
            Ok(adapter) => adapter.check_provider(&self.http, provider).await,
            Err(error) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: format!("{:?}", provider.kind),
                status: ProviderHealthStatus::Error,
                status_code: None,
                error: Some(error.to_string()),
            },
        }
    }

    fn adapter_for(&self, provider: &ProviderConfig) -> Result<Arc<dyn ProviderAdapter>> {
        self.adapters
            .get(&provider.kind)
            .cloned()
            .with_context(|| format!("provider adapter {:?} is not registered", provider.kind))
    }

    async fn acquire_permit(
        &self,
        provider: &ProviderConfig,
    ) -> Result<Option<OwnedSemaphorePermit>> {
        let Some(semaphore) = self.concurrency.get(&provider.name).cloned() else {
            return Ok(None);
        };
        let permit = if let Some(timeout_ms) = provider.queue_timeout_ms {
            timeout(Duration::from_millis(timeout_ms), semaphore.acquire_owned())
                .await
                .context("provider concurrency queue timeout exceeded")??
        } else {
            semaphore
                .try_acquire_owned()
                .with_context(|| format!("provider {} concurrency limit reached", provider.name))?
        };
        Ok(Some(permit))
    }

    pub fn error_json(message: impl Into<String>) -> Value {
        serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "autohand_router_error"
            }
        })
    }
}

fn provider_adapters() -> HashMap<ProviderKind, Arc<dyn ProviderAdapter>> {
    [
        (
            ProviderKind::OpenAiCompatible,
            Arc::new(OpenAiCompatibleAdapter::new(
                AdapterProfile::OpenAiCompatible,
            )) as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::Ollama,
            Arc::new(OpenAiCompatibleAdapter::new(AdapterProfile::Ollama))
                as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::OllamaNative,
            Arc::new(OllamaNativeAdapter) as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::LlamaCpp,
            Arc::new(OpenAiCompatibleAdapter::new(AdapterProfile::LlamaCpp))
                as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::LlamaCppNative,
            Arc::new(LlamaCppNativeAdapter) as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::Vllm,
            Arc::new(OpenAiCompatibleAdapter::new(AdapterProfile::Vllm))
                as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::OpenRouter,
            Arc::new(OpenAiCompatibleAdapter::new(AdapterProfile::OpenRouter))
                as Arc<dyn ProviderAdapter>,
        ),
        (
            ProviderKind::CloudflareAiGateway,
            Arc::new(OpenAiCompatibleAdapter::new(
                AdapterProfile::CloudflareAiGateway,
            )) as Arc<dyn ProviderAdapter>,
        ),
    ]
    .into_iter()
    .collect()
}

#[derive(Debug)]
pub struct OllamaNativeAdapter;

#[async_trait]
impl ProviderAdapter for OllamaNativeAdapter {
    fn name(&self) -> &'static str {
        "ollama_native"
    }

    async fn send_chat(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiChatRequest,
    ) -> Result<ProviderResponse> {
        let url = join_url(&provider.base_url, &provider.chat_path);
        let body = ollama_chat_body(model, request);
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return ollama_chat_response(model, response).await,
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream native Ollama chat failed"),
            }
        }

        anyhow::bail!("upstream native Ollama chat failed without response")
    }

    async fn check_provider(&self, http: &Client, provider: &ProviderConfig) -> ProviderHealth {
        let path = provider.health_path.as_deref().unwrap_or("/api/tags");
        let url = join_url(&provider.base_url, path);
        let request = authorized_provider_request(
            http.get(url).timeout(provider.timeout()),
            provider,
            self.default_headers(),
        );
        match request.send().await {
            Ok(response) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: if response.status().is_success() {
                    ProviderHealthStatus::Ok
                } else {
                    ProviderHealthStatus::Error
                },
                status_code: Some(response.status().as_u16()),
                error: None,
            },
            Err(error) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: ProviderHealthStatus::Error,
                status_code: None,
                error: Some(error.to_string()),
            },
        }
    }

    async fn send_responses(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiResponsesRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/responses")
    }

    async fn send_embeddings(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiEmbeddingsRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/embeddings")
    }

    async fn send_images(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiImagesRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/images/generations")
    }

    async fn send_speech(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiSpeechRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/audio/speech")
    }

    async fn send_audio_transcription(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/audio/transcriptions")
    }

    async fn send_audio_translation(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("ollama_native does not support /v1/audio/translations")
    }
}

fn ollama_chat_body(model: &ModelConfig, request: OpenAiChatRequest) -> Value {
    let messages = request
        .messages
        .into_iter()
        .map(|message| {
            serde_json::json!({
                "role": message.role,
                "content": provider_content_to_text(&message.content)
            })
        })
        .collect::<Vec<_>>();
    let mut body = serde_json::Map::from_iter([
        ("model".to_string(), Value::String(model.id.clone())),
        ("messages".to_string(), Value::Array(messages)),
        ("stream".to_string(), Value::Bool(false)),
    ]);
    if let Some(options) = request.extra.get("options").cloned() {
        body.insert("options".to_string(), options);
    }
    Value::Object(body)
}

async fn ollama_chat_response(model: &ModelConfig, response: Response) -> Result<ProviderResponse> {
    let status = response.status();
    let mut headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .context("failed to read native Ollama chat response")?;
    if !status.is_success() {
        return Ok(ProviderResponse::Buffered {
            status,
            headers,
            body: bytes,
        });
    }
    let value = serde_json::from_slice::<Value>(&bytes)
        .context("failed to parse native Ollama chat response")?;
    let content = value
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let prompt_tokens = value
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = value.get("eval_count").and_then(Value::as_u64).unwrap_or(0);
    let transformed = serde_json::json!({
        "id": format!("ollama-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": 0,
        "model": model.id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": if value.get("done").and_then(Value::as_bool).unwrap_or(false) {
                "stop"
            } else {
                "length"
            }
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens.saturating_add(completion_tokens)
        }
    });
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    Ok(ProviderResponse::Buffered {
        status,
        headers,
        body: Bytes::from(serde_json::to_vec(&transformed)?),
    })
}

fn provider_content_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(provider_content_to_text)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(object) => object
            .get("text")
            .or_else(|| object.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

#[derive(Debug)]
pub struct LlamaCppNativeAdapter;

#[async_trait]
impl ProviderAdapter for LlamaCppNativeAdapter {
    fn name(&self) -> &'static str {
        "llama_cpp_native"
    }

    async fn send_chat(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiChatRequest,
    ) -> Result<ProviderResponse> {
        let url = join_url(&provider.base_url, &provider.chat_path);
        let body = llama_cpp_completion_body(request);
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return llama_cpp_completion_response(model, response).await,
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => {
                    return Err(error).context("upstream native llama.cpp completion failed");
                }
            }
        }

        anyhow::bail!("upstream native llama.cpp completion failed without response")
    }

    async fn check_provider(&self, http: &Client, provider: &ProviderConfig) -> ProviderHealth {
        let path = provider.health_path.as_deref().unwrap_or("/health");
        let url = join_url(&provider.base_url, path);
        let request = authorized_provider_request(
            http.get(url).timeout(provider.timeout()),
            provider,
            self.default_headers(),
        );
        match request.send().await {
            Ok(response) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: if response.status().is_success() {
                    ProviderHealthStatus::Ok
                } else {
                    ProviderHealthStatus::Error
                },
                status_code: Some(response.status().as_u16()),
                error: None,
            },
            Err(error) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: ProviderHealthStatus::Error,
                status_code: None,
                error: Some(error.to_string()),
            },
        }
    }

    async fn send_responses(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiResponsesRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/responses")
    }

    async fn send_embeddings(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiEmbeddingsRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/embeddings")
    }

    async fn send_images(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiImagesRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/images/generations")
    }

    async fn send_speech(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiSpeechRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/audio/speech")
    }

    async fn send_audio_transcription(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/audio/transcriptions")
    }

    async fn send_audio_translation(
        &self,
        _http: &Client,
        _provider: &ProviderConfig,
        _model: &ModelConfig,
        _request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        anyhow::bail!("llama_cpp_native does not support /v1/audio/translations")
    }
}

fn llama_cpp_completion_body(request: OpenAiChatRequest) -> Value {
    let prompt = request
        .messages
        .into_iter()
        .map(|message| {
            format!(
                "{}: {}",
                message.role,
                provider_content_to_text(&message.content)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut body = serde_json::Map::from_iter([
        (
            "prompt".to_string(),
            Value::String(format!("{prompt}\nassistant:")),
        ),
        ("stream".to_string(), Value::Bool(false)),
    ]);
    if let Some(tokens) = request
        .extra
        .get("max_tokens")
        .or_else(|| request.extra.get("max_completion_tokens"))
        .cloned()
    {
        body.insert("n_predict".to_string(), tokens);
    }
    if let Some(temperature) = request.extra.get("temperature").cloned() {
        body.insert("temperature".to_string(), temperature);
    }
    Value::Object(body)
}

async fn llama_cpp_completion_response(
    model: &ModelConfig,
    response: Response,
) -> Result<ProviderResponse> {
    let status = response.status();
    let mut headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .context("failed to read native llama.cpp completion response")?;
    if !status.is_success() {
        return Ok(ProviderResponse::Buffered {
            status,
            headers,
            body: bytes,
        });
    }
    let value = serde_json::from_slice::<Value>(&bytes)
        .context("failed to parse native llama.cpp completion response")?;
    let content = value
        .get("content")
        .or_else(|| value.get("response"))
        .or_else(|| value.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let prompt_tokens = value
        .get("tokens_evaluated")
        .or_else(|| value.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = value
        .get("tokens_predicted")
        .or_else(|| value.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let transformed = serde_json::json!({
        "id": format!("llama-cpp-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": 0,
        "model": model.id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": if value.get("stopped_eos").and_then(Value::as_bool).unwrap_or(false)
                || value.get("stop").and_then(Value::as_bool).unwrap_or(false) {
                "stop"
            } else {
                "length"
            }
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens.saturating_add(completion_tokens)
        }
    });
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    Ok(ProviderResponse::Buffered {
        status,
        headers,
        body: Bytes::from(serde_json::to_vec(&transformed)?),
    })
}

fn authorized_provider_request(
    mut builder: reqwest::RequestBuilder,
    provider: &ProviderConfig,
    default_headers: &[(&'static str, &'static str)],
) -> reqwest::RequestBuilder {
    let api_key = provider.api_key.clone().or_else(|| {
        provider
            .api_key_env
            .as_ref()
            .and_then(|key| std::env::var(key).ok())
    });
    if let Some(api_key) = api_key {
        if !api_key.is_empty() {
            builder = builder.bearer_auth(api_key);
        }
    }
    for (key, value) in default_headers {
        builder = builder.header(*key, *value);
    }
    for (key, value) in &provider.extra_headers {
        builder = builder.header(key, value);
    }
    builder
}

#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    fn name(&self) -> &'static str;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    async fn send_chat(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiChatRequest,
    ) -> Result<ProviderResponse>;

    async fn check_provider(&self, http: &Client, provider: &ProviderConfig) -> ProviderHealth;

    async fn send_responses(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiResponsesRequest,
    ) -> Result<ProviderResponse>;

    async fn send_embeddings(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiEmbeddingsRequest,
    ) -> Result<ProviderResponse>;

    async fn send_images(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiImagesRequest,
    ) -> Result<ProviderResponse>;

    async fn send_speech(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiSpeechRequest,
    ) -> Result<ProviderResponse>;

    async fn send_audio_transcription(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse>;

    async fn send_audio_translation(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse>;
}

#[derive(Debug, Clone, Copy)]
enum AdapterProfile {
    OpenAiCompatible,
    Ollama,
    LlamaCpp,
    Vllm,
    OpenRouter,
    CloudflareAiGateway,
}

impl AdapterProfile {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "open_ai_compatible",
            Self::Ollama => "ollama_openai",
            Self::LlamaCpp => "llama_cpp_openai",
            Self::Vllm => "vllm_openai",
            Self::OpenRouter => "openrouter",
            Self::CloudflareAiGateway => "cloudflare_ai_gateway_openai",
        }
    }

    fn default_headers(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::OpenRouter => &[
                ("HTTP-Referer", "https://autohand.ai"),
                ("X-Title", "Autohand Router"),
            ],
            _ => &[],
        }
    }
}

#[derive(Debug)]
pub struct OpenAiCompatibleAdapter {
    profile: AdapterProfile,
}

impl OpenAiCompatibleAdapter {
    fn new(profile: AdapterProfile) -> Self {
        Self { profile }
    }
}

#[async_trait]
impl ProviderAdapter for OpenAiCompatibleAdapter {
    fn name(&self) -> &'static str {
        self.profile.name()
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        self.profile.default_headers()
    }

    async fn send_chat(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiChatRequest,
    ) -> Result<ProviderResponse> {
        let url = join_url(&provider.base_url, &provider.chat_path);
        let body = request.into_upstream(model.id.clone());
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return Ok(ProviderResponse::Upstream(response)),
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream request failed"),
            }
        }

        anyhow::bail!("upstream request failed without response")
    }

    async fn check_provider(&self, http: &Client, provider: &ProviderConfig) -> ProviderHealth {
        let Some(path) = provider.health_path.as_deref() else {
            return ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: ProviderHealthStatus::Unknown,
                status_code: None,
                error: Some("health_path is not configured".to_string()),
            };
        };
        let url = join_url(&provider.base_url, path);
        let request = authorized_provider_request(
            http.get(url).timeout(provider.timeout()),
            provider,
            self.default_headers(),
        );
        match request.send().await {
            Ok(response) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: if response.status().is_success() {
                    ProviderHealthStatus::Ok
                } else {
                    ProviderHealthStatus::Error
                },
                status_code: Some(response.status().as_u16()),
                error: None,
            },
            Err(error) => ProviderHealth {
                provider: provider.name.clone(),
                adapter: self.name().to_string(),
                status: ProviderHealthStatus::Error,
                status_code: None,
                error: Some(error.to_string()),
            },
        }
    }

    async fn send_responses(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiResponsesRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .responses_path
            .as_deref()
            .context("provider does not support /v1/responses")?;
        let url = join_url(&provider.base_url, path);
        let body = request.into_upstream(model.id.clone());
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return Ok(ProviderResponse::Upstream(response)),
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream responses request failed"),
            }
        }

        anyhow::bail!("upstream responses request failed without response")
    }

    async fn send_embeddings(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiEmbeddingsRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .embeddings_path
            .as_deref()
            .context("provider does not support /v1/embeddings")?;
        let url = join_url(&provider.base_url, path);
        let body = request.into_upstream(model.id.clone());
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return Ok(ProviderResponse::Upstream(response)),
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream embeddings request failed"),
            }
        }

        anyhow::bail!("upstream embeddings request failed without response")
    }

    async fn send_images(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiImagesRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .images_path
            .as_deref()
            .context("provider does not support /v1/images/generations")?;
        let url = join_url(&provider.base_url, path);
        let body = request.into_upstream(model.id.clone());
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return Ok(ProviderResponse::Upstream(response)),
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream images request failed"),
            }
        }

        anyhow::bail!("upstream images request failed without response")
    }

    async fn send_speech(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiSpeechRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .speech_path
            .as_deref()
            .context("provider does not support /v1/audio/speech")?;
        let url = join_url(&provider.base_url, path);
        let body = request.into_upstream(model.id.clone());
        let attempts = provider.retries.saturating_add(1);

        for attempt in 0..attempts {
            let builder = authorized_provider_request(
                http.post(&url).timeout(provider.timeout()).json(&body),
                provider,
                self.default_headers(),
            );
            match builder.send().await {
                Ok(response)
                    if is_transient_status(response.status()) && attempt + 1 < attempts =>
                {
                    sleep(backoff(attempt)).await;
                }
                Ok(response) => return Ok(ProviderResponse::Upstream(response)),
                Err(error)
                    if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    sleep(backoff(attempt)).await;
                }
                Err(error) => return Err(error).context("upstream speech request failed"),
            }
        }

        anyhow::bail!("upstream speech request failed without response")
    }

    async fn send_audio_transcription(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .audio_transcriptions_path
            .as_deref()
            .context("provider does not support /v1/audio/transcriptions")?;
        send_audio_multipart(
            http,
            provider,
            model,
            request,
            path,
            self.default_headers(),
            "upstream audio transcription request failed",
        )
        .await
    }

    async fn send_audio_translation(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        model: &ModelConfig,
        request: OpenAiAudioMultipartRequest,
    ) -> Result<ProviderResponse> {
        let path = provider
            .audio_translations_path
            .as_deref()
            .context("provider does not support /v1/audio/translations")?;
        send_audio_multipart(
            http,
            provider,
            model,
            request,
            path,
            self.default_headers(),
            "upstream audio translation request failed",
        )
        .await
    }
}

async fn send_audio_multipart(
    http: &Client,
    provider: &ProviderConfig,
    model: &ModelConfig,
    request: OpenAiAudioMultipartRequest,
    path: &str,
    default_headers: &[(&'static str, &'static str)],
    error_context: &'static str,
) -> Result<ProviderResponse> {
    let url = join_url(&provider.base_url, path);
    let attempts = provider.retries.saturating_add(1);

    for attempt in 0..attempts {
        let form = audio_multipart_form(request.clone(), model)?;
        let builder = authorized_provider_request(
            http.post(&url).timeout(provider.timeout()).multipart(form),
            provider,
            default_headers,
        );
        match builder.send().await {
            Ok(response) if is_transient_status(response.status()) && attempt + 1 < attempts => {
                sleep(backoff(attempt)).await;
            }
            Ok(response) => return Ok(ProviderResponse::Upstream(response)),
            Err(error) if attempt + 1 < attempts && (error.is_timeout() || error.is_connect()) => {
                sleep(backoff(attempt)).await;
            }
            Err(error) => return Err(error).context(error_context),
        }
    }

    anyhow::bail!("{error_context} without response")
}

fn audio_multipart_form(
    request: OpenAiAudioMultipartRequest,
    model: &ModelConfig,
) -> Result<multipart::Form> {
    let mut form = multipart::Form::new().text("model", model.id.clone());
    for part in request.parts {
        let mut upstream_part = multipart::Part::bytes(part.data.to_vec());
        if let Some(file_name) = part.file_name {
            upstream_part = upstream_part.file_name(file_name);
        }
        if let Some(content_type) = part.content_type {
            upstream_part = upstream_part
                .mime_str(&content_type)
                .with_context(|| format!("invalid multipart content type {content_type}"))?;
        }
        form = form.part(part.name, upstream_part);
    }
    Ok(form)
}

impl ProviderConfig {
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderHealth {
    pub provider: String,
    pub adapter: String,
    pub status: ProviderHealthStatus,
    pub status_code: Option<u16>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderHealthStatus {
    Ok,
    Error,
    Unknown,
}

pub fn is_transient_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

fn backoff(attempt: u8) -> Duration {
    Duration::from_millis(100 * 2_u64.pow(attempt as u32))
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{
            AuthConfig, BudgetConfig, ClassifierConfig, RouterConfig, RuntimeConfig, ScoringConfig,
            TelemetryConfig,
        },
        types::{
            ChatMessage, DomainLabel, OpenAiChatRequest, OpenAiEmbeddingsRequest,
            OpenAiImagesRequest, OpenAiResponsesRequest, OpenAiSpeechRequest, ProviderKind,
            RouterPolicy,
        },
    };
    use axum::{Json, Router, http::HeaderMap, routing::post};
    use tokio::net::TcpListener;

    #[test]
    fn transient_statuses_are_retryable() {
        assert!(is_transient_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_transient_status(StatusCode::REQUEST_TIMEOUT));
        assert!(is_transient_status(StatusCode::BAD_GATEWAY));
        assert!(!is_transient_status(StatusCode::BAD_REQUEST));
        assert!(!is_transient_status(StatusCode::UNAUTHORIZED));
    }

    #[tokio::test]
    async fn sends_responses_requests_to_configured_path_with_selected_model() {
        let base_url = spawn_provider_server().await;
        let config = provider_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_responses(
                &config,
                model,
                OpenAiResponsesRequest {
                    model: "ignored".to_string(),
                    input: Value::String("hello".to_string()),
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(value["input"], "hello");
    }

    #[tokio::test]
    async fn sends_embeddings_requests_to_configured_path_with_selected_model() {
        let base_url = spawn_provider_server().await;
        let config = provider_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_embeddings(
                &config,
                model,
                OpenAiEmbeddingsRequest {
                    model: "ignored".to_string(),
                    input: Value::String("hello".to_string()),
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(value["input"], "hello");
        assert!(value["data"][0]["embedding"].is_array());
    }

    #[tokio::test]
    async fn openrouter_adapter_adds_default_attribution_headers() {
        let base_url = spawn_provider_server().await;
        let mut config = provider_test_config(base_url);
        config.providers[0].kind = ProviderKind::OpenRouter;
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_chat(
                &config,
                model,
                OpenAiChatRequest {
                    model: "ignored".to_string(),
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: Value::String("hello".to_string()),
                    }],
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(value["http_referer"], "https://autohand.ai");
        assert_eq!(value["x_title"], "Autohand Router");
    }

    #[tokio::test]
    async fn vllm_adapter_uses_openai_paths_with_distinct_profile() {
        let base_url = spawn_provider_server().await;
        let mut config = provider_test_config(base_url);
        config.providers[0].kind = ProviderKind::Vllm;
        config.providers[0].health_path = Some("/health".to_string());
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_chat(
                &config,
                model,
                OpenAiChatRequest {
                    model: "ignored".to_string(),
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: Value::String("hello vllm".to_string()),
                    }],
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;
        let health = client.check_provider(&config.providers[0]).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(health.adapter, "vllm_openai");
    }

    #[tokio::test]
    async fn sends_images_requests_to_configured_path_with_selected_model() {
        let base_url = spawn_provider_server().await;
        let config = provider_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_images(
                &config,
                model,
                OpenAiImagesRequest {
                    model: "ignored".to_string(),
                    prompt: "draw a router".to_string(),
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(value["prompt"], "draw a router");
        assert!(value["data"][0]["url"].is_string());
    }

    #[tokio::test]
    async fn sends_speech_requests_to_configured_path_with_selected_model() {
        let base_url = spawn_provider_server().await;
        let config = provider_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("responses-model").unwrap();
        let upstream = client
            .send_speech(
                &config,
                model,
                OpenAiSpeechRequest {
                    model: "ignored".to_string(),
                    input: "read this".to_string(),
                    voice: "alloy".to_string(),
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["model"], "responses-model");
        assert_eq!(value["input"], "read this");
        assert_eq!(value["voice"], "alloy");
    }

    #[tokio::test]
    async fn ollama_native_adapter_transforms_chat_request_and_response() {
        let base_url = spawn_ollama_native_server().await;
        let config = ollama_native_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("llama3.1:8b").unwrap();
        let upstream = client
            .send_chat(
                &config,
                model,
                OpenAiChatRequest {
                    model: "ignored".to_string(),
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: Value::String("hello native ollama".to_string()),
                    }],
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["model"], "llama3.1:8b");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            "native:llama3.1:8b:hello native ollama:false"
        );
        assert_eq!(value["usage"]["prompt_tokens"], 7);
        assert_eq!(value["usage"]["completion_tokens"], 3);
        assert_eq!(value["usage"]["total_tokens"], 10);
    }

    #[tokio::test]
    async fn llama_cpp_native_adapter_transforms_completion_request_and_response() {
        let base_url = spawn_llama_cpp_native_server().await;
        let config = llama_cpp_native_test_config(base_url);
        let client = ProviderClient::new(&config).unwrap();
        let model = config.find_model("llama-cpp-q4").unwrap();
        let upstream = client
            .send_chat(
                &config,
                model,
                OpenAiChatRequest {
                    model: "ignored".to_string(),
                    messages: vec![ChatMessage {
                        role: "user".to_string(),
                        content: Value::String("hello native llama".to_string()),
                    }],
                    extra: serde_json::Map::from_iter([
                        ("max_tokens".to_string(), Value::from(12)),
                        ("temperature".to_string(), Value::from(0.2)),
                    ]),
                },
            )
            .await
            .unwrap();
        assert!(upstream.status().is_success());
        let value = response_json(upstream).await;

        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["model"], "llama-cpp-q4");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            "native:user: hello native llama\nassistant::12:0.2"
        );
        assert_eq!(value["usage"]["prompt_tokens"], 11);
        assert_eq!(value["usage"]["completion_tokens"], 4);
        assert_eq!(value["usage"]["total_tokens"], 15);
    }

    #[tokio::test]
    async fn provider_health_reports_selected_adapter() {
        let base_url = spawn_provider_server().await;
        let mut config = provider_test_config(base_url);
        config.providers[0].kind = ProviderKind::Ollama;
        config.providers[0].health_path = Some("/health".to_string());
        let client = ProviderClient::new(&config).unwrap();
        let health = client.check_provider(&config.providers[0]).await;

        assert_eq!(health.provider, "mock");
        assert_eq!(health.adapter, "ollama_openai");
        assert!(matches!(health.status, ProviderHealthStatus::Ok));
    }

    async fn spawn_provider_server() -> String {
        async fn chat(headers: HeaderMap, Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "id": "chat-test",
                "object": "chat.completion",
                "model": request["model"],
                "http_referer": headers
                    .get("HTTP-Referer")
                    .and_then(|value| value.to_str().ok()),
                "x_title": headers
                    .get("X-Title")
                    .and_then(|value| value.to_str().ok())
            }))
        }
        async fn responses(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "id": "resp-test",
                "object": "response",
                "model": request["model"],
                "input": request["input"],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5
                }
            }))
        }
        async fn embeddings(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "object": "list",
                "model": request["model"],
                "input": request["input"],
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3]
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "total_tokens": 3
                }
            }))
        }
        async fn images(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "created": 123,
                "model": request["model"],
                "prompt": request["prompt"],
                "data": [{
                    "url": "https://example.test/router.png"
                }]
            }))
        }
        async fn speech(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "input": request["input"],
                "voice": request["voice"],
                "audio": "mock"
            }))
        }
        async fn health() -> Json<Value> {
            Json(serde_json::json!({ "ok": true }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/custom/responses", post(responses))
            .route("/custom/embeddings", post(embeddings))
            .route("/custom/images", post(images))
            .route("/custom/speech", post(speech))
            .route("/health", post(health).get(health));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_ollama_native_server() -> String {
        async fn chat(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "model": request["model"],
                "message": {
                    "role": "assistant",
                    "content": format!(
                        "native:{}:{}:{}",
                        request["model"].as_str().unwrap_or_default(),
                        request["messages"][0]["content"].as_str().unwrap_or_default(),
                        request["stream"].as_bool().unwrap_or(true)
                    )
                },
                "done": true,
                "prompt_eval_count": 7,
                "eval_count": 3
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/api/chat", post(chat));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_llama_cpp_native_server() -> String {
        async fn completion(Json(request): Json<Value>) -> Json<Value> {
            Json(serde_json::json!({
                "content": format!(
                    "native:{}:{}:{}",
                    request["prompt"].as_str().unwrap_or_default(),
                    request["n_predict"].as_u64().unwrap_or_default(),
                    request["temperature"].as_f64().unwrap_or_default()
                ),
                "tokens_evaluated": 11,
                "tokens_predicted": 4,
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

    async fn response_json(response: ProviderResponse) -> Value {
        let bytes = response.bytes().await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn provider_test_config(base_url: String) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "responses-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "mock".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url,
                api_key_env: None,
                api_key: None,
                chat_path: "/v1/chat/completions".to_string(),
                responses_path: Some("/custom/responses".to_string()),
                embeddings_path: Some("/custom/embeddings".to_string()),
                images_path: Some("/custom/images".to_string()),
                speech_path: Some("/custom/speech".to_string()),
                audio_transcriptions_path: Some("/custom/transcriptions".to_string()),
                audio_translations_path: Some("/custom/translations".to_string()),
                health_path: None,
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: Default::default(),
            }],
            models: vec![ModelConfig {
                id: "responses-model".to_string(),
                provider: "mock".to_string(),
                aliases: vec![],
                capability: 0.5,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::General],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
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

    fn ollama_native_test_config(base_url: String) -> RouterConfig {
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
            models: vec![ModelConfig {
                id: "llama3.1:8b".to_string(),
                provider: "ollama-native".to_string(),
                aliases: vec!["local-native".to_string()],
                capability: 0.5,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::General],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
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

    fn llama_cpp_native_test_config(base_url: String) -> RouterConfig {
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
            models: vec![ModelConfig {
                id: "llama-cpp-q4".to_string(),
                provider: "llama-cpp-native".to_string(),
                aliases: vec!["local-llamacpp-native".to_string()],
                capability: 0.5,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::General],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
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
}
