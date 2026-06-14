use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DifficultyLabel {
    Easy,
    Medium,
    Hard,
    NeedsInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityLabel {
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DomainLabel {
    General,
    Summary,
    Coding,
    Design,
    Data,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RouterPolicy {
    Balanced,
    CostEfficient,
    CapabilityHeavy,
    DomainSkills,
}

impl Default for RouterPolicy {
    fn default() -> Self {
        Self::Balanced
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    #[serde(default)]
    pub kind: ProviderKind,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_chat_path")]
    pub chat_path: String,
    #[serde(default = "default_responses_path")]
    pub responses_path: Option<String>,
    #[serde(default = "default_embeddings_path")]
    pub embeddings_path: Option<String>,
    #[serde(default = "default_images_path")]
    pub images_path: Option<String>,
    #[serde(default = "default_speech_path")]
    pub speech_path: Option<String>,
    #[serde(default = "default_audio_transcriptions_path")]
    pub audio_transcriptions_path: Option<String>,
    #[serde(default = "default_audio_translations_path")]
    pub audio_translations_path: Option<String>,
    #[serde(default)]
    pub health_path: Option<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_retries")]
    pub retries: u8,
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    #[serde(default)]
    pub queue_timeout_ms: Option<u64>,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAiCompatible,
    Ollama,
    OllamaNative,
    LlamaCpp,
    LlamaCppNative,
    #[serde(rename = "vllm", alias = "v_llm")]
    Vllm,
    #[serde(rename = "openrouter", alias = "open_router")]
    OpenRouter,
    CloudflareAiGateway,
}

impl Default for ProviderKind {
    fn default() -> Self {
        Self::OpenAiCompatible
    }
}

fn default_chat_path() -> String {
    "/v1/chat/completions".to_string()
}

fn default_responses_path() -> Option<String> {
    Some("/v1/responses".to_string())
}

fn default_embeddings_path() -> Option<String> {
    Some("/v1/embeddings".to_string())
}

fn default_images_path() -> Option<String> {
    Some("/v1/images/generations".to_string())
}

fn default_speech_path() -> Option<String> {
    Some("/v1/audio/speech".to_string())
}

fn default_audio_transcriptions_path() -> Option<String> {
    Some("/v1/audio/transcriptions".to_string())
}

fn default_audio_translations_path() -> Option<String> {
    Some("/v1/audio/translations".to_string())
}

fn default_timeout_ms() -> u64 {
    120_000
}

fn default_retries() -> u8 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub provider: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_capability")]
    pub capability: f32,
    #[serde(default = "default_cost")]
    pub cost_per_million_input: f32,
    #[serde(default = "default_cost")]
    pub cost_per_million_output: f32,
    #[serde(default)]
    pub domains: Vec<DomainLabel>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub local: bool,
}

fn default_capability() -> f32 {
    0.5
}

fn default_cost() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyRequest {
    pub input: String,
    #[serde(default)]
    pub classes: Vec<ClassificationHead>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResponse {
    pub classifications: SelectedClassifications,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawRouterRequest {
    pub input: String,
    #[serde(default)]
    pub mode: LegacyRouterMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawRouterResponse {
    pub difficulty: DifficultyLabel,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRouterRequest {
    pub input: String,
    #[serde(default)]
    pub mode: LegacyRouterMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRouterResponse {
    pub model: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LegacyRouterMode {
    Balanced,
    Aggressive,
}

impl Default for LegacyRouterMode {
    fn default() -> Self {
        Self::Balanced
    }
}

impl LegacyRouterMode {
    pub fn policy(self) -> RouterPolicy {
        match self {
            Self::Balanced => RouterPolicy::Balanced,
            Self::Aggressive => RouterPolicy::CostEfficient,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationHead {
    Difficulty,
    Ambiguity,
    Domain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedClassifications {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<Classification<DifficultyLabel>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambiguity: Option<Classification<AmbiguityLabel>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<Classification<DomainLabel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classifications {
    pub difficulty: Classification<DifficultyLabel>,
    pub ambiguity: Classification<AmbiguityLabel>,
    pub domain: Classification<DomainLabel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification<T> {
    pub class_id: u8,
    pub label: T,
    pub confidence: f32,
    pub meets_threshold: bool,
}

impl SelectedClassifications {
    pub fn from_heads(classifications: Classifications, heads: &[ClassificationHead]) -> Self {
        let include_all = heads.is_empty();
        Self {
            difficulty: (include_all || heads.contains(&ClassificationHead::Difficulty))
                .then_some(classifications.difficulty),
            ambiguity: (include_all || heads.contains(&ClassificationHead::Ambiguity))
                .then_some(classifications.ambiguity),
            domain: (include_all || heads.contains(&ClassificationHead::Domain))
                .then_some(classifications.domain),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodelRequest {
    pub input: String,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    #[serde(default)]
    pub allowed_providers: Vec<String>,
    #[serde(default)]
    pub policy: RouterPolicy,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodelResponse {
    pub model: String,
    pub provider: String,
    pub difficulty: DifficultyLabel,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambiguity: Option<AmbiguityLabel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambiguity_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<DomainLabel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_confidence: Option<f32>,
    pub policy: RouterPolicy,
    pub reason: String,
    pub fallback: bool,
    pub estimated_input_tokens: u32,
    pub requested_output_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub candidates: Vec<RouteCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteCandidate {
    pub model: String,
    pub provider: String,
    pub score: f32,
    pub capability: f32,
    pub estimated_cost: f32,
    pub domain_match: bool,
    pub routing_priority: f32,
    pub latency_penalty: f32,
    pub health_penalty: f32,
    pub context_window: Option<u32>,
    pub context_required: u32,
    pub context_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiEmbeddingsRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiImagesRequest {
    pub model: String,
    pub prompt: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiSpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct OpenAiMultipartPart {
    pub name: String,
    pub file_name: Option<String>,
    pub content_type: Option<String>,
    pub data: Bytes,
}

#[derive(Debug, Clone)]
pub struct OpenAiAudioMultipartRequest {
    pub model: String,
    pub route_text: String,
    pub parts: Vec<OpenAiMultipartPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

impl OpenAiChatRequest {
    pub fn prompt_text(&self) -> String {
        self.messages
            .iter()
            .filter(|message| message.role == "user" || message.role == "system")
            .filter_map(|message| content_to_text(&message.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn into_upstream(mut self, model: String) -> Value {
        self.model = model;
        serde_json::to_value(self).expect("OpenAI chat request serializes")
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.extra
            .get("max_tokens")
            .or_else(|| self.extra.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    }

    pub fn stream(&self) -> bool {
        self.extra
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

impl OpenAiResponsesRequest {
    pub fn prompt_text(&self) -> String {
        content_to_text(&self.input).unwrap_or_default()
    }

    pub fn into_upstream(mut self, model: String) -> Value {
        self.model = model;
        serde_json::to_value(self).expect("OpenAI responses request serializes")
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.extra
            .get("max_output_tokens")
            .or_else(|| self.extra.get("max_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    }

    pub fn stream(&self) -> bool {
        self.extra
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

impl OpenAiEmbeddingsRequest {
    pub fn prompt_text(&self) -> String {
        content_to_text(&self.input).unwrap_or_default()
    }

    pub fn into_upstream(mut self, model: String) -> Value {
        self.model = model;
        serde_json::to_value(self).expect("OpenAI embeddings request serializes")
    }
}

impl OpenAiImagesRequest {
    pub fn prompt_text(&self) -> String {
        self.prompt.clone()
    }

    pub fn into_upstream(mut self, model: String) -> Value {
        self.model = model;
        serde_json::to_value(self).expect("OpenAI images request serializes")
    }
}

impl OpenAiSpeechRequest {
    pub fn prompt_text(&self) -> String {
        self.input.clone()
    }

    pub fn into_upstream(mut self, model: String) -> Value {
        self.model = model;
        serde_json::to_value(self).expect("OpenAI speech request serializes")
    }
}

impl OpenAiAudioMultipartRequest {
    pub fn prompt_text(&self) -> String {
        self.route_text.clone()
    }
}

fn content_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    if part.is_string() {
                        content_to_text(part)
                    } else {
                        part.get("text")
                            .or_else(|| part.get("content"))
                            .and_then(content_to_text)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        Value::Object(object) => object
            .get("text")
            .or_else(|| object.get("content"))
            .or_else(|| object.get("input"))
            .and_then(content_to_text),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AmbiguityLabel, Classification, ClassificationHead, Classifications, DifficultyLabel,
        DomainLabel, LegacyRouterMode, OpenAiEmbeddingsRequest, RawRouterResponse, RouterPolicy,
        SelectedClassifications,
    };
    use serde_json::Value;

    fn classifications() -> Classifications {
        Classifications {
            difficulty: Classification {
                class_id: 0,
                label: DifficultyLabel::Easy,
                confidence: 0.9,
                meets_threshold: true,
            },
            ambiguity: Classification {
                class_id: 0,
                label: AmbiguityLabel::Low,
                confidence: 0.8,
                meets_threshold: true,
            },
            domain: Classification {
                class_id: 2,
                label: DomainLabel::Coding,
                confidence: 0.85,
                meets_threshold: true,
            },
        }
    }

    #[test]
    fn selected_classifications_only_serializes_requested_heads() {
        let selected = SelectedClassifications::from_heads(
            classifications(),
            &[ClassificationHead::Difficulty, ClassificationHead::Domain],
        );
        let value = serde_json::to_value(selected).expect("selected classifications serialize");
        let object = value.as_object().expect("selected classifications object");

        assert!(object.contains_key("difficulty"));
        assert!(object.contains_key("domain"));
        assert!(!object.contains_key("ambiguity"));
    }

    #[test]
    fn selected_classifications_defaults_to_all_heads() {
        let selected = SelectedClassifications::from_heads(classifications(), &[]);
        let value = serde_json::to_value(selected).expect("selected classifications serialize");
        let object = value.as_object().expect("selected classifications object");

        assert!(object.contains_key("difficulty"));
        assert!(object.contains_key("ambiguity"));
        assert!(object.contains_key("domain"));
    }

    #[test]
    fn raw_router_response_matches_legacy_shape() {
        let value = serde_json::to_value(RawRouterResponse {
            difficulty: DifficultyLabel::Easy,
            confidence: 0.93,
        })
        .expect("raw router response serializes");
        let object = value.as_object().expect("raw router response object");

        assert_eq!(
            object.get("difficulty").and_then(Value::as_str),
            Some("easy")
        );
        let confidence = object
            .get("confidence")
            .and_then(Value::as_f64)
            .expect("confidence number");
        assert!((confidence - 0.93).abs() < 0.00001);
    }

    #[test]
    fn embeddings_prompt_text_reads_string_arrays() {
        let request = OpenAiEmbeddingsRequest {
            model: "embedding-model".to_string(),
            input: serde_json::json!(["first prompt", "second prompt"]),
            extra: Default::default(),
        };

        assert_eq!(request.prompt_text(), "first prompt\nsecond prompt");
    }

    #[test]
    fn legacy_aggressive_mode_maps_to_cost_efficient_policy() {
        assert_eq!(LegacyRouterMode::Balanced.policy(), RouterPolicy::Balanced);
        assert_eq!(
            LegacyRouterMode::Aggressive.policy(),
            RouterPolicy::CostEfficient
        );
    }
}
