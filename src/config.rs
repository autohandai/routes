use crate::types::{ModelCapability, ModelConfig, ModelEndpoint, ProviderConfig, RouterPolicy};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    net::SocketAddr,
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
    pub default_model: String,
    #[serde(default)]
    pub policy: RouterPolicy,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub classifier: ClassifierConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub scoring: ScoringConfig,
    #[serde(default)]
    pub budget: BudgetConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub shadow_eval: ShadowEvalConfig,
    #[serde(default)]
    pub safety: SafetyRoutingConfig,
    #[serde(default)]
    pub sticky_routing: StickyRoutingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifierConfig {
    #[serde(default)]
    pub backend: ClassifierBackend,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_easy_threshold")]
    pub easy_threshold: f32,
    #[serde(default = "default_hard_threshold")]
    pub hard_threshold: f32,
    #[serde(default)]
    pub llm_judge_model: Option<String>,
    #[serde(default = "default_llm_judge_timeout_ms")]
    pub llm_judge_timeout_ms: u64,
    #[serde(default)]
    pub adapters: ClassifierAdaptersConfig,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierBackend {
    #[default]
    Heuristic,
    LlmJudge,
    RouteLlm,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifierAdaptersConfig {
    #[serde(default)]
    pub llm_judge: ClassifierModelAdapterConfig,
    #[serde(default)]
    pub route_llm: ClassifierModelAdapterConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifierModelAdapterConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_llm_judge_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub prompt_template: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    #[serde(default)]
    pub bearer_tokens: Vec<String>,
    #[serde(default)]
    pub bearer_token_env: Vec<String>,
    #[serde(default)]
    pub allow_unauthenticated_network: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetConfig {
    #[serde(default)]
    pub max_chat_requests: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub max_estimated_cost_micros: Option<u64>,
    #[serde(default)]
    pub accounting: BudgetAccountingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetAccountingConfig {
    #[serde(default)]
    pub backend: BudgetAccountingBackend,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default = "default_budget_lock_timeout_ms")]
    pub lock_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAccountingBackend {
    Process,
    File,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub decision_log_path: Option<String>,
    #[serde(default)]
    pub include_inputs: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    #[serde(default)]
    pub semantic: SemanticCacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowEvalConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_shadow_eval_sample_rate")]
    pub sample_rate: f32,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub include_bodies: bool,
    #[serde(default = "default_shadow_eval_max_body_chars")]
    pub max_body_chars: usize,
    #[serde(default)]
    pub judge: ShadowEvalJudgeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowEvalJudgeConfig {
    #[serde(default = "default_shadow_eval_judge_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_shadow_eval_judge_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub prompt_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyRoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_safety_unsafe_action")]
    pub unsafe_action: SafetyRoutingAction,
    #[serde(default)]
    pub sensitive_action: SafetyRoutingAction,
    #[serde(default)]
    pub force_model: Option<String>,
    #[serde(default = "default_safety_redaction_replacement")]
    pub redaction_replacement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StickyRoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_sticky_routing_ttl_seconds")]
    pub ttl_seconds: u64,
    #[serde(default)]
    pub prefer_model: bool,
    #[serde(default)]
    pub backend: StickyRoutingBackend,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default = "default_sticky_routing_lock_timeout_ms")]
    pub lock_timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StickyRoutingBackend {
    Memory,
    File,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SafetyRoutingAction {
    #[default]
    Allow,
    Reject,
    Redact,
    ForceRoute,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_semantic_cache_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_semantic_cache_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_semantic_cache_ttl_seconds")]
    pub ttl_seconds: u64,
    #[serde(default = "default_semantic_cache_max_entries")]
    pub max_entries: usize,
    #[serde(default)]
    pub backend: SemanticCacheBackend,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default = "default_semantic_cache_lock_timeout_ms")]
    pub lock_timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCacheBackend {
    Memory,
    File,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default = "default_graceful_shutdown_timeout_ms")]
    pub graceful_shutdown_timeout_ms: u64,
    #[serde(default)]
    pub provider_health_sampler: ProviderHealthSamplerConfig,
    /// Optional provider-conformance matrix used to prove every declared
    /// model endpoint before the config is accepted.
    #[serde(default)]
    pub provider_conformance_artifact: Option<String>,
    #[serde(default)]
    pub ingress: IngressConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngressConfig {
    #[serde(default = "default_max_json_body_bytes")]
    pub max_json_body_bytes: usize,
    #[serde(default = "default_max_multipart_body_bytes")]
    pub max_multipart_body_bytes: usize,
    #[serde(default = "default_body_idle_timeout_ms")]
    pub body_idle_timeout_ms: u64,
    #[serde(default)]
    pub max_in_flight_requests: Option<usize>,
    #[serde(default = "default_admission_queue_timeout_ms")]
    pub admission_queue_timeout_ms: u64,
    #[serde(default)]
    pub per_credential_requests_per_minute: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderHealthSamplerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_provider_health_interval_ms")]
    pub interval_ms: u64,
    #[serde(default = "default_provider_health_initial_delay_ms")]
    pub initial_delay_ms: u64,
    #[serde(default = "default_provider_health_check_timeout_ms")]
    pub check_timeout_ms: u64,
    #[serde(default = "default_provider_health_max_concurrent_checks")]
    pub max_concurrent_checks: usize,
    #[serde(default = "default_provider_health_observation_ttl_ms")]
    pub observation_ttl_ms: u64,
    #[serde(default = "default_circuit_failure_threshold")]
    pub circuit_failure_threshold: u32,
    #[serde(default = "default_circuit_open_ms")]
    pub circuit_open_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringConfig {
    #[serde(default = "default_balanced_weights")]
    pub balanced: PolicyWeights,
    #[serde(default = "default_lowest_cost_acceptable_weights")]
    pub lowest_cost_acceptable: PolicyWeights,
    #[serde(default = "default_fastest_healthy_weights")]
    pub fastest_healthy: PolicyWeights,
    #[serde(default = "default_highest_quality_weights")]
    pub highest_quality: PolicyWeights,
    #[serde(default = "default_local_first_weights")]
    pub local_first: PolicyWeights,
    #[serde(default = "default_privacy_first_weights")]
    pub privacy_first: PolicyWeights,
    #[serde(default = "default_multimodal_first_weights")]
    pub multimodal_first: PolicyWeights,
    #[serde(default = "default_floor_weights")]
    pub floor: PolicyWeights,
    #[serde(default = "default_nitro_weights")]
    pub nitro: PolicyWeights,
    #[serde(default = "default_quality_weights")]
    pub quality: PolicyWeights,
    #[serde(default = "default_cost_efficient_weights")]
    pub cost_efficient: PolicyWeights,
    #[serde(default = "default_capability_heavy_weights")]
    pub capability_heavy: PolicyWeights,
    #[serde(default = "default_domain_skills_weights")]
    pub domain_skills: PolicyWeights,
    #[serde(default)]
    pub model_priorities: HashMap<String, f32>,
    #[serde(default)]
    pub provider_priorities: HashMap<String, f32>,
    #[serde(default)]
    pub provider_latency_p95_ms: HashMap<String, u32>,
    #[serde(default)]
    pub provider_health_penalties: HashMap<String, f32>,
    #[serde(default = "default_priority_weight")]
    pub priority_weight: f32,
    #[serde(default = "default_latency_weight")]
    pub latency_weight: f32,
    #[serde(default = "default_health_weight")]
    pub health_weight: f32,
    #[serde(default)]
    pub learned: LearnedScoringConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyWeights {
    pub capability_fit: f32,
    pub domain_bonus: f32,
    pub cost: f32,
    pub overkill: f32,
    #[serde(default)]
    pub raw_capability: f32,
    #[serde(default)]
    pub latency: f32,
    #[serde(default)]
    pub health: f32,
    #[serde(default)]
    pub local_bonus: f32,
    #[serde(default)]
    pub remote_penalty: f32,
    #[serde(default)]
    pub multimodal_capability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedScoringConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub weight: f32,
    #[serde(default)]
    pub bias: f32,
    #[serde(default)]
    pub feature_weights: HashMap<String, f32>,
    #[serde(default)]
    pub model_biases: HashMap<String, f32>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            balanced: default_balanced_weights(),
            lowest_cost_acceptable: default_lowest_cost_acceptable_weights(),
            fastest_healthy: default_fastest_healthy_weights(),
            highest_quality: default_highest_quality_weights(),
            local_first: default_local_first_weights(),
            privacy_first: default_privacy_first_weights(),
            multimodal_first: default_multimodal_first_weights(),
            floor: default_floor_weights(),
            nitro: default_nitro_weights(),
            quality: default_quality_weights(),
            cost_efficient: default_cost_efficient_weights(),
            capability_heavy: default_capability_heavy_weights(),
            domain_skills: default_domain_skills_weights(),
            model_priorities: HashMap::new(),
            provider_priorities: HashMap::new(),
            provider_latency_p95_ms: HashMap::new(),
            provider_health_penalties: HashMap::new(),
            priority_weight: default_priority_weight(),
            latency_weight: default_latency_weight(),
            health_weight: default_health_weight(),
            learned: LearnedScoringConfig::default(),
        }
    }
}

impl Default for LearnedScoringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            bias: 0.0,
            feature_weights: HashMap::new(),
            model_biases: HashMap::new(),
        }
    }
}

fn default_balanced_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.60,
        domain_bonus: 0.20,
        cost: 0.20,
        overkill: 1.0,
        raw_capability: 0.0,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_lowest_cost_acceptable_weights() -> PolicyWeights {
    default_floor_weights()
}

fn default_fastest_healthy_weights() -> PolicyWeights {
    default_nitro_weights()
}

fn default_highest_quality_weights() -> PolicyWeights {
    default_quality_weights()
}

fn default_local_first_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.48,
        domain_bonus: 0.14,
        cost: 0.18,
        overkill: 0.8,
        raw_capability: 0.04,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.34,
        remote_penalty: 0.18,
        multimodal_capability: 0.0,
    }
}

fn default_privacy_first_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.42,
        domain_bonus: 0.12,
        cost: 0.12,
        overkill: 0.6,
        raw_capability: 0.06,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.48,
        remote_penalty: 0.36,
        multimodal_capability: 0.0,
    }
}

fn default_multimodal_first_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.42,
        domain_bonus: 0.14,
        cost: 0.10,
        overkill: 0.4,
        raw_capability: 0.08,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.46,
    }
}

fn default_floor_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.34,
        domain_bonus: 0.10,
        cost: 0.56,
        overkill: 1.8,
        raw_capability: 0.0,
        latency: default_latency_weight() * 0.8,
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_nitro_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.44,
        domain_bonus: 0.12,
        cost: 0.12,
        overkill: 0.6,
        raw_capability: 0.08,
        latency: 0.42,
        health: 1.25,
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_quality_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.16,
        domain_bonus: 0.10,
        cost: 0.02,
        overkill: 0.0,
        raw_capability: 0.82,
        latency: default_latency_weight() * 0.6,
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_cost_efficient_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.42,
        domain_bonus: 0.16,
        cost: 0.42,
        overkill: 1.4,
        raw_capability: 0.0,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_capability_heavy_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.20,
        domain_bonus: 0.08,
        cost: 0.05,
        overkill: 0.0,
        raw_capability: 0.72,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_domain_skills_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.38,
        domain_bonus: 0.48,
        cost: 0.10,
        overkill: 0.0,
        raw_capability: 0.10,
        latency: default_latency_weight(),
        health: default_health_weight(),
        local_bonus: 0.0,
        remote_penalty: 0.0,
        multimodal_capability: 0.0,
    }
}

fn default_priority_weight() -> f32 {
    0.08
}

fn default_latency_weight() -> f32 {
    0.05
}

fn default_health_weight() -> f32 {
    1.0
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            backend: ClassifierBackend::Heuristic,
            confidence_threshold: default_threshold(),
            easy_threshold: default_easy_threshold(),
            hard_threshold: default_hard_threshold(),
            llm_judge_model: None,
            llm_judge_timeout_ms: default_llm_judge_timeout_ms(),
            adapters: Default::default(),
        }
    }
}

impl Default for ClassifierModelAdapterConfig {
    fn default() -> Self {
        Self {
            model: None,
            timeout_ms: default_llm_judge_timeout_ms(),
            prompt_template: None,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            graceful_shutdown_timeout_ms: default_graceful_shutdown_timeout_ms(),
            provider_health_sampler: ProviderHealthSamplerConfig::default(),
            provider_conformance_artifact: None,
            ingress: IngressConfig::default(),
        }
    }
}

impl Default for IngressConfig {
    fn default() -> Self {
        Self {
            max_json_body_bytes: default_max_json_body_bytes(),
            max_multipart_body_bytes: default_max_multipart_body_bytes(),
            body_idle_timeout_ms: default_body_idle_timeout_ms(),
            max_in_flight_requests: None,
            admission_queue_timeout_ms: default_admission_queue_timeout_ms(),
            per_credential_requests_per_minute: None,
        }
    }
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            embedding_model: default_semantic_cache_embedding_model(),
            similarity_threshold: default_semantic_cache_similarity_threshold(),
            ttl_seconds: default_semantic_cache_ttl_seconds(),
            max_entries: default_semantic_cache_max_entries(),
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: default_semantic_cache_lock_timeout_ms(),
        }
    }
}

impl Default for SemanticCacheBackend {
    fn default() -> Self {
        Self::Memory
    }
}

impl Default for ShadowEvalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate: default_shadow_eval_sample_rate(),
            output_path: None,
            include_bodies: false,
            max_body_chars: default_shadow_eval_max_body_chars(),
            judge: ShadowEvalJudgeConfig::default(),
        }
    }
}

impl Default for ShadowEvalJudgeConfig {
    fn default() -> Self {
        Self {
            enabled: default_shadow_eval_judge_enabled(),
            model: None,
            timeout_ms: default_shadow_eval_judge_timeout_ms(),
            prompt_template: None,
        }
    }
}

impl Default for SafetyRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            unsafe_action: default_safety_unsafe_action(),
            sensitive_action: SafetyRoutingAction::Allow,
            force_model: None,
            redaction_replacement: default_safety_redaction_replacement(),
        }
    }
}

impl Default for StickyRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl_seconds: default_sticky_routing_ttl_seconds(),
            prefer_model: true,
            backend: StickyRoutingBackend::Memory,
            file_path: None,
            lock_timeout_ms: default_sticky_routing_lock_timeout_ms(),
        }
    }
}

impl Default for StickyRoutingBackend {
    fn default() -> Self {
        Self::Memory
    }
}

impl Default for ProviderHealthSamplerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_ms: default_provider_health_interval_ms(),
            initial_delay_ms: default_provider_health_initial_delay_ms(),
            check_timeout_ms: default_provider_health_check_timeout_ms(),
            max_concurrent_checks: default_provider_health_max_concurrent_checks(),
            observation_ttl_ms: default_provider_health_observation_ttl_ms(),
            circuit_failure_threshold: default_circuit_failure_threshold(),
            circuit_open_ms: default_circuit_open_ms(),
        }
    }
}

fn default_bind() -> String {
    "127.0.0.1:8080".to_string()
}

fn default_threshold() -> f32 {
    0.62
}

fn default_easy_threshold() -> f32 {
    0.28
}

fn default_hard_threshold() -> f32 {
    0.62
}

fn default_llm_judge_timeout_ms() -> u64 {
    2_500
}

fn default_graceful_shutdown_timeout_ms() -> u64 {
    30_000
}

fn default_max_json_body_bytes() -> usize {
    2 * 1024 * 1024
}

fn default_max_multipart_body_bytes() -> usize {
    32 * 1024 * 1024
}

fn default_body_idle_timeout_ms() -> u64 {
    30_000
}

fn default_admission_queue_timeout_ms() -> u64 {
    100
}

fn default_provider_health_interval_ms() -> u64 {
    30_000
}

fn default_provider_health_initial_delay_ms() -> u64 {
    500
}

fn default_provider_health_check_timeout_ms() -> u64 {
    5_000
}

fn default_provider_health_max_concurrent_checks() -> usize {
    8
}

fn default_provider_health_observation_ttl_ms() -> u64 {
    90_000
}

fn default_circuit_failure_threshold() -> u32 {
    3
}

fn default_circuit_open_ms() -> u64 {
    30_000
}

fn default_semantic_cache_embedding_model() -> String {
    "local-hash".to_string()
}

fn default_semantic_cache_similarity_threshold() -> f32 {
    0.92
}

fn default_semantic_cache_ttl_seconds() -> u64 {
    3_600
}

fn default_semantic_cache_max_entries() -> usize {
    1_024
}

fn default_semantic_cache_lock_timeout_ms() -> u64 {
    1_000
}

fn default_shadow_eval_sample_rate() -> f32 {
    0.01
}

fn default_shadow_eval_max_body_chars() -> usize {
    4_096
}

fn default_shadow_eval_judge_enabled() -> bool {
    true
}

fn default_shadow_eval_judge_timeout_ms() -> u64 {
    5_000
}

fn default_safety_unsafe_action() -> SafetyRoutingAction {
    SafetyRoutingAction::Reject
}

fn default_safety_redaction_replacement() -> String {
    "[redacted]".to_string()
}

fn default_sticky_routing_ttl_seconds() -> u64 {
    1_800
}

fn default_sticky_routing_lock_timeout_ms() -> u64 {
    1_000
}

fn default_budget_lock_timeout_ms() -> u64 {
    1_000
}

impl Default for BudgetAccountingConfig {
    fn default() -> Self {
        Self {
            backend: BudgetAccountingBackend::Process,
            file_path: None,
            lock_timeout_ms: default_budget_lock_timeout_ms(),
        }
    }
}

impl Default for BudgetAccountingBackend {
    fn default() -> Self {
        Self::Process
    }
}

impl RouterConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config = serde_yaml::from_str::<Self>(&raw)
            .with_context(|| format!("failed to parse YAML config {}", path.display()))?;
        config.validate_conformance_artifact(path.parent().unwrap_or_else(|| Path::new(".")))?;
        config.validate_core()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_conformance_artifact(Path::new("."))?;
        self.validate_core()
    }

    fn validate_core(&self) -> Result<()> {
        anyhow::ensure!(!self.models.is_empty(), "at least one model is required");
        anyhow::ensure!(
            !self.providers.is_empty(),
            "at least one provider is required"
        );
        validate_unique_providers(&self.providers)?;
        validate_unique_models_and_aliases(&self.models)?;
        let providers = self
            .providers
            .iter()
            .map(|provider| provider.name.as_str())
            .collect::<HashSet<_>>();
        for model in &self.models {
            anyhow::ensure!(
                !model.provider.trim().is_empty(),
                "model {} provider cannot be empty",
                model.id
            );
            anyhow::ensure!(
                providers.contains(model.provider.as_str()),
                "model {} references unknown provider {}",
                model.id,
                model.provider
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&model.capability),
                "model {} capability must be between 0.0 and 1.0",
                model.id
            );
            let provider = self
                .providers
                .iter()
                .find(|provider| provider.name == model.provider)
                .expect("provider existence checked above");
            if let Some(endpoints) = &model.capabilities.supported_endpoints {
                anyhow::ensure!(
                    !endpoints.is_empty(),
                    "model {} capabilities.supported_endpoints cannot be empty",
                    model.id
                );
                let mut unique = HashSet::new();
                for endpoint in endpoints {
                    anyhow::ensure!(
                        unique.insert(*endpoint),
                        "model {} capabilities.supported_endpoints contains duplicate {}",
                        model.id,
                        endpoint.as_str()
                    );
                    anyhow::ensure!(
                        provider.supports_endpoint(*endpoint),
                        "model {} declares endpoint {} but provider {} ({:?}) has no compatible configured path",
                        model.id,
                        endpoint.as_str(),
                        provider.name,
                        provider.kind
                    );
                }
            }
            for capability in ModelCapability::ALL {
                anyhow::ensure!(
                    !model.capabilities.supports(&capability)
                        || provider.kind.adapter_supports_capability(&capability),
                    "model {} declares capability {} but provider adapter {} cannot preserve that request contract",
                    model.id,
                    capability.as_str(),
                    provider.kind.chat_adapter_contract().name
                );
            }
        }
        for provider in &self.providers {
            validate_provider(provider)?;
            if let Some(limit) = provider.max_concurrency {
                anyhow::ensure!(
                    limit > 0,
                    "provider {} max_concurrency must be greater than zero",
                    provider.name
                );
            }
        }
        self.validate_scoring_hints(&providers)?;
        anyhow::ensure!(
            self.find_model(&self.default_model).is_some(),
            "default_model {} does not match a configured model id or alias",
            self.default_model
        );
        self.classifier.validate(self)?;
        anyhow::ensure!(
            self.classifier.easy_threshold < self.classifier.hard_threshold,
            "classifier.easy_threshold must be lower than classifier.hard_threshold"
        );
        self.auth.validate(&self.bind)?;
        self.scoring.validate()?;
        self.budget.validate()?;
        self.telemetry.validate()?;
        self.runtime.validate()?;
        self.cache.validate(self)?;
        self.shadow_eval.validate(self)?;
        self.safety.validate(self)?;
        self.sticky_routing.validate()?;
        self.scoring.validate_model_references(self)?;
        Ok(())
    }

    pub fn provider_map(&self) -> HashMap<String, ProviderConfig> {
        self.providers
            .iter()
            .map(|provider| (provider.name.clone(), provider.clone()))
            .collect()
    }

    pub fn find_model(&self, id_or_alias: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|model| {
            model.id == id_or_alias || model.aliases.iter().any(|alias| alias == id_or_alias)
        })
    }

    fn validate_conformance_artifact(&self, config_dir: &Path) -> Result<()> {
        let Some(artifact_path) = self
            .runtime
            .provider_conformance_artifact
            .as_deref()
            .map(str::trim)
            .filter(|path| !path.is_empty())
        else {
            return Ok(());
        };
        let artifact_path = Path::new(artifact_path);
        let resolved = if artifact_path.is_absolute() {
            artifact_path.to_path_buf()
        } else {
            config_dir.join(artifact_path)
        };
        let catalog = crate::conformance::load_verified_endpoint_catalog(&resolved)?;
        for model in &self.models {
            let key = (model.provider.clone(), model.id.clone());
            let verified = catalog.get(&key).with_context(|| {
                format!(
                    "conformance artifact {} has no report for provider {} model {}",
                    resolved.display(),
                    model.provider,
                    model.id
                )
            })?;
            let declared = model
                .capabilities
                .supported_endpoints
                .clone()
                .unwrap_or_else(|| vec![ModelEndpoint::Chat]);
            for endpoint in declared {
                anyhow::ensure!(
                    verified.contains(&endpoint),
                    "conformance artifact {} does not verify endpoint {} for provider {} model {}",
                    resolved.display(),
                    endpoint.as_str(),
                    model.provider,
                    model.id
                );
            }
        }
        Ok(())
    }

    fn validate_scoring_hints(&self, providers: &HashSet<&str>) -> Result<()> {
        for model in self.scoring.model_priorities.keys() {
            anyhow::ensure!(
                self.find_model(model).is_some(),
                "scoring.model_priorities references unknown model or alias {model}"
            );
        }
        for provider in self
            .scoring
            .provider_priorities
            .keys()
            .chain(self.scoring.provider_latency_p95_ms.keys())
            .chain(self.scoring.provider_health_penalties.keys())
        {
            anyhow::ensure!(
                providers.contains(provider.as_str()),
                "scoring provider hint references unknown provider {provider}"
            );
        }
        Ok(())
    }
}

impl AuthConfig {
    pub(crate) fn validate(&self, bind: &str) -> Result<()> {
        for token in &self.bearer_tokens {
            anyhow::ensure!(
                !token.is_empty() && !token.chars().any(char::is_whitespace),
                "auth.bearer_tokens entries cannot be empty or contain whitespace"
            );
        }
        for env_name in &self.bearer_token_env {
            anyhow::ensure!(
                !env_name.trim().is_empty(),
                "auth.bearer_token_env entries cannot be empty"
            );
        }
        anyhow::ensure!(
            bind_is_loopback(bind)
                || self.allow_unauthenticated_network
                || !self.bearer_tokens.is_empty()
                || !self.bearer_token_env.is_empty(),
            "auth is required for non-loopback bind {bind}; configure bearer tokens or explicitly set auth.allow_unauthenticated_network"
        );
        Ok(())
    }
}

pub(crate) fn bind_is_loopback(bind: &str) -> bool {
    bind.parse::<SocketAddr>()
        .map(|address| address.ip().is_loopback())
        .unwrap_or_else(|_| {
            bind.rsplit_once(':')
                .is_some_and(|(host, _)| host.eq_ignore_ascii_case("localhost"))
        })
}

fn validate_unique_providers(providers: &[ProviderConfig]) -> Result<()> {
    let mut names = HashSet::new();
    for provider in providers {
        anyhow::ensure!(
            !provider.name.trim().is_empty(),
            "provider name cannot be empty"
        );
        anyhow::ensure!(
            names.insert(provider.name.as_str()),
            "duplicate provider name {}",
            provider.name
        );
    }
    Ok(())
}

fn validate_unique_models_and_aliases(models: &[ModelConfig]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut handles = HashMap::<&str, &str>::new();
    for model in models {
        anyhow::ensure!(!model.id.trim().is_empty(), "model id cannot be empty");
        anyhow::ensure!(
            ids.insert(model.id.as_str()),
            "duplicate model id {}",
            model.id
        );
        let previous = handles.insert(model.id.as_str(), model.id.as_str());
        anyhow::ensure!(
            previous.is_none(),
            "model id {} collides with another model alias",
            model.id
        );
        let mut aliases = HashSet::new();
        for alias in &model.aliases {
            anyhow::ensure!(
                !alias.trim().is_empty(),
                "model {} alias cannot be empty",
                model.id
            );
            anyhow::ensure!(
                aliases.insert(alias.as_str()),
                "model {} has duplicate alias {}",
                model.id,
                alias
            );
            let previous = handles.insert(alias.as_str(), model.id.as_str());
            anyhow::ensure!(
                previous.is_none(),
                "model alias {alias} collides with another model id or alias"
            );
        }
    }
    Ok(())
}

fn validate_provider(provider: &ProviderConfig) -> Result<()> {
    anyhow::ensure!(
        provider.base_url.starts_with("http://") || provider.base_url.starts_with("https://"),
        "provider {} base_url must start with http:// or https://",
        provider.name
    );
    if matches!(
        provider.kind,
        crate::types::ProviderKind::OllamaNative | crate::types::ProviderKind::LlamaCppNative
    ) {
        for (endpoint, configured) in [
            ("responses", provider.responses_path.is_some()),
            ("embeddings", provider.embeddings_path.is_some()),
            ("images", provider.images_path.is_some()),
            ("speech", provider.speech_path.is_some()),
            (
                "audio_transcriptions",
                provider.audio_transcriptions_path.is_some(),
            ),
            (
                "audio_translations",
                provider.audio_translations_path.is_some(),
            ),
        ] {
            anyhow::ensure!(
                !configured,
                "provider {} kind {:?} cannot configure unsupported endpoint {}",
                provider.name,
                provider.kind,
                endpoint
            );
        }
    }
    anyhow::ensure!(
        !provider.chat_path.trim().is_empty(),
        "provider {} chat_path cannot be empty",
        provider.name
    );
    anyhow::ensure!(
        provider.chat_path.starts_with('/'),
        "provider {} chat_path must start with /",
        provider.name
    );
    if let Some(responses_path) = &provider.responses_path {
        anyhow::ensure!(
            responses_path.starts_with('/'),
            "provider {} responses_path must start with /",
            provider.name
        );
    }
    if let Some(embeddings_path) = &provider.embeddings_path {
        anyhow::ensure!(
            embeddings_path.starts_with('/'),
            "provider {} embeddings_path must start with /",
            provider.name
        );
    }
    if let Some(images_path) = &provider.images_path {
        anyhow::ensure!(
            images_path.starts_with('/'),
            "provider {} images_path must start with /",
            provider.name
        );
    }
    if let Some(speech_path) = &provider.speech_path {
        anyhow::ensure!(
            speech_path.starts_with('/'),
            "provider {} speech_path must start with /",
            provider.name
        );
    }
    if let Some(audio_transcriptions_path) = &provider.audio_transcriptions_path {
        anyhow::ensure!(
            audio_transcriptions_path.starts_with('/'),
            "provider {} audio_transcriptions_path must start with /",
            provider.name
        );
    }
    if let Some(audio_translations_path) = &provider.audio_translations_path {
        anyhow::ensure!(
            audio_translations_path.starts_with('/'),
            "provider {} audio_translations_path must start with /",
            provider.name
        );
    }
    anyhow::ensure!(
        provider.timeout_ms > 0,
        "provider {} timeout_ms must be greater than zero",
        provider.name
    );
    if let Some(queue_timeout_ms) = provider.queue_timeout_ms {
        anyhow::ensure!(
            queue_timeout_ms > 0,
            "provider {} queue_timeout_ms must be greater than zero",
            provider.name
        );
    }
    if let Some(api_key_env) = &provider.api_key_env {
        anyhow::ensure!(
            !api_key_env.trim().is_empty(),
            "provider {} api_key_env cannot be empty",
            provider.name
        );
    }
    if let Some(health_path) = &provider.health_path {
        anyhow::ensure!(
            health_path.starts_with('/'),
            "provider {} health_path must start with /",
            provider.name
        );
    }
    for (key, value) in &provider.extra_headers {
        anyhow::ensure!(
            !key.trim().is_empty(),
            "provider {} extra_headers cannot contain an empty header name",
            provider.name
        );
        anyhow::ensure!(
            !value.contains('\n') && !value.contains('\r'),
            "provider {} extra header {} contains an invalid newline",
            provider.name,
            key
        );
    }
    Ok(())
}

impl RuntimeConfig {
    pub fn graceful_shutdown_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.graceful_shutdown_timeout_ms)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.graceful_shutdown_timeout_ms > 0,
            "runtime.graceful_shutdown_timeout_ms must be greater than zero"
        );
        self.provider_health_sampler.validate()?;
        if let Some(path) = &self.provider_conformance_artifact {
            anyhow::ensure!(
                !path.trim().is_empty(),
                "runtime.provider_conformance_artifact cannot be empty"
            );
        }
        self.ingress.validate()?;
        Ok(())
    }
}

impl IngressConfig {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_json_body_bytes > 0,
            "runtime.ingress.max_json_body_bytes must be greater than zero"
        );
        anyhow::ensure!(
            self.max_multipart_body_bytes > 0,
            "runtime.ingress.max_multipart_body_bytes must be greater than zero"
        );
        anyhow::ensure!(
            self.body_idle_timeout_ms > 0,
            "runtime.ingress.body_idle_timeout_ms must be greater than zero"
        );
        anyhow::ensure!(
            self.admission_queue_timeout_ms > 0,
            "runtime.ingress.admission_queue_timeout_ms must be greater than zero"
        );
        if let Some(limit) = self.max_in_flight_requests {
            anyhow::ensure!(
                limit > 0,
                "runtime.ingress.max_in_flight_requests must be greater than zero"
            );
        }
        if let Some(limit) = self.per_credential_requests_per_minute {
            anyhow::ensure!(
                limit > 0,
                "runtime.ingress.per_credential_requests_per_minute must be greater than zero"
            );
        }
        Ok(())
    }
}

impl ProviderHealthSamplerConfig {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.interval_ms > 0,
            "runtime.provider_health_sampler.interval_ms must be greater than zero"
        );
        anyhow::ensure!(
            self.check_timeout_ms > 0,
            "runtime.provider_health_sampler.check_timeout_ms must be greater than zero"
        );
        anyhow::ensure!(
            self.max_concurrent_checks > 0,
            "runtime.provider_health_sampler.max_concurrent_checks must be greater than zero"
        );
        anyhow::ensure!(
            self.observation_ttl_ms > 0,
            "runtime.provider_health_sampler.observation_ttl_ms must be greater than zero"
        );
        anyhow::ensure!(
            self.circuit_failure_threshold > 0,
            "runtime.provider_health_sampler.circuit_failure_threshold must be greater than zero"
        );
        anyhow::ensure!(
            self.circuit_open_ms > 0,
            "runtime.provider_health_sampler.circuit_open_ms must be greater than zero"
        );
        Ok(())
    }
}

impl TelemetryConfig {
    fn validate(&self) -> Result<()> {
        if let Some(path) = &self.decision_log_path {
            anyhow::ensure!(
                !path.trim().is_empty(),
                "telemetry.decision_log_path cannot be empty"
            );
        }
        Ok(())
    }
}

impl ClassifierConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        anyhow::ensure!(
            self.confidence_threshold.is_finite()
                && (0.0..=1.0).contains(&self.confidence_threshold),
            "classifier.confidence_threshold must be between 0.0 and 1.0"
        );
        anyhow::ensure!(
            self.easy_threshold.is_finite()
                && self.hard_threshold.is_finite()
                && self.easy_threshold < self.hard_threshold,
            "classifier.easy_threshold must be lower than classifier.hard_threshold"
        );
        if let Some(judge) = &self.llm_judge_model {
            anyhow::ensure!(
                config.find_model(judge).is_some(),
                "classifier.llm_judge_model {judge} does not match a configured model id or alias"
            );
        }
        let adapter = self.active_adapter();
        if matches!(
            self.active_backend(),
            ClassifierBackend::LlmJudge | ClassifierBackend::RouteLlm
        ) {
            let model = adapter.model.as_deref().unwrap_or_default().trim();
            anyhow::ensure!(
                !model.is_empty(),
                "classifier.adapters.{}.model is required when classifier.backend is {}",
                self.active_backend().config_key(),
                self.active_backend().config_key()
            );
            anyhow::ensure!(
                config.find_model(model).is_some(),
                "classifier.adapters.{}.model {model} does not match a configured model id or alias",
                self.active_backend().config_key()
            );
        }
        adapter.validate(self.active_backend())?;
        Ok(())
    }

    pub fn active_backend(&self) -> ClassifierBackend {
        if self.backend == ClassifierBackend::Heuristic && self.llm_judge_model.is_some() {
            ClassifierBackend::LlmJudge
        } else {
            self.backend
        }
    }

    pub fn active_adapter(&self) -> ClassifierModelAdapterConfig {
        match self.active_backend() {
            ClassifierBackend::Heuristic => ClassifierModelAdapterConfig::default(),
            ClassifierBackend::LlmJudge => {
                let mut adapter = self.adapters.llm_judge.clone();
                if adapter.model.is_none() {
                    adapter.model = self.llm_judge_model.clone();
                }
                if adapter.timeout_ms == default_llm_judge_timeout_ms()
                    && self.llm_judge_timeout_ms != default_llm_judge_timeout_ms()
                {
                    adapter.timeout_ms = self.llm_judge_timeout_ms;
                }
                adapter
            }
            ClassifierBackend::RouteLlm => self.adapters.route_llm.clone(),
        }
    }
}

impl ClassifierBackend {
    pub fn config_key(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::LlmJudge => "llm_judge",
            Self::RouteLlm => "route_llm",
        }
    }
}

impl ClassifierModelAdapterConfig {
    fn validate(&self, backend: ClassifierBackend) -> Result<()> {
        anyhow::ensure!(
            self.timeout_ms > 0,
            "classifier.adapters.{}.timeout_ms must be greater than zero",
            backend.config_key()
        );
        if let Some(template) = &self.prompt_template {
            anyhow::ensure!(
                template.contains("{input}"),
                "classifier.adapters.{}.prompt_template must include {{input}}",
                backend.config_key()
            );
        }
        Ok(())
    }
}

impl CacheConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        self.semantic.validate(config)
    }
}

impl SemanticCacheConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        anyhow::ensure!(
            !self.enabled || !self.embedding_model.trim().is_empty(),
            "cache.semantic.embedding_model cannot be empty when semantic cache is enabled"
        );
        if self.enabled && self.embedding_model.trim() != "local-hash" {
            let model = config.find_model(&self.embedding_model).with_context(|| {
                format!(
                    "cache.semantic.embedding_model {} does not match a configured model id or alias",
                    self.embedding_model
                )
            })?;
            let provider = config
                .providers
                .iter()
                .find(|provider| provider.name == model.provider)
                .with_context(|| {
                    format!(
                        "cache.semantic.embedding_model {} references missing provider {}",
                        self.embedding_model, model.provider
                    )
                })?;
            anyhow::ensure!(
                provider.embeddings_path.is_some(),
                "cache.semantic.embedding_model {} provider {} does not support embeddings",
                self.embedding_model,
                provider.name
            );
            anyhow::ensure!(
                model
                    .capabilities
                    .supports_endpoint(ModelEndpoint::Embeddings),
                "cache.semantic.embedding_model {} does not support embeddings",
                self.embedding_model
            );
        }
        anyhow::ensure!(
            self.similarity_threshold.is_finite()
                && (0.0..=1.0).contains(&self.similarity_threshold),
            "cache.semantic.similarity_threshold must be between 0.0 and 1.0"
        );
        anyhow::ensure!(
            self.ttl_seconds > 0,
            "cache.semantic.ttl_seconds must be greater than zero"
        );
        anyhow::ensure!(
            self.max_entries > 0,
            "cache.semantic.max_entries must be greater than zero"
        );
        anyhow::ensure!(
            self.lock_timeout_ms > 0,
            "cache.semantic.lock_timeout_ms must be greater than zero"
        );
        if self.enabled && self.backend == SemanticCacheBackend::File {
            let path = self.file_path.as_deref().unwrap_or_default().trim();
            anyhow::ensure!(
                !path.is_empty(),
                "cache.semantic.file_path is required when backend is file"
            );
        }
        Ok(())
    }
}

impl ShadowEvalConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        anyhow::ensure!(
            self.sample_rate.is_finite() && (0.0..=1.0).contains(&self.sample_rate),
            "shadow_eval.sample_rate must be between 0.0 and 1.0"
        );
        if self.enabled {
            anyhow::ensure!(
                !self
                    .output_path
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty(),
                "shadow_eval.output_path is required when shadow_eval.enabled is true"
            );
        }
        anyhow::ensure!(
            self.max_body_chars > 0,
            "shadow_eval.max_body_chars must be greater than zero"
        );
        self.judge.validate(config)?;
        Ok(())
    }
}

impl ShadowEvalJudgeConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        anyhow::ensure!(
            self.timeout_ms > 0,
            "shadow_eval.judge.timeout_ms must be greater than zero"
        );
        if let Some(model) = self
            .model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            anyhow::ensure!(
                config.find_model(model).is_some(),
                "shadow_eval.judge.model {model} does not match a configured model id or alias"
            );
        }
        if let Some(template) = &self.prompt_template {
            anyhow::ensure!(
                template.contains("{input}")
                    && template.contains("{selected_answer}")
                    && template.contains("{shadow_answer}"),
                "shadow_eval.judge.prompt_template must contain {{input}}, {{selected_answer}}, and {{shadow_answer}}"
            );
        }
        Ok(())
    }
}

impl SafetyRoutingConfig {
    fn validate(&self, config: &RouterConfig) -> Result<()> {
        anyhow::ensure!(
            !self.redaction_replacement.trim().is_empty(),
            "safety.redaction_replacement cannot be empty"
        );
        if self.enabled
            && matches!(
                (self.unsafe_action, self.sensitive_action),
                (SafetyRoutingAction::ForceRoute, _) | (_, SafetyRoutingAction::ForceRoute)
            )
        {
            let force_model = self.force_model.as_deref().unwrap_or_default().trim();
            anyhow::ensure!(
                !force_model.is_empty(),
                "safety.force_model is required when a safety action is force_route"
            );
            anyhow::ensure!(
                config.find_model(force_model).is_some(),
                "safety.force_model {force_model} does not match a configured model id or alias"
            );
        }
        Ok(())
    }
}

impl StickyRoutingConfig {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.ttl_seconds > 0,
            "sticky_routing.ttl_seconds must be greater than zero"
        );
        anyhow::ensure!(
            self.lock_timeout_ms > 0,
            "sticky_routing.lock_timeout_ms must be greater than zero"
        );
        if self.backend == StickyRoutingBackend::File {
            let path = self.file_path.as_deref().unwrap_or_default().trim();
            anyhow::ensure!(
                !path.is_empty(),
                "sticky_routing.file_path is required when backend is file"
            );
        }
        Ok(())
    }
}

impl BudgetConfig {
    fn validate(&self) -> Result<()> {
        if let Some(value) = self.max_chat_requests {
            anyhow::ensure!(
                value > 0,
                "budget.max_chat_requests must be greater than zero"
            );
        }
        if let Some(value) = self.max_total_tokens {
            anyhow::ensure!(
                value > 0,
                "budget.max_total_tokens must be greater than zero"
            );
        }
        if let Some(value) = self.max_estimated_cost_micros {
            anyhow::ensure!(
                value > 0,
                "budget.max_estimated_cost_micros must be greater than zero"
            );
        }
        anyhow::ensure!(
            self.accounting.lock_timeout_ms > 0,
            "budget.accounting.lock_timeout_ms must be greater than zero"
        );
        if self.accounting.backend == BudgetAccountingBackend::File {
            let path = self
                .accounting
                .file_path
                .as_deref()
                .unwrap_or_default()
                .trim();
            anyhow::ensure!(
                !path.is_empty(),
                "budget.accounting.file_path is required when backend is file"
            );
        }
        Ok(())
    }
}

impl ScoringConfig {
    pub fn weights_for(&self, policy: &RouterPolicy) -> PolicyWeights {
        match policy {
            RouterPolicy::Balanced => self.balanced,
            RouterPolicy::LowestCostAcceptable => self.lowest_cost_acceptable,
            RouterPolicy::FastestHealthy => self.fastest_healthy,
            RouterPolicy::HighestQuality => self.highest_quality,
            RouterPolicy::LocalFirst => self.local_first,
            RouterPolicy::PrivacyFirst => self.privacy_first,
            RouterPolicy::MultimodalFirst => self.multimodal_first,
            RouterPolicy::Floor => self.floor,
            RouterPolicy::Nitro => self.nitro,
            RouterPolicy::Quality => self.quality,
            RouterPolicy::CostEfficient => self.cost_efficient,
            RouterPolicy::CapabilityHeavy => self.capability_heavy,
            RouterPolicy::DomainSkills => self.domain_skills,
        }
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.priority_weight.is_finite()
                && self.priority_weight >= 0.0
                && self.latency_weight.is_finite()
                && self.latency_weight >= 0.0
                && self.health_weight.is_finite()
                && self.health_weight >= 0.0,
            "scoring priority_weight, latency_weight, and health_weight must be non-negative finite numbers"
        );
        for (name, weights) in [
            ("balanced", self.balanced),
            ("lowest_cost_acceptable", self.lowest_cost_acceptable),
            ("fastest_healthy", self.fastest_healthy),
            ("highest_quality", self.highest_quality),
            ("local_first", self.local_first),
            ("privacy_first", self.privacy_first),
            ("multimodal_first", self.multimodal_first),
            ("floor", self.floor),
            ("nitro", self.nitro),
            ("quality", self.quality),
            ("cost_efficient", self.cost_efficient),
            ("capability_heavy", self.capability_heavy),
            ("domain_skills", self.domain_skills),
        ] {
            anyhow::ensure!(
                weights.capability_fit >= 0.0
                    && weights.domain_bonus >= 0.0
                    && weights.cost >= 0.0
                    && weights.overkill >= 0.0
                    && weights.raw_capability >= 0.0
                    && weights.latency >= 0.0
                    && weights.health >= 0.0
                    && weights.local_bonus >= 0.0
                    && weights.remote_penalty >= 0.0
                    && weights.multimodal_capability >= 0.0,
                "scoring.{name} weights must be non-negative"
            );
            anyhow::ensure!(
                weights.capability_fit.is_finite()
                    && weights.domain_bonus.is_finite()
                    && weights.cost.is_finite()
                    && weights.overkill.is_finite()
                    && weights.raw_capability.is_finite()
                    && weights.latency.is_finite()
                    && weights.health.is_finite()
                    && weights.local_bonus.is_finite()
                    && weights.remote_penalty.is_finite()
                    && weights.multimodal_capability.is_finite(),
                "scoring.{name} weights must be finite"
            );
        }
        for (name, priority) in self
            .model_priorities
            .iter()
            .chain(self.provider_priorities.iter())
        {
            anyhow::ensure!(
                priority.is_finite() && (-1.0..=1.0).contains(priority),
                "scoring priority hint {name} must be between -1.0 and 1.0"
            );
        }
        for (provider, latency_ms) in &self.provider_latency_p95_ms {
            anyhow::ensure!(
                *latency_ms > 0,
                "scoring.provider_latency_p95_ms.{provider} must be greater than zero"
            );
        }
        for (provider, penalty) in &self.provider_health_penalties {
            anyhow::ensure!(
                penalty.is_finite() && (0.0..=1.0).contains(penalty),
                "scoring.provider_health_penalties.{provider} must be between 0.0 and 1.0"
            );
        }
        self.learned.validate()?;
        Ok(())
    }

    fn validate_model_references(&self, config: &RouterConfig) -> Result<()> {
        for model in self.learned.model_biases.keys() {
            anyhow::ensure!(
                config.find_model(model).is_some(),
                "scoring.learned.model_biases references unknown model or alias {model}"
            );
        }
        Ok(())
    }
}

impl LearnedScoringConfig {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.weight.is_finite() && self.weight >= 0.0,
            "scoring.learned.weight must be a non-negative finite number"
        );
        anyhow::ensure!(self.bias.is_finite(), "scoring.learned.bias must be finite");
        for (feature, weight) in &self.feature_weights {
            anyhow::ensure!(
                !feature.trim().is_empty(),
                "scoring.learned.feature_weights cannot contain empty feature names"
            );
            anyhow::ensure!(
                weight.is_finite(),
                "scoring.learned.feature_weights.{feature} must be finite"
            );
        }
        for (model, bias) in &self.model_biases {
            anyhow::ensure!(
                !model.trim().is_empty(),
                "scoring.learned.model_biases cannot contain empty model names"
            );
            anyhow::ensure!(
                bias.is_finite(),
                "scoring.learned.model_biases.{model} must be finite"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuthConfig, BudgetConfig, ClassifierBackend, ClassifierConfig,
        ClassifierModelAdapterConfig, ProviderHealthSamplerConfig, RouterConfig, RuntimeConfig,
        SafetyRoutingAction, SafetyRoutingConfig, ScoringConfig, SemanticCacheBackend,
        SemanticCacheConfig, ShadowEvalConfig, ShadowEvalJudgeConfig, StickyRoutingBackend,
        StickyRoutingConfig, TelemetryConfig,
    };
    use crate::types::{ModelConfig, ModelEndpoint, ProviderConfig, ProviderKind, RouterPolicy};

    fn valid_config() -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:8080".to_string(),
            default_model: "model-a".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "provider-a".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url: "http://localhost:11434".to_string(),
                api_key_env: None,
                api_key: None,
                chat_path: "/v1/chat/completions".to_string(),
                responses_path: Some("/v1/responses".to_string()),
                embeddings_path: Some("/v1/embeddings".to_string()),
                images_path: Some("/v1/images/generations".to_string()),
                speech_path: Some("/v1/audio/speech".to_string()),
                audio_transcriptions_path: Some("/v1/audio/transcriptions".to_string()),
                audio_translations_path: Some("/v1/audio/translations".to_string()),
                health_path: Some("/health".to_string()),
                timeout_ms: 1_000,
                retries: 0,
                max_concurrency: Some(1),
                queue_timeout_ms: Some(1_000),
                extra_headers: Default::default(),
            }],
            models: vec![ModelConfig {
                id: "model-a".to_string(),
                provider: "provider-a".to_string(),
                aliases: vec!["alias-a".to_string()],
                capability: 0.5,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![],
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

    #[test]
    fn rejects_unknown_fields_for_every_config_object_type() {
        let base = serde_json::to_value(valid_config()).unwrap();
        let object_paths = [
            "",
            "classifier",
            "classifier.adapters",
            "classifier.adapters.llm_judge",
            "auth",
            "budget",
            "budget.accounting",
            "telemetry",
            "cache",
            "cache.semantic",
            "shadow_eval",
            "shadow_eval.judge",
            "safety",
            "sticky_routing",
            "runtime",
            "runtime.ingress",
            "runtime.provider_health_sampler",
            "scoring",
            "scoring.balanced",
            "scoring.learned",
            "providers.0",
            "models.0",
            "models.0.capabilities",
        ];

        for path in object_paths {
            let mut value = base.clone();
            object_at_path_mut(&mut value, path)
                .insert("unknown_router_field".to_string(), serde_json::json!(true));
            let yaml = serde_yaml::to_string(&value).unwrap();
            let error = serde_yaml::from_str::<RouterConfig>(&yaml)
                .expect_err(path)
                .to_string();
            assert!(
                error.contains("unknown field `unknown_router_field`"),
                "path={path}: {error}"
            );
        }
    }

    #[test]
    fn keeps_deliberately_open_config_maps_extensible() {
        let mut value = serde_json::to_value(valid_config()).unwrap();
        object_at_path_mut(&mut value, "providers.0.extra_headers")
            .insert("x-custom-header".to_string(), serde_json::json!("value"));
        object_at_path_mut(&mut value, "scoring.model_priorities")
            .insert("future-model".to_string(), serde_json::json!(0.5));
        object_at_path_mut(&mut value, "scoring.provider_latency_p95_ms")
            .insert("future-provider".to_string(), serde_json::json!(250));
        object_at_path_mut(&mut value, "scoring.learned.feature_weights")
            .insert("future_feature".to_string(), serde_json::json!(0.25));

        let yaml = serde_yaml::to_string(&value).unwrap();
        let parsed = serde_yaml::from_str::<RouterConfig>(&yaml).unwrap();
        assert_eq!(
            parsed.providers[0].extra_headers["x-custom-header"],
            "value"
        );
        assert_eq!(parsed.scoring.model_priorities["future-model"], 0.5);
        assert_eq!(
            parsed.scoring.provider_latency_p95_ms["future-provider"],
            250
        );
        assert_eq!(
            parsed.scoring.learned.feature_weights["future_feature"],
            0.25
        );
    }

    fn object_at_path_mut<'a>(
        value: &'a mut serde_json::Value,
        path: &str,
    ) -> &'a mut serde_json::Map<String, serde_json::Value> {
        let mut current = value;
        if !path.is_empty() {
            for segment in path.split('.') {
                current = if let Ok(index) = segment.parse::<usize>() {
                    current
                        .as_array_mut()
                        .and_then(|items| items.get_mut(index))
                        .unwrap_or_else(|| panic!("missing array path segment {segment} in {path}"))
                } else {
                    current
                        .as_object_mut()
                        .and_then(|object| object.get_mut(segment))
                        .unwrap_or_else(|| {
                            panic!("missing object path segment {segment} in {path}")
                        })
                };
            }
        }
        current
            .as_object_mut()
            .unwrap_or_else(|| panic!("path {path} is not an object"))
    }

    #[test]
    fn rejects_duplicate_provider_names() {
        let mut config = valid_config();
        config.providers.push(config.providers[0].clone());

        let error = config.validate().expect_err("duplicate provider rejected");
        assert!(error.to_string().contains("duplicate provider name"));
    }

    #[test]
    fn rejects_duplicate_model_alias_handles() {
        let mut config = valid_config();
        let mut second = config.models[0].clone();
        second.id = "model-b".to_string();
        second.aliases = vec!["alias-a".to_string()];
        config.models.push(second);

        let error = config.validate().expect_err("duplicate alias rejected");
        assert!(error.to_string().contains("collides"));
    }

    #[test]
    fn rejects_provider_without_http_base_url() {
        let mut config = valid_config();
        config.providers[0].base_url = "localhost:11434".to_string();

        let error = config
            .validate()
            .expect_err("invalid provider URL rejected");
        assert!(error.to_string().contains("base_url"));
    }

    #[test]
    fn rejects_empty_provider_queue_timeout() {
        let mut config = valid_config();
        config.providers[0].queue_timeout_ms = Some(0);

        let error = config.validate().expect_err("zero queue timeout rejected");
        assert!(error.to_string().contains("queue_timeout_ms"));
    }

    #[test]
    fn rejects_unauthenticated_non_loopback_bind() {
        let mut config = valid_config();
        config.bind = "0.0.0.0:8080".to_string();

        let error = config
            .validate()
            .expect_err("public bind without auth must be explicit");
        assert!(error.to_string().contains("auth is required"));
    }

    #[test]
    fn accepts_explicit_unauthenticated_network_override() {
        let mut config = valid_config();
        config.bind = "0.0.0.0:8080".to_string();
        config.auth.allow_unauthenticated_network = true;

        config
            .validate()
            .expect("trusted gateway override is explicit");
    }

    #[test]
    fn rejects_empty_auth_sources() {
        let mut config = valid_config();
        config.auth.bearer_tokens = vec![" ".to_string()];

        let error = config.validate().expect_err("empty bearer token rejected");
        assert!(error.to_string().contains("bearer_tokens"));

        config.auth.bearer_tokens.clear();
        config.auth.bearer_token_env = vec!["".to_string()];
        let error = config
            .validate()
            .expect_err("empty bearer token env name rejected");
        assert!(error.to_string().contains("bearer_token_env"));
    }

    #[test]
    fn accepts_scoring_hints_for_known_models_aliases_and_providers() {
        let mut config = valid_config();
        config
            .scoring
            .model_priorities
            .insert("alias-a".to_string(), 0.5);
        config
            .scoring
            .provider_priorities
            .insert("provider-a".to_string(), 0.2);
        config
            .scoring
            .provider_latency_p95_ms
            .insert("provider-a".to_string(), 250);
        config
            .scoring
            .provider_health_penalties
            .insert("provider-a".to_string(), 0.1);

        config.validate().expect("valid scoring hints accepted");
    }

    #[test]
    fn accepts_learned_scoring_for_known_model_alias() {
        let mut config = valid_config();
        config.scoring.learned.enabled = true;
        config.scoring.learned.weight = 0.4;
        config
            .scoring
            .learned
            .feature_weights
            .insert("domain.coding".to_string(), 0.3);
        config
            .scoring
            .learned
            .model_biases
            .insert("alias-a".to_string(), 0.2);

        config.validate().expect("valid learned scoring accepted");
    }

    #[test]
    fn rejects_learned_scoring_unknown_model_bias() {
        let mut config = valid_config();
        config.scoring.learned.enabled = true;
        config.scoring.learned.weight = 0.4;
        config
            .scoring
            .learned
            .model_biases
            .insert("missing-model".to_string(), 0.2);

        let error = config
            .validate()
            .expect_err("unknown learned model bias rejected");
        assert!(error.to_string().contains("scoring.learned.model_biases"));
    }

    #[test]
    fn legacy_llm_judge_model_selects_llm_judge_backend() {
        let mut config = valid_config();
        config.classifier.llm_judge_model = Some("alias-a".to_string());
        config
            .validate()
            .expect("legacy llm_judge_model is still accepted");

        assert_eq!(
            config.classifier.active_backend(),
            ClassifierBackend::LlmJudge
        );
        assert_eq!(
            config.classifier.active_adapter().model.as_deref(),
            Some("alias-a")
        );
    }

    #[test]
    fn rejects_route_llm_backend_without_model() {
        let mut config = valid_config();
        config.classifier.backend = ClassifierBackend::RouteLlm;

        let error = config
            .validate()
            .expect_err("route_llm backend requires model");
        assert!(
            error
                .to_string()
                .contains("classifier.adapters.route_llm.model")
        );
    }

    #[test]
    fn rejects_classifier_prompt_template_without_input_placeholder() {
        let mut config = valid_config();
        config.classifier.backend = ClassifierBackend::RouteLlm;
        config.classifier.adapters.route_llm = ClassifierModelAdapterConfig {
            model: Some("model-a".to_string()),
            timeout_ms: 1_000,
            prompt_template: Some("classify this prompt".to_string()),
        };

        let error = config
            .validate()
            .expect_err("classifier template requires input placeholder");
        assert!(error.to_string().contains("prompt_template"));
    }

    #[test]
    fn rejects_scoring_hint_for_unknown_model() {
        let mut config = valid_config();
        config
            .scoring
            .model_priorities
            .insert("missing-model".to_string(), 0.5);

        let error = config
            .validate()
            .expect_err("unknown model priority rejected");
        assert!(
            error
                .to_string()
                .contains("scoring.model_priorities references unknown model")
        );
    }

    #[test]
    fn rejects_scoring_hint_for_unknown_provider() {
        let mut config = valid_config();
        config
            .scoring
            .provider_latency_p95_ms
            .insert("missing-provider".to_string(), 250);

        let error = config
            .validate()
            .expect_err("unknown provider hint rejected");
        assert!(
            error
                .to_string()
                .contains("scoring provider hint references unknown provider")
        );
    }

    #[test]
    fn rejects_out_of_range_scoring_hint_values() {
        let mut config = valid_config();
        config
            .scoring
            .provider_health_penalties
            .insert("provider-a".to_string(), 1.5);

        let error = config
            .validate()
            .expect_err("invalid health penalty rejected");
        assert!(
            error
                .to_string()
                .contains("scoring.provider_health_penalties.provider-a")
        );
    }

    #[test]
    fn rejects_zero_provider_health_sampler_interval() {
        let mut config = valid_config();
        config.runtime.provider_health_sampler = ProviderHealthSamplerConfig {
            enabled: true,
            interval_ms: 0,
            initial_delay_ms: 0,
            ..Default::default()
        };

        let error = config
            .validate()
            .expect_err("zero sampler interval rejected");
        assert!(
            error
                .to_string()
                .contains("runtime.provider_health_sampler.interval_ms")
        );
    }

    #[test]
    fn rejects_zero_provider_health_freshness_and_circuit_controls() {
        type SamplerMutation = fn(&mut ProviderHealthSamplerConfig);
        let cases: [(&str, SamplerMutation); 5] = [
            ("check_timeout_ms", |sampler| sampler.check_timeout_ms = 0),
            ("max_concurrent_checks", |sampler| {
                sampler.max_concurrent_checks = 0
            }),
            ("observation_ttl_ms", |sampler| {
                sampler.observation_ttl_ms = 0
            }),
            ("circuit_failure_threshold", |sampler| {
                sampler.circuit_failure_threshold = 0
            }),
            ("circuit_open_ms", |sampler| sampler.circuit_open_ms = 0),
        ];
        for (field, mutate) in cases {
            let mut config = valid_config();
            mutate(&mut config.runtime.provider_health_sampler);
            let error = config
                .validate()
                .expect_err("zero health policy control rejected");
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_zero_ingress_resource_limits() {
        type ConfigMutation = fn(&mut RouterConfig);
        let cases: [(&str, ConfigMutation); 6] = [
            ("max_json_body_bytes", |config| {
                config.runtime.ingress.max_json_body_bytes = 0
            }),
            ("max_multipart_body_bytes", |config| {
                config.runtime.ingress.max_multipart_body_bytes = 0
            }),
            ("body_idle_timeout_ms", |config| {
                config.runtime.ingress.body_idle_timeout_ms = 0
            }),
            ("max_in_flight_requests", |config| {
                config.runtime.ingress.max_in_flight_requests = Some(0)
            }),
            ("admission_queue_timeout_ms", |config| {
                config.runtime.ingress.admission_queue_timeout_ms = 0
            }),
            ("per_credential_requests_per_minute", |config| {
                config.runtime.ingress.per_credential_requests_per_minute = Some(0)
            }),
        ];
        for (field, mutate) in cases {
            let mut config = valid_config();
            mutate(&mut config);
            let error = config.validate().expect_err("zero ingress limit rejected");
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn rejects_out_of_range_semantic_cache_threshold() {
        let mut config = valid_config();
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 1.5,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("invalid semantic cache threshold rejected");
        assert!(
            error
                .to_string()
                .contains("cache.semantic.similarity_threshold")
        );
    }

    #[test]
    fn accepts_provider_backed_semantic_cache_embedding_model() {
        let mut config = valid_config();
        config.models[0].capabilities.supported_endpoints =
            Some(vec![ModelEndpoint::Chat, ModelEndpoint::Embeddings]);
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "alias-a".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        config
            .validate()
            .expect("configured embedding model alias is accepted");
    }

    #[test]
    fn rejects_semantic_cache_model_without_embedding_endpoint_support() {
        let mut config = valid_config();
        config.models[0].capabilities.supported_endpoints =
            Some(vec![crate::types::ModelEndpoint::Chat]);
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "model-a".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("model without embedding endpoint support rejected");
        assert!(error.to_string().contains("does not support embeddings"));
    }

    #[test]
    fn rejects_unknown_semantic_cache_embedding_model() {
        let mut config = valid_config();
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "missing-embedding-model".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("unknown semantic embedding model rejected");
        assert!(error.to_string().contains("cache.semantic.embedding_model"));
    }

    #[test]
    fn rejects_semantic_cache_embedding_provider_without_embeddings_path() {
        let mut config = valid_config();
        config.providers[0].embeddings_path = None;
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "model-a".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("provider without embeddings path rejected");
        assert!(error.to_string().contains("does not support embeddings"));
    }

    #[test]
    fn rejects_file_semantic_cache_without_file_path() {
        let mut config = valid_config();
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::File,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("file semantic cache requires path");
        assert!(error.to_string().contains("cache.semantic.file_path"));
    }

    #[test]
    fn rejects_zero_semantic_cache_lock_timeout() {
        let mut config = valid_config();
        config.cache.semantic = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.80,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::File,
            file_path: Some("semantic-cache.json".to_string()),
            lock_timeout_ms: 0,
        };

        let error = config
            .validate()
            .expect_err("zero semantic cache lock timeout rejected");
        assert!(error.to_string().contains("cache.semantic.lock_timeout_ms"));
    }

    #[test]
    fn rejects_enabled_shadow_eval_without_output_path() {
        let mut config = valid_config();
        config.shadow_eval = ShadowEvalConfig {
            enabled: true,
            sample_rate: 1.0,
            output_path: None,
            include_bodies: false,
            max_body_chars: 128,
            judge: Default::default(),
        };

        let error = config
            .validate()
            .expect_err("enabled shadow eval requires output path");
        assert!(error.to_string().contains("shadow_eval.output_path"));
    }

    #[test]
    fn accepts_shadow_eval_judge_model_alias() {
        let mut config = valid_config();
        config.shadow_eval = ShadowEvalConfig {
            enabled: true,
            sample_rate: 1.0,
            output_path: Some("shadow.jsonl".to_string()),
            include_bodies: false,
            max_body_chars: 128,
            judge: ShadowEvalJudgeConfig {
                enabled: true,
                model: Some("alias-a".to_string()),
                timeout_ms: 250,
                prompt_template: Some(
                    "{input}\n{selected_model}\n{selected_answer}\n{shadow_model}\n{shadow_answer}"
                        .to_string(),
                ),
            },
        };

        config.validate().expect("judge alias is accepted");
    }

    #[test]
    fn rejects_unknown_shadow_eval_judge_model() {
        let mut config = valid_config();
        config.shadow_eval.judge.model = Some("missing-judge".to_string());

        let error = config
            .validate()
            .expect_err("unknown shadow eval judge model rejected");
        assert!(error.to_string().contains("shadow_eval.judge.model"));
    }

    #[test]
    fn rejects_zero_shadow_eval_judge_timeout() {
        let mut config = valid_config();
        config.shadow_eval.judge.timeout_ms = 0;

        let error = config
            .validate()
            .expect_err("zero shadow eval judge timeout rejected");
        assert!(error.to_string().contains("shadow_eval.judge.timeout_ms"));
    }

    #[test]
    fn rejects_shadow_eval_judge_template_missing_required_slots() {
        let mut config = valid_config();
        config.shadow_eval.judge.prompt_template = Some("{input}\n{selected_answer}".to_string());

        let error = config
            .validate()
            .expect_err("shadow eval judge template slots are required");
        assert!(
            error
                .to_string()
                .contains("shadow_eval.judge.prompt_template")
        );
    }

    #[test]
    fn rejects_zero_sticky_routing_ttl() {
        let mut config = valid_config();
        config.sticky_routing = StickyRoutingConfig {
            enabled: true,
            ttl_seconds: 0,
            prefer_model: true,
            backend: StickyRoutingBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("zero sticky routing ttl rejected");
        assert!(error.to_string().contains("sticky_routing.ttl_seconds"));
    }

    #[test]
    fn rejects_file_sticky_routing_without_file_path() {
        let mut config = valid_config();
        config.sticky_routing = StickyRoutingConfig {
            enabled: true,
            ttl_seconds: 60,
            prefer_model: true,
            backend: StickyRoutingBackend::File,
            file_path: None,
            lock_timeout_ms: 1_000,
        };

        let error = config
            .validate()
            .expect_err("file sticky routing requires path");
        assert!(error.to_string().contains("sticky_routing.file_path"));
    }

    #[test]
    fn rejects_zero_sticky_routing_lock_timeout() {
        let mut config = valid_config();
        config.sticky_routing = StickyRoutingConfig {
            enabled: true,
            ttl_seconds: 60,
            prefer_model: true,
            backend: StickyRoutingBackend::File,
            file_path: Some("sticky.json".to_string()),
            lock_timeout_ms: 0,
        };

        let error = config
            .validate()
            .expect_err("zero sticky routing lock timeout rejected");
        assert!(error.to_string().contains("sticky_routing.lock_timeout_ms"));
    }

    #[test]
    fn rejects_safety_force_route_without_force_model() {
        let mut config = valid_config();
        config.safety = SafetyRoutingConfig {
            enabled: true,
            unsafe_action: SafetyRoutingAction::ForceRoute,
            sensitive_action: SafetyRoutingAction::Allow,
            force_model: None,
            redaction_replacement: "[redacted]".to_string(),
        };

        let error = config.validate().expect_err("safety force model required");
        assert!(error.to_string().contains("safety.force_model"));
    }

    #[test]
    fn rejects_model_endpoint_without_a_compatible_provider_path() {
        let mut config = valid_config();
        config.providers[0].responses_path = None;
        config.models[0].capabilities.supported_endpoints =
            Some(vec![ModelEndpoint::Chat, ModelEndpoint::Responses]);

        let error = config
            .validate()
            .expect_err("model endpoint requires provider path");

        assert!(
            error
                .to_string()
                .contains("model model-a declares endpoint responses")
        );
        assert!(error.to_string().contains("provider-a"));
    }

    #[test]
    fn rejects_duplicate_model_endpoint_declarations() {
        let mut config = valid_config();
        config.models[0].capabilities.supported_endpoints =
            Some(vec![ModelEndpoint::Chat, ModelEndpoint::Chat]);

        let error = config
            .validate()
            .expect_err("duplicate endpoint declaration rejected");

        assert!(error.to_string().contains("duplicate chat"));
    }

    #[test]
    fn native_provider_rejects_non_chat_endpoint_paths() {
        for kind in [ProviderKind::OllamaNative, ProviderKind::LlamaCppNative] {
            let mut config = valid_config();
            config.providers[0].kind = kind;
            config.providers[0].responses_path = Some("/v1/responses".to_string());

            let error = config
                .validate()
                .expect_err("native adapter endpoint path rejected");

            assert!(
                error
                    .to_string()
                    .contains("cannot configure unsupported endpoint responses")
            );
        }
    }

    #[test]
    fn native_provider_rejects_model_capabilities_its_adapter_cannot_preserve() {
        let cases = [
            (ProviderKind::OllamaNative, "vision"),
            (ProviderKind::OllamaNative, "tools"),
            (ProviderKind::OllamaNative, "audio"),
            (ProviderKind::LlamaCppNative, "vision"),
            (ProviderKind::LlamaCppNative, "tools"),
            (ProviderKind::LlamaCppNative, "audio"),
            (ProviderKind::LlamaCppNative, "json"),
        ];
        for (kind, capability) in cases {
            let mut config = valid_config();
            config.providers[0].kind = kind;
            match capability {
                "vision" => config.models[0].capabilities.supports_vision = true,
                "tools" => config.models[0].capabilities.supports_tools = true,
                "audio" => config.models[0].capabilities.supports_audio = true,
                "json" => config.models[0].capabilities.supports_json = true,
                _ => unreachable!(),
            }

            let error = config
                .validate()
                .expect_err("unsupported native adapter capability rejected");
            let message = error.to_string();
            assert!(message.contains("provider adapter"), "{message}");
            assert!(message.contains(capability), "{message}");
            assert!(
                message.contains(config.providers[0].kind.chat_adapter_contract().name),
                "{message}"
            );
        }
    }

    #[test]
    fn ollama_native_provider_accepts_json_capability() {
        let mut config = valid_config();
        config.providers[0].kind = ProviderKind::OllamaNative;
        config.providers[0].responses_path = None;
        config.providers[0].embeddings_path = None;
        config.providers[0].images_path = None;
        config.providers[0].speech_path = None;
        config.providers[0].audio_transcriptions_path = None;
        config.providers[0].audio_translations_path = None;
        config.models[0].capabilities.supports_json = true;

        config
            .validate()
            .expect("Ollama native maps OpenAI JSON response formats");
    }

    #[test]
    fn imports_conformance_artifact_and_rejects_unverified_endpoints() {
        let directory = std::env::temp_dir().join(format!(
            "autohand-router-conformance-import-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let config_path = directory.join("router.yaml");
        let artifact_path = directory.join("matrix.json");
        std::fs::write(
            &config_path,
            r#"
bind: 127.0.0.1:8080
default_model: model-a
runtime:
  provider_conformance_artifact: matrix.json
providers:
  - name: provider-a
    base_url: http://127.0.0.1:9999
    responses_path: /v1/responses
models:
  - id: model-a
    provider: provider-a
    capabilities:
      supported_endpoints: [chat, responses]
"#,
        )
        .unwrap();
        let artifact = |responses_pass| {
            serde_json::json!({
                "schema_version": 1,
                "reports": [{
                    "provider": "provider-a",
                    "model": "model-a",
                    "chat": {
                        "configured": true,
                        "status": 200,
                        "openai_chat_shape": true,
                        "response_model_matches": true,
                        "assistant_content_present": true
                    },
                    "endpoints": [{
                        "endpoint": "responses",
                        "configured": true,
                        "pass": responses_pass
                    }]
                }]
            })
        };
        std::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&artifact(true)).unwrap(),
        )
        .unwrap();

        RouterConfig::from_path(&config_path).expect("verified endpoint artifact is imported");

        std::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&artifact(false)).unwrap(),
        )
        .unwrap();
        let error = RouterConfig::from_path(&config_path)
            .expect_err("unverified endpoint must fail closed");
        assert!(
            error
                .to_string()
                .contains("does not verify endpoint responses")
        );
        let _ = std::fs::remove_dir_all(directory);
    }
}
