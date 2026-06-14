use crate::types::{ModelConfig, ProviderConfig, RouterPolicy};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub bearer_tokens: Vec<String>,
    #[serde(default)]
    pub bearer_token_env: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
pub struct TelemetryConfig {
    #[serde(default)]
    pub decision_log_path: Option<String>,
    #[serde(default)]
    pub include_inputs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "default_graceful_shutdown_timeout_ms")]
    pub graceful_shutdown_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    #[serde(default = "default_balanced_weights")]
    pub balanced: PolicyWeights,
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PolicyWeights {
    pub capability_fit: f32,
    pub domain_bonus: f32,
    pub cost: f32,
    pub overkill: f32,
    #[serde(default)]
    pub raw_capability: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            balanced: default_balanced_weights(),
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
    }
}

fn default_cost_efficient_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.42,
        domain_bonus: 0.16,
        cost: 0.42,
        overkill: 1.4,
        raw_capability: 0.0,
    }
}

fn default_capability_heavy_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.20,
        domain_bonus: 0.08,
        cost: 0.05,
        overkill: 0.0,
        raw_capability: 0.72,
    }
}

fn default_domain_skills_weights() -> PolicyWeights {
    PolicyWeights {
        capability_fit: 0.38,
        domain_bonus: 0.48,
        cost: 0.10,
        overkill: 0.0,
        raw_capability: 0.10,
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
            confidence_threshold: default_threshold(),
            easy_threshold: default_easy_threshold(),
            hard_threshold: default_hard_threshold(),
            llm_judge_model: None,
            llm_judge_timeout_ms: default_llm_judge_timeout_ms(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            graceful_shutdown_timeout_ms: default_graceful_shutdown_timeout_ms(),
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
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
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
        if let Some(judge) = &self.classifier.llm_judge_model {
            anyhow::ensure!(
                self.find_model(judge).is_some(),
                "classifier.llm_judge_model {judge} does not match a configured model id or alias"
            );
        }
        anyhow::ensure!(
            self.classifier.easy_threshold < self.classifier.hard_threshold,
            "classifier.easy_threshold must be lower than classifier.hard_threshold"
        );
        for env_name in &self.auth.bearer_token_env {
            anyhow::ensure!(
                !env_name.trim().is_empty(),
                "auth.bearer_token_env entries cannot be empty"
            );
        }
        self.scoring.validate()?;
        self.budget.validate()?;
        self.telemetry.validate()?;
        self.runtime.validate()?;
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

    pub fn auth_tokens(&self) -> Vec<String> {
        self.auth
            .bearer_tokens
            .iter()
            .cloned()
            .chain(
                self.auth
                    .bearer_token_env
                    .iter()
                    .filter_map(|name| std::env::var(name).ok()),
            )
            .filter(|token| !token.is_empty())
            .collect()
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
            ("cost_efficient", self.cost_efficient),
            ("capability_heavy", self.capability_heavy),
            ("domain_skills", self.domain_skills),
        ] {
            anyhow::ensure!(
                weights.capability_fit >= 0.0
                    && weights.domain_bonus >= 0.0
                    && weights.cost >= 0.0
                    && weights.overkill >= 0.0
                    && weights.raw_capability >= 0.0,
                "scoring.{name} weights must be non-negative"
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuthConfig, BudgetConfig, ClassifierConfig, RouterConfig, RuntimeConfig, ScoringConfig,
        TelemetryConfig,
    };
    use crate::types::{ModelConfig, ProviderConfig, ProviderKind, RouterPolicy};

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
        }
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
}
