use crate::{
    classifier::PromptClassifier,
    config::RouterConfig,
    tokens::estimate_tokens,
    types::{
        Classifications, DifficultyLabel, DomainLabel, ModelConfig, MultimodelRequest,
        MultimodelResponse, RouteCandidate, RouterPolicy,
    },
};
use std::sync::Arc;

#[derive(Clone)]
pub struct RoutingEngine<C> {
    config: Arc<RouterConfig>,
    classifier: Arc<C>,
}

#[derive(Debug, Clone, Copy)]
struct TokenBudget {
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
}

impl<C> RoutingEngine<C>
where
    C: PromptClassifier,
{
    pub fn new(config: RouterConfig, classifier: C) -> Self {
        Self {
            config: Arc::new(config),
            classifier: Arc::new(classifier),
        }
    }

    pub fn config(&self) -> Arc<RouterConfig> {
        self.config.clone()
    }

    pub fn classifier(&self) -> Arc<C> {
        self.classifier.clone()
    }

    pub async fn classify(&self, input: &str) -> Classifications {
        self.classifier.classify(input).await
    }

    pub async fn route(&self, request: MultimodelRequest) -> MultimodelResponse {
        let classifications = self.classify(&request.input).await;
        let token_budget = TokenBudget {
            estimated_input_tokens: estimate_tokens(&request.input),
            requested_output_tokens: request.max_output_tokens.unwrap_or(1024),
        };
        let default_model = request
            .default_model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let allowed_default = self.allowed_fallback_model(&request, default_model);
        let candidates = self.candidates(&request, default_model);

        if classifications.difficulty.label == DifficultyLabel::NeedsInfo {
            return match allowed_default {
                Some(model) => self.response_for_model(
                    &model.id,
                    classifications,
                    request.policy,
                    "prompt needs more information; using allowed default",
                    true,
                    token_budget,
                ),
                None => self.response_for_unknown(
                    classifications,
                    request.policy,
                    "prompt needs more information but default_model does not satisfy allowed filters",
                    true,
                    token_budget,
                ),
            };
        }

        let scored = scored_candidates(
            &candidates,
            &classifications,
            &request.policy,
            &self.config,
            token_budget.estimated_input_tokens,
            token_budget.requested_output_tokens,
        );

        let Some(best) = scored.iter().max_by(|left, right| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return match allowed_default {
                Some(model) => self.response_for_model(
                    &model.id,
                    classifications,
                    request.policy,
                    "no candidate matched filters; using allowed default",
                    true,
                    token_budget,
                ),
                None => self.response_for_unknown(
                    classifications,
                    request.policy,
                    "no candidate matched filters and default_model does not satisfy allowed filters",
                    true,
                    token_budget,
                ),
            };
        };

        let reason = format!(
            "selected by {:?} policy from {} candidate(s)",
            request.policy,
            candidates.len()
        );
        self.response_for_model(
            &best.model,
            classifications,
            request.policy,
            &reason,
            false,
            token_budget,
        )
        .with_candidates(scored)
    }

    fn candidates<'a>(
        &'a self,
        request: &MultimodelRequest,
        default_model: &'a str,
    ) -> Vec<&'a ModelConfig> {
        let mut candidates = self
            .config
            .models
            .iter()
            .filter(|model| model_allowed_by_request(model, request))
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            if let Some(default) = self.allowed_fallback_model(request, default_model) {
                candidates.push(default);
            }
        }
        candidates
    }

    fn allowed_fallback_model<'a>(
        &'a self,
        request: &MultimodelRequest,
        default_model: &str,
    ) -> Option<&'a ModelConfig> {
        self.config
            .find_model(default_model)
            .filter(|model| model_allowed_by_request(model, request))
    }

    fn response_for_model(
        &self,
        model_id: &str,
        classifications: Classifications,
        policy: RouterPolicy,
        reason: &str,
        fallback: bool,
        token_budget: TokenBudget,
    ) -> MultimodelResponse {
        let model = self
            .config
            .find_model(model_id)
            .or_else(|| self.config.find_model(&self.config.default_model));
        let (model_id, provider) = model
            .map(|model| (model.id.clone(), model.provider.clone()))
            .unwrap_or_else(|| (model_id.to_string(), "unknown".to_string()));
        let ambiguity = classifications
            .ambiguity
            .meets_threshold
            .then_some(classifications.ambiguity.label);
        let ambiguity_confidence = classifications
            .ambiguity
            .meets_threshold
            .then_some(classifications.ambiguity.confidence);
        let domain = classifications
            .domain
            .meets_threshold
            .then_some(classifications.domain.label);
        let domain_confidence = classifications
            .domain
            .meets_threshold
            .then_some(classifications.domain.confidence);

        MultimodelResponse {
            model: model_id,
            provider,
            difficulty: classifications.difficulty.label,
            confidence: classifications.difficulty.confidence,
            ambiguity,
            ambiguity_confidence,
            domain,
            domain_confidence,
            policy,
            reason: reason.to_string(),
            fallback,
            estimated_input_tokens: token_budget.estimated_input_tokens,
            requested_output_tokens: token_budget.requested_output_tokens,
            candidates: Vec::new(),
        }
    }

    fn response_for_unknown(
        &self,
        classifications: Classifications,
        policy: RouterPolicy,
        reason: &str,
        fallback: bool,
        token_budget: TokenBudget,
    ) -> MultimodelResponse {
        let ambiguity = classifications
            .ambiguity
            .meets_threshold
            .then_some(classifications.ambiguity.label);
        let ambiguity_confidence = classifications
            .ambiguity
            .meets_threshold
            .then_some(classifications.ambiguity.confidence);
        let domain = classifications
            .domain
            .meets_threshold
            .then_some(classifications.domain.label);
        let domain_confidence = classifications
            .domain
            .meets_threshold
            .then_some(classifications.domain.confidence);

        MultimodelResponse {
            model: String::new(),
            provider: String::new(),
            difficulty: classifications.difficulty.label,
            confidence: classifications.difficulty.confidence,
            ambiguity,
            ambiguity_confidence,
            domain,
            domain_confidence,
            policy,
            reason: reason.to_string(),
            fallback,
            estimated_input_tokens: token_budget.estimated_input_tokens,
            requested_output_tokens: token_budget.requested_output_tokens,
            candidates: Vec::new(),
        }
    }
}

fn model_allowed_by_request(model: &ModelConfig, request: &MultimodelRequest) -> bool {
    let has_model_filter = !request.allowed_models.is_empty();
    let has_provider_filter = !request.allowed_providers.is_empty();
    if !has_model_filter && !has_provider_filter {
        return true;
    }
    let model_allowed = request
        .allowed_models
        .iter()
        .any(|allowed| model.id == *allowed || model.aliases.iter().any(|alias| alias == allowed));
    let provider_allowed = request.allowed_providers.contains(&model.provider);
    model_allowed || provider_allowed
}

impl MultimodelResponse {
    fn with_candidates(mut self, candidates: Vec<RouteCandidate>) -> Self {
        self.candidates = candidates;
        self
    }
}

fn scored_candidates(
    candidates: &[&ModelConfig],
    classifications: &Classifications,
    policy: &RouterPolicy,
    config: &RouterConfig,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
) -> Vec<RouteCandidate> {
    let context_required = estimated_input_tokens.saturating_add(requested_output_tokens);
    let mut scored = candidates
        .iter()
        .map(|model| {
            let provider = config
                .providers
                .iter()
                .find(|provider| provider.name == model.provider);
            let context_eligible = model
                .context_window
                .is_none_or(|window| window >= context_required);
            let mut score = score_model(model, classifications, policy, config);
            let routing_priority = routing_priority(model, config);
            let latency_penalty =
                provider.map_or(0.0, |provider| latency_penalty(provider, config));
            let health_penalty = provider.map_or(0.0, |provider| health_penalty(provider, config));
            score += routing_priority * config.scoring.priority_weight;
            score -= latency_penalty * config.scoring.latency_weight;
            score -= health_penalty * config.scoring.health_weight;
            if !context_eligible {
                score -= 10.0;
            }
            RouteCandidate {
                model: model.id.clone(),
                provider: model.provider.clone(),
                score,
                capability: model.capability,
                estimated_cost: normalized_cost(model),
                domain_match: model.domains.contains(&classifications.domain.label),
                routing_priority,
                latency_penalty,
                health_penalty,
                context_window: model.context_window,
                context_required,
                context_eligible,
            }
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

pub fn score_model(
    model: &ModelConfig,
    classifications: &Classifications,
    policy: &RouterPolicy,
    config: &RouterConfig,
) -> f32 {
    let required = required_capability(&classifications.difficulty.label);
    let capability_fit = 1.0 - (required - model.capability).max(0.0);
    let overkill_penalty = (model.capability - required).max(0.0) * 0.12;
    let cost = normalized_cost(model);
    let domain_bonus = domain_bonus(model, classifications.domain.label.clone());
    let weights = config.scoring.weights_for(policy);

    capability_fit * weights.capability_fit
        + domain_bonus * weights.domain_bonus
        + model.capability * weights.raw_capability
        - cost * weights.cost
        - overkill_penalty * weights.overkill
}

pub fn required_capability(difficulty: &DifficultyLabel) -> f32 {
    match difficulty {
        DifficultyLabel::Easy => 0.25,
        DifficultyLabel::Medium => 0.58,
        DifficultyLabel::Hard => 0.84,
        DifficultyLabel::NeedsInfo => 0.55,
    }
}

pub fn normalized_cost(model: &ModelConfig) -> f32 {
    ((model.cost_per_million_input + model.cost_per_million_output) / 60.0).clamp(0.0, 1.0)
}

pub fn domain_bonus(model: &ModelConfig, domain: DomainLabel) -> f32 {
    if model.domains.contains(&domain) {
        1.0
    } else if domain == DomainLabel::General {
        0.4
    } else {
        0.0
    }
}

fn routing_priority(model: &ModelConfig, config: &RouterConfig) -> f32 {
    let model_priority = config
        .scoring
        .model_priorities
        .get(&model.id)
        .copied()
        .or_else(|| {
            model
                .aliases
                .iter()
                .find_map(|alias| config.scoring.model_priorities.get(alias).copied())
        })
        .unwrap_or_default();
    let provider_priority = config
        .scoring
        .provider_priorities
        .get(&model.provider)
        .copied()
        .unwrap_or_default();
    (model_priority + provider_priority).clamp(-1.0, 1.0)
}

fn latency_penalty(provider: &crate::types::ProviderConfig, config: &RouterConfig) -> f32 {
    config
        .scoring
        .provider_latency_p95_ms
        .get(&provider.name)
        .map(|latency_ms| (*latency_ms as f32 / 5_000.0).clamp(0.0, 1.0))
        .unwrap_or_default()
}

fn health_penalty(provider: &crate::types::ProviderConfig, config: &RouterConfig) -> f32 {
    config
        .scoring
        .provider_health_penalties
        .get(&provider.name)
        .copied()
        .unwrap_or_default()
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        classifier::HeuristicClassifier,
        types::{ProviderConfig, ProviderKind},
    };
    use std::collections::HashMap;

    fn engine() -> RoutingEngine<HeuristicClassifier> {
        let config = RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "balanced".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "local".to_string(),
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
                health_path: None,
                timeout_ms: 120_000,
                retries: 1,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: HashMap::new(),
            }],
            models: vec![
                ModelConfig {
                    id: "cheap".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.30,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding],
                    context_window: None,
                    local: true,
                },
                ModelConfig {
                    id: "balanced".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.65,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::Coding, DomainLabel::Design],
                    context_window: None,
                    local: true,
                },
                ModelConfig {
                    id: "strong".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.95,
                    cost_per_million_input: 20.0,
                    cost_per_million_output: 40.0,
                    domains: vec![DomainLabel::Design, DomainLabel::Coding],
                    context_window: None,
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring: Default::default(),
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
        };
        RoutingEngine::new(config, HeuristicClassifier::default())
    }

    fn context_engine() -> RoutingEngine<HeuristicClassifier> {
        let config = RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "large-context".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "local".to_string(),
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
                health_path: None,
                timeout_ms: 120_000,
                retries: 1,
                max_concurrency: None,
                queue_timeout_ms: None,
                extra_headers: HashMap::new(),
            }],
            models: vec![
                ModelConfig {
                    id: "small-context".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.90,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding],
                    context_window: Some(128),
                    local: true,
                },
                ModelConfig {
                    id: "large-context".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.70,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::Coding],
                    context_window: Some(4096),
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring: Default::default(),
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
        };
        RoutingEngine::new(config, HeuristicClassifier::default())
    }

    fn scoring_hint_engine() -> RoutingEngine<HeuristicClassifier> {
        let mut scoring = crate::config::ScoringConfig::default();
        scoring
            .model_priorities
            .insert("preferred-balanced".to_string(), 1.0);
        scoring
            .provider_health_penalties
            .insert("degraded".to_string(), 1.0);

        let config = RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "cheap".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![
                ProviderConfig {
                    name: "healthy".to_string(),
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
                    health_path: None,
                    timeout_ms: 120_000,
                    retries: 1,
                    max_concurrency: None,
                    queue_timeout_ms: None,
                    extra_headers: HashMap::new(),
                },
                ProviderConfig {
                    name: "degraded".to_string(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: "http://localhost:11435".to_string(),
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
                    timeout_ms: 120_000,
                    retries: 1,
                    max_concurrency: None,
                    queue_timeout_ms: None,
                    extra_headers: HashMap::new(),
                },
            ],
            models: vec![
                ModelConfig {
                    id: "cheap".to_string(),
                    provider: "healthy".to_string(),
                    aliases: vec![],
                    capability: 0.30,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding],
                    context_window: None,
                    local: true,
                },
                ModelConfig {
                    id: "preferred-balanced".to_string(),
                    provider: "healthy".to_string(),
                    aliases: vec![],
                    capability: 0.60,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::Coding],
                    context_window: None,
                    local: true,
                },
                ModelConfig {
                    id: "degraded-strong".to_string(),
                    provider: "degraded".to_string(),
                    aliases: vec![],
                    capability: 0.95,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding],
                    context_window: None,
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring,
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
        };
        RoutingEngine::new(config, HeuristicClassifier::default())
    }

    #[tokio::test]
    async fn routes_easy_work_to_cheaper_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                policy: RouterPolicy::CostEfficient,
                default_model: None,
                max_output_tokens: None,
            })
            .await;
        assert_eq!(route.model, "cheap");
        assert!(!route.fallback);
    }

    #[tokio::test]
    async fn routes_hard_architecture_work_to_strong_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Design a production multi tenant event sourcing architecture with concurrency, migration, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                policy: RouterPolicy::CapabilityHeavy,
                default_model: None,
                max_output_tokens: None,
            })
            .await;
        assert_eq!(route.model, "strong");
    }

    #[tokio::test]
    async fn avoids_models_that_cannot_fit_context() {
        let route = context_engine()
            .route(MultimodelRequest {
                input: "Add error handling to this async Rust API client and include tests"
                    .to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                policy: RouterPolicy::CapabilityHeavy,
                default_model: None,
                max_output_tokens: Some(512),
            })
            .await;
        assert_eq!(route.model, "large-context");
        assert!(route.estimated_input_tokens > 0);
        assert_eq!(route.requested_output_tokens, 512);
        assert!(
            route
                .candidates
                .iter()
                .any(|candidate| candidate.model == "small-context" && !candidate.context_eligible)
        );
    }

    #[tokio::test]
    async fn model_priority_hint_can_promote_a_preferred_candidate() {
        let route = scoring_hint_engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec!["cheap".to_string(), "preferred-balanced".to_string()],
                allowed_providers: vec![],
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "preferred-balanced");
        assert!(
            route
                .candidates
                .iter()
                .any(|candidate| candidate.model == "preferred-balanced"
                    && candidate.routing_priority > 0.0)
        );
    }

    #[tokio::test]
    async fn provider_health_penalty_demotes_degraded_provider() {
        let route = scoring_hint_engine()
            .route(MultimodelRequest {
                input: "Design a production Rust router with concurrency, tests, and retry logic"
                    .to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                policy: RouterPolicy::CapabilityHeavy,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_ne!(route.provider, "degraded");
        assert!(
            route
                .candidates
                .iter()
                .any(|candidate| candidate.provider == "degraded"
                    && candidate.health_penalty > 0.0)
        );
    }

    #[tokio::test]
    async fn does_not_return_default_model_outside_allowed_filters() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Design a production migration plan with tests and security review"
                    .to_string(),
                allowed_models: vec!["does-not-exist".to_string()],
                allowed_providers: vec![],
                policy: RouterPolicy::Balanced,
                default_model: Some("balanced".to_string()),
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert_eq!(route.provider, "");
        assert!(route.fallback);
        assert!(
            route
                .reason
                .contains("default_model does not satisfy allowed filters")
        );
        assert!(route.candidates.is_empty());
    }

    #[tokio::test]
    async fn needs_info_does_not_bypass_allowed_filters_with_default_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "hi".to_string(),
                allowed_models: vec!["cheap".to_string()],
                allowed_providers: vec![],
                policy: RouterPolicy::Balanced,
                default_model: Some("balanced".to_string()),
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert_eq!(route.provider, "");
        assert_eq!(route.difficulty, DifficultyLabel::NeedsInfo);
        assert!(route.fallback);
    }

    #[tokio::test]
    async fn needs_info_can_use_default_model_when_allowed() {
        let route = engine()
            .route(MultimodelRequest {
                input: "hi".to_string(),
                allowed_models: vec!["balanced".to_string()],
                allowed_providers: vec![],
                policy: RouterPolicy::Balanced,
                default_model: Some("balanced".to_string()),
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "balanced");
        assert_eq!(route.provider, "local");
        assert_eq!(route.difficulty, DifficultyLabel::NeedsInfo);
        assert!(route.fallback);
    }
}
