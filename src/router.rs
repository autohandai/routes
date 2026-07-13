use crate::{
    classifier::PromptClassifier,
    config::RouterConfig,
    health::ProviderHealthStore,
    tokens::estimate_tokens,
    types::{
        Classifications, DifficultyLabel, DomainLabel, LatencySensitivityLabel, ModelCapability,
        ModelConfig, MultimodelRequest, MultimodelResponse, ReasoningDepthLabel, RouteCandidate,
        RouteCandidateRejection, RouteDecisionTrace, RoutePolicyWeights, RouteScoreComponents,
        RouterPolicy,
    },
};
use std::sync::Arc;

#[derive(Clone)]
pub struct RoutingEngine<C> {
    config: Arc<RouterConfig>,
    classifier: Arc<C>,
    provider_health: ProviderHealthStore,
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
            provider_health: ProviderHealthStore::default(),
        }
    }

    pub fn config(&self) -> Arc<RouterConfig> {
        self.config.clone()
    }

    pub fn classifier(&self) -> Arc<C> {
        self.classifier.clone()
    }

    pub fn provider_health(&self) -> ProviderHealthStore {
        self.provider_health.clone()
    }

    pub async fn classify(&self, input: &str) -> Classifications {
        self.classifier.classify(input).await
    }

    pub async fn route(&self, request: MultimodelRequest) -> MultimodelResponse {
        self.route_internal(request, None).await
    }

    pub async fn route_with_estimated_input_tokens(
        &self,
        request: MultimodelRequest,
        estimated_input_tokens: u32,
    ) -> MultimodelResponse {
        self.route_internal(request, Some(estimated_input_tokens))
            .await
    }

    async fn route_internal(
        &self,
        request: MultimodelRequest,
        estimated_input_tokens: Option<u32>,
    ) -> MultimodelResponse {
        let classifications = self.classify(&request.input).await;
        let token_budget = TokenBudget {
            estimated_input_tokens: estimated_input_tokens
                .unwrap_or_else(|| estimate_tokens(&request.input)),
            requested_output_tokens: request.max_output_tokens.unwrap_or(1024),
        };
        let default_model = request
            .default_model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let allowed_default = self.allowed_fallback_model(&request, default_model);
        let candidates = self.candidates(&request, default_model);

        if classifications.difficulty.label == DifficultyLabel::NeedsInfo {
            let eligible_default = allowed_default.filter(|model| {
                context_eligible(
                    model,
                    token_budget
                        .estimated_input_tokens
                        .saturating_add(token_budget.requested_output_tokens),
                ) && capability_eligible(model, &request.required_capabilities, &self.config)
            });
            return match eligible_default {
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
                    "prompt needs more information but default_model does not satisfy allowed filters/capabilities/context",
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
            &self.provider_health,
            token_budget.estimated_input_tokens,
            token_budget.requested_output_tokens,
            &request.required_capabilities,
        );

        let Some(best) = scored.iter().max_by(|left, right| {
            candidate_selectable(left)
                .cmp(&candidate_selectable(right))
                .then_with(|| {
                    left.score
                        .partial_cmp(&right.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
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

        if !candidate_selectable(best) {
            let decision_trace = decision_trace(
                classifications.clone(),
                request.policy,
                &self.config,
                token_budget,
                &request.required_capabilities,
                &scored,
                None,
            );
            return match allowed_default.filter(|model| {
                context_eligible(
                    model,
                    token_budget.estimated_input_tokens
                        .saturating_add(token_budget.requested_output_tokens),
                ) && capability_eligible(model, &request.required_capabilities, &self.config)
            }) {
                Some(model) => self.response_for_model(
                    &model.id,
                    classifications,
                    request.policy,
                    "no eligible candidate matched required capabilities/context; using allowed default",
                    true,
                    token_budget,
                )
                .with_decision_trace(decision_trace),
                None => self.response_for_unknown(
                    classifications,
                    request.policy,
                    "no eligible candidate matched required capabilities/context",
                    true,
                    token_budget,
                )
                .with_decision_trace(decision_trace),
            };
        }

        let decision_trace = decision_trace(
            classifications.clone(),
            request.policy,
            &self.config,
            token_budget,
            &request.required_capabilities,
            &scored,
            Some(best),
        );
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
        .with_decision_trace(decision_trace)
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
        let modality = classifications
            .modality
            .meets_threshold
            .then_some(classifications.modality.label);
        let modality_confidence = classifications
            .modality
            .meets_threshold
            .then_some(classifications.modality.confidence);
        let safety = classifications
            .safety
            .meets_threshold
            .then_some(classifications.safety.label);
        let safety_confidence = classifications
            .safety
            .meets_threshold
            .then_some(classifications.safety.confidence);
        let cacheability = classifications
            .cacheability
            .meets_threshold
            .then_some(classifications.cacheability.label);
        let cacheability_confidence = classifications
            .cacheability
            .meets_threshold
            .then_some(classifications.cacheability.confidence);
        let latency_sensitivity = classifications
            .latency_sensitivity
            .meets_threshold
            .then_some(classifications.latency_sensitivity.label);
        let latency_sensitivity_confidence = classifications
            .latency_sensitivity
            .meets_threshold
            .then_some(classifications.latency_sensitivity.confidence);
        let reasoning_depth = classifications
            .reasoning_depth
            .meets_threshold
            .then_some(classifications.reasoning_depth.label);
        let reasoning_depth_confidence = classifications
            .reasoning_depth
            .meets_threshold
            .then_some(classifications.reasoning_depth.confidence);

        MultimodelResponse {
            model: model_id,
            provider,
            difficulty: classifications.difficulty.label,
            confidence: classifications.difficulty.confidence,
            ambiguity,
            ambiguity_confidence,
            domain,
            domain_confidence,
            modality,
            modality_confidence,
            safety,
            safety_confidence,
            cacheability,
            cacheability_confidence,
            latency_sensitivity,
            latency_sensitivity_confidence,
            reasoning_depth,
            reasoning_depth_confidence,
            policy,
            reason: reason.to_string(),
            fallback,
            estimated_input_tokens: token_budget.estimated_input_tokens,
            requested_output_tokens: token_budget.requested_output_tokens,
            decision_trace: None,
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
        let modality = classifications
            .modality
            .meets_threshold
            .then_some(classifications.modality.label);
        let modality_confidence = classifications
            .modality
            .meets_threshold
            .then_some(classifications.modality.confidence);
        let safety = classifications
            .safety
            .meets_threshold
            .then_some(classifications.safety.label);
        let safety_confidence = classifications
            .safety
            .meets_threshold
            .then_some(classifications.safety.confidence);
        let cacheability = classifications
            .cacheability
            .meets_threshold
            .then_some(classifications.cacheability.label);
        let cacheability_confidence = classifications
            .cacheability
            .meets_threshold
            .then_some(classifications.cacheability.confidence);
        let latency_sensitivity = classifications
            .latency_sensitivity
            .meets_threshold
            .then_some(classifications.latency_sensitivity.label);
        let latency_sensitivity_confidence = classifications
            .latency_sensitivity
            .meets_threshold
            .then_some(classifications.latency_sensitivity.confidence);
        let reasoning_depth = classifications
            .reasoning_depth
            .meets_threshold
            .then_some(classifications.reasoning_depth.label);
        let reasoning_depth_confidence = classifications
            .reasoning_depth
            .meets_threshold
            .then_some(classifications.reasoning_depth.confidence);

        MultimodelResponse {
            model: String::new(),
            provider: String::new(),
            difficulty: classifications.difficulty.label,
            confidence: classifications.difficulty.confidence,
            ambiguity,
            ambiguity_confidence,
            domain,
            domain_confidence,
            modality,
            modality_confidence,
            safety,
            safety_confidence,
            cacheability,
            cacheability_confidence,
            latency_sensitivity,
            latency_sensitivity_confidence,
            reasoning_depth,
            reasoning_depth_confidence,
            policy,
            reason: reason.to_string(),
            fallback,
            estimated_input_tokens: token_budget.estimated_input_tokens,
            requested_output_tokens: token_budget.requested_output_tokens,
            decision_trace: None,
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
    (!has_model_filter || model_allowed) && (!has_provider_filter || provider_allowed)
}

impl MultimodelResponse {
    fn with_candidates(mut self, candidates: Vec<RouteCandidate>) -> Self {
        self.candidates = candidates;
        self
    }

    fn with_decision_trace(mut self, decision_trace: RouteDecisionTrace) -> Self {
        self.decision_trace = Some(decision_trace);
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn scored_candidates(
    candidates: &[&ModelConfig],
    classifications: &Classifications,
    policy: &RouterPolicy,
    config: &RouterConfig,
    provider_health: &ProviderHealthStore,
    estimated_input_tokens: u32,
    requested_output_tokens: u32,
    required_capabilities: &[ModelCapability],
) -> Vec<RouteCandidate> {
    let context_required = estimated_input_tokens.saturating_add(requested_output_tokens);
    let mut scored = candidates
        .iter()
        .map(|model| {
            let provider = config
                .providers
                .iter()
                .find(|provider| provider.name == model.provider);
            let context_eligible = context_eligible(model, context_required);
            let missing_capabilities = missing_capabilities(model, required_capabilities, config);
            let capability_eligible = missing_capabilities.is_empty();
            let weights = config.scoring.weights_for(policy);
            let mut score_components =
                score_model_components(model, classifications, policy, config);
            let mut score = score_components.final_score;
            let routing_priority = routing_priority(model, config);
            let latency_penalty = provider.map_or(0.0, |provider| {
                latency_penalty(provider, config, provider_health)
            });
            let health_penalty = provider.map_or(0.0, |provider| {
                health_penalty(provider, config, provider_health)
            });
            let latency_weight = weights.latency
                * latency_sensitivity_multiplier(&classifications.latency_sensitivity.label);
            score_components.routing_priority_boost =
                routing_priority * config.scoring.priority_weight;
            score_components.learned_score_boost =
                learned_score_boost(model, classifications, config);
            score_components.latency_penalty = latency_penalty * latency_weight;
            score_components.health_penalty = health_penalty * weights.health;
            score += score_components.routing_priority_boost;
            score += score_components.learned_score_boost;
            score -= score_components.latency_penalty;
            score -= score_components.health_penalty;
            if !capability_eligible {
                score_components.capability_exclusion_penalty = 10.0;
                score -= 10.0;
            }
            if !context_eligible {
                score_components.context_exclusion_penalty = 10.0;
                score -= 10.0;
            }
            score_components.final_score = score;
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
                capability_eligible,
                missing_capabilities,
                context_window: model.context_window,
                context_required,
                context_eligible,
                score_components,
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

fn candidate_selectable(candidate: &RouteCandidate) -> bool {
    candidate.capability_eligible && candidate.context_eligible
}

fn decision_trace(
    classifications: Classifications,
    policy: RouterPolicy,
    config: &RouterConfig,
    token_budget: TokenBudget,
    required_capabilities: &[ModelCapability],
    candidates: &[RouteCandidate],
    selected: Option<&RouteCandidate>,
) -> RouteDecisionTrace {
    RouteDecisionTrace {
        classifier: classifications,
        policy,
        policy_weights: config.scoring.weights_for(&policy).into(),
        required_capabilities: required_capabilities.to_vec(),
        context_required: token_budget
            .estimated_input_tokens
            .saturating_add(token_budget.requested_output_tokens),
        selected_model: selected.map(|candidate| candidate.model.clone()),
        selected_provider: selected.map(|candidate| candidate.provider.clone()),
        selected_score: selected.map(|candidate| candidate.score),
        rejected_candidates: candidates.iter().filter_map(candidate_rejection).collect(),
    }
}

fn candidate_rejection(candidate: &RouteCandidate) -> Option<RouteCandidateRejection> {
    let mut reasons = Vec::new();
    if !candidate.capability_eligible {
        reasons.push(format!(
            "missing capabilities: {}",
            candidate
                .missing_capabilities
                .iter()
                .map(|capability| format!("{capability:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !candidate.context_eligible {
        reasons.push(format!(
            "context required {} exceeds window {}",
            candidate.context_required,
            candidate
                .context_window
                .map(|window| window.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ));
    }
    (!reasons.is_empty()).then(|| RouteCandidateRejection {
        model: candidate.model.clone(),
        provider: candidate.provider.clone(),
        reasons,
    })
}

impl From<crate::config::PolicyWeights> for RoutePolicyWeights {
    fn from(weights: crate::config::PolicyWeights) -> Self {
        Self {
            capability_fit: weights.capability_fit,
            domain_bonus: weights.domain_bonus,
            cost: weights.cost,
            overkill: weights.overkill,
            raw_capability: weights.raw_capability,
            latency: weights.latency,
            health: weights.health,
            local_bonus: weights.local_bonus,
            remote_penalty: weights.remote_penalty,
            multimodal_capability: weights.multimodal_capability,
        }
    }
}

fn context_eligible(model: &ModelConfig, context_required: u32) -> bool {
    model
        .context_window
        .is_none_or(|window| window >= context_required)
}

fn capability_eligible(
    model: &ModelConfig,
    required_capabilities: &[ModelCapability],
    config: &RouterConfig,
) -> bool {
    required_capabilities
        .iter()
        .all(|capability| effective_capability_support(model, capability, config))
}

fn missing_capabilities(
    model: &ModelConfig,
    required_capabilities: &[ModelCapability],
    config: &RouterConfig,
) -> Vec<ModelCapability> {
    required_capabilities
        .iter()
        .filter(|capability| !effective_capability_support(model, capability, config))
        .cloned()
        .collect()
}

fn effective_capability_support(
    model: &ModelConfig,
    capability: &ModelCapability,
    config: &RouterConfig,
) -> bool {
    model.capabilities.supports(capability)
        && config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .is_some_and(|provider| provider.kind.adapter_supports_capability(capability))
}

pub fn score_model(
    model: &ModelConfig,
    classifications: &Classifications,
    policy: &RouterPolicy,
    config: &RouterConfig,
) -> f32 {
    score_model_components(model, classifications, policy, config).final_score
}

fn score_model_components(
    model: &ModelConfig,
    classifications: &Classifications,
    policy: &RouterPolicy,
    config: &RouterConfig,
) -> RouteScoreComponents {
    let required = required_capability_for_classifications(classifications);
    let capability_fit = 1.0 - (required - model.capability).max(0.0);
    let overkill_penalty = (model.capability - required).max(0.0) * 0.12;
    let cost = normalized_cost(model);
    let domain_bonus = domain_bonus(model, classifications.domain.label.clone());
    let weights = config.scoring.weights_for(policy);

    let capability_fit_score = capability_fit * weights.capability_fit;
    let domain_bonus_score = domain_bonus * weights.domain_bonus;
    let raw_capability_score = model.capability * weights.raw_capability;
    let cost_penalty = cost * weights.cost;
    let overkill_penalty = overkill_penalty * weights.overkill;
    let local_score_boost = if model.local {
        weights.local_bonus
    } else {
        0.0
    };
    let remote_penalty = if model.local {
        0.0
    } else {
        weights.remote_penalty
    };
    let multimodal_score_boost = model_multimodal_capability(model) * weights.multimodal_capability;
    let final_score = capability_fit_score
        + domain_bonus_score
        + raw_capability_score
        + local_score_boost
        + multimodal_score_boost
        - cost_penalty
        - overkill_penalty
        - remote_penalty;

    RouteScoreComponents {
        capability_fit,
        capability_fit_score,
        domain_bonus,
        domain_bonus_score,
        raw_capability_score,
        cost_penalty,
        overkill_penalty,
        local_score_boost,
        remote_penalty,
        multimodal_score_boost,
        routing_priority_boost: 0.0,
        learned_score_boost: 0.0,
        latency_penalty: 0.0,
        health_penalty: 0.0,
        capability_exclusion_penalty: 0.0,
        context_exclusion_penalty: 0.0,
        final_score,
    }
}

fn model_multimodal_capability(model: &ModelConfig) -> f32 {
    let capabilities = [
        model.capabilities.supports_vision,
        model.capabilities.supports_audio,
        model.capabilities.supports_tools,
        model.capabilities.supports_web_apps,
    ];
    let supported = capabilities
        .into_iter()
        .filter(|supported| *supported)
        .count() as f32;
    supported / capabilities.len() as f32
}

pub fn required_capability(difficulty: &DifficultyLabel) -> f32 {
    match difficulty {
        DifficultyLabel::Easy => 0.25,
        DifficultyLabel::Medium => 0.58,
        DifficultyLabel::Hard => 0.84,
        DifficultyLabel::NeedsInfo => 0.55,
    }
}

fn required_capability_for_classifications(classifications: &Classifications) -> f32 {
    let base = required_capability(&classifications.difficulty.label);
    match classifications.reasoning_depth.label {
        ReasoningDepthLabel::Shallow => (base - 0.08).max(0.20),
        ReasoningDepthLabel::Moderate => base,
        ReasoningDepthLabel::Deep => (base + 0.10).min(0.95),
    }
}

fn latency_sensitivity_multiplier(label: &LatencySensitivityLabel) -> f32 {
    match label {
        LatencySensitivityLabel::Low => 0.55,
        LatencySensitivityLabel::Medium => 1.0,
        LatencySensitivityLabel::High => 1.65,
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

fn learned_score_boost(
    model: &ModelConfig,
    classifications: &Classifications,
    config: &RouterConfig,
) -> f32 {
    let learned = &config.scoring.learned;
    if !learned.enabled || learned.weight <= 0.0 {
        return 0.0;
    }
    let mut raw = learned.bias;
    raw += learned_model_bias(model, config);
    raw += learned_feature_weight("bias", config);
    raw += learned_feature_weight(&format!("model.{}", model.id), config);
    for alias in &model.aliases {
        raw += learned_feature_weight(&format!("model.{alias}"), config);
    }
    raw += learned_feature_weight(&format!("provider.{}", model.provider), config);
    raw += learned_feature_weight(
        &format!(
            "difficulty.{}",
            label_key(&classifications.difficulty.label)
        ),
        config,
    ) * classifications.difficulty.confidence;
    raw += learned_feature_weight(
        &format!("domain.{}", label_key(&classifications.domain.label)),
        config,
    ) * classifications.domain.confidence;
    raw += learned_feature_weight(
        &format!("modality.{}", label_key(&classifications.modality.label)),
        config,
    ) * classifications.modality.confidence;
    raw += learned_feature_weight(
        &format!("safety.{}", label_key(&classifications.safety.label)),
        config,
    ) * classifications.safety.confidence;
    raw += learned_feature_weight(
        &format!(
            "cacheability.{}",
            label_key(&classifications.cacheability.label)
        ),
        config,
    ) * classifications.cacheability.confidence;
    raw += learned_feature_weight(
        &format!(
            "latency_sensitivity.{}",
            label_key(&classifications.latency_sensitivity.label)
        ),
        config,
    ) * classifications.latency_sensitivity.confidence;
    raw += learned_feature_weight(
        &format!(
            "reasoning_depth.{}",
            label_key(&classifications.reasoning_depth.label)
        ),
        config,
    ) * classifications.reasoning_depth.confidence;
    raw += learned_feature_weight("domain_match", config)
        * domain_bonus(model, classifications.domain.label.clone());
    raw += learned_feature_weight("capability", config) * model.capability;
    raw += learned_feature_weight("cost", config) * normalized_cost(model);
    raw += learned_feature_weight("local", config) * if model.local { 1.0 } else { 0.0 };
    raw += learned_feature_weight("supports_vision", config)
        * if model.capabilities.supports_vision {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_audio", config)
        * if model.capabilities.supports_audio {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_tools", config)
        * if model.capabilities.supports_tools {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_json", config)
        * if model.capabilities.supports_json {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_code", config)
        * if model.capabilities.supports_code {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_web_apps", config)
        * if model.capabilities.supports_web_apps {
            1.0
        } else {
            0.0
        };
    raw += learned_feature_weight("supports_long_context", config)
        * if model.capabilities.supports_long_context {
            1.0
        } else {
            0.0
        };
    learned.weight * raw.clamp(-1.0, 1.0)
}

fn learned_feature_weight(feature: &str, config: &RouterConfig) -> f32 {
    config
        .scoring
        .learned
        .feature_weights
        .get(feature)
        .copied()
        .unwrap_or_default()
}

fn learned_model_bias(model: &ModelConfig, config: &RouterConfig) -> f32 {
    config
        .scoring
        .learned
        .model_biases
        .get(&model.id)
        .copied()
        .or_else(|| {
            model
                .aliases
                .iter()
                .find_map(|alias| config.scoring.learned.model_biases.get(alias).copied())
        })
        .unwrap_or_default()
}

fn label_key<T: std::fmt::Debug>(label: &T) -> String {
    let raw = format!("{label:?}");
    let mut key = String::new();
    for (index, character) in raw.chars().enumerate() {
        if character.is_uppercase() && index > 0 {
            key.push('_');
        }
        key.extend(character.to_lowercase());
    }
    key
}

fn latency_penalty(
    provider: &crate::types::ProviderConfig,
    config: &RouterConfig,
    provider_health: &ProviderHealthStore,
) -> f32 {
    let static_penalty = config
        .scoring
        .provider_latency_p95_ms
        .get(&provider.name)
        .map(|latency_ms| (*latency_ms as f32 / 5_000.0).clamp(0.0, 1.0))
        .unwrap_or_default();
    let sampled_penalty = provider_health
        .observation(&provider.name)
        .and_then(|observation| observation.latency_ms)
        .map(|latency_ms| (latency_ms as f32 / 5_000.0).clamp(0.0, 1.0))
        .unwrap_or_default();
    static_penalty.max(sampled_penalty)
}

fn health_penalty(
    provider: &crate::types::ProviderConfig,
    config: &RouterConfig,
    provider_health: &ProviderHealthStore,
) -> f32 {
    let static_penalty = config
        .scoring
        .provider_health_penalties
        .get(&provider.name)
        .copied()
        .unwrap_or_default()
        .clamp(0.0, 1.0);
    let sampled_penalty = provider_health
        .observation(&provider.name)
        .map(|observation| observation.health_penalty)
        .unwrap_or_default()
        .clamp(0.0, 1.0);
    static_penalty.max(sampled_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        classifier::HeuristicClassifier,
        provider::{ProviderHealth, ProviderHealthStatus},
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
                    capabilities: Default::default(),
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
                    capabilities: Default::default(),
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
                    capabilities: Default::default(),
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring: Default::default(),
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
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
                    capabilities: Default::default(),
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
                    capabilities: Default::default(),
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring: Default::default(),
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        };
        RoutingEngine::new(config, HeuristicClassifier::default())
    }

    fn capability_engine() -> RoutingEngine<HeuristicClassifier> {
        let config = RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "text-only".to_string(),
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
                    id: "text-only".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.70,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::General],
                    context_window: Some(4096),
                    capabilities: Default::default(),
                    local: true,
                },
                ModelConfig {
                    id: "vision-tools".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.62,
                    cost_per_million_input: 1.0,
                    cost_per_million_output: 1.0,
                    domains: vec![DomainLabel::General],
                    context_window: Some(4096),
                    capabilities: crate::types::ModelCapabilities {
                        supports_vision: true,
                        supports_tools: true,
                        supports_json: true,
                        ..Default::default()
                    },
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring: Default::default(),
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
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
                    capabilities: Default::default(),
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
                    capabilities: Default::default(),
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
                    capabilities: Default::default(),
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring,
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
        };
        RoutingEngine::new(config, HeuristicClassifier::default())
    }

    fn latency_policy_engine() -> RoutingEngine<HeuristicClassifier> {
        let mut scoring = crate::config::ScoringConfig::default();
        scoring
            .provider_latency_p95_ms
            .insert("fast".to_string(), 100);
        scoring
            .provider_latency_p95_ms
            .insert("slow".to_string(), 4_000);

        let config = RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "fast-balanced".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![
                ProviderConfig {
                    name: "fast".to_string(),
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
                    name: "slow".to_string(),
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
                    id: "fast-balanced".to_string(),
                    provider: "fast".to_string(),
                    aliases: vec![],
                    capability: 0.70,
                    cost_per_million_input: 10.0,
                    cost_per_million_output: 10.0,
                    domains: vec![DomainLabel::Coding, DomainLabel::Design],
                    context_window: None,
                    capabilities: Default::default(),
                    local: true,
                },
                ModelConfig {
                    id: "slow-strong".to_string(),
                    provider: "slow".to_string(),
                    aliases: vec![],
                    capability: 0.95,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding, DomainLabel::Design],
                    context_window: None,
                    capabilities: Default::default(),
                    local: true,
                },
            ],
            classifier: Default::default(),
            auth: Default::default(),
            scoring,
            budget: Default::default(),
            telemetry: Default::default(),
            runtime: Default::default(),
            cache: Default::default(),
            shadow_eval: Default::default(),
            safety: Default::default(),
            sticky_routing: Default::default(),
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
                required_capabilities: Vec::new(),
                policy: RouterPolicy::CostEfficient,
                default_model: None,
                max_output_tokens: None,
            })
            .await;
        assert_eq!(route.model, "cheap");
        assert!(!route.fallback);
    }

    #[tokio::test]
    async fn floor_policy_selects_cheapest_acceptable_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Floor,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "cheap");
        assert_eq!(route.policy, RouterPolicy::Floor);
    }

    #[tokio::test]
    async fn lowest_cost_acceptable_policy_selects_cheapest_acceptable_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::LowestCostAcceptable,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "cheap");
        assert_eq!(route.policy, RouterPolicy::LowestCostAcceptable);
    }

    #[tokio::test]
    async fn routes_hard_architecture_work_to_strong_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Design a production multi tenant event sourcing architecture with concurrency, migration, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::CapabilityHeavy,
                default_model: None,
                max_output_tokens: None,
            })
            .await;
        assert_eq!(route.model, "strong");
    }

    #[tokio::test]
    async fn quality_policy_selects_strongest_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Design a production multi tenant event sourcing architecture with concurrency, migration, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Quality,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "strong");
        assert_eq!(route.policy, RouterPolicy::Quality);
    }

    #[tokio::test]
    async fn highest_quality_policy_selects_strongest_model() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Design a production multi tenant event sourcing architecture with concurrency, migration, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::HighestQuality,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "strong");
        assert_eq!(route.policy, RouterPolicy::HighestQuality);
    }

    #[tokio::test]
    async fn nitro_policy_prefers_fast_healthy_provider() {
        let route = latency_policy_engine()
            .route(MultimodelRequest {
                input: "ASAP fast instant realtime: design a production architecture with concurrency, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Nitro,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "fast-balanced");
        assert_eq!(route.provider, "fast");
        assert_eq!(route.policy, RouterPolicy::Nitro);
        assert!(
            route.candidates.iter().any(
                |candidate| candidate.model == "slow-strong" && candidate.latency_penalty > 0.7
            )
        );
    }

    #[tokio::test]
    async fn fastest_healthy_policy_prefers_fast_healthy_provider() {
        let route = latency_policy_engine()
            .route(MultimodelRequest {
                input: "ASAP fast instant realtime: design a production architecture with concurrency, benchmark, and security tradeoffs".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::FastestHealthy,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "fast-balanced");
        assert_eq!(route.provider, "fast");
        assert_eq!(route.policy, RouterPolicy::FastestHealthy);
    }

    #[tokio::test]
    async fn local_first_policy_prefers_local_candidate() {
        let base_engine = engine();
        let mut config = (*base_engine.config()).clone();
        config.providers.push(ProviderConfig {
            name: "remote".to_string(),
            kind: ProviderKind::OpenAiCompatible,
            base_url: "https://remote.example.test".to_string(),
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
        });
        config.models.push(ModelConfig {
            id: "remote-strong".to_string(),
            provider: "remote".to_string(),
            aliases: vec![],
            capability: 0.95,
            cost_per_million_input: 1.0,
            cost_per_million_output: 1.0,
            domains: vec![DomainLabel::Coding, DomainLabel::Design],
            context_window: None,
            capabilities: Default::default(),
            local: false,
        });
        let engine = RoutingEngine::new(config, HeuristicClassifier::default());

        let route = engine
            .route(MultimodelRequest {
                input: "Refactor this Rust API client and add tests".to_string(),
                allowed_models: vec!["balanced".to_string(), "remote-strong".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::LocalFirst,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "balanced");
        assert_eq!(route.policy, RouterPolicy::LocalFirst);
        let selected = route
            .candidates
            .iter()
            .find(|candidate| candidate.model == "balanced")
            .expect("selected candidate present");
        assert!(selected.score_components.local_score_boost > 0.0);
    }

    #[tokio::test]
    async fn privacy_first_policy_penalizes_remote_candidates() {
        let base_engine = engine();
        let mut config = (*base_engine.config()).clone();
        config.providers.push(ProviderConfig {
            name: "remote".to_string(),
            kind: ProviderKind::OpenAiCompatible,
            base_url: "https://remote.example.test".to_string(),
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
        });
        config.models.push(ModelConfig {
            id: "remote-strong".to_string(),
            provider: "remote".to_string(),
            aliases: vec![],
            capability: 0.95,
            cost_per_million_input: 1.0,
            cost_per_million_output: 1.0,
            domains: vec![DomainLabel::Coding, DomainLabel::Design],
            context_window: None,
            capabilities: Default::default(),
            local: false,
        });
        let engine = RoutingEngine::new(config, HeuristicClassifier::default());

        let route = engine
            .route(MultimodelRequest {
                input: "Refactor this Rust API client and add tests without using cloud models"
                    .to_string(),
                allowed_models: vec!["balanced".to_string(), "remote-strong".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::PrivacyFirst,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "balanced");
        assert_eq!(route.policy, RouterPolicy::PrivacyFirst);
        let remote = route
            .candidates
            .iter()
            .find(|candidate| candidate.model == "remote-strong")
            .expect("remote candidate present");
        assert!(remote.score_components.remote_penalty > 0.0);
    }

    #[tokio::test]
    async fn multimodal_first_policy_prefers_multimodal_candidate() {
        let base_engine = engine();
        let mut config = (*base_engine.config()).clone();
        let multimodal_capabilities = crate::types::ModelCapabilities {
            supports_vision: true,
            supports_audio: true,
            supports_tools: true,
            supports_web_apps: true,
            ..Default::default()
        };
        config.models.push(ModelConfig {
            id: "multimodal-balanced".to_string(),
            provider: "local".to_string(),
            aliases: vec![],
            capability: 0.65,
            cost_per_million_input: 1.0,
            cost_per_million_output: 1.0,
            domains: vec![DomainLabel::Coding, DomainLabel::Design],
            context_window: None,
            capabilities: multimodal_capabilities,
            local: true,
        });
        let engine = RoutingEngine::new(config, HeuristicClassifier::default());

        let route = engine
            .route(MultimodelRequest {
                input: "Build a small multimodal web app from a screenshot and audio note"
                    .to_string(),
                allowed_models: vec!["balanced".to_string(), "multimodal-balanced".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::MultimodalFirst,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "multimodal-balanced");
        assert_eq!(route.policy, RouterPolicy::MultimodalFirst);
        let selected = route
            .candidates
            .iter()
            .find(|candidate| candidate.model == "multimodal-balanced")
            .expect("multimodal candidate present");
        assert!(selected.score_components.multimodal_score_boost > 0.0);
    }

    #[tokio::test]
    async fn route_response_includes_advanced_classifier_heads() {
        let route = engine()
            .route(MultimodelRequest {
                input: "ASAP fast instant realtime: analyze this screenshot and design a production architecture with root cause debugging, tradeoffs, and call the JSON schema tool with no API keys".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(
            route.modality,
            Some(crate::types::ModalityLabel::Multimodal)
        );
        assert_eq!(route.safety, Some(crate::types::SafetyLabel::Sensitive));
        assert_eq!(
            route.latency_sensitivity,
            Some(LatencySensitivityLabel::High)
        );
        assert_eq!(route.reasoning_depth, Some(ReasoningDepthLabel::Deep));
    }

    #[tokio::test]
    async fn avoids_models_that_cannot_fit_context() {
        let route = context_engine()
            .route(MultimodelRequest {
                input: "Add error handling to this async Rust API client and include tests"
                    .to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
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
        let trace = route.decision_trace.as_ref().expect("decision trace");
        assert_eq!(trace.selected_model.as_deref(), Some("large-context"));
        assert_eq!(trace.context_required, route.estimated_input_tokens + 512);
        assert!(trace.rejected_candidates.iter().any(|rejection| {
            rejection.model == "small-context"
                && rejection
                    .reasons
                    .iter()
                    .any(|reason| reason.contains("context required"))
        }));
        for candidate in &route.candidates {
            assert!((candidate.score_components.final_score - candidate.score).abs() < 0.0001);
        }
    }

    #[tokio::test]
    async fn caller_supplied_context_estimate_controls_context_eligibility() {
        let route = context_engine()
            .route_with_estimated_input_tokens(
                MultimodelRequest {
                    input: "short classifier text".to_string(),
                    allowed_models: vec![],
                    allowed_providers: vec![],
                    required_capabilities: Vec::new(),
                    policy: RouterPolicy::Balanced,
                    default_model: None,
                    max_output_tokens: Some(0),
                },
                4097,
            )
            .await;

        assert_eq!(route.model, "");
        assert!(route.fallback);
        assert_eq!(route.decision_trace.unwrap().context_required, 4097);
    }

    #[tokio::test]
    async fn required_capabilities_route_to_capable_model() {
        let route = capability_engine()
            .route(MultimodelRequest {
                input: "Describe the attached screenshot and return JSON".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Vision, ModelCapability::Json],
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "vision-tools");
        assert!(route.candidates.iter().any(|candidate| {
            candidate.model == "text-only"
                && !candidate.capability_eligible
                && candidate
                    .missing_capabilities
                    .contains(&ModelCapability::Vision)
        }));
    }

    #[tokio::test]
    async fn adapter_capability_contract_overrides_incompatible_model_metadata() {
        let base = capability_engine();
        let mut config = (*base.config()).clone();
        config.providers[0].kind = ProviderKind::OllamaNative;
        let route = RoutingEngine::new(config, HeuristicClassifier::default())
            .route(MultimodelRequest {
                input: "Use the lookup tool".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Tools],
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert!(route.fallback);
        let trace = route.decision_trace.expect("decision trace");
        assert!(trace.rejected_candidates.iter().any(|rejection| {
            rejection.model == "vision-tools"
                && rejection
                    .reasons
                    .iter()
                    .any(|reason| reason.contains("Tools"))
        }));
    }

    #[tokio::test]
    async fn route_fails_closed_when_no_model_has_required_capabilities() {
        let route = capability_engine()
            .route(MultimodelRequest {
                input: "Transcribe this audio".to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Audio],
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert!(route.fallback);
        assert!(
            route
                .reason
                .contains("no eligible candidate matched required capabilities")
        );
    }

    #[tokio::test]
    async fn model_priority_hint_can_promote_a_preferred_candidate() {
        let route = scoring_hint_engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec!["cheap".to_string(), "preferred-balanced".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
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
    async fn learned_scoring_can_promote_a_trained_candidate() {
        let base_engine = engine();
        let mut config = (*base_engine.config()).clone();
        config.scoring.learned.enabled = true;
        config.scoring.learned.weight = 0.6;
        config
            .scoring
            .learned
            .model_biases
            .insert("strong".to_string(), 1.0);
        let engine = RoutingEngine::new(config, HeuristicClassifier::default());

        let route = engine
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec!["cheap".to_string(), "strong".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Balanced,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "strong");
        let learned_candidate = route
            .candidates
            .iter()
            .find(|candidate| candidate.model == "strong")
            .expect("strong candidate is present");
        assert!(learned_candidate.score_components.learned_score_boost > 0.0);
        assert!(
            (learned_candidate.score_components.final_score - learned_candidate.score).abs()
                < 0.0001
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
                required_capabilities: Vec::new(),
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
    async fn sampled_provider_health_penalty_demotes_degraded_provider() {
        let engine = scoring_hint_engine();
        engine.provider_health().record(
            ProviderHealth {
                provider: "degraded".to_string(),
                adapter: "mock".to_string(),
                status: ProviderHealthStatus::Error,
                status_code: Some(503),
                error: Some("unavailable".to_string()),
            },
            250,
        );

        let route = engine
            .route(MultimodelRequest {
                input: "Design a production Rust router with concurrency, tests, and retry logic"
                    .to_string(),
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
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
                    && candidate.health_penalty > 0.0
                    && candidate.latency_penalty > 0.0)
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
                required_capabilities: Vec::new(),
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
                required_capabilities: Vec::new(),
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
                required_capabilities: Vec::new(),
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

    #[tokio::test]
    async fn model_and_provider_allowlists_are_intersected() {
        let route = engine()
            .route(MultimodelRequest {
                input: "Fix this typo in the Rust comment".to_string(),
                allowed_models: vec!["cheap".to_string()],
                allowed_providers: vec!["not-local".to_string()],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::CostEfficient,
                default_model: None,
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert_eq!(route.provider, "");
        assert!(route.fallback);
        assert!(route.candidates.is_empty());
    }

    #[tokio::test]
    async fn needs_info_default_must_satisfy_required_capabilities() {
        let route = capability_engine()
            .route(MultimodelRequest {
                input: "hi".to_string(),
                allowed_models: vec!["text-only".to_string()],
                allowed_providers: vec![],
                required_capabilities: vec![ModelCapability::Audio],
                policy: RouterPolicy::Balanced,
                default_model: Some("text-only".to_string()),
                max_output_tokens: None,
            })
            .await;

        assert_eq!(route.model, "");
        assert_eq!(route.provider, "");
        assert_eq!(route.difficulty, DifficultyLabel::NeedsInfo);
        assert!(route.fallback);
        assert!(route.reason.contains("capabilities/context"));
    }

    #[tokio::test]
    async fn needs_info_default_must_fit_the_requested_context() {
        let route = context_engine()
            .route(MultimodelRequest {
                input: "hi".to_string(),
                allowed_models: vec!["small-context".to_string()],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
                policy: RouterPolicy::Balanced,
                default_model: Some("small-context".to_string()),
                max_output_tokens: Some(512),
            })
            .await;

        assert_eq!(route.model, "");
        assert_eq!(route.provider, "");
        assert_eq!(route.difficulty, DifficultyLabel::NeedsInfo);
        assert!(route.fallback);
        assert!(route.reason.contains("capabilities/context"));
    }
}
