use crate::{
    classifier::{HeuristicClassifier, PromptClassifier, SmartClassifier},
    config::{PolicyWeights, RouterConfig},
    router::RoutingEngine,
    types::{DifficultyLabel, DomainLabel, ModelCapability, MultimodelRequest, RouterPolicy},
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    pub input: String,
    pub expected_tier: ExpectedTier,
    #[serde(default)]
    pub expected_domain: Option<DomainLabel>,
    #[serde(default)]
    pub expected_model: Option<String>,
    #[serde(default)]
    pub expected_provider: Option<String>,
    #[serde(default)]
    pub policy: RouterPolicy,
    #[serde(default)]
    pub allowed_models: Vec<String>,
    #[serde(default)]
    pub allowed_providers: Vec<String>,
    #[serde(default)]
    pub required_capabilities: Vec<ModelCapability>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedTier {
    Cheap,
    Balanced,
    Strong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub total: usize,
    pub exact_tier_matches: usize,
    #[serde(default)]
    pub domain_matches: usize,
    #[serde(default)]
    pub model_matches: usize,
    #[serde(default)]
    pub provider_matches: usize,
    #[serde(default)]
    pub domain_examples: usize,
    #[serde(default)]
    pub model_examples: usize,
    #[serde(default)]
    pub provider_examples: usize,
    pub average_cost: f32,
    pub average_capability: f32,
    pub accuracy: f32,
    pub domain_accuracy: f32,
    pub model_accuracy: f32,
    pub provider_accuracy: f32,
    pub misses: Vec<EvalMiss>,
    pub domain_misses: Vec<DomainEvalMiss>,
    pub model_misses: Vec<ModelEvalMiss>,
    pub provider_misses: Vec<ProviderEvalMiss>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalGateReport {
    pub schema_version: u32,
    pub dataset: DatasetArtifact,
    pub execution: EvalExecutionArtifact,
    pub selection: EvalSelectionArtifact,
    pub coverage: EvalCoverageArtifact,
    pub min_examples: usize,
    pub min_accuracy: f32,
    pub min_domain_accuracy: f32,
    pub min_model_accuracy: f32,
    pub min_provider_accuracy: f32,
    pub pass: bool,
    pub failures: Vec<String>,
    pub eval: EvalReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalSelectionArtifact {
    pub strategy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub source_examples: usize,
    pub evaluated_examples: usize,
    pub evaluated_fnv1a_64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalExecutionArtifact {
    /// Classifier implementation exercised by this report.
    pub classifier: String,
    /// Boundary exercised by this report. Eval gates are intentionally kept
    /// separate from HTTP runtime scenario gates.
    pub runtime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configured_model: Option<String>,
    pub adapter_requests: u64,
    pub adapter_successes: u64,
    pub adapter_fallbacks: u64,
    pub adapter_invalid_outputs: u64,
    pub heuristic_routes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalCoverageArtifact {
    pub min_domain_examples: usize,
    pub min_model_examples: usize,
    pub min_provider_examples: usize,
    pub domain_examples: usize,
    pub model_examples: usize,
    pub provider_examples: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EvalCoverageMinimums {
    pub domain: usize,
    pub model: usize,
    pub provider: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMiss {
    pub input: String,
    pub expected_tier: ExpectedTier,
    pub actual_tier: ExpectedTier,
    pub selected_model: String,
    pub difficulty: DifficultyLabel,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEvalMiss {
    pub input: String,
    pub expected_domain: DomainLabel,
    pub actual_domain: Option<DomainLabel>,
    pub selected_model: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvalMiss {
    pub input: String,
    pub expected_model: String,
    pub selected_model: String,
    pub selected_provider: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEvalMiss {
    pub input: String,
    pub expected_provider: String,
    pub selected_model: String,
    pub selected_provider: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub total: usize,
    pub easy_threshold: f32,
    pub hard_threshold: f32,
    pub accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub total: usize,
    pub easy_threshold: f32,
    pub hard_threshold: f32,
    pub balanced: PolicyWeights,
    pub accuracy: f32,
    pub domain_accuracy: f32,
    pub average_cost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationArtifact {
    pub schema_version: u32,
    pub optimizer: String,
    pub created_unix_seconds: u64,
    pub dataset: DatasetArtifact,
    pub split: DatasetSplitArtifact,
    pub search_space: SearchSpaceArtifact,
    pub selection_rule: String,
    pub baseline_report: EvalReport,
    pub optimized_report: OptimizationReport,
    pub validation: ValidationArtifact,
    pub config_patch: OptimizationConfigPatch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimized_config_path: Option<PathBuf>,
    pub replay: ReplayInstructions,
    pub rollback: RollbackInstructions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetArtifact {
    pub path: PathBuf,
    pub examples: usize,
    pub fnv1a_64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplitArtifact {
    pub strategy: String,
    pub train_examples: usize,
    pub holdout_examples: usize,
    pub holdout_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationArtifact {
    pub baseline_train_report: EvalReport,
    pub optimized_train_report: OptimizationReport,
    pub baseline_holdout_report: EvalReport,
    pub optimized_holdout_report: EvalReport,
    pub holdout_pass: bool,
    pub pass_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpaceArtifact {
    pub classifier_easy_threshold: ThresholdRange,
    pub classifier_hard_threshold: ThresholdRange,
    pub balanced_weight_candidates: Vec<PolicyWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfigPatch {
    pub classifier_easy_threshold: f32,
    pub classifier_hard_threshold: f32,
    pub scoring_balanced: PolicyWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayInstructions {
    pub command: String,
    pub apply_patch: OptimizationConfigPatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInstructions {
    pub action: String,
}

pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Vec<EvalExample>> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read eval dataset {}", path.display()))?;
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|(idx, line)| {
            serde_json::from_str::<EvalExample>(line)
                .with_context(|| format!("invalid JSONL at {}:{}", path.display(), idx + 1))
        })
        .collect()
}

pub async fn evaluate<C>(engine: &RoutingEngine<C>, examples: &[EvalExample]) -> EvalReport
where
    C: PromptClassifier,
{
    let mut exact_tier_matches = 0;
    let mut domain_matches = 0;
    let mut model_matches = 0;
    let mut provider_matches = 0;
    let mut domain_examples = 0;
    let mut model_examples = 0;
    let mut provider_examples = 0;
    let mut total_cost = 0.0;
    let mut total_capability = 0.0;
    let mut misses = Vec::new();
    let mut domain_misses = Vec::new();
    let mut model_misses = Vec::new();
    let mut provider_misses = Vec::new();

    for example in examples {
        let route = engine
            .route(MultimodelRequest {
                input: example.input.clone(),
                allowed_models: example.allowed_models.clone(),
                allowed_providers: example.allowed_providers.clone(),
                required_capabilities: example.required_capabilities.clone(),
                policy: example.policy,
                default_model: None,
                max_output_tokens: None,
            })
            .await;
        let config = engine.config();
        let selected = config.find_model(&route.model);
        let actual_tier = selected
            .map(|model| tier_for_capability(model.capability))
            .unwrap_or(ExpectedTier::Balanced);
        let selected_cost = selected
            .map(|model| model.cost_per_million_input + model.cost_per_million_output)
            .unwrap_or_default();
        let selected_capability = selected.map(|model| model.capability).unwrap_or_default();
        let selected_provider = selected.map(|model| model.provider.clone());

        total_cost += selected_cost;
        total_capability += selected_capability;
        if actual_tier == example.expected_tier {
            exact_tier_matches += 1;
        } else {
            misses.push(EvalMiss {
                input: example.input.clone(),
                expected_tier: example.expected_tier,
                actual_tier,
                selected_model: route.model.clone(),
                difficulty: route.difficulty.clone(),
                reason: route.reason.clone(),
            });
        }
        if let Some(expected_domain) = &example.expected_domain {
            domain_examples += 1;
            if route.domain.as_ref() == Some(expected_domain) {
                domain_matches += 1;
            } else {
                domain_misses.push(DomainEvalMiss {
                    input: example.input.clone(),
                    expected_domain: expected_domain.clone(),
                    actual_domain: route.domain.clone(),
                    selected_model: route.model.clone(),
                    reason: route.reason.clone(),
                });
            }
        }
        if let Some(expected_model) = &example.expected_model {
            model_examples += 1;
            if model_matches_expected(&config, &route.model, expected_model) {
                model_matches += 1;
            } else {
                model_misses.push(ModelEvalMiss {
                    input: example.input.clone(),
                    expected_model: expected_model.clone(),
                    selected_model: route.model.clone(),
                    selected_provider: selected_provider.clone(),
                    reason: route.reason.clone(),
                });
            }
        }
        if let Some(expected_provider) = &example.expected_provider {
            provider_examples += 1;
            if selected_provider.as_deref() == Some(expected_provider.as_str()) {
                provider_matches += 1;
            } else {
                provider_misses.push(ProviderEvalMiss {
                    input: example.input.clone(),
                    expected_provider: expected_provider.clone(),
                    selected_model: route.model.clone(),
                    selected_provider,
                    reason: route.reason.clone(),
                });
            }
        }
    }

    let total = examples.len();
    let denominator = total.max(1) as f32;
    let domain_denominator = domain_examples.max(1) as f32;
    let model_denominator = model_examples.max(1) as f32;
    let provider_denominator = provider_examples.max(1) as f32;
    EvalReport {
        total,
        exact_tier_matches,
        domain_matches,
        model_matches,
        provider_matches,
        domain_examples,
        model_examples,
        provider_examples,
        average_cost: total_cost / denominator,
        average_capability: total_capability / denominator,
        accuracy: exact_tier_matches as f32 / denominator,
        domain_accuracy: domain_matches as f32 / domain_denominator,
        model_accuracy: model_matches as f32 / model_denominator,
        provider_accuracy: provider_matches as f32 / provider_denominator,
        misses,
        domain_misses,
        model_misses,
        provider_misses,
    }
}

fn model_matches_expected(
    config: &RouterConfig,
    selected_model: &str,
    expected_model: &str,
) -> bool {
    if selected_model == expected_model {
        return true;
    }
    config
        .find_model(selected_model)
        .is_some_and(|model| model.aliases.iter().any(|alias| alias == expected_model))
}

#[allow(clippy::too_many_arguments)]
pub async fn eval_gate(
    config: &RouterConfig,
    dataset_path: impl AsRef<Path>,
    examples: &[EvalExample],
    min_examples: usize,
    min_accuracy: f32,
    min_domain_accuracy: f32,
    min_model_accuracy: f32,
    min_provider_accuracy: f32,
) -> Result<EvalGateReport> {
    eval_gate_with_coverage(
        config,
        dataset_path,
        examples,
        min_examples,
        min_accuracy,
        min_domain_accuracy,
        min_model_accuracy,
        min_provider_accuracy,
        EvalCoverageMinimums::default(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn eval_gate_with_coverage(
    config: &RouterConfig,
    dataset_path: impl AsRef<Path>,
    examples: &[EvalExample],
    min_examples: usize,
    min_accuracy: f32,
    min_domain_accuracy: f32,
    min_model_accuracy: f32,
    min_provider_accuracy: f32,
    coverage_minimums: EvalCoverageMinimums,
) -> Result<EvalGateReport> {
    let report = evaluate_with_heuristic(config, examples).await;
    finish_eval_gate(
        dataset_path.as_ref(),
        examples,
        report,
        EvalExecutionArtifact {
            classifier: "heuristic".to_string(),
            runtime: "in_memory_routing_engine".to_string(),
            configured_model: None,
            adapter_requests: 0,
            adapter_successes: 0,
            adapter_fallbacks: 0,
            adapter_invalid_outputs: 0,
            heuristic_routes: examples.len() as u64,
        },
        min_examples,
        min_accuracy,
        min_domain_accuracy,
        min_model_accuracy,
        min_provider_accuracy,
        coverage_minimums,
    )
}

#[allow(clippy::too_many_arguments)]
pub async fn configured_eval_gate(
    config: &RouterConfig,
    dataset_path: impl AsRef<Path>,
    examples: &[EvalExample],
    min_examples: usize,
    min_accuracy: f32,
    min_domain_accuracy: f32,
    min_model_accuracy: f32,
    min_provider_accuracy: f32,
    coverage_minimums: EvalCoverageMinimums,
    max_fallback_rate: f32,
) -> Result<EvalGateReport> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&max_fallback_rate),
        "configured eval-gate max_fallback_rate must be between 0.0 and 1.0"
    );
    let classifier = SmartClassifier::new(config.clone())?;
    let metrics_handle = classifier.clone();
    let engine = RoutingEngine::new(config.clone(), classifier);
    let report = evaluate(&engine, examples).await;
    let metrics = metrics_handle.judge_metrics();
    let backend = config.classifier.active_backend();
    let execution = EvalExecutionArtifact {
        classifier: backend.config_key().to_string(),
        runtime: "in_memory_routing_engine".to_string(),
        configured_model: config.classifier.active_adapter().model,
        adapter_requests: metrics.requests,
        adapter_successes: metrics.successes,
        adapter_fallbacks: metrics.fallbacks,
        adapter_invalid_outputs: metrics.invalid_outputs,
        heuristic_routes: metrics.heuristic_routes,
    };
    let mut gate = finish_eval_gate(
        dataset_path.as_ref(),
        examples,
        report,
        execution,
        min_examples,
        min_accuracy,
        min_domain_accuracy,
        min_model_accuracy,
        min_provider_accuracy,
        coverage_minimums,
    )?;
    let fallback_rate =
        gate.execution.adapter_fallbacks as f32 / gate.execution.adapter_requests.max(1) as f32;
    if fallback_rate > max_fallback_rate {
        gate.failures.push(format!(
            "classifier fallback rate {fallback_rate} is above maximum {max_fallback_rate}"
        ));
    }
    if config.classifier.active_backend() != crate::config::ClassifierBackend::Heuristic
        && gate.execution.adapter_requests != examples.len() as u64
    {
        gate.failures.push(format!(
            "configured classifier made {} adapter request(s) for {} example(s)",
            gate.execution.adapter_requests,
            examples.len()
        ));
    }
    gate.pass = gate.failures.is_empty();
    Ok(gate)
}

#[allow(clippy::too_many_arguments)]
fn finish_eval_gate(
    dataset_path: &Path,
    examples: &[EvalExample],
    report: EvalReport,
    execution: EvalExecutionArtifact,
    min_examples: usize,
    min_accuracy: f32,
    min_domain_accuracy: f32,
    min_model_accuracy: f32,
    min_provider_accuracy: f32,
    coverage_minimums: EvalCoverageMinimums,
) -> Result<EvalGateReport> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&min_accuracy),
        "eval-gate min_accuracy must be between 0.0 and 1.0"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&min_domain_accuracy),
        "eval-gate min_domain_accuracy must be between 0.0 and 1.0"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&min_model_accuracy),
        "eval-gate min_model_accuracy must be between 0.0 and 1.0"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&min_provider_accuracy),
        "eval-gate min_provider_accuracy must be between 0.0 and 1.0"
    );
    let mut failures = Vec::new();
    if examples.len() < min_examples {
        failures.push(format!(
            "dataset has {} example(s), below minimum {min_examples}",
            examples.len()
        ));
    }
    if report.accuracy < min_accuracy {
        failures.push(format!(
            "tier accuracy {} is below minimum {min_accuracy}",
            report.accuracy
        ));
    }
    if report.domain_accuracy < min_domain_accuracy {
        failures.push(format!(
            "domain accuracy {} is below minimum {min_domain_accuracy}",
            report.domain_accuracy
        ));
    }
    if report.model_accuracy < min_model_accuracy {
        failures.push(format!(
            "model accuracy {} is below minimum {min_model_accuracy}",
            report.model_accuracy
        ));
    }
    if report.provider_accuracy < min_provider_accuracy {
        failures.push(format!(
            "provider accuracy {} is below minimum {min_provider_accuracy}",
            report.provider_accuracy
        ));
    }
    for (name, actual, minimum) in [
        ("domain", report.domain_examples, coverage_minimums.domain),
        ("model", report.model_examples, coverage_minimums.model),
        (
            "provider",
            report.provider_examples,
            coverage_minimums.provider,
        ),
    ] {
        if actual < minimum {
            failures.push(format!(
                "{name} coverage has {actual} example(s), below minimum {minimum}"
            ));
        }
    }
    Ok(EvalGateReport {
        schema_version: 2,
        dataset: DatasetArtifact {
            path: dataset_path.to_path_buf(),
            examples: examples.len(),
            fnv1a_64: eval_gate_fingerprint(dataset_path, examples)?,
        },
        execution,
        selection: EvalSelectionArtifact {
            strategy: "full_dataset".to_string(),
            seed: None,
            source_examples: examples.len(),
            evaluated_examples: examples.len(),
            evaluated_fnv1a_64: format!("{:016x}", fnv1a_64(&serde_json::to_vec(examples)?)),
        },
        coverage: EvalCoverageArtifact {
            min_domain_examples: coverage_minimums.domain,
            min_model_examples: coverage_minimums.model,
            min_provider_examples: coverage_minimums.provider,
            domain_examples: report.domain_examples,
            model_examples: report.model_examples,
            provider_examples: report.provider_examples,
        },
        min_examples,
        min_accuracy,
        min_domain_accuracy,
        min_model_accuracy,
        min_provider_accuracy,
        pass: failures.is_empty(),
        failures,
        eval: report,
    })
}

fn eval_gate_fingerprint(path: &Path, examples: &[EvalExample]) -> Result<String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => serde_json::to_vec(examples)?,
    };
    Ok(format!("{:016x}", fnv1a_64(&bytes)))
}

pub fn seeded_holdout(
    examples: &[EvalExample],
    holdout_ratio: f32,
    seed: u64,
) -> Result<Vec<EvalExample>> {
    anyhow::ensure!(
        holdout_ratio.is_finite() && holdout_ratio > 0.0 && holdout_ratio <= 1.0,
        "holdout_ratio must be greater than 0.0 and at most 1.0"
    );
    if examples.is_empty() {
        return Ok(Vec::new());
    }
    let mut ranked = examples
        .iter()
        .cloned()
        .map(|example| {
            let mut bytes = seed.to_le_bytes().to_vec();
            bytes.extend(serde_json::to_vec(&example)?);
            Ok((fnv1a_64(&bytes), example))
        })
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.input.cmp(&right.1.input))
    });
    let count = ((examples.len() as f32 * holdout_ratio).ceil() as usize)
        .max(1)
        .min(examples.len());
    Ok(ranked
        .into_iter()
        .take(count)
        .map(|(_, example)| example)
        .collect())
}

pub async fn calibrate_thresholds(
    base_config: &RouterConfig,
    examples: &[EvalExample],
) -> Result<CalibrationReport> {
    let mut best = CalibrationReport {
        total: examples.len(),
        easy_threshold: base_config.classifier.easy_threshold,
        hard_threshold: base_config.classifier.hard_threshold,
        accuracy: 0.0,
    };

    for easy_step in 18..=40 {
        for hard_step in 46..=82 {
            let easy_threshold = easy_step as f32 / 100.0;
            let hard_threshold = hard_step as f32 / 100.0;
            if easy_threshold >= hard_threshold {
                continue;
            }
            let mut config = base_config.clone();
            config.classifier.easy_threshold = easy_threshold;
            config.classifier.hard_threshold = hard_threshold;
            let classifier = HeuristicClassifier::with_thresholds(
                config.classifier.confidence_threshold,
                config.classifier.easy_threshold,
                config.classifier.hard_threshold,
            );
            let engine = RoutingEngine::new(config, classifier);
            let report = evaluate(&engine, examples).await;
            if report.accuracy > best.accuracy {
                best = CalibrationReport {
                    total: examples.len(),
                    easy_threshold,
                    hard_threshold,
                    accuracy: report.accuracy,
                };
            }
        }
    }

    Ok(best)
}

pub async fn optimize_config(
    base_config: &RouterConfig,
    examples: &[EvalExample],
) -> Result<(RouterConfig, OptimizationReport)> {
    let mut best_config = base_config.clone();
    let mut best_report = evaluate_with_heuristic(&best_config, examples).await;
    let mut best = OptimizationReport {
        total: examples.len(),
        easy_threshold: best_config.classifier.easy_threshold,
        hard_threshold: best_config.classifier.hard_threshold,
        balanced: best_config.scoring.balanced,
        accuracy: best_report.accuracy,
        domain_accuracy: best_report.domain_accuracy,
        average_cost: best_report.average_cost,
    };

    for easy_step in 18..=34 {
        for hard_step in 60..=84 {
            let easy_threshold = easy_step as f32 / 100.0;
            let hard_threshold = hard_step as f32 / 100.0;
            if easy_threshold >= hard_threshold {
                continue;
            }
            for balanced in balanced_weight_candidates() {
                let mut config = base_config.clone();
                config.classifier.easy_threshold = easy_threshold;
                config.classifier.hard_threshold = hard_threshold;
                config.scoring.balanced = balanced;
                let report = evaluate_with_heuristic(&config, examples).await;
                if better_report(&report, &best_report) {
                    best_config = config;
                    best_report = report;
                    best = OptimizationReport {
                        total: examples.len(),
                        easy_threshold,
                        hard_threshold,
                        balanced,
                        accuracy: best_report.accuracy,
                        domain_accuracy: best_report.domain_accuracy,
                        average_cost: best_report.average_cost,
                    };
                }
            }
        }
    }

    Ok((best_config, best))
}

pub async fn optimize_with_artifact(
    base_config: &RouterConfig,
    config_path: &Path,
    dataset_path: &Path,
    examples: &[EvalExample],
    optimized_config_path: Option<PathBuf>,
) -> Result<(RouterConfig, OptimizationArtifact)> {
    let (train_examples, holdout_examples) = train_holdout_split(examples);
    let baseline_report = evaluate_with_heuristic(base_config, examples).await;
    let baseline_train_report = evaluate_with_heuristic(base_config, &train_examples).await;
    let baseline_holdout_report = evaluate_with_heuristic(base_config, &holdout_examples).await;
    let (optimized_config, optimized_report) =
        optimize_config(base_config, &train_examples).await?;
    let optimized_holdout_report =
        evaluate_with_heuristic(&optimized_config, &holdout_examples).await;
    let dataset_bytes = fs::read(dataset_path)
        .with_context(|| format!("failed to read eval dataset {}", dataset_path.display()))?;
    let config_patch = OptimizationConfigPatch {
        classifier_easy_threshold: optimized_report.easy_threshold,
        classifier_hard_threshold: optimized_report.hard_threshold,
        scoring_balanced: optimized_report.balanced,
    };
    let command = replay_command(config_path, dataset_path, optimized_config_path.as_deref());
    let artifact = OptimizationArtifact {
        schema_version: 1,
        optimizer: "gepa_style_replayable_grid_search".to_string(),
        created_unix_seconds: unix_seconds(),
        dataset: DatasetArtifact {
            path: dataset_path.to_path_buf(),
            examples: examples.len(),
            fnv1a_64: format!("{:016x}", fnv1a_64(&dataset_bytes)),
        },
        split: DatasetSplitArtifact {
            strategy: if examples.len() >= 5 {
                "deterministic_fnv1a_modulo_80_20".to_string()
            } else {
                "small_dataset_all_examples_for_train_and_holdout".to_string()
            },
            train_examples: train_examples.len(),
            holdout_examples: holdout_examples.len(),
            holdout_ratio: if examples.is_empty() {
                0.0
            } else {
                holdout_examples.len() as f32 / examples.len() as f32
            },
        },
        search_space: SearchSpaceArtifact {
            classifier_easy_threshold: ThresholdRange {
                min: 0.18,
                max: 0.34,
                step: 0.01,
            },
            classifier_hard_threshold: ThresholdRange {
                min: 0.60,
                max: 0.84,
                step: 0.01,
            },
            balanced_weight_candidates: balanced_weight_candidates(),
        },
        selection_rule: "maximize tier accuracy, then domain accuracy, then minimize average cost"
            .to_string(),
        baseline_report,
        optimized_report: optimized_report.clone(),
        validation: ValidationArtifact {
            baseline_train_report,
            optimized_train_report: optimized_report.clone(),
            baseline_holdout_report: baseline_holdout_report.clone(),
            optimized_holdout_report: optimized_holdout_report.clone(),
            holdout_pass: optimized_holdout_report.accuracy >= baseline_holdout_report.accuracy
                && optimized_holdout_report.domain_accuracy >= baseline_holdout_report.domain_accuracy,
            pass_rule:
                "optimized holdout tier accuracy and domain accuracy must be at least baseline"
                    .to_string(),
        },
        config_patch: config_patch.clone(),
        optimized_config_path,
        replay: ReplayInstructions {
            command,
            apply_patch: config_patch,
        },
        rollback: RollbackInstructions {
            action:
                "keep the previous config as the rollback artifact; do not overwrite it without validating this report"
                    .to_string(),
        },
    };
    Ok((optimized_config, artifact))
}

fn train_holdout_split(examples: &[EvalExample]) -> (Vec<EvalExample>, Vec<EvalExample>) {
    if examples.len() < 5 {
        return (examples.to_vec(), examples.to_vec());
    }
    let mut indexed = examples
        .iter()
        .cloned()
        .map(|example| {
            let hash = fnv1a_64(example.input.as_bytes());
            (hash, example)
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(hash, _)| *hash);
    let holdout_count = (examples.len() / 5).max(1);
    let holdout = indexed
        .iter()
        .take(holdout_count)
        .map(|(_, example)| example.clone())
        .collect::<Vec<_>>();
    let train = indexed
        .into_iter()
        .skip(holdout_count)
        .map(|(_, example)| example)
        .collect::<Vec<_>>();
    (train, holdout)
}

async fn evaluate_with_heuristic(config: &RouterConfig, examples: &[EvalExample]) -> EvalReport {
    let classifier = HeuristicClassifier::with_thresholds(
        config.classifier.confidence_threshold,
        config.classifier.easy_threshold,
        config.classifier.hard_threshold,
    );
    let engine = RoutingEngine::new(config.clone(), classifier);
    evaluate(&engine, examples).await
}

fn replay_command(
    config_path: &Path,
    dataset_path: &Path,
    optimized_config_path: Option<&Path>,
) -> String {
    let mut command = format!(
        "cargo run -- --config {} optimize {} --artifact router.optimization.json",
        shell_arg(config_path),
        shell_arg(dataset_path)
    );
    if let Some(path) = optimized_config_path {
        command.push_str(" --write-config ");
        command.push_str(&shell_arg(path));
    }
    command
}

fn shell_arg(path: &Path) -> String {
    let raw = path.display().to_string();
    if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':'))
    {
        raw
    } else {
        format!("'{}'", raw.replace('\'', "'\\''"))
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn better_report(candidate: &EvalReport, best: &EvalReport) -> bool {
    candidate.accuracy > best.accuracy
        || (candidate.accuracy == best.accuracy && candidate.domain_accuracy > best.domain_accuracy)
        || (candidate.accuracy == best.accuracy
            && candidate.domain_accuracy == best.domain_accuracy
            && candidate.average_cost < best.average_cost)
}

fn balanced_weight_candidates() -> Vec<PolicyWeights> {
    vec![
        PolicyWeights {
            capability_fit: 0.60,
            domain_bonus: 0.20,
            cost: 0.20,
            overkill: 1.0,
            raw_capability: 0.0,
            latency: 0.05,
            health: 1.0,
            local_bonus: 0.0,
            remote_penalty: 0.0,
            multimodal_capability: 0.0,
        },
        PolicyWeights {
            capability_fit: 0.58,
            domain_bonus: 0.24,
            cost: 0.28,
            overkill: 1.2,
            raw_capability: 0.0,
            latency: 0.05,
            health: 1.0,
            local_bonus: 0.0,
            remote_penalty: 0.0,
            multimodal_capability: 0.0,
        },
        PolicyWeights {
            capability_fit: 0.64,
            domain_bonus: 0.18,
            cost: 0.12,
            overkill: 0.8,
            raw_capability: 0.0,
            latency: 0.05,
            health: 1.0,
            local_bonus: 0.0,
            remote_penalty: 0.0,
            multimodal_capability: 0.0,
        },
        PolicyWeights {
            capability_fit: 0.52,
            domain_bonus: 0.34,
            cost: 0.22,
            overkill: 1.0,
            raw_capability: 0.0,
            latency: 0.05,
            health: 1.0,
            local_bonus: 0.0,
            remote_penalty: 0.0,
            multimodal_capability: 0.0,
        },
        PolicyWeights {
            capability_fit: 0.44,
            domain_bonus: 0.18,
            cost: 0.46,
            overkill: 1.6,
            raw_capability: 0.0,
            latency: 0.05,
            health: 1.0,
            local_bonus: 0.0,
            remote_penalty: 0.0,
            multimodal_capability: 0.0,
        },
    ]
}

fn tier_for_capability(capability: f32) -> ExpectedTier {
    if capability < 0.45 {
        ExpectedTier::Cheap
    } else if capability < 0.82 {
        ExpectedTier::Balanced
    } else {
        ExpectedTier::Strong
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{
            AuthConfig, BudgetConfig, ClassifierBackend, ClassifierConfig, RuntimeConfig,
            ScoringConfig, TelemetryConfig,
        },
        types::{ModelConfig, ProviderConfig, ProviderKind},
    };
    use axum::{Json, Router, routing::post};
    use tokio::net::TcpListener;

    #[test]
    fn loads_jsonl_examples() {
        let path = std::env::temp_dir().join("autohand-router-eval-test.jsonl");
        fs::write(
            &path,
            r#"{"input":"Fix typo","expected_tier":"cheap","expected_domain":"coding"}"#,
        )
        .unwrap();
        let examples = load_jsonl(&path).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].expected_tier, ExpectedTier::Cheap);
        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn optimization_artifact_is_replayable_without_serializing_provider_secrets() {
        let path = std::env::temp_dir().join("autohand-router-optimization-artifact-test.jsonl");
        fs::write(
            &path,
            [
                r#"{"input":"Fix typo","expected_tier":"cheap","expected_domain":"coding"}"#,
                r#"{"input":"Summarize docs","expected_tier":"cheap","expected_domain":"summary"}"#,
                r#"{"input":"Add async Rust tests","expected_tier":"balanced","expected_domain":"coding"}"#,
                r#"{"input":"Analyze warehouse query","expected_tier":"balanced","expected_domain":"data"}"#,
                r#"{"input":"Design event sourcing platform","expected_tier":"strong","expected_domain":"design"}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let config = test_config();
        let examples = load_jsonl(&path).unwrap();
        let (_optimized, artifact) = optimize_with_artifact(
            &config,
            Path::new("examples/router.yaml"),
            &path,
            &examples,
            Some(PathBuf::from("router.optimized.yaml")),
        )
        .await
        .unwrap();
        let artifact_json = serde_json::to_string(&artifact).unwrap();

        assert_eq!(artifact.schema_version, 1);
        assert_eq!(artifact.dataset.examples, 5);
        assert_eq!(artifact.split.train_examples, 4);
        assert_eq!(artifact.split.holdout_examples, 1);
        assert_eq!(artifact.validation.baseline_train_report.total, 4);
        assert_eq!(artifact.validation.optimized_train_report.total, 4);
        assert_eq!(artifact.validation.baseline_holdout_report.total, 1);
        assert_eq!(artifact.validation.optimized_holdout_report.total, 1);
        assert!(artifact.replay.command.contains("--write-config"));
        assert!(!artifact_json.contains("secret-token"));
        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn eval_gate_fails_undersized_dataset() {
        let config = test_config();
        let examples = vec![EvalExample {
            input: "Fix typo in Rust docs".to_string(),
            expected_tier: ExpectedTier::Cheap,
            expected_domain: Some(DomainLabel::Coding),
            expected_model: None,
            expected_provider: None,
            policy: RouterPolicy::CostEfficient,
            allowed_models: vec![],
            allowed_providers: vec![],
            required_capabilities: Vec::new(),
        }];

        let report = eval_gate(
            &config,
            Path::new("tiny.jsonl"),
            &examples,
            2,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        .await
        .unwrap();

        assert!(!report.pass);
        assert!(report.failures[0].contains("below minimum"));
    }

    #[tokio::test]
    async fn eval_gate_passes_when_size_and_accuracy_thresholds_are_met() {
        let config = test_config();
        let examples = vec![
            EvalExample {
                input: "Fix typo in Rust docs".to_string(),
                expected_tier: ExpectedTier::Cheap,
                expected_domain: Some(DomainLabel::Coding),
                expected_model: Some("cheap".to_string()),
                expected_provider: Some("local".to_string()),
                policy: RouterPolicy::CostEfficient,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            },
            EvalExample {
                input: "Design event sourcing platform with security".to_string(),
                expected_tier: ExpectedTier::Strong,
                expected_domain: Some(DomainLabel::Design),
                expected_model: Some("strong".to_string()),
                expected_provider: Some("local".to_string()),
                policy: RouterPolicy::CapabilityHeavy,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            },
        ];

        let report = eval_gate(
            &config,
            Path::new("production.jsonl"),
            &examples,
            2,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        .await
        .unwrap();

        assert!(report.pass);
        assert!(report.failures.is_empty());
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.execution.classifier, "heuristic");
        assert_eq!(report.execution.runtime, "in_memory_routing_engine");
    }

    #[tokio::test]
    async fn eval_gate_enforces_explicit_subgroup_sample_minimums() {
        let config = test_config();
        let examples = vec![cheap_example()];
        let report = eval_gate_with_coverage(
            &config,
            Path::new("coverage.jsonl"),
            &examples,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            EvalCoverageMinimums {
                domain: 2,
                model: 1,
                provider: 1,
            },
        )
        .await
        .unwrap();

        assert!(!report.pass);
        assert_eq!(report.coverage.min_domain_examples, 2);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| { failure == "domain coverage has 1 example(s), below minimum 2" })
        );
    }

    #[tokio::test]
    async fn configured_eval_gate_records_successful_adapter_execution() {
        let base_url = spawn_classifier_server(
            r#"{"difficulty":"easy","ambiguity":"low","domain":"coding","modality":"text","safety":"safe","cacheability":"high","latency_sensitivity":"low","reasoning_depth":"shallow","confidence":0.99,"ambiguity_confidence":0.99,"domain_confidence":0.99,"modality_confidence":0.99,"safety_confidence":0.99,"cacheability_confidence":0.99,"latency_sensitivity_confidence":0.99,"reasoning_depth_confidence":0.99}"#,
        )
        .await;
        let mut config = test_config();
        config.providers[0].base_url = base_url;
        config.classifier.backend = ClassifierBackend::LlmJudge;
        config.classifier.adapters.llm_judge.model = Some("cheap".to_string());
        let examples = vec![cheap_example()];

        let report = configured_eval_gate(
            &config,
            Path::new("configured.jsonl"),
            &examples,
            1,
            1.0,
            1.0,
            1.0,
            1.0,
            EvalCoverageMinimums {
                domain: 1,
                model: 1,
                provider: 1,
            },
            0.0,
        )
        .await
        .unwrap();

        assert!(
            report.pass,
            "failures={:?} execution={:?}",
            report.failures, report.execution
        );
        assert_eq!(report.execution.classifier, "llm_judge");
        assert_eq!(report.execution.configured_model.as_deref(), Some("cheap"));
        assert_eq!(report.execution.adapter_requests, 1);
        assert_eq!(report.execution.adapter_successes, 1);
        assert_eq!(report.execution.adapter_fallbacks, 0);
        assert_eq!(report.execution.heuristic_routes, 0);
    }

    #[tokio::test]
    async fn configured_eval_gate_fails_when_adapter_falls_back() {
        let base_url = spawn_classifier_server("not-json").await;
        let mut config = test_config();
        config.providers[0].base_url = base_url;
        config.classifier.backend = ClassifierBackend::LlmJudge;
        config.classifier.adapters.llm_judge.model = Some("cheap".to_string());

        let report = configured_eval_gate(
            &config,
            Path::new("configured.jsonl"),
            &[cheap_example()],
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            EvalCoverageMinimums::default(),
            0.0,
        )
        .await
        .unwrap();

        assert!(!report.pass);
        assert_eq!(report.execution.adapter_requests, 1);
        assert_eq!(report.execution.adapter_fallbacks, 1);
        assert_eq!(report.execution.adapter_invalid_outputs, 1);
        assert_eq!(report.execution.heuristic_routes, 1);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("fallback rate"))
        );
    }

    #[tokio::test]
    async fn eval_report_tracks_model_and_provider_expectations() {
        let config = test_config();
        let engine = RoutingEngine::new(config, HeuristicClassifier::default());
        let examples = vec![
            EvalExample {
                input: "Fix typo in Rust docs".to_string(),
                expected_tier: ExpectedTier::Cheap,
                expected_domain: Some(DomainLabel::Coding),
                expected_model: Some("cheap".to_string()),
                expected_provider: Some("local".to_string()),
                policy: RouterPolicy::CostEfficient,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            },
            EvalExample {
                input: "Fix typo in Rust docs".to_string(),
                expected_tier: ExpectedTier::Cheap,
                expected_domain: Some(DomainLabel::Coding),
                expected_model: Some("strong".to_string()),
                expected_provider: Some("remote".to_string()),
                policy: RouterPolicy::CostEfficient,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            },
        ];

        let report = evaluate(&engine, &examples).await;

        assert_eq!(report.model_matches, 1);
        assert_eq!(report.provider_matches, 1);
        assert_eq!(report.domain_examples, 2);
        assert_eq!(report.model_examples, 2);
        assert_eq!(report.provider_examples, 2);
        assert_eq!(report.model_misses.len(), 1);
        assert_eq!(report.provider_misses.len(), 1);
        assert_eq!(report.model_accuracy, 0.5);
        assert_eq!(report.provider_accuracy, 0.5);
    }

    #[test]
    fn train_holdout_split_is_stable_and_preserves_examples() {
        let examples = (0..10)
            .map(|index| EvalExample {
                input: format!("example-{index}"),
                expected_tier: ExpectedTier::Balanced,
                expected_domain: None,
                expected_model: None,
                expected_provider: None,
                policy: RouterPolicy::Balanced,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            })
            .collect::<Vec<_>>();

        let (train, holdout) = train_holdout_split(&examples);
        let (train_again, holdout_again) = train_holdout_split(&examples);

        assert_eq!(train.len(), 8);
        assert_eq!(holdout.len(), 2);
        assert_eq!(
            train
                .iter()
                .map(|example| &example.input)
                .collect::<Vec<_>>(),
            train_again
                .iter()
                .map(|example| &example.input)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            holdout
                .iter()
                .map(|example| &example.input)
                .collect::<Vec<_>>(),
            holdout_again
                .iter()
                .map(|example| &example.input)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn seeded_holdout_is_reproducible_and_seed_sensitive() {
        let examples = (0..20)
            .map(|index| EvalExample {
                input: format!("example-{index}"),
                expected_tier: ExpectedTier::Balanced,
                expected_domain: None,
                expected_model: None,
                expected_provider: None,
                policy: RouterPolicy::Balanced,
                allowed_models: vec![],
                allowed_providers: vec![],
                required_capabilities: Vec::new(),
            })
            .collect::<Vec<_>>();

        let first = seeded_holdout(&examples, 0.2, 7).unwrap();
        let repeat = seeded_holdout(&examples, 0.2, 7).unwrap();
        let another_seed = seeded_holdout(&examples, 0.2, 8).unwrap();

        assert_eq!(first.len(), 4);
        assert_eq!(
            first.iter().map(|item| &item.input).collect::<Vec<_>>(),
            repeat.iter().map(|item| &item.input).collect::<Vec<_>>()
        );
        assert_ne!(
            first.iter().map(|item| &item.input).collect::<Vec<_>>(),
            another_seed
                .iter()
                .map(|item| &item.input)
                .collect::<Vec<_>>()
        );
        assert!(seeded_holdout(&examples, 0.0, 7).is_err());
    }

    fn cheap_example() -> EvalExample {
        EvalExample {
            input: "Fix typo in Rust docs".to_string(),
            expected_tier: ExpectedTier::Cheap,
            expected_domain: Some(DomainLabel::Coding),
            expected_model: Some("cheap".to_string()),
            expected_provider: Some("local".to_string()),
            policy: RouterPolicy::CostEfficient,
            allowed_models: vec![],
            allowed_providers: vec![],
            required_capabilities: Vec::new(),
        }
    }

    async fn spawn_classifier_server(content: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                Json(serde_json::json!({
                    "choices": [{"message": {"content": content}}]
                }))
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{address}")
    }

    fn test_config() -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "cheap".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "local".to_string(),
                kind: ProviderKind::OpenAiCompatible,
                base_url: "http://localhost:11434".to_string(),
                api_key_env: None,
                api_key: Some("secret-token".to_string()),
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
            models: vec![
                ModelConfig {
                    id: "cheap".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.3,
                    cost_per_million_input: 0.1,
                    cost_per_million_output: 0.1,
                    domains: vec![DomainLabel::Coding],
                    context_window: Some(4096),
                    capabilities: Default::default(),
                    local: true,
                },
                ModelConfig {
                    id: "strong".to_string(),
                    provider: "local".to_string(),
                    aliases: vec![],
                    capability: 0.9,
                    cost_per_million_input: 10.0,
                    cost_per_million_output: 10.0,
                    domains: vec![DomainLabel::Design],
                    context_window: Some(4096),
                    capabilities: Default::default(),
                    local: true,
                },
            ],
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
