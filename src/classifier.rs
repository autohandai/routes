use crate::config::{ClassifierBackend, ClassifierModelAdapterConfig, RouterConfig};
use crate::provider::ProviderClient;
use crate::types::{
    AmbiguityLabel, CacheabilityLabel, ChatMessage, Classification, Classifications,
    DifficultyLabel, DomainLabel, LatencySensitivityLabel, ModalityLabel, OpenAiChatRequest,
    ReasoningDepthLabel, SafetyLabel,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::time::timeout;

#[async_trait]
pub trait PromptClassifier: Send + Sync + 'static {
    async fn classify(&self, input: &str) -> Classifications;
}

#[derive(Debug, Clone)]
pub struct HeuristicClassifier {
    confidence_threshold: f32,
    easy_threshold: f32,
    hard_threshold: f32,
}

impl HeuristicClassifier {
    pub fn new(confidence_threshold: f32) -> Self {
        Self::with_thresholds(confidence_threshold, 0.28, 0.62)
    }

    pub fn with_thresholds(
        confidence_threshold: f32,
        easy_threshold: f32,
        hard_threshold: f32,
    ) -> Self {
        Self {
            confidence_threshold,
            easy_threshold,
            hard_threshold,
        }
    }
}

impl Default for HeuristicClassifier {
    fn default() -> Self {
        Self::new(0.62)
    }
}

pub fn classify_safety_deterministically(
    input: &str,
    confidence_threshold: f32,
) -> Classification<SafetyLabel> {
    classify_safety(&PromptFeatures::from(input), confidence_threshold)
}

#[derive(Clone)]
pub struct SmartClassifier {
    heuristic: HeuristicClassifier,
    config: Arc<RouterConfig>,
    providers: ProviderClient,
    judge_metrics: Arc<JudgeMetrics>,
}

impl SmartClassifier {
    pub fn new(config: RouterConfig) -> Result<Self> {
        let heuristic = HeuristicClassifier::with_thresholds(
            config.classifier.confidence_threshold,
            config.classifier.easy_threshold,
            config.classifier.hard_threshold,
        );
        let providers = ProviderClient::new(&config)?;
        Ok(Self {
            heuristic,
            config: Arc::new(config),
            providers,
            judge_metrics: Default::default(),
        })
    }

    pub fn judge_metrics(&self) -> JudgeMetricsSnapshot {
        self.judge_metrics.snapshot()
    }

    async fn classify_with_adapter(
        &self,
        input: &str,
        backend: ClassifierBackend,
        adapter: ClassifierModelAdapterConfig,
    ) -> Result<Classifications> {
        let Some(model_id) = adapter.model.as_deref() else {
            anyhow::bail!(
                "{} classifier model is not configured",
                backend.config_key()
            );
        };
        let model = self
            .config
            .find_model(model_id)
            .with_context(|| format!("classifier model {model_id} is not configured"))?;
        let default_prompt = format!(
            "Classify the user request for model routing. Return only JSON with keys difficulty, ambiguity, domain, modality, safety, cacheability, latency_sensitivity, reasoning_depth, confidence, ambiguity_confidence, domain_confidence, modality_confidence, safety_confidence, cacheability_confidence, latency_sensitivity_confidence, reasoning_depth_confidence. Valid difficulty: easy, medium, hard, needs_info. Valid ambiguity: low, med, high. Valid domain: general, summary, coding, design, data. Valid modality: text, vision, audio, tool_use, multimodal. Valid safety: safe, sensitive, unsafe. Valid cacheability: low, medium, high. Valid latency_sensitivity: low, medium, high. Valid reasoning_depth: shallow, moderate, deep.\n\nUser request:\n{input}"
        );
        let judge_prompt = adapter
            .prompt_template
            .as_deref()
            .map(|template| template.replace("{input}", input))
            .unwrap_or(default_prompt);
        let body = OpenAiChatRequest {
            model: model.id.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Value::String(judge_prompt),
                extra: Default::default(),
            }],
            extra: serde_json::Map::from_iter([
                ("temperature".to_string(), Value::from(0.0)),
                ("max_tokens".to_string(), Value::from(180)),
            ]),
        };

        let response = timeout(
            Duration::from_millis(adapter.timeout_ms),
            self.providers.send_chat(&self.config, model, body),
        )
        .await
        .with_context(|| format!("{} classifier request timed out", backend.config_key()))??;
        let status = response.status();
        anyhow::ensure!(
            status.is_success(),
            "{} classifier provider returned HTTP status {status}",
            backend.config_key()
        );
        let bytes = response
            .bytes()
            .await
            .context("failed to read LLM judge response body")?;
        let value = serde_json::from_slice::<Value>(&bytes)?;
        let content = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .with_context(|| {
                format!(
                    "{} classifier response did not include choices[0].message.content",
                    backend.config_key()
                )
            })?;
        parse_judge_content(content)
    }
}

#[async_trait]
impl PromptClassifier for SmartClassifier {
    async fn classify(&self, input: &str) -> Classifications {
        let backend = self.config.classifier.active_backend();
        if matches!(
            backend,
            ClassifierBackend::LlmJudge | ClassifierBackend::RouteLlm
        ) {
            self.judge_metrics.requests.fetch_add(1, Ordering::Relaxed);
            match self
                .classify_with_adapter(input, backend, self.config.classifier.active_adapter())
                .await
            {
                Ok(classifications) => {
                    self.judge_metrics.successes.fetch_add(1, Ordering::Relaxed);
                    return classifications;
                }
                Err(error) => {
                    self.judge_metrics.fallbacks.fetch_add(1, Ordering::Relaxed);
                    if is_judge_output_error(&error) {
                        self.judge_metrics
                            .invalid_outputs
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    tracing::warn!(
                        ?error,
                        backend = backend.config_key(),
                        "classifier backend failed; falling back to heuristic classifier"
                    );
                }
            }
        }
        self.judge_metrics
            .heuristic_routes
            .fetch_add(1, Ordering::Relaxed);
        self.heuristic.classify(input).await
    }
}

#[derive(Default)]
pub struct JudgeMetrics {
    requests: AtomicU64,
    successes: AtomicU64,
    fallbacks: AtomicU64,
    invalid_outputs: AtomicU64,
    heuristic_routes: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeMetricsSnapshot {
    pub requests: u64,
    pub successes: u64,
    pub fallbacks: u64,
    pub invalid_outputs: u64,
    pub heuristic_routes: u64,
}

impl JudgeMetrics {
    fn snapshot(&self) -> JudgeMetricsSnapshot {
        JudgeMetricsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            invalid_outputs: self.invalid_outputs.load(Ordering::Relaxed),
            heuristic_routes: self.heuristic_routes.load(Ordering::Relaxed),
        }
    }
}

#[async_trait]
impl PromptClassifier for HeuristicClassifier {
    async fn classify(&self, input: &str) -> Classifications {
        let features = PromptFeatures::from(input);
        let difficulty = classify_difficulty(
            &features,
            self.confidence_threshold,
            self.easy_threshold,
            self.hard_threshold,
        );
        let ambiguity = classify_ambiguity(&features, self.confidence_threshold);
        let domain = classify_domain(&features, self.confidence_threshold);
        let modality = classify_modality(&features, self.confidence_threshold);
        let safety = classify_safety(&features, self.confidence_threshold);
        let cacheability = classify_cacheability(&features, self.confidence_threshold);
        let latency_sensitivity =
            classify_latency_sensitivity(&features, self.confidence_threshold);
        let reasoning_depth = classify_reasoning_depth(
            &features,
            self.confidence_threshold,
            &difficulty.label,
            &ambiguity.label,
        );
        Classifications {
            difficulty,
            ambiguity,
            domain,
            modality,
            safety,
            cacheability,
            latency_sensitivity,
            reasoning_depth,
        }
    }
}

#[derive(Debug)]
struct PromptFeatures {
    lower: String,
    token_count: usize,
    question_marks: usize,
    code_markers: usize,
    image_markers: usize,
    audio_markers: usize,
    tool_markers: usize,
    json_markers: usize,
}

impl From<&str> for PromptFeatures {
    fn from(input: &str) -> Self {
        let lower = input.to_lowercase();
        let token_count = input.split_whitespace().count();
        let question_marks = input.matches('?').count();
        let code_markers = ["```", "fn ", "class ", "const ", "let ", "impl ", "pub "]
            .iter()
            .filter(|marker| lower.contains(**marker))
            .count();
        let image_markers = count_terms(
            &lower,
            &[
                "image",
                "screenshot",
                "photo",
                "vision",
                "diagram",
                "ocr",
                "multimodal",
            ],
        );
        let audio_markers = count_terms(
            &lower,
            &[
                "audio",
                "speech",
                "voice",
                "transcribe",
                "transcription",
                "translate this recording",
                "tts",
            ],
        );
        let tool_markers = count_terms(
            &lower,
            &[
                "tool",
                "function call",
                "function calling",
                "call the api",
                "use the api",
                "webhook",
                "json schema",
            ],
        );
        let json_markers = count_terms(&lower, &["json", "schema", "structured output"]);
        Self {
            lower,
            token_count,
            question_marks,
            code_markers,
            image_markers,
            audio_markers,
            tool_markers,
            json_markers,
        }
    }
}

fn classify_difficulty(
    features: &PromptFeatures,
    threshold: f32,
    easy_threshold: f32,
    hard_threshold: f32,
) -> Classification<DifficultyLabel> {
    let hard_terms = count_terms(
        &features.lower,
        &[
            "architecture",
            "design",
            "distributed",
            "multi tenant",
            "security",
            "migration",
            "optimize",
            "refactor",
            "debug",
            "compile",
            "concurrency",
            "production",
            "benchmark",
            "agent",
            "router",
            "database",
            "queue",
            "worker",
            "retry",
            "timeout",
            "event sourcing",
            "error handling",
            "async",
            "tests",
            "api client",
            "formal",
            "proof",
            "incident response",
            "safety policy",
            "prompt injection",
            "credentials",
            "web app",
            "multimodal",
            "long context",
            "repository",
        ],
    );
    let easy_terms = count_terms(
        &features.lower,
        &["typo", "rename", "todo", "comment", "summarize", "format"],
    );
    let mut score = (features.token_count as f32 / 180.0)
        + hard_terms as f32 * 0.18
        + features.code_markers as f32 * 0.10
        - easy_terms as f32 * 0.16;
    score = score.clamp(0.0, 1.0);

    let (label, class_id, confidence) = if features.token_count < 3 {
        (DifficultyLabel::NeedsInfo, 3, 0.80)
    } else if score < easy_threshold {
        (DifficultyLabel::Easy, 0, 0.84 - score * 0.3)
    } else if score < hard_threshold {
        (DifficultyLabel::Medium, 1, 0.70)
    } else {
        (DifficultyLabel::Hard, 2, score.max(0.72))
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_ambiguity(features: &PromptFeatures, threshold: f32) -> Classification<AmbiguityLabel> {
    let vague_terms = count_terms(
        &features.lower,
        &[
            "better",
            "fix it",
            "make it work",
            "improve",
            "thing",
            "stuff",
            "asap",
            "perfect",
        ],
    );
    let precise_terms = count_terms(
        &features.lower,
        &[
            "given",
            "when",
            "then",
            "acceptance",
            "schema",
            "api",
            "error",
            "stack trace",
            "test",
            "file",
        ],
    );
    let mut score = vague_terms as f32 * 0.22 + features.question_marks as f32 * 0.06
        - precise_terms as f32 * 0.08;
    if features.token_count < 8 {
        score += 0.25;
    }
    score = score.clamp(0.0, 1.0);

    let (label, class_id, confidence) = if score < 0.25 {
        (AmbiguityLabel::Low, 0, 0.86)
    } else if score < 0.58 {
        (AmbiguityLabel::Med, 1, 0.72)
    } else {
        (AmbiguityLabel::High, 2, 0.78)
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_domain(features: &PromptFeatures, threshold: f32) -> Classification<DomainLabel> {
    let candidates = [
        (
            DomainLabel::Coding,
            2,
            count_terms(
                &features.lower,
                &[
                    "code",
                    "rust",
                    "typescript",
                    "function",
                    "compile",
                    "bug",
                    "test",
                    "api",
                    "class",
                    "module",
                    "crate",
                    "react",
                    "vite",
                    "web app",
                    "dashboard",
                    "browser",
                    "css",
                    "canvas",
                    "form",
                    "client-side",
                    "repository",
                    "dbt",
                    "ast transform",
                ],
            ) + features.code_markers,
        ),
        (
            DomainLabel::Design,
            3,
            count_terms(
                &features.lower,
                &[
                    "architecture",
                    "architect",
                    "design",
                    "system",
                    "roadmap",
                    "tradeoff",
                    "options",
                    "policy",
                    "threat model",
                    "rollout",
                    "failover",
                    "deployment",
                    "strategy",
                    "fallback",
                    "provider",
                    "safety policy",
                    "incident response",
                    "api credentials",
                    "credentials",
                    "prompt injection",
                    "mcp",
                    "protocol",
                    "pick between",
                    "high capability",
                    "multimodal app",
                    "kimi",
                    "glm",
                    "sonnet",
                    "benchmark suite",
                    "event sourcing",
                    "multi tenant",
                ],
            ),
        ),
        (
            DomainLabel::Data,
            4,
            count_terms(
                &features.lower,
                &[
                    "sql",
                    "data",
                    "analytics",
                    "metric",
                    "dataset",
                    "csv",
                    "warehouse",
                    "table",
                    "table into json",
                    "dbt",
                    "incremental",
                    "backfills",
                    "customer rows",
                ],
            ),
        ),
        (
            DomainLabel::Summary,
            1,
            count_terms(
                &features.lower,
                &[
                    "summarize",
                    "extract",
                    "tl;dr",
                    "recap",
                    "brief",
                    "transcribe",
                    "audio clip",
                    "support call",
                    "release notes",
                    "policy document",
                    "citations",
                ],
            ),
        ),
    ];

    let (label, class_id, hits) = candidates
        .into_iter()
        .max_by_key(|(_, _, hits)| *hits)
        .filter(|(_, _, hits)| *hits > 0)
        .unwrap_or((DomainLabel::General, 0, 0));
    let confidence = if hits == 0 {
        0.69
    } else {
        (0.68 + hits as f32 * 0.07).min(0.95)
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_modality(features: &PromptFeatures, threshold: f32) -> Classification<ModalityLabel> {
    let has_vision = features.image_markers > 0;
    let has_audio = features.audio_markers > 0;
    let has_tools = features.tool_markers > 0;
    let active_modes = [has_vision, has_audio, has_tools]
        .into_iter()
        .filter(|active| *active)
        .count();
    let (label, class_id, confidence) = if active_modes > 1 {
        (ModalityLabel::Multimodal, 4, 0.86)
    } else if has_vision {
        (ModalityLabel::Vision, 1, 0.84)
    } else if has_audio {
        (ModalityLabel::Audio, 2, 0.84)
    } else if has_tools {
        (ModalityLabel::ToolUse, 3, 0.80)
    } else {
        (ModalityLabel::Text, 0, 0.88)
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_safety(features: &PromptFeatures, threshold: f32) -> Classification<SafetyLabel> {
    let unsafe_terms = count_terms(
        &features.lower,
        &[
            "jailbreak",
            "bypass safety",
            "ignore previous instructions",
            "steal",
            "exfiltrate",
            "malware",
            "phishing",
            "credential theft",
            "sql injection",
            "exploit",
        ],
    );
    let sensitive_terms = count_terms(
        &features.lower,
        &[
            "pii",
            "ssn",
            "social security",
            "credit card",
            "password",
            "secret",
            "api key",
            "api_key",
            "apikey",
            "sk-",
            "pk-",
            "token",
            "authorization",
            "bearer",
            "email",
            "medical",
            "legal",
            "financial",
            "private",
            "auth",
            "security",
        ],
    ) + usize::from(
        features
            .lower
            .split_whitespace()
            .any(|token| token.contains('@') && token.contains('.')),
    );
    let (label, class_id, confidence) = if unsafe_terms > 0 {
        (
            SafetyLabel::Unsafe,
            2,
            (0.78 + unsafe_terms as f32 * 0.05).min(0.95),
        )
    } else if sensitive_terms > 0 {
        (
            SafetyLabel::Sensitive,
            1,
            (0.72 + sensitive_terms as f32 * 0.05).min(0.92),
        )
    } else {
        (SafetyLabel::Safe, 0, 0.82)
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_cacheability(
    features: &PromptFeatures,
    threshold: f32,
) -> Classification<CacheabilityLabel> {
    let stable_terms = count_terms(
        &features.lower,
        &[
            "explain",
            "summarize",
            "document",
            "docs",
            "example",
            "template",
            "how do i",
            "what is",
            "convert",
            "format",
        ],
    );
    let volatile_terms = count_terms(
        &features.lower,
        &[
            "latest",
            "today",
            "current",
            "now",
            "live",
            "my",
            "this file",
            "this screenshot",
            "private",
            "token",
            "password",
            "secret",
        ],
    );
    let mut score =
        0.45 + (stable_terms + features.json_markers) as f32 * 0.13 - volatile_terms as f32 * 0.18;
    if features.code_markers > 0 {
        score -= 0.10;
    }
    score = score.clamp(0.0, 1.0);
    let (label, class_id, confidence) = if score < 0.35 {
        (CacheabilityLabel::Low, 0, 0.78)
    } else if score < 0.68 {
        (CacheabilityLabel::Medium, 1, 0.72)
    } else {
        (CacheabilityLabel::High, 2, score.max(0.78))
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_latency_sensitivity(
    features: &PromptFeatures,
    threshold: f32,
) -> Classification<LatencySensitivityLabel> {
    let fast_terms = count_terms(
        &features.lower,
        &[
            "fast",
            "quick",
            "asap",
            "low latency",
            "interactive",
            "realtime",
            "real-time",
            "instant",
            "stream",
        ],
    );
    let patient_terms = count_terms(
        &features.lower,
        &[
            "deep", "thorough", "careful", "research", "analyze", "audit",
        ],
    );
    let mut score = 0.42 + fast_terms as f32 * 0.22 - patient_terms as f32 * 0.10;
    if features.token_count > 140 {
        score -= 0.12;
    }
    score = score.clamp(0.0, 1.0);
    let (label, class_id, confidence) = if score < 0.30 {
        (LatencySensitivityLabel::Low, 0, 0.76)
    } else if score < 0.65 {
        (LatencySensitivityLabel::Medium, 1, 0.72)
    } else {
        (LatencySensitivityLabel::High, 2, score.max(0.78))
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn classify_reasoning_depth(
    features: &PromptFeatures,
    threshold: f32,
    difficulty: &DifficultyLabel,
    ambiguity: &AmbiguityLabel,
) -> Classification<ReasoningDepthLabel> {
    let reasoning_terms = count_terms(
        &features.lower,
        &[
            "reason",
            "analyze",
            "architecture",
            "design",
            "debug",
            "root cause",
            "tradeoff",
            "proof",
            "formal",
            "plan",
            "multi step",
            "multi-step",
            "evaluate",
            "compare",
        ],
    );
    let mut score = features.token_count as f32 / 220.0 + reasoning_terms as f32 * 0.16;
    if matches!(difficulty, DifficultyLabel::Hard) {
        score += 0.28;
    }
    if matches!(ambiguity, AmbiguityLabel::High) {
        score += 0.12;
    }
    score = score.clamp(0.0, 1.0);
    let (label, class_id, confidence) = if score < 0.28 {
        (ReasoningDepthLabel::Shallow, 0, 0.82)
    } else if score < 0.65 {
        (ReasoningDepthLabel::Moderate, 1, 0.74)
    } else {
        (ReasoningDepthLabel::Deep, 2, score.max(0.80))
    };

    Classification {
        class_id,
        label,
        confidence,
        meets_threshold: confidence >= threshold,
    }
}

fn count_terms(input: &str, terms: &[&str]) -> usize {
    terms.iter().filter(|term| input.contains(**term)).count()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JudgeOutput {
    difficulty: DifficultyLabel,
    ambiguity: AmbiguityLabel,
    domain: DomainLabel,
    modality: Option<ModalityLabel>,
    safety: Option<SafetyLabel>,
    cacheability: Option<CacheabilityLabel>,
    latency_sensitivity: Option<LatencySensitivityLabel>,
    reasoning_depth: Option<ReasoningDepthLabel>,
    confidence: f32,
    ambiguity_confidence: f32,
    domain_confidence: f32,
    modality_confidence: Option<f32>,
    safety_confidence: Option<f32>,
    cacheability_confidence: Option<f32>,
    latency_sensitivity_confidence: Option<f32>,
    reasoning_depth_confidence: Option<f32>,
}

fn parse_judge_content(content: &str) -> Result<Classifications> {
    let json = extract_json_object(content).context("LLM judge content did not contain JSON")?;
    let output = serde_json::from_str::<JudgeOutput>(json)?;
    output.validate()?;
    Ok(Classifications {
        difficulty: Classification {
            class_id: difficulty_id(&output.difficulty),
            label: output.difficulty,
            confidence: output.confidence,
            meets_threshold: output.confidence >= 0.5,
        },
        ambiguity: Classification {
            class_id: ambiguity_id(&output.ambiguity),
            label: output.ambiguity,
            confidence: output.ambiguity_confidence,
            meets_threshold: output.ambiguity_confidence >= 0.5,
        },
        domain: Classification {
            class_id: domain_id(&output.domain),
            label: output.domain,
            confidence: output.domain_confidence,
            meets_threshold: output.domain_confidence >= 0.5,
        },
        modality: Classification {
            class_id: modality_id(output.modality.as_ref().unwrap_or(&ModalityLabel::Text)),
            label: output.modality.unwrap_or(ModalityLabel::Text),
            confidence: output.modality_confidence.unwrap_or(0.72),
            meets_threshold: output.modality_confidence.unwrap_or(0.72) >= 0.5,
        },
        safety: Classification {
            class_id: safety_id(output.safety.as_ref().unwrap_or(&SafetyLabel::Safe)),
            label: output.safety.unwrap_or(SafetyLabel::Safe),
            confidence: output.safety_confidence.unwrap_or(0.72),
            meets_threshold: output.safety_confidence.unwrap_or(0.72) >= 0.5,
        },
        cacheability: Classification {
            class_id: cacheability_id(
                output
                    .cacheability
                    .as_ref()
                    .unwrap_or(&CacheabilityLabel::Medium),
            ),
            label: output.cacheability.unwrap_or(CacheabilityLabel::Medium),
            confidence: output.cacheability_confidence.unwrap_or(0.68),
            meets_threshold: output.cacheability_confidence.unwrap_or(0.68) >= 0.5,
        },
        latency_sensitivity: Classification {
            class_id: latency_sensitivity_id(
                output
                    .latency_sensitivity
                    .as_ref()
                    .unwrap_or(&LatencySensitivityLabel::Medium),
            ),
            label: output
                .latency_sensitivity
                .unwrap_or(LatencySensitivityLabel::Medium),
            confidence: output.latency_sensitivity_confidence.unwrap_or(0.68),
            meets_threshold: output.latency_sensitivity_confidence.unwrap_or(0.68) >= 0.5,
        },
        reasoning_depth: Classification {
            class_id: reasoning_depth_id(
                output
                    .reasoning_depth
                    .as_ref()
                    .unwrap_or(&ReasoningDepthLabel::Moderate),
            ),
            label: output
                .reasoning_depth
                .unwrap_or(ReasoningDepthLabel::Moderate),
            confidence: output.reasoning_depth_confidence.unwrap_or(0.68),
            meets_threshold: output.reasoning_depth_confidence.unwrap_or(0.68) >= 0.5,
        },
    })
}

impl JudgeOutput {
    fn validate(&self) -> Result<()> {
        validate_confidence("confidence", self.confidence)?;
        validate_confidence("ambiguity_confidence", self.ambiguity_confidence)?;
        validate_confidence("domain_confidence", self.domain_confidence)?;
        validate_optional_confidence("modality_confidence", self.modality_confidence)?;
        validate_optional_confidence("safety_confidence", self.safety_confidence)?;
        validate_optional_confidence("cacheability_confidence", self.cacheability_confidence)?;
        validate_optional_confidence(
            "latency_sensitivity_confidence",
            self.latency_sensitivity_confidence,
        )?;
        validate_optional_confidence(
            "reasoning_depth_confidence",
            self.reasoning_depth_confidence,
        )?;
        Ok(())
    }
}

fn validate_optional_confidence(name: &str, value: Option<f32>) -> Result<()> {
    if let Some(value) = value {
        validate_confidence(name, value)?;
    }
    Ok(())
}

fn validate_confidence(name: &str, value: f32) -> Result<()> {
    anyhow::ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "LLM judge {name} must be a finite number between 0.0 and 1.0"
    );
    Ok(())
}

fn is_judge_output_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if cause.is::<serde_json::Error>() {
            return true;
        }
        let message = cause.to_string();
        message.contains("LLM judge content did not contain JSON")
            || message.contains("LLM judge response did not include")
            || message.contains("must be a finite number between 0.0 and 1.0")
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (start <= end).then_some(&content[start..=end])
}

fn difficulty_id(label: &DifficultyLabel) -> u8 {
    match label {
        DifficultyLabel::Easy => 0,
        DifficultyLabel::Medium => 1,
        DifficultyLabel::Hard => 2,
        DifficultyLabel::NeedsInfo => 3,
    }
}

fn ambiguity_id(label: &AmbiguityLabel) -> u8 {
    match label {
        AmbiguityLabel::Low => 0,
        AmbiguityLabel::Med => 1,
        AmbiguityLabel::High => 2,
    }
}

fn domain_id(label: &DomainLabel) -> u8 {
    match label {
        DomainLabel::General => 0,
        DomainLabel::Summary => 1,
        DomainLabel::Coding => 2,
        DomainLabel::Design => 3,
        DomainLabel::Data => 4,
    }
}

fn modality_id(label: &ModalityLabel) -> u8 {
    match label {
        ModalityLabel::Text => 0,
        ModalityLabel::Vision => 1,
        ModalityLabel::Audio => 2,
        ModalityLabel::ToolUse => 3,
        ModalityLabel::Multimodal => 4,
    }
}

fn safety_id(label: &SafetyLabel) -> u8 {
    match label {
        SafetyLabel::Safe => 0,
        SafetyLabel::Sensitive => 1,
        SafetyLabel::Unsafe => 2,
    }
}

fn cacheability_id(label: &CacheabilityLabel) -> u8 {
    match label {
        CacheabilityLabel::Low => 0,
        CacheabilityLabel::Medium => 1,
        CacheabilityLabel::High => 2,
    }
}

fn latency_sensitivity_id(label: &LatencySensitivityLabel) -> u8 {
    match label {
        LatencySensitivityLabel::Low => 0,
        LatencySensitivityLabel::Medium => 1,
        LatencySensitivityLabel::High => 2,
    }
}

fn reasoning_depth_id(label: &ReasoningDepthLabel) -> u8 {
    match label {
        ReasoningDepthLabel::Shallow => 0,
        ReasoningDepthLabel::Moderate => 1,
        ReasoningDepthLabel::Deep => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{
            AuthConfig, BudgetConfig, ClassifierAdaptersConfig, ClassifierBackend,
            ClassifierConfig, ClassifierModelAdapterConfig, RouterConfig, RuntimeConfig,
            ScoringConfig, TelemetryConfig,
        },
        types::{ModelConfig, ProviderConfig, ProviderKind, RouterPolicy},
    };
    use axum::{Json, Router, routing::post};
    use tokio::{net::TcpListener, time::sleep};

    #[test]
    fn parses_judge_json_inside_markdown_noise() {
        let parsed = parse_judge_content(
            "```json\n{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88}\n```",
        )
        .unwrap();
        assert_eq!(parsed.difficulty.label, DifficultyLabel::Hard);
        assert_eq!(parsed.ambiguity.label, AmbiguityLabel::Low);
        assert_eq!(parsed.domain.label, DomainLabel::Coding);
    }

    #[test]
    fn parses_judge_json_with_advanced_heads() {
        let parsed = parse_judge_content(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"modality\":\"tool_use\",\"safety\":\"sensitive\",\"cacheability\":\"low\",\"latency_sensitivity\":\"high\",\"reasoning_depth\":\"deep\",\"confidence\":0.91,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88,\"modality_confidence\":0.81,\"safety_confidence\":0.79,\"cacheability_confidence\":0.77,\"latency_sensitivity_confidence\":0.76,\"reasoning_depth_confidence\":0.9}",
        )
        .unwrap();

        assert_eq!(parsed.modality.label, ModalityLabel::ToolUse);
        assert_eq!(parsed.safety.label, SafetyLabel::Sensitive);
        assert_eq!(parsed.cacheability.label, CacheabilityLabel::Low);
        assert_eq!(
            parsed.latency_sensitivity.label,
            LatencySensitivityLabel::High
        );
        assert_eq!(parsed.reasoning_depth.label, ReasoningDepthLabel::Deep);
    }

    #[test]
    fn rejects_judge_json_missing_required_confidence() {
        let error = parse_judge_content(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"domain_confidence\":0.88}",
        )
        .expect_err("missing confidence should fail");

        assert!(error.to_string().contains("missing field"));
    }

    #[test]
    fn rejects_judge_json_with_out_of_range_confidence() {
        let error = parse_judge_content(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":1.7,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88}",
        )
        .expect_err("out of range confidence should fail");

        assert!(error.to_string().contains("between 0.0 and 1.0"));
    }

    #[tokio::test]
    async fn live_judge_success_records_success_metric() {
        let base_url = spawn_judge_server(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88}",
            0,
        )
        .await;
        let classifier = SmartClassifier::new(judge_config(base_url, 500)).unwrap();

        let classifications = classifier
            .classify("Design a distributed Rust router")
            .await;
        let metrics = classifier.judge_metrics();

        assert_eq!(classifications.difficulty.label, DifficultyLabel::Hard);
        assert_eq!(metrics.requests, 1);
        assert_eq!(metrics.successes, 1);
        assert_eq!(metrics.fallbacks, 0);
        assert_eq!(metrics.invalid_outputs, 0);
        assert_eq!(metrics.heuristic_routes, 0);
    }

    #[tokio::test]
    async fn route_llm_backend_uses_configured_adapter_model() {
        let base_url = spawn_judge_server(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"design\",\"confidence\":0.93,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.89}",
            0,
        )
        .await;
        let mut config = judge_config(base_url, 500);
        config.classifier = ClassifierConfig {
            backend: ClassifierBackend::RouteLlm,
            adapters: ClassifierAdaptersConfig {
                route_llm: ClassifierModelAdapterConfig {
                    model: Some("judge-model".to_string()),
                    timeout_ms: 500,
                    prompt_template: Some(
                        "RouteLLM classify this request and return JSON only: {input}".to_string(),
                    ),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let classifier = SmartClassifier::new(config).unwrap();

        let classifications = classifier
            .classify("Design a distributed Rust router")
            .await;
        let metrics = classifier.judge_metrics();

        assert_eq!(classifications.difficulty.label, DifficultyLabel::Hard);
        assert_eq!(classifications.domain.label, DomainLabel::Design);
        assert_eq!(metrics.requests, 1);
        assert_eq!(metrics.successes, 1);
        assert_eq!(metrics.fallbacks, 0);
    }

    #[tokio::test]
    async fn heuristic_classifier_sets_advanced_heads() {
        let classifier = HeuristicClassifier::new(0.5);
        let classifications = classifier
            .classify(
                "ASAP fast instant realtime: analyze this screenshot and design a production architecture with root cause debugging, tradeoffs, and call the JSON schema tool with no API keys",
            )
            .await;

        assert_eq!(classifications.modality.label, ModalityLabel::Multimodal);
        assert_eq!(classifications.safety.label, SafetyLabel::Sensitive);
        assert_eq!(
            classifications.latency_sensitivity.label,
            LatencySensitivityLabel::High
        );
        assert_eq!(
            classifications.reasoning_depth.label,
            ReasoningDepthLabel::Deep
        );
    }

    #[tokio::test]
    async fn heuristic_safety_detects_credential_shapes_without_descriptive_words() {
        let classifier = HeuristicClassifier::new(0.5);

        for input in [
            r#"{"api_key":"sk-live-value"}"#,
            "Authorization: Bearer opaque-value",
            "contact owner@example.com",
        ] {
            let classifications = classifier.classify(input).await;
            assert_eq!(
                classifications.safety.label,
                SafetyLabel::Sensitive,
                "input was not classified as sensitive: {input}"
            );
        }
    }

    #[tokio::test]
    async fn native_ollama_judge_uses_provider_adapter_transform() {
        let base_url = spawn_native_ollama_judge_server().await;
        let classifier = SmartClassifier::new(native_ollama_judge_config(base_url, 500)).unwrap();

        let classifications = classifier
            .classify("Design a distributed Rust router")
            .await;
        let metrics = classifier.judge_metrics();

        assert_eq!(classifications.difficulty.label, DifficultyLabel::Hard);
        assert_eq!(classifications.ambiguity.label, AmbiguityLabel::Low);
        assert_eq!(classifications.domain.label, DomainLabel::Coding);
        assert_eq!(metrics.requests, 1);
        assert_eq!(metrics.successes, 1);
        assert_eq!(metrics.fallbacks, 0);
        assert_eq!(metrics.invalid_outputs, 0);
        assert_eq!(metrics.heuristic_routes, 0);
    }

    #[tokio::test]
    async fn live_judge_invalid_output_falls_back_and_records_invalid_metric() {
        let base_url = spawn_judge_server(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"domain_confidence\":0.88}",
            0,
        )
        .await;
        let classifier = SmartClassifier::new(judge_config(base_url, 500)).unwrap();

        let classifications = classifier.classify("Fix typo in the comment").await;
        let metrics = classifier.judge_metrics();

        assert_eq!(classifications.difficulty.label, DifficultyLabel::Easy);
        assert_eq!(metrics.requests, 1);
        assert_eq!(metrics.successes, 0);
        assert_eq!(metrics.fallbacks, 1);
        assert_eq!(metrics.invalid_outputs, 1);
        assert_eq!(metrics.heuristic_routes, 1);
    }

    #[tokio::test]
    async fn live_judge_timeout_falls_back_without_invalid_output_metric() {
        let base_url = spawn_judge_server(
            "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88}",
            100,
        )
        .await;
        let classifier = SmartClassifier::new(judge_config(base_url, 20)).unwrap();

        let classifications = classifier.classify("Fix typo in the comment").await;
        let metrics = classifier.judge_metrics();

        assert_eq!(classifications.difficulty.label, DifficultyLabel::Easy);
        assert_eq!(metrics.requests, 1);
        assert_eq!(metrics.successes, 0);
        assert_eq!(metrics.fallbacks, 1);
        assert_eq!(metrics.invalid_outputs, 0);
        assert_eq!(metrics.heuristic_routes, 1);
    }

    async fn spawn_judge_server(content: &'static str, delay_ms: u64) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                if delay_ms > 0 {
                    sleep(Duration::from_millis(delay_ms)).await;
                }
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

    async fn spawn_native_ollama_judge_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/api/chat",
            post(|Json(request): Json<Value>| async move {
                assert_eq!(request["model"], "judge-model");
                assert_eq!(request["stream"], false);
                assert_eq!(request["messages"][0]["role"], "user");
                Json(serde_json::json!({
                    "model": "judge-model",
                    "message": {
                        "role": "assistant",
                        "content": "{\"difficulty\":\"hard\",\"ambiguity\":\"low\",\"domain\":\"coding\",\"confidence\":0.91,\"ambiguity_confidence\":0.82,\"domain_confidence\":0.88}"
                    },
                    "done": true,
                    "prompt_eval_count": 17,
                    "eval_count": 9
                }))
            }),
        );

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        format!("http://{addr}")
    }

    fn judge_config(base_url: String, timeout_ms: u64) -> RouterConfig {
        judge_config_with_provider_kind(base_url, timeout_ms, ProviderKind::OpenAiCompatible)
    }

    fn native_ollama_judge_config(base_url: String, timeout_ms: u64) -> RouterConfig {
        let mut config =
            judge_config_with_provider_kind(base_url, timeout_ms, ProviderKind::OllamaNative);
        config.providers[0].chat_path = "/api/chat".to_string();
        config.providers[0].responses_path = None;
        config.providers[0].embeddings_path = None;
        config.providers[0].images_path = None;
        config.providers[0].speech_path = None;
        config.providers[0].audio_transcriptions_path = None;
        config.providers[0].audio_translations_path = None;
        config
    }

    fn judge_config_with_provider_kind(
        base_url: String,
        timeout_ms: u64,
        provider_kind: ProviderKind,
    ) -> RouterConfig {
        RouterConfig {
            bind: "127.0.0.1:0".to_string(),
            default_model: "judge-model".to_string(),
            policy: RouterPolicy::Balanced,
            providers: vec![ProviderConfig {
                name: "judge-provider".to_string(),
                kind: provider_kind,
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
                capability: 0.8,
                cost_per_million_input: 1.0,
                cost_per_million_output: 1.0,
                domains: vec![DomainLabel::Coding],
                context_window: Some(4096),
                capabilities: Default::default(),
                local: true,
            }],
            classifier: ClassifierConfig {
                confidence_threshold: 0.62,
                easy_threshold: 0.28,
                hard_threshold: 0.62,
                llm_judge_model: Some("judge-model".to_string()),
                llm_judge_timeout_ms: timeout_ms,
                ..Default::default()
            },
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
