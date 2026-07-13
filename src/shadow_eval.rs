use crate::{
    config::ShadowEvalConfig,
    jsonl_writer::{AsyncJsonlWriter, JsonlWriterStats},
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowEvalEndpoint {
    Chat,
    Responses,
}

#[derive(Debug, Clone)]
pub struct ShadowEvalLogger {
    config: ShadowEvalConfig,
    writer: AsyncJsonlWriter,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShadowEvalRecord {
    pub timestamp_ms: u128,
    pub source: String,
    pub endpoint: ShadowEvalEndpoint,
    pub input_chars: usize,
    pub selected_model: String,
    pub selected_provider: String,
    pub shadow_model: String,
    pub shadow_provider: String,
    pub selected_status: u16,
    pub shadow_status: Option<u16>,
    pub selected_latency_ms: u32,
    pub shadow_latency_ms: Option<u32>,
    pub selected_body_chars: usize,
    pub shadow_body_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winner: Option<ShadowEvalWinner>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_score: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShadowEvalWinner {
    Selected,
    Shadow,
    Tie,
}

pub struct ShadowEvalRecordInput<'a> {
    pub source: &'a str,
    pub endpoint: ShadowEvalEndpoint,
    pub input: &'a str,
    pub selected_model: &'a str,
    pub selected_provider: &'a str,
    pub shadow_model: &'a str,
    pub shadow_provider: &'a str,
    pub selected_status: u16,
    pub shadow_status: Option<u16>,
    pub selected_latency_ms: u32,
    pub shadow_latency_ms: Option<u32>,
    pub selected_body: &'a [u8],
    pub shadow_body: Option<&'a [u8]>,
    pub shadow_error: Option<String>,
    pub judgement: Option<ShadowEvalJudgement>,
}

impl ShadowEvalLogger {
    pub fn new(config: &ShadowEvalConfig) -> Self {
        Self {
            config: config.clone(),
            writer: AsyncJsonlWriter::new(
                config.output_path.as_ref().map(PathBuf::from),
                config.writer_queue_capacity,
                config.max_file_bytes,
                config.retained_files,
            ),
        }
    }

    pub fn disabled() -> Self {
        Self {
            config: ShadowEvalConfig::default(),
            writer: AsyncJsonlWriter::new(None, 1, 1, 1),
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.output_path.is_some()
    }

    pub fn should_sample(&self, source: &str, input: &str) -> bool {
        if !self.enabled() || self.config.sample_rate <= 0.0 {
            return false;
        }
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        normalized_hash(source, input) < self.config.sample_rate
    }

    pub async fn record(&self, input: ShadowEvalRecordInput<'_>) {
        if !self.enabled() {
            return;
        }
        let selected_body_text = self.body_for_log(input.selected_body);
        let shadow_body_text = input.shadow_body.and_then(|body| self.body_for_log(body));
        let judgement = input
            .judgement
            .clone()
            .or_else(|| self.config.judge.enabled.then(|| judge_shadow_eval(&input)));
        let record = ShadowEvalRecord {
            timestamp_ms: unix_timestamp_ms(),
            source: input.source.to_string(),
            endpoint: input.endpoint,
            input_chars: input.input.chars().count(),
            selected_model: input.selected_model.to_string(),
            selected_provider: input.selected_provider.to_string(),
            shadow_model: input.shadow_model.to_string(),
            shadow_provider: input.shadow_provider.to_string(),
            selected_status: input.selected_status,
            shadow_status: input.shadow_status,
            selected_latency_ms: input.selected_latency_ms,
            shadow_latency_ms: input.shadow_latency_ms,
            selected_body_chars: body_chars(input.selected_body),
            shadow_body_chars: input.shadow_body.map(body_chars),
            selected_body: selected_body_text,
            shadow_body: shadow_body_text,
            shadow_error: input.shadow_error,
            winner: judgement.as_ref().map(|judgement| judgement.winner),
            judge_reason: judgement.as_ref().map(|judgement| judgement.reason.clone()),
            selected_score: judgement.as_ref().map(|judgement| judgement.selected_score),
            shadow_score: judgement.as_ref().map(|judgement| judgement.shadow_score),
        };
        let value = serde_json::to_value(record).expect("shadow eval record serializes");
        if !self.writer.try_write(value) {
            tracing::warn!("shadow eval writer queue is full; dropping record");
        }
    }

    pub async fn flush(&self, timeout: Duration) -> JsonlWriterStats {
        self.writer.flush(timeout).await
    }

    pub fn stats(&self) -> JsonlWriterStats {
        self.writer.stats()
    }

    fn body_for_log(&self, body: &[u8]) -> Option<String> {
        if !self.config.include_bodies {
            return None;
        }
        let raw = String::from_utf8_lossy(body);
        Some(raw.chars().take(self.config.max_body_chars).collect())
    }
}

#[derive(Debug, Clone)]
pub struct ShadowEvalJudgement {
    pub winner: ShadowEvalWinner,
    pub reason: String,
    pub selected_score: f32,
    pub shadow_score: f32,
}

pub fn judge_shadow_eval(input: &ShadowEvalRecordInput<'_>) -> ShadowEvalJudgement {
    let selected_score = response_score(
        Some(input.selected_status),
        Some(input.selected_body),
        Some(input.selected_latency_ms),
    );
    let shadow_score = response_score(
        input.shadow_status,
        input.shadow_body,
        input.shadow_latency_ms,
    );
    let delta = selected_score - shadow_score;
    let (winner, reason) = if input.shadow_error.is_some() {
        (
            ShadowEvalWinner::Selected,
            "selected: shadow request failed",
        )
    } else if is_success(input.selected_status) && !input.shadow_status.is_some_and(is_success) {
        (
            ShadowEvalWinner::Selected,
            "selected: only selected response succeeded",
        )
    } else if input.shadow_status.is_some_and(is_success) && !is_success(input.selected_status) {
        (
            ShadowEvalWinner::Shadow,
            "shadow: only shadow response succeeded",
        )
    } else if delta.abs() <= 0.05 {
        (ShadowEvalWinner::Tie, "tie: heuristic scores are close")
    } else if delta > 0.0 {
        (
            ShadowEvalWinner::Selected,
            "selected: higher heuristic status/content/latency score",
        )
    } else {
        (
            ShadowEvalWinner::Shadow,
            "shadow: higher heuristic status/content/latency score",
        )
    };

    ShadowEvalJudgement {
        winner,
        reason: reason.to_string(),
        selected_score,
        shadow_score,
    }
}

#[derive(Debug, Deserialize)]
struct LlmShadowEvalJudgement {
    winner: ShadowEvalWinner,
    #[serde(default)]
    reason: String,
    selected_score: Option<f32>,
    shadow_score: Option<f32>,
}

pub fn parse_llm_shadow_eval_judgement(content: &str) -> Result<ShadowEvalJudgement> {
    let json =
        extract_json_object(content).context("LLM shadow judge content did not contain JSON")?;
    let parsed: LlmShadowEvalJudgement =
        serde_json::from_str(json).context("LLM shadow judge JSON was invalid")?;
    let selected_score = validate_llm_score("selected_score", parsed.selected_score)?;
    let shadow_score = validate_llm_score("shadow_score", parsed.shadow_score)?;
    Ok(ShadowEvalJudgement {
        winner: parsed.winner,
        reason: if parsed.reason.trim().is_empty() {
            "llm_judge: no reason provided".to_string()
        } else {
            format!("llm_judge: {}", parsed.reason.trim())
        },
        selected_score,
        shadow_score,
    })
}

fn validate_llm_score(name: &str, score: Option<f32>) -> Result<f32> {
    let score = score.unwrap_or(0.5);
    anyhow::ensure!(
        score.is_finite() && (0.0..=1.0).contains(&score),
        "LLM shadow judge {name} must be between 0.0 and 1.0"
    );
    Ok(score)
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end > start).then_some(&content[start..=end])
}

fn response_score(status: Option<u16>, body: Option<&[u8]>, latency_ms: Option<u32>) -> f32 {
    let status_score = match status {
        Some(status) if is_success(status) => 1.0,
        Some(status) if status < 500 => 0.35,
        Some(_) => 0.15,
        None => 0.0,
    };
    let content_score = body.map(content_quality_score).unwrap_or(0.0);
    let latency_score = latency_ms.map(latency_quality_score).unwrap_or(0.0);
    (status_score * 0.70) + (content_score * 0.22) + (latency_score * 0.08)
}

fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

fn content_quality_score(body: &[u8]) -> f32 {
    let chars = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| extracted_text_chars(&value))
        .unwrap_or_else(|| String::from_utf8_lossy(body).trim().chars().count());
    ((chars as f32) / 200.0).clamp(0.0, 1.0)
}

fn extracted_text_chars(value: &Value) -> Option<usize> {
    let count = match value {
        Value::String(text) => text.trim().chars().count(),
        Value::Array(items) => items.iter().filter_map(extracted_text_chars).sum(),
        Value::Object(object) => {
            for key in ["content", "text", "output_text", "message"] {
                if let Some(count) = object.get(key).and_then(extracted_text_chars) {
                    return Some(count);
                }
            }
            object.values().filter_map(extracted_text_chars).sum()
        }
        _ => 0,
    };
    (count > 0).then_some(count)
}

fn latency_quality_score(latency_ms: u32) -> f32 {
    (1_000.0 / (latency_ms as f32 + 1_000.0)).clamp(0.0, 1.0)
}

fn body_chars(body: &[u8]) -> usize {
    String::from_utf8_lossy(body).chars().count()
}

fn normalized_hash(source: &str, input: &str) -> f32 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    input.hash(&mut hasher);
    let value = hasher.finish();
    (value as f64 / u64::MAX as f64) as f32
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_redacted_shadow_eval_record() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-shadow-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let logger = ShadowEvalLogger::new(&ShadowEvalConfig {
            enabled: true,
            sample_rate: 1.0,
            output_path: Some(path.to_string_lossy().to_string()),
            include_bodies: false,
            max_body_chars: 128,
            judge: Default::default(),
            ..Default::default()
        });

        logger
            .record(ShadowEvalRecordInput {
                source: "chat.auto",
                endpoint: ShadowEvalEndpoint::Chat,
                input: "secret prompt",
                selected_model: "selected",
                selected_provider: "primary",
                shadow_model: "shadow",
                shadow_provider: "secondary",
                selected_status: 200,
                shadow_status: Some(200),
                selected_latency_ms: 10,
                shadow_latency_ms: Some(12),
                selected_body: br#"{"content":"selected secret"}"#,
                shadow_body: Some(br#"{"content":"shadow secret"}"#),
                shadow_error: None,
                judgement: None,
            })
            .await;
        logger.flush(Duration::from_secs(2)).await;

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"selected_model\":\"selected\""));
        assert!(raw.contains("\"shadow_model\":\"shadow\""));
        assert!(raw.contains("\"winner\":\"tie\""));
        assert!(raw.contains("\"judge_reason\":\"tie: heuristic scores are close\""));
        assert!(!raw.contains("selected secret"));
        assert!(!raw.contains("shadow secret"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn shadow_eval_judge_prefers_successful_shadow_response() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-shadow-judge-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let logger = ShadowEvalLogger::new(&ShadowEvalConfig {
            enabled: true,
            sample_rate: 1.0,
            output_path: Some(path.to_string_lossy().to_string()),
            include_bodies: true,
            max_body_chars: 512,
            judge: Default::default(),
            ..Default::default()
        });

        logger
            .record(ShadowEvalRecordInput {
                source: "chat.auto",
                endpoint: ShadowEvalEndpoint::Chat,
                input: "compare answers",
                selected_model: "selected",
                selected_provider: "primary",
                shadow_model: "shadow",
                shadow_provider: "secondary",
                selected_status: 503,
                shadow_status: Some(200),
                selected_latency_ms: 8,
                shadow_latency_ms: Some(20),
                selected_body: br#"{"error":"busy"}"#,
                shadow_body: Some(
                    br#"{"choices":[{"message":{"content":"complete useful answer"}}]}"#,
                ),
                shadow_error: None,
                judgement: None,
            })
            .await;
        logger.flush(Duration::from_secs(2)).await;

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"winner\":\"shadow\""));
        assert!(raw.contains("\"judge_reason\":\"shadow: only shadow response succeeded\""));
        assert!(raw.contains("\"selected_score\""));
        assert!(raw.contains("\"shadow_score\""));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn parses_llm_shadow_eval_judgement_from_json_noise() {
        let judgement = parse_llm_shadow_eval_judgement(
            "Result:\n{\"winner\":\"shadow\",\"reason\":\"more complete\",\"selected_score\":0.4,\"shadow_score\":0.9}",
        )
        .unwrap();

        assert_eq!(judgement.winner, ShadowEvalWinner::Shadow);
        assert_eq!(judgement.reason, "llm_judge: more complete");
        assert_eq!(judgement.selected_score, 0.4);
        assert_eq!(judgement.shadow_score, 0.9);
    }

    #[test]
    fn rejects_out_of_range_llm_shadow_eval_score() {
        let error = parse_llm_shadow_eval_judgement(
            "{\"winner\":\"selected\",\"reason\":\"bad score\",\"selected_score\":1.5,\"shadow_score\":0.2}",
        )
        .expect_err("invalid score rejected");

        assert!(error.to_string().contains("selected_score"));
    }
}
