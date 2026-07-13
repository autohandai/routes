use crate::{
    config::TelemetryConfig,
    jsonl_writer::{AsyncJsonlWriter, JsonlWriterStats},
    types::MultimodelResponse,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone)]
pub struct DecisionLogger {
    writer: AsyncJsonlWriter,
    include_inputs: bool,
}

impl DecisionLogger {
    pub fn new(config: &TelemetryConfig) -> Self {
        Self {
            writer: AsyncJsonlWriter::new(
                config.decision_log_path.as_ref().map(PathBuf::from),
                config.queue_capacity,
                config.max_file_bytes,
                config.retained_files,
            ),
            include_inputs: config.include_inputs,
        }
    }

    pub fn disabled() -> Self {
        Self {
            writer: AsyncJsonlWriter::new(None, 1, 1, 1),
            include_inputs: false,
        }
    }

    pub async fn record_route(&self, source: &str, input: &str, response: &MultimodelResponse) {
        if !self.writer.enabled() {
            return;
        }
        let mut event = json!({
            "timestamp_ms": unix_timestamp_ms(),
            "source": source,
            "input_chars": input.chars().count(),
            "estimated_input_tokens": response.estimated_input_tokens,
            "requested_output_tokens": response.requested_output_tokens,
            "selected_model": response.model,
            "selected_provider": response.provider,
            "difficulty": response.difficulty,
            "confidence": response.confidence,
            "ambiguity": response.ambiguity,
            "ambiguity_confidence": response.ambiguity_confidence,
            "domain": response.domain,
            "domain_confidence": response.domain_confidence,
            "modality": response.modality,
            "modality_confidence": response.modality_confidence,
            "safety": response.safety,
            "safety_confidence": response.safety_confidence,
            "cacheability": response.cacheability,
            "cacheability_confidence": response.cacheability_confidence,
            "latency_sensitivity": response.latency_sensitivity,
            "latency_sensitivity_confidence": response.latency_sensitivity_confidence,
            "reasoning_depth": response.reasoning_depth,
            "reasoning_depth_confidence": response.reasoning_depth_confidence,
            "policy": response.policy,
            "fallback": response.fallback,
            "reason": response.reason,
            "decision_trace": response.decision_trace,
            "candidates": response.candidates,
        });
        if self.include_inputs {
            event["input"] = Value::String(input.to_string());
        }

        if !self.writer.try_write(event) {
            tracing::warn!("decision trace queue is full; dropping record");
        }
    }

    pub async fn flush(&self, timeout: Duration) -> JsonlWriterStats {
        self.writer.flush(timeout).await
    }

    pub fn stats(&self) -> JsonlWriterStats {
        self.writer.stats()
    }
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
    use crate::types::{DifficultyLabel, MultimodelResponse, RouterPolicy};

    #[tokio::test]
    async fn writes_redacted_decision_trace() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-trace-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let logger = DecisionLogger::new(&TelemetryConfig {
            decision_log_path: Some(path.to_string_lossy().to_string()),
            include_inputs: false,
            ..Default::default()
        });
        logger
            .record_route(
                "test",
                "secret prompt",
                &MultimodelResponse {
                    model: "m".to_string(),
                    provider: "p".to_string(),
                    difficulty: DifficultyLabel::Easy,
                    confidence: 0.9,
                    ambiguity: None,
                    ambiguity_confidence: None,
                    domain: None,
                    domain_confidence: None,
                    modality: None,
                    modality_confidence: None,
                    safety: None,
                    safety_confidence: None,
                    cacheability: None,
                    cacheability_confidence: None,
                    latency_sensitivity: None,
                    latency_sensitivity_confidence: None,
                    reasoning_depth: None,
                    reasoning_depth_confidence: None,
                    policy: RouterPolicy::Balanced,
                    reason: "test".to_string(),
                    fallback: false,
                    estimated_input_tokens: 3,
                    requested_output_tokens: 5,
                    decision_trace: None,
                    candidates: vec![],
                },
            )
            .await;
        let stats = logger.flush(Duration::from_secs(2)).await;
        assert_eq!(stats.written, 1);
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"selected_model\":\"m\""));
        assert!(!raw.contains("secret prompt"));
        let _ = std::fs::remove_file(path);
    }
}
