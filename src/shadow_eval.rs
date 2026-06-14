use crate::config::ShadowEvalConfig;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::hash_map::DefaultHasher,
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::warn;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowEvalEndpoint {
    Chat,
    Responses,
}

#[derive(Debug, Clone)]
pub struct ShadowEvalLogger {
    config: ShadowEvalConfig,
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
}

impl ShadowEvalLogger {
    pub fn new(config: &ShadowEvalConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn disabled() -> Self {
        Self {
            config: ShadowEvalConfig::default(),
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
        let Some(path) = self.config.output_path.as_ref().map(PathBuf::from) else {
            return;
        };
        let selected_body_text = self.body_for_log(input.selected_body);
        let shadow_body_text = input.shadow_body.and_then(|body| self.body_for_log(body));
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
        };
        let value = serde_json::to_value(record).expect("shadow eval record serializes");
        if let Err(error) = append_jsonl(path, value).await {
            warn!(?error, "failed to write shadow eval record");
        }
    }

    fn body_for_log(&self, body: &[u8]) -> Option<String> {
        if !self.config.include_bodies {
            return None;
        }
        let raw = String::from_utf8_lossy(body);
        Some(raw.chars().take(self.config.max_body_chars).collect())
    }
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

async fn append_jsonl(path: PathBuf, value: Value) -> std::io::Result<()> {
    tokio::task::spawn_blocking(move || append_jsonl_blocking(&path, &value))
        .await
        .unwrap_or_else(|error| Err(std::io::Error::other(error)))
}

fn append_jsonl_blocking(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
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
            })
            .await;

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"selected_model\":\"selected\""));
        assert!(raw.contains("\"shadow_model\":\"shadow\""));
        assert!(!raw.contains("selected secret"));
        assert!(!raw.contains("shadow secret"));
        let _ = std::fs::remove_file(path);
    }
}
