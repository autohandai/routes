use crate::config::SemanticCacheConfig;
use bytes::Bytes;
use serde::Serialize;
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticCacheEndpoint {
    Chat,
    Responses,
}

impl SemanticCacheEndpoint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticCacheRequest {
    pub endpoint: SemanticCacheEndpoint,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct SemanticCacheWrite {
    pub endpoint: SemanticCacheEndpoint,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct SemanticCacheHit {
    pub model: String,
    pub provider: String,
    pub status_code: u16,
    pub content_type: Option<String>,
    pub body: Bytes,
    pub similarity: f32,
    pub embedding_model: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticCacheSnapshot {
    pub entries: usize,
}

#[derive(Debug, Clone)]
struct SemanticCacheEntry {
    endpoint: SemanticCacheEndpoint,
    model: String,
    provider: String,
    embedding_model: String,
    embedding: SparseEmbedding,
    status_code: u16,
    content_type: Option<String>,
    body: Bytes,
    inserted_unix_seconds: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticCache {
    state: Arc<RwLock<Vec<SemanticCacheEntry>>>,
}

impl SemanticCache {
    pub fn lookup(
        &self,
        config: &SemanticCacheConfig,
        request: &SemanticCacheRequest,
        candidate_models: &[String],
    ) -> Option<SemanticCacheHit> {
        if !config.enabled || request.prompt.trim().is_empty() {
            return None;
        }
        let now = unix_seconds();
        let query = SparseEmbedding::from_text(&request.prompt);
        if query.values.is_empty() {
            return None;
        }
        let mut state = self.state.write().ok()?;
        state.retain(|entry| now.saturating_sub(entry.inserted_unix_seconds) <= config.ttl_seconds);
        state
            .iter()
            .filter(|entry| {
                entry.endpoint == request.endpoint
                    && entry.embedding_model == config.embedding_model
                    && candidate_models.contains(&entry.model)
            })
            .filter_map(|entry| {
                let similarity = query.cosine_similarity(&entry.embedding);
                (similarity >= config.similarity_threshold).then_some((entry, similarity))
            })
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(entry, similarity)| SemanticCacheHit {
                model: entry.model.clone(),
                provider: entry.provider.clone(),
                status_code: entry.status_code,
                content_type: entry.content_type.clone(),
                body: entry.body.clone(),
                similarity,
                embedding_model: entry.embedding_model.clone(),
            })
    }

    pub fn record(
        &self,
        config: &SemanticCacheConfig,
        write: SemanticCacheWrite,
        model: &str,
        provider: &str,
        status_code: u16,
        content_type: Option<String>,
        body: Bytes,
    ) {
        if !config.enabled || write.prompt.trim().is_empty() || body.is_empty() {
            return;
        }
        let embedding = SparseEmbedding::from_text(&write.prompt);
        if embedding.values.is_empty() {
            return;
        }
        let Ok(mut state) = self.state.write() else {
            return;
        };
        state.push(SemanticCacheEntry {
            endpoint: write.endpoint,
            model: model.to_string(),
            provider: provider.to_string(),
            embedding_model: config.embedding_model.clone(),
            embedding,
            status_code,
            content_type,
            body,
            inserted_unix_seconds: unix_seconds(),
        });
        if state.len() > config.max_entries {
            let overflow = state.len() - config.max_entries;
            state.drain(0..overflow);
        }
    }

    pub fn snapshot(&self) -> SemanticCacheSnapshot {
        let entries = self
            .state
            .read()
            .map(|state| state.len())
            .unwrap_or_default();
        SemanticCacheSnapshot { entries }
    }
}

#[derive(Debug, Clone)]
struct SparseEmbedding {
    values: HashMap<u64, f32>,
    norm: f32,
}

impl SparseEmbedding {
    fn from_text(text: &str) -> Self {
        let mut values = HashMap::<u64, f32>::new();
        for token in tokenize(text) {
            *values.entry(hash_token(&token)).or_insert(0.0) += 1.0;
        }
        let norm = values
            .values()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        Self { values, norm }
    }

    fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.norm == 0.0 || other.norm == 0.0 {
            return 0.0;
        }
        let dot = self
            .values
            .iter()
            .filter_map(|(token, left)| other.values.get(token).map(|right| left * right))
            .sum::<f32>();
        (dot / (self.norm * other.norm)).clamp(0.0, 1.0)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_lowercase();
            (token.len() >= 2).then_some(token)
        })
        .collect()
}

fn hash_token(token: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_cache_matches_similar_prompt_above_threshold() {
        let cache = SemanticCache::default();
        let config = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.70,
            ttl_seconds: 60,
            max_entries: 16,
        };
        cache.record(
            &config,
            SemanticCacheWrite {
                endpoint: SemanticCacheEndpoint::Chat,
                prompt: "Explain Rust ownership with examples".to_string(),
            },
            "model-a",
            "provider-a",
            200,
            Some("application/json".to_string()),
            Bytes::from_static(b"{\"cached\":true}"),
        );

        let hit = cache
            .lookup(
                &config,
                &SemanticCacheRequest {
                    endpoint: SemanticCacheEndpoint::Chat,
                    prompt: "Explain Rust ownership examples".to_string(),
                },
                &["model-a".to_string()],
            )
            .expect("similar prompt should hit");

        assert_eq!(hit.model, "model-a");
        assert!(hit.similarity >= 0.70);
    }

    #[test]
    fn semantic_cache_respects_ttl() {
        let cache = SemanticCache::default();
        let config = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.70,
            ttl_seconds: 1,
            max_entries: 16,
        };
        cache.record(
            &config,
            SemanticCacheWrite {
                endpoint: SemanticCacheEndpoint::Responses,
                prompt: "Summarize routing docs".to_string(),
            },
            "model-a",
            "provider-a",
            200,
            None,
            Bytes::from_static(b"{}"),
        );
        {
            let mut state = cache.state.write().unwrap();
            state[0].inserted_unix_seconds = state[0].inserted_unix_seconds.saturating_sub(5);
        }

        assert!(
            cache
                .lookup(
                    &config,
                    &SemanticCacheRequest {
                        endpoint: SemanticCacheEndpoint::Responses,
                        prompt: "Summarize routing docs".to_string(),
                    },
                    &["model-a".to_string()],
                )
                .is_none()
        );
    }
}
