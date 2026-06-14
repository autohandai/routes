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
    pub embedding: SemanticCacheEmbedding,
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
    embedding: SemanticCacheEmbedding,
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
        query: &SemanticCacheEmbedding,
    ) -> Option<SemanticCacheHit> {
        if !config.enabled || request.prompt.trim().is_empty() {
            return None;
        }
        let now = unix_seconds();
        let mut state = self.state.write().ok()?;
        state.retain(|entry| now.saturating_sub(entry.inserted_unix_seconds) <= config.ttl_seconds);
        state
            .iter()
            .filter(|entry| {
                entry.endpoint == request.endpoint
                    && entry.embedding.model == query.model
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
                embedding_model: entry.embedding.model.clone(),
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
        let Ok(mut state) = self.state.write() else {
            return;
        };
        state.push(SemanticCacheEntry {
            endpoint: write.endpoint,
            model: model.to_string(),
            provider: provider.to_string(),
            embedding: write.embedding,
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
pub struct SemanticCacheEmbedding {
    model: String,
    kind: EmbeddingKind,
}

impl SemanticCacheEmbedding {
    pub fn local_hash(text: &str) -> Option<Self> {
        let embedding = SparseEmbedding::from_text(text);
        (!embedding.values.is_empty()).then_some(Self {
            model: "local-hash".to_string(),
            kind: EmbeddingKind::Sparse(embedding),
        })
    }

    pub fn dense(model: impl Into<String>, values: Vec<f32>) -> Option<Self> {
        let embedding = DenseEmbedding::new(values)?;
        Some(Self {
            model: model.into(),
            kind: EmbeddingKind::Dense(embedding),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn cosine_similarity(&self, other: &Self) -> f32 {
        match (&self.kind, &other.kind) {
            (EmbeddingKind::Sparse(left), EmbeddingKind::Sparse(right)) => {
                left.cosine_similarity(right)
            }
            (EmbeddingKind::Dense(left), EmbeddingKind::Dense(right)) => {
                left.cosine_similarity(right)
            }
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone)]
enum EmbeddingKind {
    Sparse(SparseEmbedding),
    Dense(DenseEmbedding),
}

#[derive(Debug, Clone)]
struct DenseEmbedding {
    values: Vec<f32>,
    norm: f32,
}

impl DenseEmbedding {
    fn new(values: Vec<f32>) -> Option<Self> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        (norm > 0.0).then_some(Self { values, norm })
    }

    fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        let dot = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        (dot / (self.norm * other.norm)).clamp(0.0, 1.0)
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
                embedding: SemanticCacheEmbedding::local_hash(
                    "Explain Rust ownership with examples",
                )
                .unwrap(),
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
                &SemanticCacheEmbedding::local_hash("Explain Rust ownership examples").unwrap(),
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
                embedding: SemanticCacheEmbedding::local_hash("Summarize routing docs").unwrap(),
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
                    &SemanticCacheEmbedding::local_hash("Summarize routing docs").unwrap(),
                )
                .is_none()
        );
    }

    #[test]
    fn semantic_cache_matches_dense_provider_embeddings() {
        let cache = SemanticCache::default();
        let config = SemanticCacheConfig {
            enabled: true,
            embedding_model: "embed-model".to_string(),
            similarity_threshold: 0.95,
            ttl_seconds: 60,
            max_entries: 16,
        };
        cache.record(
            &config,
            SemanticCacheWrite {
                endpoint: SemanticCacheEndpoint::Chat,
                prompt: "Prompt A".to_string(),
                embedding: SemanticCacheEmbedding::dense("embed-model", vec![0.1, 0.2, 0.3])
                    .unwrap(),
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
                    prompt: "Prompt B".to_string(),
                },
                &["model-a".to_string()],
                &SemanticCacheEmbedding::dense("embed-model", vec![0.1, 0.2, 0.3]).unwrap(),
            )
            .expect("matching dense embedding should hit");

        assert_eq!(hit.model, "model-a");
        assert_eq!(hit.embedding_model, "embed-model");
        assert!(hit.similarity >= 0.95);
    }
}
