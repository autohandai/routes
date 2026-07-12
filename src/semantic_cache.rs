use crate::config::{SemanticCacheBackend, SemanticCacheConfig};
use anyhow::{Context, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    fs::{self, OpenOptions},
    hash::{Hash, Hasher},
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticCacheEntry {
    endpoint: SemanticCacheEndpoint,
    model: String,
    provider: String,
    embedding: SemanticCacheEmbedding,
    status_code: u16,
    content_type: Option<String>,
    #[serde(with = "bytes_serde")]
    body: Bytes,
    inserted_unix_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct SemanticCache {
    backend: SemanticCacheStoreBackend,
}

#[derive(Debug, Clone)]
enum SemanticCacheStoreBackend {
    Memory(Arc<RwLock<Vec<SemanticCacheEntry>>>),
    File(Arc<FileSemanticCacheStore>),
}

#[derive(Debug)]
struct FileSemanticCacheStore {
    path: PathBuf,
    lock_path: PathBuf,
    lock_timeout: Duration,
}

impl Default for SemanticCache {
    fn default() -> Self {
        Self {
            backend: SemanticCacheStoreBackend::Memory(Default::default()),
        }
    }
}

impl SemanticCache {
    pub fn from_config(config: &SemanticCacheConfig) -> Result<Self> {
        match config.backend {
            SemanticCacheBackend::Memory => Ok(Self::default()),
            SemanticCacheBackend::File => {
                let path = config
                    .file_path
                    .as_deref()
                    .context("cache.semantic.file_path is required when backend is file")?;
                Ok(Self {
                    backend: SemanticCacheStoreBackend::File(Arc::new(
                        FileSemanticCacheStore::new(
                            path,
                            Duration::from_millis(config.lock_timeout_ms),
                        ),
                    )),
                })
            }
        }
    }

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
        match &self.backend {
            SemanticCacheStoreBackend::Memory(state) => {
                let mut state = state.write().ok()?;
                prune_entries(&mut state, config, now);
                best_hit(config, request, candidate_models, query, &state)
            }
            SemanticCacheStoreBackend::File(store) => {
                match store.lookup(config, request, candidate_models, query, now) {
                    Ok(hit) => hit,
                    Err(error) => {
                        warn!(?error, "failed to read semantic cache store");
                        None
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        let entry = SemanticCacheEntry {
            endpoint: write.endpoint,
            model: model.to_string(),
            provider: provider.to_string(),
            embedding: write.embedding,
            status_code,
            content_type,
            body,
            inserted_unix_seconds: unix_seconds(),
        };
        match &self.backend {
            SemanticCacheStoreBackend::Memory(state) => {
                let Ok(mut state) = state.write() else {
                    return;
                };
                state.push(entry);
                prune_entries(&mut state, config, unix_seconds());
            }
            SemanticCacheStoreBackend::File(store) => {
                if let Err(error) = store.record(config, entry) {
                    warn!(?error, "failed to write semantic cache store");
                }
            }
        }
    }

    pub fn snapshot(&self) -> SemanticCacheSnapshot {
        let entries = match &self.backend {
            SemanticCacheStoreBackend::Memory(state) => {
                state.read().map(|state| state.len()).unwrap_or_default()
            }
            SemanticCacheStoreBackend::File(store) => store.snapshot().unwrap_or_default(),
        };
        SemanticCacheSnapshot { entries }
    }
}

impl FileSemanticCacheStore {
    fn new(path: impl AsRef<Path>, lock_timeout: Duration) -> Self {
        let path = path.as_ref().to_path_buf();
        let lock_path = path.with_extension(format!(
            "{}lock",
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| format!("{extension}."))
                .unwrap_or_default()
        ));
        Self {
            path,
            lock_path,
            lock_timeout,
        }
    }

    fn lookup(
        &self,
        config: &SemanticCacheConfig,
        request: &SemanticCacheRequest,
        candidate_models: &[String],
        query: &SemanticCacheEmbedding,
        now: u64,
    ) -> Result<Option<SemanticCacheHit>> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        let mut entries = self.read_entries()?;
        prune_entries(&mut entries, config, now);
        let hit = best_hit(config, request, candidate_models, query, &entries);
        self.write_entries(&entries)?;
        Ok(hit)
    }

    fn record(&self, config: &SemanticCacheConfig, entry: SemanticCacheEntry) -> Result<()> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        let mut entries = self.read_entries()?;
        entries.push(entry);
        prune_entries(&mut entries, config, unix_seconds());
        self.write_entries(&entries)
    }

    fn snapshot(&self) -> Result<usize> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        Ok(self.read_entries()?.len())
    }

    fn read_entries(&self) -> Result<Vec<SemanticCacheEntry>> {
        match fs::read_to_string(&self.path) {
            Ok(raw) => serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse semantic cache {}", self.path.display())),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error)
                .with_context(|| format!("failed to read semantic cache {}", self.path.display())),
        }
    }

    fn write_entries(&self, entries: &[SemanticCacheEntry]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create semantic cache dir {}", parent.display())
            })?;
        }
        let raw =
            serde_json::to_vec_pretty(entries).context("failed to serialize semantic cache")?;
        fs::write(&self.path, raw)
            .with_context(|| format!("failed to write semantic cache {}", self.path.display()))
    }
}

fn prune_entries(entries: &mut Vec<SemanticCacheEntry>, config: &SemanticCacheConfig, now: u64) {
    entries.retain(|entry| now.saturating_sub(entry.inserted_unix_seconds) < config.ttl_seconds);
    if entries.len() > config.max_entries {
        let overflow = entries.len() - config.max_entries;
        entries.drain(0..overflow);
    }
}

fn best_hit(
    config: &SemanticCacheConfig,
    request: &SemanticCacheRequest,
    candidate_models: &[String],
    query: &SemanticCacheEmbedding,
    entries: &[SemanticCacheEntry],
) -> Option<SemanticCacheHit> {
    entries
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EmbeddingKind {
    Sparse(SparseEmbedding),
    Dense(DenseEmbedding),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseEmbedding {
    #[serde(with = "sparse_values_serde")]
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

mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Ok(Bytes::from(bytes))
    }
}

mod sparse_values_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(values: &HashMap<u64, f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let pairs = values
            .iter()
            .map(|(token, value)| (*token, *value))
            .collect::<Vec<_>>();
        pairs.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<u64, f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let pairs = Vec::<(u64, f32)>::deserialize(deserializer)?;
        Ok(pairs.into_iter().collect())
    }
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path, timeout: Duration) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create semantic cache lock dir {}",
                    parent.display()
                )
            })?;
        }
        let start = Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(_) => {
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                    anyhow::ensure!(
                        start.elapsed() < timeout,
                        "timed out acquiring semantic cache lock {}",
                        path.display()
                    );
                    sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to acquire semantic cache lock {}", path.display())
                    });
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SemanticCacheBackend;

    #[test]
    fn semantic_cache_matches_similar_prompt_above_threshold() {
        let cache = SemanticCache::default();
        let config = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.70,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
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
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
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
        std::thread::sleep(Duration::from_millis(1100));
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
            backend: SemanticCacheBackend::Memory,
            file_path: None,
            lock_timeout_ms: 1_000,
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

    #[test]
    fn file_semantic_cache_shares_hits_across_instances() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-semantic-cache-{}.json",
            uuid::Uuid::new_v4()
        ));
        let config = SemanticCacheConfig {
            enabled: true,
            embedding_model: "local-hash".to_string(),
            similarity_threshold: 0.70,
            ttl_seconds: 60,
            max_entries: 16,
            backend: SemanticCacheBackend::File,
            file_path: Some(path.to_string_lossy().to_string()),
            lock_timeout_ms: 1_000,
        };
        let first = SemanticCache::from_config(&config).unwrap();
        let second = SemanticCache::from_config(&config).unwrap();

        first.record(
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
        let hit = second
            .lookup(
                &config,
                &SemanticCacheRequest {
                    endpoint: SemanticCacheEndpoint::Chat,
                    prompt: "Explain Rust ownership examples".to_string(),
                },
                &["model-a".to_string()],
                &SemanticCacheEmbedding::local_hash("Explain Rust ownership examples").unwrap(),
            )
            .expect("shared file cache should hit");

        assert_eq!(hit.model, "model-a");
        assert_eq!(hit.body, Bytes::from_static(b"{\"cached\":true}"));
        assert_eq!(second.snapshot().entries, 1);
        let _ = std::fs::remove_file(path);
    }
}
