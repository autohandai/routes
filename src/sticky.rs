use crate::{
    config::{StickyRoutingBackend, StickyRoutingConfig},
    types::ModelConfig,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::warn;

#[derive(Clone)]
pub struct StickyRoutingStore {
    backend: StickyRoutingStoreBackend,
}

#[derive(Clone)]
enum StickyRoutingStoreBackend {
    Memory(Arc<Mutex<HashMap<String, MemoryStickyRoute>>>),
    File(Arc<FileStickyRoutingStore>),
}

#[derive(Clone)]
pub(crate) struct StickyRoute {
    pub(crate) model: String,
    pub(crate) provider: String,
}

#[derive(Clone)]
struct MemoryStickyRoute {
    route: StickyRoute,
    expires_at: Instant,
}

#[derive(Debug)]
struct FileStickyRoutingStore {
    path: PathBuf,
    lock_path: PathBuf,
    lock_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileStickyRoute {
    model: String,
    provider: String,
    expires_at_ms: u64,
}

impl Default for StickyRoutingStore {
    fn default() -> Self {
        Self {
            backend: StickyRoutingStoreBackend::Memory(Default::default()),
        }
    }
}

impl StickyRoutingStore {
    pub fn from_config(config: &StickyRoutingConfig) -> Result<Self> {
        match config.backend {
            StickyRoutingBackend::Memory => Ok(Self::default()),
            StickyRoutingBackend::File => {
                let path = config
                    .file_path
                    .as_deref()
                    .context("sticky_routing.file_path is required when backend is file")?;
                Ok(Self {
                    backend: StickyRoutingStoreBackend::File(Arc::new(
                        FileStickyRoutingStore::new(
                            path,
                            Duration::from_millis(config.lock_timeout_ms),
                        ),
                    )),
                })
            }
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<StickyRoute> {
        match &self.backend {
            StickyRoutingStoreBackend::Memory(routes) => {
                let mut routes = routes.lock().ok()?;
                let route = routes.get(key).cloned()?;
                if route.expires_at <= Instant::now() {
                    routes.remove(key);
                    return None;
                }
                Some(route.route)
            }
            StickyRoutingStoreBackend::File(store) => match store.get(key) {
                Ok(route) => route,
                Err(error) => {
                    warn!(?error, "failed to read sticky routing store");
                    None
                }
            },
        }
    }

    pub(crate) fn record(&self, key: String, model: &ModelConfig, ttl: Duration) {
        match &self.backend {
            StickyRoutingStoreBackend::Memory(routes) => {
                if let Ok(mut routes) = routes.lock() {
                    routes.insert(
                        key,
                        MemoryStickyRoute {
                            route: StickyRoute {
                                model: model.id.clone(),
                                provider: model.provider.clone(),
                            },
                            expires_at: Instant::now() + ttl,
                        },
                    );
                }
            }
            StickyRoutingStoreBackend::File(store) => {
                if let Err(error) = store.record(key, model, ttl) {
                    warn!(?error, "failed to write sticky routing store");
                }
            }
        }
    }
}

impl FileStickyRoutingStore {
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

    fn get(&self, key: &str) -> Result<Option<StickyRoute>> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        let now_ms = now_millis();
        let mut routes = self.read_routes()?;
        purge_expired(&mut routes, now_ms);
        let route = routes.get(key).cloned();
        self.write_routes(&routes)?;
        Ok(route.map(|route| StickyRoute {
            model: route.model,
            provider: route.provider,
        }))
    }

    fn record(&self, key: String, model: &ModelConfig, ttl: Duration) -> Result<()> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        let now_ms = now_millis();
        let mut routes = self.read_routes()?;
        purge_expired(&mut routes, now_ms);
        routes.insert(
            key,
            FileStickyRoute {
                model: model.id.clone(),
                provider: model.provider.clone(),
                expires_at_ms: now_ms.saturating_add(duration_millis(ttl)),
            },
        );
        self.write_routes(&routes)
    }

    fn read_routes(&self) -> Result<HashMap<String, FileStickyRoute>> {
        match fs::read_to_string(&self.path) {
            Ok(raw) => serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse sticky routes {}", self.path.display())),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(HashMap::new()),
            Err(error) => Err(error)
                .with_context(|| format!("failed to read sticky routes {}", self.path.display())),
        }
    }

    fn write_routes(&self, routes: &HashMap<String, FileStickyRoute>) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create sticky routing dir {}", parent.display())
            })?;
        }
        let raw = serde_json::to_vec_pretty(routes).context("failed to serialize sticky routes")?;
        fs::write(&self.path, raw)
            .with_context(|| format!("failed to write sticky routes {}", self.path.display()))
    }
}

fn purge_expired(routes: &mut HashMap<String, FileStickyRoute>, now_ms: u64) {
    routes.retain(|_, route| route.expires_at_ms > now_ms);
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(duration_millis)
        .unwrap_or_default()
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path, timeout: Duration) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create sticky routing lock dir {}",
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
                        "timed out acquiring sticky routing lock {}",
                        path.display()
                    );
                    sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to acquire sticky routing lock {}", path.display())
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
    use super::StickyRoutingStore;
    use crate::{
        config::{StickyRoutingBackend, StickyRoutingConfig},
        types::{DomainLabel, ModelConfig},
    };
    use std::time::Duration;

    #[test]
    fn file_sticky_routing_store_shares_routes_across_instances() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-sticky-{}.json",
            uuid::Uuid::new_v4()
        ));
        let config = StickyRoutingConfig {
            enabled: true,
            ttl_seconds: 60,
            prefer_model: true,
            backend: StickyRoutingBackend::File,
            file_path: Some(path.to_string_lossy().to_string()),
            lock_timeout_ms: 1_000,
        };
        let first = StickyRoutingStore::from_config(&config).unwrap();
        let second = StickyRoutingStore::from_config(&config).unwrap();
        let model = sticky_model("shared-sticky", "shared-provider");

        first.record("v1:session".to_string(), &model, Duration::from_secs(60));
        let route = second.get("v1:session").expect("shared route is visible");

        assert_eq!(route.model, "shared-sticky");
        assert_eq!(route.provider, "shared-provider");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_sticky_routing_store_expires_routes() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-sticky-expired-{}.json",
            uuid::Uuid::new_v4()
        ));
        let config = StickyRoutingConfig {
            enabled: true,
            ttl_seconds: 60,
            prefer_model: true,
            backend: StickyRoutingBackend::File,
            file_path: Some(path.to_string_lossy().to_string()),
            lock_timeout_ms: 1_000,
        };
        let store = StickyRoutingStore::from_config(&config).unwrap();
        let model = sticky_model("expired-sticky", "shared-provider");

        store.record("v1:session".to_string(), &model, Duration::from_millis(0));

        assert!(store.get("v1:session").is_none());
        let _ = std::fs::remove_file(path);
    }

    fn sticky_model(id: &str, provider: &str) -> ModelConfig {
        ModelConfig {
            id: id.to_string(),
            provider: provider.to_string(),
            aliases: vec![],
            capability: 0.5,
            cost_per_million_input: 1.0,
            cost_per_million_output: 1.0,
            domains: vec![DomainLabel::General],
            context_window: Some(4096),
            capabilities: Default::default(),
            local: true,
        }
    }
}
