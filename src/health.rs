use crate::provider::{ProviderHealth, ProviderHealthStatus};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Serialize)]
pub struct ProviderHealthObservation {
    pub provider: String,
    pub status: ProviderHealthStatus,
    pub status_code: Option<u16>,
    pub error: Option<String>,
    pub latency_ms: Option<u32>,
    pub health_penalty: f32,
    pub observed_unix_seconds: u64,
}

#[derive(Debug, Default)]
struct ProviderHealthState {
    observations: HashMap<String, ProviderHealthObservation>,
}

#[derive(Debug, Clone, Default)]
pub struct ProviderHealthStore {
    state: Arc<RwLock<ProviderHealthState>>,
}

impl ProviderHealthStore {
    pub fn record(&self, health: ProviderHealth, latency_ms: u32) -> ProviderHealthObservation {
        let health_penalty = match health.status {
            ProviderHealthStatus::Ok => 0.0,
            ProviderHealthStatus::Unknown => 0.35,
            ProviderHealthStatus::Error => 1.0,
        };
        let observation = ProviderHealthObservation {
            provider: health.provider,
            status: health.status,
            status_code: health.status_code,
            error: health.error,
            latency_ms: Some(latency_ms),
            health_penalty,
            observed_unix_seconds: unix_seconds(),
        };
        if let Ok(mut state) = self.state.write() {
            state
                .observations
                .insert(observation.provider.clone(), observation.clone());
        }
        observation
    }

    pub fn snapshot(&self) -> Vec<ProviderHealthObservation> {
        let Ok(state) = self.state.read() else {
            return Vec::new();
        };
        let mut observations = state.observations.values().cloned().collect::<Vec<_>>();
        observations.sort_by(|left, right| left.provider.cmp(&right.provider));
        observations
    }

    pub fn observation(&self, provider: &str) -> Option<ProviderHealthObservation> {
        self.state
            .read()
            .ok()
            .and_then(|state| state.observations.get(provider).cloned())
    }
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
    fn records_error_observations_as_full_health_penalty() {
        let store = ProviderHealthStore::default();
        let observation = store.record(
            ProviderHealth {
                provider: "degraded".to_string(),
                adapter: "mock".to_string(),
                status: ProviderHealthStatus::Error,
                status_code: Some(503),
                error: Some("unavailable".to_string()),
            },
            250,
        );

        assert_eq!(observation.provider, "degraded");
        assert_eq!(observation.latency_ms, Some(250));
        assert_eq!(observation.health_penalty, 1.0);
        assert_eq!(
            store
                .observation("degraded")
                .expect("observation stored")
                .status_code,
            Some(503)
        );
    }
}
