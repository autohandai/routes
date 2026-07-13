use crate::{
    config::ProviderHealthSamplerConfig,
    provider::{ProviderHealth, ProviderHealthStatus},
};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderHealthObservation {
    pub provider: String,
    pub adapter: String,
    pub status: ProviderHealthStatus,
    pub status_code: Option<u16>,
    pub error: Option<String>,
    pub latency_ms: Option<u32>,
    pub health_penalty: f32,
    pub observed_unix_seconds: u64,
    pub fresh_until_unix_seconds: u64,
    pub fresh: bool,
    pub circuit_state: CircuitState,
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone)]
struct ProviderCircuit {
    state: CircuitState,
    consecutive_failures: u32,
    opened_unix_seconds: Option<u64>,
    probe_in_flight: bool,
}

impl Default for ProviderCircuit {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            opened_unix_seconds: None,
            probe_in_flight: false,
        }
    }
}

#[derive(Debug, Default)]
struct ProviderHealthState {
    observations: HashMap<String, ProviderHealthObservation>,
    circuits: HashMap<String, ProviderCircuit>,
}

#[derive(Debug, Clone)]
pub struct ProviderHealthStore {
    state: Arc<RwLock<ProviderHealthState>>,
    observation_ttl_seconds: u64,
    failure_threshold: u32,
    circuit_open_seconds: u64,
}

impl Default for ProviderHealthStore {
    fn default() -> Self {
        Self::new(&ProviderHealthSamplerConfig::default())
    }
}

impl ProviderHealthStore {
    pub fn new(config: &ProviderHealthSamplerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(ProviderHealthState::default())),
            observation_ttl_seconds: config.observation_ttl_ms.div_ceil(1_000),
            failure_threshold: config.circuit_failure_threshold,
            circuit_open_seconds: config.circuit_open_ms.div_ceil(1_000),
        }
    }

    pub fn record(&self, health: ProviderHealth, latency_ms: u32) -> ProviderHealthObservation {
        self.record_at(health, latency_ms, unix_seconds())
    }

    pub(crate) fn record_at(
        &self,
        health: ProviderHealth,
        latency_ms: u32,
        now_unix_seconds: u64,
    ) -> ProviderHealthObservation {
        let base_penalty = match health.status {
            ProviderHealthStatus::Ok => 0.0,
            ProviderHealthStatus::Unknown => 0.35,
            ProviderHealthStatus::Error => 1.0,
        };
        let mut state = self.state.write().expect("provider health lock poisoned");
        let circuit = state.circuits.entry(health.provider.clone()).or_default();
        circuit.probe_in_flight = false;
        match health.status {
            ProviderHealthStatus::Ok => {
                circuit.state = CircuitState::Closed;
                circuit.consecutive_failures = 0;
                circuit.opened_unix_seconds = None;
            }
            ProviderHealthStatus::Unknown | ProviderHealthStatus::Error => {
                circuit.consecutive_failures = circuit.consecutive_failures.saturating_add(1);
                if circuit.state == CircuitState::HalfOpen
                    || circuit.consecutive_failures >= self.failure_threshold
                {
                    circuit.state = CircuitState::Open;
                    circuit.opened_unix_seconds = Some(now_unix_seconds);
                }
            }
        }
        let observation = ProviderHealthObservation {
            provider: health.provider,
            adapter: health.adapter,
            status: health.status,
            status_code: health.status_code,
            error: health.error,
            latency_ms: Some(latency_ms),
            health_penalty: if circuit.state == CircuitState::Open {
                1.0
            } else {
                base_penalty
            },
            observed_unix_seconds: now_unix_seconds,
            fresh_until_unix_seconds: now_unix_seconds.saturating_add(self.observation_ttl_seconds),
            fresh: true,
            circuit_state: circuit.state,
            consecutive_failures: circuit.consecutive_failures,
        };
        state
            .observations
            .insert(observation.provider.clone(), observation.clone());
        observation
    }

    pub fn snapshot(&self) -> Vec<ProviderHealthObservation> {
        self.snapshot_at(unix_seconds())
    }

    fn snapshot_at(&self, now_unix_seconds: u64) -> Vec<ProviderHealthObservation> {
        let Ok(state) = self.state.read() else {
            return Vec::new();
        };
        let mut observations = state
            .observations
            .values()
            .cloned()
            .map(|mut observation| {
                observation.fresh = now_unix_seconds <= observation.fresh_until_unix_seconds;
                if !observation.fresh {
                    observation.health_penalty = 0.0;
                }
                observation
            })
            .collect::<Vec<_>>();
        observations.sort_by(|left, right| left.provider.cmp(&right.provider));
        observations
    }

    pub fn observation(&self, provider: &str) -> Option<ProviderHealthObservation> {
        self.observation_at(provider, unix_seconds())
    }

    pub(crate) fn observation_at(
        &self,
        provider: &str,
        now_unix_seconds: u64,
    ) -> Option<ProviderHealthObservation> {
        self.state
            .read()
            .ok()
            .and_then(|state| state.observations.get(provider).cloned())
            .filter(|observation| now_unix_seconds <= observation.fresh_until_unix_seconds)
    }

    pub fn provider_is_viable(&self, provider: &str) -> bool {
        self.observation(provider)
            .is_none_or(|observation| observation.status != ProviderHealthStatus::Error)
    }

    pub fn should_probe(&self, provider: &str) -> bool {
        self.should_probe_at(provider, unix_seconds())
    }

    pub(crate) fn should_probe_at(&self, provider: &str, now_unix_seconds: u64) -> bool {
        let Ok(mut state) = self.state.write() else {
            return false;
        };
        let circuit = state.circuits.entry(provider.to_string()).or_default();
        match circuit.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let retry_at = circuit
                    .opened_unix_seconds
                    .unwrap_or(now_unix_seconds)
                    .saturating_add(self.circuit_open_seconds);
                if now_unix_seconds < retry_at {
                    return false;
                }
                circuit.state = CircuitState::HalfOpen;
                circuit.probe_in_flight = true;
                true
            }
            CircuitState::HalfOpen if !circuit.probe_in_flight => {
                circuit.probe_in_flight = true;
                true
            }
            CircuitState::HalfOpen => false,
        }
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

    fn config() -> ProviderHealthSamplerConfig {
        ProviderHealthSamplerConfig {
            observation_ttl_ms: 2_000,
            circuit_failure_threshold: 2,
            circuit_open_ms: 3_000,
            ..Default::default()
        }
    }

    fn health(status: ProviderHealthStatus) -> ProviderHealth {
        ProviderHealth {
            provider: "degraded".to_string(),
            adapter: "mock".to_string(),
            status,
            status_code: Some(503),
            error: Some("unavailable".to_string()),
        }
    }

    #[test]
    fn stale_observations_stop_affecting_routing_at_the_configured_ttl() {
        let store = ProviderHealthStore::new(&config());
        store.record_at(health(ProviderHealthStatus::Error), 250, 10);
        assert!(store.observation_at("degraded", 12).is_some());
        assert!(store.observation_at("degraded", 13).is_none());
        assert_eq!(store.snapshot_at(13)[0].health_penalty, 0.0);
        assert!(!store.snapshot_at(13)[0].fresh);
    }

    #[test]
    fn circuit_opens_half_opens_once_and_recovers_after_a_successful_probe() {
        let store = ProviderHealthStore::new(&config());
        store.record_at(health(ProviderHealthStatus::Error), 250, 10);
        let opened = store.record_at(health(ProviderHealthStatus::Error), 250, 11);
        assert_eq!(opened.circuit_state, CircuitState::Open);
        assert!(!store.should_probe_at("degraded", 13));
        assert!(store.should_probe_at("degraded", 14));
        assert!(!store.should_probe_at("degraded", 14));

        let recovered = store.record_at(health(ProviderHealthStatus::Ok), 20, 14);
        assert_eq!(recovered.circuit_state, CircuitState::Closed);
        assert_eq!(recovered.consecutive_failures, 0);
        assert!(store.should_probe_at("degraded", 14));
    }
}
