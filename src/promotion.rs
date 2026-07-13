use crate::{config::RouterConfig, conformance::config_fingerprint, types::ModelEndpoint};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPromotionGateReport {
    pub schema_version: u32,
    pub evaluated_unix_seconds: u64,
    pub artifact_generated_unix_seconds: u64,
    pub artifact_age_seconds: u64,
    pub max_age_seconds: u64,
    pub config_fnv1a_64: String,
    pub require_reported_versions: bool,
    pub configured_pairs: usize,
    pub advertised_checks: usize,
    pub passed_checks: usize,
    pub failed_checks: usize,
    pub skipped_unadvertised_checks: usize,
    pub pass: bool,
    pub failures: Vec<String>,
    pub reports: Vec<PromotionPairReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPairReport {
    pub provider: String,
    pub model: String,
    pub provider_version: Option<String>,
    pub model_version: Option<String>,
    pub checks: Vec<PromotionCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCheck {
    pub kind: String,
    pub name: String,
    pub advertised: bool,
    pub status: PromotionCheckStatus,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotionCheckStatus {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Deserialize)]
struct MatrixArtifact {
    schema_version: u32,
    generated_unix_seconds: u64,
    config_fnv1a_64: String,
    reports: Vec<PairArtifact>,
}

#[derive(Debug, Deserialize)]
struct PairArtifact {
    provider: String,
    model: String,
    pass: bool,
    provider_version: ArtifactVersion,
    model_version: ArtifactVersion,
    chat: ChatArtifact,
    features: Vec<FeatureArtifact>,
    endpoints: Vec<EndpointArtifact>,
}

#[derive(Debug, Deserialize)]
struct ArtifactVersion {
    value: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatArtifact {
    configured: bool,
    skip_reason: Option<String>,
    status: u16,
    openai_chat_shape: bool,
    response_model_matches: bool,
    assistant_content_present: bool,
    usage_present: bool,
    negative_schema_rejected: bool,
}

#[derive(Debug, Deserialize)]
struct FeatureArtifact {
    feature: String,
    declared: bool,
    attempted: bool,
    pass: bool,
    skip_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EndpointArtifact {
    endpoint: String,
    configured: bool,
    pass: bool,
    positive_schema_valid: bool,
    negative_schema_rejected: bool,
    skip_reason: Option<String>,
}

pub fn evaluate_provider_promotion_gate(
    config: &RouterConfig,
    artifact_path: &Path,
    evaluated_unix_seconds: u64,
    max_age_seconds: u64,
    require_reported_versions: bool,
) -> Result<ProviderPromotionGateReport> {
    anyhow::ensure!(max_age_seconds > 0, "promotion max age must be non-zero");
    let raw = fs::read(artifact_path).with_context(|| {
        format!(
            "failed to read conformance artifact {}",
            artifact_path.display()
        )
    })?;
    let artifact = serde_json::from_slice::<MatrixArtifact>(&raw).with_context(|| {
        format!(
            "failed to parse conformance artifact {}",
            artifact_path.display()
        )
    })?;
    anyhow::ensure!(
        artifact.schema_version == 2,
        "provider promotion requires conformance schema version 2"
    );
    let artifact_age_seconds = evaluated_unix_seconds
        .checked_sub(artifact.generated_unix_seconds)
        .unwrap_or(u64::MAX);
    let expected_fingerprint = config_fingerprint(config)?;
    let mut failures = Vec::new();
    if artifact_age_seconds > max_age_seconds {
        failures.push(format!(
            "conformance artifact is {artifact_age_seconds}s old, above maximum {max_age_seconds}s"
        ));
    }
    if artifact.config_fnv1a_64 != expected_fingerprint {
        failures.push(format!(
            "conformance config fingerprint {} does not match current {}",
            artifact.config_fnv1a_64, expected_fingerprint
        ));
    }

    let mut artifact_pairs = HashMap::new();
    for pair in artifact.reports {
        let key = (pair.provider.clone(), pair.model.clone());
        if artifact_pairs.insert(key.clone(), pair).is_some() {
            failures.push(format!(
                "duplicate conformance report for provider {} model {}",
                key.0, key.1
            ));
        }
    }

    let mut reports = Vec::with_capacity(config.models.len());
    for model in &config.models {
        let key = (model.provider.clone(), model.id.clone());
        let provider = config
            .providers
            .iter()
            .find(|provider| provider.name == model.provider)
            .with_context(|| format!("provider {} is not configured", model.provider))?;
        let Some(artifact_pair) = artifact_pairs.remove(&key) else {
            failures.push(format!(
                "missing conformance report for provider {} model {}",
                model.provider, model.id
            ));
            reports.push(PromotionPairReport {
                provider: model.provider.clone(),
                model: model.id.clone(),
                provider_version: None,
                model_version: None,
                checks: vec![failed_check(
                    "pair",
                    "report",
                    true,
                    "configured provider/model pair has no live report",
                )],
            });
            continue;
        };
        let mut checks = Vec::new();
        if !artifact_pair.pass {
            checks.push(failed_check(
                "pair",
                "overall",
                true,
                "conformance report overall pass is false",
            ));
        }
        if require_reported_versions && artifact_pair.provider_version.value.is_none() {
            checks.push(failed_check(
                "version",
                "provider",
                true,
                "provider version was not reported",
            ));
        } else {
            checks.push(passed_check(
                "version",
                "provider",
                true,
                artifact_pair
                    .provider_version
                    .value
                    .as_deref()
                    .unwrap_or("unreported version allowed by gate configuration"),
            ));
        }
        if require_reported_versions && artifact_pair.model_version.value.is_none() {
            checks.push(failed_check(
                "version",
                "model",
                true,
                "model version was not reported",
            ));
        } else {
            checks.push(passed_check(
                "version",
                "model",
                true,
                artifact_pair
                    .model_version
                    .value
                    .as_deref()
                    .unwrap_or("unreported version allowed by gate configuration"),
            ));
        }

        for endpoint in ModelEndpoint::ALL {
            let advertised = provider.supports_endpoint(endpoint)
                && model.capabilities.supports_endpoint(endpoint);
            if endpoint == ModelEndpoint::Chat {
                checks.push(chat_check(&artifact_pair.chat, advertised));
            } else {
                checks.push(endpoint_check(
                    artifact_pair
                        .endpoints
                        .iter()
                        .find(|candidate| candidate.endpoint == endpoint.as_str()),
                    endpoint.as_str(),
                    advertised,
                ));
            }
        }

        let adapter = provider.kind.chat_adapter_contract();
        let chat_advertised = model.capabilities.supports_endpoint(ModelEndpoint::Chat)
            && provider.supports_endpoint(ModelEndpoint::Chat);
        for (feature, advertised) in [
            ("streaming", chat_advertised && adapter.supports_streaming),
            (
                "tools",
                chat_advertised && model.capabilities.supports_tools,
            ),
            ("json", chat_advertised && model.capabilities.supports_json),
            (
                "vision",
                chat_advertised && model.capabilities.supports_vision,
            ),
            (
                "audio",
                model.capabilities.supports_audio
                    && [
                        ModelEndpoint::Speech,
                        ModelEndpoint::AudioTranscriptions,
                        ModelEndpoint::AudioTranslations,
                    ]
                    .iter()
                    .any(|endpoint| {
                        provider.supports_endpoint(*endpoint)
                            && model.capabilities.supports_endpoint(*endpoint)
                    }),
            ),
        ] {
            checks.push(feature_check(
                artifact_pair
                    .features
                    .iter()
                    .find(|candidate| candidate.feature == feature),
                feature,
                advertised,
            ));
        }

        for check in checks
            .iter()
            .filter(|check| check.status == PromotionCheckStatus::Fail)
        {
            failures.push(format!(
                "provider {} model {} {} {}: {}",
                model.provider, model.id, check.kind, check.name, check.reason
            ));
        }
        reports.push(PromotionPairReport {
            provider: model.provider.clone(),
            model: model.id.clone(),
            provider_version: artifact_pair.provider_version.value,
            model_version: artifact_pair.model_version.value,
            checks,
        });
    }
    for ((provider, model), _) in artifact_pairs {
        failures.push(format!(
            "conformance artifact contains unconfigured provider/model pair {provider}/{model}"
        ));
    }

    let advertised_checks = reports
        .iter()
        .flat_map(|report| &report.checks)
        .filter(|check| check.advertised)
        .count();
    let passed_checks = reports
        .iter()
        .flat_map(|report| &report.checks)
        .filter(|check| check.status == PromotionCheckStatus::Pass)
        .count();
    let failed_checks = reports
        .iter()
        .flat_map(|report| &report.checks)
        .filter(|check| check.status == PromotionCheckStatus::Fail)
        .count();
    let skipped_unadvertised_checks = reports
        .iter()
        .flat_map(|report| &report.checks)
        .filter(|check| check.status == PromotionCheckStatus::Skip)
        .count();
    Ok(ProviderPromotionGateReport {
        schema_version: 1,
        evaluated_unix_seconds,
        artifact_generated_unix_seconds: artifact.generated_unix_seconds,
        artifact_age_seconds,
        max_age_seconds,
        config_fnv1a_64: expected_fingerprint,
        require_reported_versions,
        configured_pairs: config.models.len(),
        advertised_checks,
        passed_checks,
        failed_checks,
        skipped_unadvertised_checks,
        pass: failures.is_empty(),
        failures,
        reports,
    })
}

fn chat_check(chat: &ChatArtifact, advertised: bool) -> PromotionCheck {
    if !advertised {
        return skip_check(
            "endpoint",
            "chat",
            chat.skip_reason
                .as_deref()
                .unwrap_or("chat is not advertised by this provider/model pair"),
        );
    }
    if chat.configured
        && (200..300).contains(&chat.status)
        && chat.openai_chat_shape
        && chat.response_model_matches
        && chat.assistant_content_present
        && chat.usage_present
        && chat.negative_schema_rejected
    {
        passed_check("endpoint", "chat", true, "live chat schema checks passed")
    } else {
        failed_check(
            "endpoint",
            "chat",
            true,
            chat.skip_reason
                .as_deref()
                .unwrap_or("advertised chat endpoint did not pass all schema checks"),
        )
    }
}

fn endpoint_check(
    endpoint: Option<&EndpointArtifact>,
    name: &str,
    advertised: bool,
) -> PromotionCheck {
    let Some(endpoint) = endpoint else {
        return if advertised {
            failed_check(
                "endpoint",
                name,
                true,
                "artifact omitted advertised endpoint",
            )
        } else {
            skip_check("endpoint", name, "endpoint is not advertised")
        };
    };
    if !advertised {
        return skip_check(
            "endpoint",
            name,
            endpoint
                .skip_reason
                .as_deref()
                .unwrap_or("endpoint is not advertised"),
        );
    }
    if endpoint.configured
        && endpoint.pass
        && endpoint.positive_schema_valid
        && endpoint.negative_schema_rejected
    {
        passed_check("endpoint", name, true, "live endpoint schema checks passed")
    } else {
        failed_check(
            "endpoint",
            name,
            true,
            endpoint
                .skip_reason
                .as_deref()
                .unwrap_or("advertised endpoint did not pass all schema checks"),
        )
    }
}

fn feature_check(
    feature: Option<&FeatureArtifact>,
    name: &str,
    advertised: bool,
) -> PromotionCheck {
    let Some(feature) = feature else {
        return if advertised {
            failed_check("feature", name, true, "artifact omitted advertised feature")
        } else {
            skip_check("feature", name, "feature is not advertised")
        };
    };
    if !advertised {
        return skip_check(
            "feature",
            name,
            feature
                .skip_reason
                .as_deref()
                .unwrap_or("feature is not advertised"),
        );
    }
    if feature.declared && feature.attempted && feature.pass {
        passed_check("feature", name, true, "live feature probe passed")
    } else {
        failed_check(
            "feature",
            name,
            true,
            feature
                .skip_reason
                .as_deref()
                .unwrap_or("advertised feature was skipped or failed"),
        )
    }
}

fn passed_check(kind: &str, name: &str, advertised: bool, reason: &str) -> PromotionCheck {
    PromotionCheck {
        kind: kind.to_string(),
        name: name.to_string(),
        advertised,
        status: PromotionCheckStatus::Pass,
        reason: reason.to_string(),
    }
}

fn failed_check(kind: &str, name: &str, advertised: bool, reason: &str) -> PromotionCheck {
    PromotionCheck {
        kind: kind.to_string(),
        name: name.to_string(),
        advertised,
        status: PromotionCheckStatus::Fail,
        reason: reason.to_string(),
    }
}

fn skip_check(kind: &str, name: &str, reason: &str) -> PromotionCheck {
    PromotionCheck {
        kind: kind.to_string(),
        name: name.to_string(),
        advertised: false,
        status: PromotionCheckStatus::Skip,
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{
        CONFORMANCE_FILE, ControlledEvidenceConfig, controlled_router_config,
        write_controlled_evidence,
    };
    use serde_json::Value;

    #[tokio::test]
    async fn fresh_versioned_complete_matrix_passes_promotion() {
        let directory =
            std::env::temp_dir().join(format!("router-promotion-pass-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let manifest = write_controlled_evidence(ControlledEvidenceConfig {
            revision: "promotion-test".to_string(),
            output_dir: directory.clone(),
            runs: 2,
            requests_per_scenario: 1,
            concurrency: 1,
            slo_p95_ms: 2_000,
            slo_error_rate: 0.0,
        })
        .await
        .unwrap();
        let config = controlled_router_config("http://127.0.0.1:1").unwrap();
        retarget_config_fingerprint(&directory.join(CONFORMANCE_FILE), &config);
        let report = evaluate_provider_promotion_gate(
            &config,
            &directory.join(CONFORMANCE_FILE),
            manifest.generated_unix_seconds,
            60,
            true,
        )
        .unwrap();

        assert!(report.pass, "{:?}", report.failures);
        assert_eq!(report.configured_pairs, 1);
        assert_eq!(report.failed_checks, 0);
        assert!(report.advertised_checks >= 14);
        assert!(report.skipped_unadvertised_checks == 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn stale_unversioned_or_failed_evidence_blocks_promotion() {
        let directory =
            std::env::temp_dir().join(format!("router-promotion-fail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let manifest = write_controlled_evidence(ControlledEvidenceConfig {
            revision: "promotion-test".to_string(),
            output_dir: directory.clone(),
            runs: 2,
            requests_per_scenario: 1,
            concurrency: 1,
            slo_p95_ms: 2_000,
            slo_error_rate: 0.0,
        })
        .await
        .unwrap();
        let artifact_path = directory.join(CONFORMANCE_FILE);
        let mut artifact =
            serde_json::from_slice::<Value>(&fs::read(&artifact_path).unwrap()).unwrap();
        artifact["reports"][0]["provider_version"]["value"] = Value::Null;
        artifact["reports"][0]["endpoints"][0]["pass"] = Value::Bool(false);
        let config = controlled_router_config("http://127.0.0.1:1").unwrap();
        artifact["config_fnv1a_64"] = Value::String(config_fingerprint(&config).unwrap());
        fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&artifact).unwrap(),
        )
        .unwrap();
        let report = evaluate_provider_promotion_gate(
            &config,
            &artifact_path,
            manifest.generated_unix_seconds + 120,
            60,
            true,
        )
        .unwrap();

        assert!(!report.pass);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("old"))
        );
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("provider version"))
        );
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.contains("responses"))
        );
        fs::remove_dir_all(directory).unwrap();
    }

    fn retarget_config_fingerprint(path: &Path, config: &RouterConfig) {
        let mut artifact = serde_json::from_slice::<Value>(&fs::read(path).unwrap()).unwrap();
        artifact["config_fnv1a_64"] = Value::String(config_fingerprint(config).unwrap());
        fs::write(path, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
    }
}
