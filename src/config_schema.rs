use serde_json::{Value, json};

pub fn schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://autohand.ai/schemas/router.config.schema.json",
        "title": "Autohand Router Config",
        "description": "YAML/JSON configuration schema for the Autohand OpenAI-compatible LLM router.",
        "type": "object",
        "additionalProperties": false,
        "required": ["default_model"],
        "properties": {
            "bind": string_default("127.0.0.1:8080"),
            "default_model": {
                "type": "string",
                "minLength": 1,
                "description": "Configured model id or alias used as the final fail-closed fallback."
            },
            "policy": ref_schema("RouterPolicy"),
            "providers": {
                "type": "array",
                "items": ref_schema("ProviderConfig"),
                "default": []
            },
            "models": {
                "type": "array",
                "items": ref_schema("ModelConfig"),
                "default": []
            },
            "classifier": ref_schema("ClassifierConfig"),
            "auth": ref_schema("AuthConfig"),
            "scoring": ref_schema("ScoringConfig"),
            "budget": ref_schema("BudgetConfig"),
            "telemetry": ref_schema("TelemetryConfig"),
            "runtime": ref_schema("RuntimeConfig"),
            "cache": ref_schema("CacheConfig"),
            "shadow_eval": ref_schema("ShadowEvalConfig"),
            "safety": ref_schema("SafetyRoutingConfig"),
            "sticky_routing": ref_schema("StickyRoutingConfig")
        },
        "$defs": defs()
    })
}

fn defs() -> Value {
    json!({
        "RouterPolicy": string_enum(&[
            "balanced",
            "lowest_cost_acceptable",
            "fastest_healthy",
            "fast",
            "highest_quality",
            "local_first",
            "privacy_first",
            "multimodal_first",
            "floor",
            "nitro",
            "quality",
            "cost_efficient",
            "cost",
            "capability_heavy",
            "capability",
            "domain_skills",
            "domain"
        ]),
        "ProviderKind": string_enum(&[
            "open_ai_compatible",
            "ollama",
            "ollama_native",
            "llama_cpp",
            "llama_cpp_native",
            "vllm",
            "openrouter",
            "cloudflare_ai_gateway"
        ]),
        "DomainLabel": string_enum(&["general", "summary", "coding", "design", "data"]),
        "ClassifierBackend": string_enum(&["heuristic", "llm_judge", "route_llm"]),
        "StorageBackend": string_enum(&["memory", "file"]),
        "BudgetAccountingBackend": string_enum(&["process", "file"]),
        "SafetyRoutingAction": string_enum(&["allow", "reject", "redact", "force_route"]),
        "ModelEndpoint": string_enum(&[
            "chat",
            "responses",
            "embeddings",
            "images",
            "speech",
            "audio_transcriptions",
            "audio_translations"
        ]),
        "ProviderConfig": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "base_url"],
            "properties": {
                "name": non_empty_string("Provider identifier referenced by models."),
                "kind": ref_schema("ProviderKind"),
                "base_url": non_empty_string("HTTP base URL for the provider."),
                "api_key_env": nullable_string(),
                "api_key": nullable_string(),
                "chat_path": string_default("/v1/chat/completions"),
                "responses_path": nullable_string(),
                "embeddings_path": nullable_string(),
                "images_path": nullable_string(),
                "speech_path": nullable_string(),
                "audio_transcriptions_path": nullable_string(),
                "audio_translations_path": nullable_string(),
                "health_path": nullable_string(),
                "timeout_ms": integer_min_default(1, 120000),
                "retries": integer_min_default(0, 1),
                "max_concurrency": nullable_integer_min(1),
                "queue_timeout_ms": nullable_integer_min(1),
                "extra_headers": string_map()
            }
        },
        "ModelConfig": {
            "type": "object",
            "additionalProperties": false,
            "required": ["id", "provider"],
            "properties": {
                "id": non_empty_string("Model id sent upstream."),
                "provider": non_empty_string("Provider name from providers[].name."),
                "aliases": string_array(),
                "capability": number_range_default(0.0, 1.0, 0.5),
                "cost_per_million_input": number_min_default(0.0, 1.0),
                "cost_per_million_output": number_min_default(0.0, 1.0),
                "domains": {
                    "type": "array",
                    "items": ref_schema("DomainLabel"),
                    "default": []
                },
                "context_window": nullable_integer_min(1),
                "capabilities": ref_schema("ModelCapabilities"),
                "local": { "type": "boolean", "default": false }
            }
        },
        "ModelCapabilities": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "supports_vision": bool_default(false),
                "supports_audio": bool_default(false),
                "supports_tools": bool_default(false),
                "supports_json": bool_default(false),
                "supports_code": bool_default(false),
                "supports_web_apps": bool_default(false),
                "supports_long_context": bool_default(false),
                "supported_endpoints": {
                    "type": ["array", "null"],
                    "items": ref_schema("ModelEndpoint"),
                    "default": null,
                    "description": "Explicit model-level endpoint allowlist; omitted means chat only."
                }
            }
        },
        "ClassifierConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "backend": ref_schema("ClassifierBackend"),
                "confidence_threshold": number_range_default(0.0, 1.0, 0.62),
                "easy_threshold": number_range_default(0.0, 1.0, 0.28),
                "hard_threshold": number_range_default(0.0, 1.0, 0.62),
                "llm_judge_model": nullable_string(),
                "llm_judge_timeout_ms": integer_min_default(1, 2500),
                "adapters": ref_schema("ClassifierAdaptersConfig")
            }
        },
        "ClassifierAdaptersConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "llm_judge": ref_schema("ClassifierModelAdapterConfig"),
                "route_llm": ref_schema("ClassifierModelAdapterConfig")
            }
        },
        "ClassifierModelAdapterConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "model": nullable_string(),
                "timeout_ms": integer_min_default(1, 2500),
                "prompt_template": nullable_string()
            }
        },
        "AuthConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "bearer_tokens": string_array(),
                "bearer_token_env": string_array(),
                "allow_unauthenticated_network": bool_default(false)
            }
        },
        "BudgetConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "max_chat_requests": nullable_integer_min(1),
                "max_total_tokens": nullable_integer_min(1),
                "max_estimated_cost_micros": nullable_integer_min(1),
                "accounting": ref_schema("BudgetAccountingConfig")
            }
        },
        "BudgetAccountingConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "backend": ref_schema("BudgetAccountingBackend"),
                "file_path": nullable_string(),
                "lock_timeout_ms": integer_min_default(1, 1000)
            }
        },
        "TelemetryConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "decision_log_path": nullable_string(),
                "include_inputs": bool_default(false)
            }
        },
        "RuntimeConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "graceful_shutdown_timeout_ms": integer_min_default(1, 30000),
                "provider_health_sampler": ref_schema("ProviderHealthSamplerConfig"),
                "provider_conformance_artifact": nullable_string()
            }
        },
        "ProviderHealthSamplerConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "interval_ms": integer_min_default(1, 30000),
                "initial_delay_ms": integer_min_default(0, 500)
            }
        },
        "CacheConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "semantic": ref_schema("SemanticCacheConfig")
            }
        },
        "SemanticCacheConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "embedding_model": string_default("local-hash"),
                "similarity_threshold": number_range_default(0.0, 1.0, 0.92),
                "ttl_seconds": integer_min_default(1, 3600),
                "max_entries": integer_min_default(1, 1024),
                "backend": ref_schema("StorageBackend"),
                "file_path": nullable_string(),
                "lock_timeout_ms": integer_min_default(1, 1000)
            }
        },
        "ShadowEvalConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "sample_rate": number_range_default(0.0, 1.0, 0.01),
                "output_path": nullable_string(),
                "include_bodies": bool_default(false),
                "max_body_chars": integer_min_default(1, 4096),
                "judge": ref_schema("ShadowEvalJudgeConfig")
            }
        },
        "ShadowEvalJudgeConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(true),
                "model": nullable_string(),
                "timeout_ms": integer_min_default(1, 5000),
                "prompt_template": nullable_string()
            }
        },
        "SafetyRoutingConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "unsafe_action": ref_schema("SafetyRoutingAction"),
                "sensitive_action": ref_schema("SafetyRoutingAction"),
                "force_model": nullable_string(),
                "redaction_replacement": string_default("[redacted]")
            }
        },
        "StickyRoutingConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "ttl_seconds": integer_min_default(1, 1800),
                "prefer_model": bool_default(true),
                "backend": ref_schema("StorageBackend"),
                "file_path": nullable_string(),
                "lock_timeout_ms": integer_min_default(1, 1000)
            }
        },
        "ScoringConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "balanced": ref_schema("PolicyWeights"),
                "lowest_cost_acceptable": ref_schema("PolicyWeights"),
                "fastest_healthy": ref_schema("PolicyWeights"),
                "highest_quality": ref_schema("PolicyWeights"),
                "local_first": ref_schema("PolicyWeights"),
                "privacy_first": ref_schema("PolicyWeights"),
                "multimodal_first": ref_schema("PolicyWeights"),
                "floor": ref_schema("PolicyWeights"),
                "nitro": ref_schema("PolicyWeights"),
                "quality": ref_schema("PolicyWeights"),
                "cost_efficient": ref_schema("PolicyWeights"),
                "capability_heavy": ref_schema("PolicyWeights"),
                "domain_skills": ref_schema("PolicyWeights"),
                "model_priorities": number_map(),
                "provider_priorities": number_map(),
                "provider_latency_p95_ms": integer_map(),
                "provider_health_penalties": number_map(),
                "priority_weight": number_min_default(0.0, 0.08),
                "latency_weight": number_min_default(0.0, 0.05),
                "health_weight": number_min_default(0.0, 1.0),
                "learned": ref_schema("LearnedScoringConfig")
            }
        },
        "LearnedScoringConfig": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "enabled": bool_default(false),
                "weight": number_min_default(0.0, 0.0),
                "bias": { "type": "number", "default": 0.0 },
                "feature_weights": number_map(),
                "model_biases": number_map()
            }
        },
        "PolicyWeights": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "capability_fit": { "type": "number" },
                "domain_bonus": { "type": "number" },
                "cost": { "type": "number" },
                "overkill": { "type": "number" },
                "raw_capability": { "type": "number", "default": 0.0 },
                "latency": { "type": "number", "default": 0.05 },
                "health": { "type": "number", "default": 1.0 },
                "local_bonus": { "type": "number", "default": 0.0 },
                "remote_penalty": { "type": "number", "default": 0.0 },
                "multimodal_capability": { "type": "number", "default": 0.0 }
            }
        }
    })
}

fn ref_schema(name: &str) -> Value {
    json!({ "$ref": format!("#/$defs/{name}") })
}

fn string_enum(values: &[&str]) -> Value {
    json!({
        "type": "string",
        "enum": values
    })
}

fn non_empty_string(description: &str) -> Value {
    json!({
        "type": "string",
        "minLength": 1,
        "description": description
    })
}

fn string_default(default: &str) -> Value {
    json!({
        "type": "string",
        "default": default
    })
}

fn nullable_string() -> Value {
    json!({
        "type": ["string", "null"]
    })
}

fn string_array() -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" },
        "default": []
    })
}

fn string_map() -> Value {
    json!({
        "type": "object",
        "additionalProperties": { "type": "string" },
        "default": {}
    })
}

fn number_map() -> Value {
    json!({
        "type": "object",
        "additionalProperties": { "type": "number" },
        "default": {}
    })
}

fn integer_map() -> Value {
    json!({
        "type": "object",
        "additionalProperties": { "type": "integer", "minimum": 0 },
        "default": {}
    })
}

fn bool_default(default: bool) -> Value {
    json!({
        "type": "boolean",
        "default": default
    })
}

fn integer_min_default(minimum: u64, default: u64) -> Value {
    json!({
        "type": "integer",
        "minimum": minimum,
        "default": default
    })
}

fn nullable_integer_min(minimum: u64) -> Value {
    json!({
        "type": ["integer", "null"],
        "minimum": minimum
    })
}

fn number_min_default(minimum: f64, default: f64) -> Value {
    json!({
        "type": "number",
        "minimum": minimum,
        "default": default
    })
}

fn number_range_default(minimum: f64, maximum: f64, default: f64) -> Value {
    json!({
        "type": "number",
        "minimum": minimum,
        "maximum": maximum,
        "default": default
    })
}

#[cfg(test)]
mod tests {
    use super::schema;

    #[test]
    fn config_schema_contains_router_config_sections() {
        let schema = schema();

        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["required"][0], "default_model");
        assert!(schema["properties"]["providers"].is_object());
        assert!(schema["properties"]["models"].is_object());
        assert!(schema["$defs"]["ProviderConfig"].is_object());
        assert!(schema["$defs"]["ModelConfig"].is_object());
        assert!(schema["$defs"]["ModelEndpoint"].is_object());
        assert_eq!(
            schema["$defs"]["ModelCapabilities"]["properties"]["supported_endpoints"]["items"]["$ref"],
            "#/$defs/ModelEndpoint"
        );
        assert!(schema["$defs"]["SemanticCacheConfig"].is_object());
        assert!(schema["$defs"]["StickyRoutingConfig"].is_object());
        assert!(schema["$defs"]["ShadowEvalJudgeConfig"].is_object());
        assert!(schema["$defs"]["LearnedScoringConfig"].is_object());
        assert_eq!(
            schema["$defs"]["AuthConfig"]["properties"]["allow_unauthenticated_network"]["default"],
            false
        );
        assert_eq!(
            schema["$defs"]["SemanticCacheConfig"]["properties"]["backend"]["$ref"],
            "#/$defs/StorageBackend"
        );
        assert_eq!(
            schema["$defs"]["ScoringConfig"]["properties"]["learned"]["$ref"],
            "#/$defs/LearnedScoringConfig"
        );
        assert_eq!(
            schema["$defs"]["ScoringConfig"]["properties"]["local_first"]["$ref"],
            "#/$defs/PolicyWeights"
        );
        assert_eq!(
            schema["$defs"]["ScoringConfig"]["properties"]["privacy_first"]["$ref"],
            "#/$defs/PolicyWeights"
        );
        assert_eq!(
            schema["$defs"]["ScoringConfig"]["properties"]["multimodal_first"]["$ref"],
            "#/$defs/PolicyWeights"
        );
        assert_eq!(
            schema["$defs"]["PolicyWeights"]["properties"]["local_bonus"]["default"],
            0.0
        );
        assert_eq!(
            schema["$defs"]["PolicyWeights"]["properties"]["remote_penalty"]["default"],
            0.0
        );
        assert_eq!(
            schema["$defs"]["PolicyWeights"]["properties"]["multimodal_capability"]["default"],
            0.0
        );
        assert_eq!(
            schema["$defs"]["ModelCapabilities"]["properties"]["supports_web_apps"]["type"],
            "boolean"
        );
        assert!(
            schema["$defs"]["ProviderConfig"]["properties"]["responses_path"]
                .get("default")
                .is_none()
        );
        assert_eq!(
            schema["$defs"]["ModelCapabilities"]["properties"]["supported_endpoints"]["description"],
            "Explicit model-level endpoint allowlist; omitted means chat only."
        );
        assert!(
            schema["$defs"]["RuntimeConfig"]["properties"]["provider_conformance_artifact"]
                .is_object()
        );
    }
}
