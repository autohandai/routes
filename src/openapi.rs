use serde_json::{Value, json};

pub fn spec() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Autohand Router",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "OpenAI-compatible Rust LLM router with classification and multimodel routing endpoints."
        },
        "servers": [
            { "url": "http://127.0.0.1:8080" }
        ],
        "security": [
            { "bearerAuth": [] },
            {}
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Router is alive",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ok": { "type": "boolean" },
                                            "service": { "type": "string" }
                                        },
                                        "required": ["ok", "service"]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "OpenAPI document",
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "OpenAPI JSON",
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                }
                            }
                        }
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List configured models",
                    "responses": {
                        "200": {
                            "description": "OpenAI-compatible model list",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ModelList" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/v1/router/classify": {
                "post": {
                    "summary": "Classify prompt routing heads",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/ClassifyRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Classification result",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ClassifyResponse" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/v1/router/raw": {
                "post": {
                    "summary": "Compatibility difficulty-only router",
                    "deprecated": true,
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/RawRouterRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Difficulty-only classification",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/RawRouterResponse" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/v1/router/{provider}": {
                "post": {
                    "summary": "Provider-specific model selector",
                    "deprecated": true,
                    "parameters": [
                        {
                            "name": "provider",
                            "in": "path",
                            "required": true,
                            "schema": { "type": "string" }
                        }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/ProviderRouterRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Provider-constrained model selection",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ProviderRouterResponse" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "404": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/router/multimodel": {
                "post": {
                    "summary": "Select a model/provider for an input",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/MultimodelRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Routing decision",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/MultimodelResponse" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/v1/router/providers": {
                "get": {
                    "summary": "Provider health status",
                    "responses": {
                        "200": {
                            "description": "Provider health statuses",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "providers": {
                                                "type": "array",
                                                "items": { "$ref": "#/components/schemas/ProviderHealth" }
                                            },
                                            "sampled": {
                                                "type": "array",
                                                "items": { "$ref": "#/components/schemas/ProviderHealthObservation" }
                                            }
                                        },
                                        "required": ["providers", "sampled"]
                                    }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "summary": "OpenAI-compatible chat completions proxy",
                    "description": "Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/OpenAiChatRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-cache": { "schema": { "type": "string", "enum": ["hit", "miss"] } },
                                "x-autohand-router-cache-similarity": { "schema": { "type": "number" } },
                                "x-autohand-router-cache-embedding-model": { "schema": { "type": "string" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                },
                                "text/event-stream": {
                                    "schema": { "type": "string" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/responses": {
                "post": {
                    "summary": "OpenAI-compatible responses proxy",
                    "description": "Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/OpenAiResponsesRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-cache": { "schema": { "type": "string", "enum": ["hit", "miss"] } },
                                "x-autohand-router-cache-similarity": { "schema": { "type": "number" } },
                                "x-autohand-router-cache-embedding-model": { "schema": { "type": "string" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                },
                                "text/event-stream": {
                                    "schema": { "type": "string" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/embeddings": {
                "post": {
                    "summary": "OpenAI-compatible embeddings proxy",
                    "description": "Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding to providers with embeddings support.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/OpenAiEmbeddingsRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible embeddings response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/images/generations": {
                "post": {
                    "summary": "OpenAI-compatible image generations proxy",
                    "description": "Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding to providers with image generation support.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/OpenAiImagesRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible image generation response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/audio/speech": {
                "post": {
                    "summary": "OpenAI-compatible audio speech proxy",
                    "description": "Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding to providers with speech support.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/OpenAiSpeechRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible speech response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/octet-stream": {
                                    "schema": { "type": "string", "format": "binary" }
                                },
                                "audio/mpeg": {
                                    "schema": { "type": "string", "format": "binary" }
                                },
                                "application/json": {
                                    "schema": { "type": "object" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/audio/transcriptions": {
                "post": {
                    "summary": "OpenAI-compatible audio transcription proxy",
                    "description": "Accepts OpenAI-compatible multipart audio transcription requests. Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": { "$ref": "#/components/schemas/OpenAiAudioMultipartRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible transcription response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                },
                                "text/plain": {
                                    "schema": { "type": "string" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/v1/audio/translations": {
                "post": {
                    "summary": "OpenAI-compatible audio translation proxy",
                    "description": "Accepts OpenAI-compatible multipart audio translation requests. Use model `auto` or `router-*` for automatic routing, or any configured model id/alias for strict forwarding.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "multipart/form-data": {
                                "schema": { "$ref": "#/components/schemas/OpenAiAudioMultipartRequest" }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upstream OpenAI-compatible translation response",
                            "headers": {
                                "x-autohand-router-model": { "schema": { "type": "string" } },
                                "x-autohand-router-provider": { "schema": { "type": "string" } },
                                "x-autohand-router-failovers": { "schema": { "type": "integer" } },
                                "x-autohand-router-input-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-output-tokens": { "schema": { "type": "integer" } },
                                "x-autohand-router-request-id": { "schema": { "type": "string" } }
                            },
                            "content": {
                                "application/json": {
                                    "schema": { "type": "object" }
                                },
                                "text/plain": {
                                    "schema": { "type": "string" }
                                }
                            }
                        },
                        "400": { "$ref": "#/components/responses/RouterError" },
                        "401": { "$ref": "#/components/responses/Unauthorized" },
                        "429": { "$ref": "#/components/responses/RouterError" },
                        "502": { "$ref": "#/components/responses/RouterError" }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Router metrics and accounting snapshot",
                    "responses": {
                        "200": {
                            "description": "Metrics snapshot",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/MetricsSnapshot" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            },
            "/metrics/prometheus": {
                "get": {
                    "summary": "Prometheus-compatible router metrics",
                    "responses": {
                        "200": {
                            "description": "Prometheus text exposition metrics",
                            "content": {
                                "text/plain": {
                                    "schema": { "type": "string" }
                                }
                            }
                        },
                        "401": { "$ref": "#/components/responses/Unauthorized" }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            },
            "responses": {
                "Unauthorized": {
                    "description": "Missing or invalid bearer token",
                    "headers": {
                        "WWW-Authenticate": {
                            "schema": { "type": "string", "const": "Bearer" }
                        },
                        "x-autohand-router-request-id": {
                            "schema": { "type": "string" }
                        }
                    },
                    "content": {
                        "application/json": {
                            "schema": { "$ref": "#/components/schemas/RouterError" }
                        }
                    }
                },
                "RouterError": {
                    "description": "Router error",
                    "content": {
                        "application/json": {
                            "schema": { "$ref": "#/components/schemas/RouterError" }
                        }
                    }
                }
            },
            "schemas": schemas()
        }
    })
}

fn schemas() -> Value {
    json!({
        "RouterError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" },
                        "type": { "type": "string" }
                    },
                    "required": ["message", "type"]
                }
            },
            "required": ["error"]
        },
        "ClassifyRequest": {
            "type": "object",
            "properties": {
                "input": { "type": "string" },
                "classes": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/ClassificationHead" }
                }
            },
            "required": ["input"]
        },
        "ClassifyResponse": {
            "type": "object",
            "properties": {
                "classifications": { "$ref": "#/components/schemas/Classifications" }
            },
            "required": ["classifications"]
        },
        "RawRouterRequest": {
            "type": "object",
            "properties": {
                "input": { "type": "string" },
                "mode": { "$ref": "#/components/schemas/LegacyRouterMode" }
            },
            "required": ["input"]
        },
        "RawRouterResponse": {
            "type": "object",
            "properties": {
                "difficulty": { "type": "string", "enum": ["easy", "medium", "hard", "needs_info"] },
                "confidence": { "type": "number" }
            },
            "required": ["difficulty", "confidence"]
        },
        "ProviderRouterRequest": {
            "type": "object",
            "properties": {
                "input": { "type": "string" },
                "mode": { "$ref": "#/components/schemas/LegacyRouterMode" }
            },
            "required": ["input"]
        },
        "ProviderRouterResponse": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "confidence": { "type": "number" }
            },
            "required": ["model", "confidence"]
        },
        "LegacyRouterMode": {
            "type": "string",
            "enum": ["balanced", "aggressive"]
        },
        "Classifications": {
            "type": "object",
            "properties": {
                "difficulty": { "$ref": "#/components/schemas/DifficultyClassification" },
                "ambiguity": { "$ref": "#/components/schemas/AmbiguityClassification" },
                "domain": { "$ref": "#/components/schemas/DomainClassification" },
                "modality": { "$ref": "#/components/schemas/ModalityClassification" },
                "safety": { "$ref": "#/components/schemas/SafetyClassification" },
                "cacheability": { "$ref": "#/components/schemas/CacheabilityClassification" },
                "latency_sensitivity": { "$ref": "#/components/schemas/LatencySensitivityClassification" },
                "reasoning_depth": { "$ref": "#/components/schemas/ReasoningDepthClassification" }
            }
        },
        "ClassificationHead": {
            "type": "string",
            "enum": ["difficulty", "ambiguity", "domain", "modality", "safety", "cacheability", "latency_sensitivity", "reasoning_depth"]
        },
        "DifficultyClassification": classification_schema(["easy", "medium", "hard", "needs_info"]),
        "AmbiguityClassification": classification_schema(["low", "med", "high"]),
        "DomainClassification": classification_schema(["general", "summary", "coding", "design", "data"]),
        "ModalityClassification": classification_schema(["text", "vision", "audio", "tool_use", "multimodal"]),
        "SafetyClassification": classification_schema(["safe", "sensitive", "unsafe"]),
        "CacheabilityClassification": classification_schema(["low", "medium", "high"]),
        "LatencySensitivityClassification": classification_schema(["low", "medium", "high"]),
        "ReasoningDepthClassification": classification_schema(["shallow", "moderate", "deep"]),
        "MultimodelRequest": {
            "type": "object",
            "properties": {
                "input": { "type": "string" },
                "allowed_models": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional model id/alias allowlist. When provider filters are also present, a candidate must satisfy both allowlists."
                },
                "allowed_providers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional provider allowlist. When model filters are also present, a candidate must satisfy both allowlists."
                },
                "required_capabilities": { "type": "array", "items": { "$ref": "#/components/schemas/ModelCapability" } },
                "policy": { "$ref": "#/components/schemas/RouterPolicy" },
                "default_model": { "type": ["string", "null"] },
                "max_output_tokens": { "type": ["integer", "null"], "minimum": 1 }
            },
            "required": ["input"]
        },
        "MultimodelResponse": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "provider": { "type": "string" },
                "difficulty": { "type": "string", "enum": ["easy", "medium", "hard", "needs_info"] },
                "confidence": { "type": "number" },
                "ambiguity": { "type": ["string", "null"], "enum": ["low", "med", "high", null] },
                "ambiguity_confidence": { "type": ["number", "null"] },
                "domain": { "type": ["string", "null"], "enum": ["general", "summary", "coding", "design", "data", null] },
                "domain_confidence": { "type": ["number", "null"] },
                "modality": { "type": ["string", "null"], "enum": ["text", "vision", "audio", "tool_use", "multimodal", null] },
                "modality_confidence": { "type": ["number", "null"] },
                "safety": { "type": ["string", "null"], "enum": ["safe", "sensitive", "unsafe", null] },
                "safety_confidence": { "type": ["number", "null"] },
                "cacheability": { "type": ["string", "null"], "enum": ["low", "medium", "high", null] },
                "cacheability_confidence": { "type": ["number", "null"] },
                "latency_sensitivity": { "type": ["string", "null"], "enum": ["low", "medium", "high", null] },
                "latency_sensitivity_confidence": { "type": ["number", "null"] },
                "reasoning_depth": { "type": ["string", "null"], "enum": ["shallow", "moderate", "deep", null] },
                "reasoning_depth_confidence": { "type": ["number", "null"] },
                "policy": { "$ref": "#/components/schemas/RouterPolicy" },
                "reason": { "type": "string" },
                "fallback": { "type": "boolean" },
                "estimated_input_tokens": { "type": "integer" },
                "requested_output_tokens": { "type": "integer" },
                "decision_trace": { "$ref": "#/components/schemas/RouteDecisionTrace" },
                "candidates": { "type": "array", "items": { "$ref": "#/components/schemas/RouteCandidate" } }
            },
            "required": ["model", "provider", "difficulty", "confidence", "policy", "reason", "fallback", "estimated_input_tokens", "requested_output_tokens"]
        },
        "RouteCandidate": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "provider": { "type": "string" },
                "score": { "type": "number" },
                "capability": { "type": "number" },
                "estimated_cost": { "type": "number" },
                "domain_match": { "type": "boolean" },
                "routing_priority": { "type": "number" },
                "latency_penalty": { "type": "number" },
                "health_penalty": { "type": "number" },
                "capability_eligible": { "type": "boolean" },
                "missing_capabilities": { "type": "array", "items": { "$ref": "#/components/schemas/ModelCapability" } },
                "context_window": { "type": ["integer", "null"] },
                "context_required": { "type": "integer" },
                "context_eligible": { "type": "boolean" },
                "score_components": { "$ref": "#/components/schemas/RouteScoreComponents" }
            },
            "required": ["model", "provider", "score", "capability", "estimated_cost", "domain_match", "routing_priority", "latency_penalty", "health_penalty", "capability_eligible", "missing_capabilities", "context_required", "context_eligible", "score_components"]
        },
        "RouteDecisionTrace": {
            "type": "object",
            "properties": {
                "classifier": { "$ref": "#/components/schemas/Classifications" },
                "policy": { "$ref": "#/components/schemas/RouterPolicy" },
                "policy_weights": { "$ref": "#/components/schemas/RoutePolicyWeights" },
                "required_capabilities": { "type": "array", "items": { "$ref": "#/components/schemas/ModelCapability" } },
                "context_required": { "type": "integer" },
                "selected_model": { "type": ["string", "null"] },
                "selected_provider": { "type": ["string", "null"] },
                "selected_score": { "type": ["number", "null"] },
                "rejected_candidates": { "type": "array", "items": { "$ref": "#/components/schemas/RouteCandidateRejection" } }
            },
            "required": ["classifier", "policy", "policy_weights", "required_capabilities", "context_required", "rejected_candidates"]
        },
        "RoutePolicyWeights": {
            "type": "object",
            "properties": {
                "capability_fit": { "type": "number" },
                "domain_bonus": { "type": "number" },
                "cost": { "type": "number" },
                "overkill": { "type": "number" },
                "raw_capability": { "type": "number" },
                "latency": { "type": "number" },
                "health": { "type": "number" },
                "local_bonus": { "type": "number" },
                "remote_penalty": { "type": "number" },
                "multimodal_capability": { "type": "number" }
            },
            "required": ["capability_fit", "domain_bonus", "cost", "overkill", "raw_capability", "latency", "health", "local_bonus", "remote_penalty", "multimodal_capability"]
        },
        "RouteCandidateRejection": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "provider": { "type": "string" },
                "reasons": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["model", "provider", "reasons"]
        },
        "RouteScoreComponents": {
            "type": "object",
            "properties": {
                "capability_fit": { "type": "number" },
                "capability_fit_score": { "type": "number" },
                "domain_bonus": { "type": "number" },
                "domain_bonus_score": { "type": "number" },
                "raw_capability_score": { "type": "number" },
                "cost_penalty": { "type": "number" },
                "overkill_penalty": { "type": "number" },
                "local_score_boost": { "type": "number" },
                "remote_penalty": { "type": "number" },
                "multimodal_score_boost": { "type": "number" },
                "routing_priority_boost": { "type": "number" },
                "learned_score_boost": { "type": "number" },
                "latency_penalty": { "type": "number" },
                "health_penalty": { "type": "number" },
                "capability_exclusion_penalty": { "type": "number" },
                "context_exclusion_penalty": { "type": "number" },
                "final_score": { "type": "number" }
            },
            "required": ["capability_fit", "capability_fit_score", "domain_bonus", "domain_bonus_score", "raw_capability_score", "cost_penalty", "overkill_penalty", "local_score_boost", "remote_penalty", "multimodal_score_boost", "routing_priority_boost", "learned_score_boost", "latency_penalty", "health_penalty", "capability_exclusion_penalty", "context_exclusion_penalty", "final_score"]
        },
        "ModelCapability": {
            "type": "string",
            "enum": ["vision", "audio", "tools", "json", "code", "web_apps", "long_context"]
        },
        "ModelCapabilities": {
            "type": "object",
            "properties": {
                "supports_vision": { "type": "boolean" },
                "supports_audio": { "type": "boolean" },
                "supports_tools": { "type": "boolean" },
                "supports_json": { "type": "boolean" },
                "supports_code": { "type": "boolean" },
                "supports_web_apps": { "type": "boolean" },
                "supports_long_context": { "type": "boolean" }
            }
        },
        "RouterPolicy": {
            "type": "string",
            "enum": ["balanced", "lowest_cost_acceptable", "fastest_healthy", "highest_quality", "local_first", "privacy_first", "multimodal_first", "floor", "nitro", "quality", "cost_efficient", "capability_heavy", "domain_skills"]
        },
        "OpenAiChatRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "messages": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/ChatMessage" }
                },
                "stream": { "type": "boolean" },
                "max_tokens": { "type": "integer" },
                "max_completion_tokens": { "type": "integer" }
            },
            "additionalProperties": true,
            "required": ["model", "messages"]
        },
        "OpenAiResponsesRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "input": {},
                "stream": { "type": "boolean" },
                "max_output_tokens": { "type": "integer" },
                "max_tokens": { "type": "integer" }
            },
            "additionalProperties": true,
            "required": ["model", "input"]
        },
        "OpenAiEmbeddingsRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "input": {},
                "encoding_format": { "type": "string" },
                "dimensions": { "type": "integer" },
                "user": { "type": "string" }
            },
            "additionalProperties": true,
            "required": ["model", "input"]
        },
        "OpenAiImagesRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "prompt": { "type": "string" },
                "n": { "type": "integer" },
                "size": { "type": "string" },
                "quality": { "type": "string" },
                "response_format": { "type": "string" },
                "user": { "type": "string" }
            },
            "additionalProperties": true,
            "required": ["model", "prompt"]
        },
        "OpenAiSpeechRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "input": { "type": "string" },
                "voice": { "type": "string" },
                "response_format": { "type": "string" },
                "speed": { "type": "number" }
            },
            "additionalProperties": true,
            "required": ["model", "input", "voice"]
        },
        "OpenAiAudioMultipartRequest": {
            "type": "object",
            "properties": {
                "model": { "type": "string" },
                "file": { "type": "string", "format": "binary" },
                "prompt": { "type": "string" },
                "language": { "type": "string" },
                "response_format": { "type": "string" },
                "temperature": { "type": "number" }
            },
            "additionalProperties": true,
            "required": ["model", "file"]
        },
        "ChatMessage": {
            "type": "object",
            "properties": {
                "role": { "type": "string" },
                "content": {}
            },
            "required": ["role"]
        },
        "ModelList": {
            "type": "object",
            "properties": {
                "object": { "type": "string" },
                "data": { "type": "array", "items": { "$ref": "#/components/schemas/ModelInfo" } }
            },
            "required": ["object", "data"]
        },
        "ModelInfo": {
            "type": "object",
            "properties": {
                "id": { "type": "string" },
                "object": { "type": "string" },
                "owned_by": { "type": "string" },
                "aliases": { "type": "array", "items": { "type": "string" } },
                "local": { "type": "boolean" },
                "context_window": { "type": ["integer", "null"] },
                "capabilities": { "$ref": "#/components/schemas/ModelCapabilities" }
            },
            "required": ["id", "object", "owned_by", "aliases", "local", "capabilities"]
        },
        "ProviderHealth": {
            "type": "object",
            "properties": {
                "provider": { "type": "string" },
                "adapter": { "type": "string" },
                "status": { "type": "string", "enum": ["ok", "error", "unknown"] },
                "status_code": { "type": ["integer", "null"] },
                "error": { "type": ["string", "null"] }
            },
            "required": ["provider", "adapter", "status"]
        },
        "ProviderHealthObservation": {
            "type": "object",
            "properties": {
                "provider": { "type": "string" },
                "status": { "type": "string", "enum": ["ok", "error", "unknown"] },
                "status_code": { "type": ["integer", "null"] },
                "error": { "type": ["string", "null"] },
                "latency_ms": { "type": ["integer", "null"] },
                "health_penalty": { "type": "number" },
                "observed_unix_seconds": { "type": "integer" }
            },
            "required": ["provider", "status", "health_penalty", "observed_unix_seconds"]
        },
        "MetricsSnapshot": {
            "type": "object",
            "additionalProperties": true,
            "properties": {
                "route_requests": { "type": "integer" },
                "classify_requests": { "type": "integer" },
                "chat_requests": { "type": "integer" },
                "responses_requests": { "type": "integer" },
                "embeddings_requests": { "type": "integer" },
                "images_requests": { "type": "integer" },
                "speech_requests": { "type": "integer" },
                "audio_transcription_requests": { "type": "integer" },
                "audio_translation_requests": { "type": "integer" },
                "fallback_routes": { "type": "integer" },
                "failover_attempts": { "type": "integer" },
                "failover_successes": { "type": "integer" },
                "auth_failures": { "type": "integer" },
                "upstream_errors": { "type": "integer" },
                "budget_rejections": { "type": "integer" },
                "semantic_cache_hits": { "type": "integer" },
                "semantic_cache_misses": { "type": "integer" },
                "shadow_eval_samples": { "type": "integer" },
                "shadow_eval_successes": { "type": "integer" },
                "shadow_eval_errors": { "type": "integer" },
                "safety_rejections": { "type": "integer" },
                "safety_redactions": { "type": "integer" },
                "safety_force_routes": { "type": "integer" },
                "sticky_routing_hits": { "type": "integer" },
                "sticky_routing_writes": { "type": "integer" },
                "selected_models": { "type": "integer" },
                "prompt_tokens": { "type": "integer" },
                "completion_tokens": { "type": "integer" },
                "total_tokens": { "type": "integer" },
                "estimated_cost_micros": { "type": "integer" },
                "estimated_cost_usd": { "type": "number" },
                "per_model": { "type": "array", "items": { "$ref": "#/components/schemas/SelectionMetrics" } },
                "per_provider": { "type": "array", "items": { "$ref": "#/components/schemas/SelectionMetrics" } },
                "budget": { "$ref": "#/components/schemas/BudgetSnapshot" },
                "judge": { "$ref": "#/components/schemas/JudgeMetricsSnapshot" }
            }
        },
        "JudgeMetricsSnapshot": {
            "type": "object",
            "properties": {
                "requests": { "type": "integer" },
                "successes": { "type": "integer" },
                "fallbacks": { "type": "integer" },
                "invalid_outputs": { "type": "integer" },
                "heuristic_routes": { "type": "integer" }
            },
            "required": ["requests", "successes", "fallbacks", "invalid_outputs", "heuristic_routes"]
        },
        "SelectionMetrics": {
            "type": "object",
            "properties": {
                "id": { "type": "string" },
                "requests": { "type": "integer" },
                "prompt_tokens": { "type": "integer" },
                "completion_tokens": { "type": "integer" },
                "total_tokens": { "type": "integer" },
                "estimated_cost_micros": { "type": "integer" },
                "estimated_cost_usd": { "type": "number" }
            },
            "required": ["id", "requests", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_micros", "estimated_cost_usd"]
        },
        "BudgetSnapshot": {
            "type": "object",
            "properties": {
                "accounting_backend": { "type": "string", "enum": ["disabled", "process", "file"] },
                "max_chat_requests": { "type": ["integer", "null"] },
                "max_total_tokens": { "type": ["integer", "null"] },
                "max_estimated_cost_micros": { "type": ["integer", "null"] },
                "used_chat_requests": { "type": "integer" },
                "used_total_tokens": { "type": "integer" },
                "used_estimated_cost_micros": { "type": "integer" },
                "chat_requests_remaining": { "type": ["integer", "null"] },
                "total_tokens_remaining": { "type": ["integer", "null"] },
                "estimated_cost_micros_remaining": { "type": ["integer", "null"] }
            }
        }
    })
}

fn classification_schema(labels: impl IntoIterator<Item = &'static str>) -> Value {
    json!({
        "type": "object",
        "properties": {
            "class_id": { "type": "integer" },
            "label": { "type": "string", "enum": labels.into_iter().collect::<Vec<_>>() },
            "confidence": { "type": "number" },
            "meets_threshold": { "type": "boolean" }
        },
        "required": ["class_id", "label", "confidence", "meets_threshold"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_spec_contains_core_paths() {
        let spec = spec();
        assert_eq!(spec["openapi"], "3.1.0");
        assert!(spec["paths"]["/v1/router/classify"].is_object());
        assert!(spec["paths"]["/v1/router/raw"].is_object());
        assert!(spec["paths"]["/v1/router/{provider}"].is_object());
        assert!(spec["paths"]["/v1/router/multimodel"].is_object());
        assert!(spec["paths"]["/v1/chat/completions"].is_object());
        assert!(spec["paths"]["/v1/responses"].is_object());
        assert!(spec["paths"]["/v1/embeddings"].is_object());
        assert!(spec["paths"]["/v1/images/generations"].is_object());
        assert!(spec["paths"]["/v1/audio/speech"].is_object());
        assert!(spec["paths"]["/v1/audio/transcriptions"].is_object());
        assert!(spec["paths"]["/v1/audio/translations"].is_object());
        assert!(spec["paths"]["/metrics"].is_object());
        assert!(spec["paths"]["/metrics/prometheus"].is_object());
        assert!(spec["components"]["schemas"]["MultimodelResponse"].is_object());
        assert!(spec["components"]["schemas"]["OpenAiResponsesRequest"].is_object());
        assert!(spec["components"]["schemas"]["OpenAiEmbeddingsRequest"].is_object());
        assert!(spec["components"]["schemas"]["OpenAiImagesRequest"].is_object());
        assert!(spec["components"]["schemas"]["OpenAiSpeechRequest"].is_object());
        assert!(spec["components"]["schemas"]["OpenAiAudioMultipartRequest"].is_object());
        assert!(spec["components"]["schemas"]["ProviderHealthObservation"].is_object());
        assert!(spec["components"]["schemas"]["ModalityClassification"].is_object());
        assert!(spec["components"]["schemas"]["SafetyClassification"].is_object());
        assert!(spec["components"]["schemas"]["CacheabilityClassification"].is_object());
        assert!(spec["components"]["schemas"]["LatencySensitivityClassification"].is_object());
        assert!(spec["components"]["schemas"]["ReasoningDepthClassification"].is_object());
        assert!(spec["components"]["schemas"]["JudgeMetricsSnapshot"].is_object());
        assert_eq!(
            spec["paths"]["/v1/models"]["get"]["responses"]["401"]["$ref"],
            "#/components/responses/Unauthorized"
        );
        assert_eq!(
            spec["components"]["responses"]["Unauthorized"]["headers"]["WWW-Authenticate"]["schema"]
                ["const"],
            "Bearer"
        );
        assert!(
            spec["components"]["schemas"]["RouterPolicy"]["enum"]
                .as_array()
                .expect("router policy enum")
                .iter()
                .any(|value| value == "privacy_first")
        );
        assert!(
            spec["components"]["schemas"]["RouteScoreComponents"]["properties"]
                ["local_score_boost"]
                .is_object()
        );
        assert!(
            spec["components"]["schemas"]["RouteScoreComponents"]["properties"]
                ["multimodal_score_boost"]
                .is_object()
        );
    }
}
