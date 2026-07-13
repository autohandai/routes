use anyhow::Result;
use autohand_router::{
    RouterConfig,
    classifier::SmartClassifier,
    classifier_gate::{ClassifierLiveGateConfig, run_classifier_live_gate},
    config_schema,
    conformance::{run_provider_conformance, run_provider_conformance_matrix},
    deployment_gate::{DeploymentLiveGateConfig, run_budget_worker, run_deployment_live_gate},
    eval::{
        EvalCoverageMinimums, calibrate_thresholds, configured_eval_gate, eval_gate_with_coverage,
        evaluate, load_jsonl, optimize_with_artifact, seeded_holdout,
    },
    evidence::{ControlledEvidenceConfig, validate_evidence_bundle, write_controlled_evidence},
    judge::run_judge_smoke,
    load::{
        LoadSuiteConfig, LoadTestConfig, default_load_suite_scenarios, default_multimodel_body,
        run_load_suite, run_load_test,
    },
    openapi,
    promotion::evaluate_provider_promotion_gate,
    router::RoutingEngine,
    runtime_gate::run_runtime_gate,
    server::{self, AppState},
    stream_gate::{StreamLiveGateConfig, run_stream_live_gate},
    types::{ClassifyResponse, MultimodelRequest, RouterPolicy, SelectedClassifications},
};
use clap::{Parser, Subcommand};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
#[command(name = "routes")]
#[command(about = "OpenAI-compatible Rust LLM router for local and hosted providers")]
struct Cli {
    #[arg(
        long,
        env = "AUTOHAND_ROUTER_CONFIG",
        default_value = "examples/router.yaml"
    )]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve,
    Validate,
    Openapi,
    ConfigSchema,
    InitConfig {
        #[arg(default_value = "router.yaml")]
        output: PathBuf,
    },
    Classify {
        input: String,
    },
    Route {
        input: String,
        #[arg(long)]
        policy: Option<RouterPolicyArg>,
        #[arg(long = "allow-model")]
        allowed_models: Vec<String>,
        #[arg(long = "allow-provider")]
        allowed_providers: Vec<String>,
    },
    Eval {
        dataset: PathBuf,
    },
    EvalGate {
        dataset: PathBuf,
        #[arg(long, default_value_t = 24)]
        min_examples: usize,
        #[arg(long, default_value_t = 0.90)]
        min_accuracy: f32,
        #[arg(long, default_value_t = 0.90)]
        min_domain_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_model_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_provider_accuracy: f32,
        #[arg(long, default_value_t = 1)]
        min_domain_examples: usize,
        #[arg(long, default_value_t = 1)]
        min_model_examples: usize,
        #[arg(long, default_value_t = 1)]
        min_provider_examples: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ConfiguredEvalGate {
        dataset: PathBuf,
        #[arg(long, default_value_t = 10)]
        min_examples: usize,
        #[arg(long, default_value_t = 0.90)]
        min_accuracy: f32,
        #[arg(long, default_value_t = 0.90)]
        min_domain_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_model_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_provider_accuracy: f32,
        #[arg(long, default_value_t = 1)]
        min_domain_examples: usize,
        #[arg(long, default_value_t = 1)]
        min_model_examples: usize,
        #[arg(long, default_value_t = 1)]
        min_provider_examples: usize,
        #[arg(long, default_value_t = 0.0)]
        max_fallback_rate: f32,
        #[arg(long, default_value_t = 0.20)]
        holdout_ratio: f32,
        #[arg(long, default_value_t = 0xA17E_2026)]
        holdout_seed: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    RuntimeGate {
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ControlledEvidence {
        #[arg(long, env = "GITHUB_SHA", default_value = "working-tree")]
        revision: String,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long, default_value_t = 20)]
        requests_per_scenario: u64,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        #[arg(long, default_value_t = 250)]
        slo_p95_ms: u64,
        #[arg(long, default_value_t = 0.0)]
        slo_error_rate: f64,
        #[arg(long, default_value = "artifacts/release-evidence")]
        output_dir: PathBuf,
    },
    EvidenceValidate {
        #[arg(default_value = "artifacts/release-evidence")]
        directory: PathBuf,
        #[arg(long)]
        expected_revision: Option<String>,
    },
    Calibrate {
        dataset: PathBuf,
        #[arg(long)]
        write_config: Option<PathBuf>,
    },
    Optimize {
        dataset: PathBuf,
        #[arg(long)]
        write_config: Option<PathBuf>,
        #[arg(long)]
        artifact: Option<PathBuf>,
    },
    LoadTest {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, default_value = "/v1/router/multimodel")]
        path: String,
        #[arg(long, default_value_t = 100)]
        requests: u64,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long, default_value_t = 250)]
        slo_p95_ms: u64,
        #[arg(long, default_value_t = 0.001)]
        slo_error_rate: f64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    LoadSuite {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, default_value_t = 100)]
        requests_per_scenario: u64,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long, default_value_t = 250)]
        slo_p95_ms: u64,
        #[arg(long, default_value_t = 0.001)]
        slo_error_rate: f64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    JudgeSmoke {
        #[arg(default_value = "Design a production Rust LLM router with provider failover")]
        input: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ProviderConformance {
        model: String,
        #[arg(default_value = "Verify provider adapter conformance")]
        input: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ProviderConformanceMatrix {
        #[arg(default_value = "Verify provider adapter conformance")]
        input: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ProviderPromotionGate {
        artifact: PathBuf,
        #[arg(long, default_value_t = 86_400)]
        max_age_seconds: u64,
        #[arg(long)]
        allow_unreported_versions: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    ClassifierLiveGate {
        dataset: PathBuf,
        #[arg(long, env = "GITHUB_SHA", default_value = "working-tree")]
        revision: String,
        #[arg(long, default_value_t = 5)]
        smoke_runs: usize,
        #[arg(long, default_value_t = 0.95)]
        min_adapter_success_rate: f64,
        #[arg(long, default_value_t = 0.05)]
        max_fallback_rate: f64,
        #[arg(long, default_value_t = 5_000)]
        max_smoke_p95_ms: u64,
        #[arg(long, default_value_t = 0.20)]
        holdout_ratio: f32,
        #[arg(long, default_value_t = 0xA17E_2026)]
        holdout_seed: u64,
        #[arg(long, default_value_t = 10)]
        min_holdout_examples: usize,
        #[arg(long, default_value_t = 0.90)]
        min_tier_accuracy: f32,
        #[arg(long, default_value_t = 0.90)]
        min_domain_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_model_accuracy: f32,
        #[arg(long, default_value_t = 0.0)]
        min_provider_accuracy: f32,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    StreamLiveGate {
        #[arg(long, env = "GITHUB_SHA", default_value = "working-tree")]
        revision: String,
        #[arg(long, default_value_t = 5_000)]
        max_first_chunk_ms: u64,
        #[arg(long, default_value_t = 30_000)]
        max_completion_ms: u64,
        #[arg(long, default_value_t = 5_000)]
        cancellation_timeout_ms: u64,
        #[arg(long, default_value_t = 5_000)]
        shutdown_timeout_ms: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    DeploymentLiveGate {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, env = "AUTOHAND_ROUTER_TOKEN", hide_env_values = true)]
        bearer_token: Option<String>,
        #[arg(long, env = "GITHUB_SHA", default_value = "working-tree")]
        revision: String,
        #[arg(long, default_value_t = 300)]
        duration_seconds: u64,
        #[arg(long, default_value_t = 4)]
        requests_per_second: u64,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long, default_value_t = 25)]
        min_samples_per_scenario: u64,
        #[arg(long, default_value_t = 5_000)]
        max_p95_ms: u64,
        #[arg(long, default_value_t = 10_000)]
        max_p99_ms: u64,
        #[arg(long, default_value_t = 0.01)]
        max_error_rate: f64,
        #[arg(long, default_value_t = 1_000)]
        max_queue_p95_ms: u64,
        #[arg(long, default_value_t = 1_073_741_824)]
        max_peak_rss_bytes: u64,
        #[arg(long)]
        allow_missing_rss: bool,
        #[arg(long)]
        allow_unreported_target_revision: bool,
        #[arg(long, default_value_t = 10_000)]
        max_recovery_ms: u64,
        #[arg(long, default_value_t = 4)]
        accounting_processes: usize,
        #[arg(long, default_value_t = 100)]
        accounting_limit: u64,
        #[arg(long, default_value_t = 67_108_864)]
        max_upload_probe_bytes: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(hide = true)]
    DeploymentBudgetWorker {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        limit: u64,
        #[arg(long)]
        attempts: u64,
        #[arg(long)]
        worker_id: String,
        #[arg(long)]
        start_file: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum RouterPolicyArg {
    Balanced,
    LowestCostAcceptable,
    FastestHealthy,
    HighestQuality,
    LocalFirst,
    PrivacyFirst,
    MultimodalFirst,
    Floor,
    Nitro,
    Quality,
    CostEfficient,
    CapabilityHeavy,
    DomainSkills,
}

impl From<RouterPolicyArg> for RouterPolicy {
    fn from(value: RouterPolicyArg) -> Self {
        match value {
            RouterPolicyArg::Balanced => Self::Balanced,
            RouterPolicyArg::LowestCostAcceptable => Self::LowestCostAcceptable,
            RouterPolicyArg::FastestHealthy => Self::FastestHealthy,
            RouterPolicyArg::HighestQuality => Self::HighestQuality,
            RouterPolicyArg::LocalFirst => Self::LocalFirst,
            RouterPolicyArg::PrivacyFirst => Self::PrivacyFirst,
            RouterPolicyArg::MultimodalFirst => Self::MultimodalFirst,
            RouterPolicyArg::Floor => Self::Floor,
            RouterPolicyArg::Nitro => Self::Nitro,
            RouterPolicyArg::Quality => Self::Quality,
            RouterPolicyArg::CostEfficient => Self::CostEfficient,
            RouterPolicyArg::CapabilityHeavy => Self::CapabilityHeavy,
            RouterPolicyArg::DomainSkills => Self::DomainSkills,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Validate => {
            let config = RouterConfig::from_path(&cli.config)?;
            println!(
                "valid config: {} provider(s), {} model(s)",
                config.providers.len(),
                config.models.len()
            );
            Ok(())
        }
        Command::Openapi => {
            println!("{}", serde_json::to_string_pretty(&openapi::spec())?);
            Ok(())
        }
        Command::ConfigSchema => {
            println!(
                "{}",
                serde_json::to_string_pretty(&config_schema::schema())?
            );
            Ok(())
        }
        Command::InitConfig { output } => {
            fs::write(&output, include_str!("../examples/router.yaml"))?;
            println!("wrote {}", output.display());
            Ok(())
        }
        Command::Serve => {
            let config = RouterConfig::from_path(&cli.config)?;
            let bind = config.bind.clone();
            let state = AppState::from_config(&config)?;
            server::serve(state, &bind).await
        }
        Command::Classify { input } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let classifier = SmartClassifier::new(config.clone())?;
            let engine = RoutingEngine::new(config.clone(), classifier);
            let response = ClassifyResponse {
                classifications: SelectedClassifications::from_heads(
                    engine.classify(&input).await,
                    &[],
                ),
            };
            println!("{}", serde_json::to_string_pretty(&response)?);
            Ok(())
        }
        Command::Route {
            input,
            policy,
            allowed_models,
            allowed_providers,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let classifier = SmartClassifier::new(config.clone())?;
            let engine = RoutingEngine::new(config.clone(), classifier);
            let response = engine
                .route(MultimodelRequest {
                    input,
                    allowed_models,
                    allowed_providers,
                    required_capabilities: Vec::new(),
                    policy: policy.map(Into::into).unwrap_or(config.policy),
                    default_model: None,
                    max_output_tokens: None,
                })
                .await;
            println!("{}", serde_json::to_string_pretty(&response)?);
            Ok(())
        }
        Command::Eval { dataset } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let classifier = SmartClassifier::new(config.clone())?;
            let engine = RoutingEngine::new(config, classifier);
            let examples = load_jsonl(dataset)?;
            let report = evaluate(&engine, &examples).await;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::EvalGate {
            dataset,
            min_examples,
            min_accuracy,
            min_domain_accuracy,
            min_model_accuracy,
            min_provider_accuracy,
            min_domain_examples,
            min_model_examples,
            min_provider_examples,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let examples = load_jsonl(&dataset)?;
            let report = eval_gate_with_coverage(
                &config,
                &dataset,
                &examples,
                min_examples,
                min_accuracy,
                min_domain_accuracy,
                min_model_accuracy,
                min_provider_accuracy,
                EvalCoverageMinimums {
                    domain: min_domain_examples,
                    model: min_model_examples,
                    provider: min_provider_examples,
                },
            )
            .await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote eval-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!("eval-gate failed: {}", report.failures.join("; "));
            }
            Ok(())
        }
        Command::ConfiguredEvalGate {
            dataset,
            min_examples,
            min_accuracy,
            min_domain_accuracy,
            min_model_accuracy,
            min_provider_accuracy,
            min_domain_examples,
            min_model_examples,
            min_provider_examples,
            max_fallback_rate,
            holdout_ratio,
            holdout_seed,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let examples = load_jsonl(&dataset)?;
            let holdout = seeded_holdout(&examples, holdout_ratio, holdout_seed)?;
            let mut report = configured_eval_gate(
                &config,
                &dataset,
                &holdout,
                min_examples,
                min_accuracy,
                min_domain_accuracy,
                min_model_accuracy,
                min_provider_accuracy,
                EvalCoverageMinimums {
                    domain: min_domain_examples,
                    model: min_model_examples,
                    provider: min_provider_examples,
                },
                max_fallback_rate,
            )
            .await?;
            report.selection.strategy = format!("seeded_holdout_{holdout_ratio:.4}");
            report.selection.seed = Some(holdout_seed);
            report.selection.source_examples = examples.len();
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote configured eval-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "configured eval-gate failed: {}",
                    report.failures.join("; ")
                );
            }
            Ok(())
        }
        Command::RuntimeGate { output } => {
            let report = run_runtime_gate().await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote runtime-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!("runtime-gate failed: {}", report.failures.join("; "));
            }
            Ok(())
        }
        Command::ControlledEvidence {
            revision,
            runs,
            requests_per_scenario,
            concurrency,
            slo_p95_ms,
            slo_error_rate,
            output_dir,
        } => {
            let manifest = write_controlled_evidence(ControlledEvidenceConfig {
                revision,
                output_dir,
                runs,
                requests_per_scenario,
                concurrency,
                slo_p95_ms,
                slo_error_rate,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::EvidenceValidate {
            directory,
            expected_revision,
        } => {
            validate_evidence_bundle(&directory, expected_revision.as_deref())?;
            println!("valid evidence bundle: {}", directory.display());
            Ok(())
        }
        Command::Calibrate {
            dataset,
            write_config,
        } => {
            let mut config = RouterConfig::from_path(&cli.config)?;
            let examples = load_jsonl(dataset)?;
            let report = calibrate_thresholds(&config, &examples).await?;
            if let Some(path) = write_config {
                config.classifier.easy_threshold = report.easy_threshold;
                config.classifier.hard_threshold = report.hard_threshold;
                fs::write(&path, serde_yaml::to_string(&config)?)?;
                println!("wrote calibrated config {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Optimize {
            dataset,
            write_config,
            artifact,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let examples = load_jsonl(&dataset)?;
            let (optimized, report_artifact) = optimize_with_artifact(
                &config,
                &cli.config,
                &dataset,
                &examples,
                write_config.clone(),
            )
            .await?;
            if let Some(path) = write_config {
                fs::write(&path, serde_yaml::to_string(&optimized)?)?;
                println!("wrote optimized config {}", path.display());
            }
            if let Some(path) = artifact {
                fs::write(&path, serde_json::to_string_pretty(&report_artifact)?)?;
                println!("wrote optimization artifact {}", path.display());
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&report_artifact.optimized_report)?
            );
            Ok(())
        }
        Command::LoadTest {
            url,
            path,
            requests,
            concurrency,
            slo_p95_ms,
            slo_error_rate,
            output,
        } => {
            let report = run_load_test(LoadTestConfig {
                base_url: url,
                path,
                requests,
                concurrency,
                slo_p95_ms,
                slo_error_rate,
                body: default_multimodel_body(),
            })
            .await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote load-test report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.slo.pass {
                anyhow::bail!(
                    "load-test SLO failed: p95={}ms threshold={}ms error_rate={} threshold={}",
                    report.latency.p95_ms,
                    report.slo.p95_ms_threshold,
                    report.error_rate,
                    report.slo.error_rate_threshold
                );
            }
            Ok(())
        }
        Command::LoadSuite {
            url,
            requests_per_scenario,
            concurrency,
            slo_p95_ms,
            slo_error_rate,
            output,
        } => {
            let report = run_load_suite(LoadSuiteConfig {
                base_url: url,
                requests_per_scenario,
                concurrency,
                slo_p95_ms,
                slo_error_rate,
                scenarios: default_load_suite_scenarios(),
            })
            .await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote load-suite report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "load-suite SLO failed: {} of {} scenario(s) passed",
                    report
                        .reports
                        .iter()
                        .filter(|entry| entry.report.slo.pass)
                        .count(),
                    report.reports.len()
                );
            }
            Ok(())
        }
        Command::JudgeSmoke { input, output } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let report = run_judge_smoke(config, input).await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote judge-smoke report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "judge smoke failed: requests={} successes={} fallbacks={} heuristic_routes={}",
                    report
                        .metrics_after
                        .requests
                        .saturating_sub(report.metrics_before.requests),
                    report
                        .metrics_after
                        .successes
                        .saturating_sub(report.metrics_before.successes),
                    report
                        .metrics_after
                        .fallbacks
                        .saturating_sub(report.metrics_before.fallbacks),
                    report
                        .metrics_after
                        .heuristic_routes
                        .saturating_sub(report.metrics_before.heuristic_routes)
                );
            }
            Ok(())
        }
        Command::ProviderConformance {
            model,
            input,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let report = run_provider_conformance(config, model, input).await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote provider conformance report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "provider conformance failed: provider={} model={} status={} openai_chat_shape={}",
                    report.provider,
                    report.model,
                    report.chat.status,
                    report.chat.openai_chat_shape
                );
            }
            Ok(())
        }
        Command::ProviderConformanceMatrix { input, output } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let report = run_provider_conformance_matrix(config, input).await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!(
                    "wrote provider conformance matrix report {}",
                    path.display()
                );
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "provider conformance matrix failed: passed={} failed={} total={}",
                    report.passed,
                    report.failed,
                    report.total
                );
            }
            Ok(())
        }
        Command::ProviderPromotionGate {
            artifact,
            max_age_seconds,
            allow_unreported_versions,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let evaluated_unix_seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or_default();
            let report = evaluate_provider_promotion_gate(
                &config,
                &artifact,
                evaluated_unix_seconds,
                max_age_seconds,
                !allow_unreported_versions,
            )?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote provider promotion report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "provider promotion gate failed: {}",
                    report.failures.join("; ")
                );
            }
            Ok(())
        }
        Command::ClassifierLiveGate {
            dataset,
            revision,
            smoke_runs,
            min_adapter_success_rate,
            max_fallback_rate,
            max_smoke_p95_ms,
            holdout_ratio,
            holdout_seed,
            min_holdout_examples,
            min_tier_accuracy,
            min_domain_accuracy,
            min_model_accuracy,
            min_provider_accuracy,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let report = run_classifier_live_gate(
                &config,
                &dataset,
                ClassifierLiveGateConfig {
                    revision,
                    smoke_runs,
                    min_adapter_success_rate,
                    max_fallback_rate,
                    max_smoke_p95_ms,
                    holdout_ratio,
                    holdout_seed,
                    min_holdout_examples,
                    min_tier_accuracy,
                    min_domain_accuracy,
                    min_model_accuracy,
                    min_provider_accuracy,
                },
            )
            .await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote classifier live-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "classifier live gate failed: {}",
                    report.failures.join("; ")
                );
            }
            Ok(())
        }
        Command::StreamLiveGate {
            revision,
            max_first_chunk_ms,
            max_completion_ms,
            cancellation_timeout_ms,
            shutdown_timeout_ms,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let report = run_stream_live_gate(
                &config,
                StreamLiveGateConfig {
                    revision,
                    max_first_chunk_ms,
                    max_completion_ms,
                    cancellation_timeout_ms,
                    shutdown_timeout_ms,
                },
            )
            .await?;
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote stream live-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!("stream live gate failed: {}", report.failures.join("; "));
            }
            Ok(())
        }
        Command::DeploymentLiveGate {
            url,
            bearer_token,
            revision,
            duration_seconds,
            requests_per_second,
            concurrency,
            min_samples_per_scenario,
            max_p95_ms,
            max_p99_ms,
            max_error_rate,
            max_queue_p95_ms,
            max_peak_rss_bytes,
            allow_missing_rss,
            allow_unreported_target_revision,
            max_recovery_ms,
            accounting_processes,
            accounting_limit,
            max_upload_probe_bytes,
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let failure_revision = revision.clone();
            let failure_output = output.clone();
            let report = match run_deployment_live_gate(
                &config,
                DeploymentLiveGateConfig {
                    base_url: url,
                    bearer_token,
                    revision,
                    duration: std::time::Duration::from_secs(duration_seconds),
                    requests_per_second,
                    concurrency,
                    min_samples_per_scenario,
                    max_p95_ms,
                    max_p99_ms,
                    max_error_rate,
                    max_queue_p95_ms,
                    max_peak_rss_bytes,
                    require_rss: !allow_missing_rss,
                    require_target_revision: !allow_unreported_target_revision,
                    max_recovery_ms,
                    accounting_processes,
                    accounting_limit,
                    max_upload_probe_bytes,
                    worker_executable: std::env::current_exe()?,
                },
            )
            .await
            {
                Ok(report) => report,
                Err(error) => {
                    if let Some(path) = failure_output {
                        fs::write(
                            &path,
                            serde_json::to_string_pretty(&serde_json::json!({
                                "schema_version": 1,
                                "artifact_kind": "staging_deployment_live_gate_error",
                                "source_revision": failure_revision,
                                "payloads_redacted": true,
                                "pass": false,
                                "error": error.to_string()
                            }))?,
                        )?;
                        println!("wrote deployment live-gate failure {}", path.display());
                    }
                    return Err(error);
                }
            };
            if let Some(path) = output {
                fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("wrote deployment live-gate report {}", path.display());
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.pass {
                anyhow::bail!(
                    "deployment live gate failed: {}",
                    report.failures.join("; ")
                );
            }
            Ok(())
        }
        Command::DeploymentBudgetWorker {
            ledger,
            limit,
            attempts,
            worker_id,
            start_file,
            output,
        } => {
            run_budget_worker(&ledger, limit, attempts, worker_id, &start_file, &output).await?;
            Ok(())
        }
    }
}
