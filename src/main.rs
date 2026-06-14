use anyhow::Result;
use autohand_router::{
    RouterConfig,
    accounting::BudgetAccounting,
    classifier::SmartClassifier,
    conformance::{run_provider_conformance, run_provider_conformance_matrix},
    eval::{calibrate_thresholds, eval_gate, evaluate, load_jsonl, optimize_with_artifact},
    judge::run_judge_smoke,
    load::{
        LoadSuiteConfig, LoadTestConfig, default_load_suite_scenarios, default_multimodel_body,
        run_load_suite, run_load_test,
    },
    openapi,
    provider::ProviderClient,
    router::RoutingEngine,
    server::{self, AppState},
    telemetry::DecisionLogger,
    types::{ClassifyResponse, MultimodelRequest, RouterPolicy, SelectedClassifications},
};
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
#[command(name = "autohand-router")]
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
        #[arg(long)]
        output: Option<PathBuf>,
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
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum RouterPolicyArg {
    Balanced,
    CostEfficient,
    CapabilityHeavy,
    DomainSkills,
}

impl From<RouterPolicyArg> for RouterPolicy {
    fn from(value: RouterPolicyArg) -> Self {
        match value {
            RouterPolicyArg::Balanced => Self::Balanced,
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
        Command::InitConfig { output } => {
            fs::write(&output, include_str!("../examples/router.yaml"))?;
            println!("wrote {}", output.display());
            Ok(())
        }
        Command::Serve => {
            let config = RouterConfig::from_path(&cli.config)?;
            let classifier = SmartClassifier::new(config.clone())?;
            let engine = RoutingEngine::new(config.clone(), classifier);
            let bind = config.bind.clone();
            let state = AppState {
                engine,
                providers: ProviderClient::new(&config)?,
                metrics: Default::default(),
                accounting: BudgetAccounting::from_budget_config(&config.budget)?,
                telemetry: DecisionLogger::new(&config.telemetry),
            };
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
            output,
        } => {
            let config = RouterConfig::from_path(&cli.config)?;
            let examples = load_jsonl(&dataset)?;
            let report = eval_gate(
                &config,
                &dataset,
                &examples,
                min_examples,
                min_accuracy,
                min_domain_accuracy,
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
    }
}
