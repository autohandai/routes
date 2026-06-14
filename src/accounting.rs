use crate::{
    config::{BudgetAccountingBackend, BudgetConfig},
    types::ModelConfig,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

#[derive(Clone)]
pub enum BudgetAccounting {
    Process,
    File(Arc<FileBudgetLedger>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetUsageSnapshot {
    pub request_count: u64,
    pub total_tokens: u64,
    pub estimated_cost_micros: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetReservation {
    pub request_count: u64,
    pub total_tokens: u64,
    pub estimated_cost_micros: u64,
}

impl BudgetAccounting {
    pub fn from_budget_config(budget: &BudgetConfig) -> Result<Self> {
        match budget.accounting.backend {
            BudgetAccountingBackend::Process => Ok(Self::Process),
            BudgetAccountingBackend::File => {
                let path = budget
                    .accounting
                    .file_path
                    .as_deref()
                    .context("budget.accounting.file_path is required when backend is file")?;
                Ok(Self::File(Arc::new(FileBudgetLedger::new(
                    path,
                    Duration::from_millis(budget.accounting.lock_timeout_ms),
                ))))
            }
        }
    }

    pub fn reserve(&self, budget: &BudgetConfig, reservation: BudgetReservation) -> Result<()> {
        match self {
            Self::Process => Ok(()),
            Self::File(ledger) => ledger.reserve(budget, reservation),
        }
    }

    pub fn snapshot(&self) -> Option<BudgetUsageSnapshot> {
        match self {
            Self::Process => None,
            Self::File(ledger) => ledger.snapshot().ok(),
        }
    }
}

impl BudgetReservation {
    pub fn new(
        model: &ModelConfig,
        estimated_input_tokens: u32,
        requested_output_tokens: u32,
    ) -> Self {
        let total_tokens =
            u64::from(estimated_input_tokens).saturating_add(u64::from(requested_output_tokens));
        let input_cost = (u64::from(estimated_input_tokens) as f64
            * model.cost_per_million_input as f64)
            / 1_000_000.0;
        let output_cost = (u64::from(requested_output_tokens) as f64
            * model.cost_per_million_output as f64)
            / 1_000_000.0;
        let estimated_cost_micros = ((input_cost + output_cost) * 1_000_000.0).round() as u64;
        Self {
            request_count: 1,
            total_tokens,
            estimated_cost_micros,
        }
    }
}

#[derive(Debug)]
pub struct FileBudgetLedger {
    path: PathBuf,
    lock_path: PathBuf,
    lock_timeout: Duration,
}

impl FileBudgetLedger {
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

    fn reserve(&self, budget: &BudgetConfig, reservation: BudgetReservation) -> Result<()> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        let mut usage = self.read_usage()?;
        ensure_within_budget(budget, &usage, reservation)?;
        usage.request_count = usage
            .request_count
            .saturating_add(reservation.request_count);
        usage.total_tokens = usage.total_tokens.saturating_add(reservation.total_tokens);
        usage.estimated_cost_micros = usage
            .estimated_cost_micros
            .saturating_add(reservation.estimated_cost_micros);
        self.write_usage(&usage)
    }

    fn snapshot(&self) -> Result<BudgetUsageSnapshot> {
        let _lock = FileLock::acquire(&self.lock_path, self.lock_timeout)?;
        self.read_usage()
    }

    fn read_usage(&self) -> Result<BudgetUsageSnapshot> {
        match fs::read_to_string(&self.path) {
            Ok(raw) => serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse budget ledger {}", self.path.display())),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(BudgetUsageSnapshot::default()),
            Err(error) => Err(error)
                .with_context(|| format!("failed to read budget ledger {}", self.path.display())),
        }
    }

    fn write_usage(&self, usage: &BudgetUsageSnapshot) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create budget ledger dir {}", parent.display())
            })?;
        }
        let raw = serde_json::to_vec_pretty(usage).context("failed to serialize budget ledger")?;
        fs::write(&self.path, raw)
            .with_context(|| format!("failed to write budget ledger {}", self.path.display()))
    }
}

fn ensure_within_budget(
    budget: &BudgetConfig,
    usage: &BudgetUsageSnapshot,
    reservation: BudgetReservation,
) -> Result<()> {
    if let Some(limit) = budget.max_chat_requests {
        let requested = usage
            .request_count
            .saturating_add(reservation.request_count);
        anyhow::ensure!(
            requested <= limit,
            "model request budget exceeded: current={}, limit={limit}",
            usage.request_count
        );
    }
    if let Some(limit) = budget.max_total_tokens {
        let requested = usage.total_tokens.saturating_add(reservation.total_tokens);
        anyhow::ensure!(
            requested <= limit,
            "token budget exceeded: current={}, requested={}, limit={limit}",
            usage.total_tokens,
            reservation.total_tokens
        );
    }
    if let Some(limit) = budget.max_estimated_cost_micros {
        let requested = usage
            .estimated_cost_micros
            .saturating_add(reservation.estimated_cost_micros);
        anyhow::ensure!(
            requested <= limit,
            "cost budget exceeded: current_micros={}, requested_micros={}, limit_micros={limit}",
            usage.estimated_cost_micros,
            reservation.estimated_cost_micros
        );
    }
    Ok(())
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path, timeout: Duration) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create budget lock dir {}", parent.display())
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
                        "timed out acquiring budget ledger lock {}",
                        path.display()
                    );
                    sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to acquire budget ledger lock {}", path.display())
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
    use super::{BudgetAccounting, BudgetReservation};
    use crate::{
        config::{BudgetAccountingBackend, BudgetAccountingConfig, BudgetConfig},
        types::ModelConfig,
    };
    use std::{fs, time::SystemTime};

    #[test]
    fn file_accounting_enforces_shared_request_limit() {
        let path = std::env::temp_dir().join(format!(
            "autohand-router-budget-{}.json",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let budget = BudgetConfig {
            max_chat_requests: Some(1),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: BudgetAccountingConfig {
                backend: BudgetAccountingBackend::File,
                file_path: Some(path.to_string_lossy().to_string()),
                lock_timeout_ms: 1_000,
            },
        };
        let first = BudgetAccounting::from_budget_config(&budget).unwrap();
        let second = BudgetAccounting::from_budget_config(&budget).unwrap();
        let reservation = BudgetReservation::new(&test_model(), 10, 0);

        first.reserve(&budget, reservation).unwrap();
        let error = second.reserve(&budget, reservation).unwrap_err();

        assert!(error.to_string().contains("model request budget"));
        assert_eq!(first.snapshot().unwrap().request_count, 1);
        let _ = fs::remove_file(path.with_extension("json.lock"));
        let _ = fs::remove_file(path);
    }

    fn test_model() -> ModelConfig {
        ModelConfig {
            id: "priced".to_string(),
            provider: "test".to_string(),
            aliases: vec![],
            capability: 0.5,
            cost_per_million_input: 2.0,
            cost_per_million_output: 10.0,
            domains: vec![],
            context_window: None,
            local: false,
        }
    }
}
