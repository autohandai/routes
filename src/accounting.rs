use crate::{
    config::{BudgetAccountingBackend, BudgetConfig},
    file_state::{BlockingFileGate, FileLeaseLock, atomic_write},
    types::ModelConfig,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone)]
pub enum BudgetAccounting {
    Process(Arc<ProcessBudgetLedger>),
    File(Arc<FileBudgetLedger>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetUsageSnapshot {
    pub request_count: u64,
    pub total_tokens: u64,
    pub estimated_cost_micros: u64,
    #[serde(default)]
    pub by_scope: HashMap<String, BudgetScopeUsageSnapshot>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetScopeUsageSnapshot {
    pub request_count: u64,
    pub total_tokens: u64,
    pub estimated_cost_micros: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetReservation {
    pub id: uuid::Uuid,
    pub class: BudgetChargeClass,
    pub request_count: u64,
    pub total_tokens: u64,
    pub estimated_cost_micros: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetChargeClass {
    ForegroundLogicalRequest,
}

impl BudgetAccounting {
    pub fn from_budget_config(budget: &BudgetConfig) -> Result<Self> {
        match budget.accounting.backend {
            BudgetAccountingBackend::Process => {
                Ok(Self::Process(Arc::new(ProcessBudgetLedger::default())))
            }
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

    pub async fn reserve(
        &self,
        budget: &BudgetConfig,
        reservation: BudgetReservation,
    ) -> Result<()> {
        self.reserve_scoped(budget, "global", reservation).await
    }

    pub async fn reserve_scoped(
        &self,
        budget: &BudgetConfig,
        scope: &str,
        reservation: BudgetReservation,
    ) -> Result<()> {
        anyhow::ensure!(!scope.trim().is_empty(), "budget scope cannot be empty");
        let result = match self {
            Self::Process(ledger) => ledger.reserve(budget, scope, reservation),
            Self::File(ledger) => {
                ledger
                    .reserve(budget.clone(), scope.to_string(), reservation)
                    .await
            }
        };
        if result.is_ok() {
            tracing::debug!(
                reservation_id = %reservation.id,
                scope,
                class = ?reservation.class,
                tokens = reservation.total_tokens,
                cost_micros = reservation.estimated_cost_micros,
                "logical request budget reserved"
            );
        }
        result
    }

    pub async fn snapshot(&self) -> Option<BudgetUsageSnapshot> {
        match self {
            Self::Process(ledger) => ledger.snapshot().ok(),
            Self::File(ledger) => ledger.snapshot().await.ok(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ProcessBudgetLedger {
    usage: Mutex<BudgetUsageSnapshot>,
}

impl ProcessBudgetLedger {
    fn reserve(
        &self,
        budget: &BudgetConfig,
        scope: &str,
        reservation: BudgetReservation,
    ) -> Result<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| anyhow::anyhow!("process budget ledger lock poisoned"))?;
        let scoped = usage.by_scope.entry(scope.to_string()).or_default();
        ensure_within_budget(budget, scoped, reservation)?;
        add_scope_reservation(scoped, reservation);
        add_reservation(&mut usage, reservation);
        Ok(())
    }

    fn snapshot(&self) -> Result<BudgetUsageSnapshot> {
        self.usage
            .lock()
            .map(|usage| usage.clone())
            .map_err(|_| anyhow::anyhow!("process budget ledger lock poisoned"))
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
            id: uuid::Uuid::new_v4(),
            class: BudgetChargeClass::ForegroundLogicalRequest,
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
    blocking: BlockingFileGate,
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
            blocking: BlockingFileGate::default(),
        }
    }

    async fn reserve(
        self: &Arc<Self>,
        budget: BudgetConfig,
        scope: String,
        reservation: BudgetReservation,
    ) -> Result<()> {
        let store = Arc::clone(self);
        self.blocking
            .run(move || store.reserve_blocking(&budget, &scope, reservation))
            .await
    }

    async fn snapshot(self: &Arc<Self>) -> Result<BudgetUsageSnapshot> {
        let store = Arc::clone(self);
        self.blocking.run(move || store.snapshot_blocking()).await
    }

    fn reserve_blocking(
        &self,
        budget: &BudgetConfig,
        scope: &str,
        reservation: BudgetReservation,
    ) -> Result<()> {
        let _lock = FileLeaseLock::acquire(&self.lock_path, self.lock_timeout, "budget ledger")?;
        let mut usage = self.read_usage()?;
        let scoped = usage.by_scope.entry(scope.to_string()).or_default();
        ensure_within_budget(budget, scoped, reservation)?;
        add_scope_reservation(scoped, reservation);
        add_reservation(&mut usage, reservation);
        self.write_usage(&usage)
    }

    fn snapshot_blocking(&self) -> Result<BudgetUsageSnapshot> {
        let _lock = FileLeaseLock::acquire(&self.lock_path, self.lock_timeout, "budget ledger")?;
        self.read_usage()
    }

    fn read_usage(&self) -> Result<BudgetUsageSnapshot> {
        match fs::read_to_string(&self.path) {
            Ok(raw) => {
                let mut usage =
                    serde_json::from_str::<BudgetUsageSnapshot>(&raw).with_context(|| {
                        format!("failed to parse budget ledger {}", self.path.display())
                    })?;
                if usage.by_scope.is_empty()
                    && (usage.request_count > 0
                        || usage.total_tokens > 0
                        || usage.estimated_cost_micros > 0)
                {
                    usage.by_scope.insert(
                        "global".to_string(),
                        BudgetScopeUsageSnapshot {
                            request_count: usage.request_count,
                            total_tokens: usage.total_tokens,
                            estimated_cost_micros: usage.estimated_cost_micros,
                        },
                    );
                }
                Ok(usage)
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(BudgetUsageSnapshot::default()),
            Err(error) => Err(error)
                .with_context(|| format!("failed to read budget ledger {}", self.path.display())),
        }
    }

    fn write_usage(&self, usage: &BudgetUsageSnapshot) -> Result<()> {
        let raw = serde_json::to_vec_pretty(usage).context("failed to serialize budget ledger")?;
        atomic_write(&self.path, &raw, "budget ledger")
    }
}

fn add_reservation(usage: &mut BudgetUsageSnapshot, reservation: BudgetReservation) {
    usage.request_count = usage
        .request_count
        .saturating_add(reservation.request_count);
    usage.total_tokens = usage.total_tokens.saturating_add(reservation.total_tokens);
    usage.estimated_cost_micros = usage
        .estimated_cost_micros
        .saturating_add(reservation.estimated_cost_micros);
}

fn add_scope_reservation(usage: &mut BudgetScopeUsageSnapshot, reservation: BudgetReservation) {
    usage.request_count = usage
        .request_count
        .saturating_add(reservation.request_count);
    usage.total_tokens = usage.total_tokens.saturating_add(reservation.total_tokens);
    usage.estimated_cost_micros = usage
        .estimated_cost_micros
        .saturating_add(reservation.estimated_cost_micros);
}

fn ensure_within_budget(
    budget: &BudgetConfig,
    usage: &BudgetScopeUsageSnapshot,
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

#[cfg(test)]
mod tests {
    use super::{BudgetAccounting, BudgetReservation};
    use crate::{
        config::{
            BudgetAccountingBackend, BudgetAccountingConfig, BudgetAccountingScope, BudgetConfig,
        },
        file_state::FileLeaseLock,
        types::ModelConfig,
    };
    use std::{
        fs,
        process::{Command, Stdio},
        time::Duration,
    };

    #[tokio::test]
    async fn file_accounting_enforces_shared_request_limit() {
        let path = temp_path("shared-limit");
        let budget = BudgetConfig {
            max_chat_requests: Some(1),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: BudgetAccountingConfig {
                backend: BudgetAccountingBackend::File,
                file_path: Some(path.to_string_lossy().to_string()),
                lock_timeout_ms: 1_000,
                ..Default::default()
            },
        };
        let first = BudgetAccounting::from_budget_config(&budget).unwrap();
        let second = BudgetAccounting::from_budget_config(&budget).unwrap();
        let reservation = BudgetReservation::new(&test_model(), 10, 0);

        first.reserve(&budget, reservation).await.unwrap();
        let error = second.reserve(&budget, reservation).await.unwrap_err();

        assert!(error.to_string().contains("model request budget"));
        assert_eq!(first.snapshot().await.unwrap().request_count, 1);
        let _ = fs::remove_file(path.with_extension("json.lock"));
        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn credential_scopes_are_independent_and_survive_file_ledger_restart() {
        let path = temp_path("credential-scopes");
        let budget = BudgetConfig {
            max_chat_requests: Some(1),
            max_total_tokens: Some(20),
            max_estimated_cost_micros: None,
            accounting: BudgetAccountingConfig {
                backend: BudgetAccountingBackend::File,
                file_path: Some(path.to_string_lossy().to_string()),
                lock_timeout_ms: 1_000,
                scope: BudgetAccountingScope::Credential,
                ..Default::default()
            },
        };
        let reservation = BudgetReservation::new(&test_model(), 10, 0);
        let first = BudgetAccounting::from_budget_config(&budget).unwrap();
        first
            .reserve_scoped(&budget, "credential-0", reservation)
            .await
            .unwrap();
        first
            .reserve_scoped(&budget, "credential-1", reservation)
            .await
            .unwrap();
        assert!(
            first
                .reserve_scoped(&budget, "credential-0", reservation)
                .await
                .is_err()
        );

        let restarted = BudgetAccounting::from_budget_config(&budget).unwrap();
        let snapshot = restarted.snapshot().await.unwrap();
        assert_eq!(snapshot.request_count, 2);
        assert_eq!(snapshot.by_scope["credential-0"].request_count, 1);
        assert_eq!(snapshot.by_scope["credential-1"].request_count, 1);
        assert!(
            restarted
                .reserve_scoped(&budget, "credential-1", reservation)
                .await
                .is_err()
        );
        cleanup(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn process_accounting_reserves_concurrent_requests_atomically() {
        let budget = BudgetConfig {
            max_chat_requests: Some(1),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: Default::default(),
        };
        let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();
        let reservation = BudgetReservation::new(&test_model(), 10, 0);
        let handles = (0..8)
            .map(|_| {
                let accounting = accounting.clone();
                let budget = budget.clone();
                tokio::spawn(async move { accounting.reserve(&budget, reservation).await.is_ok() })
            })
            .collect::<Vec<_>>();
        let mut successes = 0;
        for handle in handles {
            if handle.await.unwrap() {
                successes += 1;
            }
        }

        assert_eq!(successes, 1);
        assert_eq!(accounting.snapshot().await.unwrap().request_count, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn file_accounting_serializes_multithreaded_contention() {
        let path = temp_path("contention");
        let mut budget = file_budget(&path, 16);
        // This test proves serialization, not timeout behavior. Give heavily loaded
        // hosted runners enough time for every fsync-backed writer to take the lock.
        budget.accounting.lock_timeout_ms = 30_000;
        let reservation = BudgetReservation::new(&test_model(), 10, 0);
        let handles = (0..16)
            .map(|_| {
                let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();
                let budget = budget.clone();
                tokio::spawn(async move { accounting.reserve(&budget, reservation).await })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.await.unwrap().unwrap();
        }
        let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();
        assert_eq!(accounting.snapshot().await.unwrap().request_count, 16);
        cleanup(&path);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn contended_file_accounting_does_not_starve_the_tokio_worker() {
        let path = temp_path("worker-starvation");
        let budget = file_budget(&path, 1);
        let lock_path = path.with_extension("json.lock");
        let held_lock =
            FileLeaseLock::acquire(&lock_path, Duration::from_secs(1), "budget ledger").unwrap();
        let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();
        let task_budget = budget.clone();
        let task = tokio::spawn(async move {
            accounting
                .reserve(&task_budget, BudgetReservation::new(&test_model(), 10, 0))
                .await
        });

        tokio::time::timeout(
            Duration::from_millis(100),
            tokio::time::sleep(Duration::from_millis(20)),
        )
        .await
        .expect("Tokio timer was starved by file lock contention");
        assert!(!task.is_finished());
        drop(held_lock);
        task.await.unwrap().unwrap();
        cleanup(&path);
    }

    #[tokio::test]
    async fn corrupt_budget_ledger_fails_closed() {
        let path = temp_path("corrupt");
        fs::write(&path, b"{partial").unwrap();
        let budget = file_budget(&path, 1);
        let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();

        let error = accounting
            .reserve(&budget, BudgetReservation::new(&test_model(), 10, 0))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("failed to parse budget ledger"));
        assert_eq!(fs::read(&path).unwrap(), b"{partial");
        cleanup(&path);
    }

    #[test]
    fn file_budget_is_enforced_across_processes() {
        if std::env::var_os("AUTOHAND_BUDGET_CHILD_PATH").is_some() {
            return;
        }
        let path = temp_path("cross-process");
        run_budget_child(&path, true);
        run_budget_child(&path, false);
        cleanup(&path);
    }

    #[tokio::test]
    async fn budget_child_process_reservation() {
        let Some(path) = std::env::var_os("AUTOHAND_BUDGET_CHILD_PATH") else {
            return;
        };
        let expect_success = std::env::var("AUTOHAND_BUDGET_CHILD_SUCCESS").unwrap() == "true";
        let path = std::path::PathBuf::from(path);
        let budget = file_budget(&path, 1);
        let accounting = BudgetAccounting::from_budget_config(&budget).unwrap();
        let result = accounting
            .reserve(&budget, BudgetReservation::new(&test_model(), 10, 0))
            .await;
        assert_eq!(
            result.is_ok(),
            expect_success,
            "reservation result: {result:?}"
        );
    }

    fn run_budget_child(path: &std::path::Path, expect_success: bool) {
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "accounting::tests::budget_child_process_reservation",
                "--nocapture",
            ])
            .env("AUTOHAND_BUDGET_CHILD_PATH", path)
            .env("AUTOHAND_BUDGET_CHILD_SUCCESS", expect_success.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "budget child process failed: {status}");
    }

    fn file_budget(path: &std::path::Path, limit: u64) -> BudgetConfig {
        BudgetConfig {
            max_chat_requests: Some(limit),
            max_total_tokens: None,
            max_estimated_cost_micros: None,
            accounting: BudgetAccountingConfig {
                backend: BudgetAccountingBackend::File,
                file_path: Some(path.to_string_lossy().to_string()),
                lock_timeout_ms: 1_000,
                ..Default::default()
            },
        }
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "autohand-router-budget-{label}-{}.json",
            uuid::Uuid::new_v4()
        ))
    }

    fn cleanup(path: &std::path::Path) {
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
            capabilities: Default::default(),
            local: false,
        }
    }
}
