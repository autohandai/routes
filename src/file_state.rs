use anyhow::{Context, Result};
use fs4::{FileExt, TryLockError};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(10);
const MINIMUM_LEASE: Duration = Duration::from_secs(30);
const MAX_BLOCKING_FILE_OPERATIONS: usize = 4;
static FILE_OPERATION_PERMITS: OnceLock<Arc<Semaphore>> = OnceLock::new();

#[derive(Debug, Clone)]
pub(crate) struct BlockingFileGate {
    permits: Arc<Semaphore>,
}

impl Default for BlockingFileGate {
    fn default() -> Self {
        Self {
            permits: Arc::clone(
                FILE_OPERATION_PERMITS
                    .get_or_init(|| Arc::new(Semaphore::new(MAX_BLOCKING_FILE_OPERATIONS))),
            ),
        }
    }
}

impl BlockingFileGate {
    pub(crate) async fn run<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .context("file-state blocking worker closed")?;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            operation()
        })
        .await
        .context("file-state blocking worker panicked")?
    }
}

#[derive(Debug, Serialize)]
struct LockOwner {
    pid: u32,
    owner_id: String,
    acquired_unix_ms: u64,
    lease_expires_unix_ms: u64,
}

pub(crate) struct FileLeaseLock {
    file: File,
}

impl FileLeaseLock {
    pub(crate) fn acquire(path: &Path, timeout: Duration, purpose: &str) -> Result<Self> {
        create_parent_dir(path, purpose)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("failed to open {purpose} lock {}", path.display()))?;
        let started = Instant::now();
        loop {
            match try_lock_exclusive(&file) {
                Ok(true) => break,
                Ok(false) => {
                    anyhow::ensure!(
                        started.elapsed() < timeout,
                        "timed out acquiring {purpose} lock {}",
                        path.display()
                    );
                    sleep(LOCK_RETRY_INTERVAL);
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to acquire {purpose} lock {}", path.display())
                    });
                }
            }
        }

        let acquired_unix_ms = unix_millis();
        let lease = timeout.saturating_mul(4).max(MINIMUM_LEASE);
        let owner = LockOwner {
            pid: std::process::id(),
            owner_id: uuid::Uuid::new_v4().to_string(),
            acquired_unix_ms,
            lease_expires_unix_ms: acquired_unix_ms.saturating_add(duration_millis(lease)),
        };
        if let Err(error) = write_lock_owner(&mut file, &owner) {
            let _ = unlock_file(&file);
            return Err(error).with_context(|| {
                format!("failed to record {purpose} lock owner {}", path.display())
            });
        }
        Ok(Self { file })
    }
}

impl Drop for FileLeaseLock {
    fn drop(&mut self) {
        let _ = unlock_file(&self.file);
    }
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8], purpose: &str) -> Result<()> {
    atomic_write_with_before_rename(path, bytes, purpose, || Ok(()))
}

fn atomic_write_with_before_rename<F>(
    path: &Path,
    bytes: &[u8],
    purpose: &str,
    before_rename: F,
) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    create_parent_dir(path, purpose)?;
    let parent = parent_dir(path);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("file-state path must have a UTF-8 file name")?;
    let temp_path = parent.join(format!(
        ".{file_name}.tmp.{}.{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let mut cleanup = TempFileCleanup::new(temp_path.clone());
    let mut temp = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| {
            format!(
                "failed to create temporary {purpose} {}",
                temp_path.display()
            )
        })?;
    temp.write_all(bytes).with_context(|| {
        format!(
            "failed to write temporary {purpose} {}",
            temp_path.display()
        )
    })?;
    temp.sync_all()
        .with_context(|| format!("failed to sync temporary {purpose} {}", temp_path.display()))?;
    before_rename()?;
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to atomically replace {purpose} {} from {}",
            path.display(),
            temp_path.display()
        )
    })?;
    cleanup.disarm();
    sync_parent_dir(parent, purpose)?;
    Ok(())
}

fn write_lock_owner(file: &mut File, owner: &LockOwner) -> Result<()> {
    file.set_len(0)
        .context("failed to truncate lock owner file")?;
    file.seek(SeekFrom::Start(0))
        .context("failed to seek lock owner file")?;
    serde_json::to_writer(&mut *file, owner).context("failed to serialize lock owner")?;
    file.write_all(b"\n")
        .context("failed to terminate lock owner record")?;
    file.sync_all().context("failed to sync lock owner file")
}

fn create_parent_dir(path: &Path, purpose: &str) -> Result<()> {
    let parent = parent_dir(path);
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {purpose} dir {}", parent.display()))
}

fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(not(windows))]
fn sync_parent_dir(parent: &Path, purpose: &str) -> Result<()> {
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("failed to sync {purpose} dir {}", parent.display()))
}

#[cfg(windows)]
fn sync_parent_dir(_parent: &Path, _purpose: &str) -> Result<()> {
    // `std::fs::File::open` cannot open directories on Windows. The temporary
    // file is already flushed before the atomic rename, and std has no portable
    // Windows API for flushing the parent directory's metadata.
    Ok(())
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(duration_millis)
        .unwrap_or_default()
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn try_lock_exclusive(file: &File) -> std::io::Result<bool> {
    match FileExt::try_lock(file) {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

fn unlock_file(file: &File) -> std::io::Result<()> {
    FileExt::unlock(file)
}

struct TempFileCleanup {
    path: PathBuf,
    armed: bool,
}

impl TempFileCleanup {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FileLeaseLock, atomic_write, atomic_write_with_before_rename};
    use anyhow::anyhow;
    use std::{
        fs,
        process::{Command, Stdio},
        thread::sleep,
        time::{Duration, Instant},
    };

    #[test]
    fn atomic_write_keeps_previous_state_when_interrupted_before_rename() {
        let path = temp_path("partial-write");
        fs::write(&path, b"previous-valid-state").unwrap();

        let error = atomic_write_with_before_rename(&path, b"partial-new-state", "test", || {
            Err(anyhow!("injected interruption"))
        })
        .unwrap_err();

        assert!(error.to_string().contains("injected interruption"));
        assert_eq!(fs::read(&path).unwrap(), b"previous-valid-state");
        let prefix = format!(".{}.tmp.", path.file_name().unwrap().to_string_lossy());
        assert!(
            fs::read_dir(path.parent().unwrap())
                .unwrap()
                .flatten()
                .all(|entry| !entry.file_name().to_string_lossy().starts_with(&prefix))
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn atomic_write_replaces_existing_state() {
        let path = temp_path("replace-existing");
        fs::write(&path, b"previous-state").unwrap();

        atomic_write(&path, b"replacement-state", "test").unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"replacement-state");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn killed_lock_owner_is_recovered_without_deleting_the_lock_file() {
        if std::env::var_os("AUTOHAND_FILE_LOCK_CHILD").is_some() {
            return;
        }
        let path = temp_path("killed-lock");
        let ready_path = path.with_extension("ready");
        let current_exe = std::env::current_exe().unwrap();
        let mut child = Command::new(current_exe)
            .args([
                "--exact",
                "file_state::tests::hold_file_lock_until_killed",
                "--nocapture",
            ])
            .env("AUTOHAND_FILE_LOCK_CHILD", &path)
            .env("AUTOHAND_FILE_LOCK_READY", &ready_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready_path.exists() && Instant::now() < deadline {
            sleep(Duration::from_millis(10));
        }
        assert!(ready_path.exists(), "child did not acquire the test lock");
        child.kill().unwrap();
        child.wait().unwrap();

        let recovered = FileLeaseLock::acquire(&path, Duration::from_secs(1), "test").unwrap();
        assert!(path.exists(), "advisory lock metadata remains reusable");
        drop(recovered);
        let _ = fs::remove_file(ready_path);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn hold_file_lock_until_killed() {
        let Some(path) = std::env::var_os("AUTOHAND_FILE_LOCK_CHILD") else {
            return;
        };
        let ready_path = std::env::var_os("AUTOHAND_FILE_LOCK_READY").unwrap();
        let _lock =
            FileLeaseLock::acquire(std::path::Path::new(&path), Duration::from_secs(1), "test")
                .unwrap();
        atomic_write(
            std::path::Path::new(&ready_path),
            b"ready",
            "test readiness",
        )
        .unwrap();
        loop {
            sleep(Duration::from_secs(1));
        }
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("autohand-router-{label}-{}", uuid::Uuid::new_v4()))
    }
}
