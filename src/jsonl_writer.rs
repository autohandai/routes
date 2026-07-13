use serde::Serialize;
use serde_json::Value;
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};

enum WriterMessage {
    Record(Value),
    Flush(oneshot::Sender<()>),
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct JsonlWriterStats {
    pub accepted: u64,
    pub written: u64,
    pub dropped: u64,
    pub errors: u64,
    pub rotations: u64,
}

#[derive(Debug, Default)]
struct WriterCounters {
    accepted: AtomicU64,
    written: AtomicU64,
    dropped: AtomicU64,
    errors: AtomicU64,
    rotations: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct AsyncJsonlWriter {
    sender: Option<mpsc::Sender<WriterMessage>>,
    counters: Arc<WriterCounters>,
}

impl AsyncJsonlWriter {
    pub fn new(
        path: Option<PathBuf>,
        queue_capacity: usize,
        max_file_bytes: u64,
        retained_files: usize,
    ) -> Self {
        let counters = Arc::new(WriterCounters::default());
        let Some(path) = path else {
            return Self {
                sender: None,
                counters,
            };
        };
        let (sender, mut receiver) = mpsc::channel(queue_capacity);
        let worker_counters = counters.clone();
        tokio::spawn(async move {
            while let Some(message) = receiver.recv().await {
                match message {
                    WriterMessage::Record(value) => {
                        let path = path.clone();
                        let log_path = path.clone();
                        let result = tokio::task::spawn_blocking(move || {
                            append_rotating_jsonl(&path, &value, max_file_bytes, retained_files)
                        })
                        .await;
                        match result {
                            Ok(Ok(rotated)) => {
                                worker_counters.written.fetch_add(1, Ordering::Relaxed);
                                if rotated {
                                    worker_counters.rotations.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            Ok(Err(error)) => {
                                worker_counters.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(?error, path = %log_path.display(), "JSONL writer failed");
                            }
                            Err(error) => {
                                worker_counters.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::warn!(?error, path = %log_path.display(), "JSONL writer task failed");
                            }
                        }
                    }
                    WriterMessage::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        Self {
            sender: Some(sender),
            counters,
        }
    }

    pub fn try_write(&self, value: Value) -> bool {
        let Some(sender) = &self.sender else {
            return false;
        };
        match sender.try_send(WriterMessage::Record(value)) {
            Ok(()) => {
                self.counters.accepted.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn enabled(&self) -> bool {
        self.sender.is_some()
    }

    pub async fn flush(&self, timeout_duration: Duration) -> JsonlWriterStats {
        if let Some(sender) = &self.sender {
            let (done, receiver) = oneshot::channel();
            let _ = tokio::time::timeout(timeout_duration, async {
                if sender.send(WriterMessage::Flush(done)).await.is_ok() {
                    let _ = receiver.await;
                }
            })
            .await;
        }
        self.stats()
    }

    pub fn stats(&self) -> JsonlWriterStats {
        JsonlWriterStats {
            accepted: self.counters.accepted.load(Ordering::Relaxed),
            written: self.counters.written.load(Ordering::Relaxed),
            dropped: self.counters.dropped.load(Ordering::Relaxed),
            errors: self.counters.errors.load(Ordering::Relaxed),
            rotations: self.counters.rotations.load(Ordering::Relaxed),
        }
    }
}

fn append_rotating_jsonl(
    path: &Path,
    value: &Value,
    max_file_bytes: u64,
    retained_files: usize,
) -> std::io::Result<bool> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    let current_len = std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let rotate = current_len > 0 && current_len.saturating_add(line.len() as u64) > max_file_bytes;
    if rotate {
        rotate_files(path, retained_files)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(&line)?;
    file.flush()?;
    Ok(rotate)
}

fn rotate_files(path: &Path, retained_files: usize) -> std::io::Result<()> {
    if retained_files == 0 {
        let _ = std::fs::remove_file(path);
        return Ok(());
    }
    let oldest = rotated_path(path, retained_files);
    let _ = std::fs::remove_file(oldest);
    for index in (1..retained_files).rev() {
        let source = rotated_path(path, index);
        let target = rotated_path(path, index + 1);
        if source.exists() {
            std::fs::rename(source, target)?;
        }
    }
    if path.exists() {
        std::fs::rename(path, rotated_path(path, 1))?;
    }
    Ok(())
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    PathBuf::from(format!("{}.{}", path.display(), index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queue_is_bounded_rotates_and_flushes_in_fifo_order() {
        let directory = std::env::temp_dir().join(format!(
            "autohand-router-jsonl-writer-{}",
            uuid::Uuid::new_v4()
        ));
        let path = directory.join("events.jsonl");
        let writer = AsyncJsonlWriter::new(Some(path.clone()), 2, 10, 2);
        for index in 0..20 {
            writer.try_write(serde_json::json!({"index": index}));
        }
        let stats = writer.flush(Duration::from_secs(2)).await;
        assert_eq!(stats.accepted, stats.written);
        assert!(stats.dropped > 0);
        assert!(stats.rotations > 0);
        assert!(path.exists());
        assert!(rotated_path(&path, 1).exists());
        assert!(!rotated_path(&path, 3).exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn disk_errors_are_observable_without_blocking_callers() {
        let directory = std::env::temp_dir().join(format!(
            "autohand-router-jsonl-error-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&directory, b"not a directory").unwrap();
        let writer = AsyncJsonlWriter::new(Some(directory.join("events.jsonl")), 4, 1024, 1);
        assert!(writer.try_write(serde_json::json!({"event": true})));
        let stats = writer.flush(Duration::from_secs(2)).await;
        assert_eq!(stats.errors, 1);
        let _ = std::fs::remove_file(directory);
    }
}
