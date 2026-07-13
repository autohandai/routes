use serde::Serialize;
use std::{
    collections::HashMap,
    fmt::Write,
    sync::{Mutex, OnceLock},
    time::Instant,
};

const BUCKETS_MS: [u64; 14] = [
    1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000,
];

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct HistogramKey {
    metric: &'static str,
    endpoint: &'static str,
    provider: String,
    model: String,
    outcome: &'static str,
}

#[derive(Debug, Clone, Default)]
struct Histogram {
    buckets: [u64; BUCKETS_MS.len()],
    count: u64,
    sum_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistogramSnapshot {
    pub metric: &'static str,
    pub endpoint: &'static str,
    pub provider: String,
    pub model: String,
    pub outcome: &'static str,
    pub count: u64,
    pub sum_ms: u64,
}

static HISTOGRAMS: OnceLock<Mutex<HashMap<HistogramKey, Histogram>>> = OnceLock::new();

pub fn observe(
    metric: &'static str,
    endpoint: &'static str,
    provider: impl Into<String>,
    model: impl Into<String>,
    outcome: &'static str,
    elapsed_ms: u64,
) {
    let key = HistogramKey {
        metric,
        endpoint,
        provider: provider.into(),
        model: model.into(),
        outcome,
    };
    let Ok(mut histograms) = HISTOGRAMS.get_or_init(Default::default).lock() else {
        return;
    };
    let histogram = histograms.entry(key).or_default();
    histogram.count = histogram.count.saturating_add(1);
    histogram.sum_ms = histogram.sum_ms.saturating_add(elapsed_ms);
    for (index, boundary) in BUCKETS_MS.iter().enumerate() {
        if elapsed_ms <= *boundary {
            histogram.buckets[index] = histogram.buckets[index].saturating_add(1);
        }
    }
}

pub fn timer(
    metric: &'static str,
    endpoint: &'static str,
    provider: impl Into<String>,
    model: impl Into<String>,
    outcome: &'static str,
) -> HistogramTimer {
    HistogramTimer {
        metric,
        endpoint,
        provider: provider.into(),
        model: model.into(),
        outcome,
        started: Instant::now(),
    }
}

pub struct HistogramTimer {
    metric: &'static str,
    endpoint: &'static str,
    provider: String,
    model: String,
    outcome: &'static str,
    started: Instant,
}

impl Drop for HistogramTimer {
    fn drop(&mut self) {
        observe(
            self.metric,
            self.endpoint,
            self.provider.clone(),
            self.model.clone(),
            self.outcome,
            elapsed_ms(self.started),
        );
    }
}

pub fn snapshots() -> Vec<HistogramSnapshot> {
    let Ok(histograms) = HISTOGRAMS.get_or_init(Default::default).lock() else {
        return Vec::new();
    };
    let mut snapshots = histograms
        .iter()
        .map(|(key, histogram)| HistogramSnapshot {
            metric: key.metric,
            endpoint: key.endpoint,
            provider: key.provider.clone(),
            model: key.model.clone(),
            outcome: key.outcome,
            count: histogram.count,
            sum_ms: histogram.sum_ms,
        })
        .collect::<Vec<_>>();
    snapshots.sort_by(|left, right| {
        left.metric
            .cmp(right.metric)
            .then_with(|| left.endpoint.cmp(right.endpoint))
            .then_with(|| left.provider.cmp(&right.provider))
            .then_with(|| left.model.cmp(&right.model))
            .then_with(|| left.outcome.cmp(right.outcome))
    });
    snapshots
}

pub fn prometheus() -> String {
    let Ok(histograms) = HISTOGRAMS.get_or_init(Default::default).lock() else {
        return String::new();
    };
    let mut output = String::new();
    let mut entries = histograms.iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| {
        left.metric
            .cmp(right.metric)
            .then_with(|| left.endpoint.cmp(right.endpoint))
            .then_with(|| left.provider.cmp(&right.provider))
            .then_with(|| left.model.cmp(&right.model))
            .then_with(|| left.outcome.cmp(right.outcome))
    });
    for (key, histogram) in entries {
        for (index, boundary) in BUCKETS_MS.iter().enumerate() {
            write_sample(
                &mut output,
                &format!("{}_bucket", key.metric),
                key,
                Some(&boundary.to_string()),
                histogram.buckets[index],
            );
        }
        write_sample(
            &mut output,
            &format!("{}_bucket", key.metric),
            key,
            Some("+Inf"),
            histogram.count,
        );
        write_sample(
            &mut output,
            &format!("{}_sum", key.metric),
            key,
            None,
            histogram.sum_ms,
        );
        write_sample(
            &mut output,
            &format!("{}_count", key.metric),
            key,
            None,
            histogram.count,
        );
    }
    output
}

fn write_sample(
    output: &mut String,
    metric: &str,
    key: &HistogramKey,
    le: Option<&str>,
    value: u64,
) {
    let _ = write!(
        output,
        "{metric}{{endpoint=\"{}\",provider=\"{}\",model=\"{}\",outcome=\"{}\"",
        escape(key.endpoint),
        escape(&key.provider),
        escape(&key.model),
        escape(key.outcome)
    );
    if let Some(le) = le {
        let _ = write!(output, ",le=\"{le}\"");
    }
    let _ = writeln!(output, "}} {value}");
}

fn escape(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

pub fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_use_cumulative_buckets_and_fixed_labels() {
        observe(
            "autohand_router_test_duration_ms",
            "chat",
            "provider",
            "model",
            "ok",
            7,
        );
        let output = prometheus();
        assert!(output.contains("autohand_router_test_duration_ms_bucket{"));
        assert!(output.contains("le=\"10\"} 1"));
        assert!(output.contains("le=\"5\"} 0"));
        assert!(output.contains("autohand_router_test_duration_ms_count{"));
    }
}
