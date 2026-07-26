//! GPU (NVML) and host (sysinfo) sampling. Every field is optional: the TUI
//! must stay fully usable on a machine with no NVIDIA GPU at all.

use std::ffi::OsStr;
use std::time::{Duration, Instant};

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use sysinfo::System;

/// Sampling period. NVML calls are sub-millisecond so this could run per
/// frame, but 4 Hz is plenty for a human and leaves the GPU alone.
const PERIOD: Duration = Duration::from_millis(250);

#[derive(Default, Clone, Copy)]
pub struct Gpu {
    pub vram_used: u64,
    pub vram_total: u64,
    pub util: u32,
    pub temp_c: u32,
    pub power_w: f32,
}

#[derive(Default, Clone, Copy)]
pub struct Host {
    pub ram_used: u64,
    pub ram_total: u64,
    pub cpu: f32,
}

pub struct Sampler {
    nvml: Option<Nvml>,
    /// Why NVML is unavailable, for the "n/a" pane.
    pub gpu_error: Option<String>,
    system: System,
    last: Option<Instant>,
    /// `refresh_cpu_usage` needs two samples to compute a delta; the first
    /// reading is documented to be meaningless, so it is discarded.
    cpu_primed: bool,
    pub gpu: Option<Gpu>,
    pub host: Host,
}

impl Sampler {
    pub fn new() -> Self {
        // This container ships only `libnvidia-ml.so.1`; there is no
        // unversioned `libnvidia-ml.so`, which is what `Nvml::init()` looks
        // for by default.
        let (nvml, gpu_error) = match Nvml::builder()
            .lib_path(OsStr::new("libnvidia-ml.so.1"))
            .init()
        {
            Ok(n) => (Some(n), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Sampler {
            nvml,
            gpu_error,
            system: System::new_all(),
            last: None,
            cpu_primed: false,
            gpu: None,
            host: Host::default(),
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.nvml.is_some()
    }

    /// Refresh if `PERIOD` has elapsed. Cheap to call every frame.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let due = match self.last {
            // Also respect sysinfo's documented CPU floor: refreshing faster
            // than MINIMUM_CPU_UPDATE_INTERVAL yields garbage percentages.
            Some(t) => now.duration_since(t) >= PERIOD.max(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL),
            None => true,
        };
        if !due {
            return;
        }
        self.last = Some(now);

        self.system.refresh_memory();
        self.system.refresh_cpu_usage();
        self.host.ram_used = self.system.used_memory(); // bytes, not KB
        self.host.ram_total = self.system.total_memory();
        if self.cpu_primed {
            self.host.cpu = self.system.global_cpu_usage();
        } else {
            self.cpu_primed = true;
        }

        self.gpu = self.nvml.as_ref().and_then(|nvml| {
            let dev = nvml.device_by_index(0).ok()?;
            let mem = dev.memory_info().ok()?;
            Some(Gpu {
                // Device-wide, not per-process: `nvidia-smi
                // --query-compute-apps` returns no rows inside a PID
                // namespace, so per-process attribution is not available here.
                vram_used: mem.used,
                vram_total: mem.total,
                util: dev.utilization_rates().map(|u| u.gpu).unwrap_or(0),
                temp_c: dev.temperature(TemperatureSensor::Gpu).unwrap_or(0),
                power_w: dev
                    .power_usage()
                    .map(|mw| mw as f32 / 1000.0)
                    .unwrap_or(0.0),
            })
        });
    }
}

/// Rolling tokens/s over the stream of per-token timestamps.
///
/// Prefill is tracked separately: the first `logits_step` covers the whole
/// prompt and is far slower than a decode step, so folding it into tokens/s
/// makes the number meaningless.
pub struct Throughput {
    /// EMA window, in tokens.
    window: usize,
    inter: std::collections::VecDeque<Duration>,
    prev: Option<Instant>,
    pub started: Option<Instant>,
    pub ttft: Option<Duration>,
    pub tokens: usize,
}

impl Throughput {
    pub fn new(window: usize) -> Self {
        Throughput {
            window,
            inter: std::collections::VecDeque::with_capacity(window),
            prev: None,
            started: None,
            ttft: None,
            tokens: 0,
        }
    }

    pub fn start(&mut self, at: Instant) {
        *self = Throughput::new(self.window);
        self.started = Some(at);
    }

    pub fn record(&mut self, at: Instant) {
        self.tokens += 1;
        match self.prev {
            None => {
                // First token: the gap since `start` is prefill (TTFT), not a
                // decode interval, so it never enters the tok/s window.
                self.ttft = self.started.map(|s| at.duration_since(s));
            }
            Some(p) => {
                if self.inter.len() == self.window {
                    self.inter.pop_front();
                }
                self.inter.push_back(at.duration_since(p));
            }
        }
        self.prev = Some(at);
    }

    /// Instantaneous rate: EMA over the last `window` decode intervals.
    pub fn instant(&self) -> Option<f32> {
        if self.inter.is_empty() {
            return None;
        }
        let total: f64 = self.inter.iter().map(|d| d.as_secs_f64()).sum();
        (total > 0.0).then(|| (self.inter.len() as f64 / total) as f32)
    }

    /// Session average over decode only — excludes prefill, so it is
    /// comparable with the instantaneous figure.
    pub fn average(&self) -> Option<f32> {
        let (start, ttft, last) = (self.started?, self.ttft?, self.prev?);
        let decode = last.duration_since(start).checked_sub(ttft)?.as_secs_f64();
        (self.tokens > 1 && decode > 0.0).then(|| ((self.tokens - 1) as f64 / decode) as f32)
    }

    /// Recent inter-token rates, oldest first — the sparkline series.
    pub fn history(&self) -> Vec<u64> {
        self.inter
            .iter()
            .map(|d| {
                let s = d.as_secs_f64();
                if s > 0.0 { (1.0 / s) as u64 } else { 0 }
            })
            .collect()
    }
}
