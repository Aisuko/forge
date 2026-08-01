//! `forge-top` — a terminal model browser and live run dashboard.
//!
//! ```bash
//! cargo run --release -p forge-top -- --path models/ --path checkpoints/
//! ```
//!
//! Three threads over one `mpsc` channel: **main** owns the terminal and
//! redraws at ~15 FPS, **scanner** streams discovered models, **runner** drives
//! generation. Inference never touches the main thread — a single CPU
//! `logits_step` is long enough to visibly freeze input handling.

mod metrics;
mod run;
mod scan;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use crossterm::event::{self, KeyCode, KeyEventKind, KeyModifiers};
use forge::Sampling;

use metrics::{Sampler, Throughput};
use run::{Backend, RunSpec};
use scan::ModelInfo;

/// ~15 FPS. Also the input-poll timeout, so a keypress is never delayed by
/// more than one frame.
const FRAME: Duration = Duration::from_millis(66);
/// Tokens of context kept in the output pane.
const OUTPUT_CHARS: usize = 8192;

pub enum Event {
    ModelFound(Box<ModelInfo>),
    ScanWarning(String),
    ScanDone,
    RunStarted {
        device: String,
        load: Duration,
        at: Instant,
    },
    Token {
        id: u32,
        text: String,
        at: Instant,
    },
    /// `Some(error)` when the run failed or was rejected.
    RunFinished(Option<String>),
}

#[derive(Clone, Copy, PartialEq)]
pub enum Detail {
    Summary,
    Tensors,
}

pub enum Phase {
    Idle,
    Loading,
    Generating,
    Cancelling,
    Failed(String),
}

pub struct AppState {
    pub roots: Vec<PathBuf>,
    pub models: Vec<ModelInfo>,
    pub selected: usize,
    pub detail: Detail,
    pub scanning: bool,
    pub phase: Phase,
    pub backend: Backend,
    pub run_device: Option<String>,
    pub output: String,
    pub message: Option<String>,
    pub throughput: Throughput,
    pub metrics: Sampler,
    cancel: Option<Arc<AtomicBool>>,
    prompt: Option<String>,
    max_new_tokens: usize,
    sampling: Sampling,
}

impl AppState {
    pub fn model(&self) -> Option<&ModelInfo> {
        self.models.get(self.selected)
    }

    pub fn roots_display(&self) -> String {
        self.roots
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn busy(&self) -> bool {
        matches!(
            self.phase,
            Phase::Loading | Phase::Generating | Phase::Cancelling
        )
    }

    /// The prompt for the selected model: `--prompt` when given, otherwise
    /// something the model's own vocabulary can actually encode.
    fn prompt_for(&self, m: &ModelInfo) -> String {
        if let Some(p) = &self.prompt {
            return p.clone();
        }
        match m.tokenizer {
            scan::TokenizerKind::Char => "ROMEO:".into(),
            _ => "The old lighthouse keeper".into(),
        }
    }

    fn start_run(&mut self, tx: &Sender<Event>) {
        if self.busy() {
            return;
        }
        let Some(m) = self.model().cloned() else {
            self.message = Some("no model selected".into());
            return;
        };
        if let Some(reason) = &m.blocked {
            self.phase = Phase::Failed(reason.clone());
            return;
        }
        self.output.clear();
        self.run_device = None;
        self.throughput.start(Instant::now());
        self.phase = Phase::Loading;
        self.cancel = Some(run::spawn(
            RunSpec {
                prompt: self.prompt_for(&m),
                model: m,
                backend: self.backend,
                max_new_tokens: self.max_new_tokens,
                sampling: self.sampling,
            },
            tx.clone(),
        ));
    }

    fn cancel_run(&mut self) {
        if let Some(flag) = &self.cancel {
            flag.store(true, Ordering::Relaxed);
            self.phase = Phase::Cancelling;
        }
    }

    fn apply(&mut self, ev: Event) {
        match ev {
            Event::ModelFound(m) => self.models.push(*m),
            Event::ScanWarning(w) => self.message = Some(w),
            Event::ScanDone => {
                self.scanning = false;
                if self.models.is_empty() {
                    self.message = Some(format!("no models under {}", self.roots_display()));
                }
            }
            Event::RunStarted { device, load, at } => {
                self.phase = Phase::Generating;
                self.run_device = Some(format!("{device} (loaded in {:.1}s)", load.as_secs_f32()));
                // Restart the clock: weight loading is not prefill, and
                // folding it into TTFT would misreport both.
                self.throughput.start(at);
            }
            Event::Token { id: _, text, at } => {
                self.throughput.record(at);
                self.output.push_str(&text);
                if self.output.len() > OUTPUT_CHARS {
                    let cut = self.output.len() - OUTPUT_CHARS;
                    // Never split a UTF-8 character.
                    let cut = (cut..self.output.len())
                        .find(|&i| self.output.is_char_boundary(i))
                        .unwrap_or(self.output.len());
                    self.output.drain(..cut);
                }
            }
            Event::RunFinished(err) => {
                self.cancel = None;
                self.phase = match err {
                    Some(e) => Phase::Failed(e),
                    None => Phase::Idle,
                };
                self.message = self.throughput.average().map(|avg| {
                    format!(
                        "{} tokens, {avg:.1} tok/s average — backend: {}",
                        self.throughput.tokens,
                        self.backend.label()
                    )
                });
            }
        }
    }
}

struct Args {
    roots: Vec<PathBuf>,
    backend: Backend,
    prompt: Option<String>,
    tokens: usize,
    topk: Option<usize>,
    temp: f32,
    /// Print the scan and exit, without touching the terminal. Makes the
    /// discovery path scriptable — and its memory footprint measurable.
    list: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut roots = Vec::new();
    let mut a = Args {
        roots: Vec::new(),
        backend: Backend::Wgpu,
        prompt: None,
        tokens: 120,
        topk: Some(40),
        temp: 0.8,
        list: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("missing value for {name}"));
        match flag.as_str() {
            // Repeatable: this repo keeps weights in both models/ and
            // checkpoints/, and other users will have their own layout.
            "--path" => roots.push(PathBuf::from(val("--path")?)),
            "--backend" => {
                a.backend = match val("--backend")?.as_str() {
                    "cpu" => Backend::Cpu,
                    "wgpu" => Backend::Wgpu,
                    other => return Err(format!("unknown backend {other:?} (use cpu|wgpu)")),
                }
            }
            "--prompt" => a.prompt = Some(val("--prompt")?),
            "--tokens" => a.tokens = val("--tokens")?.parse().map_err(|_| "bad --tokens")?,
            "--topk" => {
                let k: usize = val("--topk")?.parse().map_err(|_| "bad --topk")?;
                a.topk = (k > 0).then_some(k);
            }
            "--temp" => a.temp = val("--temp")?.parse().map_err(|_| "bad --temp")?,
            "--list" => a.list = true,
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }
    // Default to the current directory, not models/: weights live in several
    // places here, and defaulting to one of them makes the browser look broken.
    a.roots = if roots.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        roots
    };
    Ok(a)
}

fn usage() -> String {
    "forge-top — model browser + live run dashboard\n\n\
     usage: forge-top [--path DIR]... [--backend cpu|wgpu] [--prompt TEXT]\n\
     \x20                [--tokens N] [--topk K] [--temp T] [--list]\n\n\
     --path may be repeated; it defaults to the current directory.\n\
     --list prints discovered models and exits."
        .to_string()
}

/// `--list`: the scan, printed. No terminal, no threads.
fn list(roots: Vec<PathBuf>) {
    let start = Instant::now();
    let (tx, rx) = std::sync::mpsc::channel();
    scan::scan(roots, tx);
    let mut n = 0;
    for ev in rx {
        match ev {
            Event::ModelFound(m) => {
                n += 1;
                let cfg = m.config.as_ref().map_or("no config".to_string(), |c| {
                    format!("{}L/{}H/{}d ctx {}", c.n_layer, c.n_head, c.n_embd, c.n_ctx)
                });
                println!(
                    "{:<40} {:>9}  {:>7} params  {:>3} tensors  {cfg}  tok:{}{}",
                    m.path.display(),
                    scan::human_bytes(m.file_size),
                    scan::human_params(m.params),
                    m.tensor_count(),
                    m.tokenizer.label(),
                    m.blocked
                        .as_ref()
                        .map_or(String::new(), |b| format!("  [not runnable: {b}]")),
                );
            }
            Event::ScanWarning(w) => eprintln!("warning: {w}"),
            _ => {}
        }
    }
    println!("{n} model(s) in {:.3}s", start.elapsed().as_secs_f32());
}

fn main() -> std::io::Result<()> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if args.list {
        list(args.roots);
        return Ok(());
    }

    let (tx, rx) = std::sync::mpsc::channel();
    {
        let (roots, tx) = (args.roots.clone(), tx.clone());
        std::thread::spawn(move || scan::scan(roots, tx));
    }

    let app = AppState {
        roots: args.roots,
        models: Vec::new(),
        selected: 0,
        detail: Detail::Summary,
        scanning: true,
        phase: Phase::Idle,
        backend: args.backend,
        run_device: None,
        output: String::new(),
        message: None,
        throughput: Throughput::new(8),
        metrics: Sampler::new(),
        cancel: None,
        prompt: args.prompt,
        max_new_tokens: args.tokens,
        sampling: match args.topk {
            Some(k) => Sampling::TopK {
                k,
                temperature: args.temp,
                seed: 42,
            },
            None => Sampling::Greedy,
        },
    };

    // `run` handles init, restore, *and* the panic hook — a panic inside raw
    // mode would otherwise leave the user's terminal unusable.
    ratatui::run(|terminal| event_loop(terminal, app, tx, rx))
}

fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    mut app: AppState,
    tx: Sender<Event>,
    rx: Receiver<Event>,
) -> std::io::Result<()> {
    loop {
        // Metrics sampling is cheap and self-throttling, so it can live here.
        app.metrics.tick();
        while let Ok(ev) = rx.try_recv() {
            app.apply(ev);
        }
        terminal.draw(|frame| ui::draw(frame, &mut app))?;

        if !event::poll(FRAME)? {
            continue;
        }
        let event::Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => break,
            KeyCode::Char('q') => break,
            KeyCode::Up | KeyCode::Char('k') => {
                app.selected = app.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if app.selected + 1 < app.models.len() {
                    app.selected += 1;
                }
            }
            KeyCode::Tab => {
                app.detail = match app.detail {
                    Detail::Summary => Detail::Tensors,
                    Detail::Tensors => Detail::Summary,
                };
            }
            KeyCode::Char('b') if !app.busy() => {
                app.backend = app.backend.toggled();
                app.message = Some(format!("backend: {}", app.backend.label()));
            }
            KeyCode::Enter => app.start_run(&tx),
            KeyCode::Esc => app.cancel_run(),
            _ => {}
        }
    }
    // Stop the runner before the terminal is restored, so a late token cannot
    // print over the user's shell.
    if let Some(flag) = &app.cancel {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}
