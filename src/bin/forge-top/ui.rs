//! Rendering. A pure function of `&AppState` — no I/O, no sampling, no
//! mutation, so a frame can never block on a device or a disk.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Gauge, List, ListItem, ListState, Paragraph, Sparkline, Wrap};

use crate::scan::{human_bytes, human_params};
use crate::{AppState, Detail, Phase};

/// Below this width the two columns are stacked instead of side by side.
const NARROW: u16 = 80;

const ACCENT: Color = Color::Rgb(255, 138, 76);

pub fn draw(frame: &mut Frame, app: &mut AppState) {
    let area = frame.area();
    // A zero-sized Rect reaches here when the terminal is being resized; every
    // widget below would panic on it.
    if area.width < 4 || area.height < 4 {
        return;
    }
    let [body, status] = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);

    let (left, right) = if body.width < NARROW {
        // Single column: the dashboard keeps the top half, the list the rest.
        let [a, b] = Layout::vertical([Constraint::Percentage(50); 2]).areas(body);
        (a, b)
    } else {
        let [a, b] = Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)])
            .areas(body);
        (a, b)
    };

    draw_models(frame, app, left);
    draw_dashboard(frame, app, right);
    frame.render_widget(status_line(app), status);
}

fn draw_models(frame: &mut Frame, app: &mut AppState, area: Rect) {
    let [list_area, detail_area] =
        Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area);

    let block = Block::bordered().title(Line::from(vec![
        " models ".fg(ACCENT).bold(),
        Span::raw(format!("({}) ", app.models.len())),
    ]));

    if app.models.is_empty() {
        let msg = if app.scanning {
            format!("scanning {}…", app.roots_display())
        } else {
            format!(
                "no *.safetensors found under {}\n\npass --path <dir> to search elsewhere",
                app.roots_display()
            )
        };
        frame.render_widget(
            Paragraph::new(msg).wrap(Wrap { trim: false }).block(block),
            list_area,
        );
        frame.render_widget(Block::bordered().title(" details "), detail_area);
        return;
    }

    let items: Vec<ListItem> = app
        .models
        .iter()
        .map(|m| {
            let size = human_bytes(m.file_size);
            let mark = if m.runnable() { "" } else { " ✗" };
            ListItem::new(Line::from(vec![
                Span::raw(format!("{:<22}", truncate(&m.name, 22))),
                Span::raw(format!("{size:>8}")),
                Span::styled(mark, Style::default().fg(Color::Red)),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_symbol("> ")
            .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        list_area,
        &mut state,
    );

    draw_detail(frame, app, detail_area);
}

fn draw_detail(frame: &mut Frame, app: &AppState, area: Rect) {
    let Some(m) = app.model() else { return };
    let title = match app.detail {
        Detail::Summary => " details  [tab] tensors ",
        Detail::Tensors => " tensors  [tab] details ",
    };
    let block = Block::bordered().title(title.fg(ACCENT));

    let text: Vec<Line> = match app.detail {
        Detail::Tensors => m
            .tensors
            .iter()
            .map(|t| {
                Line::from(format!(
                    "{:<28} {:?} {}",
                    truncate(&t.name, 28),
                    t.shape,
                    t.dtype
                ))
            })
            .collect(),
        Detail::Summary => {
            let c = m.config.as_ref();
            let field = |label: &str, v: String| {
                Line::from(vec![
                    Span::styled(format!("{label:<9}"), Style::default().fg(Color::DarkGray)),
                    Span::raw(v),
                ])
            };
            let dash = "—".to_string();
            let mut lines = vec![
                field("layers", c.map_or(dash.clone(), |c| c.n_layer.to_string())),
                field("heads", c.map_or(dash.clone(), |c| c.n_head.to_string())),
                field("d_model", c.map_or(dash.clone(), |c| c.n_embd.to_string())),
                field("n_ctx", c.map_or(dash.clone(), |c| c.n_ctx.to_string())),
                field("vocab", c.map_or(dash, |c| c.vocab_size.to_string())),
                field("params", human_params(m.params)),
                field("tensors", m.tensor_count().to_string()),
                field("on disk", human_bytes(m.file_size)),
                field("tokenizer", m.tokenizer.label().to_string()),
                Line::from(""),
                Line::from(vec![
                    check("config.json", m.config_path.is_some()),
                    Span::raw("  "),
                    check("vocab.json", m.vocab_path.is_some()),
                    Span::raw("  "),
                    check("merges.txt", m.merges_path.is_some()),
                ]),
            ];
            if let Some(reason) = &m.blocked {
                lines.push(Line::from(""));
                lines.push(Line::from(format!("not runnable: {reason}").fg(Color::Red)));
            }
            lines
        }
    };
    frame.render_widget(
        Paragraph::new(text).block(block).wrap(Wrap { trim: false }),
        area,
    );
}

fn check(label: &str, ok: bool) -> Span<'static> {
    let (mark, color) = if ok {
        ("✓", Color::Green)
    } else {
        ("✗", Color::Red)
    };
    Span::styled(format!("{mark} {label}"), Style::default().fg(color))
}

fn draw_dashboard(frame: &mut Frame, app: &AppState, area: Rect) {
    let [rate_area, gauges_area, out_area] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(9),
        Constraint::Min(3),
    ])
    .areas(area);

    draw_rate(frame, app, rate_area);
    draw_gauges(frame, app, gauges_area);

    frame.render_widget(
        Paragraph::new(app.output.as_str())
            .wrap(Wrap { trim: false })
            .block(Block::bordered().title(" output ".fg(ACCENT))),
        out_area,
    );
}

fn draw_rate(frame: &mut Frame, app: &AppState, area: Rect) {
    let block = Block::bordered().title(" forge ".fg(ACCENT).bold());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let t = &app.throughput;
    let headline = Line::from(vec![
        Span::styled("tok/s ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            t.instant().map_or("  —  ".into(), |v| format!("{v:6.1}")),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   avg ", Style::default().fg(Color::DarkGray)),
        Span::raw(t.average().map_or("—".into(), |v| format!("{v:.1}"))),
        Span::styled("   ttft ", Style::default().fg(Color::DarkGray)),
        Span::raw(t.ttft.map_or("—".into(), |d| {
            format!("{:.0}ms", d.as_secs_f32() * 1000.0)
        })),
        Span::styled("   tok ", Style::default().fg(Color::DarkGray)),
        Span::raw(t.tokens.to_string()),
    ]);

    let [head, spark] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);
    frame.render_widget(Paragraph::new(headline), head);
    let data = t.history();
    if !data.is_empty() && spark.height > 0 {
        frame.render_widget(
            Sparkline::default()
                .data(&data)
                .style(Style::default().fg(ACCENT)),
            spark,
        );
    }
}

fn draw_gauges(frame: &mut Frame, app: &AppState, area: Rect) {
    let block = Block::bordered().title(" system ".fg(ACCENT));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 2 {
        return;
    }
    let rows: [Rect; 7] = Layout::vertical([Constraint::Length(1); 7]).areas(inner);
    let m = &app.metrics;

    match m.gpu {
        Some(g) => {
            frame.render_widget(
                label_line(
                    "device VRAM",
                    &format!("{:.1}/{:.1} GB", gb(g.vram_used), gb(g.vram_total)),
                ),
                rows[0],
            );
            frame.render_widget(bar(ratio(g.vram_used, g.vram_total)), rows[1]);
            frame.render_widget(label_line("GPU util", &format!("{}%", g.util)), rows[2]);
            frame.render_widget(bar(g.util as f64 / 100.0), rows[3]);
            frame.render_widget(
                label_line("GPU", &format!("{}C   {:.0}W", g.temp_c, g.power_w)),
                rows[4],
            );
        }
        None => {
            let why = app
                .metrics
                .gpu_error
                .as_deref()
                .unwrap_or("no NVIDIA device");
            frame.render_widget(label_line("device VRAM", "n/a"), rows[0]);
            frame.render_widget(
                Paragraph::new(truncate(why, inner.width as usize).fg(Color::DarkGray)),
                rows[1],
            );
            frame.render_widget(label_line("GPU util", "n/a"), rows[2]);
        }
    }

    frame.render_widget(
        label_line(
            "host RAM",
            &format!("{:.1}/{:.1} GB", gb(m.host.ram_used), gb(m.host.ram_total)),
        ),
        rows[5],
    );
    frame.render_widget(bar(ratio(m.host.ram_used, m.host.ram_total)), rows[6]);
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

fn ratio(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64).clamp(0.0, 1.0)
    }
}

fn label_line(label: &str, value: &str) -> Paragraph<'static> {
    Paragraph::new(Line::from(vec![
        Span::styled(format!("{label:<12}"), Style::default().fg(Color::DarkGray)),
        Span::raw(value.to_string()),
    ]))
}

fn bar(ratio: f64) -> Gauge<'static> {
    Gauge::default()
        .ratio(ratio)
        .label(format!("{:.0}%", ratio * 100.0))
        .gauge_style(Style::default().fg(ACCENT))
}

fn status_line(app: &AppState) -> Paragraph<'static> {
    let keys = "[↑↓] nav  [enter] run  [esc] cancel  [tab] detail  [b] backend  [q] quit";
    let (msg, color) = match &app.phase {
        Phase::Idle => (
            app.message
                .clone()
                .unwrap_or_else(|| format!("backend: {}", app.backend.label())),
            Color::DarkGray,
        ),
        Phase::Loading => ("loading weights…".to_string(), Color::Yellow),
        Phase::Generating => (
            match &app.run_device {
                Some(d) => format!("generating on {d}"),
                None => "generating…".to_string(),
            },
            Color::Green,
        ),
        Phase::Cancelling => (
            "cancelling — stops after the current token".to_string(),
            Color::Yellow,
        ),
        Phase::Failed(e) => (format!("error: {e}"), Color::Red),
    };
    Paragraph::new(Line::from(vec![
        Span::styled(format!(" {msg} "), Style::default().fg(color)),
        Span::styled(keys, Style::default().fg(Color::DarkGray)),
    ]))
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}
