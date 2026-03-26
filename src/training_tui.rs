use crate::training_hud::{TrainingHud, TrainingHudState};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use std::cell::RefCell;
use std::io::{self, stdout, Write};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use wgpu::{Backends, Instance};

// ── Palette ───────────────────────────────────────────────────────────────────
const BG: Color = Color::Rgb(40, 40, 40); // gruvbox dark0
const OUTER_BORD: Color = Color::Rgb(60, 56, 54); // provided neutral
const TEL_BORD: Color = Color::Rgb(69, 133, 136); // provided aqua
const GAUGE_BORD: Color = Color::Rgb(142, 192, 124); // provided green
const WORK_BORD: Color = Color::Rgb(215, 153, 33); // provided yellow
const ROCKET_BORD: Color = Color::Rgb(131, 165, 152); // gruvbox dark aqua soft
const GAUGE_FILL: Color = Color::Rgb(142, 192, 124); // provided green
const GAUGE_EMPTY: Color = Color::Rgb(60, 56, 54); // provided neutral

const LABEL: Color = Color::Rgb(215, 153, 33); // provided yellow
const VALUE: Color = Color::Rgb(142, 192, 124); // provided green
const LILAC: Color = Color::Rgb(184, 187, 38); // gruvbox green bright
const SKY: Color = Color::Rgb(69, 133, 136); // provided aqua
const GRAPE: Color = Color::Rgb(211, 134, 155); // gruvbox purple
const MUTED: Color = Color::Rgb(146, 131, 116); // gruvbox gray/brown
const TITLE: Color = Color::Rgb(235, 219, 178); // gruvbox light text

const C_WROTE: Color = Color::Rgb(142, 192, 124); // success
const C_FAIL: Color = Color::Rgb(204, 36, 29); // provided red
const C_SKIP: Color = Color::Rgb(146, 131, 116); // muted
const C_TOPR2: Color = Color::Rgb(69, 133, 136); // aqua
const C_BOTR2: Color = MUTED;
const PERF_BORD: Color = Color::Rgb(215, 153, 33); // yellow

const HARDWARE_POLL_INTERVAL: Duration = Duration::from_secs(3 * 60);

// ── Rocket ────────────────────────────────────────────────────────────────────
// Compact ASCII rocket — every line is exactly 14 display columns.
// Panel = 14 content + 2 border = 16 terminal columns.

const ROCKET_PANEL_W: u16 = 16;
const WINDOW_IDX: usize = 2;

const BODY: [(&str, Color); 8] = [
    ("      /\\      ", SKY),
    ("     /  \\     ", SKY),
    ("    / ** \\    ", SKY), // window — rendered with colored ◆◆ spans
    ("    |    |    ", TITLE),
    ("    |    |    ", TITLE),
    ("   /| || |\\   ", LILAC),
    ("  / |____| \\  ", LILAC),
    ("      ||      ", GRAPE),
];

const FIRE: [[&str; 2]; 4] = [
    ["     \\||/     ", "      \\/      "],
    ["     |**|     ", "      **      "],
    ["     /||\\     ", "      /\\      "],
    ["     *||*     ", "      **      "],
];

const FIRE_RGB: [[(u8, u8, u8); 2]; 4] = [
    [(255, 220, 100), (255, 150, 50)],
    [(255, 190, 70), (255, 120, 30)],
    [(255, 240, 150), (255, 180, 60)],
    [(255, 200, 90), (255, 100, 20)],
];

const STARFIELD: [&str; 8] = [
    "  ·         · ",
    "        ✦     ",
    "   ·       ·  ",
    "         ·    ",
    " ✦     ·      ",
    "       ·    ✦ ",
    "  ·           ",
    "     ✦    ·   ",
];

fn rocket_lines(frame: usize) -> Vec<Line<'static>> {
    let f = frame % 4;
    let shimmer = (frame / 3) % BODY.len();
    let win_c = if frame % 8 < 4 {
        GRAPE
    } else {
        Color::Rgb(180, 150, 230)
    };

    let mut lines = Vec::with_capacity(32);
    lines.push(Line::from(Span::raw("              ")));
    lines.push(Line::from(Span::raw("              ")));

    for (i, (text, base)) in BODY.iter().enumerate() {
        if i == WINDOW_IDX {
            lines.push(Line::from(vec![
                Span::styled("    / ", Style::default().fg(*base)),
                Span::styled(
                    "◆◆",
                    Style::default().fg(win_c).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" \\    ", Style::default().fg(*base)),
            ]));
        } else {
            let c = if i == shimmer {
                brighten(*base, 35)
            } else {
                *base
            };
            let mut s = Style::default().fg(c);
            if i == shimmer {
                s = s.add_modifier(Modifier::BOLD);
            }
            lines.push(Line::from(Span::styled(*text, s)));
        }
    }

    for (ri, text) in FIRE[f].iter().enumerate() {
        let (r, g, b) = FIRE_RGB[f][ri];
        lines.push(Line::from(Span::styled(
            *text,
            Style::default()
                .fg(Color::Rgb(r, g, b))
                .add_modifier(Modifier::BOLD),
        )));
    }

    for row in 0..20 {
        let idx = (row + frame / 4) % STARFIELD.len();
        let v = 75 + ((row * 13 + frame * 3) % 55) as u8;
        lines.push(Line::from(Span::styled(
            STARFIELD[idx],
            Style::default().fg(Color::Rgb(v, v.saturating_sub(10), v.saturating_add(25))),
        )));
    }

    lines
}

fn brighten(c: Color, amt: u8) -> Color {
    if let Color::Rgb(r, g, b) = c {
        Color::Rgb(
            r.saturating_add(amt),
            g.saturating_add(amt),
            b.saturating_add(amt),
        )
    } else {
        c
    }
}

// ── Workers: multi-column ─────────────────────────────────────────────────────
const GENE_PAD: usize = 16;
const STAT_PAD: usize = 26;
const ENTRY_W: usize = 2 + GENE_PAD + 5 + STAT_PAD;

fn workers_in_columns(active: &[(&String, &String)], content_w: usize) -> Vec<ListItem<'static>> {
    if active.is_empty() {
        return vec![ListItem::new(Line::from(Span::styled(
            "  ·  idle  ·",
            Style::default().fg(MUTED),
        )))];
    }
    let n_cols = ((content_w + 3) / (ENTRY_W + 3)).max(1);
    active
        .chunks(n_cols)
        .map(|chunk| {
            let mut spans: Vec<Span<'static>> = Vec::new();
            for (i, (gene, status)) in chunk.iter().enumerate() {
                if i > 0 {
                    spans.push(Span::styled(" │ ", Style::default().fg(MUTED)));
                }
                let pc = if status.contains("export") {
                    C_WROTE
                } else if status.contains("lasso") || status.contains("cnn") {
                    GRAPE
                } else if status.contains("fail") {
                    C_FAIL
                } else if status.contains("skip") {
                    C_SKIP
                } else {
                    LILAC
                };
                spans.push(Span::styled("✿ ", Style::default().fg(LABEL)));
                spans.push(Span::styled(
                    format!("{:<w$}", gene, w = GENE_PAD),
                    Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled("  ·  ", Style::default().fg(MUTED)));
                spans.push(Span::styled(
                    format!("{:<w$}", status, w = STAT_PAD),
                    Style::default().fg(pc),
                ));
            }
            ListItem::new(Line::from(spans))
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn format_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{} B", b)
    }
}

fn scan_dir(dir: &str) -> (u64, usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (0, 0);
    };
    let (mut bytes, mut count) = (0u64, 0usize);
    for e in entries.flatten() {
        if let Ok(m) = e.metadata() {
            if m.is_file() {
                bytes += m.len();
                count += 1;
            }
        }
    }
    (bytes, count)
}

fn format_t(secs: f64) -> String {
    let s = secs as u64;
    if s >= 3600 {
        format!("T+{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60)
    } else if s >= 60 {
        format!("T+{}m{:02}s", s / 60, s % 60)
    } else {
        format!("T+{}s", s)
    }
}

fn truncate_label(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", t)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn probe_wgpu_adapter_names() -> Vec<String> {
    let instance = Instance::default();
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out = Vec::new();
    for adapter in instance.enumerate_adapters(Backends::all()) {
        let name = adapter.get_info().name;
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    out
}

#[cfg(target_arch = "wasm32")]
fn probe_wgpu_adapter_names() -> Vec<String> {
    Vec::new()
}

fn cpu_brand_label(sys: &System) -> String {
    let b = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    b.to_string()
}

fn logical_core_count(sys: &System) -> usize {
    let n = sys.cpus().len();
    if n > 0 {
        return n;
    }
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
}

fn build_machine_hardware_line(
    sys: &System,
    gpu_names: &[String],
    train_backend: &str,
    max_chars: usize,
) -> String {
    let logical = logical_core_count(sys);
    let physical = sys.physical_core_count();
    let cpu = cpu_brand_label(sys);

    let cores = match physical {
        Some(p) if p > 0 => format!("{logical} logical / {p} physical cores"),
        _ => format!("{logical} logical cores"),
    };

    let gpu_part = if gpu_names.is_empty() {
        "0 GPUs (wgpu)".to_string()
    } else {
        let joined = gpu_names.join("; ");
        format!("{} GPU(s): {}", gpu_names.len(), joined)
    };

    let s = format!(
        "{cores}  ·  CPU: {cpu}  ·  {gpu_part}  ·  train: {train_backend}"
    );
    truncate_label(&s, max_chars)
}

fn fmt_r2_fixed(r: f64) -> String {
    if r.is_finite() {
        format!("{:>6.2}", r)
    } else {
        format!("{:>6}", "—")
    }
}

fn perf_r2_columns(inner_w: usize) -> (usize, usize) {
    const MID: usize = 2;
    const R2_COL: usize = 7;
    let w = inner_w.max(MID + 2);
    let half = (w - MID) / 2;
    let gene_w = half.saturating_sub(R2_COL);
    (half, gene_w)
}

fn rule_line(inner_w: usize) -> Line<'static> {
    let n = inner_w.clamp(8, 128);
    Line::from(Span::styled("─".repeat(n), Style::default().fg(MUTED)))
}

fn build_perf_panel_lines(st: &TrainingHudState, inner_w: usize) -> Vec<Line<'static>> {
    let n_genes = st.gene_r2_mean.len();
    if n_genes == 0 {
        return vec![Line::from(Span::styled(
            "  ·  no R² yet  ·",
            Style::default().fg(MUTED),
        ))];
    }

    let (half, gene_w) = perf_r2_columns(inner_w);

    let mut v = st.gene_r2_mean.clone();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(String, f64)> = v.iter().take(5).cloned().collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let bot5: Vec<(String, f64)> = v.iter().take(5).cloned().collect();

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(8);
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<hw$}", "▲ best R²", hw = half),
            Style::default().fg(C_TOPR2).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default().fg(MUTED)),
        Span::styled(
            format!("{:<hw$}", "▼ worst R²", hw = half),
            Style::default().fg(C_BOTR2).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(rule_line(inner_w));
    for ((g_hi, r_hi), (g_lo, r_lo)) in top5.into_iter().zip(bot5.into_iter()) {
        let g_hi_s = truncate_label(&g_hi, gene_w);
        let g_lo_s = truncate_label(&g_lo, gene_w);
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<gw$}", g_hi_s, gw = gene_w),
                Style::default().fg(TITLE),
            ),
            Span::styled(
                fmt_r2_fixed(r_hi),
                Style::default().fg(C_TOPR2).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default().fg(MUTED)),
            Span::styled(
                format!("{:<gw$}", g_lo_s, gw = gene_w),
                Style::default().fg(TITLE),
            ),
            Span::styled(
                fmt_r2_fixed(r_lo),
                Style::default().fg(C_BOTR2).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines
}

fn expand_user_path_input(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    if s == "~" {
        return std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| s.to_string());
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(h) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return format!("{}/{}", h.trim_end_matches('/'), rest);
        }
    }
    s.to_string()
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    let x = area.x.saturating_add(area.width.saturating_sub(width) / 2);
    let y = area.y.saturating_add(area.height.saturating_sub(height) / 2);
    Rect::new(x, y, width, height)
}

fn prompt_aborts(key: &event::KeyEvent) -> bool {
    key.code == KeyCode::Esc
        || (key.modifiers.contains(KeyModifiers::SHIFT)
            && matches!(key.code, KeyCode::Char('q' | 'Q')))
}

/// When no `.h5ad` path is configured: compact centered prompts for AnnData then output directory.
/// **Enter** confirms each step. **Esc** or **Shift+Q** exits without starting training.
pub fn run_dataset_paths_prompt(default_output_dir: &str) -> anyhow::Result<Option<(String, String)>> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, crossterm::cursor::Show)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;

    let default_out = default_output_dir.trim().to_string();
    let mut step: u8 = 0;
    let mut adata_input = String::new();
    let mut output_input = default_out.clone();
    let mut err_line: Option<String> = None;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            f.render_widget(Block::default().style(Style::default().bg(BG)), area);

            let popup_w = ((area.width * 55) / 100).clamp(48, 72).min(area.width.saturating_sub(4));
            let popup_h = 13u16.min(area.height.saturating_sub(2));
            let popup_area = centered_rect(area, popup_w, popup_h);

            let (title, help): (&str, Vec<Line>) = if step == 0 {
                (
                    " AnnData (.h5ad) ",
                    vec![
                        Line::from(Span::styled(
                            "No dataset path in config or CLI.",
                            Style::default().fg(MUTED),
                        )),
                        Line::from(Span::styled(
                            "Path to spatial .h5ad, then Enter.",
                            Style::default().fg(VALUE),
                        )),
                        Line::from(Span::styled(
                            "Esc / Shift+Q exit · ~/ expanded",
                            Style::default().fg(MUTED),
                        )),
                    ],
                )
            } else {
                let adata_disp = if adata_input.chars().count() > 48 {
                    format!(
                        "{}…",
                        adata_input.chars().take(45).collect::<String>()
                    )
                } else {
                    adata_input.clone()
                };
                (
                    " Output directory ",
                    vec![
                        Line::from(Span::styled(
                            format!("AnnData: {adata_disp}"),
                            Style::default().fg(MUTED),
                        )),
                        Line::from(Span::styled(
                            "Directory for *_betadata.feather (created if missing).",
                            Style::default().fg(VALUE),
                        )),
                        Line::from(Span::styled(
                            "Esc / Shift+Q exit · Enter confirm",
                            Style::default().fg(MUTED),
                        )),
                    ],
                )
            };

            let input_ref = if step == 0 {
                &adata_input
            } else {
                &output_input
            };

            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(TEL_BORD))
                .title(Span::styled(
                    title,
                    Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(popup_area);
            f.render_widget(block, popup_area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(6),
                    Constraint::Length(3),
                    Constraint::Length(2),
                ])
                .split(inner);

            let help_w = Paragraph::new(help)
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(OUTER_BORD)),
                );
            f.render_widget(help_w, chunks[0]);

            let path_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(SKY))
                .title(Span::styled(" path ", Style::default().fg(LABEL)));
            let path_para = Paragraph::new(Line::from(vec![Span::styled(
                if input_ref.is_empty() {
                    " "
                } else {
                    input_ref.as_str()
                },
                Style::default().fg(TITLE),
            )]))
            .block(path_block);
            f.render_widget(path_para, chunks[1]);

            let msg = err_line.as_deref().unwrap_or(" ");
            let err_c = if err_line.is_some() {
                C_FAIL
            } else {
                MUTED
            };
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(msg, Style::default().fg(err_c))))
                    .wrap(Wrap { trim: true }),
                chunks[2],
            );
        })?;

        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if prompt_aborts(&key) {
            break None;
        }

        let input_mut = if step == 0 {
            &mut adata_input
        } else {
            &mut output_input
        };

        match key.code {
            KeyCode::Enter => {
                let expanded = expand_user_path_input(input_mut);
                if step == 0 {
                    if expanded.is_empty() {
                        err_line = Some("Path cannot be empty.".to_string());
                        continue;
                    }
                    if !Path::new(&expanded).exists() {
                        err_line = Some(format!("Not found: {}", expanded));
                        continue;
                    }
                    if !expanded.to_lowercase().ends_with(".h5ad") {
                        err_line = Some("File should end with .h5ad".to_string());
                        continue;
                    }
                    adata_input = expanded;
                    err_line = None;
                    if output_input.trim().is_empty() {
                        if let Ok(s) = crate::config::default_output_dir_for_adata_path(
                            Path::new(&adata_input),
                        ) {
                            output_input = s;
                        }
                    }
                    step = 1;
                    continue;
                }
                if expanded.is_empty() {
                    err_line = Some("Output directory cannot be empty.".to_string());
                    continue;
                }
                let p = Path::new(&expanded);
                if p.exists() && !p.is_dir() {
                    err_line = Some("Path exists but is not a directory.".to_string());
                    continue;
                }
                break Some((adata_input.clone(), expanded));
            }
            KeyCode::Backspace => {
                input_mut.pop();
                err_line = None;
            }
            KeyCode::Char(c) if !c.is_control() => {
                input_mut.push(c);
                err_line = None;
            }
            _ => {}
        }
    };

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    let _ = io::stdout().flush();
    Ok(result)
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingDashboardExit {
    Completed,
    /// User pressed Shift+Q — terminal restored; caller should exit the process without waiting on workers.
    ForceQuit,
}

fn is_shift_q(key: &event::KeyEvent) -> bool {
    key.kind == KeyEventKind::Press
        && key.modifiers.contains(KeyModifiers::SHIFT)
        && matches!(key.code, KeyCode::Char('q' | 'Q'))
}

pub fn run_training_dashboard(hud: TrainingHud) -> anyhow::Result<TrainingDashboardExit> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;

    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );
    let mut last_sys = Instant::now();
    let mut last_dir_scan = Instant::now();
    let mut last_gpu_probe = Instant::now();
    let mut cached_gpu_names = probe_wgpu_adapter_names();
    let t0 = Instant::now();

    let mut dir_bytes: u64 = 0;
    let mut dir_files: usize = 0;

    let output_dir = hud
        .lock()
        .map(|st| st.output_dir.clone())
        .unwrap_or_default();

    let mut dashboard_exit = TrainingDashboardExit::Completed;
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "operator".to_string());

    let perf_cell: RefCell<(u64, Instant, usize, Vec<Line<'static>>)> = RefCell::new((
        0,
        Instant::now() - Duration::from_secs(3600),
        0,
        vec![Line::from(Span::styled(
            "  ·  no R² yet  ·",
            Style::default().fg(MUTED),
        ))],
    ));

    loop {
        if last_sys.elapsed() > HARDWARE_POLL_INTERVAL {
            sys.refresh_cpu_all();
            sys.refresh_memory();
            last_sys = Instant::now();
        }
        if last_dir_scan.elapsed() > Duration::from_secs(2) {
            (dir_bytes, dir_files) = scan_dir(&output_dir);
            last_dir_scan = Instant::now();
        }
        if last_gpu_probe.elapsed() > HARDWARE_POLL_INTERVAL {
            cached_gpu_names = probe_wgpu_adapter_names();
            last_gpu_probe = Instant::now();
        }
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if is_shift_q(&key) {
                    if let Ok(st) = hud.lock() {
                        st.cancel_requested.store(true, Ordering::Relaxed);
                    }
                    dashboard_exit = TrainingDashboardExit::ForceQuit;
                    break;
                }
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    if let Ok(st) = hud.lock() {
                        st.cancel_requested.store(true, Ordering::Relaxed);
                    }
                }
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('v') {
                    if let Ok(mut st) = hud.lock() {
                        st.show_pipeline_timing = !st.show_pipeline_timing;
                    }
                }
            }
        }

        let done = hud.lock().map(|g| g.finished.is_some()).unwrap_or(true);
        let frame = (t0.elapsed().as_millis() / 200) as usize;

        terminal.draw(|f| {
            let area = f.area();
            let bg = Style::default().bg(BG);
            f.render_widget(Block::default().style(bg), area);

            let Ok(st) = hud.lock() else { return };

            let cpu_pct = sys.global_cpu_usage();
            let used_mem = sys.used_memory();
            let total_mem = sys.total_memory().max(1);
            let mem_pct = (used_mem as f64 / total_mem as f64) * 100.0;
            let elapsed = st.elapsed_secs();

            let outer = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(OUTER_BORD).add_modifier(Modifier::DIM))
                .style(bg);
            let inner = outer.inner(area);
            f.render_widget(outer, area);

            let vchunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(20),
                    Constraint::Length(3),
                    Constraint::Length(1),
                ])
                .split(inner);

            // ── Header ────────────────────────────────────────────────────────
            let mode = match st.run_config.cnn_training_mode.as_str() {
                "full" => "full",
                "hybrid" => "hybrid",
                _ => "seed",
            };
            let status_txt = if st.should_cancel() {
                "Stopping"
            } else {
                "Running"
            };
            let status_c = if st.should_cancel() { C_FAIL } else { VALUE };
            let hw_w = vchunks[0].width.saturating_sub(2) as usize;
            let hw_line = build_machine_hardware_line(
                &sys,
                &cached_gpu_names,
                st.run_config.compute_backend.as_str(),
                hw_w.max(12),
            );

            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled(
                            " ✿ ",
                            Style::default().fg(GRAPE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "SpaceTravLR",
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(MUTED)),
                        Span::styled(
                            format!("@{}", username),
                            Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(MUTED)),
                        Span::styled(
                            format_t(elapsed),
                            Style::default().fg(VALUE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(MUTED)),
                        Span::styled(mode, Style::default().fg(SKY)),
                        Span::styled("  ·  ", Style::default().fg(MUTED)),
                        Span::styled(status_txt, Style::default().fg(status_c)),
                        Span::styled(" ✿", Style::default().fg(GRAPE)),
                    ]),
                    Line::from(Span::styled(
                        hw_line,
                        Style::default().fg(MUTED),
                    )),
                ]),
                vchunks[0],
            );

            // ── Main: [telemetry + workers] | rocket ──────────────────────────
            let hchunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(1), Constraint::Length(ROCKET_PANEL_W)])
                .split(vchunks[1]);

            let left = if st.show_pipeline_timing {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(10),
                        Constraint::Length(8),
                        Constraint::Min(4),
                    ])
                    .split(hchunks[0])
            } else {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(10), Constraint::Min(4)])
                    .split(hchunks[0])
            };
            let top_panels = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(left[0]);

            if st.show_pipeline_timing {
                let act_w = left[1].width.saturating_sub(2) as usize;
                let mut act_lines: Vec<Line> = Vec::new();

                let eta = st
                    .eta_secs()
                    .map(|s| format!("{:.0}s", s))
                    .unwrap_or_else(|| "—".into());
                let gpm = if elapsed > 1.0 {
                    format!("{:.2}/min", (st.genes_rounds as f64) * 60.0 / elapsed)
                } else {
                    "—".into()
                };
                act_lines.push(Line::from(vec![
                    Span::styled("ETA ", Style::default().fg(LABEL)),
                    Span::styled(eta, Style::default().fg(VALUE)),
                    Span::styled("  ·  ", Style::default().fg(MUTED)),
                    Span::styled("RATE ", Style::default().fg(LABEL)),
                    Span::styled(gpm, Style::default().fg(SKY)),
                ]));

                for entry in st.activity_log.iter() {
                    let truncated = truncate_label(entry, act_w.max(12));
                    act_lines.push(Line::from(Span::styled(
                        truncated,
                        Style::default().fg(LILAC),
                    )));
                }
                if act_lines.len() <= 1 {
                    act_lines.push(Line::from(Span::styled(
                        "  ·  waiting for pipeline…",
                        Style::default().fg(MUTED),
                    )));
                }
                f.render_widget(
                    Paragraph::new(act_lines)
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_style(Style::default().fg(TEL_BORD))
                                .title(Span::styled(
                                    " Pipeline & timing (v) ",
                                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                                )),
                        )
                        .style(bg),
                    left[1],
                );
            }

            let work_area = if st.show_pipeline_timing { left[2] } else { left[1] };
            let work_row = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(10), Constraint::Min(32), Constraint::Min(16)])
                .split(work_area);

            let sep = || Span::styled("  ·  ", Style::default().fg(MUTED));
            let lbl = |s: &'static str| Span::styled(s, Style::default().fg(LABEL));
            let val = |s: String, c: Color| Span::styled(s, Style::default().fg(c));

            let rc = &st.run_config;
            let cfg_disp = truncate_label(
                &rc.config_source,
                (top_panels[0].width as usize).saturating_sub(24),
            );
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![lbl("CONFIG  "), val(cfg_disp, MUTED)]),
                    Line::from(vec![
                        lbl("BACKEND  "),
                        val(rc.compute_backend.clone(), VALUE),
                        sep(),
                        lbl("LAYER  "),
                        val(rc.layer.clone(), SKY),
                    ]),
                    Line::from(vec![
                        lbl("OBS  "),
                        val(rc.cluster_annot.clone(), LILAC),
                        sep(),
                        lbl("SPATIAL  "),
                        val(
                            format!(
                                "r={:.1}  dim={}  contact={:.1}",
                                rc.spatial_radius, rc.spatial_dim, rc.contact_distance
                            ),
                            VALUE,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("LASSO  "),
                        val(
                            format!(
                                "l1={:.3}  group={:.3}  n_iter={}  tol={:.0e}",
                                rc.l1_reg, rc.group_reg, rc.n_iter, rc.tol
                            ),
                            GRAPE,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("TRAIN  "),
                        val(
                            format!(
                                "lr={:.3}  score≥{:.2}  epochs={}/gene",
                                rc.learning_rate, rc.score_threshold, rc.epochs_per_gene
                            ),
                            SKY,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("GRN  "),
                        val(
                            format!(
                                "tf_lig≥{:.2}  max_lr={}  top_lr={}",
                                rc.tf_ligand_cutoff, rc.max_lr_pairs, rc.top_lr_pairs
                            ),
                            MUTED,
                        ),
                    ]),
                    Line::from(vec![lbl("GENES  "), val(rc.gene_selection.clone(), TITLE)]),
                    Line::from(vec![
                        lbl("TRAIN_MODE  "),
                        val(rc.cnn_training_mode.clone(), SKY),
                    ]),
                ])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(TEL_BORD))
                        .title(Span::styled(
                            " Run configuration ",
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        )),
                )
                .style(bg),
                top_panels[0],
            );

            let path_s = if st.dataset_path.len() > 55 {
                format!("…{}", &st.dataset_path[st.dataset_path.len() - 52..])
            } else {
                st.dataset_path.clone()
            };
            let dir_s = if st.output_dir.len() > 40 {
                format!("…{}", &st.output_dir[st.output_dir.len() - 37..])
            } else {
                st.output_dir.clone()
            };
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![lbl("SRC  "), val(path_s, MUTED)]),
                    Line::from(vec![
                        lbl("GRID  "),
                        val(
                            format!("{} cells  ·  {} clusters", st.n_cells, st.n_clusters),
                            VALUE,
                        ),
                        sep(),
                        lbl("WORKERS  "),
                        val(format!("{}", st.n_parallel), GRAPE),
                    ]),
                    Line::from(vec![
                        lbl("EXPORT  "),
                        val(
                            format!(
                                "seed-only {}  ·  CNN {}",
                                st.genes_exported_seed_only, st.genes_exported_cnn
                            ),
                            VALUE,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("EPOCH  "),
                        val(format!("{}/gene", st.epochs_per_gene), LILAC),
                    ]),
                    Line::from(vec![
                        lbl("CPU  "),
                        val(format!("{cpu_pct:5.1}%"), SKY),
                        sep(),
                        lbl("MEM  "),
                        val(format!("{mem_pct:5.1}%"), SKY),
                        sep(),
                        lbl("RAM  "),
                        val(
                            format!("{}/{} MiB", used_mem / 1024 / 1024, total_mem / 1024 / 1024),
                            MUTED,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("OUT  "),
                        val(dir_s, MUTED),
                        sep(),
                        lbl("SIZE  "),
                        val(format_bytes(dir_bytes), VALUE),
                        sep(),
                        lbl("FILES  "),
                        val(format!("{}", dir_files), LILAC),
                    ]),
                ])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(TEL_BORD))
                        .title(Span::styled(
                            " Live telemetry ",
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        )),
                )
                .style(bg),
                top_panels[1],
            );

            // ── Workers ───────────────────────────────────────────────────────
            let mut active: Vec<(&String, &String)> = st.active_genes.iter().collect();
            active.sort_by_key(|(g, _)| g.as_str());
            let cw = (work_row[0].width as usize).saturating_sub(2);

            f.render_widget(
                List::new(workers_in_columns(&active, cw))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(WORK_BORD))
                            .title(Span::styled(
                                format!(
                                    " ✦ ACTIVE WORKERS ({}/{}) ",
                                    st.active_genes.len(),
                                    st.n_parallel
                                ),
                                Style::default().fg(GRAPE).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .style(bg),
                work_row[0],
            );

            let perf_inner = work_row[1].width.saturating_sub(2) as usize;
            {
                let now = Instant::now();
                let mut cell = perf_cell.borrow_mut();
                let stale_gen = st.perf_stats_generation != cell.0;
                let stale_t = now.duration_since(cell.1) > Duration::from_millis(450);
                let stale_w = cell.2 != perf_inner;
                if stale_gen || stale_t || stale_w {
                    cell.3 = build_perf_panel_lines(&st, perf_inner);
                    cell.0 = st.perf_stats_generation;
                    cell.1 = now;
                    cell.2 = perf_inner;
                }
            }
            let perf_snapshot = perf_cell.borrow().3.clone();
            f.render_widget(
                Paragraph::new(perf_snapshot)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(PERF_BORD))
                            .title(Span::styled(
                                " ✦ LASSO R² ",
                                Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .style(bg),
                work_row[1],
            );

            // ── Cell types ────────────────────────────────────────────────
            let cell_panel_w = work_row[2].width.saturating_sub(2) as usize;
            let cell_panel_h = work_row[2].height.saturating_sub(2) as usize;
            let cell_panel_rows = cell_panel_h.max(1);

            let mut cell_counts = st.cell_type_counts.clone();
            cell_counts.sort_by(|a, b| b.1.cmp(&a.1));

            let label_w = cell_panel_w.saturating_sub(10).max(3);
            let count_w = cell_panel_w.saturating_sub(label_w).max(1);

            let mut cell_lines: Vec<Line<'static>> = Vec::new();
            if cell_counts.is_empty() {
                cell_lines.push(Line::from(Span::styled(
                    "  ·  obs.cell_type not found  ·",
                    Style::default().fg(MUTED),
                )));
            } else {
                for (ct, count) in cell_counts.iter().take(cell_panel_rows) {
                    let ct_disp = truncate_label(ct, label_w);
                    let left = format!("{:<lw$}", ct_disp, lw = label_w);
                    let right = format!("{:>cw$}", count, cw = count_w);
                    cell_lines.push(Line::from(vec![
                        Span::styled(
                            left,
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            right,
                            Style::default().fg(VALUE).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                if cell_counts.len() > cell_panel_rows {
                    cell_lines.push(Line::from(Span::styled(
                        format!("… +{} more", cell_counts.len() - cell_panel_rows),
                        Style::default().fg(MUTED),
                    )));
                }
            }

            f.render_widget(
                Paragraph::new(cell_lines)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(TEL_BORD))
                            .title(Span::styled(
                                " Cell Types ",
                                Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .style(bg),
                work_row[2],
            );

            // ── Rocket ────────────────────────────────────────────────────────
            f.render_widget(
                Paragraph::new(rocket_lines(frame))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(ROCKET_BORD))
                            .title(Span::styled(" Status ", Style::default().fg(GRAPE))),
                    )
                    .style(bg),
                hchunks[1],
            );

            // ── Gene progress ─────────────────────────────────────────────────
            let total = st.total_genes.max(1) as u64;
            let pos = st.genes_rounds.min(st.total_genes) as u64;
            let ratio = (pos as f64 / total as f64).clamp(0.0, 1.0);
            f.render_widget(
                Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(GAUGE_BORD))
                            .title(Span::styled(
                                " Gene progress ",
                                Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .gauge_style(
                        Style::default()
                            .fg(GAUGE_FILL)
                            .bg(GAUGE_EMPTY)
                            .add_modifier(Modifier::BOLD),
                    )
                    .use_unicode(true)
                    .ratio(ratio)
                    .label(Span::styled(
                        format!(
                            "{}/{}  ·  ok {}  skip {}  fail {}  orphan {}",
                            pos,
                            total,
                            st.genes_done,
                            st.genes_skipped,
                            st.genes_failed,
                            st.genes_orphan
                        ),
                        Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                    )),
                vchunks[2],
            );

            // ── Footer ────────────────────────────────────────────────────────
            let footer = if st.should_cancel() {
                Line::from(Span::styled(
                    " Stopping after in-flight genes finish… ",
                    Style::default().fg(C_FAIL).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(Span::styled(
                    " q: graceful exit   shift+q: french leave ",
                    Style::default().fg(MUTED),
                ))
            };
            f.render_widget(Paragraph::new(footer).wrap(Wrap { trim: true }), vchunks[3]);
        })?;

        if done {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        crossterm::cursor::Show
    )?;
    Ok(dashboard_exit)
}
