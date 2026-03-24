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
const C_TOPR2: Color = Color::Rgb(69, 133, 136); // provided aqua
const C_BOTR2: Color = Color::Rgb(204, 36, 29); // provided red

const PERF_BORD: Color = Color::Rgb(215, 153, 33); // provided yellow

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

fn compute_hardware_summary(notice: &str, max_chars: usize) -> String {
    let raw = if let Some((_, rest)) = notice.split_once(':') {
        rest.trim()
    } else {
        notice.trim()
    };
    truncate_label(raw, max_chars)
}

fn fmt_r2_fixed(r: f64) -> String {
    if r.is_finite() {
        format!("{:>7.3}", r)
    } else {
        format!("{:>7}", "—")
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

    let grand: f64 = st.gene_r2_mean.iter().map(|(_, r)| r).sum::<f64>() / n_genes as f64;

    let mut v = st.gene_r2_mean.clone();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(String, f64)> = v.iter().take(5).cloned().collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let bot5: Vec<(String, f64)> = v.iter().take(5).cloned().collect();

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(12);
    lines.push(Line::from(vec![
        Span::styled("μ R²/gene ", Style::default().fg(LABEL)),
        Span::styled(
            format!("{:.3}", grand),
            Style::default().fg(VALUE).add_modifier(Modifier::BOLD),
        ),
    ]));

    let mut cluster_parts: Vec<(usize, f64)> = st
        .cluster_r2_sum
        .iter()
        .zip(st.cluster_r2_count.iter())
        .enumerate()
        .filter_map(|(i, (&sum, &cnt))| {
            if cnt > 0 {
                Some((i, sum / f64::from(cnt)))
            } else {
                None
            }
        })
        .collect();
    cluster_parts.sort_by_key(|(i, _)| *i);

    if !cluster_parts.is_empty() {
        let cl_mean: f64 =
            cluster_parts.iter().map(|(_, r)| r).sum::<f64>() / cluster_parts.len() as f64;
        lines.push(Line::from(vec![
            Span::styled("μ R²/cluster ", Style::default().fg(LABEL)),
            Span::styled(
                format!("{:.3}", cl_mean),
                Style::default().fg(SKY).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines.push(rule_line(inner_w));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<hw$}", "▲ best", hw = half),
            Style::default().fg(C_TOPR2).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default().fg(MUTED)),
        Span::styled(
            format!("{:<hw$}", "▼ worst", hw = half),
            Style::default().fg(C_BOTR2).add_modifier(Modifier::BOLD),
        ),
    ]));
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

/// Full-screen prompt when no `.h5ad` path is configured. **Enter** confirms, **Esc** cancels.
pub fn run_adata_path_prompt() -> anyhow::Result<Option<String>> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, crossterm::cursor::Show)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;

    let mut input = String::new();
    let mut err_line: Option<String> = None;

    let result = loop {
        terminal.draw(|f| {
            let area = f.area();
            let bg = Style::default().bg(BG);
            f.render_widget(Block::default().style(bg), area);

            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(TEL_BORD))
                .title(Span::styled(
                    " Select AnnData (.h5ad) ",
                    Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(area);
            f.render_widget(block, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(4),
                    Constraint::Min(2),
                    Constraint::Length(2),
                ])
                .split(inner);

            let help = Paragraph::new(vec![
                Line::from(Span::styled(
                    "No dataset path was set in config or on the command line.",
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    "Type the full path to your spatial .h5ad file, then press Enter.",
                    Style::default().fg(VALUE),
                )),
                Line::from(Span::styled(
                    "Esc cancel  ·  ~ and ~/ are expanded",
                    Style::default().fg(MUTED),
                )),
            ])
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(OUTER_BORD)),
            );
            f.render_widget(help, chunks[0]);

            let path_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(SKY))
                .title(Span::styled(" path ", Style::default().fg(LABEL)));
            let path_para = Paragraph::new(Line::from(vec![Span::styled(
                if input.is_empty() {
                    " "
                } else {
                    input.as_str()
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
        match key.code {
            KeyCode::Esc => break None,
            KeyCode::Enter => {
                let expanded = expand_user_path_input(&input);
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
                break Some(expanded);
            }
            KeyCode::Backspace => {
                input.pop();
                err_line = None;
            }
            KeyCode::Char(c) => {
                input.push(c);
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
        if last_sys.elapsed() > Duration::from_millis(350) {
            sys.refresh_cpu_all();
            sys.refresh_memory();
            last_sys = Instant::now();
        }
        if last_dir_scan.elapsed() > Duration::from_secs(2) {
            (dir_bytes, dir_files) = scan_dir(&output_dir);
            last_dir_scan = Instant::now();
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
                    Constraint::Length(1),
                    Constraint::Min(20),
                    Constraint::Length(5),
                    Constraint::Length(1),
                ])
                .split(inner);

            // ── Header ────────────────────────────────────────────────────────
            let mode = match st.run_config.cnn_training_mode.as_str() {
                "full" => "CNN spatial",
                "hybrid" => "Hybrid gated CNN",
                _ => "Seed-only lasso",
            };
            let status_txt = if st.should_cancel() {
                "Stopping"
            } else {
                "Running"
            };
            let status_c = if st.should_cancel() { C_FAIL } else { VALUE };
            let hw = compute_hardware_summary(&st.run_config.compute_notice, 44);

            f.render_widget(
                Paragraph::new(Line::from(vec![
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
                    Span::styled("  ·  ", Style::default().fg(MUTED)),
                    Span::styled(hw, Style::default().fg(MUTED)),
                    Span::styled(" ✿", Style::default().fg(GRAPE)),
                ])),
                vchunks[0],
            );

            // ── Main: [telemetry + workers] | rocket ──────────────────────────
            let hchunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(1), Constraint::Length(ROCKET_PANEL_W)])
                .split(vchunks[1]);

            let left = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(10), Constraint::Min(4)])
                .split(hchunks[0]);
            let top_panels = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(left[0]);

            let work_row = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(10), Constraint::Min(40)])
                .split(left[1]);

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
                                "r={:.4}  dim={}  contact={:.4}",
                                rc.spatial_radius, rc.spatial_dim, rc.contact_distance
                            ),
                            VALUE,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("LASSO  "),
                        val(
                            format!(
                                "l1={:.4}  group={:.4}  n_iter={}  tol={:.1e}",
                                rc.l1_reg, rc.group_reg, rc.n_iter, rc.tol
                            ),
                            GRAPE,
                        ),
                    ]),
                    Line::from(vec![
                        lbl("TRAIN  "),
                        val(
                            format!(
                                "lr={:.4}  score≥{:.3}  epochs={}/gene",
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
            let eta = st
                .eta_secs()
                .map(|s| format!("{:.0}s", s))
                .unwrap_or_else(|| "—".into());
            let gpm = if elapsed > 1.0 {
                format!("{:.2}/min", (st.genes_rounds as f64) * 60.0 / elapsed)
            } else {
                "—".into()
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
                        sep(),
                        lbl("ETA  "),
                        val(eta, VALUE),
                        sep(),
                        lbl("RATE  "),
                        val(gpm, SKY),
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

            // ── R² performance (throttled rebuild; width-aware columns) ─────
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

            // ── Gauge ─────────────────────────────────────────────────────────
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
                    " q: graceful stop   Shift+Q: exit immediately ",
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
