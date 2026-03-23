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
use std::io::stdout;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

// ── Palette ───────────────────────────────────────────────────────────────────
const BG:          Color = Color::Rgb(18, 12, 28);
const OUTER_BORD:  Color = Color::Rgb(180, 140, 210);
const TEL_BORD:    Color = Color::Rgb(210, 140, 180);
const GAUGE_BORD:  Color = Color::Rgb(140, 180, 230);
const WORK_BORD:   Color = Color::Rgb(200, 140, 220);
const ROCKET_BORD: Color = Color::Rgb(200, 160, 240);
const GAUGE_FILL:  Color = Color::Rgb(200, 140, 200);
const GAUGE_EMPTY: Color = Color::Rgb(35, 22, 45);

const LABEL:  Color = Color::Rgb(255, 160, 185);
const VALUE:  Color = Color::Rgb(160, 230, 200);
const LILAC:  Color = Color::Rgb(195, 170, 240);
const SKY:    Color = Color::Rgb(155, 205, 250);
const GRAPE:  Color = Color::Rgb(230, 180, 255);
const MUTED:  Color = Color::Rgb(120, 100, 145);
const TITLE:  Color = Color::Rgb(240, 190, 220);

const C_WROTE: Color = Color::Rgb(150, 230, 195);
const C_FAIL:  Color = Color::Rgb(255, 130, 155);
const C_SKIP:  Color = Color::Rgb(110, 95,  130);
const C_TOPR2: Color = Color::Rgb(130, 235, 190);
const C_BOTR2: Color = Color::Rgb(255, 160, 170);

const PERF_BORD: Color = Color::Rgb(160, 150, 220);

// ── Rocket ────────────────────────────────────────────────────────────────────
// Compact ASCII rocket — every line is exactly 14 display columns.
// Panel = 14 content + 2 border = 16 terminal columns.

const ROCKET_PANEL_W: u16 = 16;
const WINDOW_IDX: usize = 2;

const BODY: [(&str, Color); 8] = [
    ("      /\\      ", SKY),
    ("     /  \\     ", SKY),
    ("    / ** \\    ", SKY),     // window — rendered with colored ◆◆ spans
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
    [(255, 190,  70), (255, 120, 30)],
    [(255, 240, 150), (255, 180, 60)],
    [(255, 200,  90), (255, 100, 20)],
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
    let f       = frame % 4;
    let shimmer = (frame / 3) % BODY.len();
    let win_c   = if frame % 8 < 4 { GRAPE } else { Color::Rgb(180, 150, 230) };

    let mut lines = Vec::with_capacity(32);

    for (i, (text, base)) in BODY.iter().enumerate() {
        if i == WINDOW_IDX {
            lines.push(Line::from(vec![
                Span::styled("    / ", Style::default().fg(*base)),
                Span::styled("◆◆", Style::default().fg(win_c).add_modifier(Modifier::BOLD)),
                Span::styled(" \\    ", Style::default().fg(*base)),
            ]));
        } else {
            let c = if i == shimmer { brighten(*base, 35) } else { *base };
            let mut s = Style::default().fg(c);
            if i == shimmer { s = s.add_modifier(Modifier::BOLD); }
            lines.push(Line::from(Span::styled(*text, s)));
        }
    }

    for (ri, text) in FIRE[f].iter().enumerate() {
        let (r, g, b) = FIRE_RGB[f][ri];
        lines.push(Line::from(Span::styled(
            *text,
            Style::default().fg(Color::Rgb(r, g, b)).add_modifier(Modifier::BOLD),
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
        Color::Rgb(r.saturating_add(amt), g.saturating_add(amt), b.saturating_add(amt))
    } else {
        c
    }
}

// ── Workers: multi-column ─────────────────────────────────────────────────────
const GENE_PAD: usize = 16;
const STAT_PAD: usize = 26;
const ENTRY_W:  usize = 2 + GENE_PAD + 5 + STAT_PAD;

fn workers_in_columns(
    active: &[(&String, &String)],
    content_w: usize,
) -> Vec<ListItem<'static>> {
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
                if i > 0 { spans.push(Span::styled(" │ ", Style::default().fg(MUTED))); }
                let pc = if status.contains("export") { C_WROTE }
                    else if status.contains("lasso") || status.contains("cnn") { GRAPE }
                    else if status.contains("fail") { C_FAIL }
                    else if status.contains("skip") { C_SKIP }
                    else { LILAC };
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
    if b >= 1 << 30      { format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64) }
    else if b >= 1 << 20 { format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64) }
    else if b >= 1 << 10 { format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64) }
    else                 { format!("{} B", b) }
}

fn scan_dir(dir: &str) -> (u64, usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return (0, 0) };
    let (mut bytes, mut count) = (0u64, 0usize);
    for e in entries.flatten() {
        if let Ok(m) = e.metadata() {
            if m.is_file() { bytes += m.len(); count += 1; }
        }
    }
    (bytes, count)
}

fn format_t(secs: f64) -> String {
    let s = secs as u64;
    if s >= 3600 { format!("T+{}h{:02}m{:02}s", s / 3600, (s / 60) % 60, s % 60) }
    else if s >= 60 { format!("T+{}m{:02}s", s / 60, s % 60) }
    else { format!("T+{}s", s) }
}

fn truncate_label(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", t)
    }
}

fn build_perf_panel_lines(st: &TrainingHudState) -> Vec<Line<'static>> {
    let n_genes = st.gene_r2_mean.len();
    if n_genes == 0 {
        return vec![Line::from(Span::styled(
            "  ·  no R² yet  ·",
            Style::default().fg(MUTED),
        ))];
    }

    let grand: f64 =
        st.gene_r2_mean.iter().map(|(_, r)| r).sum::<f64>() / n_genes as f64;

    let mut v = st.gene_r2_mean.clone();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top5: Vec<(String, f64)> = v.iter().take(5).cloned().collect();
    v.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

    lines.push(Line::from(Span::styled("────────────────────────────", Style::default().fg(MUTED))));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<18}", "▲ best"),
            Style::default().fg(C_TOPR2).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default().fg(MUTED)),
        Span::styled(
            format!("{:<18}", "▼ worst"),
            Style::default().fg(C_BOTR2).add_modifier(Modifier::BOLD),
        ),
    ]));
    for ((g_hi, r_hi), (g_lo, r_lo)) in top5.into_iter().zip(bot5.into_iter()) {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<12}", truncate_label(&g_hi, 11)),
                Style::default().fg(TITLE),
            ),
            Span::styled(
                format!(" {:>5}", format!("{:.3}", r_hi)),
                Style::default().fg(C_TOPR2).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default().fg(MUTED)),
            Span::styled(
                format!("{:<12}", truncate_label(&g_lo, 11)),
                Style::default().fg(TITLE),
            ),
            Span::styled(
                format!(" {:>5}", format!("{:.3}", r_lo)),
                Style::default().fg(C_BOTR2).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines
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
    let mut last_sys      = Instant::now();
    let mut last_dir_scan = Instant::now();
    let t0                = Instant::now();

    let mut dir_bytes: u64   = 0;
    let mut dir_files: usize = 0;

    let output_dir = hud.lock().map(|st| st.output_dir.clone()).unwrap_or_default();

    let mut dashboard_exit = TrainingDashboardExit::Completed;

    let perf_cell: RefCell<(u64, Instant, Vec<Line<'static>>)> = RefCell::new((
        0,
        Instant::now() - Duration::from_secs(3600),
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

        let done  = hud.lock().map(|g| g.finished.is_some()).unwrap_or(true);
        let frame = (t0.elapsed().as_millis() / 200) as usize;

        terminal.draw(|f| {
            let area = f.area();
            let bg   = Style::default().bg(BG);
            f.render_widget(Block::default().style(bg), area);

            let Ok(st) = hud.lock() else { return };

            let cpu_pct   = sys.global_cpu_usage();
            let used_mem  = sys.used_memory();
            let total_mem = sys.total_memory().max(1);
            let mem_pct   = (used_mem as f64 / total_mem as f64) * 100.0;
            let elapsed   = st.elapsed_secs();

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
                    Constraint::Min(16),
                    Constraint::Length(5),
                    Constraint::Length(1),
                ])
                .split(inner);

            // ── Header ────────────────────────────────────────────────────────
            let mode = if st.full_cnn { "CNN ✦ SPATIAL" } else { "SEED ✦ LASSO" };
            let status_txt = if st.should_cancel() { "DRAINING" } else { "NOMINAL" };
            let status_c   = if st.should_cancel() { C_FAIL } else { VALUE };

            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(" ✿ ", Style::default().fg(GRAPE).add_modifier(Modifier::BOLD)),
                    Span::styled("SPACETRAVLR", Style::default().fg(TITLE).add_modifier(Modifier::BOLD)),
                    Span::styled("  ·  ", Style::default().fg(MUTED)),
                    Span::styled(format_t(elapsed), Style::default().fg(VALUE).add_modifier(Modifier::BOLD)),
                    Span::styled("  ·  ", Style::default().fg(MUTED)),
                    Span::styled(mode, Style::default().fg(SKY)),
                    Span::styled("  ·  ", Style::default().fg(MUTED)),
                    Span::styled(status_txt, Style::default().fg(status_c)),
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
                .constraints([Constraint::Length(7), Constraint::Min(6)])
                .split(hchunks[0]);

            let work_row = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(10), Constraint::Length(36)])
                .split(left[1]);

            // ── Telemetry ─────────────────────────────────────────────────────
            let path_s = if st.dataset_path.len() > 55 {
                format!("…{}", &st.dataset_path[st.dataset_path.len() - 52..])
            } else { st.dataset_path.clone() };
            let dir_s = if st.output_dir.len() > 40 {
                format!("…{}", &st.output_dir[st.output_dir.len() - 37..])
            } else { st.output_dir.clone() };
            let eta = st.eta_secs().map(|s| format!("{:.0}s", s)).unwrap_or_else(|| "—".into());
            let gpm = if elapsed > 1.0 {
                format!("{:.2}/min", (st.genes_rounds as f64) * 60.0 / elapsed)
            } else { "—".into() };

            let sep = || Span::styled("  ·  ", Style::default().fg(MUTED));
            let lbl = |s: &'static str| Span::styled(s, Style::default().fg(LABEL));
            let val = |s: String, c: Color| Span::styled(s, Style::default().fg(c));

            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![lbl("SRC  "), val(path_s, MUTED)]),
                    Line::from(vec![
                        lbl("GRID  "),
                        val(format!("{} cells  ·  {} clusters", st.n_cells, st.n_clusters), VALUE),
                        sep(), lbl("WORKERS  "), val(format!("{}", st.n_parallel), GRAPE),
                    ]),
                    Line::from(vec![
                        lbl("EPOCH  "), val(format!("{}/gene", st.epochs_per_gene), LILAC),
                        sep(), lbl("ETA  "), val(eta, VALUE),
                        sep(), lbl("RATE  "), val(gpm, SKY),
                    ]),
                    Line::from(vec![
                        lbl("CPU  "), val(format!("{cpu_pct:5.1}%"), SKY),
                        sep(), lbl("MEM  "), val(format!("{mem_pct:5.1}%"), SKY),
                        sep(), lbl("RAM  "),
                        val(format!("{}/{} MiB", used_mem / 1024 / 1024, total_mem / 1024 / 1024), MUTED),
                    ]),
                    Line::from(vec![
                        lbl("OUT  "), val(dir_s, MUTED),
                        sep(), lbl("SIZE  "), val(format_bytes(dir_bytes), VALUE),
                        sep(), lbl("FILES  "), val(format!("{}", dir_files), LILAC),
                    ]),
                ])
                .block(Block::default().borders(Borders::ALL)
                    .border_style(Style::default().fg(TEL_BORD))
                    .title(Span::styled(" ✦ TELEMETRY ", Style::default().fg(TITLE).add_modifier(Modifier::BOLD))))
                .style(bg),
                left[0],
            );

            // ── Workers ───────────────────────────────────────────────────────
            let mut active: Vec<(&String, &String)> = st.active_genes.iter().collect();
            active.sort_by_key(|(g, _)| g.as_str());
            let cw = (work_row[0].width as usize).saturating_sub(2);

            f.render_widget(
                List::new(workers_in_columns(&active, cw))
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(WORK_BORD))
                        .title(Span::styled(
                            format!(" ✦ ACTIVE WORKERS ({}/{}) ", st.active_genes.len(), st.n_parallel),
                            Style::default().fg(GRAPE).add_modifier(Modifier::BOLD),
                        )))
                    .style(bg),
                work_row[0],
            );

            // ── R² performance (throttled rebuild) ────────────────────────────
            {
                let now = Instant::now();
                let mut cell = perf_cell.borrow_mut();
                let stale_gen = st.perf_stats_generation != cell.0;
                let stale_t = now.duration_since(cell.1) > Duration::from_millis(450);
                if stale_gen || stale_t {
                    cell.2 = build_perf_panel_lines(&st);
                    cell.0 = st.perf_stats_generation;
                    cell.1 = now;
                }
            }
            let perf_snapshot = perf_cell.borrow().2.clone();
            f.render_widget(
                Paragraph::new(perf_snapshot)
                    .wrap(Wrap { trim: false })
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
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(ROCKET_BORD))
                        .title(Span::styled(" 🚀 ", Style::default().fg(GRAPE))))
                    .style(bg),
                hchunks[1],
            );

            // ── Gauge ─────────────────────────────────────────────────────────
            let total = st.total_genes.max(1) as u64;
            let pos   = st.genes_rounds.min(st.total_genes) as u64;
            let ratio = (pos as f64 / total as f64).clamp(0.0, 1.0);
            f.render_widget(
                Gauge::default()
                    .block(Block::default().borders(Borders::ALL)
                        .border_style(Style::default().fg(GAUGE_BORD))
                        .title(Span::styled(" ✦ GENE INDEX ",
                            Style::default().fg(SKY).add_modifier(Modifier::BOLD))))
                    .gauge_style(Style::default().fg(GAUGE_FILL).bg(GAUGE_EMPTY).add_modifier(Modifier::BOLD))
                    .ratio(ratio)
                    .label(Span::styled(
                        format!("{}/{}  ·  ok {}  skip {}  fail {}  orphan {}",
                            pos, total, st.genes_done, st.genes_skipped, st.genes_failed, st.genes_orphan),
                        Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                    )),
                vchunks[2],
            );

            // ── Footer ────────────────────────────────────────────────────────
            let footer = if st.should_cancel() {
                Line::from(Span::styled(
                    " draining — will abort after current genes finish ",
                    Style::default().fg(C_FAIL).add_modifier(Modifier::BOLD)))
            } else {
                Line::from(Span::styled(
                    " [q] graceful stop   [Shift+Q] kill CLI now ",
                    Style::default().fg(MUTED)))
            };
            f.render_widget(Paragraph::new(footer).wrap(Wrap { trim: true }), vchunks[3]);
        })?;

        if done { break; }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, crossterm::cursor::Show)?;
    Ok(dashboard_exit)
}
