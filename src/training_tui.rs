use crate::training_hud::TrainingHud;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use std::io::stdout;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

pub fn run_training_dashboard(hud: TrainingHud) -> anyhow::Result<()> {
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

    loop {
        if last_sys.elapsed() > Duration::from_millis(350) {
            sys.refresh_cpu_all();
            sys.refresh_memory();
            last_sys = Instant::now();
        }

        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    if let Ok(mut st) = hud.lock() {
                        st.cancel_requested.store(true, Ordering::Relaxed);
                        st.push_log(">> Q: draining workers…".to_string());
                    }
                }
            }
        }

        let done = hud.lock().map(|g| g.finished.is_some()).unwrap_or(true);

        terminal.draw(|frame| {
            let area = frame.area();
            let bg = Style::default().bg(Color::Rgb(8, 12, 22));
            frame.render_widget(Block::default().style(bg), area);

            let Ok(st) = hud.lock() else { return };

            let cpu_pct  = sys.global_cpu_usage();
            let used_mem = sys.used_memory();
            let total_mem = sys.total_memory().max(1);
            let mem_pct  = (used_mem as f64 / total_mem as f64) * 100.0;

            let outer = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM))
                .style(bg);
            let inner = outer.inner(area);
            frame.render_widget(outer, area);

            // How many rows to give the ACTIVE WORKERS panel
            let active_rows = (st.active_genes.len().max(1) as u16 + 2).min(10);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(1),            // header
                    Constraint::Length(5),            // TELEMETRY
                    Constraint::Length(3),            // GENE INDEX gauge
                    Constraint::Length(active_rows),  // ACTIVE WORKERS (dynamic)
                    Constraint::Min(4),               // EVENT LOG
                    Constraint::Length(1),            // footer
                ])
                .split(inner);

            // ── Header ────────────────────────────────────────────────────────
            let mode = if st.full_cnn { "CNN // SPATIAL" } else { "SEED // LASSO-ONLY" };
            let header = Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                Span::styled("SPACETRAVLR", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::styled(" :: ", Style::default().fg(Color::DarkGray)),
                Span::styled("NEUROLINK", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" :: {} ", mode), Style::default().fg(Color::Cyan)),
                Span::styled("◆", Style::default().fg(Color::LightCyan)),
            ]);
            frame.render_widget(Paragraph::new(header), chunks[0]);

            // ── Telemetry ─────────────────────────────────────────────────────
            let path_short = if st.dataset_path.len() > 60 {
                format!("…{}", &st.dataset_path[st.dataset_path.len() - 57..])
            } else {
                st.dataset_path.clone()
            };
            let eta = st
                .eta_secs()
                .map(|s| format!("{:.0}s", s))
                .unwrap_or_else(|| "—".to_string());
            let elapsed = st.elapsed_secs();
            let gpm = if elapsed > 1.0 {
                format!("{:.2}/min", (st.genes_rounds as f64) * 60.0 / elapsed)
            } else {
                "—".to_string()
            };

            let tel = vec![
                Line::from(vec![
                    Span::styled("SRC ", Style::default().fg(Color::Yellow)),
                    Span::styled(path_short, Style::default().fg(Color::Gray)),
                ]),
                Line::from(vec![
                    Span::styled("GRID ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{} cells  │  {} clusters", st.n_cells, st.n_clusters),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("WORKERS ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{}", st.n_parallel),
                        Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("EPOCH ", Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{}/gene", st.epochs_per_gene), Style::default().fg(Color::White)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("ELAPSED ", Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{:.0}s", elapsed), Style::default().fg(Color::Green)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("ETA ", Style::default().fg(Color::Yellow)),
                    Span::styled(eta, Style::default().fg(Color::Green)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("RATE ", Style::default().fg(Color::Yellow)),
                    Span::styled(gpm, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![
                    Span::styled("CPU ", Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{cpu_pct:5.1}%"), Style::default().fg(Color::Cyan)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("MEM ", Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{mem_pct:5.1}%"), Style::default().fg(Color::Cyan)),
                    Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("RAM ", Style::default().fg(Color::Yellow)),
                    Span::styled(
                        format!("{}/{} MiB", used_mem / 1024 / 1024, total_mem / 1024 / 1024),
                        Style::default().fg(Color::Gray),
                    ),
                ]),
            ];
            frame.render_widget(
                Paragraph::new(tel)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Rgb(0, 140, 160)))
                            .title(Span::styled(" TELEMETRY ", Style::default().fg(Color::LightCyan))),
                    )
                    .style(bg),
                chunks[1],
            );

            // ── Gene index gauge ──────────────────────────────────────────────
            let total = st.total_genes.max(1) as u64;
            let pos   = st.genes_rounds.min(st.total_genes) as u64;
            let ratio = (pos as f64 / total as f64).clamp(0.0, 1.0);
            frame.render_widget(
                Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Rgb(80, 200, 255)))
                            .title(Span::styled(" GENE INDEX ", Style::default().fg(Color::LightCyan))),
                    )
                    .gauge_style(
                        Style::default()
                            .fg(Color::Cyan)
                            .bg(Color::Rgb(20, 30, 45))
                            .add_modifier(Modifier::BOLD),
                    )
                    .ratio(ratio)
                    .label(Span::styled(
                        format!(
                            "{}/{}  │  ok {}  skip {}  fail {}  orphan {}",
                            pos, total, st.genes_done, st.genes_skipped, st.genes_failed, st.genes_orphan
                        ),
                        Style::default().fg(Color::Gray),
                    )),
                chunks[2],
            );

            // ── Active workers ────────────────────────────────────────────────
            let mut active_sorted: Vec<(&String, &String)> = st.active_genes.iter().collect();
            active_sorted.sort_by_key(|(g, _)| g.as_str());

            let active_items: Vec<ListItem> = if active_sorted.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  —  idle  —",
                    Style::default().fg(Color::DarkGray),
                )))]
            } else {
                active_sorted
                    .iter()
                    .map(|(gene, status)| {
                        let phase_color = if status.contains("export") {
                            Color::LightGreen
                        } else if status.contains("lasso") || status.contains("cnn") {
                            Color::Yellow
                        } else {
                            Color::Gray
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled("◈ ", Style::default().fg(Color::LightMagenta)),
                            Span::styled(
                                gene.as_str(),
                                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled("  //  ", Style::default().fg(Color::DarkGray)),
                            Span::styled(status.as_str(), Style::default().fg(phase_color)),
                        ]))
                    })
                    .collect()
            };

            frame.render_widget(
                List::new(active_items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Rgb(140, 80, 255)))
                            .title(Span::styled(
                                format!(
                                    " ACTIVE WORKERS ({}/{}) ",
                                    st.active_genes.len(),
                                    st.n_parallel,
                                ),
                                Style::default().fg(Color::LightMagenta),
                            )),
                    )
                    .style(bg),
                chunks[3],
            );

            // ── Event log ─────────────────────────────────────────────────────
            let log_items: Vec<ListItem> = st
                .log
                .iter()
                .map(|l| {
                    let color = if l.contains("fail") || l.contains("ERROR") {
                        Color::Red
                    } else if l.contains("wrote") {
                        Color::LightGreen
                    } else if l.contains("orphan") || l.contains("skip") {
                        Color::DarkGray
                    } else {
                        Color::Rgb(140, 190, 215)
                    };
                    ListItem::new(Line::from(Span::styled(l.as_str(), Style::default().fg(color))))
                })
                .collect();
            frame.render_widget(
                List::new(log_items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Rgb(60, 100, 120)))
                            .title(Span::styled(" EVENT LOG ", Style::default().fg(Color::Cyan))),
                    )
                    .style(bg),
                chunks[4],
            );

            // ── Footer ────────────────────────────────────────────────────────
            let footer = if st.should_cancel() {
                Line::from(Span::styled(
                    " ⚠ draining workers — waiting for current genes to finish ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(Span::styled(
                    " [q] signal stop after current gene ",
                    Style::default().fg(Color::DarkGray),
                ))
            };
            frame.render_widget(
                Paragraph::new(footer).wrap(Wrap { trim: true }),
                chunks[5],
            );
        })?;

        if done {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        crossterm::cursor::Show,
    )?;
    Ok(())
}
