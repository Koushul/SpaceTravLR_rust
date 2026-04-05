use std::collections::HashSet;
use std::io::stdout;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::widgets::{Block, Borders, LineGauge, List, ListItem, Paragraph, Wrap};

use crate::betadata::BetadataUiProgress;
use crate::config::expand_user_path;
use crate::perturb::{
    PerturbConfig, PerturbResult, PerturbTarget, PerturbTimings, perturb_with_targets,
};
use crate::perturb_mode::{
    GeneCellTypeScopes, PerturbRuntime, export_joint_perturb_result, single_perturb_target,
};

// Palette aligned with `training_tui` (spacetravlr dashboard).
const BG: Color = Color::Rgb(40, 40, 40);
const OUTER_BORD: Color = Color::Rgb(60, 56, 54);
const TEL_BORD: Color = Color::Rgb(69, 133, 136);
const WORK_BORD: Color = Color::Rgb(215, 153, 33);
const ROCKET_BORD: Color = Color::Rgb(131, 165, 152);
const LABEL: Color = Color::Rgb(215, 153, 33);
const VALUE: Color = Color::Rgb(142, 192, 124);
const LILAC: Color = Color::Rgb(184, 187, 38);
const SKY: Color = Color::Rgb(69, 133, 136);
const GRAPE: Color = Color::Rgb(211, 134, 155);
const MUTED: Color = Color::Rgb(146, 131, 116);
const TITLE: Color = Color::Rgb(235, 219, 178);
const C_WROTE: Color = Color::Rgb(142, 192, 124);
const C_FAIL: Color = Color::Rgb(204, 36, 29);
const GAUGE_EMPTY: Color = Color::Rgb(60, 56, 54);

const LOAD_GAUGE_LINES: symbols::line::Set = symbols::line::Set {
    horizontal: "█",
    ..symbols::line::THICK
};

fn block_panel<'a>(title: impl Into<Line<'a>>, border: Color) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .style(Style::default().bg(BG))
        .title(title)
}

fn styled_result_line(s: &str) -> Line<'static> {
    let style = if s.starts_with("Δ ") {
        Style::default().fg(SKY)
    } else if s.starts_with("Joint export") {
        Style::default().fg(C_WROTE)
    } else if s.starts_with("Export error") || s.starts_with("Perturbation failed") {
        Style::default().fg(C_FAIL)
    } else if s.starts_with("Per-step") {
        Style::default().fg(LABEL).add_modifier(Modifier::BOLD)
    } else if s.starts_with("  ") {
        Style::default().fg(MUTED)
    } else if s.contains("Enter / Esc") {
        Style::default().fg(MUTED)
    } else {
        Style::default().fg(TITLE)
    };
    Line::from(Span::styled(s.to_string(), style))
}

pub struct PerturbTuiOptions {
    pub run_toml: Option<PathBuf>,
    pub default_desired_expr: f64,
    pub n_propagation_initial: Option<usize>,
    pub verbose: bool,
    pub toml_path_hint_for_error: Option<String>,
}

enum Screen {
    PickToml {
        path_input: String,
        err: Option<String>,
    },
    Loading,
    Main,
    EditDesired {
        buf: String,
    },
    EditNPropagation {
        buf: String,
    },
    EditCellScope {
        gene: String,
        cell_types: Vec<usize>,
        picked: HashSet<usize>,
        list_state: ratatui::widgets::ListState,
    },
    Running,
    ResultView {
        lines: Vec<String>,
        scroll: usize,
    },
}

enum BgMsg {
    Loaded(Result<PerturbRuntime, String>),
    Perturbed(Result<PerturbOutcome, String>),
}

struct PerturbOutcome {
    result: PerturbResult,
    timings: Option<PerturbTimings>,
    elapsed: Duration,
    export_dir: Option<PathBuf>,
    export_err: Option<String>,
}

pub async fn run(opts: PerturbTuiOptions) -> anyhow::Result<()> {
    tokio::task::spawn_blocking(move || run_sync(opts))
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?
}

fn run_sync(opts: PerturbTuiOptions) -> anyhow::Result<()> {
    let (tx_bg, rx_bg) = mpsc::channel::<BgMsg>();
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    let pick = if opts.run_toml.is_none() {
        Screen::PickToml {
            path_input: String::new(),
            err: None,
        }
    } else {
        Screen::Loading
    };

    let load_progress_permille = Arc::new(AtomicU32::new(0));
    let load_progress_message = Arc::new(Mutex::new(String::from("Starting…")));
    let perturb_progress_permille = Arc::new(AtomicU32::new(0));
    let perturb_progress_message = Arc::new(Mutex::new(String::new()));

    let mut app = App {
        screen: pick,
        runtime: None,
        desired_expr: opts.default_desired_expr,
        n_propagation: 0,
        gene_filter: String::new(),
        list_state: ratatui::widgets::ListState::default(),
        filtered_cursor: 0,
        list_viewport_height: 8,
        perturb_targets: Vec::new(),
        target_cell_scopes: GeneCellTypeScopes::new(),
        status_line: String::new(),
        status_is_error: false,
        verbose: opts.verbose,
        pending_verbose: opts.verbose,
        spinner_frame: 0u8,
        bg_rx: rx_bg,
        load_applied_n_prop: opts.n_propagation_initial,
        toml_path_hint_for_error: opts.toml_path_hint_for_error.clone(),
        last_perturbed_targets: Vec::new(),
        filtered_gene_indices: Vec::new(),
        run_cancel: None,
        load_progress_permille: load_progress_permille.clone(),
        load_progress_message: load_progress_message.clone(),
        perturb_progress_permille: perturb_progress_permille.clone(),
        perturb_progress_message: perturb_progress_message.clone(),
    };

    if let Some(path) = opts.run_toml.clone() {
        let tx = tx_bg.clone();
        let p = load_progress_permille.clone();
        let m = load_progress_message.clone();
        std::thread::spawn(move || {
            let dummy_ui = Arc::new(BetadataUiProgress::new());
            let r = PerturbRuntime::from_run_toml_with_progress(
                path.as_path(),
                Some(p),
                Some(m),
                Some(dummy_ui),
            )
            .map_err(|e| e.to_string());
            let _ = tx.send(BgMsg::Loaded(r));
        });
    }

    let tick = Duration::from_millis(120);
    let mut last_tick = Instant::now();

    let res = loop {
        if last_tick.elapsed() >= tick {
            app.spinner_frame = app.spinner_frame.wrapping_add(1);
            last_tick = Instant::now();
        }

        while let Ok(msg) = app.bg_rx.try_recv() {
            match msg {
                BgMsg::Loaded(r) => match r {
                    Ok(mut rt) => {
                        if let Some(n) = app.load_applied_n_prop {
                            rt.perturb_cfg.n_propagation = n;
                        }
                        app.n_propagation = rt.perturb_cfg.n_propagation;
                        app.runtime = Some(std::sync::Arc::new(rt));
                        app.screen = Screen::Main;
                        app.rebuild_filter();
                        app.clear_status();
                    }
                    Err(e) => {
                        app.screen = Screen::PickToml {
                            path_input: app.toml_path_hint_for_error.clone().unwrap_or_default(),
                            err: Some(e),
                        };
                    }
                },
                BgMsg::Perturbed(r) => match r {
                    Ok(out) => {
                        let lines = app.format_outcome(&out);
                        app.screen = Screen::ResultView { lines, scroll: 0 };
                        app.verbose = app.pending_verbose;
                        app.run_cancel = None;
                    }
                    Err(e) => {
                        app.screen = Screen::ResultView {
                            lines: vec!["Perturbation failed:".into(), e],
                            scroll: 0,
                        };
                        app.run_cancel = None;
                    }
                },
            }
        }

        terminal.draw(|f| app.render(f))?;

        let poll_ms = tick.saturating_sub(last_tick.elapsed());
        let poll_ms = poll_ms.as_millis().min(100) as u64;
        if event::poll(Duration::from_millis(poll_ms.max(16)))? {
            let ev = event::read()?;
            match app.handle_event(ev, &tx_bg) {
                Ok(Some(())) => break Ok(()),
                Ok(None) => {}
                Err(e) => break Err(e),
            }
        }
    };

    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    res
}

struct App {
    screen: Screen,
    runtime: Option<std::sync::Arc<PerturbRuntime>>,
    desired_expr: f64,
    n_propagation: usize,
    gene_filter: String,
    list_state: ratatui::widgets::ListState,
    filtered_cursor: usize,
    list_viewport_height: u16,
    perturb_targets: Vec<String>,
    target_cell_scopes: GeneCellTypeScopes,
    status_line: String,
    status_is_error: bool,
    verbose: bool,
    pending_verbose: bool,
    spinner_frame: u8,
    bg_rx: mpsc::Receiver<BgMsg>,
    load_applied_n_prop: Option<usize>,
    filtered_gene_indices: Vec<usize>,
    toml_path_hint_for_error: Option<String>,
    last_perturbed_targets: Vec<String>,
    run_cancel: Option<Arc<AtomicBool>>,
    load_progress_permille: Arc<AtomicU32>,
    load_progress_message: Arc<Mutex<String>>,
    perturb_progress_permille: Arc<AtomicU32>,
    perturb_progress_message: Arc<Mutex<String>>,
}

impl App {
    fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_line = msg.into();
        self.status_is_error = is_error;
    }

    fn clear_status(&mut self) {
        self.status_line.clear();
        self.status_is_error = false;
    }

    fn rebuild_filter(&mut self) {
        let Some(rt) = self.runtime.as_ref() else {
            self.filtered_gene_indices.clear();
            return;
        };
        let q = self.gene_filter.to_ascii_lowercase();
        if q.is_empty() {
            self.filtered_gene_indices = (0..rt.gene_names.len()).collect();
        } else {
            self.filtered_gene_indices = rt
                .gene_names
                .iter()
                .enumerate()
                .filter(|(_, g)| g.to_ascii_lowercase().contains(&q))
                .map(|(i, _)| i)
                .take(50_000)
                .collect();
        }
        let n = self.filtered_gene_indices.len();
        if n == 0 {
            self.filtered_cursor = 0;
        } else {
            self.filtered_cursor = self.filtered_cursor.min(n - 1);
        }
    }

    fn list_offset_for_view(cursor: usize, view_len: usize, total: usize) -> usize {
        if total == 0 || view_len == 0 {
            return 0;
        }
        let c = cursor.min(total - 1);
        if total <= view_len {
            return 0;
        }
        let max_start = total - view_len;
        c.saturating_sub(view_len / 2).min(max_start)
    }

    fn selected_gene_name(&self) -> Option<String> {
        let rt = self.runtime.as_ref()?;
        let gi = *self.filtered_gene_indices.get(self.filtered_cursor)?;
        rt.gene_names.get(gi).cloned()
    }

    fn toggle_perturb_target(&mut self, gene: String) {
        if let Some(i) = self.perturb_targets.iter().position(|g| g == &gene) {
            self.perturb_targets.remove(i);
            self.target_cell_scopes.remove(&gene);
        } else {
            self.perturb_targets.push(gene.clone());
            self.target_cell_scopes.insert(gene, None);
        }
    }

    fn open_cell_scope_editor(&mut self, gene: String, rt: &PerturbRuntime) {
        let mut cell_types: Vec<usize> = rt.cell_types.iter().copied().collect();
        cell_types.sort_unstable();
        cell_types.dedup();
        let all_set: HashSet<usize> = cell_types.iter().copied().collect();
        let picked = match self.target_cell_scopes.get(&gene) {
            Some(Some(sub)) => sub.clone(),
            _ => all_set,
        };
        let mut list_state = ratatui::widgets::ListState::default();
        list_state.select(if cell_types.is_empty() { None } else { Some(0) });
        self.screen = Screen::EditCellScope {
            gene,
            cell_types,
            picked,
            list_state,
        };
    }

    fn render(&mut self, f: &mut Frame) {
        f.render_widget(Block::default().style(Style::default().bg(BG)), f.area());

        let outer_title = Line::from(vec![
            Span::styled("✿ ", Style::default().fg(GRAPE)),
            Span::styled(
                "spacetravlr-perturb",
                Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);
        let block = block_panel(outer_title, TEL_BORD);
        let inner = block.inner(f.area());
        f.render_widget(block, f.area());

        match &mut self.screen {
            Screen::PickToml { path_input, err } => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(3), Constraint::Length(3)])
                    .split(inner);
                let mut txt = vec![
                    Line::from(vec![
                        Span::styled("Path to ", Style::default().fg(TITLE)),
                        Span::styled(
                            "spacetravlr_run_repro.toml",
                            Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  Enter load  ·  Esc quit", Style::default().fg(MUTED)),
                    ]),
                    Line::from(""),
                ];
                if let Some(e) = err {
                    txt.push(Line::from(vec![Span::styled(
                        format!("Error: {e}"),
                        Style::default().fg(C_FAIL).add_modifier(Modifier::BOLD),
                    )]));
                }
                f.render_widget(
                    Paragraph::new(txt)
                        .style(Style::default().bg(BG))
                        .wrap(Wrap { trim: true }),
                    chunks[0],
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        path_input.as_str(),
                        Style::default().fg(TITLE),
                    ))
                    .block(block_panel(
                        Line::from(Span::styled(" Path ", Style::default().fg(LABEL))),
                        SKY,
                    )),
                    chunks[1],
                );
            }
            Screen::Loading => {
                let permille = self.load_progress_permille.load(Ordering::Relaxed);
                let ratio = (permille as f64 / 1000.0).clamp(0.0, 1.0);
                let status = self
                    .load_progress_message
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_default();
                let spin = ["|", "/", "-", "\\"][self.spinner_frame as usize % 4];
                let pct = permille / 10;
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),
                        Constraint::Length(1),
                        Constraint::Min(2),
                        Constraint::Length(1),
                    ])
                    .split(inner);

                let title = Line::from(vec![
                    Span::styled("✿ ", Style::default().fg(GRAPE)),
                    Span::styled(
                        "Loading PerturbRuntime",
                        Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ·  {pct}%"),
                        Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                    ),
                ]);
                f.render_widget(
                    Paragraph::new(title)
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(BG)),
                    chunks[0],
                );

                let gauge = LineGauge::default()
                    .style(Style::default().bg(BG))
                    .filled_style(Style::default().fg(SKY).add_modifier(Modifier::BOLD))
                    .unfilled_style(Style::default().fg(GAUGE_EMPTY))
                    .line_set(LOAD_GAUGE_LINES)
                    .label(Line::from(""))
                    .ratio(ratio);
                f.render_widget(gauge, chunks[1]);

                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            format!("{spin}  "),
                            Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(status, Style::default().fg(VALUE)),
                    ]))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true })
                    .style(Style::default().bg(BG)),
                    chunks[2],
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        "Esc — cancel load",
                        Style::default().fg(MUTED),
                    ))
                    .alignment(Alignment::Center)
                    .style(Style::default().bg(BG)),
                    chunks[3],
                );
            }
            Screen::Main => self.render_main(f, inner),
            Screen::EditDesired { buf } => {
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from(vec![
                            Span::styled("Desired ", Style::default().fg(TITLE)),
                            Span::styled("desired_expr", Style::default().fg(VALUE)),
                            Span::styled(
                                "  ·  Enter OK  ·  Esc cancel",
                                Style::default().fg(MUTED),
                            ),
                        ]),
                        Line::from(""),
                        Line::from(Span::styled(
                            buf.as_str(),
                            Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                        )),
                    ])
                    .block(block_panel(
                        Line::from(Span::styled(" desired_expr ", Style::default().fg(LABEL))),
                        WORK_BORD,
                    )),
                    inner,
                );
            }
            Screen::EditNPropagation { buf } => {
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from(vec![
                            Span::styled("Integer ", Style::default().fg(TITLE)),
                            Span::styled("n_propagation", Style::default().fg(VALUE)),
                            Span::styled(
                                "  ·  Enter OK  ·  Esc cancel",
                                Style::default().fg(MUTED),
                            ),
                        ]),
                        Line::from(""),
                        Line::from(Span::styled(
                            buf.as_str(),
                            Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
                        )),
                    ])
                    .block(block_panel(
                        Line::from(Span::styled(" n_propagation ", Style::default().fg(LABEL))),
                        ROCKET_BORD,
                    )),
                    inner,
                );
            }
            Screen::EditCellScope {
                gene,
                cell_types,
                picked,
                list_state,
            } => {
                let items: Vec<ListItem> = cell_types
                    .iter()
                    .map(|ct| {
                        let (m, mc) = if picked.contains(ct) {
                            ("[•]", GRAPE)
                        } else {
                            ("[ ]", MUTED)
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled(format!("{m} "), Style::default().fg(mc)),
                            Span::styled("cell_type_int ", Style::default().fg(MUTED)),
                            Span::styled(format!("{ct}"), Style::default().fg(TITLE)),
                        ]))
                    })
                    .collect();
                let hint = vec![
                    Line::from(vec![
                        Span::styled("Cell scope  ", Style::default().fg(LABEL)),
                        Span::styled("·  ", Style::default().fg(MUTED)),
                        Span::styled(
                            gene.as_str(),
                            Style::default().fg(GRAPE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "  ·  Space toggle  ·  Enter save  ·  Esc cancel",
                            Style::default().fg(MUTED),
                        ),
                    ]),
                    Line::from(vec![Span::styled(
                        "All types on = whole tissue. Subset = only those cell types get the perturbation.",
                        Style::default().fg(MUTED),
                    )]),
                    Line::from(""),
                ];
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(4), Constraint::Min(3)])
                    .split(inner);
                f.render_widget(
                    Paragraph::new(hint).style(Style::default().bg(BG)),
                    chunks[0],
                );
                let list = List::new(items)
                    .block(block_panel(
                        Line::from(Span::styled(" cell types ", Style::default().fg(LABEL))),
                        ROCKET_BORD,
                    ))
                    .highlight_style(Style::default().bg(SKY).fg(BG).add_modifier(Modifier::BOLD));
                f.render_stateful_widget(list, chunks[1], list_state);
            }
            Screen::Running => {
                let permille = self.perturb_progress_permille.load(Ordering::Relaxed);
                let ratio = (permille as f64 / 1000.0).clamp(0.0, 1.0);
                let status = self
                    .perturb_progress_message
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_default();
                let spin = ["|", "/", "-", "\\"][self.spinner_frame as usize % 4];
                let pct = permille / 10;
                let tgt_line = if self.last_perturbed_targets.is_empty() {
                    Line::from(vec![
                        Span::styled("Targets ", Style::default().fg(LABEL)),
                        Span::styled("—", Style::default().fg(MUTED)),
                    ])
                } else {
                    let s = self.last_perturbed_targets.join(", ");
                    let summary = if s.len() > 72 {
                        format!("{}…", &s[..69])
                    } else {
                        s
                    };
                    Line::from(vec![
                        Span::styled("Targets ", Style::default().fg(LABEL)),
                        Span::styled("· ", Style::default().fg(MUTED)),
                        Span::styled(summary, Style::default().fg(LILAC)),
                    ])
                };
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),
                        Constraint::Length(1),
                        Constraint::Length(2),
                        Constraint::Min(2),
                        Constraint::Length(1),
                    ])
                    .split(inner);

                let title = Line::from(vec![
                    Span::styled("◆ ", Style::default().fg(WORK_BORD)),
                    Span::styled(
                        "GRN perturbation",
                        Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ·  {pct}%"),
                        Style::default().fg(WORK_BORD).add_modifier(Modifier::BOLD),
                    ),
                ]);
                f.render_widget(
                    Paragraph::new(title)
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(BG)),
                    chunks[0],
                );

                let gauge = LineGauge::default()
                    .style(Style::default().bg(BG))
                    .filled_style(Style::default().fg(WORK_BORD).add_modifier(Modifier::BOLD))
                    .unfilled_style(Style::default().fg(GAUGE_EMPTY))
                    .line_set(LOAD_GAUGE_LINES)
                    .label(Line::from(""))
                    .ratio(ratio);
                f.render_widget(gauge, chunks[1]);

                f.render_widget(
                    Paragraph::new(tgt_line)
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(BG))
                        .wrap(Wrap { trim: true }),
                    chunks[2],
                );

                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            format!("{spin}  "),
                            Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(status, Style::default().fg(VALUE)),
                    ]))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true })
                    .style(Style::default().bg(BG)),
                    chunks[3],
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        "Esc or Ctrl+C — cancel",
                        Style::default().fg(MUTED),
                    ))
                    .alignment(Alignment::Center)
                    .style(Style::default().bg(BG)),
                    chunks[4],
                );
            }
            Screen::ResultView { lines, scroll } => {
                let view_h = inner.height.saturating_sub(2) as usize;
                let max_scroll = lines.len().saturating_sub(view_h.max(1));
                let s = (*scroll).min(max_scroll);
                *scroll = s;
                let end = (s + view_h.max(1)).min(lines.len());
                let slice: Vec<Line> = lines[s..end]
                    .iter()
                    .map(|x| styled_result_line(x))
                    .collect();
                let title_line = if lines.len() > view_h.max(1) {
                    Line::from(vec![
                        Span::styled(
                            " Result ",
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("·", Style::default().fg(MUTED)),
                        Span::styled(" scroll ", Style::default().fg(SKY)),
                        Span::styled(
                            format!("({}/{}) ", s + 1, lines.len()),
                            Style::default().fg(LILAC),
                        ),
                        Span::styled("· Enter/Esc back ", Style::default().fg(MUTED)),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(
                            " Result ",
                            Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("· Enter/Esc back ", Style::default().fg(MUTED)),
                    ])
                };
                f.render_widget(
                    Paragraph::new(slice)
                        .style(Style::default().bg(BG))
                        .block(block_panel(title_line, TEL_BORD))
                        .wrap(Wrap { trim: false }),
                    inner,
                );
            }
        }
    }

    fn render_main(&mut self, f: &mut Frame, area: Rect) {
        let rt = self.runtime.as_ref().unwrap();
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(area);
        let body = main_chunks[0];
        let status_area = main_chunks[1];

        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(body);

        self.list_viewport_height = chunks[0].height.saturating_sub(2).max(1);

        let n = self.filtered_gene_indices.len();
        let h = self.list_viewport_height as usize;
        let offset = Self::list_offset_for_view(self.filtered_cursor, h, n);
        let end = (offset + h).min(n);
        let list_items: Vec<ListItem> = if offset < end {
            self.filtered_gene_indices[offset..end]
                .iter()
                .filter_map(|&gi| {
                    rt.gene_names.get(gi).map(|g| {
                        let (mark, mc) = if self.perturb_targets.iter().any(|t| t == g) {
                            ("[•] ", GRAPE)
                        } else {
                            ("[ ] ", MUTED)
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled(mark, Style::default().fg(mc)),
                            Span::styled(g.as_str(), Style::default().fg(TITLE)),
                        ]))
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        let local_sel = self.filtered_cursor.saturating_sub(offset);
        self.list_state.select(if list_items.is_empty() {
            None
        } else {
            Some(local_sel.min(list_items.len().saturating_sub(1)))
        });

        let count_note = if n > 0 {
            format!(" {} / {} ", (self.filtered_cursor + 1).min(n), n)
        } else {
            " (empty)".into()
        };

        let list = List::new(list_items)
            .block(block_panel(
                Line::from(vec![
                    Span::styled(
                        " Genes ",
                        Style::default().fg(TITLE).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(count_note, Style::default().fg(SKY)),
                    Span::styled(
                        " · PgUp/PgDn Home/End · Ctrl+J/K · type · Space ",
                        Style::default().fg(MUTED),
                    ),
                ]),
                ROCKET_BORD,
            ))
            .highlight_style(Style::default().bg(SKY).fg(BG).add_modifier(Modifier::BOLD));

        f.render_stateful_widget(list, chunks[0], &mut self.list_state);

        let scope_hint = self
            .selected_gene_name()
            .and_then(|g| self.target_cell_scopes.get(&g).map(|sc| (g, sc)))
            .and_then(|(g, sc)| match sc {
                None => None,
                Some(sub) => {
                    let mut v: Vec<_> = sub.iter().copied().collect();
                    v.sort_unstable();
                    Some(format!("{} → types {:?}", g, v))
                }
            })
            .unwrap_or_default();

        let hdr = vec![
            Line::from(vec![
                Span::styled("Run ", Style::default().fg(LABEL)),
                Span::styled("· ", Style::default().fg(MUTED)),
                Span::styled(
                    rt.run_toml_path.display().to_string(),
                    Style::default().fg(VALUE),
                ),
            ]),
            Line::from(vec![
                Span::styled("Cells × genes ", Style::default().fg(LABEL)),
                Span::styled("· ", Style::default().fg(MUTED)),
                Span::styled(
                    format!("{} × {}", rt.obs_names.len(), rt.gene_names.len()),
                    Style::default().fg(SKY),
                ),
            ]),
            Line::from(vec![
                Span::styled("Targets ", Style::default().fg(LABEL)),
                Span::styled("· ", Style::default().fg(MUTED)),
                Span::styled(
                    if self.perturb_targets.is_empty() {
                        "— (none)".to_string()
                    } else {
                        self.perturb_targets.join(", ")
                    },
                    Style::default().fg(if self.perturb_targets.is_empty() {
                        MUTED
                    } else {
                        LILAC
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled("Row scope ", Style::default().fg(LABEL)),
                Span::styled("· ", Style::default().fg(MUTED)),
                Span::styled(
                    if scope_hint.is_empty() {
                        "—".into()
                    } else {
                        scope_hint
                    },
                    Style::default().fg(GRAPE),
                ),
            ]),
            Line::from(vec![
                Span::styled("desired_expr ", Style::default().fg(LABEL)),
                Span::styled(
                    format!("{}", self.desired_expr),
                    Style::default().fg(VALUE).add_modifier(Modifier::BOLD),
                ),
                Span::styled("   n_prop ", Style::default().fg(LABEL)),
                Span::styled(
                    format!("{}", self.n_propagation),
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Verbose ", Style::default().fg(LABEL)),
                Span::styled(
                    if self.pending_verbose { "on" } else { "off" },
                    Style::default()
                        .fg(if self.pending_verbose { C_WROTE } else { MUTED })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" (Ctrl+V)", Style::default().fg(MUTED)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "Ctrl+R",
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" run  ", Style::default().fg(MUTED)),
                Span::styled(
                    "Ctrl+T",
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" scope  ", Style::default().fg(MUTED)),
                Span::styled(
                    "Ctrl+E",
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" d.expr  ", Style::default().fg(MUTED)),
                Span::styled(
                    "Ctrl+P",
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" n  ", Style::default().fg(MUTED)),
                Span::styled("Esc", Style::default().fg(SKY).add_modifier(Modifier::BOLD)),
                Span::styled(" clr  ", Style::default().fg(MUTED)),
                Span::styled(
                    "Ctrl+Q",
                    Style::default().fg(SKY).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" quit", Style::default().fg(MUTED)),
            ]),
        ];

        let filter_title = Line::from(vec![
            Span::styled(" Filter ", Style::default().fg(LABEL)),
            Span::styled("· type to narrow", Style::default().fg(MUTED)),
        ]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(hdr.len() as u16 + 1), Constraint::Min(3)])
            .split(chunks[1]);

        f.render_widget(Paragraph::new(hdr).style(Style::default().bg(BG)), right[0]);
        f.render_widget(
            Paragraph::new(Span::styled(
                self.gene_filter.as_str(),
                Style::default().fg(LILAC).add_modifier(Modifier::BOLD),
            ))
            .block(block_panel(filter_title, OUTER_BORD)),
            right[1],
        );

        let st_style = if self.status_is_error {
            Style::default().fg(C_FAIL).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(C_WROTE)
        };
        let status_txt = if self.status_line.is_empty() {
            Line::from(vec![
                Span::styled("✿ ", Style::default().fg(GRAPE)),
                Span::styled("Joint export ", Style::default().fg(LABEL)),
                Span::styled("· auto after successful run · ", Style::default().fg(MUTED)),
                Span::styled("see result panel", Style::default().fg(SKY)),
            ])
        } else {
            Line::from(vec![Span::styled(self.status_line.as_str(), st_style)])
        };
        f.render_widget(
            Paragraph::new(status_txt).style(Style::default().bg(BG)),
            status_area,
        );
    }

    fn handle_event(
        &mut self,
        ev: Event,
        tx_bg: &mpsc::Sender<BgMsg>,
    ) -> anyhow::Result<Option<()>> {
        if let Event::Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return Ok(None);
            }
            match &mut self.screen {
                Screen::PickToml { path_input, err } => {
                    match key.code {
                        KeyCode::Esc => return Ok(Some(())),
                        KeyCode::Enter => {
                            let expanded = expand_user_path(path_input.trim());
                            let p = PathBuf::from(expanded);
                            if !p.is_file() {
                                *err = Some(format!("not a file: {}", p.display()));
                                return Ok(None);
                            }
                            *err = None;
                            self.load_progress_permille.store(0, Ordering::Relaxed);
                            if let Ok(mut g) = self.load_progress_message.lock() {
                                *g = "Starting…".to_string();
                            }
                            self.screen = Screen::Loading;
                            let tx = tx_bg.clone();
                            let prog_p = self.load_progress_permille.clone();
                            let prog_m = self.load_progress_message.clone();
                            std::thread::spawn(move || {
                                let dummy_ui = Arc::new(BetadataUiProgress::new());
                                let r = PerturbRuntime::from_run_toml_with_progress(
                                    p.as_path(),
                                    Some(prog_p),
                                    Some(prog_m),
                                    Some(dummy_ui),
                                )
                                .map_err(|e| e.to_string());
                                let _ = tx.send(BgMsg::Loaded(r));
                            });
                        }
                        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                            path_input.push(c);
                            err.take();
                        }
                        KeyCode::Backspace => {
                            path_input.pop();
                            err.take();
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Screen::Loading => {
                    if key.code == KeyCode::Esc {
                        return Ok(Some(()));
                    }
                    return Ok(None);
                }
                Screen::EditDesired { buf } => match key.code {
                    KeyCode::Esc => self.screen = Screen::Main,
                    KeyCode::Enter => {
                        if let Ok(v) = buf.parse::<f64>() {
                            if v.is_finite() {
                                self.desired_expr = v;
                            }
                        }
                        self.screen = Screen::Main;
                    }
                    KeyCode::Char(c) => buf.push(c),
                    KeyCode::Backspace => {
                        buf.pop();
                    }
                    _ => {}
                },
                Screen::EditNPropagation { buf } => match key.code {
                    KeyCode::Esc => self.screen = Screen::Main,
                    KeyCode::Enter => {
                        if let Ok(v) = buf.parse::<usize>() {
                            if v > 0 {
                                self.n_propagation = v;
                            }
                        }
                        self.screen = Screen::Main;
                    }
                    KeyCode::Char(c) => buf.push(c),
                    KeyCode::Backspace => {
                        buf.pop();
                    }
                    _ => {}
                },
                Screen::EditCellScope {
                    gene,
                    cell_types,
                    picked,
                    list_state,
                } => {
                    match key.code {
                        KeyCode::Esc => self.screen = Screen::Main,
                        KeyCode::Enter => {
                            let all_set: HashSet<usize> = cell_types.iter().copied().collect();
                            if picked.is_empty() {
                                self.set_status("Select at least one cell type", true);
                                return Ok(None);
                            }
                            if *picked == all_set {
                                self.target_cell_scopes.insert(gene.clone(), None);
                            } else {
                                self.target_cell_scopes
                                    .insert(gene.clone(), Some(picked.clone()));
                            }
                            self.screen = Screen::Main;
                            self.clear_status();
                        }
                        KeyCode::Up => {
                            let i = list_state.selected().unwrap_or(0);
                            list_state.select(Some(i.saturating_sub(1)));
                        }
                        KeyCode::Down => {
                            let n = cell_types.len();
                            if n == 0 {
                                return Ok(None);
                            }
                            let i = list_state.selected().unwrap_or(0);
                            list_state.select(Some((i + 1).min(n - 1)));
                        }
                        KeyCode::Char(' ') => {
                            if let Some(i) = list_state.selected() {
                                if let Some(&ct) = cell_types.get(i) {
                                    if picked.contains(&ct) {
                                        picked.remove(&ct);
                                    } else {
                                        picked.insert(ct);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Screen::Running => {
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('c')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            if let Some(c) = self.run_cancel.as_ref() {
                                c.store(true, Ordering::Relaxed);
                            }
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Screen::ResultView { lines, scroll } => {
                    let view_h = 10usize;
                    match key.code {
                        KeyCode::Esc | KeyCode::Enter => {
                            self.screen = Screen::Main;
                        }
                        KeyCode::Up => {
                            *scroll = scroll.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            let max = lines.len().saturating_sub(1);
                            *scroll = (*scroll + 1).min(max);
                        }
                        KeyCode::PageUp => {
                            *scroll = scroll.saturating_sub(view_h);
                        }
                        KeyCode::PageDown => {
                            let max = lines.len().saturating_sub(1);
                            *scroll = (*scroll + view_h).min(max);
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Screen::Main => {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('v')
                    {
                        self.pending_verbose = !self.pending_verbose;
                        return Ok(None);
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(Some(())),
                            KeyCode::Char('e') | KeyCode::Char('E') => {
                                self.screen = Screen::EditDesired {
                                    buf: format!("{}", self.desired_expr),
                                };
                            }
                            KeyCode::Char('p') | KeyCode::Char('P') => {
                                self.screen = Screen::EditNPropagation {
                                    buf: format!("{}", self.n_propagation),
                                };
                            }
                            KeyCode::Char('t') | KeyCode::Char('T') => {
                                let Some(gene) = self.selected_gene_name() else {
                                    self.set_status("No row selected", true);
                                    return Ok(None);
                                };
                                if !self.perturb_targets.iter().any(|g| g == &gene) {
                                    self.set_status(
                                        "Toggle gene as target (Space) before cell scope",
                                        true,
                                    );
                                    return Ok(None);
                                }
                                let rt = self.runtime.as_ref().unwrap().clone();
                                self.open_cell_scope_editor(gene, rt.as_ref());
                                self.clear_status();
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                if self.perturb_targets.is_empty() {
                                    self.set_status(
                                        "No targets — Space toggles genes in the list",
                                        true,
                                    );
                                    return Ok(None);
                                }
                                let rt = self.runtime.as_ref().unwrap();
                                let mut targets: Vec<PerturbTarget> =
                                    Vec::with_capacity(self.perturb_targets.len());
                                for gene in &self.perturb_targets {
                                    let Ok(base) = single_perturb_target(
                                        gene.as_str(),
                                        self.desired_expr,
                                        &rt.gene_names,
                                    ) else {
                                        self.set_status(format!("Invalid gene: {gene}"), true);
                                        return Ok(None);
                                    };
                                    let cell_indices = self
                                        .target_cell_scopes
                                        .get(gene)
                                        .and_then(|scope| scope.as_ref())
                                        .map(|cell_types| {
                                            rt.cell_types
                                                .iter()
                                                .enumerate()
                                                .filter_map(|(idx, ct)| {
                                                    if cell_types.contains(ct) {
                                                        Some(idx)
                                                    } else {
                                                        None
                                                    }
                                                })
                                                .collect::<Vec<_>>()
                                        });
                                    targets.push(PerturbTarget {
                                        gene: base.gene,
                                        desired_expr: base.desired_expr,
                                        cell_indices,
                                    });
                                }
                                let rt = rt.clone();
                                let mut cfg: PerturbConfig = rt.perturb_cfg.clone();
                                cfg.n_propagation = self.n_propagation;
                                let capture_timings = self.pending_verbose;
                                let scopes = self.target_cell_scopes.clone();
                                let genes = self.perturb_targets.clone();
                                let desired = self.desired_expr;
                                let n_prop = self.n_propagation;
                                self.last_perturbed_targets
                                    .clone_from(&self.perturb_targets);
                                self.perturb_progress_permille.store(0, Ordering::Relaxed);
                                if let Ok(mut g) = self.perturb_progress_message.lock() {
                                    *g = "GRN perturbation · starting…".to_string();
                                }
                                self.screen = Screen::Running;
                                let cancel = Arc::new(AtomicBool::new(false));
                                self.run_cancel = Some(cancel.clone());
                                let job_p = self.perturb_progress_permille.clone();
                                let job_m = self.perturb_progress_message.clone();
                                let tx = tx_bg.clone();
                                std::thread::spawn(move || {
                                    let t0 = Instant::now();
                                    let mut timings = if capture_timings {
                                        Some(PerturbTimings::default())
                                    } else {
                                        None
                                    };
                                    let res = perturb_with_targets(
                                        &rt.bb,
                                        &rt.gene_mtx,
                                        &rt.gene_names,
                                        &rt.xy,
                                        &rt.rw_ligands_init,
                                        &rt.rw_tfligands_init,
                                        &targets,
                                        &cfg,
                                        &rt.lr_radii,
                                        Some(&job_p),
                                        Some(&job_m),
                                        Some(cancel.as_ref()),
                                        Some(&rt.baseline_splash_cache),
                                        &mut timings,
                                    );
                                    let elapsed = t0.elapsed();
                                    let msg = match res {
                                        Ok(result) => {
                                            let (export_dir, export_err) =
                                                match export_joint_perturb_result(
                                                    &rt,
                                                    &result.simulated,
                                                    &genes,
                                                    desired,
                                                    n_prop,
                                                    &scopes,
                                                ) {
                                                    Ok(p) => (Some(p), None),
                                                    Err(e) => (None, Some(e.to_string())),
                                                };
                                            BgMsg::Perturbed(Ok(PerturbOutcome {
                                                result,
                                                timings,
                                                elapsed,
                                                export_dir,
                                                export_err,
                                            }))
                                        }
                                        Err(()) => {
                                            let msg = if cancel.load(Ordering::Relaxed) {
                                                "Perturbation canceled".into()
                                            } else {
                                                "perturb_with_targets failed".into()
                                            };
                                            BgMsg::Perturbed(Err(msg))
                                        }
                                    };
                                    let _ = tx.send(msg);
                                });
                            }
                            _ => {}
                        }
                        return Ok(None);
                    }

                    let page = self.list_viewport_height.max(1) as usize;
                    match key.code {
                        KeyCode::Esc => {
                            self.gene_filter.clear();
                            self.rebuild_filter();
                            self.clear_status();
                        }
                        KeyCode::Down => self.select_next(),
                        KeyCode::Up => self.select_prev(),
                        KeyCode::PageDown => {
                            let n = self.filtered_gene_indices.len();
                            if n > 0 {
                                self.filtered_cursor = (self.filtered_cursor + page).min(n - 1);
                            }
                        }
                        KeyCode::PageUp => {
                            self.filtered_cursor = self.filtered_cursor.saturating_sub(page);
                        }
                        KeyCode::Home => {
                            if !self.filtered_gene_indices.is_empty() {
                                self.filtered_cursor = 0;
                            }
                        }
                        KeyCode::End => {
                            let n = self.filtered_gene_indices.len();
                            if n > 0 {
                                self.filtered_cursor = n - 1;
                            }
                        }
                        KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.select_next();
                        }
                        KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.select_prev();
                        }
                        KeyCode::Char(' ') | KeyCode::Enter => {
                            if let Some(g) = self.selected_gene_name() {
                                self.toggle_perturb_target(g);
                                self.clear_status();
                            }
                        }
                        KeyCode::Backspace => {
                            self.gene_filter.pop();
                            self.rebuild_filter();
                        }
                        KeyCode::Char(c) => {
                            self.gene_filter.push(c);
                            self.rebuild_filter();
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(None)
    }

    fn select_next(&mut self) {
        let n = self.filtered_gene_indices.len();
        if n == 0 {
            return;
        }
        self.filtered_cursor = (self.filtered_cursor + 1).min(n - 1);
    }

    fn select_prev(&mut self) {
        let n = self.filtered_gene_indices.len();
        if n == 0 {
            return;
        }
        self.filtered_cursor = self.filtered_cursor.saturating_sub(1);
    }

    fn format_outcome(&self, out: &PerturbOutcome) -> Vec<String> {
        let rt = self.runtime.as_ref().unwrap();
        let genes_label = if self.last_perturbed_targets.is_empty() {
            "?".to_string()
        } else {
            self.last_perturbed_targets.join(", ")
        };
        let mut lines = vec![
            format!("Genes: {genes_label}"),
            format!("desired_expr: {}", self.desired_expr),
            format!("n_propagation: {}", self.n_propagation),
            format!("Wall time: {:?}", out.elapsed),
        ];

        if let Some(ref path) = out.export_dir {
            lines.push(format!(
                "Joint export (full simulated matrix): {}",
                path.display()
            ));
        }
        if let Some(ref e) = out.export_err {
            lines.push(format!("Export error: {e}"));
        }

        for gene in &self.last_perturbed_targets {
            if let Some(j) = rt.gene_names.iter().position(|g| g == gene) {
                let col = out.result.delta.column(j);
                let mut min: f64 = f64::INFINITY;
                let mut max: f64 = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut n = 0usize;
                for &v in col.iter() {
                    if v.is_finite() {
                        min = min.min(v);
                        max = max.max(v);
                        sum += v;
                        n += 1;
                    }
                }
                let mean = if n > 0 { sum / n as f64 } else { 0.0 };
                lines.push(format!(
                    "Δ {gene}: min={min:.6} max={max:.6} mean={mean:.6} (n={n})"
                ));
            }
        }

        if self.pending_verbose {
            if let Some(t) = out.timings.as_ref() {
                lines.push("Per-step timings:".into());
                for (label, d) in &t.entries {
                    lines.push(format!("  {label}: {d:?}"));
                }
            }
        }

        lines.push("".into());
        lines.push("Enter / Esc: back · scroll with arrows / PgUp / PgDn".into());
        lines
    }
}
