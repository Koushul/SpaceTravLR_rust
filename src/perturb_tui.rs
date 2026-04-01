use std::io::stdout;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use crate::config::expand_user_path;
use crate::perturb::{PerturbConfig, PerturbResult, PerturbTimings, perturb_with_targets};
use crate::perturb_mode::{PerturbRuntime, single_perturb_target};

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
    Loading {
        label: String,
    },
    Main,
    EditDesired {
        buf: String,
    },
    EditNPropagation {
        buf: String,
    },
    Running,
    ResultView {
        lines: Vec<String>,
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
        Screen::Loading {
            label: "Loading PerturbRuntime…".into(),
        }
    };

    let mut app = App {
        screen: pick,
        runtime: None,
        desired_expr: opts.default_desired_expr,
        n_propagation: 0,
        gene_filter: String::new(),
        filter_focus: false,
        list_state: {
            let mut s = ratatui::widgets::ListState::default();
            s.select(Some(0));
            s
        },
        selected_gene: None,
        verbose: opts.verbose,
        pending_verbose: opts.verbose,
        spinner_frame: 0u8,
        bg_rx: rx_bg,
        load_applied_n_prop: opts.n_propagation_initial,
        toml_path_hint_for_error: opts.toml_path_hint_for_error.clone(),
        last_perturbed_gene: None,
        filtered_gene_indices: Vec::new(),
    };

    if let Some(path) = opts.run_toml.clone() {
        let tx = tx_bg.clone();
        std::thread::spawn(move || {
            let r = PerturbRuntime::from_run_toml(path.as_path()).map_err(|e| e.to_string());
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
                    }
                    Err(e) => {
                        app.screen = Screen::PickToml {
                            path_input: app
                                .toml_path_hint_for_error
                                .clone()
                                .unwrap_or_default(),
                            err: Some(e),
                        };
                    }
                },
                BgMsg::Perturbed(r) => match r {
                    Ok(out) => {
                        let lines = app.format_outcome(&out);
                        app.screen = Screen::ResultView { lines };
                        app.verbose = app.pending_verbose;
                    }
                    Err(e) => {
                        app.screen = Screen::ResultView {
                            lines: vec!["Perturbation failed:".into(), e],
                        };
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
    filter_focus: bool,
    list_state: ratatui::widgets::ListState,
    selected_gene: Option<String>,
    verbose: bool,
    pending_verbose: bool,
    spinner_frame: u8,
    bg_rx: mpsc::Receiver<BgMsg>,
    load_applied_n_prop: Option<usize>,
    filtered_gene_indices: Vec<usize>,
    toml_path_hint_for_error: Option<String>,
    last_perturbed_gene: Option<String>,
}

impl App {
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
        if self.filtered_gene_indices.is_empty() {
            self.list_state.select(None);
        } else {
            let sel = self.list_state.selected().unwrap_or(0).min(self.filtered_gene_indices.len() - 1);
            self.list_state.select(Some(sel));
        }
    }

    fn selected_gene_name(&self) -> Option<String> {
        let rt = self.runtime.as_ref()?;
        let i = self.list_state.selected()?;
        let gi = *self.filtered_gene_indices.get(i)?;
        rt.gene_names.get(gi).cloned()
    }

    fn render(&mut self, f: &mut Frame) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" spacetravlr-perturb ");
        let inner = block.inner(f.area());
        f.render_widget(block, f.area());

        match &self.screen {
            Screen::PickToml { path_input, err } => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(3), Constraint::Length(3)])
                    .split(inner);
                let mut txt = vec![
                    Line::from("Path to spacetravlr_run_repro.toml (Enter to load, Esc to quit)"),
                    Line::from(""),
                ];
                if let Some(e) = err {
                    txt.push(Line::from(vec![Span::styled(
                        format!("Error: {e}"),
                        Style::default().fg(Color::Red),
                    )]));
                }
                f.render_widget(Paragraph::new(txt).wrap(Wrap { trim: true }), chunks[0]);
                f.render_widget(
                    Paragraph::new(path_input.as_str()).block(Block::default().borders(Borders::ALL).title(" Path ")),
                    chunks[1],
                );
            }
            Screen::Loading { label } => {
                let spin = ["|", "/", "-", "\\"][self.spinner_frame as usize % 4];
                f.render_widget(
                    Paragraph::new(format!("{spin} {label}")).alignment(Alignment::Center),
                    inner,
                );
            }
            Screen::Main => self.render_main(f, inner),
            Screen::EditDesired { buf } => {
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from("Desired expression for target gene (Enter = OK, Esc = cancel)"),
                        Line::from(""),
                        Line::from(buf.as_str()),
                    ])
                    .block(Block::default().borders(Borders::ALL).title(" desired_expr ")),
                    inner,
                );
            }
            Screen::EditNPropagation { buf } => {
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from("n_propagation (Enter = OK, Esc = cancel)"),
                        Line::from(""),
                        Line::from(buf.as_str()),
                    ])
                    .block(Block::default().borders(Borders::ALL).title(" n_propagation ")),
                    inner,
                );
            }
            Screen::Running => {
                let spin = ["|", "/", "-", "\\"][self.spinner_frame as usize % 4];
                f.render_widget(
                    Paragraph::new(format!("{spin} Running perturbation…")).alignment(Alignment::Center),
                    inner,
                );
            }
            Screen::ResultView { lines } => {
                let text: Vec<Line> = lines
                    .iter()
                    .map(|s| Line::from(s.as_str()))
                    .collect();
                f.render_widget(
                    Paragraph::new(text)
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(" Result — any key: back "),
                        )
                        .wrap(Wrap { trim: false }),
                    inner,
                );
            }
        }
    }

    fn render_main(&mut self, f: &mut Frame, area: Rect) {
        let rt = self.runtime.as_ref().unwrap();
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(area);

        let list_items: Vec<ListItem> = self
            .filtered_gene_indices
            .iter()
            .filter_map(|&gi| rt.gene_names.get(gi).map(|g| ListItem::new(g.as_str())))
            .collect();

        let list = List::new(list_items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Genes (↑/↓ j/k / filter) "),
            )
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        f.render_stateful_widget(list, chunks[0], &mut self.list_state);

        let hdr = vec![
            Line::from(vec![
                Span::styled("Run: ", Style::default().fg(Color::Yellow)),
                Span::raw(rt.run_toml_path.display().to_string()),
            ]),
            Line::from(vec![
                Span::styled("Cells × genes: ", Style::default().fg(Color::Yellow)),
                Span::raw(format!("{} × {}", rt.obs_names.len(), rt.gene_names.len())),
            ]),
            Line::from(vec![
                Span::styled("Highlight: ", Style::default().fg(Color::Yellow)),
                Span::raw(
                    self.selected_gene_name()
                        .or_else(|| self.selected_gene.clone())
                        .unwrap_or_else(|| "—".into()),
                ),
            ]),
            Line::from(vec![
                Span::styled("desired_expr: ", Style::default().fg(Color::Yellow)),
                Span::raw(format!("{}", self.desired_expr)),
                Span::raw("  "),
                Span::styled("n_propagation: ", Style::default().fg(Color::Yellow)),
                Span::raw(format!("{}", self.n_propagation)),
            ]),
            Line::from(vec![
                Span::styled("Verbose timings: ", Style::default().fg(Color::Yellow)),
                Span::raw(if self.pending_verbose { "on (Ctrl+V)" } else { "off (Ctrl+V)" }),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("r", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" run  "),
                Span::styled("e", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" edit desired  "),
                Span::styled("p", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" edit n_prop  "),
                Span::styled("/", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" filter  "),
                Span::styled("Enter", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" pick gene  "),
                Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" quit"),
            ]),
        ];

        let filter_title = if self.filter_focus {
            " Filter (typing) — Esc exit "
        } else {
            " Filter (press /) "
        };
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(hdr.len() as u16 + 1), Constraint::Min(3)])
            .split(chunks[1]);

        f.render_widget(Paragraph::new(hdr), right[0]);
        f.render_widget(
            Paragraph::new(self.gene_filter.as_str())
                .block(Block::default().borders(Borders::ALL).title(filter_title)),
            right[1],
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
                            self.screen = Screen::Loading {
                                label: "Loading PerturbRuntime…".into(),
                            };
                            let tx = tx_bg.clone();
                            std::thread::spawn(move || {
                                let r = PerturbRuntime::from_run_toml(p.as_path()).map_err(|e| e.to_string());
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
                Screen::Loading { .. } => {
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
                Screen::Running => return Ok(None),
                Screen::ResultView { .. } => {
                    self.screen = Screen::Main;
                    return Ok(None);
                }
                Screen::Main => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('v')
                    {
                        self.pending_verbose = !self.pending_verbose;
                        return Ok(None);
                    }
                    if self.filter_focus {
                        match key.code {
                            KeyCode::Esc => {
                                self.filter_focus = false;
                            }
                            KeyCode::Char(c) => {
                                self.gene_filter.push(c);
                                self.rebuild_filter();
                            }
                            KeyCode::Backspace => {
                                self.gene_filter.pop();
                                self.rebuild_filter();
                            }
                            _ => {}
                        }
                        return Ok(None);
                    }

                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(Some(())),
                        KeyCode::Char('/') => {
                            self.filter_focus = true;
                        }
                        KeyCode::Esc => {
                            self.gene_filter.clear();
                            self.rebuild_filter();
                        }
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
                        KeyCode::Char('r') | KeyCode::Char('R') => {
                            let Some(gene) = self
                                .selected_gene_name()
                                .or_else(|| self.selected_gene.clone())
                            else {
                                return Ok(None);
                            };
                            let Ok(target) = single_perturb_target(
                                gene.as_str(),
                                self.desired_expr,
                                &self.runtime.as_ref().unwrap().gene_names,
                            ) else {
                                return Ok(None);
                            };
                            let rt = self.runtime.as_ref().unwrap().clone();
                            let mut cfg: PerturbConfig = rt.perturb_cfg.clone();
                            cfg.n_propagation = self.n_propagation;
                            let targets = vec![target];
                            let capture_timings = self.pending_verbose;
                            self.last_perturbed_gene = Some(gene.clone());
                            self.screen = Screen::Running;
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
                                    None,
                                    None,
                                    None,
                                    Some(&rt.baseline_splash_cache),
                                    &mut timings,
                                );
                                let elapsed = t0.elapsed();
                                let msg = match res {
                                    Ok(result) => BgMsg::Perturbed(Ok(PerturbOutcome {
                                        result,
                                        timings,
                                        elapsed,
                                    })),
                                    Err(_) => BgMsg::Perturbed(Err("perturb_with_targets failed".into())),
                                };
                                let _ = tx.send(msg);
                            });
                        }
                        KeyCode::Down | KeyCode::Char('j') => self.select_next(),
                        KeyCode::Up | KeyCode::Char('k') => self.select_prev(),
                        KeyCode::Enter => {
                            if let Some(g) = self.selected_gene_name() {
                                self.selected_gene = Some(g);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(None)
    }

    fn select_next(&mut self) {
        if self.filtered_gene_indices.is_empty() {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        let n = self.filtered_gene_indices.len();
        self.list_state.select(Some((i + 1).min(n - 1)));
    }

    fn select_prev(&mut self) {
        if self.filtered_gene_indices.is_empty() {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        self.list_state.select(Some(i.saturating_sub(1)));
    }

    fn format_outcome(&self, out: &PerturbOutcome) -> Vec<String> {
        let rt = self.runtime.as_ref().unwrap();
        let gene = self
            .last_perturbed_gene
            .as_deref()
            .unwrap_or("?");
        let mut lines = vec![
            format!("Gene: {gene}"),
            format!("desired_expr: {}", self.desired_expr),
            format!("n_propagation: {}", self.n_propagation),
            format!("Wall time: {:?}", out.elapsed),
        ];

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
                "Δ target column: min={min:.6} max={max:.6} mean={mean:.6} (n={n})"
            ));
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
        lines.push("Press any key to return.".into());
        lines
    }
}
