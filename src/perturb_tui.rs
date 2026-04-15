use std::collections::{HashMap, HashSet, VecDeque};
use std::io::stdout;
use std::panic::AssertUnwindSafe;
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
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

use crate::betadata::BetadataUiProgress;
use crate::config::expand_user_path;
use std::path::Path;
use crate::perturb::{
    PerturbConfig, PerturbResult, PerturbTarget, PerturbTimings, PerturbWithTargetsInputs,
    perturb_with_targets,
};
use crate::perturb_mode::{
    ExportJointPerturbArgs, GeneCellTypeScopes, JointCellsCsvExportSummary, ObsColumnsCsv,
    PerturbRuntime, export_joint_perturb_result, merge_csv_and_type_cell_indices,
    parse_obs_columns_csv, single_perturb_target,
};
use crate::tui_theme::TuiColors;

const SYS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

const LOAD_GAUGE_LINES: symbols::line::Set = symbols::line::Set {
    horizontal: "█",
    ..symbols::line::THICK
};

fn block_panel<'a>(pal: TuiColors, title: impl Into<Line<'a>>, border: Color) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .style(Style::default().bg(pal.bg))
        .title(title)
}

fn styled_result_line(pal: TuiColors, s: &str) -> Line<'static> {
    let style = if s.starts_with("Δ ") {
        Style::default().fg(pal.sky)
    } else if s.starts_with("Joint export") {
        Style::default().fg(pal.c_wrote)
    } else if s.starts_with("Export error") || s.starts_with("Perturbation failed") {
        Style::default().fg(pal.c_fail)
    } else if s.starts_with("Per-step") {
        Style::default().fg(pal.label).add_modifier(Modifier::BOLD)
    } else if s.starts_with("  ") || s.contains("Enter / Esc") {
        Style::default().fg(pal.muted)
    } else {
        Style::default().fg(pal.title)
    };
    Line::from(Span::styled(s.to_string(), style))
}

struct PerturbJobSpec {
    id: u64,
    label: String,
    genes: Vec<String>,
    targets: Vec<PerturbTarget>,
    config: PerturbConfig,
    scopes: GeneCellTypeScopes,
    desired_expr: f64,
    n_propagation: usize,
    joint_csv_summary: Option<JointCellsCsvExportSummary>,
    capture_timings: bool,
}

struct ActivePerturb {
    id: u64,
    label: String,
    progress: Arc<AtomicU32>,
    message: Arc<Mutex<String>>,
    cancel: Arc<AtomicBool>,
}

pub struct PerturbTuiOptions {
    pub run_toml: Option<PathBuf>,
    pub default_desired_expr: f64,
    pub n_propagation_initial: Option<usize>,
    pub verbose: bool,
    pub toml_path_hint_for_error: Option<String>,
    pub cells_csv: Option<PathBuf>,
    pub cells_csv_column: Option<String>,
    /// When set, merged into the repro TOML for the initial auto-load path (not when picking a path manually).
    pub config_merge_overlay: Option<toml::Value>,
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
    PickCsvColumn {
        list_state: ratatui::widgets::ListState,
    },
}

enum BgMsg {
    Loaded {
        generation: u64,
        result: Result<Box<PerturbRuntime>, String>,
    },
    JobDone {
        id: u64,
        outcome: Result<Box<PerturbOutcome>, String>,
    },
}

struct PerturbOutcome {
    job_id: u64,
    genes: Vec<String>,
    desired_expr: f64,
    n_propagation: usize,
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
    let max_parallel = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(2);

    let pick_toml_path_draft = opts
        .run_toml
        .as_ref()
        .map(|p| p.display().to_string())
        .or(opts.toml_path_hint_for_error.clone())
        .unwrap_or_default();

    let mut app = App {
        screen: pick,
        pending_load_generation: 0,
        pick_toml_path_draft,
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
        filtered_gene_indices: Vec::new(),
        load_progress_permille: load_progress_permille.clone(),
        load_progress_message: load_progress_message.clone(),
        cells_csv_path: opts.cells_csv.clone(),
        cells_csv_initial_column: opts.cells_csv_column.clone(),
        cells_csv_data: None,
        cells_csv_selected_col: None,
        last_perturb_lines: Vec::new(),
        last_perturb_scroll: 0,
        job_queue: VecDeque::new(),
        active_jobs: Vec::new(),
        next_job_id: 1,
        max_parallel,
        sys: System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        ),
        last_sys_refresh: Instant::now()
            .checked_sub(SYS_REFRESH_INTERVAL)
            .unwrap_or_else(Instant::now),
        theme_slot: TuiColors::default_theme_slot(),
    };

    if let Some(path) = opts.run_toml.clone() {
        app.pending_load_generation = app.pending_load_generation.wrapping_add(1);
        let load_gen = app.pending_load_generation;
        let tx = tx_bg.clone();
        let p = load_progress_permille.clone();
        let m = load_progress_message.clone();
        let overlay = opts.config_merge_overlay.clone();
        std::thread::spawn(move || {
            let dummy_ui = Arc::new(BetadataUiProgress::new());
            let result = PerturbRuntime::from_run_toml_with_progress(
                path.as_path(),
                Some(p),
                Some(m),
                Some(dummy_ui),
                overlay.as_ref(),
            )
            .map_err(|e| e.to_string())
                .map(Box::new);
            let _ = tx.send(BgMsg::Loaded {
                generation: load_gen,
                result,
            });
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
                BgMsg::Loaded { generation, result } => {
                    if generation != app.pending_load_generation {
                        continue;
                    }
                    match result {
                        Ok(rt_box) => {
                            let mut rt = *rt_box;
                            if let Some(n) = app.load_applied_n_prop {
                                rt.perturb_cfg.n_propagation = n;
                            }
                            app.n_propagation = rt.perturb_cfg.n_propagation;
                            let run_parent = rt
                                .run_toml_path
                                .parent()
                                .filter(|p| !p.as_os_str().is_empty())
                                .unwrap_or_else(|| Path::new("."));
                            if app.cells_csv_path.is_none() {
                                if let Some(ref rel) = rt.cfg.perturbation.cells_csv {
                                    if !rel.trim().is_empty() {
                                        let exp = expand_user_path(rel.trim());
                                        let pb = Path::new(&exp);
                                        app.cells_csv_path = Some(if pb.is_absolute() {
                                            pb.to_path_buf()
                                        } else {
                                            run_parent.join(pb)
                                        });
                                    }
                                }
                            }
                            if app.cells_csv_initial_column.is_none() {
                                app.cells_csv_initial_column =
                                    rt.cfg.perturbation.cells_csv_column.clone();
                            }
                            let rt_arc = std::sync::Arc::new(rt);
                            app.apply_cells_csv_after_load(rt_arc.as_ref());
                            app.runtime = Some(rt_arc);
                            app.screen = Screen::Main;
                            app.rebuild_filter();
                        }
                        Err(e) => {
                            app.screen = Screen::PickToml {
                                path_input: app.pick_toml_path_draft.clone(),
                                err: Some(e),
                            };
                        }
                    }
                }
                BgMsg::JobDone { id, outcome } => {
                    app.active_jobs.retain(|j| j.id != id);
                    match outcome {
                        Ok(out) => {
                            let lines = app.format_outcome(out.as_ref());
                            app.append_job_log(lines);
                            app.last_perturb_scroll = 0;
                            app.verbose = app.pending_verbose;
                        }
                        Err(e) => {
                            app.append_job_log(vec![
                                format!("Job {id} failed: {e}"),
                                String::new(),
                                "Alt+PgUp/Dn scroll · Ctrl+L clear".into(),
                            ]);
                            app.last_perturb_scroll = 0;
                        }
                    }
                    app.drain_job_queue(&tx_bg);
                }
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
    pending_load_generation: u64,
    pick_toml_path_draft: String,
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
    load_progress_permille: Arc<AtomicU32>,
    load_progress_message: Arc<Mutex<String>>,
    cells_csv_path: Option<PathBuf>,
    cells_csv_initial_column: Option<String>,
    cells_csv_data: Option<ObsColumnsCsv>,
    cells_csv_selected_col: Option<usize>,
    last_perturb_lines: Vec<String>,
    last_perturb_scroll: usize,
    job_queue: VecDeque<PerturbJobSpec>,
    active_jobs: Vec<ActivePerturb>,
    next_job_id: u64,
    max_parallel: usize,
    sys: System,
    last_sys_refresh: Instant,
    theme_slot: usize,
}

impl App {
    fn pal(&self) -> TuiColors {
        TuiColors::resolve(self.theme_slot)
    }

    fn refresh_sys_if_due(&mut self) {
        if self.last_sys_refresh.elapsed() >= SYS_REFRESH_INTERVAL {
            self.sys.refresh_cpu_all();
            self.sys.refresh_memory();
            self.last_sys_refresh = Instant::now();
        }
    }

    fn sys_resource_spans(&self) -> Vec<Span<'static>> {
        let p = self.pal();
        let cpu_pct = self.sys.global_cpu_usage();
        let used_mem = self.sys.used_memory();
        let total_mem = self.sys.total_memory().max(1);
        let mem_pct = (used_mem as f64 / total_mem as f64) * 100.0;
        vec![
            Span::styled("CPU ", Style::default().fg(p.label)),
            Span::styled(
                format!("{cpu_pct:5.1}%"),
                Style::default().fg(p.sky).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  MEM ", Style::default().fg(p.label)),
            Span::styled(
                format!("{mem_pct:5.1}%"),
                Style::default().fg(p.sky).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  RAM ", Style::default().fg(p.label)),
            Span::styled(
                format!(
                    "{} / {} MiB",
                    used_mem / 1024 / 1024,
                    total_mem / 1024 / 1024
                ),
                Style::default().fg(p.muted),
            ),
        ]
    }

    fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_line = msg.into();
        self.status_is_error = is_error;
    }

    fn clear_status(&mut self) {
        self.status_line.clear();
        self.status_is_error = false;
    }

    fn append_job_log(&mut self, mut block: Vec<String>) {
        if !self.last_perturb_lines.is_empty() && !block.is_empty() {
            block.push(String::new());
        }
        block.extend(std::mem::take(&mut self.last_perturb_lines));
        self.last_perturb_lines = block;
        const CAP: usize = 500;
        if self.last_perturb_lines.len() > CAP {
            self.last_perturb_lines.truncate(CAP);
        }
    }

    fn drain_job_queue(&mut self, tx_bg: &mpsc::Sender<BgMsg>) {
        let Some(rt) = self.runtime.as_ref() else {
            return;
        };
        let rt = Arc::clone(rt);
        while self.active_jobs.len() < self.max_parallel {
            let Some(spec) = self.job_queue.pop_front() else {
                break;
            };
            let id = spec.id;
            let job_p = Arc::new(AtomicU32::new(0));
            let job_m = Arc::new(Mutex::new(String::from("GRN perturbation · starting…")));
            let cancel = Arc::new(AtomicBool::new(false));
            self.active_jobs.push(ActivePerturb {
                id,
                label: spec.label.clone(),
                progress: job_p.clone(),
                message: job_m.clone(),
                cancel: cancel.clone(),
            });
            let tx = tx_bg.clone();
            let rt_t = Arc::clone(&rt);
            std::thread::spawn(move || {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(
                    || -> Result<PerturbOutcome, String> {
                        let t0 = Instant::now();
                        let mut timings = if spec.capture_timings {
                            Some(PerturbTimings::default())
                        } else {
                            None
                        };
                        let res = perturb_with_targets(
                            &PerturbWithTargetsInputs {
                                bb: &rt_t.bb,
                                gene_mtx: &rt_t.gene_mtx,
                                gene_names: &rt_t.gene_names,
                                xy: &rt_t.xy,
                                rw_ligands_init: &rt_t.rw_ligands_init,
                                rw_tfligands_init: &rt_t.rw_tfligands_init,
                                targets: &spec.targets,
                                config: &spec.config,
                                lr_radii: &rt_t.lr_radii,
                                job_progress: Some(&job_p),
                                job_message: Some(&job_m),
                                cancel: Some(cancel.as_ref()),
                                baseline_splash_cache: Some(&rt_t.baseline_splash_cache),
                            },
                            &mut timings,
                        );
                        let elapsed = t0.elapsed();
                        match res {
                            Some(result) => {
                                let genes = spec.genes.clone();
                                let joint_csv = spec.joint_csv_summary;
                                let (export_dir, export_err) = match export_joint_perturb_result(
                                    ExportJointPerturbArgs {
                                        runtime: rt_t.as_ref(),
                                        simulated: &result.simulated,
                                        selected_genes: &genes,
                                        desired_expr: spec.desired_expr,
                                        n_propagation: spec.n_propagation,
                                        selected_cell_types_per_gene: &spec.scopes,
                                        cells_csv_summary: joint_csv,
                                        job_id: Some(id),
                                    },
                                ) {
                                    Ok(p) => (Some(p), None),
                                    Err(e) => (None, Some(e.to_string())),
                                };
                                Ok(PerturbOutcome {
                                    job_id: id,
                                    genes,
                                    desired_expr: spec.desired_expr,
                                    n_propagation: spec.n_propagation,
                                    result,
                                    timings,
                                    elapsed,
                                    export_dir,
                                    export_err,
                                })
                            }
                            None => {
                                let msg = if cancel.load(Ordering::Relaxed) {
                                    "canceled".into()
                                } else {
                                    "perturb_with_targets failed".into()
                                };
                                Err(msg)
                            }
                        }
                    },
                ));
                let outcome = match outcome {
                    Ok(inner) => inner.map(Box::new),
                    Err(_) => Err("job panicked".into()),
                };
                let _ = tx.send(BgMsg::JobDone { id, outcome });
            });
        }
    }

    fn rebuild_filter(&mut self) {
        let Some(rt) = self.runtime.as_ref() else {
            self.filtered_gene_indices.clear();
            return;
        };
        const CAP: usize = 50_000;
        let q = self.gene_filter.to_lowercase();
        if q.is_empty() {
            self.filtered_gene_indices = (0..rt.gene_names.len()).collect();
            if self.filtered_gene_indices.len() > CAP {
                self.filtered_gene_indices.truncate(CAP);
                if !self.status_is_error {
                    self.set_status(
                        format!(
                            "Listing first {CAP} genes only ({} total) — type to filter",
                            rt.gene_names.len()
                        ),
                        false,
                    );
                }
            } else if self.status_line.contains("Listing first") && !self.status_is_error {
                self.clear_status();
            }
        } else {
            let mut v: Vec<usize> = rt
                .gene_names
                .iter()
                .enumerate()
                .filter(|(_, g)| g.to_lowercase().contains(&q))
                .map(|(i, _)| i)
                .collect();
            let total_matched = v.len();
            if total_matched > CAP {
                v.truncate(CAP);
                if !self.status_is_error {
                    self.set_status(
                        format!(
                            "Filter matched {total_matched} genes; showing first {CAP} — narrow your filter"
                        ),
                        false,
                    );
                }
            } else if self.status_line.contains("narrow your filter") && !self.status_is_error {
                self.clear_status();
            }
            self.filtered_gene_indices = v;
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

    fn apply_cells_csv_after_load(&mut self, rt: &PerturbRuntime) {
        self.cells_csv_data = None;
        self.cells_csv_selected_col = None;
        let Some(ref path) = self.cells_csv_path else {
            self.clear_status();
            return;
        };
        match parse_obs_columns_csv(path.as_path(), &rt.obs_names) {
            Ok(csv) => {
                if let Some(ref want) = self.cells_csv_initial_column {
                    if let Some(i) = csv.column_names.iter().position(|c| c == want) {
                        self.cells_csv_selected_col = Some(i);
                        self.cells_csv_data = Some(csv);
                        self.clear_status();
                    } else {
                        self.cells_csv_data = Some(csv);
                        self.set_status(
                            format!(
                                "cells CSV: column '{}' not in header (Ctrl+O to pick)",
                                want
                            ),
                            true,
                        );
                    }
                } else {
                    self.cells_csv_data = Some(csv);
                    self.clear_status();
                }
            }
            Err(e) => self.set_status(format!("cells CSV: {e}"), true),
        }
    }

    fn cells_csv_selected_column_name(&self) -> Option<String> {
        let i = self.cells_csv_selected_col?;
        let csv = self.cells_csv_data.as_ref()?;
        csv.column_names.get(i).cloned()
    }

    fn csv_indices_slice(&self) -> Option<&[usize]> {
        let name = self.cells_csv_selected_column_name()?;
        self.cells_csv_data
            .as_ref()?
            .indices_for_column(name.as_str())
    }

    fn open_pick_csv_column(&mut self) {
        let Some(csv) = self.cells_csv_data.as_ref() else {
            self.set_status("No cells CSV loaded (use --cells-csv)", true);
            return;
        };
        if csv.column_names.is_empty() {
            self.set_status("cells CSV has no columns", true);
            return;
        }
        let mut list_state = ratatui::widgets::ListState::default();
        let sel = self
            .cells_csv_selected_col
            .unwrap_or(0)
            .min(csv.column_names.len().saturating_sub(1));
        list_state.select(Some(sel));
        self.screen = Screen::PickCsvColumn { list_state };
        self.clear_status();
    }

    fn expression_preview_content(&self, inner_width: u16) -> (Line<'static>, Vec<Line<'static>>) {
        use std::collections::HashMap;

        let p = self.pal();
        let Some(rt) = self.runtime.as_ref() else {
            return (
                Line::from(Span::styled(
                    " Expression preview ",
                    Style::default().fg(p.label).add_modifier(Modifier::BOLD),
                )),
                vec![Line::from(Span::styled(
                    "Load a run to preview.",
                    Style::default().fg(p.muted),
                ))],
            );
        };
        let title_layer = rt.cfg.data.layer.clone();
        let Some(gene) = self.selected_gene_name() else {
            return (
                Line::from(vec![
                    Span::styled(" Mean expr ", Style::default().fg(p.label)),
                    Span::styled(format!("· {} · ", title_layer), Style::default().fg(p.muted)),
                ]),
                vec![Line::from(Span::styled(
                    "Select a gene (↑/↓) for cluster means (input layer).",
                    Style::default().fg(p.muted),
                ))],
            );
        };
        let Some(gi) = rt.gene_names.iter().position(|g| g == &gene) else {
            return (
                Line::from(Span::styled(
                    " Expression preview ",
                    Style::default().fg(p.label).add_modifier(Modifier::BOLD),
                )),
                vec![Line::from("Gene index error.")],
            );
        };
        let n = rt.gene_mtx.nrows();
        if rt.betadata_cluster_key.len() != n {
            return (
                Line::from(Span::styled(
                    " Expression preview ",
                    Style::default().fg(p.label).add_modifier(Modifier::BOLD),
                )),
                vec![Line::from(Span::styled(
                    "Cluster key length mismatch.",
                    Style::default().fg(p.c_fail),
                ))],
            );
        }
        let mut acc: HashMap<String, (f64, usize)> = HashMap::new();
        for r in 0..n {
            let k = rt.betadata_cluster_key[r].clone();
            let v = rt.gene_mtx[[r, gi]];
            let e = acc.entry(k).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
        let mut pairs: Vec<(String, f64)> = acc
            .into_iter()
            .map(|(k, (s, c))| (k, s / c as f64))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        const CAP: usize = 15;
        pairs.truncate(CAP);
        let vmax = pairs
            .first()
            .map(|(_, m)| *m)
            .filter(|m| m.is_finite())
            .unwrap_or(0.0)
            .abs()
            .max(1e-12);

        let inner_w = inner_width.max(8) as usize;
        let label_w = (inner_w / 4).clamp(6, 18);
        let bar_w = inner_w.saturating_sub(label_w + 8 + 3).max(4);

        let trunc_label = |s: &str, max_c: usize| -> String {
            let nch = s.chars().count();
            if nch <= max_c {
                return s.to_string();
            }
            let take = max_c.saturating_sub(1);
            format!("{}…", s.chars().take(take).collect::<String>())
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        for (label, mean) in pairs {
            let lab = trunc_label(&label, label_w);
            let frac = (mean.abs() / vmax).clamp(0.0, 1.0);
            let nf = (frac * bar_w as f64).round() as usize;
            let bar: String = "█".repeat(nf) + &"░".repeat(bar_w.saturating_sub(nf));
            let num = format!("{:>7.3}", mean);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{lab:<label_w$}"),
                    Style::default().fg(p.title),
                ),
                Span::raw(" "),
                Span::styled(bar, Style::default().fg(p.sky)),
                Span::raw(" "),
                Span::styled(num, Style::default().fg(p.muted)),
            ]));
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No rows to aggregate.",
                Style::default().fg(p.muted),
            )));
        }
        let block_title = Line::from(vec![
            Span::styled(" Mean ", Style::default().fg(p.label)),
            Span::styled(gene.clone(), Style::default().fg(p.value).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(" · {} ", title_layer),
                Style::default().fg(p.muted),
            ),
        ]);
        (block_title, lines)
    }

    fn open_cell_scope_editor(&mut self, gene: String, rt: &PerturbRuntime) {
        let mut cell_types: Vec<usize> = rt.cell_types.to_vec();
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
        self.refresh_sys_if_due();
        let pal = self.pal();
        f.render_widget(
            Block::default().style(Style::default().bg(pal.bg)),
            f.area(),
        );

        let outer_title = Line::from(vec![
            Span::styled("✿ ", Style::default().fg(pal.grape)),
            Span::styled(
                "spacetravlr-perturb",
                Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
            ),
        ]);
        let block = block_panel(pal, outer_title, pal.tel_bord);
        let inner = block.inner(f.area());
        f.render_widget(block, f.area());

        match &mut self.screen {
            Screen::PickToml { path_input, err } => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(3),
                        Constraint::Length(3),
                        Constraint::Length(1),
                    ])
                    .split(inner);
                let mut txt = vec![
                    Line::from(vec![
                        Span::styled("Path to ", Style::default().fg(pal.title)),
                        Span::styled(
                            "spacetravlr_run_repro.toml",
                            Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "  ·  Enter load  ·  Esc quit",
                            Style::default().fg(pal.muted),
                        ),
                    ]),
                    Line::from(""),
                ];
                if let Some(e) = err {
                    txt.push(Line::from(vec![Span::styled(
                        format!("Error: {e}"),
                        Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD),
                    )]));
                }
                f.render_widget(
                    Paragraph::new(txt)
                        .style(Style::default().bg(pal.bg))
                        .wrap(Wrap { trim: true }),
                    chunks[0],
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        path_input.as_str(),
                        Style::default().fg(pal.title),
                    ))
                    .block(block_panel(
                        pal,
                        Line::from(Span::styled(" Path ", Style::default().fg(pal.label))),
                        pal.sky,
                    )),
                    chunks[1],
                );
                f.render_widget(
                    Paragraph::new(Line::from(self.sys_resource_spans()))
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(pal.bg)),
                    chunks[2],
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
                        Constraint::Length(1),
                        Constraint::Min(2),
                        Constraint::Length(1),
                    ])
                    .split(inner);

                let title = Line::from(vec![
                    Span::styled("✿ ", Style::default().fg(pal.grape)),
                    Span::styled(
                        "Loading PerturbRuntime",
                        Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ·  {pct}%"),
                        Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                    ),
                ]);
                f.render_widget(
                    Paragraph::new(title)
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(pal.bg)),
                    chunks[0],
                );

                let gauge = LineGauge::default()
                    .style(Style::default().bg(pal.bg))
                    .filled_style(Style::default().fg(pal.sky).add_modifier(Modifier::BOLD))
                    .unfilled_style(Style::default().fg(pal.gauge_empty))
                    .filled_symbol(LOAD_GAUGE_LINES.horizontal)
                    .unfilled_symbol(symbols::line::THICK.horizontal)
                    .label(Line::from(""))
                    .ratio(ratio);
                f.render_widget(gauge, chunks[1]);

                f.render_widget(
                    Paragraph::new(Line::from(self.sys_resource_spans()))
                        .alignment(Alignment::Center)
                        .style(Style::default().bg(pal.bg)),
                    chunks[2],
                );

                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            format!("{spin}  "),
                            Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(status, Style::default().fg(pal.value)),
                    ]))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true })
                    .style(Style::default().bg(pal.bg)),
                    chunks[3],
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        "Esc — return to path entry (load continues in background; result ignored)",
                        Style::default().fg(pal.muted),
                    ))
                    .alignment(Alignment::Center)
                    .style(Style::default().bg(pal.bg)),
                    chunks[4],
                );
            }
            Screen::Main => self.render_main(f, inner, pal),
            Screen::EditDesired { buf } => {
                let mut lines = vec![
                    Line::from(vec![
                        Span::styled("Desired ", Style::default().fg(pal.title)),
                        Span::styled("desired_expr", Style::default().fg(pal.value)),
                        Span::styled(
                            "  ·  Enter OK  ·  Esc cancel",
                            Style::default().fg(pal.muted),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(
                        buf.as_str(),
                        Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                    )),
                ];
                if !self.status_line.is_empty() {
                    let st = if self.status_is_error {
                        Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(pal.c_wrote)
                    };
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(self.status_line.as_str(), st)));
                }
                f.render_widget(
                    Paragraph::new(lines)
                        .block(block_panel(
                            pal,
                            Line::from(Span::styled(
                                " desired_expr ",
                                Style::default().fg(pal.label),
                            )),
                            pal.work_bord,
                        ))
                        .style(Style::default().bg(pal.bg)),
                    inner,
                );
            }
            Screen::EditNPropagation { buf } => {
                let mut lines = vec![
                    Line::from(vec![
                        Span::styled("Integer ", Style::default().fg(pal.title)),
                        Span::styled("n_propagation", Style::default().fg(pal.value)),
                        Span::styled(
                            "  ·  Enter OK  ·  Esc cancel",
                            Style::default().fg(pal.muted),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(
                        buf.as_str(),
                        Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                    )),
                ];
                if !self.status_line.is_empty() {
                    let st = if self.status_is_error {
                        Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(pal.c_wrote)
                    };
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(self.status_line.as_str(), st)));
                }
                f.render_widget(
                    Paragraph::new(lines)
                        .block(block_panel(
                            pal,
                            Line::from(Span::styled(
                                " n_propagation ",
                                Style::default().fg(pal.label),
                            )),
                            pal.rocket_bord,
                        ))
                        .style(Style::default().bg(pal.bg)),
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
                            ("[•]", pal.grape)
                        } else {
                            ("[ ]", pal.muted)
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled(format!("{m} "), Style::default().fg(mc)),
                            Span::styled("cell_type ", Style::default().fg(pal.muted)),
                            Span::styled(format!("{ct}"), Style::default().fg(pal.title)),
                        ]))
                    })
                    .collect();
                let hint = vec![
                    Line::from(vec![
                        Span::styled("Cell scope  ", Style::default().fg(pal.label)),
                        Span::styled("·  ", Style::default().fg(pal.muted)),
                        Span::styled(
                            gene.as_str(),
                            Style::default().fg(pal.grape).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "  ·  Space toggle  ·  Enter save  ·  Esc cancel",
                            Style::default().fg(pal.muted),
                        ),
                    ]),
                    Line::from(vec![Span::styled(
                        "All types on = whole tissue. Subset = only those cell types get the perturbation.",
                        Style::default().fg(pal.muted),
                    )]),
                    Line::from(""),
                ];
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(4), Constraint::Min(3)])
                    .split(inner);
                f.render_widget(
                    Paragraph::new(hint).style(Style::default().bg(pal.bg)),
                    chunks[0],
                );
                let list = List::new(items)
                    .block(block_panel(
                        pal,
                        Line::from(Span::styled(" cell types ", Style::default().fg(pal.label))),
                        pal.rocket_bord,
                    ))
                    .highlight_style(
                        Style::default()
                            .bg(pal.sky)
                            .fg(pal.bg)
                            .add_modifier(Modifier::BOLD),
                    );
                f.render_stateful_widget(list, chunks[1], list_state);
            }
            Screen::PickCsvColumn { list_state } => {
                let names: &[String] = self
                    .cells_csv_data
                    .as_ref()
                    .map(|c| c.column_names.as_slice())
                    .unwrap_or(&[]);
                let items: Vec<ListItem> = names
                    .iter()
                    .map(|n| {
                        let n_cells = self
                            .cells_csv_data
                            .as_ref()
                            .and_then(|c| c.indices_for_column(n.as_str()))
                            .map(|v| v.len())
                            .unwrap_or(0);
                        ListItem::new(Line::from(vec![
                            Span::styled(n.as_str(), Style::default().fg(pal.title)),
                            Span::styled(
                                format!("  ({n_cells} cells)"),
                                Style::default().fg(pal.muted),
                            ),
                        ]))
                    })
                    .collect();
                let hint = vec![
                    Line::from(vec![
                        Span::styled("Cells CSV column ", Style::default().fg(pal.label)),
                        Span::styled(
                            "  ·  Enter save  ·  Esc cancel",
                            Style::default().fg(pal.muted),
                        ),
                    ]),
                    Line::from(vec![Span::styled(
                        "Values must be obs_names; merged with per-gene cell-type scope (intersection).",
                        Style::default().fg(pal.muted),
                    )]),
                    Line::from(""),
                ];
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(4), Constraint::Min(3)])
                    .split(inner);
                f.render_widget(
                    Paragraph::new(hint).style(Style::default().bg(pal.bg)),
                    chunks[0],
                );
                let list = List::new(items)
                    .block(block_panel(
                        pal,
                        Line::from(Span::styled(" CSV columns ", Style::default().fg(pal.label))),
                        pal.rocket_bord,
                    ))
                    .highlight_style(
                        Style::default()
                            .bg(pal.sky)
                            .fg(pal.bg)
                            .add_modifier(Modifier::BOLD),
                    );
                f.render_stateful_widget(list, chunks[1], list_state);
            }
        }
    }

    fn render_jobs_strip(&self, f: &mut Frame, area: Rect, pal: TuiColors) {
        let head = Line::from(vec![
            Span::styled("Parallel jobs ", Style::default().fg(pal.label)),
            Span::styled(
                format!("{} running ", self.active_jobs.len()),
                Style::default().fg(pal.sky),
            ),
            Span::styled("· ", Style::default().fg(pal.muted)),
            Span::styled(
                format!("{} queued ", self.job_queue.len()),
                Style::default().fg(pal.lilac),
            ),
            Span::styled("· ", Style::default().fg(pal.muted)),
            Span::styled(
                format!("≤{} at once ", self.max_parallel),
                Style::default().fg(pal.muted),
            ),
            Span::styled(
                "· Ctrl+R queue · Ctrl+X cancel all",
                Style::default().fg(pal.muted),
            ),
        ]);
        let mut lines: Vec<Line> = vec![head];
        for j in &self.active_jobs {
            let permille = j.progress.load(Ordering::Relaxed);
            let pct = permille / 10;
            let msg = j.message.lock().map(|g| g.clone()).unwrap_or_default();
            let short = if msg.chars().count() > 72 {
                let t: String = msg.chars().take(69).collect();
                format!("{t}…")
            } else {
                msg
            };
            lines.push(Line::from(vec![
                Span::styled(format!("#{} ", j.id), Style::default().fg(pal.grape)),
                Span::styled(j.label.as_str(), Style::default().fg(pal.title)),
                Span::styled(format!(" {pct}% "), Style::default().fg(pal.work_bord)),
                Span::styled(short, Style::default().fg(pal.value)),
            ]));
        }
        f.render_widget(
            Paragraph::new(lines)
                .style(Style::default().bg(pal.bg))
                .block(block_panel(
                    pal,
                    Line::from(Span::styled(" Queue ", Style::default().fg(pal.label))),
                    pal.work_bord,
                ))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn render_last_perturb_panel(&mut self, f: &mut Frame, area: Rect, pal: TuiColors) {
        let lines = &self.last_perturb_lines;
        let view_h = area.height.saturating_sub(2) as usize;
        let max_scroll = lines.len().saturating_sub(view_h.max(1));
        let s = self.last_perturb_scroll.min(max_scroll);
        self.last_perturb_scroll = s;
        let end = (s + view_h.max(1)).min(lines.len());
        let slice: Vec<Line> = lines[s..end]
            .iter()
            .map(|x| styled_result_line(pal, x))
            .collect();
        let title_line = if lines.len() > view_h.max(1) {
            Line::from(vec![
                Span::styled(
                    " Latest run ",
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ),
                Span::styled("·", Style::default().fg(pal.muted)),
                Span::styled(" Alt+PgUp/Dn ", Style::default().fg(pal.sky)),
                Span::styled(
                    format!("({}/{}) ", s + 1, lines.len()),
                    Style::default().fg(pal.lilac),
                ),
                Span::styled("· Ctrl+L clear ", Style::default().fg(pal.muted)),
            ])
        } else {
            Line::from(vec![
                Span::styled(
                    " Latest run ",
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ),
                Span::styled("· Ctrl+L clear ", Style::default().fg(pal.muted)),
            ])
        };
        f.render_widget(
            Paragraph::new(slice)
                .style(Style::default().bg(pal.bg))
                .block(block_panel(pal, title_line, pal.tel_bord))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn render_main(&mut self, f: &mut Frame, area: Rect, pal: TuiColors) {
        let has_summary = !self.last_perturb_lines.is_empty();
        let show_jobs = !self.active_jobs.is_empty() || !self.job_queue.is_empty();
        let summary_h = if has_summary {
            7u16.min(area.height.saturating_sub(5)).max(4)
        } else {
            0
        };
        let jobs_h = if show_jobs {
            let want = (3u16.saturating_add(self.active_jobs.len() as u16)).clamp(3, 10);
            want.min(area.height.saturating_sub(summary_h).saturating_sub(4))
        } else {
            0
        };
        let status_h = 1u16;
        let body_h = area
            .height
            .saturating_sub(summary_h)
            .saturating_sub(jobs_h)
            .saturating_sub(status_h)
            .max(3);
        let summary_rect = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: summary_h,
        };
        let jobs_rect = Rect {
            x: area.x,
            y: area.y + summary_h,
            width: area.width,
            height: jobs_h,
        };
        let body = Rect {
            x: area.x,
            y: area.y + summary_h + jobs_h,
            width: area.width,
            height: body_h,
        };
        let status_area = Rect {
            x: area.x,
            y: area.y + summary_h + jobs_h + body_h,
            width: area.width,
            height: status_h,
        };
        if has_summary {
            self.render_last_perturb_panel(f, summary_rect, pal);
        }
        if show_jobs {
            self.render_jobs_strip(f, jobs_rect, pal);
        }

        let Some(rt) = self.runtime.as_ref() else {
            let fill = Rect {
                x: area.x,
                y: area.y + summary_h + jobs_h,
                width: area.width,
                height: area.height.saturating_sub(summary_h + jobs_h),
            };
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        "Internal error: PerturbRuntime missing on Main screen. Press Ctrl+Q to quit.",
                        Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD),
                    )),
                ])
                .alignment(Alignment::Center)
                .style(Style::default().bg(pal.bg)),
                fill,
            );
            return;
        };
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
                            ("[•] ", pal.grape)
                        } else {
                            ("[ ] ", pal.muted)
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled(mark, Style::default().fg(mc)),
                            Span::styled(g.as_str(), Style::default().fg(pal.title)),
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
                pal,
                Line::from(vec![
                    Span::styled(
                        " Genes ",
                        Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(count_note, Style::default().fg(pal.sky)),
                    Span::styled(
                        " · PgUp/PgDn Home/End · Ctrl+J/K · type · Space ",
                        Style::default().fg(pal.muted),
                    ),
                ]),
                pal.rocket_bord,
            ))
            .highlight_style(
                Style::default()
                    .bg(pal.sky)
                    .fg(pal.bg)
                    .add_modifier(Modifier::BOLD),
            );

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

        let csv_hint = match (&self.cells_csv_path, self.cells_csv_selected_column_name()) {
            (Some(p), Some(col)) => format!("{} → {}", p.display(), col),
            (Some(p), None) => format!("{} (Ctrl+O pick column)", p.display()),
            (None, _) => String::new(),
        };

        let theme_label = TuiColors::theme_label(self.theme_slot);
        let hdr = vec![
            Line::from(vec![
                Span::styled("Run ", Style::default().fg(pal.label)),
                Span::styled("· ", Style::default().fg(pal.muted)),
                Span::styled(
                    rt.run_toml_path.display().to_string(),
                    Style::default().fg(pal.value),
                ),
            ]),
            Line::from(vec![
                Span::styled("Cells × genes ", Style::default().fg(pal.label)),
                Span::styled("· ", Style::default().fg(pal.muted)),
                Span::styled(
                    format!("{} × {}", rt.obs_names.len(), rt.gene_names.len()),
                    Style::default().fg(pal.sky),
                ),
            ]),
            Line::from(self.sys_resource_spans()),
            Line::from(vec![
                Span::styled("Targets ", Style::default().fg(pal.label)),
                Span::styled("· ", Style::default().fg(pal.muted)),
                Span::styled(
                    if self.perturb_targets.is_empty() {
                        "— (none)".to_string()
                    } else {
                        self.perturb_targets.join(", ")
                    },
                    Style::default().fg(if self.perturb_targets.is_empty() {
                        pal.muted
                    } else {
                        pal.lilac
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled("Row scope ", Style::default().fg(pal.label)),
                Span::styled("· ", Style::default().fg(pal.muted)),
                Span::styled(
                    if scope_hint.is_empty() {
                        "—".into()
                    } else {
                        scope_hint
                    },
                    Style::default().fg(pal.grape),
                ),
            ]),
            Line::from(vec![
                Span::styled("CSV cells ", Style::default().fg(pal.label)),
                Span::styled("· ", Style::default().fg(pal.muted)),
                Span::styled(
                    if csv_hint.is_empty() {
                        "—".into()
                    } else {
                        csv_hint
                    },
                    Style::default().fg(if self.cells_csv_data.is_some() {
                        pal.sky
                    } else {
                        pal.muted
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled("desired_expr ", Style::default().fg(pal.label)),
                Span::styled(
                    format!("{}", self.desired_expr),
                    Style::default().fg(pal.value).add_modifier(Modifier::BOLD),
                ),
                Span::styled("   n_prop ", Style::default().fg(pal.label)),
                Span::styled(
                    format!("{}", self.n_propagation),
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Theme ", Style::default().fg(pal.label)),
                Span::styled(
                    format!("{theme_label} · "),
                    Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                ),
                Span::styled("t cycle", Style::default().fg(pal.muted)),
            ]),
            Line::from(vec![
                Span::styled("Verbose ", Style::default().fg(pal.label)),
                Span::styled(
                    if self.pending_verbose { "on" } else { "off" },
                    Style::default()
                        .fg(if self.pending_verbose {
                            pal.c_wrote
                        } else {
                            pal.muted
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" (Ctrl+V)", Style::default().fg(pal.muted)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "Ctrl+R",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" queue  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+T",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" scope  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+O",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" csv  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+E",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" d.expr  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+P",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" n  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+X",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" halt q  ", Style::default().fg(pal.muted)),
                Span::styled("Esc", Style::default().fg(pal.sky).add_modifier(Modifier::BOLD)),
                Span::styled(" clr  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+Q",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" quit  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Ctrl+L",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" clr run log", Style::default().fg(pal.muted)),
            ]),
        ];

        let filter_title = Line::from(vec![
            Span::styled(" Filter ", Style::default().fg(pal.label)),
            Span::styled("· type to narrow", Style::default().fg(pal.muted)),
        ]);
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(hdr.len() as u16 + 1),
                Constraint::Min(2),
                Constraint::Min(5),
            ])
            .split(chunks[1]);

        f.render_widget(
            Paragraph::new(hdr).style(Style::default().bg(pal.bg)),
            right[0],
        );
        f.render_widget(
            Paragraph::new(Span::styled(
                self.gene_filter.as_str(),
                Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
            ))
            .block(block_panel(pal, filter_title, pal.outer_bord)),
            right[1],
        );
        let (preview_title, preview_lines) = self.expression_preview_content(right[2].width);
        f.render_widget(
            Paragraph::new(preview_lines).block(block_panel(pal, preview_title, pal.work_bord)),
            right[2],
        );

        let st_style = if self.status_is_error {
            Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pal.c_wrote)
        };
        let status_txt = if self.status_line.is_empty() {
            Line::from(vec![
                Span::styled("✿ ", Style::default().fg(pal.grape)),
                Span::styled("Joint export ", Style::default().fg(pal.label)),
                Span::styled("· auto after run · ", Style::default().fg(pal.muted)),
                Span::styled("summary above · Ctrl+L clear", Style::default().fg(pal.sky)),
            ])
        } else {
            Line::from(vec![Span::styled(self.status_line.as_str(), st_style)])
        };
        f.render_widget(
            Paragraph::new(status_txt).style(Style::default().bg(pal.bg)),
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
                            self.pick_toml_path_draft = p.display().to_string();
                            self.pending_load_generation =
                                self.pending_load_generation.wrapping_add(1);
                            let load_gen = self.pending_load_generation;
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
                                let result = PerturbRuntime::from_run_toml_with_progress(
                                    p.as_path(),
                                    Some(prog_p),
                                    Some(prog_m),
                                    Some(dummy_ui),
                                    None,
                                )
                                .map_err(|e| e.to_string())
                                .map(Box::new);
                                let _ = tx.send(BgMsg::Loaded {
                                    generation: load_gen,
                                    result,
                                });
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
                        self.pending_load_generation = self.pending_load_generation.wrapping_add(1);
                        self.screen = Screen::PickToml {
                            path_input: self.pick_toml_path_draft.clone(),
                            err: None,
                        };
                    }
                    return Ok(None);
                }
                Screen::EditDesired { buf } => match key.code {
                    KeyCode::Esc => {
                        self.clear_status();
                        self.screen = Screen::Main;
                    }
                    KeyCode::Enter => match buf.trim().parse::<f64>() {
                        Ok(v) if v.is_finite() => {
                            self.clear_status();
                            self.desired_expr = v;
                            self.screen = Screen::Main;
                        }
                        Ok(_) => self.set_status("desired_expr must be finite", true),
                        Err(_) => self.set_status("desired_expr must be a number", true),
                    },
                    KeyCode::Char(c) => buf.push(c),
                    KeyCode::Backspace => {
                        buf.pop();
                    }
                    _ => {}
                },
                Screen::EditNPropagation { buf } => match key.code {
                    KeyCode::Esc => {
                        self.clear_status();
                        self.screen = Screen::Main;
                    }
                    KeyCode::Enter => match buf.trim().parse::<usize>() {
                        Ok(v) if v > 0 => {
                            self.clear_status();
                            self.n_propagation = v;
                            self.screen = Screen::Main;
                        }
                        Ok(_) => self.set_status("n_propagation must be at least 1", true),
                        Err(_) => self.set_status("n_propagation must be a positive integer", true),
                    },
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
                Screen::PickCsvColumn { list_state } => {
                    let n = self
                        .cells_csv_data
                        .as_ref()
                        .map(|c| c.column_names.len())
                        .unwrap_or(0);
                    match key.code {
                        KeyCode::Esc => self.screen = Screen::Main,
                        KeyCode::Enter => {
                            if let Some(i) = list_state.selected() {
                                self.cells_csv_selected_col = Some(i);
                            }
                            self.screen = Screen::Main;
                            self.clear_status();
                        }
                        KeyCode::Up => {
                            let i = list_state.selected().unwrap_or(0);
                            list_state.select(Some(i.saturating_sub(1)));
                        }
                        KeyCode::Down => {
                            if n == 0 {
                                return Ok(None);
                            }
                            let i = list_state.selected().unwrap_or(0);
                            list_state.select(Some((i + 1).min(n - 1)));
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Screen::Main => {
                    if matches!(key.code, KeyCode::Char('t' | 'T'))
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    {
                        self.theme_slot = TuiColors::advance_slot(self.theme_slot);
                        return Ok(None);
                    }
                    if key.modifiers.contains(KeyModifiers::ALT)
                        && !self.last_perturb_lines.is_empty()
                    {
                        match key.code {
                            KeyCode::PageUp => {
                                self.last_perturb_scroll =
                                    self.last_perturb_scroll.saturating_sub(3);
                                return Ok(None);
                            }
                            KeyCode::PageDown => {
                                self.last_perturb_scroll =
                                    self.last_perturb_scroll.saturating_add(3);
                                return Ok(None);
                            }
                            _ => {}
                        }
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('l')
                    {
                        self.last_perturb_lines.clear();
                        self.last_perturb_scroll = 0;
                        return Ok(None);
                    }
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
                                let Some(rt_arc) = self.runtime.clone() else {
                                    self.set_status("PerturbRuntime not available", true);
                                    return Ok(None);
                                };
                                self.open_cell_scope_editor(gene, rt_arc.as_ref());
                                self.clear_status();
                            }
                            KeyCode::Char('o') | KeyCode::Char('O') => {
                                self.open_pick_csv_column();
                            }
                            KeyCode::Char('x') | KeyCode::Char('X') => {
                                for j in &self.active_jobs {
                                    j.cancel.store(true, Ordering::Relaxed);
                                }
                                let nq = self.job_queue.len();
                                self.job_queue.clear();
                                self.set_status(
                                    format!(
                                        "Signaled cancel for running jobs; cleared {nq} queued"
                                    ),
                                    false,
                                );
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                if self.perturb_targets.is_empty() {
                                    self.set_status(
                                        "No targets — Space toggles genes in the list",
                                        true,
                                    );
                                    return Ok(None);
                                }
                                let Some(rt) = self.runtime.as_ref() else {
                                    self.set_status("PerturbRuntime not available", true);
                                    return Ok(None);
                                };
                                let mut targets: Vec<PerturbTarget> =
                                    Vec::with_capacity(self.perturb_targets.len());
                                let csv_part = self.csv_indices_slice();
                                for gene in &self.perturb_targets {
                                    let Ok(base) = single_perturb_target(
                                        gene.as_str(),
                                        self.desired_expr,
                                        &rt.gene_names,
                                    ) else {
                                        self.set_status(format!("Invalid gene: {gene}"), true);
                                        return Ok(None);
                                    };
                                    let type_rows = self
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
                                    let cell_indices =
                                        merge_csv_and_type_cell_indices(csv_part, type_rows);
                                    if let Some(ref v) = cell_indices {
                                        if v.is_empty() {
                                            self.set_status(
                                                "No cells match CSV column and cell-type scope",
                                                true,
                                            );
                                            return Ok(None);
                                        }
                                    }
                                    targets.push(PerturbTarget {
                                        gene: base.gene,
                                        desired_expr: base.desired_expr,
                                        cell_indices,
                                    });
                                }
                                let mut cfg: PerturbConfig = rt.perturb_cfg.clone();
                                cfg.n_propagation = self.n_propagation;
                                let capture_timings = self.pending_verbose;
                                let scopes = self.target_cell_scopes.clone();
                                let genes = self.perturb_targets.clone();
                                let joint_csv_summary = self
                                    .cells_csv_path
                                    .as_ref()
                                    .zip(self.cells_csv_selected_column_name())
                                    .map(|(p, col)| {
                                        let n_cells_per_target_gene: HashMap<String, usize> =
                                            targets
                                                .iter()
                                                .map(|t| {
                                                    let n = t
                                                        .cell_indices
                                                        .as_ref()
                                                        .map(|v| v.len())
                                                        .unwrap_or(rt.obs_names.len());
                                                    (t.gene.clone(), n)
                                                })
                                                .collect();
                                        JointCellsCsvExportSummary {
                                            path: p.display().to_string(),
                                            column: col,
                                            n_cells_per_target_gene,
                                        }
                                    });
                                let id = self.next_job_id;
                                self.next_job_id = self.next_job_id.wrapping_add(1);
                                let label = if genes.len() <= 4 {
                                    genes.join(", ")
                                } else {
                                    format!("{} genes", genes.len())
                                };
                                self.job_queue.push_back(PerturbJobSpec {
                                    id,
                                    label,
                                    genes,
                                    targets,
                                    config: cfg,
                                    scopes,
                                    desired_expr: self.desired_expr,
                                    n_propagation: self.n_propagation,
                                    joint_csv_summary,
                                    capture_timings,
                                });
                                self.set_status(
                                    format!(
                                        "Queued job #{id} ({} pending, {} running)",
                                        self.job_queue.len(),
                                        self.active_jobs.len()
                                    ),
                                    false,
                                );
                                self.drain_job_queue(tx_bg);
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
        let genes_label = out.genes.join(", ");
        let mut lines = vec![
            format!("Job {} · Genes: {genes_label}", out.job_id),
            format!("desired_expr: {}", out.desired_expr),
            format!("n_propagation: {}", out.n_propagation),
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

        if let Some(rt) = self.runtime.as_ref() {
            for gene in &out.genes {
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
        } else {
            lines.push("Δ per-gene summary skipped (runtime unavailable)".into());
        }

        if let Some(t) = out.timings.as_ref() {
            lines.push("Per-step timings:".into());
            for (label, d) in &t.entries {
                lines.push(format!("  {label}: {d:?}"));
            }
        }

        lines.push("".into());
        lines.push("Alt+PgUp / Alt+PgDn: scroll · Ctrl+L: clear this panel".into());
        lines
    }
}
