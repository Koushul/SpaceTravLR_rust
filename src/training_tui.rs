use crate::adata_terminal_scatter::{
    self as adata_scatter, ratatui_color_for_cell_type_label, sorted_unique_labels_from_counts,
};
use crate::training_hud::{TrainingHud, TrainingHudState};
use crate::tui_theme::TuiColors;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, LineGauge, List, ListItem, Paragraph, Wrap};
use std::cell::RefCell;
use std::collections::HashSet;
use std::io::{self, Write, stdout};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
/// Best / worst gene counts in the LASSO R² perf panel (paired columns).
const PERF_R2_LEADERBOARD_LEN: usize = 10;

const GENE_PROGRESS_LINE_SET: symbols::line::Set = symbols::line::Set {
    horizontal: "█",
    ..symbols::line::THICK
};

const HARDWARE_POLL_INTERVAL: Duration = Duration::from_secs(3 * 60);

// ── Rocket ────────────────────────────────────────────────────────────────────
// Compact ASCII rocket — art is ~14 display columns, centered in the panel.

const ROCKET_PANEL_W: u16 = 20;
const WINDOW_IDX: usize = 2;

const BODY_TEXT: [&str; 8] = [
    "      /\\      ",
    "     /  \\     ",
    "    / ** \\    ",
    "    |    |    ",
    "    |    |    ",
    "   /| || |\\   ",
    "  / |____| \\  ",
    "      ||      ",
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

const ROCKET_BODY_FIRE_LINES: usize = 8 + 2;
const SHEEP: &str = "🐑\u{FE0F}";
const ROCKET_HEADER_LINES: usize = 2;
const ROCKET_MIN_TOP_MARGIN: usize = 1;
const ROCKET_MIN_BOTTOM_MARGIN: usize = 1;
const ROCKET_MIN_STARS: usize = 3;
const SHEEP_MAX_SIMULTANEOUS: usize = 12;
const SHEEP_RETAIN: Duration = Duration::from_secs(4);
/// Falling sheep on the rocket panel (demo). Set `true` to re-enable.
const SHEEP_DROP_ENABLED: bool = false;

fn rocket_line_centered(spans: Vec<Span<'static>>, inner_w: usize) -> Line<'static> {
    let w: usize = spans.iter().map(Span::width).sum();
    let left = inner_w.saturating_sub(w) / 2;
    let mut v = Vec::with_capacity(spans.len() + 1);
    v.push(Span::raw(" ".repeat(left)));
    v.extend(spans);
    Line::from(v)
}

fn rocket_vertical_pad(gene_ratio: f64, inner_h: usize) -> usize {
    let ratio = gene_ratio.clamp(0.0, 1.0);
    let eased = 1.0 - (1.0 - ratio).powi(2);
    let core = ROCKET_BODY_FIRE_LINES
        + ROCKET_MIN_TOP_MARGIN
        + ROCKET_MIN_BOTTOM_MARGIN
        + ROCKET_MIN_STARS;
    let max_top_extra = inner_h.saturating_sub(core);
    ROCKET_MIN_TOP_MARGIN + ((1.0 - eased) * max_top_extra as f64).round() as usize
}

fn rocket_lines(
    frame: usize,
    gene_ratio: f64,
    gene_pct: u32,
    inner_w: usize,
    inner_h: usize,
    now: Instant,
    falling_sheep: &[Instant],
    pal: TuiColors,
) -> Vec<Line<'static>> {
    let f = frame % 4;
    let shimmer = (frame / 3) % BODY_TEXT.len();
    let win_c = if frame % 8 < 4 {
        pal.grape
    } else {
        Color::Rgb(180, 150, 230)
    };
    let body_line_fg = [
        pal.sky,
        pal.sky,
        pal.sky,
        pal.title,
        pal.title,
        pal.lilac,
        pal.lilac,
        pal.grape,
    ];

    let canvas_h = inner_h.saturating_sub(ROCKET_HEADER_LINES);
    let canvas_h =
        canvas_h.max(ROCKET_BODY_FIRE_LINES + ROCKET_MIN_TOP_MARGIN + ROCKET_MIN_BOTTOM_MARGIN);

    let top_pad = rocket_vertical_pad(gene_ratio, canvas_h)
        .min(canvas_h.saturating_sub(ROCKET_BODY_FIRE_LINES + ROCKET_MIN_BOTTOM_MARGIN));
    let stars_h = canvas_h
        .saturating_sub(ROCKET_BODY_FIRE_LINES + top_pad + ROCKET_MIN_BOTTOM_MARGIN)
        .max(ROCKET_MIN_STARS);

    let mut lines =
        Vec::with_capacity(ROCKET_HEADER_LINES + top_pad + ROCKET_BODY_FIRE_LINES + stars_h + 2);

    lines.push(rocket_line_centered(
        vec![Span::styled(
            "Status",
            Style::default().fg(pal.grape).add_modifier(Modifier::BOLD),
        )],
        inner_w,
    ));
    lines.push(rocket_line_centered(
        vec![Span::styled(
            format!("{gene_pct}%"),
            Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
        )],
        inner_w,
    ));

    for _ in 0..top_pad {
        lines.push(Line::from(Span::raw(" ".repeat(inner_w))));
    }

    for (i, text) in BODY_TEXT.iter().enumerate() {
        if i == WINDOW_IDX {
            let base = body_line_fg[i];
            lines.push(rocket_line_centered(
                vec![
                    Span::styled("    / ", Style::default().fg(base)),
                    Span::styled(
                        "◆◆",
                        Style::default().fg(win_c).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" \\    ", Style::default().fg(base)),
                ],
                inner_w,
            ));
        } else {
            let base = body_line_fg[i];
            let c = if i == shimmer {
                brighten(base, 35)
            } else {
                base
            };
            let mut s = Style::default().fg(c);
            if i == shimmer {
                s = s.add_modifier(Modifier::BOLD);
            }
            lines.push(rocket_line_centered(vec![Span::styled(*text, s)], inner_w));
        }
    }

    for (ri, text) in FIRE[f].iter().enumerate() {
        let (r, g, b) = FIRE_RGB[f][ri];
        lines.push(rocket_line_centered(
            vec![Span::styled(
                *text,
                Style::default()
                    .fg(Color::Rgb(r, g, b))
                    .add_modifier(Modifier::BOLD),
            )],
            inner_w,
        ));
    }

    for row in 0..stars_h {
        let idx = (row + frame / 4) % STARFIELD.len();
        let v = 75 + ((row * 13 + frame * 3) % 55) as u8;
        lines.push(rocket_line_centered(
            vec![Span::styled(
                STARFIELD[idx],
                Style::default().fg(Color::Rgb(v, v.saturating_sub(10), v.saturating_add(25))),
            )],
            inner_w,
        ));
    }

    apply_rocket_sheep_overlays(&mut lines, inner_w, top_pad, now, falling_sheep, pal);

    lines
}

fn apply_rocket_sheep_overlays(
    lines: &mut Vec<Line<'static>>,
    inner_w: usize,
    top_pad: usize,
    now: Instant,
    falling_sheep: &[Instant],
    pal: TuiColors,
) {
    if !SHEEP_DROP_ENABLED {
        return;
    }
    if falling_sheep.is_empty() || lines.is_empty() {
        return;
    }
    let sheep_w = SHEEP.width();
    if sheep_w == 0 || inner_w <= sheep_w {
        return;
    }
    let first_row_below_rocket = ROCKET_HEADER_LINES + top_pad + ROCKET_BODY_FIRE_LINES;
    if first_row_below_rocket >= lines.len() {
        return;
    }

    for &started in falling_sheep {
        let age = now.checked_duration_since(started).unwrap_or(Duration::ZERO);
        let t = age.as_secs_f64();
        let fall_rows = ((t * 10.0).powf(1.35)).floor() as usize;
        let line_idx = first_row_below_rocket + fall_rows;
        if line_idx >= lines.len() {
            continue;
        }
        let center = (inner_w / 2) as i32;
        let half = (sheep_w / 2) as i32;
        let max_col = inner_w.saturating_sub(sheep_w) as i32;
        let col = (center - half).clamp(0, max_col) as usize;

        let mut style = Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD);
        if t > 0.9 {
            style = Style::default().fg(pal.value).add_modifier(Modifier::DIM);
        }
        if t > 1.5 {
            style = Style::default().fg(pal.muted).add_modifier(Modifier::DIM);
        }

        lines[line_idx] = Line::from(vec![
            Span::raw(" ".repeat(col)),
            Span::styled(SHEEP, style),
            Span::raw(" ".repeat(inner_w.saturating_sub(col + sheep_w))),
        ]);
    }
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

// ── Workers: multi-column (display-width–fixed cells so │ stays aligned) ─────
const GENE_DISP: usize = 16;
const STAT_DISP: usize = 38;
const GENE_STAT_SEP: &str = " · ";

fn pad_or_trunc_display(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let w = s.width();
    if w <= width {
        let pad = width - w;
        return format!("{}{}", s, " ".repeat(pad));
    }
    let mut out = String::new();
    let mut used = 0usize;
    let el = '…';
    let el_w = el.width().unwrap_or(1);
    let budget = width.saturating_sub(el_w);
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if used + cw > budget {
            break;
        }
        out.push(ch);
        used += cw;
    }
    out.push(el);
    let rem = width.saturating_sub(out.width());
    if rem > 0 {
        out.push_str(&" ".repeat(rem));
    }
    out
}

fn pad_left_trunc_display(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let w = s.width();
    if w <= width {
        let pad = width - w;
        return format!("{}{}", " ".repeat(pad), s);
    }
    let mut out = String::new();
    let mut used = 0usize;
    let el = '…';
    let el_w = el.width().unwrap_or(1);
    let budget = width.saturating_sub(el_w);
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if used + cw > budget {
            break;
        }
        out.push(ch);
        used += cw;
    }
    out.push(el);
    let rem = width.saturating_sub(out.width());
    if rem > 0 {
        out.push_str(&" ".repeat(rem));
    }
    out
}

fn workers_in_columns(
    active: &[(&String, &String)],
    content_w: usize,
    lasso_ct: &std::collections::HashMap<String, (usize, usize)>,
    pal: TuiColors,
) -> Vec<ListItem<'static>> {
    if active.is_empty() {
        return vec![ListItem::new(Line::from(Span::styled(
            "  ·  idle  ·",
            Style::default().fg(pal.muted),
        )))];
    }
    let prefix_w = "✿ ".width();
    let mid_w = GENE_STAT_SEP.width();
    let sep_w = " │ ".width();
    let entry_w = prefix_w + GENE_DISP + mid_w + STAT_DISP;
    let n_cols = ((content_w + sep_w) / (entry_w + sep_w)).max(1);
    active
        .chunks(n_cols)
        .map(|chunk| {
            let mut spans: Vec<Span<'static>> = Vec::new();
            for (i, (gene, status)) in chunk.iter().enumerate() {
                if i > 0 {
                    spans.push(Span::styled(" │ ", Style::default().fg(pal.muted)));
                }
                let pc = if status.contains("export") {
                    pal.c_wrote
                } else if status.contains("lasso") || status.contains("cnn") {
                    pal.grape
                } else if status.contains("fail") {
                    pal.c_fail
                } else if status.contains("skip") {
                    pal.c_skip
                } else {
                    pal.lilac
                };
                let status_line: String = if let Some(&(done, total)) = lasso_ct.get(gene.as_str())
                {
                    if total > 0 {
                        format!("{} · {}/{}", status, done, total)
                    } else {
                        (*status).clone()
                    }
                } else {
                    (*status).clone()
                };
                spans.push(Span::styled("✿ ", Style::default().fg(pal.label)));
                spans.push(Span::styled(
                    pad_left_trunc_display(gene.as_str(), GENE_DISP),
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(GENE_STAT_SEP, Style::default().fg(pal.muted)));
                spans.push(Span::styled(
                    pad_or_trunc_display(&status_line, STAT_DISP),
                    Style::default().fg(pc),
                ));
            }
            ListItem::new(Line::from(spans))
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────
/// Display width for `lbl("SRC  ")` / `lbl("OUT  ")` (used to indent path continuations).
const PATH_LABEL_COLS: usize = 5;
const PATH_CONT_INDENT: &str = "     ";

/// Break `path` into lines of at most `max_width` display columns, preferring `/` and `\` breaks.
fn wrap_full_path(path: &str, max_width: usize) -> Vec<String> {
    if max_width < 2 {
        return vec![path.to_string()];
    }
    if path.width() <= max_width {
        return vec![path.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut rest = path;
    while !rest.is_empty() {
        if rest.width() <= max_width {
            lines.push(rest.to_string());
            break;
        }
        let mut acc_w = 0usize;
        let mut end_byte = 0usize;
        let mut last_sep_byte: Option<usize> = None;
        for (i, ch) in rest.char_indices() {
            let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            if acc_w + cw > max_width {
                break;
            }
            acc_w += cw;
            end_byte = i + ch.len_utf8();
            if ch == '/' || ch == '\\' {
                last_sep_byte = Some(end_byte);
            }
        }
        let split = if let Some(sb) = last_sep_byte.filter(|&b| b > 0) {
            sb
        } else if end_byte > 0 {
            end_byte
        } else {
            rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1)
        };
        lines.push(rest[..split].to_string());
        rest = &rest[split..];
    }
    lines
}

fn pad_label_to(s: &str, cols: usize) -> String {
    let w = s.width();
    if w >= cols {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(cols - w))
    }
}

fn break_long_word(s: &str, max_width: usize) -> Vec<String> {
    if max_width < 1 {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if cur_w + cw > max_width && !cur.is_empty() {
            out.push(cur);
            cur = String::new();
            cur_w = 0;
        }
        cur.push(ch);
        cur_w += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        vec![String::new()]
    } else {
        out
    }
}

/// Word-wrap to display width `max_width` (spaces; long tokens hard-broken).
fn wrap_words(text: &str, max_width: usize) -> Vec<String> {
    if max_width < 2 {
        return vec![text.to_string()];
    }
    if text.width() <= max_width {
        return vec![text.to_string()];
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for w in words {
        let ww = w.width();
        if ww > max_width {
            if !cur.is_empty() {
                lines.push(cur);
                cur = String::new();
                cur_w = 0;
            }
            lines.extend(break_long_word(w, max_width));
            continue;
        }
        let add = if cur.is_empty() { ww } else { ww + 1 };
        if cur_w + add > max_width && !cur.is_empty() {
            lines.push(cur);
            cur = String::new();
            cur_w = 0;
        }
        if !cur.is_empty() {
            cur.push(' ');
            cur_w += 1;
        }
        cur.push_str(w);
        cur_w += ww;
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

fn append_cfg_wrapped_lines(
    out: &mut Vec<Line<'static>>,
    label_w: usize,
    val_wrap: usize,
    label: &'static str,
    text: &str,
    label_style: Style,
    value_color: Color,
) {
    for (i, chunk) in wrap_words(text, val_wrap).into_iter().enumerate() {
        if i == 0 {
            out.push(Line::from(vec![
                Span::styled(pad_label_to(label, label_w), label_style),
                Span::styled(chunk, Style::default().fg(value_color)),
            ]));
        } else {
            out.push(Line::from(vec![
                Span::raw(" ".repeat(label_w)),
                Span::styled(chunk, Style::default().fg(value_color)),
            ]));
        }
    }
}

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

fn scan_output_metrics(
    dir: &str,
    active_local_genes: &HashSet<String>,
) -> (u64, usize, usize, usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (0, 0, 0, 0);
    };
    let mut bytes = 0u64;
    let mut n_files = 0usize;
    let mut disk_done = 0usize;
    let mut external_locks = 0usize;
    for e in entries.flatten() {
        let Ok(m) = e.metadata() else {
            continue;
        };
        if !m.is_file() {
            continue;
        }
        bytes += m.len();
        n_files += 1;
        let name = e.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.ends_with(".feather") || name.ends_with(".orphan") {
            disk_done += 1;
        } else if name.ends_with(".lock") {
            let stem = name.strip_suffix(".lock").unwrap_or(name);
            if !active_local_genes.contains(stem) {
                external_locks += 1;
            }
        }
    }
    (bytes, n_files, disk_done, external_locks)
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

fn format_duration_human(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "—".to_string();
    }
    let s = secs.round().max(1.0) as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        let m = s / 60;
        let r = s % 60;
        if r == 0 {
            format!("{m}m")
        } else {
            format!("{m}m {r}s")
        }
    } else {
        let h = s / 3600;
        let rem = s % 3600;
        let m = rem / 60;
        if m == 0 {
            format!("{h}h")
        } else {
            format!("{h}h {m}m")
        }
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
    train_backend: &str,
    train_device_detail: &str,
    max_chars: usize,
) -> String {
    let logical = logical_core_count(sys);
    let physical = sys.physical_core_count();
    let cpu = cpu_brand_label(sys);

    let cores = match physical {
        Some(p) if p > 0 => format!("{logical} logical / {p} physical cores"),
        _ => format!("{logical} logical cores"),
    };

    let s = format!("{cores}  ·  CPU: {cpu}  ·  train {train_backend}: {train_device_detail}");
    truncate_label(&s, max_chars)
}

fn fmt_r2_fixed(r: f64) -> String {
    if r.is_finite() {
        format!("{:>6.2}", r)
    } else {
        format!("{:>6}", "—")
    }
}

fn fmt_lasso_float(x: f64) -> String {
    if x.is_finite() {
        format!("{:.3e}", x)
    } else {
        "—".to_string()
    }
}

fn perf_r2_columns(inner_w: usize) -> (usize, usize) {
    const MID: usize = 2;
    const R2_COL: usize = 7;
    const MOD_COL: usize = 5;
    let w = inner_w.max(MID + 2);
    let half = (w - MID) / 2;
    let gene_w = half.saturating_sub(R2_COL + MOD_COL);
    (half, gene_w)
}

fn fmt_n_mod(n: usize) -> String {
    format!("{:>5}", n)
}

fn rule_line(inner_w: usize, pal: TuiColors) -> Line<'static> {
    let n = inner_w.clamp(8, 128);
    Line::from(Span::styled("─".repeat(n), Style::default().fg(pal.muted)))
}

fn build_perf_panel_lines(st: &TrainingHudState, inner_w: usize, pal: TuiColors) -> Vec<Line<'static>> {
    let n_genes = st.gene_r2_mean.len();
    if n_genes == 0 {
        return vec![Line::from(Span::styled(
            "  ·  no R² yet  ·",
            Style::default().fg(pal.muted),
        ))];
    }

    let (half, gene_w) = perf_r2_columns(inner_w);

    let mut v = st.gene_r2_mean.clone();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_n: Vec<(String, f64, usize)> =
        v.iter().take(PERF_R2_LEADERBOARD_LEN).cloned().collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let bot_n: Vec<(String, f64, usize)> =
        v.iter().take(PERF_R2_LEADERBOARD_LEN).cloned().collect();

    let mut lines: Vec<Line<'static>> =
        Vec::with_capacity(2 + PERF_R2_LEADERBOARD_LEN.min(n_genes));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<hw$}", "▲ best R²", hw = half),
            Style::default().fg(pal.c_topr2).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default().fg(pal.muted)),
        Span::styled(
            format!("{:<hw$}", "▼ worst R²", hw = half),
            Style::default().fg(pal.c_botr2).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(rule_line(inner_w, pal));
    for ((g_hi, r_hi, m_hi), (g_lo, r_lo, m_lo)) in top_n.into_iter().zip(bot_n.into_iter()) {
        let g_hi_s = truncate_label(&g_hi, gene_w);
        let g_lo_s = truncate_label(&g_lo, gene_w);
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<gw$}", g_hi_s, gw = gene_w),
                Style::default().fg(pal.title),
            ),
            Span::styled(fmt_n_mod(m_hi), Style::default().fg(pal.muted)),
            Span::styled(
                fmt_r2_fixed(r_hi),
                Style::default().fg(pal.c_topr2).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default().fg(pal.muted)),
            Span::styled(
                format!("{:<gw$}", g_lo_s, gw = gene_w),
                Style::default().fg(pal.title),
            ),
            Span::styled(fmt_n_mod(m_lo), Style::default().fg(pal.muted)),
            Span::styled(
                fmt_r2_fixed(r_lo),
                Style::default().fg(pal.c_botr2).add_modifier(Modifier::BOLD),
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
    let y = area
        .y
        .saturating_add(area.height.saturating_sub(height) / 2);
    Rect::new(x, y, width, height)
}

fn prompt_aborts(key: &event::KeyEvent) -> bool {
    key.code == KeyCode::Esc
        || (key.modifiers.contains(KeyModifiers::SHIFT)
            && matches!(key.code, KeyCode::Char('q' | 'Q')))
}

/// When no `.h5ad` path is configured: compact centered prompts for AnnData then output directory.
/// **Enter** confirms each step. **Esc** or **Shift+Q** exits without starting training.
pub fn run_dataset_paths_prompt(
    default_output_dir: &str,
) -> anyhow::Result<Option<(String, String)>> {
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
            let pal = TuiColors::default_palette();
            let area = f.area();
            f.render_widget(
                Block::default().style(Style::default().bg(pal.bg)),
                area,
            );

            let popup_w = ((area.width * 55) / 100)
                .clamp(48, 72)
                .min(area.width.saturating_sub(4));
            let popup_h = 13u16.min(area.height.saturating_sub(2));
            let popup_area = centered_rect(area, popup_w, popup_h);

            let (title, help): (&str, Vec<Line>) = if step == 0 {
                (
                    " AnnData (.h5ad) ",
                    vec![
                        Line::from(Span::styled(
                            "Make sure `cell_type` is in .obs and `spatial` is in .obsm",
                            Style::default().fg(pal.muted),
                        )),
                        Line::from(Span::styled(
                            "Enter path to .h5ad. Press shift+q to quit",
                            Style::default().fg(pal.value),
                        )),
                    ],
                )
            } else {
                let adata_disp = if adata_input.chars().count() > 48 {
                    format!("{}…", adata_input.chars().take(45).collect::<String>())
                } else {
                    adata_input.clone()
                };
                (
                    " Output directory ",
                    vec![
                        Line::from(Span::styled(
                            format!("AnnData: {adata_disp}"),
                            Style::default().fg(pal.muted),
                        )),
                        Line::from(Span::styled(
                            "Directory for *_betadata.feather (created if missing).",
                            Style::default().fg(pal.value),
                        )),
                        Line::from(Span::styled(
                            "Press shift+q to quit · Enter confirm",
                            Style::default().fg(pal.muted),
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
                .border_style(Style::default().fg(pal.tel_bord))
                .title(Span::styled(
                    title,
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
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

            let help_w = Paragraph::new(help).wrap(Wrap { trim: true }).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(pal.outer_bord)),
            );
            f.render_widget(help_w, chunks[0]);

            let path_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pal.sky))
                .title(Span::styled(" path ", Style::default().fg(pal.label)));
            let path_para = Paragraph::new(Line::from(vec![Span::styled(
                if input_ref.is_empty() {
                    " "
                } else {
                    input_ref.as_str()
                },
                Style::default().fg(pal.title),
            )]))
            .block(path_block);
            f.render_widget(path_para, chunks[1]);

            let msg = err_line.as_deref().unwrap_or(" ");
            let err_c = if err_line.is_some() {
                pal.c_fail
            } else {
                pal.muted
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
                        if let Ok(s) = crate::config::default_output_dir_for_adata_path(Path::new(
                            &adata_input,
                        )) {
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

fn is_theme_cycle_key(key: &event::KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
        && matches!(key.code, KeyCode::Char('t' | 'T'))
        && !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
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
    let mut disk_genes_done: usize = 0;
    let mut external_workers: usize = 0;

    if let Ok(st) = hud.lock() {
        let active: HashSet<String> = st.active_genes.keys().cloned().collect();
        (dir_bytes, dir_files, disk_genes_done, external_workers) =
            scan_output_metrics(&st.output_dir, &active);
    }

    let mut dashboard_exit = TrainingDashboardExit::Completed;
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "operator".to_string());

    let perf_cell: RefCell<(u64, Instant, usize, usize, Vec<Line<'static>>)> = RefCell::new((
        0,
        Instant::now() - Duration::from_secs(3600),
        0,
        0,
        vec![Line::from(Span::styled(
            "  ·  no R² yet  ·",
            Style::default().fg(TuiColors::default_palette().muted),
        ))],
    ));

    const ETA_RATE_REFRESH: Duration = Duration::from_secs(3);
    let eta_rate_cache: RefCell<(Instant, usize, usize, usize, String, String)> = RefCell::new((
        Instant::now() - Duration::from_secs(3600),
        0usize,
        0usize,
        0usize,
        "—".to_string(),
        "—".to_string(),
    ));
    let scatter_panel_cache: RefCell<Option<(String, u16, u16, usize, Vec<Line<'static>>)>> =
        RefCell::new(None);

    let mut prev_genes_rounds_heartbeat = 0usize;
    let mut heart_beat_until: Option<Instant> = None;
    let mut prev_genes_rounds_sheep = 0usize;
    let mut falling_sheep: Vec<Instant> = Vec::new();
    let mut theme_slot = TuiColors::default_theme_slot();

    'dashboard: loop {
        if last_sys.elapsed() > HARDWARE_POLL_INTERVAL {
            sys.refresh_cpu_all();
            sys.refresh_memory();
            last_sys = Instant::now();
        }
        if last_dir_scan.elapsed() > Duration::from_secs(2) {
            if let Ok(st) = hud.lock() {
                let active: HashSet<String> = st.active_genes.keys().cloned().collect();
                (dir_bytes, dir_files, disk_genes_done, external_workers) =
                    scan_output_metrics(&st.output_dir, &active);
            }
            last_dir_scan = Instant::now();
        }
        if event::poll(Duration::from_millis(40))? {
            loop {
                if let Event::Key(key) = event::read()? {
                    if is_shift_q(&key) {
                        if let Ok(st) = hud.lock() {
                            st.cancel_requested.store(true, Ordering::Relaxed);
                        }
                        dashboard_exit = TrainingDashboardExit::ForceQuit;
                        break 'dashboard;
                    }
                    if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                        if let Ok(st) = hud.lock() {
                            st.cancel_requested.store(true, Ordering::Relaxed);
                        }
                    }
                    if is_theme_cycle_key(&key) {
                        theme_slot = TuiColors::advance_slot(theme_slot);
                    }
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        let done = hud.lock().map(|g| g.finished.is_some()).unwrap_or(true);
        let frame = (t0.elapsed().as_millis() / 200) as usize;

        let now_heartbeat = Instant::now();
        if let Ok(st) = hud.lock() {
            let rounds = st.genes_rounds;
            if rounds > prev_genes_rounds_heartbeat {
                heart_beat_until = Some(now_heartbeat + Duration::from_millis(220));
                prev_genes_rounds_heartbeat = rounds;
            } else if rounds < prev_genes_rounds_heartbeat {
                prev_genes_rounds_heartbeat = rounds;
            }
            if rounds > prev_genes_rounds_sheep {
                if SHEEP_DROP_ENABLED {
                    for _ in 0..(rounds - prev_genes_rounds_sheep) {
                        if falling_sheep.len() < SHEEP_MAX_SIMULTANEOUS {
                            falling_sheep.push(now_heartbeat);
                        }
                    }
                }
                prev_genes_rounds_sheep = rounds;
            } else if rounds < prev_genes_rounds_sheep {
                prev_genes_rounds_sheep = rounds;
            }
        }
        falling_sheep.retain(|t| now_heartbeat.saturating_duration_since(*t) < SHEEP_RETAIN);
        let heart_peak = heart_beat_until.is_some_and(|deadline| now_heartbeat < deadline);

        let now_rocket = now_heartbeat;
        let sheep_snapshot: Vec<Instant> = falling_sheep.clone();
        terminal.draw(|f| {
            let area = f.area();
            let pal = TuiColors::resolve(theme_slot);
            let bg = Style::default().bg(pal.bg);
            f.render_widget(Block::default().style(bg), area);

            let Ok(st) = hud.lock() else { return };

            let cpu_pct = sys.global_cpu_usage();
            let used_mem = sys.used_memory();
            let total_mem = sys.total_memory().max(1);
            let mem_pct = (used_mem as f64 / total_mem as f64) * 100.0;
            let elapsed = st.elapsed_secs();

            let outer = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pal.outer_bord).add_modifier(Modifier::DIM))
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
                    Constraint::Length(2),
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
            let status_c = if st.should_cancel() {
                pal.c_fail
            } else {
                pal.value
            };
            let hw_w = vchunks[0].width.saturating_sub(2) as usize;
            let hw_line = build_machine_hardware_line(
                &sys,
                st.run_config.compute_backend.as_str(),
                st.run_config.compute_device_detail.as_str(),
                hw_w.max(12),
            );

            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled(
                            " ✿ ",
                            Style::default().fg(pal.grape).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "SpaceTravLR",
                            Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(pal.muted)),
                        Span::styled(
                            format!("@{}", username),
                            Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(pal.muted)),
                        Span::styled(
                            format_t(elapsed),
                            Style::default().fg(pal.value).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("  ·  ", Style::default().fg(pal.muted)),
                        Span::styled(mode, Style::default().fg(pal.sky)),
                        Span::styled("  ·  ", Style::default().fg(pal.muted)),
                        Span::styled(status_txt, Style::default().fg(status_c)),
                        Span::styled(" ✿", Style::default().fg(pal.grape)),
                    ]),
                    Line::from(Span::styled(hw_line, Style::default().fg(pal.muted))),
                ]),
                vchunks[0],
            );

            // ── Main: [telemetry + workers] | rocket ──────────────────────────
            let hchunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(1), Constraint::Length(ROCKET_PANEL_W)])
                .split(vchunks[1]);

            let left = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(22), Constraint::Min(4)])
                .split(hchunks[0]);
            let top_panels = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(33),
                    Constraint::Percentage(34),
                    Constraint::Percentage(33),
                ])
                .split(left[0]);

            let now_er = Instant::now();
            {
                let mut erc = eta_rate_cache.borrow_mut();
                let n_times = st.gene_train_times.len();
                let rounds = st.genes_rounds;
                let total_g = st.total_genes;
                let refresh = now_er.duration_since(erc.0) >= ETA_RATE_REFRESH
                    || n_times != erc.1
                    || rounds != erc.2
                    || total_g != erc.3;
                if refresh {
                    erc.4 = match st.eta_secs() {
                        Some(s) if s <= 0.0 => "0s".to_string(),
                        Some(s) => format_duration_human(s),
                        None => "—".to_string(),
                    };
                    erc.5 = match st.parallel_wall_secs_per_gene() {
                        Some(avg) => format!("{:.1}s/gene", avg),
                        None => "—".to_string(),
                    };
                    erc.0 = now_er;
                    erc.1 = n_times;
                    erc.2 = rounds;
                    erc.3 = total_g;
                }
            }
            let (eta_s, rate_s) = {
                let erc = eta_rate_cache.borrow();
                (erc.4.clone(), erc.5.clone())
            };

            let work_area = left[1];
            let work_row = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Min(10),
                    Constraint::Min(38),
                    Constraint::Min(16),
                ])
                .split(work_area);

            let sep = || Span::styled("  ·  ", Style::default().fg(pal.muted));
            let lbl = |s: &'static str| Span::styled(s, Style::default().fg(pal.label));
            let val = |s: String, c: Color| Span::styled(s, Style::default().fg(c));

            let rc = &st.run_config;
            let cfg_panel = top_panels[0];
            let cfg_inner_w = cfg_panel.width.saturating_sub(2) as usize;
            const CFG_LABEL_W: usize = 11;
            let cfg_val_wrap = cfg_inner_w.saturating_sub(CFG_LABEL_W).max(6);
            let cfg_lbl_style = Style::default().fg(pal.label);
            let mut cfg_lines: Vec<Line<'static>> = Vec::new();

            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "CONFIG",
                &rc.config_source,
                cfg_lbl_style,
                pal.muted,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "RUNTIME",
                &format!("{}  ·  {}", rc.compute_backend, rc.compute_device_detail),
                cfg_lbl_style,
                pal.value,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "LAYER",
                &rc.layer,
                cfg_lbl_style,
                pal.sky,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "OBS",
                &rc.cluster_annot,
                cfg_lbl_style,
                pal.lilac,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "SPATIAL",
                &format!(
                    "r={:.1}  ·  dim={}  ·  contact={:.1}  ·  wl={:.3}",
                    rc.spatial_radius,
                    rc.spatial_dim,
                    rc.contact_distance,
                    rc.weighted_ligand_scale_factor,
                ),
                cfg_lbl_style,
                pal.value,
            );

            cfg_lines.push(Line::default());

            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "LASSO",
                &format!(
                    "l1={}  ·  group={}  ·  n_iter={}  ·  tol={:.1e}",
                    fmt_lasso_float(rc.l1_reg),
                    fmt_lasso_float(rc.group_reg),
                    rc.n_iter,
                    rc.tol,
                ),
                cfg_lbl_style,
                pal.grape,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "TRAIN",
                &format!(
                    "lr={}  ·  score≥{:.2}  ·  ep/gene={}",
                    fmt_lasso_float(rc.learning_rate),
                    rc.score_threshold,
                    rc.epochs_per_gene,
                ),
                cfg_lbl_style,
                pal.sky,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "GRN",
                &format!(
                    "tf_lig≥{:.2}  ·  max_ligands={}",
                    rc.tf_ligand_cutoff, rc.max_ligands
                ),
                cfg_lbl_style,
                pal.muted,
            );

            cfg_lines.push(Line::default());

            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "GENES",
                &rc.gene_selection,
                cfg_lbl_style,
                pal.title,
            );
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "MODE",
                &rc.cnn_training_mode,
                cfg_lbl_style,
                pal.sky,
            );
            let cond_txt = if rc.condition_split == "—" {
                "—  ·  single run".to_string()
            } else {
                format!("obs.{}  ·  multi-run (--condition)", rc.condition_split)
            };
            let cond_c = if rc.condition_split == "—" {
                pal.muted
            } else {
                pal.lilac
            };
            append_cfg_wrapped_lines(
                &mut cfg_lines,
                CFG_LABEL_W,
                cfg_val_wrap,
                "COND",
                &cond_txt,
                cfg_lbl_style,
                cond_c,
            );

            let cfg_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pal.tel_bord))
                .title(Span::styled(
                    " Run config ",
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ))
                .style(bg);
            let cfg_inner = cfg_block.inner(cfg_panel);

            f.render_widget(cfg_block, cfg_panel);
            f.render_widget(
                Paragraph::new(cfg_lines)
                    .wrap(Wrap { trim: true })
                    .style(bg),
                cfg_inner,
            );

            let tel_inner_w = top_panels[1].width.saturating_sub(2) as usize;
            let path_wrap_w = tel_inner_w.saturating_sub(PATH_LABEL_COLS).max(12);

            let mut mission_lines: Vec<Line> = Vec::new();
            for (i, chunk) in wrap_full_path(&st.dataset_path, path_wrap_w)
                .into_iter()
                .enumerate()
            {
                if i == 0 {
                    mission_lines.push(Line::from(vec![lbl("SRC  "), val(chunk, pal.muted)]));
                } else {
                    mission_lines.push(Line::from(vec![
                        Span::raw(PATH_CONT_INDENT),
                        val(chunk, pal.muted),
                    ]));
                }
            }
            mission_lines.push(Line::from(vec![
                lbl("ETA  "),
                val(eta_s, pal.value),
                sep(),
                lbl("RATE  "),
                val(rate_s, pal.sky),
            ]));
            if rc.condition_split != "—" {
                let active = match (
                    st.current_condition_value.as_deref(),
                    st.condition_split_progress,
                ) {
                    (Some(v), Some((i, n))) => format!("{v}  ·  {i}/{n}"),
                    (Some(v), None) => v.to_string(),
                    (None, Some((i, n))) => format!("…  ·  {i}/{n}"),
                    (None, None) => "preparing splits…".to_string(),
                };
                mission_lines.push(Line::from(vec![lbl("ACTIVE  "), val(active, pal.lilac)]));
            }
            mission_lines.push(Line::from(vec![
                lbl("GRID  "),
                val(
                    format!("{} cells  ·  {} clusters", st.n_cells, st.n_clusters),
                    pal.value,
                ),
                sep(),
                lbl("WORKERS  "),
                val(
                    format!("local {}  ·  external {}", st.n_parallel, external_workers),
                    pal.grape,
                ),
            ]));
            mission_lines.push(Line::from(vec![
                lbl("EXPORT  "),
                val(
                    format!(
                        "seed-only {}  ·  CNN {}",
                        st.genes_exported_seed_only, st.genes_exported_cnn
                    ),
                    pal.value,
                ),
            ]));
            mission_lines.push(Line::from(vec![
                lbl("EPOCH  "),
                val(format!("{}/gene", st.epochs_per_gene), pal.lilac),
            ]));
            mission_lines.push(Line::from(vec![
                lbl("CPU  "),
                val(format!("{cpu_pct:5.1}%"), pal.sky),
                sep(),
                lbl("MEM  "),
                val(format!("{mem_pct:5.1}%"), pal.sky),
                sep(),
                lbl("RAM  "),
                val(
                    format!("{}/{} MiB", used_mem / 1024 / 1024, total_mem / 1024 / 1024),
                    pal.muted,
                ),
            ]));
            for (i, chunk) in wrap_full_path(&st.output_dir, path_wrap_w)
                .into_iter()
                .enumerate()
            {
                if i == 0 {
                    mission_lines.push(Line::from(vec![lbl("OUT  "), val(chunk, pal.muted)]));
                } else {
                    mission_lines.push(Line::from(vec![
                        Span::raw(PATH_CONT_INDENT),
                        val(chunk, pal.muted),
                    ]));
                }
            }
            mission_lines.push(Line::from(vec![
                lbl("SIZE  "),
                val(format_bytes(dir_bytes), pal.value),
                sep(),
                lbl("FILES  "),
                val(format!("{}", dir_files), pal.lilac),
            ]));
            f.render_widget(
                Paragraph::new(mission_lines)
                    .wrap(Wrap { trim: true })
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(pal.tel_bord))
                            .title(Span::styled(
                                " Live telemetry ",
                                Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .style(bg),
                top_panels[1],
            );

            {
                let scatter_area = top_panels[2];
                let scatter_block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(pal.tel_bord))
                    .title(Span::styled(
                        if st.is_demo {
                            " Spatial (obsm spatial · cached) "
                        } else {
                            " Spatial (obsm) "
                        },
                        Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                    ))
                    .style(bg);
                let inner = scatter_block.inner(scatter_area);
                let iw = inner.width as usize;
                let ih = inner.height as usize;
                let (cw, ch) = adata_scatter::optimal_square_chart_dims(iw, ih);
                let content_w = cw as u16;
                let content_h = ch as u16;
                let tb = inner.height.saturating_sub(content_h);
                let top_pad = tb / 2;
                let bottom_pad = tb - top_pad;
                let lr = inner.width.saturating_sub(content_w);
                let left_pad = lr / 2;
                let right_pad = lr - left_pad;
                let cache_key = (
                    st.dataset_path.clone(),
                    scatter_area.width,
                    scatter_area.height,
                    st.n_cells,
                );
                let mut borrow = scatter_panel_cache.borrow_mut();
                let reload = match borrow.as_ref() {
                    None => true,
                    Some((p, w, h, n, _)) => {
                        p != &cache_key.0
                            || *w != cache_key.1
                            || *h != cache_key.2
                            || *n != cache_key.3
                    }
                };
                if reload {
                    if st.is_demo {
                        match crate::training_demo::demo_kidney_spatial_scatter_lines_for_tui(cw, ch) {
                            Ok(mut v) => {
                                if v.len() > ch {
                                    v.truncate(ch);
                                }
                                *borrow = Some((
                                    cache_key.0,
                                    cache_key.1,
                                    cache_key.2,
                                    cache_key.3,
                                    v,
                                ));
                            }
                            Err(e) => {
                                *borrow = Some((
                                    cache_key.0,
                                    cache_key.1,
                                    cache_key.2,
                                    cache_key.3,
                                    vec![Line::from(Span::styled(
                                        format!("  ·  demo spatial: {}  ·", e),
                                        Style::default().fg(pal.c_fail),
                                    ))],
                                ));
                            }
                        }
                    } else if st.n_cells == 0 {
                        *borrow = Some((
                            cache_key.0,
                            cache_key.1,
                            cache_key.2,
                            cache_key.3,
                            vec![Line::from(Span::styled(
                                "  ·  loading…  ·",
                                Style::default().fg(pal.muted),
                            ))],
                        ));
                    } else {
                        match adata_scatter::spatial_scatter_lines_for_tui(
                            Path::new(&st.dataset_path),
                            st.run_config.cluster_annot.as_str(),
                            cw,
                            ch,
                            None,
                        ) {
                            Ok(mut v) => {
                                if v.len() > ch {
                                    v.truncate(ch);
                                }
                                *borrow = Some((
                                    cache_key.0,
                                    cache_key.1,
                                    cache_key.2,
                                    cache_key.3,
                                    v,
                                ));
                            }
                            Err(e) => {
                                *borrow = Some((
                                    cache_key.0,
                                    cache_key.1,
                                    cache_key.2,
                                    cache_key.3,
                                    vec![Line::from(Span::styled(
                                        format!("  ·  {}  ·", e),
                                        Style::default().fg(pal.c_fail),
                                    ))],
                                ));
                            }
                        }
                    }
                }
                let scatter_lines = borrow
                    .as_ref()
                    .map(|(_, _, _, _, lines)| lines.clone())
                    .unwrap_or_default();
                drop(borrow);
                let rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(top_pad),
                        Constraint::Length(content_h),
                        Constraint::Length(bottom_pad),
                    ])
                    .split(inner);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Length(left_pad),
                        Constraint::Length(content_w),
                        Constraint::Length(right_pad),
                    ])
                    .split(rows[1]);
                f.render_widget(scatter_block, scatter_area);
                f.render_widget(
                    Paragraph::new(Text::from(scatter_lines))
                        .style(bg)
                        .alignment(Alignment::Center)
                        .wrap(Wrap { trim: false }),
                    cols[1],
                );
            }

            // ── Workers ───────────────────────────────────────────────────────
            let mut active: Vec<(&String, &String)> = st.active_genes.iter().collect();
            active.sort_by_key(|(g, _)| g.as_str());
            let cw = (work_row[0].width as usize).saturating_sub(2);

            let (heart_glyph, heart_style) = if heart_peak {
                ("♥", Style::default().fg(pal.grape).add_modifier(Modifier::BOLD))
            } else {
                ("♡", Style::default().fg(pal.muted))
            };
            let rest_style = Style::default().fg(pal.grape).add_modifier(Modifier::BOLD);

            f.render_widget(
                List::new(workers_in_columns(
                    &active,
                    cw,
                    &st.gene_lasso_cluster_progress,
                    pal,
                ))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(pal.work_bord))
                        .title(Line::from(vec![
                            Span::styled(" ", rest_style),
                            Span::styled(heart_glyph, heart_style),
                            Span::styled(
                                format!(
                                    " ACTIVE WORKERS ({}/{}) ",
                                    st.active_genes.len(),
                                    st.n_parallel
                                ),
                                rest_style,
                            ),
                        ])),
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
                let stale_theme = cell.3 != theme_slot;
                if stale_gen || stale_t || stale_w || stale_theme {
                    cell.4 = build_perf_panel_lines(&st, perf_inner, pal);
                    cell.0 = st.perf_stats_generation;
                    cell.1 = now;
                    cell.2 = perf_inner;
                    cell.3 = theme_slot;
                }
            }
            let perf_snapshot = perf_cell.borrow().4.clone();
            f.render_widget(
                Paragraph::new(perf_snapshot)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(pal.perf_bord))
                            .title(Span::styled(
                                format!(
                                    " ✦ LASSO R²  · n = β cols in feather  · {} best / {} worst ",
                                    PERF_R2_LEADERBOARD_LEN, PERF_R2_LEADERBOARD_LEN
                                ),
                                Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
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
            let sorted_type_labels = sorted_unique_labels_from_counts(&cell_counts);

            let label_w = cell_panel_w.saturating_sub(10).max(3);
            let count_w = cell_panel_w.saturating_sub(label_w).max(1);

            let mut cell_lines: Vec<Line<'static>> = Vec::new();
            if st.n_cells == 0 {
                cell_lines.push(Line::from(Span::styled(
                    "  ·  loading obs…  ·",
                    Style::default().fg(pal.muted),
                )));
            } else if cell_counts.is_empty() {
                cell_lines.push(Line::from(Span::styled(
                    "  ·  no cell-type label column  ·",
                    Style::default().fg(pal.muted),
                )));
            } else {
                for (ct, count) in cell_counts.iter().take(cell_panel_rows) {
                    let ct_disp = truncate_label(ct, label_w);
                    let left = format!("{:<lw$}", ct_disp, lw = label_w);
                    let right = format!("{:>cw$}", count, cw = count_w);
                    let ct_c = ratatui_color_for_cell_type_label(ct, &sorted_type_labels);
                    cell_lines.push(Line::from(vec![
                        Span::styled(
                            left,
                            Style::default()
                                .fg(ct_c)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            right,
                            Style::default().fg(pal.value).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                if cell_counts.len() > cell_panel_rows {
                    cell_lines.push(Line::from(Span::styled(
                        format!("… +{} more", cell_counts.len() - cell_panel_rows),
                        Style::default().fg(pal.muted),
                    )));
                }
            }

            f.render_widget(
                Paragraph::new(cell_lines)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(pal.tel_bord))
                            .title(Span::styled(
                                " Cell Types ",
                                Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                            )),
                    )
                    .style(bg),
                work_row[2],
            );

            let total = st.total_genes.max(1) as u64;
            let pos = if !st.is_demo && Path::new(&st.output_dir).is_dir() {
                disk_genes_done.min(st.total_genes)
            } else {
                st.genes_rounds.min(st.total_genes)
            } as u64;
            let ratio = (pos as f64 / total as f64).clamp(0.0, 1.0);
            let gene_pct = (ratio * 100.0).round().clamp(0.0, 100.0) as u32;

            let rocket_panel = hchunks[1];
            let rocket_inner_w = rocket_panel.width.saturating_sub(2) as usize;
            let rocket_min_h = (ROCKET_HEADER_LINES
                + ROCKET_BODY_FIRE_LINES
                + ROCKET_MIN_TOP_MARGIN
                + ROCKET_MIN_BOTTOM_MARGIN
                + ROCKET_MIN_STARS) as u16;
            let rocket_inner_h = rocket_panel.height.saturating_sub(2).max(rocket_min_h) as usize;

            // ── Rocket ────────────────────────────────────────────────────────
            f.render_widget(
                Paragraph::new(rocket_lines(
                    frame,
                    ratio,
                    gene_pct,
                    rocket_inner_w,
                    rocket_inner_h,
                    now_rocket,
                    &sheep_snapshot,
                    pal,
                ))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(pal.rocket_bord)),
                )
                .style(bg),
                rocket_panel,
            );

            // ── Gene progress (title + stats on border; one row, full-block glyphs)
            let prog_area = vchunks[2];
            let sky_bold = Style::default().fg(pal.sky).add_modifier(Modifier::BOLD);
            let title_bold = Style::default().fg(pal.title).add_modifier(Modifier::BOLD);
            let prog_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pal.sky))
                .title(Line::from(vec![
                    Span::styled(" Gene progress ", sky_bold),
                    Span::styled(" · ", Style::default().fg(pal.muted)),
                    Span::styled(format!("{}/{}", pos, total), title_bold),
                    Span::styled("  ·  ok ", Style::default().fg(pal.muted)),
                    Span::styled(format!("{}", st.genes_done), title_bold),
                    Span::styled("  fail ", Style::default().fg(pal.muted)),
                    Span::styled(format!("{}", st.genes_failed), title_bold),
                    Span::styled("  orphan ", Style::default().fg(pal.muted)),
                    Span::styled(format!("{}", st.genes_orphan), title_bold),
                ]));
            let prog_inner = prog_block.inner(prog_area);
            f.render_widget(prog_block, prog_area);
            let gauge = LineGauge::default()
                .style(bg)
                .filled_style(Style::default().fg(pal.sky).add_modifier(Modifier::BOLD))
                .unfilled_style(Style::default().fg(pal.gauge_empty))
                .filled_symbol(GENE_PROGRESS_LINE_SET.horizontal)
                .unfilled_symbol(symbols::line::THICK.horizontal)
                .label(Line::from(""))
                .ratio(ratio);
            f.render_widget(gauge, prog_inner);

            // ── Footer ────────────────────────────────────────────────────────
            let theme_hint = TuiColors::theme_label(theme_slot);
            let credit_spans = vec![
                Span::styled(
                    "© ",
                    Style::default().fg(pal.sky).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "2026",
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ·  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "Koushul & Ally",
                    Style::default().fg(pal.title).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  @  ", Style::default().fg(pal.muted)),
                Span::styled(
                    "jishnulab.org",
                    Style::default().fg(pal.lilac).add_modifier(Modifier::BOLD),
                ),
            ];
            let footer_bar_w = vchunks[3].width as usize;
            let credit_w: usize = credit_spans.iter().map(Span::width).sum();
            let credit_pad = footer_bar_w.saturating_sub(credit_w);
            let mut credit_line_spans = Vec::with_capacity(1 + credit_spans.len());
            if credit_pad > 0 {
                credit_line_spans.push(Span::raw(" ".repeat(credit_pad)));
            }
            credit_line_spans.extend(credit_spans);
            let credit = Line::from(credit_line_spans);
            let footer = if st.should_cancel() {
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        " Stopping after in-flight genes finish… ",
                        Style::default().fg(pal.c_fail).add_modifier(Modifier::BOLD),
                    )),
                    credit.clone(),
                ])
            } else {
                let mut hint = format!(
                    " q: graceful exit   shift+q: french leave   t: theme ({theme_hint})"
                );
                if st.is_demo && SHEEP_DROP_ENABLED {
                    hint.push_str("   ·   demo: sheep on each gene finish ");
                } else {
                    hint.push(' ');
                }
                Paragraph::new(vec![
                    Line::from(Span::styled(hint, Style::default().fg(pal.muted))),
                    credit,
                ])
            };
            f.render_widget(footer.wrap(Wrap { trim: true }), vchunks[3]);
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
