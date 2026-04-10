use std::collections::HashMap;
use std::path::Path;

use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use colored::Color;
use ndarray::{Array2, s};
use polars::datatypes::DataType;
use polars::prelude::*;
use termplot_rs::ChartContext;

pub const SPATIAL_OBSM_KEYS: &[&str] = &["spatial", "X_spatial", "spatial_loc"];

pub const CELL_TYPE_PALETTE_RGB: [(u8, u8, u8); 13] = [
    (97, 214, 214),
    (214, 97, 214),
    (255, 220, 100),
    (129, 241, 129),
    (97, 160, 255),
    (255, 105, 105),
    (0, 214, 214),
    (214, 0, 214),
    (255, 215, 0),
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (255, 255, 255),
];

pub fn colored_for_cell_type_index(i: usize) -> Color {
    let (r, g, b) = CELL_TYPE_PALETTE_RGB[i % CELL_TYPE_PALETTE_RGB.len()];
    Color::TrueColor { r, g, b }
}

#[cfg(feature = "tui")]
pub fn ratatui_color_for_cell_type_index(i: usize) -> ratatui::style::Color {
    let (r, g, b) = CELL_TYPE_PALETTE_RGB[i % CELL_TYPE_PALETTE_RGB.len()];
    ratatui::style::Color::Rgb(r, g, b)
}

#[cfg(feature = "tui")]
pub fn ratatui_color_for_cell_type_label(label: &str, sorted_unique: &[String]) -> ratatui::style::Color {
    let idx = sorted_unique
        .iter()
        .position(|s| s == label)
        .unwrap_or(0);
    ratatui_color_for_cell_type_index(idx)
}

pub fn sorted_unique_labels_from_counts(counts: &[(String, usize)]) -> Vec<String> {
    let mut u: Vec<String> = counts.iter().map(|(s, _)| s.clone()).collect();
    u.sort();
    u.dedup();
    u
}

fn obsm_xy_f64<B: Backend>(adata: &AnnData<B>, key: &str) -> anyhow::Result<Array2<f64>> {
    if let Ok(Some(arr)) = adata.obsm().get_item::<Array2<f32>>(key) {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(Some(arr)) = adata.obsm().get_item::<Array2<f64>>(key) {
        return Ok(arr);
    }
    anyhow::bail!("obsm[{key:?}] is missing or not a dense f32/f64 matrix")
}

pub fn detect_spatial_obsm_key<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<String> {
    for key in SPATIAL_OBSM_KEYS {
        if let Ok(arr) = obsm_xy_f64(adata, key) {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok((*key).to_string());
            }
        }
    }
    let keys = adata.obsm().keys();
    anyhow::bail!("no usable 2D spatial in obsm (tried {:?}). Keys: {:?}", SPATIAL_OBSM_KEYS, keys)
}

fn obs_column_strings(obs: &DataFrame, name: &str) -> anyhow::Result<Vec<String>> {
    let col = obs.column(name).with_context(|| format!("obs column {name:?}"))?;
    let series = col.as_materialized_series();
    let as_str = series.cast(&DataType::String)?;
    Ok(as_str
        .str()?
        .into_iter()
        .map(|o| o.map(str::to_string).unwrap_or_else(|| "NA".into()))
        .collect())
}

fn assign_colors(labels: &[String]) -> (HashMap<String, Color>, Vec<Color>) {
    let mut uniq: Vec<String> = Vec::new();
    for s in labels {
        if !uniq.iter().any(|u| u == s) {
            uniq.push(s.clone());
        }
    }
    uniq.sort();
    let map: HashMap<String, Color> = uniq
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, colored_for_cell_type_index(i)))
        .collect();
    let per_point: Vec<Color> = labels.iter().map(|l| map[l]).collect();
    (map, per_point)
}

fn chart_size_square_pixels(max_chart_w: usize, max_chart_h: usize) -> (usize, usize) {
    let ch = max_chart_h.min(max_chart_w / 2).max(1);
    let cw = (ch * 2).min(max_chart_w);
    (cw, ch)
}

/// Largest square braille chart that fits in a terminal `inner_w × inner_h` cell rect (equal data aspect; see `chart_size_square_pixels`).
/// `inner_w` / `inner_h` are terminal columns / rows; termplot uses one terminal column per braille cell.
pub fn optimal_square_chart_dims(inner_w: usize, inner_h: usize) -> (usize, usize) {
    let max_chart_w = inner_w.max(1);
    let max_chart_h = inner_h.max(1);
    chart_size_square_pixels(max_chart_w, max_chart_h)
}

fn draw_spatial_scatter_canvas_from_points(
    points: &[(f64, f64)],
    labels: &[String],
    chart_w: usize,
    chart_h: usize,
) -> anyhow::Result<(String, String, HashMap<String, Color>)> {
    anyhow::ensure!(
        points.len() == labels.len(),
        "points (len {}) != labels (len {})",
        points.len(),
        labels.len()
    );
    let n_obs = points.len();
    let (xr, yr) = ChartContext::get_auto_range(&points.to_vec(), 0.04);
    let (legend_map, colors) = assign_colors(labels);

    let mut chart = ChartContext::new(chart_w, chart_h);
    let w_px = chart.canvas.pixel_width();
    let h_px = chart.canvas.pixel_height();
    for i in 0..n_obs {
        if let Some((px, py)) =
            map_to_pixel_equal_aspect(w_px, h_px, points[i].0, points[i].1, xr, yr)
        {
            chart.canvas.set_pixel(px, py, Some(colors[i]));
        }
    }

    let canvas_no_border = chart.canvas.render_with_options(false, None);
    let canvas_with_border = chart.canvas.render();
    Ok((canvas_no_border, canvas_with_border, legend_map))
}

fn map_to_pixel_equal_aspect(
    w_px: usize,
    h_px: usize,
    x: f64,
    y: f64,
    xr: (f64, f64),
    yr: (f64, f64),
) -> Option<(usize, usize)> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let rx = (xr.1 - xr.0).max(1e-9);
    let ry = (yr.1 - yr.0).max(1e-9);
    let xc = (xr.0 + xr.1) / 2.0;
    let yc = (yr.0 + yr.1) / 2.0;
    let wm = (w_px.saturating_sub(1)) as f64;
    let hm = (h_px.saturating_sub(1)) as f64;
    let pu = (wm / rx).min(hm / ry);
    let px = wm / 2.0 + (x - xc) * pu;
    let py = hm / 2.0 + (y - yc) * pu;
    let pxi = px.round() as isize;
    let pyi = py.round() as isize;
    if pxi >= 0 && pyi >= 0 && (pxi as usize) < w_px && (pyi as usize) < h_px {
        Some((pxi as usize, pyi as usize))
    } else {
        None
    }
}

fn build_spatial_scatter_canvas_fixed_dims(
    path: &Path,
    color_by: &str,
    chart_w: usize,
    chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<(usize, String, String, String, Vec<(String, Color)>)> {
    let file = H5::open(path)?;
    let adata = AnnData::<H5>::open(file)?;
    let n_obs = adata.n_obs();
    let key = match obsm_key {
        Some(k) => k.to_string(),
        None => detect_spatial_obsm_key(&adata)?,
    };
    let xy = obsm_xy_f64(&adata, &key).with_context(|| format!("read obsm[{key}]"))?;
    anyhow::ensure!(
        xy.nrows() == n_obs,
        "obsm[{key}] rows {} != n_obs {n_obs}",
        xy.nrows()
    );
    anyhow::ensure!(xy.ncols() >= 2, "obsm[{key}] needs ≥2 columns");
    let slice = xy.slice(s![.., ..2]);
    let obs = adata.read_obs()?;
    let labels = obs_column_strings(&obs, color_by)?;

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        points.push((slice[[i, 0]], slice[[i, 1]]));
    }
    let (canvas_no_border, canvas_with_border, legend_map) =
        draw_spatial_scatter_canvas_from_points(&points, &labels, chart_w, chart_h)?;
    let mut legend: Vec<(String, Color)> = legend_map.into_iter().collect();
    legend.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((n_obs, key, canvas_no_border, canvas_with_border, legend))
}

pub fn build_spatial_scatter_canvas(
    path: &Path,
    color_by: &str,
    max_chart_w: usize,
    max_chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<(usize, String, String, String, Vec<(String, Color)>)> {
    let (cw, ch) = chart_size_square_pixels(max_chart_w, max_chart_h);
    build_spatial_scatter_canvas_fixed_dims(path, color_by, cw, ch, obsm_key)
}

#[cfg(feature = "tui")]
pub fn ansi_braille_to_lines(s: &str) -> Vec<ratatui::text::Line<'static>> {
    s.lines().map(|line| ansi_line_to_line(line)).collect()
}

#[cfg(feature = "tui")]
fn ansi_line_to_line(line: &str) -> ratatui::text::Line<'static> {
    use ratatui::style::Style;
    use ratatui::text::{Line, Span};

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut run = String::new();
    let mut style = Style::default();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\x1b' && i + 1 < chars.len() && chars[i + 1] == '[' {
            if !run.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut run), style));
            }
            i += 2;
            let start = i;
            while i < chars.len() && chars[i] != 'm' {
                i += 1;
            }
            let seq: String = chars[start..i].iter().collect();
            i += 1;
            style = apply_ansi_seq(style, &seq);
        } else {
            run.push(chars[i]);
            i += 1;
        }
    }
    if !run.is_empty() {
        spans.push(Span::styled(run, style));
    }
    if spans.is_empty() {
        Line::default()
    } else {
        Line::from(spans)
    }
}

#[cfg(feature = "tui")]
fn apply_ansi_seq(
    _base: ratatui::style::Style,
    seq: &str,
) -> ratatui::style::Style {
    use ratatui::style::{Color, Style};
    if seq == "0" || seq.is_empty() {
        return Style::default();
    }
    let parts: Vec<&str> = seq.split(';').collect();
    if parts.len() >= 5 && parts[0] == "38" && parts[1] == "2" {
        let r: u8 = parts[2].parse().unwrap_or(255);
        let g: u8 = parts[3].parse().unwrap_or(255);
        let b: u8 = parts[4].parse().unwrap_or(255);
        return Style::default().fg(Color::Rgb(r, g, b));
    }
    if parts.len() == 1 {
        if let Ok(n) = parts[0].parse::<u8>() {
            return match n {
                0 => Style::default(),
                30 => Style::default().fg(Color::Black),
                31 => Style::default().fg(Color::Red),
                32 => Style::default().fg(Color::Green),
                33 => Style::default().fg(Color::Yellow),
                34 => Style::default().fg(Color::Blue),
                35 => Style::default().fg(Color::Magenta),
                36 => Style::default().fg(Color::Cyan),
                37 => Style::default().fg(Color::Gray),
                39 => Style::default(),
                90 => Style::default().fg(Color::DarkGray),
                91 => Style::default().fg(Color::LightRed),
                92 => Style::default().fg(Color::LightGreen),
                93 => Style::default().fg(Color::LightYellow),
                94 => Style::default().fg(Color::LightBlue),
                95 => Style::default().fg(Color::LightMagenta),
                96 => Style::default().fg(Color::LightCyan),
                97 => Style::default().fg(Color::White),
                _ => Style::default(),
            };
        }
    }
    Style::default()
}

#[cfg(feature = "tui")]
pub fn spatial_scatter_lines_for_tui(
    path: &Path,
    color_by: &str,
    chart_w: usize,
    chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<Vec<ratatui::text::Line<'static>>> {
    let (_n, _key, canvas, _, _) =
        build_spatial_scatter_canvas_fixed_dims(path, color_by, chart_w, chart_h, obsm_key)?;
    Ok(ansi_braille_to_lines(&canvas))
}

#[cfg(feature = "tui")]
pub fn spatial_scatter_lines_from_xy_labels(
    points: &[(f64, f64)],
    labels: &[String],
    chart_w: usize,
    chart_h: usize,
) -> anyhow::Result<Vec<ratatui::text::Line<'static>>> {
    let (canvas_no_border, _, _) =
        draw_spatial_scatter_canvas_from_points(points, labels, chart_w, chart_h)?;
    Ok(ansi_braille_to_lines(&canvas_no_border))
}

/// Prefer string cell-type label columns for terminal spatial plots (`cell_type`, …);
/// otherwise use `fallback` (typically `[data].cluster_annot`, e.g. `cell_type_int`).
pub fn resolve_plot_h5ad_color_column(path: &Path, fallback: &str) -> anyhow::Result<String> {
    let file = H5::open(path)?;
    let adata = AnnData::<H5>::open(file)?;
    let obs = adata.read_obs()?;
    let names = obs.get_column_names();
    const PREFERRED: &[&str] = &["cell_type", "cell_types", "celltype", "major_cell_type"];
    for p in PREFERRED {
        if let Some(n) = names.iter().find(|n| n.as_str() == *p) {
            return Ok(n.to_string());
        }
    }
    if let Some(n) = names
        .iter()
        .find(|n| n.as_str().eq_ignore_ascii_case("cell_type"))
    {
        return Ok(n.to_string());
    }
    Ok(fallback.to_string())
}

pub fn print_h5ad_scatter(path: &Path, color_by: &str) -> anyhow::Result<()> {
    use colored::Colorize;
    let term = terminal_size::terminal_size();
    let cols = term.map(|(w, _)| w.0 as usize).unwrap_or(100).max(60);
    let rows = term.map(|(_, h)| h.0 as usize).unwrap_or(32);
    let max_chart_w = ((cols.saturating_sub(4)) / 2).clamp(15, 90);
    let max_chart_h = rows.saturating_sub(10).clamp(8, 48);
    let (n_obs, key, _, canvas, legend) =
        build_spatial_scatter_canvas(path, color_by, max_chart_w, max_chart_h, None)?;
    println!(
        "{}  {}  n={}  obsm[{key}]  color=obs[{color_by}]",
        "AnnData spatial".bold(),
        path.display(),
        n_obs
    );
    print!("{}", canvas);
    let legend: String = legend
        .iter()
        .map(|(name, c)| {
            let (r, g, b) = match c {
                Color::TrueColor { r, g, b } => (*r, *g, *b),
                _ => (255, 255, 255),
            };
            format!("{}", name.as_str().truecolor(r, g, b))
        })
        .collect::<Vec<_>>()
        .join("  ·  ");
    println!("{}", legend);
    Ok(())
}
