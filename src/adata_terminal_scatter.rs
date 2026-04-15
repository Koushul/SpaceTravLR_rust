use std::collections::HashMap;
use std::path::Path;

use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend};
use anyhow::{Context, bail};
use colored::Color;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset as H5Dataset, File as H5File, Group, LocationType};
use ndarray::{Array1, Array2, s};
use termplot_rs::ChartContext;

pub const SPATIAL_OBSM_KEYS: &[&str] = &["spatial", "X_spatial", "spatial_loc"];

pub struct SpatialScatterCanvas {
    pub n_obs: usize,
    pub obsm_key: String,
    pub canvas_no_border: String,
    pub canvas_with_border: String,
    pub legend: Vec<(String, Color)>,
}

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

fn obs_index_dataset_name(obs: &Group) -> Option<String> {
    let a = obs.attr("_index").ok()?;
    let names: Array1<VarLenUnicode> = a.read_1d().ok()?;
    if names.is_empty() {
        return None;
    }
    Some(names[[0]].to_string())
}

fn h5ad_obs_column_names_from_members(obs: &Group) -> anyhow::Result<Vec<String>> {
    let mut names = obs.member_names().context("h5ad: list obs members")?;
    if let Some(ix) = obs_index_dataset_name(obs) {
        names.retain(|n| n != &ix);
    }
    names.retain(|n| !n.starts_with("__"));
    names.sort();
    anyhow::ensure!(
        !names.is_empty(),
        "h5ad obs: no data columns (after excluding cell index)"
    );
    Ok(names)
}

fn h5ad_obs_column_names_for_plot(root: &Group) -> anyhow::Result<Vec<String>> {
    let obs = root.group("obs").context("h5ad: missing obs group")?;
    if let Ok(attr) = obs.attr("column-order") {
        if let Ok(raw) = attr.read_1d::<VarLenUnicode>() {
            let v: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
    h5ad_obs_column_names_from_members(&obs)
}

fn h5ad_obs_labels_from_dataset(ds: &H5Dataset) -> anyhow::Result<Vec<String>> {
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 1, "expected 1d obs column, got shape {:?}", sh);
    if let Ok(v) = ds.read_1d::<VarLenUnicode>() {
        return Ok(v.iter().map(|s| s.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<i64>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<i32>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<u32>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<f64>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<f32>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<bool>() {
        return Ok(v.iter().map(|b| if *b { "true" } else { "false" }.to_string()).collect());
    }
    bail!("unsupported obs column element type for plot-h5ad");
}

fn h5ad_obs_labels_from_categorical_group(g: &Group) -> anyhow::Result<Vec<String>> {
    let codes = g.dataset("codes")?.read_1d::<i32>()?;
    let cats: Array1<VarLenUnicode> = g.dataset("categories")?.read_1d()?;
    let mut out = Vec::with_capacity(codes.len());
    for &c in codes.iter() {
        let i = c as usize;
        let s = if c < 0 || i >= cats.len() {
            "NA".to_string()
        } else {
            cats[[i]].to_string()
        };
        out.push(s);
    }
    Ok(out)
}

fn h5ad_obs_labels(obs: &Group, col: &str) -> anyhow::Result<Vec<String>> {
    anyhow::ensure!(obs.link_exists(col), "obs column {col:?} not found");
    match obs.loc_type_by_name(col).with_context(|| format!("obs[{col}]"))? {
        LocationType::Dataset => h5ad_obs_labels_from_dataset(&obs.dataset(col)?),
        LocationType::Group => h5ad_obs_labels_from_categorical_group(&obs.group(col)?),
        _ => bail!("obs[{col}] unsupported HDF5 type for plot-h5ad"),
    }
}

fn h5ad_read_obsm_xy_two_cols(ds: &H5Dataset) -> anyhow::Result<Array2<f64>> {
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 2, "expected 2d obsm matrix, got shape {:?}", sh);
    let ncols = sh[1];
    anyhow::ensure!(ncols >= 2, "obsm matrix needs ≥2 columns, got {ncols}");
    if ncols == 2 {
        if let Ok(a) = ds.read_2d::<f32>() {
            return Ok(a.mapv(|x| x as f64));
        }
        if let Ok(a) = ds.read_2d::<f64>() {
            return Ok(a);
        }
    } else if let Ok(a) = ds.read_slice_2d::<f32, _>(s![.., 0..2]) {
        return Ok(a.mapv(|x| x as f64));
    } else if let Ok(a) = ds.read_slice_2d::<f64, _>(s![.., 0..2]) {
        return Ok(a);
    }
    bail!("unsupported obsm dtype (expected f32/f64)");
}

fn h5ad_obsm_xy_two_cols(obsm: &Group, key: &str) -> anyhow::Result<Array2<f64>> {
    anyhow::ensure!(obsm.link_exists(key), "missing obsm[{key}]");
    match obsm.loc_type_by_name(key)? {
        LocationType::Dataset => h5ad_read_obsm_xy_two_cols(&obsm.dataset(key)?),
        LocationType::Group => {
            let g = obsm.group(key)?;
            if g.link_exists("array") && matches!(g.loc_type_by_name("array")?, LocationType::Dataset)
            {
                h5ad_read_obsm_xy_two_cols(&g.dataset("array")?)
            } else {
                bail!("obsm[{key}] group has no numeric array dataset");
            }
        }
        _ => bail!("obsm[{key}] unsupported HDF5 type"),
    }
}

fn h5ad_spatial_obsm_key(obsm: &Group, forced: Option<&str>) -> anyhow::Result<String> {
    if let Some(k) = forced {
        anyhow::ensure!(obsm.link_exists(k), "obsm missing key {k:?}");
        let m = h5ad_obsm_xy_two_cols(obsm, k)?;
        anyhow::ensure!(m.nrows() > 0, "obsm[{k}] is empty");
        return Ok(k.to_string());
    }
    for key in SPATIAL_OBSM_KEYS {
        if !obsm.link_exists(key) {
            continue;
        }
        if let Ok(m) = h5ad_obsm_xy_two_cols(obsm, key) {
            if m.nrows() > 0 {
                return Ok((*key).to_string());
            }
        }
    }
    let keys = obsm.member_names().unwrap_or_default();
    anyhow::bail!(
        "no usable 2D spatial in obsm (tried {:?}). obsm keys: {keys:?}",
        SPATIAL_OBSM_KEYS
    )
}

fn assign_colors(labels: &[String]) -> (HashMap<String, Color>, Vec<Color>) {
    let mut uniq: Vec<String> = Vec::new();
    for s in labels {
        if !uniq.contains(s) {
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

fn char_term_display_width(ch: char) -> usize {
    #[cfg(feature = "tui")]
    {
        use unicode_width::UnicodeWidthChar;
        ch.width().unwrap_or(0)
    }
    #[cfg(not(feature = "tui"))]
    {
        let _ = ch;
        1usize
    }
}

#[allow(clippy::while_let_on_iterator)]
fn ansi_stripped_display_width(s: &str) -> usize {
    let mut total = 0usize;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' && it.peek() == Some(&'[') {
            it.next();
            for ch in it.by_ref() {
                if ch == 'm' {
                    break;
                }
            }
            continue;
        }
        total += char_term_display_width(c);
    }
    total
}

fn pad_right_ansi_to_display_width(line: &str, target: usize) -> String {
    let w = ansi_stripped_display_width(line);
    if w >= target {
        line.to_string()
    } else {
        format!("{}{}", line, " ".repeat(target - w))
    }
}

fn legend_plain_width_budget(labels: &[String], n_obs: usize) -> usize {
    let mut uniq: Vec<&str> = Vec::new();
    for s in labels {
        if !uniq.contains(&s.as_str()) {
            uniq.push(s.as_str());
        }
    }
    uniq.sort();
    let name_w = uniq
        .iter()
        .map(|n| n.chars().map(char_term_display_width).sum::<usize>())
        .max()
        .unwrap_or(0)
        .min(48);
    let num_w = format!("{}", n_obs).len().max(1);
    name_w + 2 + num_w
}

fn zip_spatial_canvas_and_legend_lines(canvas: &str, legend_lines: &[String]) -> String {
    let plot_lines: Vec<&str> = canvas.trim_end_matches('\n').lines().collect();
    let plot_w = plot_lines
        .iter()
        .map(|l| ansi_stripped_display_width(l))
        .max()
        .unwrap_or(0);
    let n = plot_lines.len().max(legend_lines.len());
    let mut out = String::new();
    for i in 0..n {
        let left = plot_lines.get(i).copied().unwrap_or("");
        let padded = pad_right_ansi_to_display_width(left, plot_w);
        out.push_str(&padded);
        if let Some(leg) = legend_lines.get(i) {
            out.push_str("  ");
            out.push_str(leg);
        }
        out.push('\n');
    }
    out
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
    let (xr, yr) = ChartContext::get_auto_range(points, 0.04);
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

fn build_spatial_scatter_canvas_fixed_dims_from_root(
    root: &Group,
    color_by: &str,
    chart_w: usize,
    chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<SpatialScatterCanvas> {
    let obsm = root.group("obsm").context("h5ad: missing obsm group")?;
    let key = h5ad_spatial_obsm_key(&obsm, obsm_key)?;
    let xy = h5ad_obsm_xy_two_cols(&obsm, &key).with_context(|| format!("read obsm[{key}]"))?;
    let n_obs = xy.nrows();
    anyhow::ensure!(xy.ncols() == 2, "internal: expected two spatial columns");
    let obs = root.group("obs").context("h5ad: missing obs group")?;
    let labels = h5ad_obs_labels(&obs, color_by).with_context(|| format!("obs[{color_by}]"))?;
    anyhow::ensure!(
        labels.len() == n_obs,
        "obs[{color_by}] length {} != n_obs {n_obs}",
        labels.len()
    );

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        points.push((xy[[i, 0]], xy[[i, 1]]));
    }
    let (canvas_no_border, canvas_with_border, legend_map) =
        draw_spatial_scatter_canvas_from_points(&points, &labels, chart_w, chart_h)?;
    let mut legend: Vec<(String, Color)> = legend_map.into_iter().collect();
    legend.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(SpatialScatterCanvas {
        n_obs,
        obsm_key: key,
        canvas_no_border,
        canvas_with_border,
        legend,
    })
}

fn build_spatial_scatter_canvas_fixed_dims(
    path: &Path,
    color_by: &str,
    chart_w: usize,
    chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<SpatialScatterCanvas> {
    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;
    build_spatial_scatter_canvas_fixed_dims_from_root(&h5, color_by, chart_w, chart_h, obsm_key)
}

pub fn build_spatial_scatter_canvas(
    path: &Path,
    color_by: &str,
    max_chart_w: usize,
    max_chart_h: usize,
    obsm_key: Option<&str>,
) -> anyhow::Result<SpatialScatterCanvas> {
    let (cw, ch) = chart_size_square_pixels(max_chart_w, max_chart_h);
    build_spatial_scatter_canvas_fixed_dims(path, color_by, cw, ch, obsm_key)
}

#[cfg(feature = "tui")]
pub fn ansi_braille_to_lines(s: &str) -> Vec<ratatui::text::Line<'static>> {
    s.lines().map(ansi_line_to_line).collect()
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
    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;
    let c = build_spatial_scatter_canvas_fixed_dims_from_root(
        &h5,
        color_by,
        chart_w,
        chart_h,
        obsm_key,
    )?;
    Ok(ansi_braille_to_lines(&c.canvas_no_border))
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

/// Resolve a cell-type label column in obs, returning `None` when nothing suitable exists.
fn resolve_plot_h5ad_color_column_opt(root: &Group) -> Option<String> {
    let names = h5ad_obs_column_names_for_plot(root).ok()?;
    const PREFERRED: &[&str] = &["cell_type", "cell_types", "celltype", "major_cell_type"];
    for p in PREFERRED {
        if let Some(n) = names.iter().find(|n| n == p) {
            return Some(n.clone());
        }
    }
    if let Some(n) = names.iter().find(|n| n.eq_ignore_ascii_case("cell_type")) {
        return Some(n.clone());
    }
    None
}

/// Prefer string cell-type label columns for terminal spatial plots (`cell_type`, …);
/// otherwise use `fallback` (typically `[data].cluster_annot`, e.g. `cell_type`).
///
/// Uses only HDF5 metadata and the chosen `obs` column — no full `AnnData` load.
pub fn resolve_plot_h5ad_color_column_from_root(
    root: &Group,
    fallback: &str,
) -> anyhow::Result<String> {
    if let Some(c) = resolve_plot_h5ad_color_column_opt(root) {
        return Ok(c);
    }
    let obs = root.group("obs").ok();
    if let Some(ref obs) = obs {
        if obs.link_exists(fallback) {
            return Ok(fallback.to_string());
        }
    }
    Ok(fallback.to_string())
}

pub fn resolve_plot_h5ad_color_column(path: &Path, fallback: &str) -> anyhow::Result<String> {
    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;
    resolve_plot_h5ad_color_column_from_root(&h5, fallback)
}

const UMAP_OBSM_KEYS: &[&str] = &["X_umap", "umap"];

fn h5ad_umap_obsm_key(obsm: &Group) -> anyhow::Result<String> {
    for key in UMAP_OBSM_KEYS {
        if !obsm.link_exists(key) {
            continue;
        }
        if let Ok(m) = h5ad_obsm_xy_two_cols(obsm, key) {
            if m.nrows() > 0 {
                return Ok((*key).to_string());
            }
        }
    }
    let keys = obsm.member_names().unwrap_or_default();
    anyhow::bail!(
        "no usable 2D UMAP in obsm (tried {:?}). obsm keys: {keys:?}",
        UMAP_OBSM_KEYS
    )
}

/// Print a terminal UMAP scatter from a preprocessed `.h5ad` (obsm `X_umap` or `umap`).
pub fn print_h5ad_umap_scatter(path: &Path) -> anyhow::Result<()> {
    use colored::Colorize;
    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;

    let obsm = h5
        .group("obsm")
        .context("h5ad: missing obsm group — cannot plot without UMAP coordinates")?;
    let umap_key = h5ad_umap_obsm_key(&obsm)?;
    let xy = h5ad_obsm_xy_two_cols(&obsm, &umap_key)
        .with_context(|| format!("failed to read obsm['{umap_key}']"))?;
    let n_obs = xy.nrows();
    anyhow::ensure!(n_obs > 0, "obsm['{umap_key}'] is empty");

    let color_col = resolve_plot_h5ad_color_column_opt(&h5)
        .or_else(|| {
            let obs = h5.group("obs").ok()?;
            for fallback in ["cell_type", "leiden"] {
                if obs.link_exists(fallback) {
                    return Some(fallback.to_string());
                }
            }
            None
        });

    let labels: Vec<String> = if let Some(ref col) = color_col {
        let obs = h5.group("obs").context("h5ad: missing obs group")?;
        match h5ad_obs_labels(&obs, col) {
            Ok(l) if l.len() == n_obs => l,
            _ => vec!["cell".to_string(); n_obs],
        }
    } else {
        vec!["cell".to_string(); n_obs]
    };

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        points.push((xy[[i, 0]], xy[[i, 1]]));
    }

    let term = terminal_size::terminal_size();
    let cols = term.map(|(w, _)| w.0 as usize).unwrap_or(100).max(60);
    let rows = term.map(|(_, h)| h.0 as usize).unwrap_or(32);
    let margin = 4usize;
    let legend_gap = 2usize;
    let border_cols = 2usize;
    let legend_budget = legend_plain_width_budget(&labels, n_obs);
    let max_chart_w = cols
        .saturating_sub(margin)
        .saturating_sub(legend_gap)
        .saturating_sub(legend_budget)
        .saturating_sub(border_cols)
        .clamp(15, 90);
    let max_chart_h = rows.saturating_sub(10).clamp(8, 48);
    let (cw, ch) = chart_size_square_pixels(max_chart_w, max_chart_h);

    let color_label = color_col
        .as_deref()
        .filter(|_| labels.iter().any(|l| l != "cell"))
        .unwrap_or("(none)");

    let (_, canvas, legend_map) =
        draw_spatial_scatter_canvas_from_points(&points, &labels, cw, ch)?;
    let mut legend: Vec<(String, Color)> = legend_map.into_iter().collect();
    legend.sort_by(|a, b| a.0.cmp(&b.0));

    println!(
        "{}  {}  n={}  obsm[{umap_key}]  color=obs[{color_label}]",
        "AnnData UMAP".bold(),
        path.display(),
        n_obs,
    );
    let mut label_counts: HashMap<&str, usize> = HashMap::new();
    for l in &labels {
        *label_counts.entry(l.as_str()).or_insert(0) += 1;
    }
    let legend_lines: Vec<String> = legend
        .iter()
        .map(|(name, c)| {
            let cnt = label_counts.get(name.as_str()).copied().unwrap_or(0);
            let (r, g, b) = match c {
                Color::TrueColor { r, g, b } => (*r, *g, *b),
                _ => (255, 255, 255),
            };
            format!(
                "{} {}",
                name.as_str().truecolor(r, g, b),
                cnt.to_string().dimmed()
            )
        })
        .collect();
    print!("{}", zip_spatial_canvas_and_legend_lines(&canvas, &legend_lines));
    Ok(())
}

/// Opens the `.h5ad` **once** (read-only HDF5), reads only `obsm` (two columns) and optionally one
/// `obs` column for coloring. No `AnnData` open, no expression data, no preprocessing.
///
/// If no suitable cell-type column exists, every cell is drawn in a single color.
pub fn print_h5ad_scatter(path: &Path, cluster_annot_fallback: &str) -> anyhow::Result<()> {
    use colored::Colorize;
    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;

    let obsm = h5
        .group("obsm")
        .context("h5ad: missing obsm group — cannot plot without spatial coordinates")?;
    anyhow::ensure!(
        obsm.link_exists("spatial"),
        "obsm['spatial'] not found — --plot-h5ad requires spatial coordinates in obsm['spatial']"
    );
    let xy = h5ad_obsm_xy_two_cols(&obsm, "spatial")
        .context("failed to read obsm['spatial']")?;
    let n_obs = xy.nrows();
    anyhow::ensure!(n_obs > 0, "obsm['spatial'] is empty");

    let color_col = resolve_plot_h5ad_color_column_opt(&h5)
        .or_else(|| {
            let obs = h5.group("obs").ok()?;
            if obs.link_exists(cluster_annot_fallback) {
                Some(cluster_annot_fallback.to_string())
            } else {
                None
            }
        });

    let labels: Vec<String> = if let Some(ref col) = color_col {
        let obs = h5.group("obs").context("h5ad: missing obs group")?;
        match h5ad_obs_labels(&obs, col) {
            Ok(l) if l.len() == n_obs => l,
            _ => vec!["cell".to_string(); n_obs],
        }
    } else {
        vec!["cell".to_string(); n_obs]
    };

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        points.push((xy[[i, 0]], xy[[i, 1]]));
    }

    let term = terminal_size::terminal_size();
    let cols = term.map(|(w, _)| w.0 as usize).unwrap_or(100).max(60);
    let rows = term.map(|(_, h)| h.0 as usize).unwrap_or(32);
    let margin = 4usize;
    let legend_gap = 2usize;
    let border_cols = 2usize;
    let legend_budget = legend_plain_width_budget(&labels, n_obs);
    let max_chart_w = cols
        .saturating_sub(margin)
        .saturating_sub(legend_gap)
        .saturating_sub(legend_budget)
        .saturating_sub(border_cols)
        .clamp(15, 90);
    let max_chart_h = rows.saturating_sub(10).clamp(8, 48);
    let (cw, ch) = chart_size_square_pixels(max_chart_w, max_chart_h);

    let color_label = color_col
        .as_deref()
        .filter(|_| labels.iter().any(|l| l != "cell"))
        .unwrap_or("(none)");

    let (_, canvas, legend_map) =
        draw_spatial_scatter_canvas_from_points(&points, &labels, cw, ch)?;
    let mut legend: Vec<(String, Color)> = legend_map.into_iter().collect();
    legend.sort_by(|a, b| a.0.cmp(&b.0));

    println!(
        "{}  {}  n={}  obsm[spatial]  color=obs[{color_label}]",
        "AnnData spatial".bold(),
        path.display(),
        n_obs
    );
    let mut label_counts: HashMap<&str, usize> = HashMap::new();
    for l in &labels {
        *label_counts.entry(l.as_str()).or_insert(0) += 1;
    }
    let legend_lines: Vec<String> = legend
        .iter()
        .map(|(name, c)| {
            let cnt = label_counts.get(name.as_str()).copied().unwrap_or(0);
            let (r, g, b) = match c {
                Color::TrueColor { r, g, b } => (*r, *g, *b),
                _ => (255, 255, 255),
            };
            format!(
                "{} {}",
                name.as_str().truecolor(r, g, b),
                cnt.to_string().dimmed()
            )
        })
        .collect();
    print!("{}", zip_spatial_canvas_and_legend_lines(&canvas, &legend_lines));
    Ok(())
}
