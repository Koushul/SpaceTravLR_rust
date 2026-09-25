use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use crate::adata_terminal_scatter::read_h5ad_obs_column_str_h5;
use anyhow::Context;
use colored::Colorize;
use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
use hdf5_metno::{Attribute, Dataset, File as H5File, Group, LocationType};
use ndarray::s;

const PEEK_LW: usize = 8;

/// Gruvbox (light) accents — https://github.com/morhetz/gruvbox
const GB_BRIGHT_RED: (u8, u8, u8) = (251, 73, 52); // #fb4934 — shape highlight (pink-red)
const GB_BRIGHT_AQUA: (u8, u8, u8) = (142, 192, 124); // #8ec07c
const GB_BRIGHT_YELLOW: (u8, u8, u8) = (250, 189, 47); // #fabd2f
const GB_BRIGHT_BLUE: (u8, u8, u8) = (131, 165, 152); // #83a598
const GB_BRIGHT_PURPLE: (u8, u8, u8) = (211, 134, 155); // #d3869b
const GB_BRIGHT_GREEN: (u8, u8, u8) = (184, 187, 38); // #b8bb26
const GB_FG3: (u8, u8, u8) = (189, 174, 147); // #bdae93
const GB_GRAY: (u8, u8, u8) = (146, 131, 116); // #928374

fn peek_color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none()
}

fn peek_head_prefix(label: &str) -> (String, usize) {
    if peek_color_enabled() {
        let label_s = format!(
            "{}{}",
            "› ".truecolor(GB_GRAY.0, GB_GRAY.1, GB_GRAY.2),
            format!("{:<width$}", label, width = PEEK_LW)
                .truecolor(GB_GRAY.0, GB_GRAY.1, GB_GRAY.2)
        );
        let head = format!("{label_s}  ");
        (head, 2 + PEEK_LW + 2)
    } else {
        let head = format!("› {:<width$}  ", label, width = PEEK_LW);
        let cols = head.chars().count();
        (head, cols)
    }
}

#[derive(Clone, Copy)]
struct PeekStyle {
    rgb: Option<(u8, u8, u8)>,
    bold: bool,
}

impl PeekStyle {
    const PLAIN: Self = Self {
        rgb: None,
        bold: false,
    };

    const PATH: Self = Self {
        rgb: Some(GB_BRIGHT_YELLOW),
        bold: false,
    };
    const SIZE: Self = Self {
        rgb: Some(GB_BRIGHT_AQUA),
        bold: false,
    };
    const SHAPE: Self = Self {
        rgb: Some(GB_BRIGHT_RED),
        bold: true,
    };
    const META: Self = Self {
        rgb: Some(GB_FG3),
        bold: false,
    };
    const OBS_GRID: Self = Self {
        rgb: Some(GB_BRIGHT_BLUE),
        bold: false,
    };
    const VAR_GRID: Self = Self {
        rgb: Some(GB_BRIGHT_PURPLE),
        bold: false,
    };

    fn paint(self, s: &str) -> String {
        if !peek_color_enabled() {
            return s.to_string();
        }
        match self.rgb {
            Some((r, g, b)) => {
                let c = s.truecolor(r, g, b);
                if self.bold {
                    c.bold().to_string()
                } else {
                    c.to_string()
                }
            }
            None => {
                if self.bold {
                    s.bold().to_string()
                } else {
                    s.to_string()
                }
            }
        }
    }
}

fn axis_index_dataset_name(axis: &Group) -> Option<String> {
    let a = axis.attr("_index").ok()?;
    let name = h5_attr_string(&a)?;
    if name.is_empty() { None } else { Some(name) }
}

fn h5_dataframe_column_names(axis: &Group) -> anyhow::Result<Vec<String>> {
    if let Ok(attr) = axis.attr("column-order") {
        if let Ok(raw) = attr.read_1d::<VarLenUnicode>() {
            let v: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
    let mut names = axis
        .member_names()
        .with_context(|| "list HDF5 group members")?;
    if let Some(ix) = axis_index_dataset_name(axis) {
        names.retain(|n| n != &ix);
    }
    names.retain(|n| !n.starts_with("__"));
    names.sort();
    Ok(names)
}

fn fmt_usize_sep(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn format_file_size(bytes: u64) -> String {
    const KB: u128 = 1024;
    let b = bytes as u128;
    if b >= KB * KB * KB {
        format!("{:.2} GiB", b as f64 / (KB * KB * KB) as f64)
    } else if b >= KB * KB {
        format!("{:.2} MiB", b as f64 / (KB * KB) as f64)
    } else if b >= KB {
        format!("{:.2} KiB", b as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn peek_terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(88)
        .clamp(48, 200)
}

fn floor_char_boundary_stable(s: &str, i: usize) -> usize {
    let mut j = i.min(s.len());
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}

fn byte_index_at_char_count(s: &str, max_chars: usize) -> usize {
    for (count, (i, _)) in s.char_indices().enumerate() {
        if count >= max_chars {
            return floor_char_boundary_stable(s, i);
        }
    }
    s.len()
}

fn wrap_fill_lines(text: &str, width: usize) -> Vec<String> {
    if width < 8 || text.is_empty() {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut rest = text.trim_end();
    while !rest.is_empty() {
        if rest.chars().count() <= width {
            lines.push(rest.to_string());
            break;
        }
        let mut cut = byte_index_at_char_count(rest, width);
        if let Some(sp) = rest[..cut].rfind(' ') {
            if sp > width / 4 {
                cut = floor_char_boundary_stable(rest, sp + 1);
            }
        }
        let (line, tail) = rest.split_at(cut);
        lines.push(line.trim_end().to_string());
        rest = tail.trim_start();
    }
    lines
}

fn peek_row_wrapped(
    out: &mut String,
    label: &str,
    value: &str,
    term_w: usize,
    value_style: PeekStyle,
) {
    let (head, head_cols) = peek_head_prefix(label);
    let budget = term_w.saturating_sub(head_cols).max(12);
    let first_chunks = wrap_fill_lines(value, budget);
    let cont = " ".repeat(head_cols);
    for (i, chunk) in first_chunks.iter().enumerate() {
        let chunk_styled = value_style.paint(chunk);
        if i == 0 {
            let _ = writeln!(out, "{}{}", head, chunk_styled);
        } else {
            let _ = writeln!(out, "{}{}", cont, chunk_styled);
        }
    }
}

fn peek_value_start_cols() -> usize {
    if peek_color_enabled() {
        2 + PEEK_LW + 2
    } else {
        format!("› {:<width$}  ", "", width = PEEK_LW)
            .chars()
            .count()
    }
}

fn format_name_grid(names: &[String], term_w: usize) -> String {
    if names.is_empty() {
        return String::new();
    }
    let head_cols = peek_value_start_cols();
    let usable = term_w.saturating_sub(head_cols).max(16);
    let max_name = names
        .iter()
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(1)
        .min(40);
    let gutter = 2usize;
    let col_w = (max_name + gutter).clamp(6, usable);
    let ncols = (usable / col_w).clamp(1, 12);
    let col_w = usable / ncols;

    let mut lines = Vec::new();
    let mut idx = 0;
    while idx < names.len() {
        let mut row = String::new();
        for _ in 0..ncols {
            if idx >= names.len() {
                break;
            }
            let s = &names[idx];
            let mut cell: String = s.chars().take(col_w.saturating_sub(gutter)).collect();
            if s.chars().count() > cell.chars().count() {
                cell.push('…');
            }
            row.push_str(&format!("{:<cw$}", cell, cw = col_w));
            idx += 1;
        }
        lines.push(row.trim_end().to_string());
    }
    lines.join("\n")
}

fn peek_label_grid(
    out: &mut String,
    label: &str,
    names: &[String],
    term_w: usize,
    grid_style: PeekStyle,
) {
    if names.is_empty() {
        return;
    }
    let (head, head_cols) = peek_head_prefix(label);
    let grid = format_name_grid(names, term_w);
    let cont = " ".repeat(head_cols);
    let mut gi = grid.lines();
    if let Some(first) = gi.next() {
        let _ = writeln!(out, "{}{}", head, grid_style.paint(first));
        for line in gi {
            let _ = writeln!(out, "{}{}", cont, grid_style.paint(line));
        }
    }
}

fn sorted_mapping_keys(root: &Group, name: &str) -> Vec<String> {
    if !root.link_exists(name) {
        return Vec::new();
    }
    let Ok(g) = root.group(name) else {
        return Vec::new();
    };
    let Ok(mut names) = g.member_names() else {
        return Vec::new();
    };
    names.sort();
    names
}

fn read_shape_attr_2d(g: &Group) -> Option<(usize, usize)> {
    let a = g.attr("shape").ok()?;
    if let Ok(v) = a.read_1d::<u64>() {
        if v.len() >= 2 {
            return Some((v[0] as usize, v[1] as usize));
        }
    }
    if let Ok(v) = a.read_1d::<i64>() {
        if v.len() >= 2 {
            return Some((v[0] as usize, v[1] as usize));
        }
    }
    if let Ok(v) = a.read_1d::<u32>() {
        if v.len() >= 2 {
            return Some((v[0] as usize, v[1] as usize));
        }
    }
    None
}

fn index_axis_len(axis: &Group) -> anyhow::Result<Option<usize>> {
    let Some(ix) = axis_index_dataset_name(axis) else {
        return Ok(None);
    };
    Ok(Some(axis.dataset(&ix)?.size()))
}

fn dense_inner_2d_shape(g: &Group) -> anyhow::Result<Option<(usize, usize)>> {
    let names = g.member_names().unwrap_or_default();
    let mut best: Option<(usize, usize)> = None;
    let mut best_n = 0usize;
    for name in names {
        let Ok(LocationType::Dataset) = g.loc_type_by_name(&name) else {
            continue;
        };
        let Ok(ds) = g.dataset(&name) else {
            continue;
        };
        let sh = ds.shape();
        if sh.len() == 2 {
            let n = sh[0].saturating_mul(sh[1]);
            if n >= best_n {
                best_n = n;
                best = Some((sh[0], sh[1]));
            }
        }
    }
    Ok(best)
}

fn x_n_obs_n_vars(root: &Group) -> anyhow::Result<Option<(usize, usize)>> {
    if !root.link_exists("X") {
        return Ok(None);
    }
    match root.loc_type_by_name("X").context("X link type")? {
        LocationType::Dataset => {
            let sh = root.dataset("X")?.shape();
            match sh.len() {
                2 => Ok(Some((sh[0], sh[1]))),
                1 => Ok(Some((sh[0], 1))),
                _ => anyhow::bail!("X dataset has unexpected rank {}", sh.len()),
            }
        }
        LocationType::Group => {
            let xg = root.group("X")?;
            if let Some(p) = read_shape_attr_2d(&xg) {
                return Ok(Some(p));
            }
            if xg.link_exists("indptr") {
                let ip_len = xg.dataset("indptr")?.size();
                anyhow::ensure!(ip_len >= 2, "csr X: indptr length {ip_len} is too short");
                let n_obs = ip_len - 1;
                return Ok(Some((n_obs, 0)));
            }
            if let Some(p) = dense_inner_2d_shape(&xg)? {
                return Ok(Some(p));
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn infer_n_obs_n_vars(root: &Group) -> anyhow::Result<(usize, usize)> {
    let n_obs_idx = root
        .group("obs")
        .ok()
        .map(|g| index_axis_len(&g))
        .transpose()?
        .flatten();
    let n_var_idx = root
        .group("var")
        .ok()
        .map(|g| index_axis_len(&g))
        .transpose()?
        .flatten();

    let (mut n_obs, mut n_var) = match x_n_obs_n_vars(root)? {
        None => (None, None),
        Some((o, 0)) => (Some(o), None),
        Some((o, v)) => (Some(o), Some(v)),
    };

    if n_obs.is_none() {
        n_obs = n_obs_idx;
    }
    if n_var.is_none() {
        n_var = n_var_idx;
    }

    if let (Some(o), Some(v)) = (n_obs, n_var) {
        return Ok((o, v));
    }
    if let (Some(o), None) = (n_obs, n_var) {
        let v = n_var_idx.context(
            "could not infer n_vars: X has no second dimension and var index is missing",
        )?;
        return Ok((o, v));
    }
    if let (None, Some(v)) = (n_obs, n_var) {
        let o = n_obs_idx
            .context("could not infer n_obs: X has no row count and obs index is missing")?;
        return Ok((o, v));
    }

    anyhow::bail!(
        "could not infer obs × var shape (tried X and obs/var _index lengths); is this AnnData HDF5?"
    );
}

fn read_dataset_len2_usize(ds: &Dataset) -> Option<(usize, usize)> {
    macro_rules! try_ty {
        ($t:ty) => {
            if let Ok(v) = ds.read_1d::<$t>() {
                if v.len() >= 2 {
                    return Some((v[0] as usize, v[1] as usize));
                }
            }
        };
    }
    try_ty!(i32);
    try_ty!(i64);
    try_ty!(u32);
    try_ty!(u64);
    None
}

fn sorted_group_member_summaries(g: &Group) -> anyhow::Result<Vec<String>> {
    let mut names = g
        .member_names()
        .with_context(|| "list HDF5 group members")?;
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let tag = match g.loc_type_by_name(&n) {
            Ok(LocationType::Dataset) => "dataset",
            Ok(LocationType::Group) => "group",
            Ok(_) => "other",
            Err(_) => "?",
        };
        out.push(format!("{n} ({tag})"));
    }
    Ok(out)
}

fn try_peek_tenx_filtered_matrix(
    root: &Group,
    path: &Path,
    file_bytes: u64,
    term_w: usize,
) -> anyhow::Result<Option<String>> {
    if !root.link_exists("matrix") {
        return Ok(None);
    }
    let Ok(matrix) = root.group("matrix") else {
        return Ok(None);
    };
    let Ok(shape_ds) = matrix.dataset("shape") else {
        return Ok(None);
    };
    let Some((n_features, n_barcodes)) = read_dataset_len2_usize(&shape_ds) else {
        return Ok(None);
    };
    if !matrix.link_exists("data") || !matrix.link_exists("indptr") {
        return Ok(None);
    }

    let n_obs = n_barcodes;
    let n_vars = n_features;

    let nnz = matrix.dataset("data").ok().map(|d| d.size());
    let n_bc = matrix
        .link_exists("barcodes")
        .then(|| matrix.dataset("barcodes").ok().map(|d| d.size()))
        .flatten();

    let matrix_members = sorted_group_member_summaries(&matrix).unwrap_or_default();
    let mut feature_names: Vec<String> = matrix
        .group("features")
        .ok()
        .and_then(|fg| fg.member_names().ok())
        .unwrap_or_default();
    feature_names.sort();
    feature_names.retain(|n| !n.starts_with('_'));

    let mut out = String::new();
    peek_row_wrapped(
        &mut out,
        "path",
        &path.display().to_string(),
        term_w,
        PeekStyle::PATH,
    );
    peek_row_wrapped(
        &mut out,
        "size",
        &format_file_size(file_bytes),
        term_w,
        PeekStyle::SIZE,
    );
    peek_row_wrapped(
        &mut out,
        "format",
        "10x sparse matrix (Cell Ranger-style /matrix, CSR)",
        term_w,
        PeekStyle::META,
    );
    peek_row_wrapped(
        &mut out,
        "shape",
        &format!(
            "{}×{}  (cells × genes, same axes as AnnData obs×var)",
            fmt_usize_sep(n_obs),
            fmt_usize_sep(n_vars)
        ),
        term_w,
        PeekStyle::SHAPE,
    );
    if let Some(n) = nnz {
        peek_row_wrapped(
            &mut out,
            "nnz",
            &format!("{} nonzeros in /matrix/data", fmt_usize_sep(n)),
            term_w,
            PeekStyle::META,
        );
    }
    if let Some(nb) = n_bc {
        let mut s = format!("{} barcodes", fmt_usize_sep(nb));
        if nb != n_obs {
            s.push_str(&format!(
                "  (note: /matrix/shape implies {} cells)",
                fmt_usize_sep(n_obs)
            ));
        }
        peek_row_wrapped(&mut out, "barcodes", &s, term_w, PeekStyle::META);
    }

    if !feature_names.is_empty() {
        peek_label_grid(
            &mut out,
            "features",
            &feature_names,
            term_w,
            PeekStyle::VAR_GRID,
        );
    }
    if !matrix_members.is_empty() {
        peek_row_wrapped(
            &mut out,
            "matrix",
            &matrix_members.join(", "),
            term_w,
            PeekStyle::META,
        );
    }

    Ok(Some(out))
}

fn peek_generic_hdf5_report(
    root: &Group,
    path: &Path,
    file_bytes: u64,
    term_w: usize,
) -> anyhow::Result<String> {
    let members = sorted_group_member_summaries(root)?;
    let mut out = String::new();
    peek_row_wrapped(
        &mut out,
        "path",
        &path.display().to_string(),
        term_w,
        PeekStyle::PATH,
    );
    peek_row_wrapped(
        &mut out,
        "size",
        &format_file_size(file_bytes),
        term_w,
        PeekStyle::SIZE,
    );
    peek_row_wrapped(
        &mut out,
        "format",
        "HDF5 (not AnnData or 10x /matrix layout)",
        term_w,
        PeekStyle::META,
    );
    peek_label_grid(&mut out, "root", &members, term_w, PeekStyle::OBS_GRID);
    Ok(out)
}

fn value_counts_block(
    col: &str,
    cells: &[String],
    n_obs: usize,
    term_w: usize,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        cells.len() == n_obs,
        "obs['{col}'] length {} != n_obs {n_obs}",
        cells.len()
    );
    let n = cells.len();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for v in cells {
        let key = if v.trim().is_empty() {
            "(empty)".to_string()
        } else {
            v.clone()
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut out = String::new();
    peek_row_wrapped(
        &mut out,
        "counts",
        &format!(
            "obs['{col}']  value_counts  n={}  unique={}  (sorted by count, desc)",
            fmt_usize_sep(n),
            pairs.len()
        ),
        term_w,
        PeekStyle::PLAIN,
    );
    let hdr = "  #\tcount\tpct\tcategory";
    let _ = writeln!(
        out,
        "{}",
        if peek_color_enabled() {
            hdr.truecolor(GB_BRIGHT_BLUE.0, GB_BRIGHT_BLUE.1, GB_BRIGHT_BLUE.2)
                .bold()
                .to_string()
        } else {
            hdr.to_string()
        }
    );
    let table_budget = term_w.saturating_sub(4).max(32);
    for (i, (label, cnt)) in pairs.iter().enumerate() {
        let pct = if n > 0 {
            100.0 * (*cnt as f64) / (n as f64)
        } else {
            0.0
        };
        let safe: String = label
            .chars()
            .map(|c| if matches!(c, '\t' | '\n') { ' ' } else { c })
            .collect();
        let pct_s = format!("{:.1}", pct);
        let row0 = if peek_color_enabled() {
            format!(
                "  {}\t{}\t{}%\t",
                (i + 1)
                    .to_string()
                    .truecolor(GB_GRAY.0, GB_GRAY.1, GB_GRAY.2),
                fmt_usize_sep(*cnt)
                    .truecolor(GB_BRIGHT_YELLOW.0, GB_BRIGHT_YELLOW.1, GB_BRIGHT_YELLOW.2)
                    .bold(),
                pct_s.truecolor(GB_BRIGHT_AQUA.0, GB_BRIGHT_AQUA.1, GB_BRIGHT_AQUA.2),
            )
        } else {
            format!("  {}\t{}\t{}%\t", i + 1, fmt_usize_sep(*cnt), pct_s)
        };
        let row0_cols = format!("  {}\t{}\t{}%\t", i + 1, fmt_usize_sep(*cnt), pct_s)
            .chars()
            .count();
        let cat_budget = table_budget.saturating_sub(row0_cols).max(8);
        let cat_lines = wrap_fill_lines(&safe, cat_budget);
        let cat_style = PeekStyle {
            rgb: Some(GB_BRIGHT_GREEN),
            bold: false,
        };
        for (j, cl) in cat_lines.iter().enumerate() {
            if j == 0 {
                let _ = writeln!(out, "{}{}", row0, cat_style.paint(cl));
            } else {
                let pad = " ".repeat(row0_cols);
                let _ = writeln!(out, "{}{}", pad, cat_style.paint(cl));
            }
        }
    }
    Ok(out)
}

fn peek_annadata_h5_report(
    h5: &Group,
    path: &Path,
    file_bytes: u64,
    n_obs: usize,
    n_vars: usize,
    term_w: usize,
    obs_column: Option<&str>,
    genes: &[String],
) -> anyhow::Result<String> {
    let layers = sorted_mapping_keys(h5, "layers");
    let obsm = sorted_mapping_keys(h5, "obsm");
    let obsp = sorted_mapping_keys(h5, "obsp");
    let varm = sorted_mapping_keys(h5, "varm");
    let varp = sorted_mapping_keys(h5, "varp");
    let uns = sorted_mapping_keys(h5, "uns");

    let obs_cols = h5
        .group("obs")
        .ok()
        .map(|g| h5_dataframe_column_names(&g))
        .transpose()?
        .unwrap_or_default();
    let var_cols = h5
        .group("var")
        .ok()
        .map(|g| h5_dataframe_column_names(&g))
        .transpose()?
        .unwrap_or_default();

    let mut raw_note: Option<String> = None;
    if h5.link_exists("raw") {
        if let Ok(raw) = h5.group("raw") {
            let mut s = String::from("raw: present");
            if raw.link_exists("var") {
                if let Ok(vg) = raw.group("var") {
                    if let Ok(cols) = h5_dataframe_column_names(&vg) {
                        if !cols.is_empty() {
                            s.push_str(&format!(" (var: {})", cols.join(", ")));
                        }
                    }
                }
            }
            raw_note = Some(s);
        }
    }

    let mut out = String::new();
    peek_row_wrapped(
        &mut out,
        "path",
        &path.display().to_string(),
        term_w,
        PeekStyle::PATH,
    );
    peek_row_wrapped(
        &mut out,
        "size",
        &format_file_size(file_bytes),
        term_w,
        PeekStyle::SIZE,
    );
    peek_row_wrapped(
        &mut out,
        "format",
        "AnnData HDF5 (.h5ad-compatible layout)",
        term_w,
        PeekStyle::META,
    );
    peek_row_wrapped(
        &mut out,
        "shape",
        &format!("{}×{}", fmt_usize_sep(n_obs), fmt_usize_sep(n_vars)),
        term_w,
        PeekStyle::SHAPE,
    );

    peek_label_grid(&mut out, "obs", &obs_cols, term_w, PeekStyle::OBS_GRID);
    peek_label_grid(&mut out, "var", &var_cols, term_w, PeekStyle::VAR_GRID);
    if !layers.is_empty() {
        peek_row_wrapped(
            &mut out,
            "layers",
            &layers.join(", "),
            term_w,
            PeekStyle::META,
        );
    }
    if !obsm.is_empty() {
        peek_row_wrapped(&mut out, "obsm", &obsm.join(", "), term_w, PeekStyle::META);
    }
    if !obsp.is_empty() {
        peek_row_wrapped(&mut out, "obsp", &obsp.join(", "), term_w, PeekStyle::META);
    }
    if !varm.is_empty() {
        peek_row_wrapped(&mut out, "varm", &varm.join(", "), term_w, PeekStyle::META);
    }
    if !varp.is_empty() {
        peek_row_wrapped(&mut out, "varp", &varp.join(", "), term_w, PeekStyle::META);
    }
    if !uns.is_empty() {
        peek_row_wrapped(&mut out, "uns", &uns.join(", "), term_w, PeekStyle::META);
    }
    if let Some(r) = raw_note {
        peek_row_wrapped(&mut out, "raw", &r, term_w, PeekStyle::META);
    }

    if !genes.is_empty() && obs_column.map(str::trim).unwrap_or("").is_empty() {
        anyhow::bail!("--gene/--genes require --obs COLUMN");
    }

    if let Some(col) = obs_column {
        let col = col.trim();
        anyhow::ensure!(!col.is_empty(), "--obs column name is empty");
        let obs = h5
            .group("obs")
            .context("h5ad: missing obs group — cannot read --obs")?;
        let cells =
            read_h5ad_obs_column_str_h5(&obs, col).with_context(|| format!("read obs[{col:?}]"))?;
        let _ = writeln!(out);
        if !genes.is_empty() {
            out.push_str(&gene_expression_block(
                h5, genes, col, &cells, n_obs, n_vars, term_w,
            )?);
        } else {
            out.push_str(&value_counts_block(col, &cells, n_obs, term_w)?);
        }
    }

    Ok(out)
}

pub fn h5ad_peek_report(
    path: &Path,
    obs_column: Option<&str>,
    genes: &[String],
) -> anyhow::Result<String> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let bytes = meta.len();

    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;
    let tw = peek_terminal_width();

    if let Ok((n_obs, n_vars)) = infer_n_obs_n_vars(&h5) {
        return peek_annadata_h5_report(&h5, path, bytes, n_obs, n_vars, tw, obs_column, genes);
    }

    if let Some(s) = try_peek_tenx_filtered_matrix(&h5, path, bytes, tw)? {
        if obs_column.is_some() || !genes.is_empty() {
            anyhow::bail!(
                "--obs/--gene apply only to AnnData HDF5 with an `obs` group (e.g. .h5ad); \
                 this file looks like a 10x-style `/matrix` HDF5 with no `obs` metadata"
            );
        }
        return Ok(s);
    }

    if obs_column.is_some() || !genes.is_empty() {
        anyhow::bail!(
            "--obs/--gene apply only to AnnData HDF5 with an `obs` group (e.g. .h5ad); \
             this HDF5 layout does not expose `obs` columns"
        );
    }

    peek_generic_hdf5_report(&h5, path, bytes, tw)
}

pub fn print_h5ad_peek(
    path: &Path,
    obs_column: Option<&str>,
    genes: &[String],
) -> anyhow::Result<()> {
    print!("{}", h5ad_peek_report(path, obs_column, genes)?);
    Ok(())
}

const GENE_CSR_CHUNK: usize = 1 << 20;

fn h5_attr_string(attr: &Attribute) -> Option<String> {
    if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<VarLenAscii>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_1d::<VarLenUnicode>() {
        if let Some(s) = v.first() {
            return Some(s.to_string());
        }
    }
    if let Ok(v) = attr.read_1d::<VarLenAscii>() {
        if let Some(s) = v.first() {
            return Some(s.to_string());
        }
    }
    None
}

fn read_string_1d(ds: &Dataset) -> anyhow::Result<Vec<String>> {
    if let Ok(v) = ds.read_1d::<VarLenUnicode>() {
        return Ok(v.iter().map(|s| s.to_string()).collect());
    }
    if let Ok(v) = ds.read_1d::<VarLenAscii>() {
        return Ok(v.iter().map(|s| s.to_string()).collect());
    }
    anyhow::bail!("expected a variable-length string dataset")
}

fn read_var_index(var: &Group) -> anyhow::Result<Vec<String>> {
    let name = axis_index_dataset_name(var).unwrap_or_else(|| "_index".to_string());
    let ds = var
        .dataset(&name)
        .with_context(|| format!("var index dataset {name:?}"))?;
    read_string_1d(&ds).with_context(|| format!("read var index {name:?}"))
}

fn resolve_gene_index(names: &[String], query: &str) -> anyhow::Result<(usize, String)> {
    let q = query.trim();
    anyhow::ensure!(!q.is_empty(), "--gene name is empty");
    let exact: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() == q)
        .map(|(i, _)| i)
        .collect();
    if exact.len() == 1 {
        return Ok((exact[0], names[exact[0]].clone()));
    }
    if exact.len() > 1 {
        anyhow::bail!(
            "gene {q:?} matches {} var columns; var names are not unique",
            exact.len()
        );
    }
    let folded: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.eq_ignore_ascii_case(q))
        .map(|(i, _)| i)
        .collect();
    match folded.len() {
        0 => anyhow::bail!("gene {q:?} not found in var (n_vars={})", names.len()),
        1 => Ok((folded[0], names[folded[0]].clone())),
        n => {
            let shown: Vec<&str> = folded.iter().take(8).map(|&i| names[i].as_str()).collect();
            anyhow::bail!(
                "gene {q:?} matches {n} var names ({}); pass the exact symbol",
                shown.join(", ")
            )
        }
    }
}

fn read_usize_range(ds: &Dataset, start: usize, end: usize) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(end >= start, "invalid dataset range {start}..{end}");
    if start == end {
        return Ok(Vec::new());
    }
    if let Ok(a) = ds.read_slice_1d::<i64, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| v.max(0) as usize).collect());
    }
    if let Ok(a) = ds.read_slice_1d::<i32, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| v.max(0) as usize).collect());
    }
    if let Ok(a) = ds.read_slice_1d::<u64, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    if let Ok(a) = ds.read_slice_1d::<u32, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    anyhow::bail!("unsupported integer dtype")
}

fn read_f64_range(ds: &Dataset, start: usize, end: usize) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(end >= start, "invalid dataset range {start}..{end}");
    if start == end {
        return Ok(Vec::new());
    }
    if let Ok(a) = ds.read_slice_1d::<f64, _>(s![start..end]) {
        return Ok(a.to_vec());
    }
    if let Ok(a) = ds.read_slice_1d::<f32, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| f64::from(v)).collect());
    }
    if let Ok(a) = ds.read_slice_1d::<i32, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| f64::from(v)).collect());
    }
    if let Ok(a) = ds.read_slice_1d::<i64, _>(s![start..end]) {
        return Ok(a.iter().map(|&v| v as f64).collect());
    }
    anyhow::bail!("unsupported numeric dtype")
}

fn csc_gene_column(
    indptr_ds: &Dataset,
    indices_ds: &Dataset,
    data_ds: &Dataset,
    col: usize,
    n_obs: usize,
    n_vars: usize,
) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(
        indptr_ds.size() == n_vars + 1,
        "csc indptr len {} != n_vars+1",
        indptr_ds.size()
    );
    let pair = read_usize_range(indptr_ds, col, col + 2)?;
    anyhow::ensure!(pair.len() == 2, "csc indptr pair at column {col}");
    let (start, end) = (pair[0], pair[1]);
    anyhow::ensure!(end >= start, "csc indptr decreases at column {col}");
    let rows = read_usize_range(indices_ds, start, end)?;
    let vals = read_f64_range(data_ds, start, end)?;
    anyhow::ensure!(
        rows.len() == vals.len(),
        "csc indices/data length mismatch at column {col}"
    );
    let mut out = vec![0.0; n_obs];
    for (row, val) in rows.into_iter().zip(vals) {
        if row < n_obs {
            out[row] += val;
        }
    }
    Ok(out)
}

fn dense_row_slab(ds: &Dataset, start: usize, end: usize) -> anyhow::Result<Vec<f64>> {
    if start == end {
        return Ok(Vec::new());
    }
    if let Ok(a) = ds.read_slice_2d::<f32, _>(s![start..end, ..]) {
        return Ok(a.iter().copied().map(f64::from).collect());
    }
    if let Ok(a) = ds.read_slice_2d::<f64, _>(s![start..end, ..]) {
        return Ok(a.iter().copied().collect());
    }
    if let Ok(a) = ds.read_slice_2d::<i32, _>(s![start..end, ..]) {
        return Ok(a.iter().map(|&v| f64::from(v)).collect());
    }
    if let Ok(a) = ds.read_slice_2d::<i64, _>(s![start..end, ..]) {
        return Ok(a.iter().map(|&v| v as f64).collect());
    }
    anyhow::bail!("unsupported X dtype for a gene-column read")
}

fn dense_gene_columns(
    ds: &Dataset,
    cols: &[usize],
    n_obs: usize,
    n_vars: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 2, "X: expected 2D dataset, got shape {sh:?}");
    anyhow::ensure!(
        sh[0] == n_obs && sh[1] == n_vars,
        "X shape {sh:?} does not match obs×var {n_obs}×{n_vars}"
    );
    for &col in cols {
        anyhow::ensure!(
            col < n_vars,
            "gene column {col} out of range (n_vars={n_vars})"
        );
    }
    let mut out = vec![vec![0.0; n_obs]; cols.len()];
    let row_bytes = n_vars.saturating_mul(8).max(1);
    let slab_rows = (64 * 1024 * 1024 / row_bytes).clamp(1, 8192);
    let mut row = 0usize;
    while row < n_obs {
        let end = (row + slab_rows).min(n_obs);
        let slab = dense_row_slab(ds, row, end)?;
        let width = end - row;
        anyhow::ensure!(
            slab.len() == width * n_vars,
            "dense X slab length {} != {}×{n_vars}",
            slab.len(),
            width
        );
        for r in 0..width {
            let base = r * n_vars;
            for (i, &col) in cols.iter().enumerate() {
                out[i][row + r] = slab[base + col];
            }
        }
        row = end;
    }
    Ok(out)
}

/// CSR scatters each gene across rows, so one chunked scan of `indices`/`data` fills every
/// requested column. The dense matrix is never built.
fn csr_gene_columns(
    indptr_ds: &Dataset,
    indices_ds: &Dataset,
    data_ds: &Dataset,
    cols: &[usize],
    n_obs: usize,
    n_vars: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let indptr_len = indptr_ds.size();
    anyhow::ensure!(
        indptr_len == n_obs + 1,
        "csr indptr len {indptr_len} != n_obs+1"
    );
    let nnz = indices_ds.size();
    anyhow::ensure!(
        data_ds.size() == nnz,
        "csr indices/data length mismatch ({nnz} vs {})",
        data_ds.size()
    );
    let mut slot = vec![usize::MAX; n_vars];
    for (i, &col) in cols.iter().enumerate() {
        anyhow::ensure!(
            col < n_vars,
            "gene column {col} out of range (n_vars={n_vars})"
        );
        slot[col] = i;
    }
    let indptr = read_usize_range(indptr_ds, 0, indptr_len)?;
    let mut out = vec![vec![0.0; n_obs]; cols.len()];
    let mut row = 0usize;
    let mut k = 0usize;
    while k < nnz {
        let end = (k + GENE_CSR_CHUNK).min(nnz);
        let indices = read_usize_range(indices_ds, k, end)?;
        let data = read_f64_range(data_ds, k, end)?;
        anyhow::ensure!(
            indices.len() == data.len(),
            "csr chunk length mismatch at {k}..{end}"
        );
        for (j, col_i) in indices.into_iter().enumerate() {
            let abs = k + j;
            while row + 1 < indptr.len() && indptr[row + 1] <= abs {
                row += 1;
            }
            if col_i < n_vars && row < n_obs {
                let slot_i = slot[col_i];
                if slot_i != usize::MAX {
                    out[slot_i][row] += data[j];
                }
            }
        }
        k = end;
    }
    Ok(out)
}

fn sparse_gene_columns(
    g: &Group,
    cols: &[usize],
    n_obs: usize,
    n_vars: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    if let Some((nr, nc)) = read_shape_attr_2d(g) {
        anyhow::ensure!(
            nr == n_obs && nc == n_vars,
            "sparse X shape {nr}×{nc} does not match obs×var {n_obs}×{n_vars}"
        );
    }
    let indptr_ds = g.dataset("indptr").context("sparse X indptr")?;
    let indices_ds = g.dataset("indices").context("sparse X indices")?;
    let data_ds = g.dataset("data").context("sparse X data")?;
    let indptr_len = indptr_ds.size();
    let enc = g
        .attr("encoding-type")
        .ok()
        .and_then(|a| h5_attr_string(&a))
        .unwrap_or_default();
    let csr_by_len = indptr_len == n_obs + 1;
    let csc_by_len = indptr_len == n_vars + 1;
    if enc == "csc_matrix" || (enc.is_empty() && csc_by_len && !csr_by_len) {
        let mut out = Vec::with_capacity(cols.len());
        for &col in cols {
            out.push(csc_gene_column(
                &indptr_ds,
                &indices_ds,
                &data_ds,
                col,
                n_obs,
                n_vars,
            )?);
        }
        return Ok(out);
    }
    if enc == "csr_matrix" || (enc.is_empty() && csr_by_len) {
        return csr_gene_columns(&indptr_ds, &indices_ds, &data_ds, cols, n_obs, n_vars);
    }
    anyhow::bail!(
        "X sparse group: encoding-type {enc:?}, indptr len {indptr_len} (expected csr n_obs+1 or csc n_vars+1)"
    )
}

fn read_x_gene_columns(
    root: &Group,
    cols: &[usize],
    n_obs: usize,
    n_vars: usize,
) -> anyhow::Result<Vec<Vec<f64>>> {
    anyhow::ensure!(root.link_exists("X"), "h5ad: missing X");
    match root.loc_type_by_name("X").context("X link type")? {
        LocationType::Dataset => dense_gene_columns(&root.dataset("X")?, cols, n_obs, n_vars),
        LocationType::Group => sparse_gene_columns(&root.group("X")?, cols, n_obs, n_vars),
        _ => anyhow::bail!("X is not a dataset or sparse matrix group"),
    }
}

fn uns_marks_log1p(root: &Group) -> bool {
    root.group("uns")
        .ok()
        .is_some_and(|uns| uns.link_exists("log1p"))
}

fn column_is_already_log1p(values: &[f64]) -> bool {
    let mut sample = Vec::new();
    let step = (values.len() / 4096).max(1);
    for v in values.iter().step_by(step) {
        if v.is_finite() {
            sample.push(*v);
            if sample.len() == 4096 {
                break;
            }
        }
    }
    if sample.is_empty() {
        return false;
    }
    let mx = sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n_int = sample
        .iter()
        .filter(|v| (*v - v.round()).abs() < 1e-5)
        .count();
    let frac = n_int as f64 / sample.len() as f64;
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sample[sample.len() / 2];
    if mx > 30.0 || (frac > 0.72 && mx > 12.0) {
        return false;
    }
    mx <= 12.0 && med <= 3.5 && frac < 0.55
}

struct GeneGroupRow {
    label: String,
    n: usize,
    min: f64,
    mean: f64,
    median: f64,
    max: f64,
}

fn obs_group_key(label: &str) -> &str {
    if label.trim().is_empty() {
        "(empty)"
    } else {
        label
    }
}

fn median_of(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

fn gene_group_rows(labels: &[String], raw: &[f64], apply_log1p: bool) -> Vec<GeneGroupRow> {
    let mut idxs: Vec<usize> = (0..labels.len()).collect();
    idxs.sort_unstable_by(|&a, &b| obs_group_key(&labels[a]).cmp(obs_group_key(&labels[b])));
    let mut rows = Vec::new();
    let mut i = 0usize;
    while i < idxs.len() {
        let key = obs_group_key(&labels[idxs[i]]).to_string();
        let mut j = i + 1;
        while j < idxs.len() && obs_group_key(&labels[idxs[j]]) == key {
            j += 1;
        }
        let mut vals = Vec::with_capacity(j - i);
        for &k in &idxs[i..j] {
            let v = raw[k];
            let y = if apply_log1p && v.is_finite() {
                v.ln_1p()
            } else {
                v
            };
            if y.is_finite() {
                vals.push(y);
            }
        }
        let n = j - i;
        let mean = if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        let median = median_of(&mut vals);
        let (min, max) = if vals.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (vals[0], vals[vals.len() - 1])
        };
        rows.push(GeneGroupRow {
            label: key,
            n,
            min,
            mean,
            median,
            max,
        });
        i = j;
    }
    rows.sort_by(|a, b| {
        b.mean
            .partial_cmp(&a.mean)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });
    rows
}

fn fmt_stat(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "NA".to_string()
    }
}

fn pad_left(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - n))
    }
}

fn pad_right(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - n))
    }
}

fn trunc_chars(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let take = max_chars.saturating_sub(1);
    format!("{}…", s.chars().take(take).collect::<String>())
}

fn mean_bar(frac: f64, width: usize) -> (String, String) {
    if width == 0 {
        return (String::new(), String::new());
    }
    let frac = if frac.is_finite() {
        frac.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = ((frac * width as f64).round() as usize).min(width);
    ("█".repeat(filled), "░".repeat(width - filled))
}

fn paint_rgb(s: &str, rgb: (u8, u8, u8), bold: bool) -> String {
    if !peek_color_enabled() {
        return s.to_string();
    }
    let c = s.truecolor(rgb.0, rgb.1, rgb.2);
    if bold {
        c.bold().to_string()
    } else {
        c.to_string()
    }
}

fn format_gene_groups(
    query: &str,
    resolved: &str,
    obs_col: &str,
    stored_log1p: bool,
    rows: &[GeneGroupRow],
    n_obs: usize,
    term_w: usize,
) -> String {
    let mut out = String::new();
    let gene_shown = if resolved != query && resolved.eq_ignore_ascii_case(query) {
        format!("{resolved}  (query {query})")
    } else {
        resolved.to_string()
    };
    peek_row_wrapped(&mut out, "gene", &gene_shown, term_w, PeekStyle::VAR_GRID);
    let how = if stored_log1p {
        "log1p  (X already log1p)"
    } else {
        "log1p(X)"
    };
    peek_row_wrapped(
        &mut out,
        "expr",
        &format!(
            "{how}  obs['{obs_col}']  n={}  groups={}  bars scaled to the highest mean",
            fmt_usize_sep(n_obs),
            rows.len()
        ),
        term_w,
        PeekStyle::PLAIN,
    );

    let n_w = rows
        .iter()
        .map(|r| fmt_usize_sep(r.n).chars().count())
        .max()
        .unwrap_or(1)
        .max("n".len());
    let min_w = stat_col_width(rows, "min", |r| r.min);
    let mean_w = stat_col_width(rows, "mean", |r| r.mean);
    let med_w = stat_col_width(rows, "median", |r| r.median);
    let max_w = stat_col_width(rows, "max", |r| r.max);
    let stats_w = n_w + min_w + mean_w + med_w + max_w + 8;
    let label_natural = rows
        .iter()
        .map(|r| r.label.chars().count())
        .max()
        .unwrap_or(1)
        .max("group".len());
    let mut label_w = label_natural.min(28);
    let mut bar_w = term_w.saturating_sub(6 + label_w + stats_w).clamp(8, 36);
    while 2 + label_w + 2 + bar_w + 2 + stats_w > term_w && bar_w > 8 {
        bar_w -= 1;
    }
    while 2 + label_w + 2 + bar_w + 2 + stats_w > term_w && label_w > 5 {
        label_w -= 1;
    }
    let scale = rows
        .iter()
        .filter_map(|r| r.mean.is_finite().then_some(r.mean))
        .fold(0.0_f64, f64::max)
        .max(0.0);

    let hdr = format!(
        "  {}  {}  {}  {}  {}  {}  {}",
        pad_right("group", label_w),
        " ".repeat(bar_w),
        pad_left("n", n_w),
        pad_left("min", min_w),
        pad_left("mean", mean_w),
        pad_left("median", med_w),
        pad_left("max", max_w),
    );
    let _ = writeln!(
        out,
        "{}",
        if peek_color_enabled() {
            hdr.truecolor(GB_BRIGHT_BLUE.0, GB_BRIGHT_BLUE.1, GB_BRIGHT_BLUE.2)
                .bold()
                .to_string()
        } else {
            hdr
        }
    );
    for row in rows {
        let safe: String = row
            .label
            .chars()
            .map(|c| if matches!(c, '\t' | '\n') { ' ' } else { c })
            .collect();
        let frac = if scale > 0.0 && row.mean.is_finite() {
            (row.mean / scale).max(0.0)
        } else {
            0.0
        };
        let (fill, track) = mean_bar(frac, bar_w);
        let _ = writeln!(
            out,
            "  {}  {}{}  {}  {}  {}  {}  {}",
            paint_rgb(
                &pad_right(&trunc_chars(&safe, label_w), label_w),
                GB_BRIGHT_GREEN,
                false
            ),
            paint_rgb(&fill, GB_BRIGHT_GREEN, true),
            paint_rgb(&track, GB_GRAY, false),
            paint_rgb(
                &pad_left(&fmt_usize_sep(row.n), n_w),
                GB_BRIGHT_YELLOW,
                true
            ),
            paint_rgb(&pad_left(&fmt_stat(row.min), min_w), GB_GRAY, false),
            paint_rgb(
                &pad_left(&fmt_stat(row.mean), mean_w),
                GB_BRIGHT_YELLOW,
                true
            ),
            pad_left(&fmt_stat(row.median), med_w),
            paint_rgb(&pad_left(&fmt_stat(row.max), max_w), GB_BRIGHT_AQUA, false),
        );
    }
    out
}

fn stat_col_width(
    rows: &[GeneGroupRow],
    header: &str,
    value: impl Fn(&GeneGroupRow) -> f64,
) -> usize {
    rows.iter()
        .map(|r| fmt_stat(value(r)).chars().count())
        .max()
        .unwrap_or(header.len())
        .max(header.len())
}

struct MarkerRow {
    label: String,
    n: usize,
    means: Vec<f64>,
}

fn marker_rows(labels: &[String], columns: &[Vec<f64>], apply_log1p: bool) -> Vec<MarkerRow> {
    let n_genes = columns.len();
    let mut idxs: Vec<usize> = (0..labels.len()).collect();
    idxs.sort_unstable_by(|&a, &b| obs_group_key(&labels[a]).cmp(obs_group_key(&labels[b])));
    let mut rows = Vec::new();
    let mut i = 0usize;
    while i < idxs.len() {
        let key = obs_group_key(&labels[idxs[i]]).to_string();
        let mut j = i + 1;
        while j < idxs.len() && obs_group_key(&labels[idxs[j]]) == key {
            j += 1;
        }
        let mut sums = vec![0.0; n_genes];
        let mut counts = vec![0usize; n_genes];
        for &k in &idxs[i..j] {
            for g in 0..n_genes {
                let v = columns[g][k];
                let y = if apply_log1p && v.is_finite() {
                    v.ln_1p()
                } else {
                    v
                };
                if y.is_finite() {
                    sums[g] += y;
                    counts[g] += 1;
                }
            }
        }
        let means = sums
            .iter()
            .zip(counts)
            .map(|(&sum, n)| if n == 0 { f64::NAN } else { sum / n as f64 })
            .collect();
        rows.push(MarkerRow {
            label: key,
            n: j - i,
            means,
        });
        i = j;
    }
    rows.sort_by(|a, b| b.n.cmp(&a.n).then_with(|| a.label.cmp(&b.label)));
    rows
}

fn fmt_marker_mean(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "NA".to_string()
    }
}

fn format_marker_panel(
    resolved: &[String],
    obs_col: &str,
    stored_log1p: bool,
    rows: &[MarkerRow],
    n_obs: usize,
    term_w: usize,
) -> String {
    let mut out = String::new();
    peek_row_wrapped(
        &mut out,
        "markers",
        &resolved.join("  "),
        term_w,
        PeekStyle::VAR_GRID,
    );
    let how = if stored_log1p {
        "log1p  (X already log1p)"
    } else {
        "log1p(X)"
    };
    peek_row_wrapped(
        &mut out,
        "expr",
        &format!(
            "{how}  obs['{obs_col}']  n={}  groups={}  (bar = mean, scaled within each gene)",
            fmt_usize_sep(n_obs),
            rows.len()
        ),
        term_w,
        PeekStyle::PLAIN,
    );

    let n_w = rows
        .iter()
        .map(|r| fmt_usize_sep(r.n).chars().count())
        .max()
        .unwrap_or(1)
        .max("n".len());
    let label_natural = rows
        .iter()
        .map(|r| r.label.chars().count())
        .max()
        .unwrap_or(1)
        .max("group".len());
    let label_w = label_natural.min(22).max("group".len());
    let mean_w = rows
        .iter()
        .flat_map(|r| r.means.iter().copied())
        .map(fmt_marker_mean)
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(4)
        .max(4);
    let bar_w = 4usize;
    let cell_body = bar_w + 1 + mean_w;
    let mut col_w = Vec::with_capacity(resolved.len());
    for name in resolved {
        col_w.push(name.chars().count().max(cell_body));
    }
    let prefix = 2 + label_w + 2 + n_w + 2;
    let mut batches: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0usize;
    while start < resolved.len() {
        let mut used = 0usize;
        let mut end = start;
        while end < resolved.len() {
            let gap = if end == start { 0 } else { 2 };
            let need = gap + col_w[end];
            if end > start && used + need + prefix > term_w {
                break;
            }
            used += need;
            end += 1;
        }
        if end == start {
            end = start + 1;
        }
        batches.push(start..end);
        start = end;
    }

    let scales: Vec<f64> = (0..resolved.len())
        .map(|g| {
            rows.iter()
                .filter_map(|r| r.means.get(g).copied())
                .filter(|m| m.is_finite())
                .fold(0.0_f64, f64::max)
                .max(0.0)
        })
        .collect();

    for batch in batches {
        let _ = writeln!(out);
        let mut hdr = format!("  {}  {}", pad_right("group", label_w), pad_left("n", n_w));
        for g in batch.clone() {
            hdr.push_str("  ");
            hdr.push_str(&pad_right(&trunc_chars(&resolved[g], col_w[g]), col_w[g]));
        }
        let _ = writeln!(
            out,
            "{}",
            if peek_color_enabled() {
                hdr.truecolor(GB_BRIGHT_BLUE.0, GB_BRIGHT_BLUE.1, GB_BRIGHT_BLUE.2)
                    .bold()
                    .to_string()
            } else {
                hdr
            }
        );
        for row in rows {
            let safe: String = row
                .label
                .chars()
                .map(|c| if matches!(c, '\t' | '\n') { ' ' } else { c })
                .collect();
            let mut line = format!(
                "  {}  {}",
                paint_rgb(
                    &pad_right(&trunc_chars(&safe, label_w), label_w),
                    GB_BRIGHT_GREEN,
                    false
                ),
                paint_rgb(
                    &pad_left(&fmt_usize_sep(row.n), n_w),
                    GB_BRIGHT_YELLOW,
                    true
                ),
            );
            for g in batch.clone() {
                let mean = row.means.get(g).copied().unwrap_or(f64::NAN);
                let frac = if scales[g] > 0.0 && mean.is_finite() {
                    (mean / scales[g]).max(0.0)
                } else {
                    0.0
                };
                let (fill, track) = mean_bar(frac, bar_w);
                let body = format!(
                    "{}{} {}",
                    paint_rgb(&fill, GB_BRIGHT_GREEN, true),
                    paint_rgb(&track, GB_GRAY, false),
                    paint_rgb(&fmt_marker_mean(mean), GB_BRIGHT_YELLOW, true),
                );
                let pad = col_w[g].saturating_sub(cell_body);
                line.push_str("  ");
                line.push_str(&body);
                if pad > 0 {
                    line.push_str(&" ".repeat(pad));
                }
            }
            let _ = writeln!(out, "{line}");
        }
    }
    out
}

fn columns_already_log1p(columns: &[Vec<f64>]) -> bool {
    let mut sample = Vec::new();
    for col in columns {
        let step = (col.len() / 1024).max(1);
        for v in col.iter().step_by(step) {
            if v.is_finite() {
                sample.push(*v);
                if sample.len() == 4096 {
                    return column_is_already_log1p(&sample);
                }
            }
        }
    }
    column_is_already_log1p(&sample)
}

fn gene_expression_block(
    h5: &Group,
    genes: &[String],
    obs_col: &str,
    labels: &[String],
    n_obs: usize,
    n_vars: usize,
    term_w: usize,
) -> anyhow::Result<String> {
    anyhow::ensure!(!genes.is_empty(), "--gene name is empty");
    anyhow::ensure!(
        labels.len() == n_obs,
        "obs['{obs_col}'] length {} != n_obs {n_obs}",
        labels.len()
    );
    let var = h5
        .group("var")
        .context("h5ad: missing var group — cannot resolve --gene")?;
    let var_names = read_var_index(&var)?;
    anyhow::ensure!(
        var_names.len() == n_vars,
        "var index length {} != n_vars {n_vars}",
        var_names.len()
    );
    let mut idxs = Vec::with_capacity(genes.len());
    let mut resolved = Vec::with_capacity(genes.len());
    for gene in genes {
        let (idx, name) = resolve_gene_index(&var_names, gene)?;
        idxs.push(idx);
        resolved.push(name);
    }
    let raw = read_x_gene_columns(h5, &idxs, n_obs, n_vars)
        .with_context(|| format!("read X columns for {}", resolved.join(", ")))?;
    for (col, name) in raw.iter().zip(&resolved) {
        anyhow::ensure!(
            col.len() == n_obs,
            "gene column {name} length {} != n_obs {n_obs}",
            col.len()
        );
    }
    let stored = uns_marks_log1p(h5) || columns_already_log1p(&raw);
    if genes.len() == 1 {
        let rows = gene_group_rows(labels, &raw[0], !stored);
        return Ok(format_gene_groups(
            genes[0].trim(),
            &resolved[0],
            obs_col,
            stored,
            &rows,
            n_obs,
            term_w,
        ));
    }
    let rows = marker_rows(labels, &raw, !stored);
    Ok(format_marker_panel(
        &resolved, obs_col, stored, &rows, n_obs, term_w,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::ArrayData;
    use anndata::{AnnData, AnnDataOp};
    use anndata_hdf5::H5;
    use ndarray::Array2;
    use polars::prelude::{DataFrame, NamedFrom, Series};
    #[test]
    fn peek_tiny_h5ad_report() {
        let dir = std::env::temp_dir().join(format!("h5ad_peek_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let a = AnnData::<H5>::new(&path).unwrap();
        a.set_obs_names(vec!["c0".into(), "c1".into()].into())
            .unwrap();
        a.set_var_names(vec!["G0".into(), "G1".into()].into())
            .unwrap();
        let obs = DataFrame::new(vec![
            Series::new("cell_type".into(), vec!["a".to_string(), "b".to_string()]).into(),
        ])
        .unwrap();
        a.set_obs(obs).unwrap();
        let var = DataFrame::new(vec![Series::new("mt".into(), vec![false, true]).into()]).unwrap();
        a.set_var(var).unwrap();
        let gem = Array2::<f64>::zeros((2, 2));
        a.set_x(ArrayData::from(gem)).unwrap();
        a.close().unwrap();

        let s = h5ad_peek_report(&path, None, &[]).unwrap();
        assert!(s.contains("2×2"), "report:\n{s}");
        assert!(s.contains("cell_type"));
        assert!(s.contains("mt"));
        assert!(s.contains("›"), "expected tag prefix, got:\n{s}");

        let s2 = h5ad_peek_report(&path, Some("cell_type"), &[]).unwrap();
        assert!(s2.contains("value_counts"));
        assert!(s2.contains("a"));
        assert!(s2.contains("b"));
    }

    #[test]
    fn peek_tenx_matrix_h5_e14s_if_present() {
        let p = Path::new("/tmp/E14S.h5");
        if !p.is_file() {
            return;
        }
        let s = h5ad_peek_report(p, None, &[]).expect("peek");
        assert!(s.contains("10x sparse matrix"), "report:\n{s}");
        assert!(s.contains("5,292") || s.contains("5292"), "report:\n{s}");
    }

    #[test]
    fn gene_resolve_prefers_exact_and_folds_case() {
        let names = vec!["Gata3".into(), "GATA3".into(), "ACTB".into()];
        let (i, s) = resolve_gene_index(&names, "GATA3").unwrap();
        assert_eq!((i, s.as_str()), (1, "GATA3"));
        let (i, s) = resolve_gene_index(&names, "actb").unwrap();
        assert_eq!((i, s.as_str()), (2, "ACTB"));
        assert!(resolve_gene_index(&names, "gata3").is_err());
        assert!(resolve_gene_index(&names, "SOX2").is_err());
    }

    fn write_grouped_expr(path: &std::path::Path, x: ArrayData) {
        let a = AnnData::<H5>::new(path).unwrap();
        a.set_obs_names(vec!["c0".into(), "c1".into(), "c2".into()].into())
            .unwrap();
        a.set_var_names(vec!["Actb".into(), "Gata3".into()].into())
            .unwrap();
        let obs = DataFrame::new(vec![
            Series::new(
                "cell_type_2".into(),
                vec!["A".to_string(), "A".to_string(), "B".to_string()],
            )
            .into(),
        ])
        .unwrap();
        a.set_obs(obs).unwrap();
        a.set_x(x).unwrap();
        a.close().unwrap();
    }

    fn assert_gata3_log1p_groups(report: &str) {
        assert!(report.contains("Gata3"), "report:\n{report}");
        assert!(
            report.to_ascii_lowercase().contains("query gata3"),
            "report:\n{report}"
        );
        assert!(report.contains("log1p(X)"), "report:\n{report}");
        assert!(report.contains("0.693"), "report:\n{report}");
        assert!(report.contains("1.040"), "report:\n{report}");
        assert!(report.contains("1.386"), "report:\n{report}");
        assert!(report.contains("0.000"), "report:\n{report}");
        assert!(report.contains('█'), "report:\n{report}");
        assert!(report.contains('░'), "report:\n{report}");
        assert!(
            !report.contains("6.908") && !report.contains("6.912"),
            "decoy gene leaked into Gata3:\n{report}"
        );
    }

    #[test]
    fn peek_gene_dense_log1p_by_obs_ignores_case() {
        let dir = std::env::temp_dir().join(format!("h5ad_peek_gene_dense_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let gem = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 5.0, 3.0, 0.0, 0.0]).unwrap();
        write_grouped_expr(&path, ArrayData::from(gem));
        let gata3 = vec!["gata3".to_string()];
        let s = h5ad_peek_report(&path, Some("cell_type_2"), &gata3).unwrap();
        assert_gata3_log1p_groups(&s);
        assert!(s.contains("cell_type_2"), "report:\n{s}");
        assert!(!s.contains("value_counts"), "report:\n{s}");
        let missing = h5ad_peek_report(&path, Some("cell_type_2"), &["SOX2".to_string()]);
        assert!(missing.is_err(), "expected missing gene to fail");
    }

    #[test]
    fn peek_gene_csr_reads_only_the_requested_column() {
        use nalgebra_sparse::CsrMatrix;
        let dir = std::env::temp_dir().join(format!("h5ad_peek_gene_csr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let csr = CsrMatrix::try_from_csr_data(
            3,
            2,
            vec![0, 1, 3, 3],
            vec![1, 0, 1],
            vec![1.0_f64, 1000.0, 3.0],
        )
        .unwrap();
        write_grouped_expr(&path, ArrayData::from(csr));
        let s = h5ad_peek_report(&path, Some("cell_type_2"), &["GATA3".to_string()]).unwrap();
        assert_gata3_log1p_groups(&s);
    }

    #[test]
    fn peek_gene_csc_reads_only_the_requested_column() {
        use nalgebra_sparse::CscMatrix;
        let dir = std::env::temp_dir().join(format!("h5ad_peek_gene_csc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let csc = CscMatrix::try_from_csc_data(
            3,
            2,
            vec![0, 1, 3],
            vec![1, 0, 1],
            vec![1000.0_f64, 1.0, 3.0],
        )
        .unwrap();
        write_grouped_expr(&path, ArrayData::from(csc));
        let s = h5ad_peek_report(&path, Some("cell_type_2"), &["gata3".to_string()]).unwrap();
        assert_gata3_log1p_groups(&s);
    }

    #[test]
    fn peek_gene_keeps_values_already_in_log1p_space() {
        let dir = std::env::temp_dir().join(format!("h5ad_peek_gene_log_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let gem = Array2::from_shape_vec((3, 2), vec![0.0, 0.5, 0.2, 1.5, 0.0, 0.0]).unwrap();
        write_grouped_expr(&path, ArrayData::from(gem));
        let s = h5ad_peek_report(&path, Some("cell_type_2"), &["gata3".to_string()]).unwrap();
        assert!(s.contains("already log1p"), "report:\n{s}");
        assert!(s.contains("1.000"), "report:\n{s}");
        assert!(!s.contains("log1p(X)"), "report:\n{s}");
    }

    fn write_marker_h5ad(path: &std::path::Path, x: ArrayData) {
        let a = AnnData::<H5>::new(path).unwrap();
        a.set_obs_names(vec!["c0".into(), "c1".into(), "c2".into()].into())
            .unwrap();
        a.set_var_names(vec!["Actb".into(), "Gata3".into(), "Decoy".into()].into())
            .unwrap();
        let obs = DataFrame::new(vec![
            Series::new(
                "cell_type".into(),
                vec!["A".to_string(), "A".to_string(), "B".to_string()],
            )
            .into(),
        ])
        .unwrap();
        a.set_obs(obs).unwrap();
        a.set_x(x).unwrap();
        a.close().unwrap();
    }

    fn assert_marker_panel(report: &str) {
        assert!(report.contains("markers"), "report:\n{report}");
        assert!(report.contains("Actb"), "report:\n{report}");
        assert!(report.contains("Gata3"), "report:\n{report}");
        assert!(report.contains("1.04"), "report:\n{report}");
        assert!(report.contains("0.69"), "report:\n{report}");
        assert!(report.contains("0.00"), "report:\n{report}");
        assert!(report.contains('█'), "report:\n{report}");
        assert!(report.contains('░'), "report:\n{report}");
        assert!(
            !report.contains("Decoy") && !report.contains("6.91"),
            "decoy gene leaked:\n{report}"
        );
        let a_line = report
            .lines()
            .find(|l| l.contains("  A") && l.contains("1.04"))
            .unwrap_or_else(|| panic!("missing group A:\n{report}"));
        let b_line = report
            .lines()
            .find(|l| l.contains("  B") && l.contains("0.69"))
            .unwrap_or_else(|| panic!("missing group B:\n{report}"));
        let a_gata = a_line.split("1.04").next().unwrap_or("");
        assert!(
            a_gata.contains('█'),
            "Gata3 peak should fill the bar:\n{a_line}"
        );
        let b_actb = b_line.split("0.69").next().unwrap_or("");
        assert!(
            b_actb.contains('█'),
            "Actb peak should fill the bar:\n{b_line}"
        );
    }

    #[test]
    fn peek_marker_panel_scales_each_gene_and_skips_decoy() {
        let dir = std::env::temp_dir().join(format!("h5ad_peek_markers_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let gem = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, 1000.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1000.0],
        )
        .unwrap();
        write_marker_h5ad(&path, ArrayData::from(gem));
        let genes = vec!["actb".to_string(), "gata3".to_string()];
        let s = h5ad_peek_report(&path, Some("cell_type"), &genes).unwrap();
        assert_marker_panel(&s);
    }

    #[test]
    fn peek_marker_panel_csr_one_pass_skips_decoy() {
        use nalgebra_sparse::CsrMatrix;
        let dir =
            std::env::temp_dir().join(format!("h5ad_peek_markers_csr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.h5ad");
        let csr = CsrMatrix::try_from_csr_data(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![1, 2, 1, 0, 2],
            vec![1.0_f64, 1000.0, 3.0, 1.0, 1000.0],
        )
        .unwrap();
        write_marker_h5ad(&path, ArrayData::from(csr));
        let genes = vec!["actb".to_string(), "gata3".to_string()];
        let s = h5ad_peek_report(&path, Some("cell_type"), &genes).unwrap();
        assert_marker_panel(&s);
    }
}
