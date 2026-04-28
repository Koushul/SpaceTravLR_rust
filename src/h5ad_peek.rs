use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

use anyhow::Context;
use colored::Colorize;
use crate::adata_terminal_scatter::read_h5ad_obs_column_str_h5;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset, File as H5File, Group, LocationType};
use ndarray::Array1;

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
    let names: Array1<VarLenUnicode> = a.read_1d().ok()?;
    if names.is_empty() {
        return None;
    }
    Some(names[[0]].to_string())
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

fn byte_index_at_char_count(s: &str, max_chars: usize) -> usize {
    let mut n = 0usize;
    for (i, _) in s.char_indices() {
        if n >= max_chars {
            return s.floor_char_boundary(i);
        }
        n += 1;
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
                cut = rest.floor_char_boundary(sp + 1);
            }
        }
        let (line, tail) = rest.split_at(cut);
        lines.push(line.trim_end().to_string());
        rest = tail.trim_start();
    }
    lines
}

fn peek_row_wrapped(out: &mut String, label: &str, value: &str, term_w: usize, value_style: PeekStyle) {
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
    let ncols = (usable / col_w).max(1).min(12);
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

fn peek_label_grid(out: &mut String, label: &str, names: &[String], term_w: usize, grid_style: PeekStyle) {
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
                2 => return Ok(Some((sh[0], sh[1]))),
                1 => return Ok(Some((sh[0], 1))),
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
                anyhow::ensure!(
                    ip_len >= 2,
                    "csr X: indptr length {ip_len} is too short"
                );
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
        let o = n_obs_idx.context(
            "could not infer n_obs: X has no row count and obs index is missing",
        )?;
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
    let mut names = g.member_names().with_context(|| "list HDF5 group members")?;
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
    peek_row_wrapped(&mut out, "path", &path.display().to_string(), term_w, PeekStyle::PATH);
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
    peek_row_wrapped(&mut out, "path", &path.display().to_string(), term_w, PeekStyle::PATH);
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
    peek_label_grid(
        &mut out,
        "root",
        &members,
        term_w,
        PeekStyle::OBS_GRID,
    );
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
        let safe = label.replace('\t', " ").replace('\n', " ");
        let pct_s = format!("{:.1}", pct);
        let row0 = if peek_color_enabled() {
            format!(
                "  {}\t{}\t{}%\t",
                (i + 1).to_string().truecolor(GB_GRAY.0, GB_GRAY.1, GB_GRAY.2),
                fmt_usize_sep(*cnt)
                    .truecolor(GB_BRIGHT_YELLOW.0, GB_BRIGHT_YELLOW.1, GB_BRIGHT_YELLOW.2)
                    .bold(),
                pct_s.truecolor(GB_BRIGHT_AQUA.0, GB_BRIGHT_AQUA.1, GB_BRIGHT_AQUA.2),
            )
        } else {
            format!(
                "  {}\t{}\t{}%\t",
                i + 1,
                fmt_usize_sep(*cnt),
                pct_s
            )
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
    peek_row_wrapped(&mut out, "path", &path.display().to_string(), term_w, PeekStyle::PATH);
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

    if let Some(col) = obs_column {
        let col = col.trim();
        anyhow::ensure!(!col.is_empty(), "--obs column name is empty");
        let obs = h5
            .group("obs")
            .context("h5ad: missing obs group — cannot read --obs")?;
        let cells = read_h5ad_obs_column_str_h5(&obs, col)
            .with_context(|| format!("read obs[{col:?}]"))?;
        let _ = writeln!(out);
        out.push_str(&value_counts_block(col, &cells, n_obs, term_w)?);
    }

    Ok(out)
}

pub fn h5ad_peek_report(path: &Path, obs_column: Option<&str>) -> anyhow::Result<String> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let bytes = meta.len();

    let h5 = H5File::open(path).with_context(|| format!("open {}", path.display()))?;
    let tw = peek_terminal_width();

    if let Ok((n_obs, n_vars)) = infer_n_obs_n_vars(&h5) {
        return peek_annadata_h5_report(&h5, path, bytes, n_obs, n_vars, tw, obs_column);
    }

    if let Some(s) = try_peek_tenx_filtered_matrix(&h5, path, bytes, tw)? {
        if obs_column.is_some() {
            anyhow::bail!(
                "--obs applies only to AnnData HDF5 with an `obs` group (e.g. .h5ad); \
                 this file looks like a 10x-style `/matrix` HDF5 with no `obs` metadata"
            );
        }
        return Ok(s);
    }

    if obs_column.is_some() {
        anyhow::bail!(
            "--obs applies only to AnnData HDF5 with an `obs` group (e.g. .h5ad); \
             this HDF5 layout does not expose `obs` columns"
        );
    }

    peek_generic_hdf5_report(&h5, path, bytes, tw)
}

pub fn print_h5ad_peek(path: &Path, obs_column: Option<&str>) -> anyhow::Result<()> {
    print!("{}", h5ad_peek_report(path, obs_column)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::{AnnData, AnnDataOp};
    use anndata::data::ArrayData;
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
        a.set_obs_names(vec!["c0".into(), "c1".into()].into()).unwrap();
        a.set_var_names(vec!["G0".into(), "G1".into()].into()).unwrap();
        let obs = DataFrame::new(vec![Series::new(
            "cell_type".into(),
            vec!["a".to_string(), "b".to_string()],
        )
        .into()])
        .unwrap();
        a.set_obs(obs).unwrap();
        let var = DataFrame::new(vec![Series::new("mt".into(), vec![false, true]).into()]).unwrap();
        a.set_var(var).unwrap();
        let gem = Array2::<f64>::zeros((2, 2));
        a.set_x(ArrayData::from(gem)).unwrap();
        a.close().unwrap();

        let s = h5ad_peek_report(&path, None).unwrap();
        assert!(s.contains("2×2"), "report:\n{s}");
        assert!(s.contains("cell_type"));
        assert!(s.contains("mt"));
        assert!(s.contains("›"), "expected tag prefix, got:\n{s}");

        let s2 = h5ad_peek_report(&path, Some("cell_type")).unwrap();
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
        let s = h5ad_peek_report(p, None).expect("peek");
        assert!(s.contains("10x sparse matrix"), "report:\n{s}");
        assert!(s.contains("5,292") || s.contains("5292"), "report:\n{s}");
    }
}
