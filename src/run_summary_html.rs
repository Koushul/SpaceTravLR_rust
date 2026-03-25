use crate::config::{CnnTrainingMode, SpaceshipConfig};
use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use build_html::{escape_html, Html, HtmlChild, HtmlContainer, HtmlElement, HtmlPage, HtmlTag};
use chrono::Utc;
use glob::Pattern;
use ndarray::Array2;
use polars::prelude::DataType;
use serde_json::Value;
use std::path::{Path, PathBuf};

const RUN_SUMMARY_CSS: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/run_summary_report.css"));

const GOOGLE_FONTS_STYLESHEET: &str = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,600;0,700;1,300&family=IBM+Plex+Mono:wght@400;500;600&display=swap";

fn fmt_usize(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn count_betadata_files(output_dir: &Path, pattern: &str) -> anyhow::Result<usize> {
    let pat = Pattern::new(pattern).with_context(|| format!("invalid glob: {pattern}"))?;
    let rd = match std::fs::read_dir(output_dir) {
        Ok(r) => r,
        Err(_) => return Ok(0),
    };
    let mut n = 0;
    for e in rd {
        let e = e?;
        if !e.file_type()?.is_file() {
            continue;
        }
        if let Some(name) = e.file_name().to_str() {
            if pat.matches(name) {
                n += 1;
            }
        }
    }
    Ok(n)
}

fn obsm_has_spatial_coords<B: Backend>(adata: &AnnData<B>, key: &str) -> anyhow::Result<bool> {
    if let Some(a) = adata.obsm().get_item::<Array2<f64>>(key)? {
        return Ok(a.nrows() > 0 && a.ncols() >= 2);
    }
    if let Some(a) = adata.obsm().get_item::<Array2<f32>>(key)? {
        return Ok(a.nrows() > 0 && a.ncols() >= 2);
    }
    Ok(false)
}

fn detect_spatial_key<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Option<&'static str>> {
    for key in ["spatial", "X_spatial", "spatial_loc"] {
        if obsm_has_spatial_coords(adata, key)? {
            return Ok(Some(key));
        }
    }
    Ok(None)
}

fn obsm_has_umap<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<bool> {
    obsm_has_spatial_coords(adata, "X_umap")
}

fn cluster_n_unique(
    obs: &polars::frame::DataFrame,
    cluster_key: &str,
) -> anyhow::Result<Option<usize>> {
    let col = match obs.column(cluster_key) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };
    let cast = col.cast(&DataType::String)?;
    Ok(Some(cast.n_unique()?))
}

fn cnn_mode_label(mode: CnnTrainingMode) -> &'static str {
    match mode {
        CnnTrainingMode::Seed => "seed",
        CnnTrainingMode::Full => "full",
        CnnTrainingMode::Hybrid => "hybrid",
    }
}

fn manifest_training_mode(manifest: &Value) -> Option<String> {
    manifest.get("training_mode").map(|v| {
        if let Some(s) = v.as_str() {
            s.to_string()
        } else {
            v.to_string()
        }
    })
}

fn kv_tr(k: &str, v: &str) -> HtmlElement {
    HtmlElement::new(HtmlTag::TableRow)
        .with_child(
            HtmlElement::new(HtmlTag::TableCell)
                .with_attribute("class", "k")
                .with_child(escape_html(k).into())
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::TableCell)
                .with_attribute("class", "v")
                .with_child(escape_html(v).into())
                .into(),
        )
}

fn kv_table(rows: Vec<HtmlElement>) -> HtmlElement {
    let mut t = HtmlElement::new(HtmlTag::Table).with_attribute("class", "kv-table");
    for r in rows {
        t.add_child(r.into());
    }
    t
}

fn stat_cell(label: &str, value: &str, sub: &str, tone: Option<&str>) -> HtmlElement {
    let class = match tone {
        Some(t) => format!("stat-cell {t}"),
        None => "stat-cell".to_string(),
    };
    HtmlElement::new(HtmlTag::Div)
        .with_attribute("class", class)
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_attribute("class", "label")
                .with_child(label.into())
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_attribute("class", "value")
                .with_child(value.into())
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_attribute("class", "sub")
                .with_child(sub.into())
                .into(),
        )
}

fn meta_line(label: &str, value: &str) -> HtmlElement {
    HtmlElement::new(HtmlTag::Div)
        .with_child(HtmlElement::new(HtmlTag::Strong).with_child(label.into()).into())
        .with_child(format!("\u{00a0}\u{00a0}{}", escape_html(value)).into())
}

pub struct RunSummaryParams<'a> {
    pub adata_path: &'a Path,
    pub output_dir: &'a Path,
    pub cfg: &'a SpaceshipConfig,
    pub cluster_key: Option<&'a str>,
    pub layer_override: Option<&'a str>,
    pub run_id: Option<&'a str>,
    pub manifest: Option<&'a Value>,
    pub betadata_pattern: &'a str,
}

pub fn write_run_summary_html(p: RunSummaryParams<'_>) -> anyhow::Result<PathBuf> {
    let RunSummaryParams {
        adata_path,
        output_dir,
        cfg,
        cluster_key,
        layer_override,
        run_id,
        manifest,
        betadata_pattern,
    } = p;

    std::fs::create_dir_all(output_dir)?;

    let cluster_key = cluster_key.unwrap_or(cfg.data.cluster_annot.as_str());
    let layer = layer_override.unwrap_or(cfg.data.layer.as_str());
    let manifest = manifest.cloned().unwrap_or_else(|| Value::Object(Default::default()));

    let file = H5::open(adata_path).with_context(|| format!("open {}", adata_path.display()))?;
    let adata = AnnData::<H5>::open(file)?;

    let n_obs = adata.n_obs();
    let n_vars = adata.var_names().into_vec().len();

    let sp_key = detect_spatial_key(&adata)?;
    let sp_display = sp_key.map(|s| s.to_string()).unwrap_or_else(|| "—".to_string());

    let obs = adata.read_obs()?;
    let n_clusters = cluster_n_unique(&obs, cluster_key)?;
    let n_clusters_str = n_clusters
        .map(fmt_usize)
        .unwrap_or_else(|| "—".to_string());

    let has_umap = obsm_has_umap(&adata)?;
    let umap_display = if has_umap {
        "yes (obsm['X_umap'])"
    } else {
        "no"
    };

    let n_beta = manifest
        .get("betadata_count")
        .and_then(|v| v.as_u64())
        .map(|u| u as usize)
        .map(Ok)
        .unwrap_or_else(|| count_betadata_files(output_dir, betadata_pattern))?;

    let run_id = run_id
        .map(|s| s.to_string())
        .or_else(|| manifest.get("run_id").and_then(|v| v.as_str().map(String::from)))
        .unwrap_or_else(|| {
            adata_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("run")
                .to_string()
        });

    let started = manifest
        .get("started")
        .and_then(|v| v.as_str().map(String::from));

    let finished = manifest
        .get("finished")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string());

    let mode_str = manifest_training_mode(&manifest).unwrap_or_else(|| {
        cnn_mode_label(cfg.resolved_cnn_mode()).to_string()
    });

    let epochs = manifest
        .get("epochs")
        .and_then(|v| v.as_u64())
        .map(|u| u.to_string())
        .unwrap_or_else(|| cfg.training.epochs.to_string());

    let n_parallel = manifest
        .get("n_parallel")
        .and_then(|v| v.as_u64())
        .map(|u| u.to_string())
        .unwrap_or_else(|| cfg.execution.n_parallel.to_string());

    let spatial_training = if sp_key.is_some() {
        format!("adata.obsm['{sp_display}']")
    } else {
        "not detected".to_string()
    };

    let adata_path_disp = adata_path.display().to_string();
    let output_dir_disp = output_dir.display().to_string();

    let training_rows = vec![
        kv_tr("training.mode / manifest", &mode_str),
        kv_tr("training.epochs", &epochs),
        kv_tr("execution.n_parallel", &n_parallel),
        kv_tr("data.layer", layer),
        kv_tr("data.cluster_annot", cluster_key),
        kv_tr("spatial coords", &spatial_training),
        kv_tr("betadata files (*_betadata.feather)", &n_beta.to_string()),
    ];

    let adata_rows = vec![
        kv_tr(
            "Shape (obs × vars)",
            &format!("{} × {}", fmt_usize(n_obs), fmt_usize(n_vars)),
        ),
        kv_tr("Cluster key (obs)", cluster_key),
        kv_tr("Unique clusters", &n_clusters_str),
        kv_tr("Spatial obsm key", &sp_display),
        kv_tr("UMAP present", umap_display),
    ];

    let path_bar_item = |label: &str, inner: HtmlElement| {
        HtmlElement::new(HtmlTag::Div)
            .with_attribute("class", "path-bar-item")
            .with_child(
                HtmlElement::new(HtmlTag::Span)
                    .with_attribute("class", "pb-label")
                    .with_child(label.into())
                    .into(),
            )
            .with_child(
                HtmlElement::new(HtmlTag::Span)
                    .with_attribute("class", "pb-path")
                    .with_child(inner.into())
                    .into(),
            )
    };

    let report_pb = format!(
        r#"<span class="pb-dir">{}/</span>spacetravlr_run_summary.html"#,
        escape_html(&output_dir_disp)
    );

    let beta_pb = format!(
        r#"<span class="pb-dir">{}/</span>{}"#,
        escape_html(&output_dir_disp),
        escape_html(betadata_pattern)
    );

    let mut run_meta_children: Vec<HtmlChild> = vec![
        meta_line("Run ID", &run_id).into(),
        meta_line("AnnData", &adata_path_disp).into(),
        meta_line("Finished", &finished).into(),
    ];
    if let Some(s) = &started {
        run_meta_children.push(meta_line("Started", s).into());
    }

    let mut run_meta = HtmlElement::new(HtmlTag::Div).with_attribute("class", "run-meta");
    for c in run_meta_children {
        run_meta.add_child(c);
    }

    let header = HtmlElement::new(HtmlTag::Header)
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_attribute("class", "header-top")
                .with_child(
                    HtmlElement::new(HtmlTag::Div)
                        .with_attribute("class", "logo-block")
                        .with_child(
                            HtmlElement::new(HtmlTag::Heading1)
                                .with_child("Space".into())
                                .with_child(
                                    HtmlElement::new(HtmlTag::Span)
                                        .with_child("TravLR".into())
                                        .into(),
                                )
                                .into(),
                        )
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "subtitle")
                                .with_child(
                                    "Spatially Perturbing Transcription Factors, Ligands & Receptors · Run Report".into(),
                                )
                                .into(),
                        )
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "status-badge")
                                .with_child("Completed Successfully".into())
                                .into(),
                        )
                        .into(),
                )
                .with_child(run_meta.into())
                .into(),
        );

    let path_bar = HtmlElement::new(HtmlTag::Div)
        .with_attribute("class", "path-bar")
        .with_child(
            path_bar_item(
                "AnnData",
                HtmlElement::new(HtmlTag::Span).with_child(escape_html(&adata_path_disp).into()),
            )
            .into(),
        )
        .with_child(
            path_bar_item(
                "Beta outputs",
                HtmlElement::new(HtmlTag::Span).with_child(beta_pb.into()),
            )
            .into(),
        )
        .with_child(
            path_bar_item(
                "Report",
                HtmlElement::new(HtmlTag::Span).with_child(report_pb.into()),
            )
            .into(),
        );

    let note_html = format!(
        r#"Spatial / UMAP figures are not generated here; use scanpy or your notebook on the same <code>.h5ad</code> if needed. Clusters in the table use <strong>{}</strong>; spatial key detected: <strong>{}</strong>."#,
        escape_html(cluster_key),
        escape_html(&sp_display)
    );

    let container = HtmlElement::new(HtmlTag::Div)
        .with_attribute("class", "container")
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_child(
                    HtmlElement::new(HtmlTag::Div)
                        .with_attribute("class", "section-label")
                        .with_child("Overview".into())
                        .into(),
                )
                .with_child(
                    HtmlElement::new(HtmlTag::Div)
                        .with_attribute("class", "stat-grid")
                        .with_child(
                            stat_cell(
                                "Cells (obs)",
                                &fmt_usize(n_obs),
                                "in report AnnData",
                                None,
                            )
                            .into(),
                        )
                        .with_child(
                            stat_cell(
                                "Genes (vars)",
                                &fmt_usize(n_vars),
                                "expression matrix",
                                Some("good"),
                            )
                            .into(),
                        )
                        .with_child(
                            stat_cell(
                                "Betadata files",
                                &n_beta.to_string(),
                                &format!("matching {}", escape_html(betadata_pattern)),
                                Some("sky"),
                            )
                            .into(),
                        )
                        .into(),
                )
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::ParagraphText)
                .with_attribute("class", "note-panel")
                .with_child(note_html.into())
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::Div)
                .with_attribute("class", "two-col")
                .with_child(
                    HtmlElement::new(HtmlTag::Div)
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "section-label")
                                .with_child("AnnData".into())
                                .into(),
                        )
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "card")
                                .with_child(
                                    HtmlElement::new(HtmlTag::Div)
                                        .with_attribute("class", "card-title")
                                        .with_child("⬡ summary".into())
                                        .into(),
                                )
                                .with_child(kv_table(adata_rows).into())
                                .into(),
                        )
                        .into(),
                )
                .with_child(
                    HtmlElement::new(HtmlTag::Div)
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "section-label")
                                .with_child("Training (config / manifest)".into())
                                .into(),
                        )
                        .with_child(
                            HtmlElement::new(HtmlTag::Div)
                                .with_attribute("class", "card")
                                .with_child(
                                    HtmlElement::new(HtmlTag::Div)
                                        .with_attribute("class", "card-title")
                                        .with_child("⚙ run".into())
                                        .into(),
                                )
                                .with_child(kv_table(training_rows).into())
                                .into(),
                        )
                        .into(),
                )
                .into(),
        );

    let gen_stamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let footer = HtmlElement::new(HtmlTag::Footer)
        .with_child(
            HtmlElement::new(HtmlTag::Span)
                .with_child("SpaceTravLR · Rust training report".into())
                .into(),
        )
        .with_child(
            HtmlElement::new(HtmlTag::Span)
                .with_child(
                    format!(
                        "Generated {} · {}",
                        gen_stamp,
                        escape_html(&run_id)
                    )
                    .into(),
                )
                .into(),
        );

    let body = HtmlElement::new(HtmlTag::Div)
        .with_child(header.into())
        .with_child(path_bar.into())
        .with_child(container.into())
        .with_child(footer.into());

    let mut page = HtmlPage::new()
        .with_title("SpaceTravLR · Run Summary")
        .with_meta([("charset", "UTF-8")])
        .with_meta([
            ("name", "viewport"),
            (
                "content",
                "width=device-width, initial-scale=1.0",
            ),
        ])
        .with_head_link("https://fonts.googleapis.com", "preconnect")
        .with_head_link_attr(
            "https://fonts.gstatic.com",
            "preconnect",
            [("crossorigin", "anonymous")],
        )
        .with_head_link(GOOGLE_FONTS_STYLESHEET, "stylesheet");

    page.add_style(RUN_SUMMARY_CSS);
    page.add_html(body);

    let html = page
        .to_html_string()
        .replacen("<html>", r#"<html lang="en">"#, 1);

    let html_path = output_dir.join("spacetravlr_run_summary.html");
    std::fs::write(&html_path, html)?;
    Ok(html_path)
}
