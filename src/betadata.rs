use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView1};
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use polars::prelude::*;
use rayon::prelude::*;
use serde::Serialize;

/// One human-readable label from an `obs` column row (`cell_type`, `--condition`, etc.).
/// Uses [`Series::str_value`] so categoricals resolve to category names, not `AnyValue::to_string()` / `Debug` quoting.
pub fn obs_series_row_str(series: &Series, i: usize) -> anyhow::Result<String> {
    let v = series
        .str_value(i)
        .map_err(|e| anyhow::anyhow!("obs row {i}: {e}"))?;
    let t = v.as_ref();
    Ok(if t == "null" {
        String::new()
    } else {
        t.to_string()
    })
}

/// Named matrix for gene expression or weighted ligand data.
/// Wraps a dense 2D array with gene-name → column-index lookup.
pub struct GeneMatrix {
    pub data: Array2<f32>,
    pub col_names: Vec<String>,
    col_map: HashMap<String, usize>,
}

impl GeneMatrix {
    pub fn new(data: Array2<f32>, col_names: Vec<String>) -> Self {
        let col_map = col_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Self {
            data,
            col_names,
            col_map,
        }
    }

    pub fn col(&self, name: &str) -> Option<ArrayView1<'_, f32>> {
        self.col_map.get(name).map(|&i| self.data.column(i))
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.col_map.get(name).copied()
    }

    pub fn n_rows(&self) -> usize {
        self.data.nrows()
    }

    pub fn n_cols(&self) -> usize {
        self.data.ncols()
    }
}

/// Column used for feather join keys (same as training `cluster_annot`).
pub fn resolve_betadata_cluster_key_column(_obs: &DataFrame, cluster_annot: &str) -> String {
    cluster_annot.to_string()
}

fn obs_cluster_column_is_numeric_id(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// One string per cell for matching feather `Cluster` / `CellID` row labels.
///
/// Categorical / enum obs columns use **category names** (e.g. `"10"`), not internal codes (`2`),
/// so they align with seed-only betadata exported from training. Purely numeric columns use the
/// same `Int64`→`String` normalization as feather files (`3.0` → `"3"`).
pub fn betadata_cluster_keys_from_obs_dataframe(
    obs: &DataFrame,
    cluster_annot: &str,
) -> Result<Vec<String>> {
    let col = obs
        .column(cluster_annot)
        .with_context(|| format!("obs column {:?} not found", cluster_annot))?;
    let series = col.as_materialized_series();
    let keys: Vec<String> = match series.dtype() {
        DataType::Categorical(_, _) | DataType::Enum(_, _) => series
            .cast(&DataType::String)?
            .str()?
            .into_iter()
            .map(|opt| opt.map(|s| s.trim().to_string()).unwrap_or_default())
            .collect(),
        dt if obs_cluster_column_is_numeric_id(dt) => series
            .cast(&DataType::Int64)?
            .cast(&DataType::String)?
            .str()?
            .into_iter()
            .map(|opt| opt.map(|s| s.to_string()).unwrap_or_else(|| "0".into()))
            .collect(),
        _ => series
            .cast(&DataType::String)?
            .str()?
            .into_iter()
            .map(|opt| opt.map(|s| s.trim().to_string()).unwrap_or_default())
            .collect(),
    };
    Ok(keys)
}

/// Maps each training cluster id (same indexing as Lasso / seed betadata rows) to the
/// per-cell string used in [`betadata_cluster_keys_from_obs_dataframe`] for that cluster,
/// so seed `*_betadata.feather` **`Cluster`** values match perturbation / viewer join keys
/// (e.g. `cell_type` names instead of internal `0..K-1` codes for string columns).
pub fn build_cluster_id_to_betadata_cluster_key_map(
    obs: &DataFrame,
    cluster_annot: &str,
    clusters: &Array1<usize>,
) -> Result<HashMap<usize, String>> {
    let keys = betadata_cluster_keys_from_obs_dataframe(obs, cluster_annot)?;
    anyhow::ensure!(
        keys.len() == clusters.len(),
        "cluster key len {} != clusters len {}",
        keys.len(),
        clusters.len()
    );
    let mut out = HashMap::new();
    for (i, &cid) in clusters.iter().enumerate() {
        out.entry(cid).or_insert_with(|| keys[i].clone());
    }
    Ok(out)
}

/// `usize` cluster codes for UI / colormap grouping (Float64 cast + round). For betadata row
/// matching use [`betadata_cluster_keys_from_obs_dataframe`] instead when the column is categorical.
pub fn clusters_usize_from_obs_dataframe(
    obs: &DataFrame,
    cluster_annot: &str,
) -> Result<Vec<usize>> {
    let col = obs.column(cluster_annot).map_err(|_| {
        let preview: Vec<String> = obs
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .take(25)
            .collect();
        anyhow::anyhow!(
            "obs column {:?} not found. First columns: {:?}",
            cluster_annot,
            preview
        )
    })?;
    let f = col.cast(&DataType::Float64).map_err(|e| {
        anyhow::anyhow!(
            "obs column {:?} must be numeric (cluster ids): {}",
            cluster_annot,
            e
        )
    })?;
    let ca = f.f64()?;
    Ok(ca
        .into_iter()
        .map(|v| v.unwrap_or(0.0).round() as i64 as usize)
        .collect())
}

/// Beta coefficients for a single target gene.
///
/// Betas are stored compactly (`n_beta_rows` rows — clusters for seed-only, cells
/// for CNN). A shared `cell_to_beta_row` mapping lets `splash()` always produce
/// one output row per cell without duplicating storage.
pub struct BetaFrame {
    pub gene_name: String,

    /// Number of rows in the beta arrays (clusters for seed, cells for CNN).
    pub n_beta_rows: usize,
    /// Labels for the beta rows (cluster IDs or cell IDs from the file).
    pub row_labels: Vec<String>,

    pub intercepts: Array1<f32>,
    pub tf_betas: Array2<f32>,
    pub lr_betas: Array2<f32>,
    pub tfl_betas: Array2<f32>,

    /// Number of output cells (== obs_names.len() after expand_to_cells).
    pub n_cells: usize,
    /// Per-cell obs names (shared across all frames in a Betabase).
    pub cell_labels: Arc<Vec<String>>,
    /// Maps cell index → beta row index (shared across frames with identical row_labels).
    pub cell_to_beta_row: Arc<Vec<usize>>,

    pub tfs: Vec<String>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,

    /// Sorted unique modulator gene names with "beta_" prefix.
    pub modulator_genes: Vec<String>,
    /// Maps each modulator gene to its index in a global gene list (set externally).
    pub modulator_gene_indices: Option<Vec<usize>>,

    /// Feather rows are keyed by **`CellID`** (per-cell CNN): map cells **only** via `obs_names`.
    /// When `false` (e.g. **`Cluster`** column), map via `cluster_keys` first, then obs name.
    pub join_by_obs_name: bool,
}

pub struct BetaFrameFromParts {
    pub gene_name: String,
    pub row_labels: Vec<String>,
    pub intercepts: Array1<f32>,
    pub tf_betas: Array2<f32>,
    pub tfs: Vec<String>,
    pub lr_betas: Array2<f32>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_betas: Array2<f32>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,
}

impl BetaFrameFromParts {
    #[must_use]
    pub fn into_beta_frame(self) -> BetaFrame {
        self.into()
    }
}

impl From<BetaFrameFromParts> for BetaFrame {
    fn from(parts: BetaFrameFromParts) -> Self {
        let BetaFrameFromParts {
            gene_name,
            row_labels,
            intercepts,
            tf_betas,
            tfs,
            lr_betas,
            ligands,
            receptors,
            tfl_betas,
            tfl_ligands,
            tfl_regulators,
        } = parts;
        let n = row_labels.len();
        let modulator_genes = Self::compute_modulator_genes(
            &tfs,
            &ligands,
            &receptors,
            &tfl_ligands,
            &tfl_regulators,
        );

        Self {
            gene_name,
            n_beta_rows: n,
            cell_labels: Arc::new(row_labels.clone()),
            cell_to_beta_row: Arc::new((0..n).collect()),
            n_cells: n,
            row_labels,
            intercepts,
            tf_betas,
            lr_betas,
            tfl_betas,
            tfs,
            ligands,
            receptors,
            tfl_ligands,
            tfl_regulators,
            modulator_genes,
            modulator_gene_indices: None,
            join_by_obs_name: false,
        }
    }
}

/// Write betadata as Feather-compatible Arrow IPC (LZ4). `id_col` is `Cluster` (seed-only) or `CellID` (per-cell CNN).
pub fn write_betadata_feather_to_writer<W: std::io::Write>(
    writer: W,
    id_col: &str,
    ids: &[String],
    data_columns: &[String],
    data: &Array2<f64>,
) -> Result<()> {
    anyhow::ensure!(
        ids.len() == data.nrows(),
        "id count {} != data rows {}",
        ids.len(),
        data.nrows()
    );
    anyhow::ensure!(
        data_columns.len() == data.ncols(),
        "data_columns len {} != data ncols {}",
        data_columns.len(),
        data.ncols()
    );

    let mut columns: Vec<Column> = Vec::with_capacity(1 + data_columns.len());
    columns.push(Series::new(id_col.into(), ids.to_vec()).into());
    for (j, name) in data_columns.iter().enumerate() {
        let col: Vec<f64> = data.column(j).iter().copied().collect();
        columns.push(Series::new(name.as_str().into(), col).into());
    }
    let mut df = DataFrame::new(columns)?;
    let mut w = IpcWriter::new(writer).with_compression(Some(IpcCompression::LZ4));
    w.finish(&mut df).context("write IPC / feather bytes")?;
    Ok(())
}

/// Write betadata as Feather-compatible Arrow IPC (LZ4). `id_col` is `Cluster` (seed-only) or `CellID` (per-cell CNN).
pub fn write_betadata_feather(
    path: &str,
    id_col: &str,
    ids: &[String],
    data_columns: &[String],
    data: &Array2<f64>,
) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    write_betadata_feather_to_writer(f, id_col, ids, data_columns, data)
        .with_context(|| format!("write IPC {}", path))
}

impl BetaFrame {
    pub fn from_path(path: &str) -> Result<Self> {
        Self::from_feather(path)
    }

    pub fn from_feather(path: &str) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path))?;
        let df = IpcReader::new(f)
            .finish()
            .with_context(|| format!("read IPC {}", path))?;

        let all_col_names: Vec<String> = df
            .get_columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();

        let label_col_idx = betadata_feather_label_column_index(&all_col_names);

        let join_by_obs_name = label_col_idx
            .and_then(|i| all_col_names.get(i).map(|s| s.as_str()))
            .is_some_and(label_name_is_per_cell_identity);

        let (row_labels, data_col_names) = if let Some(idx) = label_col_idx {
            let label_name = &all_col_names[idx];
            let labels = feather_id_column_to_strings(df.column(label_name)?)?;
            let data_names: Vec<String> = all_col_names
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, n)| n.clone())
                .collect();
            (labels, data_names)
        } else {
            let labels: Vec<String> = (0..df.height()).map(|i| i.to_string()).collect();
            (labels, all_col_names)
        };

        let n_rows = row_labels.len();
        let n_cols = data_col_names.len();
        let mut raw = Array2::<f32>::zeros((n_rows, n_cols));

        for (j, col_name) in data_col_names.iter().enumerate() {
            let casted = df.column(col_name)?.cast(&DataType::Float32)?;
            let ca = casted.f32()?;
            for (i, val) in ca.into_iter().enumerate() {
                raw[[i, j]] = val.unwrap_or(0.0);
            }
        }

        let gene_name = Self::extract_gene_name(path);
        Self::from_raw(gene_name, row_labels, data_col_names, raw, join_by_obs_name)
    }

    /// Construct directly from typed arrays (useful for tests and programmatic construction).
    /// Same as `BetaFrame::from(parts)` for a [`BetaFrameFromParts`].
    /// Starts with an identity cell→beta mapping (n_cells == n_beta_rows).
    pub fn from_parts(parts: BetaFrameFromParts) -> Self {
        Self::from(parts)
    }

    /// Given obs_names and per-cell cluster assignments, build the mapping from
    /// cell index → beta row index. For seed-only betadata (rows = clusters) this
    /// maps each cell to its cluster's row; for CNN betadata (rows = cells) it
    /// matches by obs_name.
    ///
    /// Both `Arc`s are typically shared across every `BetaFrame` in a `Betabase`
    /// so the per-gene overhead is just two pointer-sized fields.
    pub fn expand_to_cells(
        &mut self,
        cell_labels: Arc<Vec<String>>,
        cell_to_beta_row: Arc<Vec<usize>>,
    ) {
        self.n_cells = cell_labels.len();
        self.cell_labels = cell_labels;
        self.cell_to_beta_row = cell_to_beta_row;
    }

    /// Determine how to map cell indices to beta rows for a given set of row_labels.
    /// Returns a Vec<usize> of length obs_names.len().
    ///
    /// For each cell: match `cluster_keys[i]` to a row label (seed-only `Cluster` column), else
    /// **obs name** (per-cell `CellID` export), else row `0` with a warning. `cluster_keys` must
    /// use the same strings as in the feather (e.g. categorical **names** `"10"`, not code `2`).
    pub fn compute_cell_mapping(
        row_labels: &[String],
        obs_names: &[String],
        cluster_keys: &[String],
    ) -> Vec<usize> {
        debug_assert_eq!(
            obs_names.len(),
            cluster_keys.len(),
            "compute_cell_mapping length mismatch"
        );
        let row_map: HashMap<&str, usize> = row_labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        let mut n_via_key = 0usize;
        let mut n_via_obs = 0usize;
        let mut n_default = 0usize;
        let mut mapping = Vec::with_capacity(obs_names.len());

        for (name, ck) in obs_names.iter().zip(cluster_keys.iter()) {
            let idx = if let Some(&i) = row_map.get(ck.as_str()) {
                n_via_key += 1;
                i
            } else if let Some(&i) = row_map.get(name.as_str()) {
                n_via_obs += 1;
                i
            } else {
                n_default += 1;
                0
            };
            mapping.push(idx);
        }

        if n_default > 0 {
            eprintln!(
                "Warning: {} of {} cells could not map to a betadata row; using row 0 for those. ({} cells mapped via cluster key, {} via obs id.)",
                n_default,
                obs_names.len(),
                n_via_key,
                n_via_obs
            );
        }

        mapping
    }

    /// Map each AnnData cell to a feather row by **`obs_names[i]`** matching `row_labels`.
    /// Used when the feather id column is **`CellID`**. Do not use `cluster_keys` here: those
    /// are cell-type (or other) labels and can spuriously match row ids or force row `0`,
    /// destroying per-cell spatial β variation.
    pub fn compute_cell_mapping_cellid_rows(
        row_labels: &[String],
        obs_names: &[String],
    ) -> Vec<usize> {
        let mut row_map: HashMap<&str, usize> = HashMap::new();
        for (i, l) in row_labels.iter().enumerate() {
            row_map.entry(l.as_str()).or_insert(i);
        }
        let mut n_default = 0usize;
        let mapping: Vec<usize> = obs_names
            .iter()
            .map(|name| {
                row_map.get(name.as_str()).copied().unwrap_or_else(|| {
                    n_default += 1;
                    0
                })
            })
            .collect();
        if n_default > 0 {
            eprintln!(
                "Warning: {} of {} cells could not map to a betadata CellID row; using row 0 for those.",
                n_default,
                obs_names.len()
            );
        }
        mapping
    }

    fn extract_gene_name(path: &str) -> String {
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .strip_suffix("_betadata")
            .unwrap_or("")
            .to_string()
    }

    fn compute_modulator_genes(
        tfs: &[String],
        ligands: &[String],
        receptors: &[String],
        tfl_ligands: &[String],
        tfl_regulators: &[String],
    ) -> Vec<String> {
        let mut unique = HashSet::new();
        for g in tfs
            .iter()
            .chain(ligands.iter())
            .chain(receptors.iter())
            .chain(tfl_ligands.iter())
            .chain(tfl_regulators.iter())
        {
            unique.insert(g.clone());
        }
        let mut genes: Vec<String> = unique.into_iter().collect();
        genes.sort();
        genes.iter().map(|g| format!("beta_{}", g)).collect()
    }

    fn from_raw(
        gene_name: String,
        row_labels: Vec<String>,
        data_col_names: Vec<String>,
        data: Array2<f32>,
        join_by_obs_name: bool,
    ) -> Result<Self> {
        let n_rows = row_labels.len();

        let has_prefix = data_col_names
            .iter()
            .any(|c| c.starts_with("beta_") && c != "beta0");

        let mut tfs = Vec::new();
        let mut ligands = Vec::new();
        let mut receptors = Vec::new();
        let mut tfl_ligands = Vec::new();
        let mut tfl_regulators = Vec::new();

        let mut intercept_idx = None;
        let mut tf_indices = Vec::new();
        let mut lr_indices = Vec::new();
        let mut tfl_indices = Vec::new();

        for (i, col) in data_col_names.iter().enumerate() {
            if col == "beta0" || col == "beta_0" {
                intercept_idx = Some(i);
                continue;
            }

            let modulator = if has_prefix {
                match col.strip_prefix("beta_") {
                    Some(m) => m,
                    None => continue,
                }
            } else {
                col.as_str()
            };

            if modulator.contains('$') {
                let parts: Vec<&str> = modulator.splitn(2, '$').collect();
                ligands.push(parts[0].to_string());
                receptors.push(parts[1].to_string());
                lr_indices.push(i);
            } else if modulator.contains('#') {
                let parts: Vec<&str> = modulator.splitn(2, '#').collect();
                tfl_ligands.push(parts[0].to_string());
                tfl_regulators.push(parts[1].to_string());
                tfl_indices.push(i);
            } else {
                tfs.push(modulator.to_string());
                tf_indices.push(i);
            }
        }

        let intercepts = intercept_idx
            .map(|i| data.column(i).to_owned())
            .unwrap_or_else(|| Array1::zeros(n_rows));

        let tf_betas = Self::extract_cols(&data, &tf_indices, n_rows);
        let lr_betas = Self::extract_cols(&data, &lr_indices, n_rows);
        let tfl_betas = Self::extract_cols(&data, &tfl_indices, n_rows);

        let modulator_genes = Self::compute_modulator_genes(
            &tfs,
            &ligands,
            &receptors,
            &tfl_ligands,
            &tfl_regulators,
        );

        Ok(Self {
            gene_name,
            n_beta_rows: n_rows,
            cell_labels: Arc::new(row_labels.clone()),
            cell_to_beta_row: Arc::new((0..n_rows).collect()),
            n_cells: n_rows,
            row_labels,
            intercepts,
            tf_betas,
            lr_betas,
            tfl_betas,
            tfs,
            ligands,
            receptors,
            tfl_ligands,
            tfl_regulators,
            modulator_genes,
            modulator_gene_indices: None,
            join_by_obs_name,
        })
    }

    fn extract_cols(data: &Array2<f32>, indices: &[usize], n_rows: usize) -> Array2<f32> {
        if indices.is_empty() {
            return Array2::zeros((n_rows, 0));
        }
        let mut out = Array2::zeros((n_rows, indices.len()));
        for (j, &col_idx) in indices.iter().enumerate() {
            out.column_mut(j).assign(&data.column(col_idx));
        }
        out
    }

    /// Compute partial derivatives of target gene expression w.r.t. each modulator gene.
    ///
    /// Mirrors the Python `BetaFrame.splash()`:
    ///   dy/dTF    = beta_TF                                (no scale_factor)
    ///   dy/dR     = beta_LR * wL        (where gex[R] > 0, × scale_factor)
    ///   dy/dL(lr) = beta_LR * gex[R]                       (× scale_factor)
    ///   dy/dL(tfl)= beta_TFL * gex[reg]                    (× scale_factor)
    ///   dy/dTF(tfl)= beta_TFL * wL_tfl                     (× scale_factor)
    pub fn splash(
        &self,
        rw_ligands: &GeneMatrix,
        rw_ligands_tfl: &GeneMatrix,
        gex_df: &GeneMatrix,
        scale_factor: f32,
        beta_cap: Option<f32>,
    ) -> GeneMatrix {
        let n = self.n_cells;
        let map = self.cell_to_beta_row.as_slice();
        let n_out = self.modulator_genes.len();
        let n_tfs = self.tfs.len();
        let n_lr = self.ligands.len();
        let n_tfl = self.tfl_ligands.len();

        let gene_to_out: HashMap<&str, usize> = self
            .modulator_genes
            .iter()
            .enumerate()
            .map(|(i, g)| (g.strip_prefix("beta_").unwrap_or(g.as_str()), i))
            .collect();

        let tf_oi: Vec<usize> = self
            .tfs
            .iter()
            .map(|t| gene_to_out.get(t.as_str()).copied().unwrap_or(0))
            .collect();

        // LR work items with pre-resolved flat indices into input matrices
        #[derive(Clone)]
        struct LrWork {
            beta_col: usize,
            rec_oi: usize,
            lig_oi: usize,
            wl_col: usize,
            gex_col: usize,
        }
        let lr_work: Vec<LrWork> = (0..n_lr)
            .filter_map(|j| {
                Some(LrWork {
                    beta_col: j,
                    rec_oi: gene_to_out.get(self.receptors[j].as_str()).copied()?,
                    lig_oi: gene_to_out.get(self.ligands[j].as_str()).copied()?,
                    wl_col: rw_ligands.col_index(&self.ligands[j])?,
                    gex_col: gex_df.col_index(&self.receptors[j])?,
                })
            })
            .collect();

        #[derive(Clone)]
        struct TflWork {
            beta_col: usize,
            lig_oi: usize,
            reg_oi: usize,
            gex_col: usize,
            wl_col: usize,
        }
        let tfl_work: Vec<TflWork> = (0..n_tfl)
            .filter_map(|j| {
                Some(TflWork {
                    beta_col: j,
                    lig_oi: gene_to_out.get(self.tfl_ligands[j].as_str()).copied()?,
                    reg_oi: gene_to_out.get(self.tfl_regulators[j].as_str()).copied()?,
                    gex_col: gex_df.col_index(&self.tfl_regulators[j])?,
                    wl_col: rw_ligands_tfl.col_index(&self.tfl_ligands[j])?,
                })
            })
            .collect();

        // Flat views: beta arrays are tiny (n_clusters × n_cols), always in cache
        let tf_flat = self.tf_betas.as_slice_memory_order().unwrap_or(&[]);
        let lr_flat = self.lr_betas.as_slice_memory_order().unwrap_or(&[]);
        let tfl_flat = self.tfl_betas.as_slice_memory_order().unwrap_or(&[]);

        // Flat views of input matrices (zero-allocation direct access)
        let rw_flat = rw_ligands.data.as_slice().unwrap();
        let rw_nc = rw_ligands.data.ncols();
        let rw_tfl_flat = rw_ligands_tfl.data.as_slice().unwrap();
        let rw_tfl_nc = rw_ligands_tfl.data.ncols();
        let gex_flat = gex_df.data.as_slice().unwrap();
        let gex_nc = gex_df.data.ncols();

        // Row-by-row parallel processing: each cell's result row (~2KB) fits in L1
        let mut result = vec![0.0f32; n * n_out];

        result.par_chunks_mut(n_out).enumerate().for_each(|(i, r)| {
            let br = unsafe { *map.get_unchecked(i) };
            let rw_base = i * rw_nc;
            let rw_tfl_base = i * rw_tfl_nc;
            let gex_base = i * gex_nc;

            // 1. TF derivatives (no scale_factor)
            let tf_base = br * n_tfs;
            for j in 0..n_tfs {
                unsafe {
                    *r.get_unchecked_mut(*tf_oi.get_unchecked(j)) +=
                        *tf_flat.get_unchecked(tf_base + j);
                }
            }

            // 2+3. LR derivatives
            let lr_beta_base = br * n_lr;
            for lw in &lr_work {
                let beta = unsafe { *lr_flat.get_unchecked(lr_beta_base + lw.beta_col) };
                let wl = unsafe { *rw_flat.get_unchecked(rw_base + lw.wl_col) };
                let gex = unsafe { *gex_flat.get_unchecked(gex_base + lw.gex_col) };

                if gex > 0.0f32 {
                    unsafe { *r.get_unchecked_mut(lw.rec_oi) += beta * wl * scale_factor };
                }
                unsafe { *r.get_unchecked_mut(lw.lig_oi) += beta * gex * scale_factor };
            }

            // 4+5. TFL derivatives
            let tfl_beta_base = br * n_tfl;
            for tw in &tfl_work {
                let beta = unsafe { *tfl_flat.get_unchecked(tfl_beta_base + tw.beta_col) };
                let gex_reg = unsafe { *gex_flat.get_unchecked(gex_base + tw.gex_col) };
                let wl = unsafe { *rw_tfl_flat.get_unchecked(rw_tfl_base + tw.wl_col) };

                unsafe { *r.get_unchecked_mut(tw.lig_oi) += beta * gex_reg * scale_factor };
                unsafe { *r.get_unchecked_mut(tw.reg_oi) += beta * wl * scale_factor };
            }
        });

        let mut result_arr = Array2::from_shape_vec((n, n_out), result).unwrap();

        if let Some(cap) = beta_cap {
            result_arr.mapv_inplace(|v| v.clamp(-cap, cap));
        }

        GeneMatrix::new(result_arr, self.modulator_genes.clone())
    }
}

/// Collection of BetaFrames for all trained genes, expanded to cell level.
pub struct Betabase {
    pub data: HashMap<String, BetaFrame>,
    pub ligands_set: HashSet<String>,
    pub receptors_set: HashSet<String>,
    pub tfl_ligands_set: HashSet<String>,
    pub tfs_set: HashSet<String>,
}

/// Feather read vs cell expansion while [`Betabase::from_directory`] runs.
#[derive(Clone, Copy, Debug)]
pub enum BetadataProgressPhase {
    ReadingFeathers { done: usize, total: usize },
    ExpandingToCells { done: usize, total: usize },
}

/// Atomics updated during betadata load for HTTP/UI progress (spatial viewer).
pub struct BetadataUiProgress {
    pub done: AtomicU32,
    pub total: AtomicU32,
    pub phase: AtomicU8,
}

const BETADATA_UI_PHASE_IDLE: u8 = 0;

impl Default for BetadataUiProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl BetadataUiProgress {
    pub fn new() -> Self {
        Self {
            done: AtomicU32::new(0),
            total: AtomicU32::new(0),
            phase: AtomicU8::new(BETADATA_UI_PHASE_IDLE),
        }
    }

    pub fn reset(&self) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
        self.phase.store(BETADATA_UI_PHASE_IDLE, Ordering::Relaxed);
    }
}

impl Betabase {
    pub fn apply_modulator_gene_indices(&mut self, gene2index: &HashMap<String, usize>) {
        for frame in self.data.values_mut() {
            frame.modulator_gene_indices = Some(
                frame
                    .modulator_genes
                    .iter()
                    .map(|g| {
                        let plain = g.strip_prefix("beta_").unwrap_or(g);
                        *gene2index.get(plain).unwrap_or(&0)
                    })
                    .collect(),
            );
        }
    }

    /// Load all `*_betadata.feather` files from `dir` in parallel (rayon),
    /// then expand every frame to cell level using the given obs_names + `cluster_keys`.
    ///
    /// `on_subprogress`: optional callback with sub-progress in **permille** (0–1000) for this
    /// stage only (roughly 0–700 while reading feathers, 700–1000 while expanding to cells), plus
    /// the current [`BetadataProgressPhase`].
    pub fn from_directory(
        dir: &str,
        obs_names: &[String],
        cluster_keys: &[String],
        gene2index: Option<&HashMap<String, usize>>,
        on_subprogress: Option<Arc<dyn Fn(u32, BetadataProgressPhase) + Send + Sync>>,
    ) -> Result<Self> {
        let dir_path = Path::new(dir);
        anyhow::ensure!(dir_path.exists(), "Directory {} does not exist", dir);
        anyhow::ensure!(
            obs_names.len() == cluster_keys.len(),
            "obs_names len {} != cluster_keys len {}",
            obs_names.len(),
            cluster_keys.len()
        );

        let paths: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let p = entry.path();
                let name = p.file_name()?.to_str()?;
                if name.ends_with("_betadata.feather") {
                    Some(p.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect();

        // When a higher-level UI (e.g. spatial_viewer) is tracking betadata progress via
        // `on_subprogress`, avoid drawing a second terminal progress bar here; it causes
        // duplicated / glitchy output. Also skip the bar entirely when stderr is not a TTY
        // (e.g. in notebook / web hosts where it would get stuck at the bottom of the page).
        let pb = if on_subprogress.is_none() && std::io::stderr().is_terminal() {
            let pb = indicatif::ProgressBar::new(paths.len() as u64);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Reading betadata")?
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        let total_n = paths.len().max(1) as u64;
        let processed = Arc::new(AtomicU32::new(0));

        let mut frames: Vec<BetaFrame> = paths
            .par_iter()
            .filter_map(|path| {
                let result = BetaFrame::from_path(path);
                if let Some(pb) = &pb {
                    pb.inc(1);
                }
                let pn = processed.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(f) = &on_subprogress {
                    let sub = ((pn as u64 * 700u64) / total_n).min(700) as u32;
                    f(
                        sub,
                        BetadataProgressPhase::ReadingFeathers {
                            done: pn as usize,
                            total: paths.len(),
                        },
                    );
                }
                match result {
                    Ok(frame) => Some(frame),
                    Err(e) => {
                        eprintln!("Warning: failed to load {}: {}", path, e);
                        None
                    }
                }
            })
            .collect();

        if let Some(pb) = pb {
            pb.finish_with_message("Done loading betadata");
        }

        // Expand all frames to cell level. Compute the mapping once per unique
        // set of row_labels and share via Arc to avoid duplicating per gene.
        let cell_labels = Arc::new(obs_names.to_vec());
        let mut last_row_labels: Option<Vec<String>> = None;
        let mut last_join_by_obs: Option<bool> = None;
        let mut last_mapping: Option<Arc<Vec<usize>>> = None;

        let mut data = HashMap::new();
        let mut ligands_set = HashSet::new();
        let mut receptors_set = HashSet::new();
        let mut tfl_ligands_set = HashSet::new();
        let mut tfs_set = HashSet::new();

        let n_expand = frames.len().max(1);
        for (fi, mut frame) in frames.drain(..).enumerate() {
            if let Some(f) = &on_subprogress {
                let sub = (700u64 + ((fi as u64 + 1) * 300) / n_expand as u64).min(1000) as u32;
                f(
                    sub,
                    BetadataProgressPhase::ExpandingToCells {
                        done: fi + 1,
                        total: n_expand,
                    },
                );
            }
            ligands_set.extend(frame.ligands.iter().cloned());
            receptors_set.extend(frame.receptors.iter().cloned());
            tfl_ligands_set.extend(frame.tfl_ligands.iter().cloned());
            tfs_set.extend(frame.tfs.iter().cloned());

            // Reuse the mapping Arc when row_labels and join mode haven't changed
            let mapping = if last_row_labels.as_ref() == Some(&frame.row_labels)
                && last_join_by_obs == Some(frame.join_by_obs_name)
            {
                last_mapping.as_ref().unwrap().clone()
            } else {
                let m = Arc::new(if frame.join_by_obs_name {
                    BetaFrame::compute_cell_mapping_cellid_rows(&frame.row_labels, obs_names)
                } else {
                    BetaFrame::compute_cell_mapping(&frame.row_labels, obs_names, cluster_keys)
                });
                last_row_labels = Some(frame.row_labels.clone());
                last_join_by_obs = Some(frame.join_by_obs_name);
                last_mapping = Some(m.clone());
                m
            };

            frame.expand_to_cells(cell_labels.clone(), mapping);

            if let Some(g2i) = gene2index {
                frame.modulator_gene_indices = Some(
                    frame
                        .modulator_genes
                        .iter()
                        .map(|g| {
                            let plain = g.strip_prefix("beta_").unwrap_or(g);
                            *g2i.get(plain).unwrap_or(&0)
                        })
                        .collect(),
                );
            }

            data.insert(frame.gene_name.clone(), frame);
        }

        Ok(Self {
            data,
            ligands_set,
            receptors_set,
            tfl_ligands_set,
            tfs_set,
        })
    }
}

/// Feather column used as β row key. **Order matters:** many CNN exports include both `Cluster`
/// and `CellID`; Arrow column order often has `Cluster` first — taking the first match used to
/// collapse all cells in the same cluster onto one feather row (no spatial variance in the UI).
pub(crate) fn betadata_feather_label_column_index(all_names: &[String]) -> Option<usize> {
    const EXACT: &[&str] = &["CellID", "obs_names", "cell_id", "Cluster"];
    for &name in EXACT {
        if let Some(i) = all_names.iter().position(|c| c == name) {
            return Some(i);
        }
    }
    all_names
        .iter()
        .position(|c| c.starts_with("__index") || c == "index")
}

#[inline]
fn label_name_is_per_cell_identity(name: &str) -> bool {
    name == "CellID" || name == "obs_names" || name == "cell_id"
}

fn betadata_feather_cell_mapping(
    all_names: &[String],
    label_idx: Option<usize>,
    row_labels: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
) -> Vec<usize> {
    let per_cell = label_idx
        .and_then(|i| all_names.get(i).map(|s| s.as_str()))
        .is_some_and(label_name_is_per_cell_identity);
    if per_cell {
        BetaFrame::compute_cell_mapping_cellid_rows(row_labels, obs_names)
    } else {
        BetaFrame::compute_cell_mapping(row_labels, obs_names, cluster_keys)
    }
}

fn feather_id_label_dtype_is_numeric(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// `Cluster` / `CellID` values as strings aligned with AnnData cluster codes (`"0"`, `"1"`, …).
/// **Numeric** feather columns are cast through `Int64` so float IDs like `3.0` become `"3"`, not `"3.0"`.
/// String `CellID` values (e.g. `c0`) stay on the direct string path — Utf8→Int64 in Polars yields nulls,
/// which would drop rows if we always normalized through integers.
fn feather_id_column_to_strings(col: &Column) -> Result<Vec<String>> {
    let string_col = if feather_id_label_dtype_is_numeric(col.dtype()) {
        col.cast(&DataType::Int64)?.cast(&DataType::String)?
    } else {
        col.cast(&DataType::String)?
    };
    Ok(string_col
        .str()?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect())
}

/// Detects how betadata rows map to cells: same precedence as [`betadata_feather_label_column_index`]
/// (**`CellID`** / **`obs_names`** / **`cell_id`** before **`Cluster`**). Used by the spatial viewer meta.
pub fn betadata_feather_row_id_column(path: &str) -> Result<Option<String>> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    Ok(betadata_feather_label_column_index(&all_names).and_then(|i| all_names.get(i).cloned()))
}

/// Numeric data columns suitable for spatial coloring (excludes id / label column).
pub fn betadata_feather_plottable_columns(path: &str) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let mut out = Vec::new();
    for (i, name) in all_names.iter().enumerate() {
        if Some(i) == label_idx {
            continue;
        }
        let col = df.column(name.as_str())?;
        if col.cast(&DataType::Float64).is_ok() {
            out.push(name.clone());
        }
    }
    out.sort();
    Ok(out)
}

/// One scalar per AnnData cell: feather row → cell via cluster-key + obs mapping for **`Cluster`**
/// feathers, or **obs-name-only** mapping when the id column is **`CellID`** (spatial CNN export).
pub fn betadata_feather_per_cell_column(
    path: &str,
    column: &str,
    obs_names: &[String],
    cluster_keys: &[String],
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let row_labels: Vec<String> = if let Some(idx) = label_idx {
        let label_name = &all_names[idx];
        feather_id_column_to_strings(df.column(label_name.as_str())?)?
    } else {
        (0..df.height()).map(|i| i.to_string()).collect()
    };
    let mapping =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let series = df
        .column(column)
        .with_context(|| format!("column {:?}", column))?
        .cast(&DataType::Float64)?;
    let ca = series.f64()?;
    let n_obs = obs_names.len();
    let mut out = vec![0f32; n_obs];
    for i in 0..n_obs {
        let r = mapping[i];
        let v = ca.get(r).unwrap_or(0.0);
        out[i] = v as f32;
    }
    Ok(out)
}

#[derive(Clone, Serialize)]
pub struct TopBetaCoefficient {
    pub column: String,
    pub mean: f64,
    pub mean_abs: f64,
}

fn is_intercept_column(name: &str) -> bool {
    name == "beta0" || name == "beta_0"
}

/// Mean and mean |β| per coefficient column over the given **cell** indices (obs order),
/// ranked by `mean_abs` descending. Skips intercept columns (`beta0` / `beta_0`).
pub fn betadata_feather_top_coefficients_for_selection(
    path: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_indices: &[usize],
    top_k: usize,
) -> Result<Vec<TopBetaCoefficient>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    if cell_indices.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }

    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let row_labels: Vec<String> = if let Some(idx) = label_idx {
        let label_name = &all_names[idx];
        feather_id_column_to_strings(df.column(label_name.as_str())?)?
    } else {
        (0..df.height()).map(|i| i.to_string()).collect()
    };
    let mapping =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let n_obs = obs_names.len();

    let mut columns: Vec<String> = Vec::new();
    for (i, name) in all_names.iter().enumerate() {
        if Some(i) == label_idx {
            continue;
        }
        if is_intercept_column(name) {
            continue;
        }
        let col = df.column(name.as_str())?;
        if col.cast(&DataType::Float64).is_ok() {
            columns.push(name.clone());
        }
    }

    let mut scores: Vec<(String, f64, f64)> = Vec::with_capacity(columns.len());

    for col_name in columns {
        let series = df.column(col_name.as_str())?.cast(&DataType::Float64)?;
        let ca = series.f64()?;
        let mut sum = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut cnt = 0usize;
        for &ci in cell_indices {
            if ci >= n_obs {
                continue;
            }
            let r = mapping[ci];
            let v = ca.get(r).unwrap_or(0.0);
            sum += v;
            sum_abs += v.abs();
            cnt += 1;
        }
        if cnt == 0 {
            continue;
        }
        scores.push((col_name, sum / cnt as f64, sum_abs / cnt as f64));
    }

    scores.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scores.truncate(top_k.min(scores.len()));

    Ok(scores
        .into_iter()
        .map(|(column, mean, mean_abs)| TopBetaCoefficient {
            column,
            mean,
            mean_abs,
        })
        .collect())
}

fn feather_column_modulator_key(name: &str) -> String {
    name.strip_prefix("beta_")
        .unwrap_or(name)
        .to_ascii_uppercase()
}

/// Mean β per modulator column (column name stripped of `beta_` prefix, ASCII-uppercase match) over
/// the given **cell** indices. One result per `modulators` entry; `None` if no numeric column matches.
pub fn betadata_feather_modulator_beta_means_for_cells(
    path: &str,
    modulators: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_indices: &[usize],
) -> Result<Vec<Option<f64>>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    if modulators.is_empty() {
        return Ok(Vec::new());
    }
    if cell_indices.is_empty() {
        return Ok(modulators.iter().map(|_| None).collect());
    }

    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let row_labels: Vec<String> = if let Some(idx) = label_idx {
        let label_name = &all_names[idx];
        feather_id_column_to_strings(df.column(label_name.as_str())?)?
    } else {
        (0..df.height()).map(|i| i.to_string()).collect()
    };
    let mapping =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let n_obs = obs_names.len();

    let mut col_by_mod: HashMap<String, String> = HashMap::new();
    for (i, name) in all_names.iter().enumerate() {
        if Some(i) == label_idx {
            continue;
        }
        if is_intercept_column(name) {
            continue;
        }
        let col = df.column(name.as_str())?;
        if col.cast(&DataType::Float64).is_err() {
            continue;
        }
        let key = feather_column_modulator_key(name);
        col_by_mod.entry(key).or_insert_with(|| name.clone());
    }

    let mut out = Vec::with_capacity(modulators.len());
    for m in modulators {
        let key = m.trim().to_ascii_uppercase();
        let Some(col_name) = col_by_mod.get(&key) else {
            out.push(None);
            continue;
        };
        let series = df.column(col_name.as_str())?.cast(&DataType::Float64)?;
        let ca = series.f64()?;
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for &ci in cell_indices {
            if ci >= n_obs {
                continue;
            }
            let r = mapping[ci];
            let v = ca.get(r).unwrap_or(0.0);
            sum += v;
            cnt += 1;
        }
        out.push(if cnt == 0 {
            None
        } else {
            Some(sum / cnt as f64)
        });
    }
    Ok(out)
}

/// One row of aggregated β across cells of a chosen type/cluster (Python `Betabase.collect_interactions`).
#[derive(Clone, Debug, Serialize)]
pub struct CollectedInteraction {
    pub interaction: String,
    pub gene: String,
    pub beta: f64,
    pub interaction_type: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetadataCollectAggregate {
    Mean,
    Min,
    Max,
    Sum,
    Positive,
    Negative,
}

impl BetadataCollectAggregate {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Some(Self::Mean),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "sum" => Some(Self::Sum),
            "positive" => Some(Self::Positive),
            "negative" => Some(Self::Negative),
            _ => None,
        }
    }
}

fn classify_betadata_column_type(col: &str) -> &'static str {
    let body = col.strip_prefix("beta_").unwrap_or(col);
    if body.contains('#') {
        "ligand-tf"
    } else if body.contains('$') {
        "ligand-receptor"
    } else {
        "tf"
    }
}

fn aggregate_values(vals: &[f64], mode: BetadataCollectAggregate) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    match mode {
        BetadataCollectAggregate::Mean => Some(vals.iter().sum::<f64>() / vals.len() as f64),
        BetadataCollectAggregate::Min => vals.iter().copied().reduce(f64::min),
        BetadataCollectAggregate::Max => vals.iter().copied().reduce(f64::max),
        BetadataCollectAggregate::Sum => Some(vals.iter().sum::<f64>()),
        BetadataCollectAggregate::Positive => {
            let p: Vec<f64> = vals.iter().copied().filter(|x| *x > 0.0).collect();
            if p.is_empty() {
                None
            } else {
                Some(p.iter().sum::<f64>() / p.len() as f64)
            }
        }
        BetadataCollectAggregate::Negative => {
            let p: Vec<f64> = vals.iter().copied().filter(|x| *x < 0.0).collect();
            if p.is_empty() {
                None
            } else {
                Some(p.iter().sum::<f64>() / p.len() as f64)
            }
        }
    }
}

/// Aggregates every β column in one target-gene feather for cells matching `cell_include_mask`.
pub fn betadata_collect_interactions_one_gene(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
    mode: BetadataCollectAggregate,
) -> Result<Vec<CollectedInteraction>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    anyhow::ensure!(
        obs_names.len() == cell_include_mask.len(),
        "obs_names len {} != mask len {}",
        obs_names.len(),
        cell_include_mask.len()
    );

    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let row_labels: Vec<String> = if let Some(idx) = label_idx {
        let label_name = &all_names[idx];
        feather_id_column_to_strings(df.column(label_name.as_str())?)?
    } else {
        (0..df.height()).map(|i| i.to_string()).collect()
    };
    let mapping =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let n_obs = obs_names.len();

    let mut out = Vec::new();
    for (i, col_name) in all_names.iter().enumerate() {
        if Some(i) == label_idx {
            continue;
        }
        if is_intercept_column(col_name) {
            continue;
        }
        let col = match df.column(col_name.as_str()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Ok(series) = col.cast(&DataType::Float64) else {
            continue;
        };
        let ca = series.f64()?;
        let mut vals: Vec<f64> = Vec::new();
        for (ci, &inc) in cell_include_mask.iter().enumerate() {
            if !inc || ci >= n_obs {
                continue;
            }
            let r = mapping[ci];
            let v = ca.get(r).unwrap_or(0.0);
            vals.push(v);
        }
        let Some(beta) = aggregate_values(&vals, mode) else {
            continue;
        };
        if beta == 0.0 || !beta.is_finite() {
            continue;
        }
        let itype = classify_betadata_column_type(col_name);
        out.push(CollectedInteraction {
            interaction: col_name.clone(),
            gene: target_gene.to_string(),
            beta,
            interaction_type: itype.to_string(),
        });
    }
    Ok(out)
}

/// Parallel scan of `genes.len()` feather files (Rayon). Missing files are skipped.
pub fn betadata_collect_interactions_parallel(
    dir: &str,
    genes: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
    mode: BetadataCollectAggregate,
) -> Result<Vec<CollectedInteraction>> {
    let dir_path = PathBuf::from(dir);
    let results: Vec<Result<Vec<CollectedInteraction>>> = genes
        .par_iter()
        .map(|gene| {
            let path = dir_path.join(format!("{}_betadata.feather", gene));
            if !path.is_file() {
                return Ok(Vec::new());
            }
            let ps = path.to_string_lossy().into_owned();
            betadata_collect_interactions_one_gene(
                &ps,
                gene.as_str(),
                obs_names,
                cluster_keys,
                cell_include_mask,
                mode,
            )
        })
        .collect();

    let mut merged = Vec::new();
    for r in results {
        merged.extend(r?);
    }
    merged.sort_by(|a, b| {
        b.beta
            .abs()
            .partial_cmp(&a.beta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.gene.cmp(&b.gene))
            .then_with(|| a.interaction.cmp(&b.interaction))
    });
    Ok(merged)
}

#[derive(Clone, Debug, Serialize)]
pub struct PairLrBetaRow {
    pub target_gene: String,
    pub interaction: String,
    pub beta_cell_a: f64,
    pub beta_cell_b: f64,
    /// `max(|beta_cell_a|, |beta_cell_b|)` for ranking.
    pub score: f64,
}

/// Per target-gene feather: ligand–receptor β at the feather rows mapped to `cell_a` and `cell_b`.
pub fn betadata_pair_lr_one_gene(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_a: usize,
    cell_b: usize,
) -> Result<Vec<PairLrBetaRow>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    anyhow::ensure!(
        cell_a < obs_names.len() && cell_b < obs_names.len(),
        "cell index out of range (n_obs = {})",
        obs_names.len()
    );
    anyhow::ensure!(cell_a != cell_b, "cell_a and cell_b must differ");

    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let df = IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))?;
    let all_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let label_idx = betadata_feather_label_column_index(&all_names);
    let row_labels: Vec<String> = if let Some(idx) = label_idx {
        let label_name = &all_names[idx];
        feather_id_column_to_strings(df.column(label_name.as_str())?)?
    } else {
        (0..df.height()).map(|i| i.to_string()).collect()
    };
    let mapping =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let ra = mapping[cell_a];
    let rb = mapping[cell_b];

    let mut out = Vec::new();
    for (i, col_name) in all_names.iter().enumerate() {
        if Some(i) == label_idx {
            continue;
        }
        if is_intercept_column(col_name) {
            continue;
        }
        if classify_betadata_column_type(col_name) != "ligand-receptor" {
            continue;
        }
        let col = match df.column(col_name.as_str()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Ok(series) = col.cast(&DataType::Float64) else {
            continue;
        };
        let ca = series.f64()?;
        let v_a = ca.get(ra).unwrap_or(0.0);
        let v_b = ca.get(rb).unwrap_or(0.0);
        if !v_a.is_finite() || !v_b.is_finite() {
            continue;
        }
        let score = v_a.abs().max(v_b.abs());
        if score == 0.0 || !score.is_finite() {
            continue;
        }
        out.push(PairLrBetaRow {
            target_gene: target_gene.to_string(),
            interaction: col_name.clone(),
            beta_cell_a: v_a,
            beta_cell_b: v_b,
            score,
        });
    }
    Ok(out)
}

/// Parallel scan of target-gene feathers; merges and sorts by `score` descending.
pub fn betadata_pair_lr_parallel(
    dir: &str,
    genes: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_a: usize,
    cell_b: usize,
) -> Result<Vec<PairLrBetaRow>> {
    let dir_path = PathBuf::from(dir);
    let results: Vec<Result<Vec<PairLrBetaRow>>> = genes
        .par_iter()
        .map(|gene| {
            let path = dir_path.join(format!("{}_betadata.feather", gene));
            if !path.is_file() {
                return Ok(Vec::new());
            }
            let ps = path.to_string_lossy().into_owned();
            betadata_pair_lr_one_gene(&ps, gene.as_str(), obs_names, cluster_keys, cell_a, cell_b)
        })
        .collect();

    let mut merged = Vec::new();
    for r in results {
        merged.extend(r?);
    }
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.target_gene.cmp(&b.target_gene))
            .then_with(|| a.interaction.cmp(&b.interaction))
    });
    Ok(merged)
}

#[cfg(test)]
mod feather_label_tests {
    use super::betadata_feather_label_column_index;

    #[test]
    fn label_index_prefers_cellid_when_cluster_is_first_column() {
        let names = vec!["Cluster".into(), "CellID".into(), "beta0".into()];
        assert_eq!(betadata_feather_label_column_index(&names), Some(1));
    }

    #[test]
    fn label_index_falls_back_to_cluster_when_no_cellid() {
        let names = vec!["Cluster".into(), "beta0".into()];
        assert_eq!(betadata_feather_label_column_index(&names), Some(0));
    }
}
