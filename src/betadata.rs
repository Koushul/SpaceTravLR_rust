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

/// Mapping stats from [`BetaFrame::compute_cell_mapping`] / [`BetaFrame::compute_cell_mapping_cellid_rows`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMappingSummary {
    pub n_unmapped: usize,
    pub n_cells: usize,
    pub n_via_cluster_key: usize,
    pub n_via_obs_id: usize,
    pub per_cell_join: bool,
}

impl CellMappingSummary {
    pub fn log_unmapped_warning(&self) {
        if self.n_unmapped == 0 {
            return;
        }
        if self.per_cell_join {
            eprintln!(
                "Warning: {} of {} cells could not map to a betadata CellID row; using zero betas for those.",
                self.n_unmapped, self.n_cells
            );
        } else {
            eprintln!(
                "Warning: {} of {} cells could not map to a betadata row; using zero betas for those. ({} cells mapped via cluster key, {} via obs id.)",
                self.n_unmapped,
                self.n_cells,
                self.n_via_cluster_key,
                self.n_via_obs_id
            );
        }
    }
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
    /// Returns mapping + summary (caller logs the summary once per load, not per target gene).
    ///
    /// For each cell: match `cluster_keys[i]` to a row label (seed-only `Cluster` column), else
    /// **obs name** (per-cell `CellID` export), else [`Self::missing_beta_row_index`] (all β = 0).
    /// `cluster_keys` must use the same strings as in the feather (e.g. categorical **names**
    /// `"10"`, not code `2`).
    pub fn compute_cell_mapping(
        row_labels: &[String],
        obs_names: &[String],
        cluster_keys: &[String],
    ) -> (Vec<usize>, CellMappingSummary) {
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
        let missing = Self::missing_beta_row_index(row_labels.len());

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
                missing
            };
            mapping.push(idx);
        }

        (
            mapping,
            CellMappingSummary {
                n_unmapped: n_default,
                n_cells: obs_names.len(),
                n_via_cluster_key: n_via_key,
                n_via_obs_id: n_via_obs,
                per_cell_join: false,
            },
        )
    }

    /// `cell_to_beta_row` value when no feather row matches: [`Self::splash`] treats this as all-zero β.
    #[inline]
    pub fn missing_beta_row_index(n_feather_rows: usize) -> usize {
        n_feather_rows
    }

    /// Map each AnnData cell to a feather row by **`obs_names[i]`** matching `row_labels`.
    /// Used when the feather id column is **`CellID`**. Do not use `cluster_keys` here: those
    /// are cell-type (or other) labels and can spuriously match row ids or assign zero β via the
    /// missing-row sentinel, destroying per-cell spatial β variation.
    pub fn compute_cell_mapping_cellid_rows(
        row_labels: &[String],
        obs_names: &[String],
    ) -> (Vec<usize>, CellMappingSummary) {
        let mut row_map: HashMap<&str, usize> = HashMap::new();
        for (i, l) in row_labels.iter().enumerate() {
            row_map.entry(l.as_str()).or_insert(i);
        }
        let missing = Self::missing_beta_row_index(row_labels.len());
        let mut n_default = 0usize;
        let mut n_via_obs = 0usize;
        let mapping: Vec<usize> = obs_names
            .iter()
            .map(|name| {
                if let Some(&i) = row_map.get(name.as_str()) {
                    n_via_obs += 1;
                    i
                } else {
                    n_default += 1;
                    missing
                }
            })
            .collect();
        (
            mapping,
            CellMappingSummary {
                n_unmapped: n_default,
                n_cells: obs_names.len(),
                n_via_cluster_key: 0,
                n_via_obs_id: n_via_obs,
                per_cell_join: true,
            },
        )
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
    /// `[perturbation].beta_scale_factor` is passed as `ligand_beta_scale_factor` at splash time
    /// (not when loading seed-only `Cluster` feathers).
    ///
    ///   dy/dTF       = beta_TF
    ///   dy/dR        = beta_LR * wL        (gex[R] > 0, × ligand_beta_scale_factor)
    ///   dy/dL(lr)    = beta_LR * gex[R]   (× ligand_beta_scale_factor)
    ///   dy/dL(tfl)   = beta_TFL * gex[reg] (× ligand_beta_scale_factor)
    ///   dy/dTF(tfl)  = beta_TFL * wL_tfl  (TF regulator; no ligand scale)
    pub fn splash(
        &self,
        rw_ligands: &GeneMatrix,
        rw_ligands_tfl: &GeneMatrix,
        gex_df: &GeneMatrix,
        ligand_beta_scale_factor: f32,
        beta_cap: Option<f32>,
    ) -> GeneMatrix {
        let n = self.n_cells;
        let map = self.cell_to_beta_row.as_slice();
        let n_out = self.modulator_genes.len();
        if n_out == 0 {
            return GeneMatrix::new(Array2::zeros((n, 0)), vec![]);
        }
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

        let n_beta_rows = self.n_beta_rows;
        result.par_chunks_mut(n_out).enumerate().for_each(|(i, r)| {
            let br = unsafe { *map.get_unchecked(i) };
            if br >= n_beta_rows {
                return;
            }
            let rw_base = i * rw_nc;
            let rw_tfl_base = i * rw_tfl_nc;
            let gex_base = i * gex_nc;

            // 1. TF derivatives (plain TF modulators; no ligand_beta_scale_factor)
            let tf_base = br * n_tfs;
            for j in 0..n_tfs {
                unsafe {
                    *r.get_unchecked_mut(*tf_oi.get_unchecked(j)) +=
                        *tf_flat.get_unchecked(tf_base + j);
                }
            }

            let lbs = ligand_beta_scale_factor;

            // 2+3. LR derivatives (ligand + receptor)
            let lr_beta_base = br * n_lr;
            for lw in &lr_work {
                let beta = unsafe { *lr_flat.get_unchecked(lr_beta_base + lw.beta_col) };
                let wl = unsafe { *rw_flat.get_unchecked(rw_base + lw.wl_col) };
                let gex = unsafe { *gex_flat.get_unchecked(gex_base + lw.gex_col) };

                if gex > 0.0f32 {
                    unsafe { *r.get_unchecked_mut(lw.rec_oi) += beta * wl * lbs };
                }
                unsafe { *r.get_unchecked_mut(lw.lig_oi) += beta * gex * lbs };
            }

            // 4+5. TFL: scale ligand leg only; regulator is a TF modulator
            let tfl_beta_base = br * n_tfl;
            for tw in &tfl_work {
                let beta = unsafe { *tfl_flat.get_unchecked(tfl_beta_base + tw.beta_col) };
                let gex_reg = unsafe { *gex_flat.get_unchecked(gex_base + tw.gex_col) };
                let wl = unsafe { *rw_tfl_flat.get_unchecked(rw_tfl_base + tw.wl_col) };

                unsafe { *r.get_unchecked_mut(tw.lig_oi) += beta * gex_reg * lbs };
                unsafe { *r.get_unchecked_mut(tw.reg_oi) += beta * wl };
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
        let mut unmapped_summary: Option<CellMappingSummary> = None;

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
                let (mapping_vec, summary) = if frame.join_by_obs_name {
                    BetaFrame::compute_cell_mapping_cellid_rows(&frame.row_labels, obs_names)
                } else {
                    BetaFrame::compute_cell_mapping(&frame.row_labels, obs_names, cluster_keys)
                };
                if summary.n_unmapped > 0 {
                    unmapped_summary = Some(match unmapped_summary {
                        None => summary,
                        Some(prev) if summary.n_unmapped >= prev.n_unmapped => summary,
                        Some(prev) => prev,
                    });
                }
                let m = Arc::new(mapping_vec);
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

        if let Some(summary) = unmapped_summary {
            summary.log_unmapped_warning();
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
) -> (Vec<usize>, CellMappingSummary) {
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
    let (mapping, _) =
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
    let (mapping, _) =
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
    let (mapping, _) =
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

/// One row of the multi–cell-type interactions database (`interaction`, `target_gene`, `beta`, …).
#[derive(Clone, Debug, Serialize)]
pub struct CollectedInteractionRow {
    pub interaction: String,
    pub target_gene: String,
    pub beta: f64,
    pub interaction_type: String,
    pub cell_type: String,
}

/// All β aggregation modes computed in one pass over matching cells.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct BetaAggregates {
    pub mean: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub sum: Option<f64>,
    pub positive: Option<f64>,
    pub negative: Option<f64>,
}

/// Multi–cell-type interaction row with every aggregation as its own column.
#[derive(Clone, Debug, Serialize)]
pub struct CollectedInteractionRowFull {
    pub interaction: String,
    pub target_gene: String,
    pub interaction_type: String,
    pub cell_type: String,
    /// Set when collecting independently per `obs[cluster_col]` partition.
    pub cluster: Option<String>,
    pub aggregates: BetaAggregates,
}

/// Single-selection interaction row with every aggregation as its own column.
#[derive(Clone, Debug, Serialize)]
pub struct CollectedInteractionFull {
    pub interaction: String,
    pub gene: String,
    pub interaction_type: String,
    pub cluster: Option<String>,
    pub aggregates: BetaAggregates,
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

fn aggregates_have_signal(agg: &BetaAggregates) -> bool {
    [agg.mean, agg.min, agg.max, agg.sum, agg.positive, agg.negative]
        .into_iter()
        .flatten()
        .any(|v| v.is_finite() && v.abs() > 1e-15)
}

fn aggregate_sort_key(agg: &BetaAggregates) -> f64 {
    [agg.mean, agg.min, agg.max, agg.sum, agg.positive, agg.negative]
        .into_iter()
        .flatten()
        .map(f64::abs)
        .fold(0.0f64, f64::max)
}

fn unique_sorted_cell_types(labels: &[String]) -> Vec<String> {
    let mut seen = HashSet::with_capacity(labels.len().min(256));
    for l in labels {
        seen.insert(l.as_str());
    }
    let mut out: Vec<String> = seen.into_iter().map(str::to_string).collect();
    out.sort();
    out
}

fn ensure_obs_slices_same_len(
    obs_names: usize,
    cluster_keys: usize,
    cell_type_labels: usize,
    cluster_obs: Option<usize>,
) -> Result<()> {
    anyhow::ensure!(
        obs_names == cluster_keys,
        "obs_names len {obs_names} != cluster_keys len {cluster_keys}"
    );
    anyhow::ensure!(
        obs_names == cell_type_labels,
        "obs_names len {obs_names} != cell_type_labels len {cell_type_labels}"
    );
    if let Some(n) = cluster_obs {
        anyhow::ensure!(
            obs_names == n,
            "obs_names len {obs_names} != cluster_obs len {n}"
        );
    }
    Ok(())
}

fn partition_indices_by_label(labels: &[String]) -> Vec<(String, Vec<usize>)> {
    unique_sorted_cell_types(labels)
        .into_iter()
        .map(|lab| {
            let indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, l)| l.as_str() == lab.as_str())
                .map(|(i, _)| i)
                .collect();
            (lab, indices)
        })
        .collect()
}

/// Precomputed per-cell-type row indices into the feather (through the `mapping`).
/// Avoids scanning the full `n_obs` boolean mask for every (column × cell_type).
struct CellTypeIndices {
    labels: Vec<Arc<str>>,
    mapped_rows: Vec<Vec<usize>>,
}

impl CellTypeIndices {
    fn build(
        cell_type_labels: &[String],
        unique_cell_types: &[Arc<str>],
        mapping: &[usize],
    ) -> Self {
        let n_obs = cell_type_labels.len();
        let mut label_to_idx: HashMap<&str, usize> = HashMap::with_capacity(unique_cell_types.len());
        for (i, ct) in unique_cell_types.iter().enumerate() {
            label_to_idx.insert(ct.as_ref(), i);
        }
        let mut mapped_rows: Vec<Vec<usize>> = vec![Vec::new(); unique_cell_types.len()];
        for ci in 0..n_obs {
            if let Some(&ct_i) = label_to_idx.get(cell_type_labels[ci].as_str()) {
                mapped_rows[ct_i].push(mapping[ci]);
            }
        }
        CellTypeIndices {
            labels: unique_cell_types.to_vec(),
            mapped_rows,
        }
    }

    /// Like [`Self::build`], but only includes obs rows in `obs_subset` (for cluster partitions).
    fn build_for_obs_subset(
        cell_type_labels: &[String],
        unique_cell_types: &[Arc<str>],
        mapping: &[usize],
        obs_subset: &[usize],
    ) -> Self {
        let mut label_to_idx: HashMap<&str, usize> = HashMap::with_capacity(unique_cell_types.len());
        for (i, ct) in unique_cell_types.iter().enumerate() {
            label_to_idx.insert(ct.as_ref(), i);
        }
        let mut mapped_rows: Vec<Vec<usize>> = vec![Vec::new(); unique_cell_types.len()];
        for &ci in obs_subset {
            if let Some(&ct_i) = label_to_idx.get(cell_type_labels[ci].as_str()) {
                mapped_rows[ct_i].push(mapping[ci]);
            }
        }
        CellTypeIndices {
            labels: unique_cell_types.to_vec(),
            mapped_rows,
        }
    }
}

/// Per–obs-cluster partition: cell-type row lists for cells in that cluster only.
struct ClusterPartitionCellTypes {
    labels: Vec<String>,
    per_partition: Vec<CellTypeIndices>,
}

enum InteractionGrouping {
    CellTypes(CellTypeIndices),
    /// One independent aggregation per `obs[cluster_col]` value; feather read once.
    ClusterPartitions(ClusterPartitionCellTypes),
}

/// One β coefficient column materialized as `f64` for hot aggregation loops.
struct CoefColumnData {
    name: Arc<str>,
    interaction_type: &'static str,
    values: Vec<f64>,
}

/// Per-feather state shared across all coefficient columns (one IPC read per file).
struct CollectGeneWorkspace {
    target_gene: Arc<str>,
    coef_columns: Vec<CoefColumnData>,
    grouping: InteractionGrouping,
}

enum MaskInteractionGrouping {
    Single(Vec<usize>),
    ClusterPartitions {
        labels: Vec<String>,
        masked_rows: Vec<Vec<usize>>,
    },
}

/// Per-feather state for mask-filtered collection (one IPC read per file).
struct CollectGeneMaskWorkspace {
    target_gene: Arc<str>,
    coef_columns: Vec<CoefColumnData>,
    grouping: MaskInteractionGrouping,
}

fn build_cluster_partition_cell_types(
    cell_type_labels: &[String],
    mapping: &[usize],
    cluster_obs: &[String],
) -> ClusterPartitionCellTypes {
    let partitions = partition_indices_by_label(cluster_obs);
    let mut labels = Vec::with_capacity(partitions.len());
    let mut per_partition = Vec::with_capacity(partitions.len());
    for (lab, obs_indices) in partitions {
        let ct_in_part: Vec<String> = obs_indices
            .iter()
            .map(|&i| cell_type_labels[i].clone())
            .collect();
        let unique_ct: Vec<Arc<str>> = unique_sorted_cell_types(&ct_in_part)
            .into_iter()
            .map(|s| Arc::from(s.as_str()))
            .collect();
        labels.push(lab);
        per_partition.push(CellTypeIndices::build_for_obs_subset(
            cell_type_labels,
            &unique_ct,
            mapping,
            &obs_indices,
        ));
    }
    ClusterPartitionCellTypes {
        labels,
        per_partition,
    }
}

fn read_feather_coef_columns_and_mapping(
    path: &str,
    obs_names: &[String],
    cluster_keys: &[String],
) -> Result<(Vec<CoefColumnData>, Vec<usize>)> {
    let df = read_betadata_feather_df(path)?;
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
    let (mapping, _) =
        betadata_feather_cell_mapping(&all_names, label_idx, &row_labels, obs_names, cluster_keys);
    let coef_columns = materialize_coef_columns(&df, &all_names, label_idx)?;
    Ok((coef_columns, mapping))
}

fn list_betadata_feather_gene_paths(
    dir: &Path,
    max_genes: Option<usize>,
) -> Result<Vec<(String, PathBuf)>> {
    let mut paths: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_str()?;
            let stem = name.strip_suffix("_betadata.feather")?;
            if stem.is_empty() {
                return None;
            }
            Some((stem.to_string(), p))
        })
        .collect();
    paths.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(cap) = max_genes {
        paths.truncate(cap.min(paths.len()));
    }
    Ok(paths)
}

fn read_betadata_feather_df(path: &str) -> Result<DataFrame> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    IpcReader::new(f)
        .finish()
        .with_context(|| format!("read IPC {}", path))
}

fn materialize_coef_columns(
    df: &DataFrame,
    all_names: &[String],
    label_idx: Option<usize>,
) -> Result<Vec<CoefColumnData>> {
    let mut out = Vec::new();
    for (i, name) in all_names.iter().enumerate() {
        if Some(i) == label_idx || is_intercept_column(name) {
            continue;
        }
        let col = match df.column(name.as_str()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Ok(series) = col.cast(&DataType::Float64) else {
            continue;
        };
        let ca = series.f64()?;
        let n_rows = ca.len();
        let values: Vec<f64> = (0..n_rows).map(|i| ca.get(i).unwrap_or(0.0)).collect();
        out.push(CoefColumnData {
            name: Arc::from(name.as_str()),
            interaction_type: classify_betadata_column_type(name),
            values,
        });
    }
    Ok(out)
}

fn load_collect_gene_workspace(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    cluster_obs: Option<&[String]>,
    unique_cell_types: &[Arc<str>],
) -> Result<CollectGeneWorkspace> {
    let (coef_columns, mapping) =
        read_feather_coef_columns_and_mapping(path, obs_names, cluster_keys)?;
    let grouping = if let Some(labels) = cluster_obs {
        InteractionGrouping::ClusterPartitions(build_cluster_partition_cell_types(
            cell_type_labels,
            &mapping,
            labels,
        ))
    } else {
        InteractionGrouping::CellTypes(CellTypeIndices::build(
            cell_type_labels,
            unique_cell_types,
            &mapping,
        ))
    };
    Ok(CollectGeneWorkspace {
        target_gene: Arc::from(target_gene),
        coef_columns,
        grouping,
    })
}

fn load_collect_gene_workspace_all_ct(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    unique_cell_types: &[Arc<str>],
) -> Result<CollectGeneWorkspace> {
    load_collect_gene_workspace(
        path,
        target_gene,
        obs_names,
        cluster_keys,
        cell_type_labels,
        None,
        unique_cell_types,
    )
}

fn load_collect_gene_mask_workspace(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
    cluster_obs: Option<&[String]>,
) -> Result<CollectGeneMaskWorkspace> {
    let (coef_columns, mapping) =
        read_feather_coef_columns_and_mapping(path, obs_names, cluster_keys)?;
    let n_obs = obs_names.len();
    let grouping = if let Some(labels) = cluster_obs {
        let partitions = partition_indices_by_label(labels);
        let mut cluster_labels = Vec::with_capacity(partitions.len());
        let mut masked_rows = Vec::with_capacity(partitions.len());
        for (lab, obs_indices) in partitions {
            let rows: Vec<usize> = obs_indices
                .iter()
                .filter(|&&ci| ci < n_obs && cell_include_mask[ci])
                .map(|&ci| mapping[ci])
                .collect();
            cluster_labels.push(lab);
            masked_rows.push(rows);
        }
        MaskInteractionGrouping::ClusterPartitions {
            labels: cluster_labels,
            masked_rows,
        }
    } else {
        let rows: Vec<usize> = cell_include_mask
            .iter()
            .enumerate()
            .filter(|(ci, inc)| **inc && *ci < n_obs)
            .map(|(ci, _)| mapping[ci])
            .collect();
        MaskInteractionGrouping::Single(rows)
    };
    Ok(CollectGeneMaskWorkspace {
        target_gene: Arc::from(target_gene),
        coef_columns,
        grouping,
    })
}

fn collect_rows_for_cell_type_indices(
    coef: &CoefColumnData,
    target_gene: &Arc<str>,
    ct_idx: &CellTypeIndices,
    mode: BetadataCollectAggregate,
) -> Vec<CollectedInteractionRow> {
    let mut local = Vec::with_capacity(ct_idx.mapped_rows.len());
    for (ct_i, rows) in ct_idx.mapped_rows.iter().enumerate() {
        let Some(beta) = aggregate_mapped_column(&coef.values, rows, mode) else {
            continue;
        };
        if !beta.is_finite() || beta.abs() <= 1e-15 {
            continue;
        }
        local.push(CollectedInteractionRow {
            interaction: coef.name.to_string(),
            target_gene: target_gene.to_string(),
            beta,
            interaction_type: coef.interaction_type.to_string(),
            cell_type: ct_idx.labels[ct_i].to_string(),
        });
    }
    local
}

fn collect_rows_full_for_cell_type_indices(
    coef: &CoefColumnData,
    target_gene: &Arc<str>,
    ct_idx: &CellTypeIndices,
    cluster: Option<&str>,
) -> Vec<CollectedInteractionRowFull> {
    let mut local = Vec::with_capacity(ct_idx.mapped_rows.len());
    for (ct_i, rows) in ct_idx.mapped_rows.iter().enumerate() {
        let aggregates = aggregate_mapped_column_all(&coef.values, rows);
        if !aggregates_have_signal(&aggregates) {
            continue;
        }
        local.push(CollectedInteractionRowFull {
            interaction: coef.name.to_string(),
            target_gene: target_gene.to_string(),
            interaction_type: coef.interaction_type.to_string(),
            cell_type: ct_idx.labels[ct_i].to_string(),
            cluster: cluster.map(str::to_string),
            aggregates,
        });
    }
    local
}

fn collect_interactions_all_cell_types_from_workspace(
    ws: &CollectGeneWorkspace,
    mode: BetadataCollectAggregate,
) -> Vec<CollectedInteractionRow> {
    match &ws.grouping {
        InteractionGrouping::CellTypes(ct_idx) => ws
            .coef_columns
            .par_iter()
            .map(|coef| collect_rows_for_cell_type_indices(coef, &ws.target_gene, ct_idx, mode))
            .reduce(
                Vec::new,
                |mut acc, mut chunk| {
                    acc.append(&mut chunk);
                    acc
                },
            ),
        InteractionGrouping::ClusterPartitions(parts) => parts
            .labels
            .par_iter()
            .zip(parts.per_partition.par_iter())
            .map(|(_lab, ct_idx)| {
                ws.coef_columns
                    .par_iter()
                    .map(|coef| {
                        collect_rows_for_cell_type_indices(coef, &ws.target_gene, ct_idx, mode)
                    })
                    .reduce(
                        Vec::new,
                        |mut acc, mut chunk| {
                            acc.append(&mut chunk);
                            acc
                        },
                    )
            })
            .reduce(
                Vec::new,
                |mut acc, mut chunk| {
                    acc.append(&mut chunk);
                    acc
                },
            ),
    }
}

fn collect_interactions_all_cell_types_full_from_workspace(
    ws: &CollectGeneWorkspace,
) -> Vec<CollectedInteractionRowFull> {
    match &ws.grouping {
        InteractionGrouping::CellTypes(ct_idx) => ws
            .coef_columns
            .par_iter()
            .map(|coef| {
                collect_rows_full_for_cell_type_indices(coef, &ws.target_gene, ct_idx, None)
            })
            .reduce(
                Vec::new,
                |mut acc, mut chunk| {
                    acc.append(&mut chunk);
                    acc
                },
            ),
        InteractionGrouping::ClusterPartitions(parts) => parts
            .labels
            .par_iter()
            .zip(parts.per_partition.par_iter())
            .map(|(lab, ct_idx)| {
                ws.coef_columns
                    .par_iter()
                    .map(|coef| {
                        collect_rows_full_for_cell_type_indices(
                            coef,
                            &ws.target_gene,
                            ct_idx,
                            Some(lab.as_str()),
                        )
                    })
                    .reduce(
                        Vec::new,
                        |mut acc, mut chunk| {
                            acc.append(&mut chunk);
                            acc
                        },
                    )
            })
            .reduce(
                Vec::new,
                |mut acc, mut chunk| {
                    acc.append(&mut chunk);
                    acc
                },
            ),
    }
}

fn collect_interactions_mask_from_workspace(
    ws: &CollectGeneMaskWorkspace,
    mode: BetadataCollectAggregate,
) -> Vec<CollectedInteraction> {
    let MaskInteractionGrouping::Single(rows) = &ws.grouping else {
        unreachable!("mask single-β collection does not use cluster partitions");
    };
    if rows.is_empty() {
        return Vec::new();
    }
    ws.coef_columns
        .par_iter()
        .filter_map(|coef| {
            let beta = aggregate_mapped_column(&coef.values, rows, mode)?;
            if !beta.is_finite() || beta.abs() <= 1e-15 {
                return None;
            }
            Some(CollectedInteraction {
                interaction: coef.name.to_string(),
                gene: ws.target_gene.to_string(),
                beta,
                interaction_type: coef.interaction_type.to_string(),
            })
        })
        .collect()
}

fn collect_interactions_mask_full_from_workspace(
    ws: &CollectGeneMaskWorkspace,
) -> Vec<CollectedInteractionFull> {
    match &ws.grouping {
        MaskInteractionGrouping::Single(rows) if rows.is_empty() => Vec::new(),
        MaskInteractionGrouping::Single(rows) => ws
            .coef_columns
            .par_iter()
            .filter_map(|coef| {
                let aggregates = aggregate_mapped_column_all(&coef.values, rows);
                if !aggregates_have_signal(&aggregates) {
                    return None;
                }
                Some(CollectedInteractionFull {
                    interaction: coef.name.to_string(),
                    gene: ws.target_gene.to_string(),
                    interaction_type: coef.interaction_type.to_string(),
                    cluster: None,
                    aggregates,
                })
            })
            .collect(),
        MaskInteractionGrouping::ClusterPartitions { labels, masked_rows } => labels
            .par_iter()
            .zip(masked_rows.par_iter())
            .flat_map(|(lab, rows)| {
                if rows.is_empty() {
                    return Vec::new();
                }
                ws.coef_columns
                    .par_iter()
                    .filter_map(|coef| {
                        let aggregates = aggregate_mapped_column_all(&coef.values, rows);
                        if !aggregates_have_signal(&aggregates) {
                            return None;
                        }
                        Some(CollectedInteractionFull {
                            interaction: coef.name.to_string(),
                            gene: ws.target_gene.to_string(),
                            interaction_type: coef.interaction_type.to_string(),
                            cluster: Some(lab.clone()),
                            aggregates,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    }
}

#[inline]
fn aggregate_mapped_column_all(col_data: &[f64], feather_rows: &[usize]) -> BetaAggregates {
    if feather_rows.is_empty() {
        return BetaAggregates::default();
    }
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut pos_sum = 0.0f64;
    let mut pos_cnt = 0usize;
    let mut neg_sum = 0.0f64;
    let mut neg_cnt = 0usize;
    for &r in feather_rows {
        let x = *col_data.get(r).unwrap_or(&0.0);
        sum += x;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        if x > 0.0 {
            pos_sum += x;
            pos_cnt += 1;
        }
        if x < 0.0 {
            neg_sum += x;
            neg_cnt += 1;
        }
    }
    let n = feather_rows.len() as f64;
    BetaAggregates {
        mean: Some(sum / n),
        min: Some(min),
        max: Some(max),
        sum: Some(sum),
        positive: if pos_cnt == 0 {
            None
        } else {
            Some(pos_sum / pos_cnt as f64)
        },
        negative: if neg_cnt == 0 {
            None
        } else {
            Some(neg_sum / neg_cnt as f64)
        },
    }
}

#[inline]
fn aggregate_mapped_column(
    col_data: &[f64],
    feather_rows: &[usize],
    mode: BetadataCollectAggregate,
) -> Option<f64> {
    if feather_rows.is_empty() {
        return None;
    }
    match mode {
        BetadataCollectAggregate::Mean => {
            let mut sum = 0.0f64;
            for &r in feather_rows {
                sum += *col_data.get(r).unwrap_or(&0.0);
            }
            Some(sum / feather_rows.len() as f64)
        }
        BetadataCollectAggregate::Sum => {
            let mut sum = 0.0f64;
            for &r in feather_rows {
                sum += *col_data.get(r).unwrap_or(&0.0);
            }
            Some(sum)
        }
        BetadataCollectAggregate::Min => {
            let mut v = f64::INFINITY;
            for &r in feather_rows {
                let x = *col_data.get(r).unwrap_or(&0.0);
                if x < v { v = x; }
            }
            Some(v)
        }
        BetadataCollectAggregate::Max => {
            let mut v = f64::NEG_INFINITY;
            for &r in feather_rows {
                let x = *col_data.get(r).unwrap_or(&0.0);
                if x > v { v = x; }
            }
            Some(v)
        }
        BetadataCollectAggregate::Positive => {
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for &r in feather_rows {
                let x = *col_data.get(r).unwrap_or(&0.0);
                if x > 0.0 { sum += x; cnt += 1; }
            }
            if cnt == 0 { None } else { Some(sum / cnt as f64) }
        }
        BetadataCollectAggregate::Negative => {
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for &r in feather_rows {
                let x = *col_data.get(r).unwrap_or(&0.0);
                if x < 0.0 { sum += x; cnt += 1; }
            }
            if cnt == 0 { None } else { Some(sum / cnt as f64) }
        }
    }
}

/// Like [`betadata_collect_interactions_one_gene`], but emits one row per (coefficient × cell type).
///
/// Uses precomputed per-cell-type row-index lists so the inner loop touches only matching cells,
/// and materializes each Polars column as a contiguous `&[f64]` slice for zero-overhead access.
pub fn betadata_collect_interactions_all_cell_types_one_gene(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    unique_cell_types: &[Arc<str>],
    mode: BetadataCollectAggregate,
) -> Result<Vec<CollectedInteractionRow>> {
    let ws = load_collect_gene_workspace_all_ct(
        path,
        target_gene,
        obs_names,
        cluster_keys,
        cell_type_labels,
        unique_cell_types,
    )?;
    Ok(collect_interactions_all_cell_types_from_workspace(&ws, mode))
}

/// Like [`betadata_collect_interactions_all_cell_types_one_gene`], but emits mean/min/max/sum/positive/negative columns.
pub fn betadata_collect_interactions_all_cell_types_one_gene_full(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    unique_cell_types: &[Arc<str>],
) -> Result<Vec<CollectedInteractionRowFull>> {
    let ws = load_collect_gene_workspace_all_ct(
        path,
        target_gene,
        obs_names,
        cluster_keys,
        cell_type_labels,
        unique_cell_types,
    )?;
    Ok(collect_interactions_all_cell_types_full_from_workspace(&ws))
}

fn sort_collected_interaction_rows_full(merged: &mut [CollectedInteractionRowFull]) {
    merged.par_sort_unstable_by(|a, b| {
        aggregate_sort_key(&b.aggregates)
            .partial_cmp(&aggregate_sort_key(&a.aggregates))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cluster.cmp(&b.cluster))
            .then_with(|| a.cell_type.cmp(&b.cell_type))
            .then_with(|| a.target_gene.cmp(&b.target_gene))
            .then_with(|| a.interaction.cmp(&b.interaction))
    });
}

/// Scan pre-listed feathers with all aggregation columns (parallel per gene, one read per file).
fn betadata_collect_interactions_all_cell_types_full_paths(
    _dir: &str,
    paths: &[(String, PathBuf)],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    cluster_obs: Option<&[String]>,
) -> Result<Vec<CollectedInteractionRowFull>> {
    if let Some(labels) = cluster_obs {
        anyhow::ensure!(
            !unique_sorted_cell_types(labels).is_empty(),
            "no distinct cluster values in cluster_obs — check obs column"
        );
    } else {
        let unique_cell_types = unique_sorted_cell_types(cell_type_labels);
        anyhow::ensure!(
            !unique_cell_types.is_empty(),
            "no distinct cell types in labels — check obs annotation column"
        );
    }

    let unique_cell_types = unique_sorted_cell_types(cell_type_labels);
    let unique_arcs: Vec<Arc<str>> = unique_cell_types.iter().map(|s| Arc::from(s.as_str())).collect();
    let unique_arcs = Arc::new(unique_arcs);
    let obs_names = Arc::new(obs_names.to_vec());
    let cluster_keys = Arc::new(cluster_keys.to_vec());
    let cell_type_labels_arc = Arc::new(cell_type_labels.to_vec());
    let cluster_obs_arc = cluster_obs.map(|s| Arc::new(s.to_vec()));

    let n_total = paths.len();
    let pb = if std::io::stderr().is_terminal() && n_total > 0 {
        let bar = indicatif::ProgressBar::new(n_total as u64);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {per_sec} eta {eta} collect-interactions",
                )?
                .progress_chars("#>-"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(200));
        Some(Arc::new(bar))
    } else if n_total > 0 {
        eprintln!("Scanning {n_total} betadata feathers…");
        None
    } else {
        None
    };

    let row_counts: Arc<std::sync::atomic::AtomicUsize> =
        Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let results: Vec<Vec<CollectedInteractionRowFull>> = paths
        .par_iter()
        .filter_map(|(gene, path)| {
            let ps = path.to_string_lossy();
            let r = load_collect_gene_workspace(
                &ps,
                gene.as_str(),
                obs_names.as_slice(),
                cluster_keys.as_slice(),
                cell_type_labels_arc.as_slice(),
                cluster_obs_arc.as_deref().map(|a| a.as_slice()),
                unique_arcs.as_slice(),
            )
            .map(|ws| collect_interactions_all_cell_types_full_from_workspace(&ws));
            if let Some(ref p) = pb {
                p.inc(1);
            }
            match r {
                Ok(v) => {
                    row_counts.fetch_add(v.len(), std::sync::atomic::Ordering::Relaxed);
                    Some(v)
                }
                Err(e) => {
                    eprintln!("Warning: failed to load {}: {:#}", path.display(), e);
                    None
                }
            }
        })
        .collect();

    if let Some(p) = &pb {
        p.finish_with_message("Done collecting interactions");
    }

    let total_rows = row_counts.load(std::sync::atomic::Ordering::Relaxed);
    let mut merged = Vec::with_capacity(total_rows);
    for v in results {
        merged.extend(v);
    }
    Ok(merged)
}

/// Parallel scan of every `*_betadata.feather` under `dir` with all aggregation columns (Rayon).
///
/// When `cluster_obs` is set (one label per obs cell, e.g. `adata.obs['cluster']`), interactions
/// are collected independently within each distinct cluster value; output rows include `cluster`.
pub fn betadata_collect_interactions_all_cell_types_full(
    dir: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    max_genes: Option<usize>,
    cluster_obs: Option<&[String]>,
) -> Result<Vec<CollectedInteractionRowFull>> {
    ensure_obs_slices_same_len(
        obs_names.len(),
        cluster_keys.len(),
        cell_type_labels.len(),
        cluster_obs.map(|s| s.len()),
    )?;

    let paths = list_betadata_feather_gene_paths(Path::new(dir), max_genes)?;

    let mut merged = betadata_collect_interactions_all_cell_types_full_paths(
        dir,
        &paths,
        obs_names,
        cluster_keys,
        cell_type_labels,
        cluster_obs,
    )?;
    sort_collected_interaction_rows_full(&mut merged);
    Ok(merged)
}

/// Parallel scan of every `*_betadata.feather` under `dir` (Rayon). Corrupt files are skipped with a warning.
///
/// Each feather is read exactly once; per-cell-type aggregation uses precomputed row-index lists.
/// Progress is shown via an `indicatif` bar with ETA and throughput when stderr is a TTY.
pub fn betadata_collect_interactions_all_cell_types(
    dir: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_type_labels: &[String],
    mode: BetadataCollectAggregate,
    max_genes: Option<usize>,
) -> Result<Vec<CollectedInteractionRow>> {
    anyhow::ensure!(
        obs_names.len() == cluster_keys.len(),
        "obs_names len {} != cluster_keys len {}",
        obs_names.len(),
        cluster_keys.len()
    );
    anyhow::ensure!(
        obs_names.len() == cell_type_labels.len(),
        "obs_names len {} != cell_type_labels len {}",
        obs_names.len(),
        cell_type_labels.len()
    );

    let unique_cell_types = unique_sorted_cell_types(cell_type_labels);
    anyhow::ensure!(
        !unique_cell_types.is_empty(),
        "no distinct cell types in labels — check obs annotation column"
    );
    let unique_arcs: Vec<Arc<str>> = unique_cell_types.iter().map(|s| Arc::from(s.as_str())).collect();
    let unique_arcs = Arc::new(unique_arcs);
    let cell_type_labels_arc = Arc::new(cell_type_labels.to_vec());

    let paths = list_betadata_feather_gene_paths(Path::new(dir), max_genes)?;

    let n_total = paths.len();
    let pb = if std::io::stderr().is_terminal() && n_total > 0 {
        let bar = indicatif::ProgressBar::new(n_total as u64);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {per_sec} eta {eta} collect-interactions",
                )?
                .progress_chars("#>-"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(200));
        Some(Arc::new(bar))
    } else {
        eprintln!("Scanning {} betadata feathers…", n_total);
        None
    };

    let row_counts: Arc<std::sync::atomic::AtomicUsize> =
        Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let results: Vec<Vec<CollectedInteractionRow>> = paths
        .par_iter()
        .filter_map(|(gene, path)| {
            let ps = path.to_string_lossy();
            let r = load_collect_gene_workspace_all_ct(
                &ps,
                gene.as_str(),
                obs_names,
                cluster_keys,
                cell_type_labels_arc.as_slice(),
                unique_arcs.as_slice(),
            )
            .map(|ws| collect_interactions_all_cell_types_from_workspace(&ws, mode));
            if let Some(ref p) = pb {
                p.inc(1);
            }
            match r {
                Ok(v) => {
                    row_counts.fetch_add(v.len(), std::sync::atomic::Ordering::Relaxed);
                    Some(v)
                }
                Err(e) => {
                    eprintln!("Warning: failed to load {}: {:#}", path.display(), e);
                    None
                }
            }
        })
        .collect();

    if let Some(p) = &pb {
        p.finish_with_message("Done collecting interactions");
    }

    let total_rows = row_counts.load(std::sync::atomic::Ordering::Relaxed);
    let mut merged = Vec::with_capacity(total_rows);
    for v in results {
        merged.extend(v);
    }
    merged.par_sort_unstable_by(|a, b| {
        b.beta
            .abs()
            .partial_cmp(&a.beta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cell_type.cmp(&b.cell_type))
            .then_with(|| a.target_gene.cmp(&b.target_gene))
            .then_with(|| a.interaction.cmp(&b.interaction))
    });
    Ok(merged)
}

fn optional_f64_series(name: &str, values: &[Option<f64>]) -> Column {
    let v: Vec<Option<f64>> = values
        .iter()
        .map(|x| x.filter(|v| v.is_finite()))
        .collect();
    Series::new(name.into(), v).into()
}

/// Write [`CollectedInteractionRowFull`] as Feather-compatible Arrow IPC (LZ4).
pub fn write_collected_interactions_full_feather(
    path: &str,
    rows: &[CollectedInteractionRowFull],
) -> Result<()> {
    let interaction: Vec<String> = rows.iter().map(|r| r.interaction.clone()).collect();
    let target_gene: Vec<String> = rows.iter().map(|r| r.target_gene.clone()).collect();
    let interaction_type: Vec<String> = rows.iter().map(|r| r.interaction_type.clone()).collect();
    let cell_type: Vec<String> = rows.iter().map(|r| r.cell_type.clone()).collect();
    let cluster: Vec<Option<String>> = rows.iter().map(|r| r.cluster.clone()).collect();
    let mean: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.mean).collect();
    let min: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.min).collect();
    let max: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.max).collect();
    let sum: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.sum).collect();
    let positive: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.positive).collect();
    let negative: Vec<Option<f64>> = rows.iter().map(|r| r.aggregates.negative).collect();

    let mut columns: Vec<Column> = vec![
        Series::new("interaction".into(), interaction).into(),
        Series::new("target_gene".into(), target_gene).into(),
        Series::new("interaction_type".into(), interaction_type).into(),
        Series::new("cell_type".into(), cell_type).into(),
        optional_f64_series("mean", &mean),
        optional_f64_series("min", &min),
        optional_f64_series("max", &max),
        optional_f64_series("sum", &sum),
        optional_f64_series("positive", &positive),
        optional_f64_series("negative", &negative),
    ];
    if rows.iter().any(|r| r.cluster.is_some()) {
        columns.insert(4, Series::new("cluster".into(), cluster).into());
    }

    let mut df = DataFrame::new(columns)?;

    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    let mut w = IpcWriter::new(f).with_compression(Some(IpcCompression::LZ4));
    w.finish(&mut df).context("write IPC / feather")?;
    Ok(())
}

/// Write [`CollectedInteractionRow`] as Feather-compatible Arrow IPC (LZ4).
pub fn write_collected_interactions_feather(path: &str, rows: &[CollectedInteractionRow]) -> Result<()> {
    let interaction: Vec<String> = rows.iter().map(|r| r.interaction.clone()).collect();
    let target_gene: Vec<String> = rows.iter().map(|r| r.target_gene.clone()).collect();
    let beta: Vec<f64> = rows.iter().map(|r| r.beta).collect();
    let interaction_type: Vec<String> = rows.iter().map(|r| r.interaction_type.clone()).collect();
    let cell_type: Vec<String> = rows.iter().map(|r| r.cell_type.clone()).collect();

    let mut df = DataFrame::new(vec![
        Series::new("interaction".into(), interaction).into(),
        Series::new("target_gene".into(), target_gene).into(),
        Series::new("beta".into(), beta).into(),
        Series::new("interaction_type".into(), interaction_type).into(),
        Series::new("cell_type".into(), cell_type).into(),
    ])?;

    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    let mut w = IpcWriter::new(f).with_compression(Some(IpcCompression::LZ4));
    w.finish(&mut df).context("write IPC / feather")?;
    Ok(())
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

    let ws = load_collect_gene_mask_workspace(
        path,
        target_gene,
        obs_names,
        cluster_keys,
        cell_include_mask,
        None,
    )?;
    Ok(collect_interactions_mask_from_workspace(&ws, mode))
}

/// Like [`betadata_collect_interactions_one_gene`], but emits mean/min/max/sum/positive/negative columns.
pub fn betadata_collect_interactions_one_gene_full(
    path: &str,
    target_gene: &str,
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
) -> Result<Vec<CollectedInteractionFull>> {
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

    let ws = load_collect_gene_mask_workspace(
        path,
        target_gene,
        obs_names,
        cluster_keys,
        cell_include_mask,
        None,
    )?;
    Ok(collect_interactions_mask_full_from_workspace(&ws))
}

fn betadata_collect_interactions_parallel_full_impl(
    dir: &str,
    genes: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
    cluster_obs: Option<&[String]>,
) -> Result<Vec<CollectedInteractionFull>> {
    let dir_path = PathBuf::from(dir);
    let row_counts: Arc<std::sync::atomic::AtomicUsize> =
        Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let chunks: Vec<Result<Vec<CollectedInteractionFull>>> = genes
        .par_iter()
        .map(|gene| {
            let path = dir_path.join(format!("{}_betadata.feather", gene));
            if !path.is_file() {
                return Ok(Vec::new());
            }
            let ps = path.to_string_lossy();
            let ws = load_collect_gene_mask_workspace(
                &ps,
                gene.as_str(),
                obs_names,
                cluster_keys,
                cell_include_mask,
                cluster_obs,
            )?;
            let rows = collect_interactions_mask_full_from_workspace(&ws);
            row_counts.fetch_add(rows.len(), std::sync::atomic::Ordering::Relaxed);
            Ok(rows)
        })
        .collect();

    let total_rows = row_counts.load(std::sync::atomic::Ordering::Relaxed);
    let mut merged = Vec::with_capacity(total_rows);
    for chunk in chunks {
        merged.extend(chunk?);
    }
    Ok(merged)
}

fn sort_collected_interaction_full(merged: &mut [CollectedInteractionFull]) {
    merged.par_sort_unstable_by(|a, b| {
        aggregate_sort_key(&b.aggregates)
            .partial_cmp(&aggregate_sort_key(&a.aggregates))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cluster.cmp(&b.cluster))
            .then_with(|| a.gene.cmp(&b.gene))
            .then_with(|| a.interaction.cmp(&b.interaction))
    });
}

/// Parallel scan of `genes.len()` feather files (Rayon) with all aggregation columns.
///
/// When `cluster_obs` is set, collects independently per distinct obs cluster label (mask applied
/// within each partition). Output rows include `cluster` when partitioned.
pub fn betadata_collect_interactions_parallel_full(
    dir: &str,
    genes: &[String],
    obs_names: &[String],
    cluster_keys: &[String],
    cell_include_mask: &[bool],
    cluster_obs: Option<&[String]>,
) -> Result<Vec<CollectedInteractionFull>> {
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
    if let Some(labels) = cluster_obs {
        anyhow::ensure!(
            obs_names.len() == labels.len(),
            "obs_names len {} != cluster_obs len {}",
            obs_names.len(),
            labels.len()
        );
    }

    let mut merged = betadata_collect_interactions_parallel_full_impl(
        dir,
        genes,
        obs_names,
        cluster_keys,
        cell_include_mask,
        cluster_obs,
    )?;
    sort_collected_interaction_full(&mut merged);
    Ok(merged)
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
    let row_counts: Arc<std::sync::atomic::AtomicUsize> =
        Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let chunks: Vec<Result<Vec<CollectedInteraction>>> = genes
        .par_iter()
        .map(|gene| {
            let path = dir_path.join(format!("{}_betadata.feather", gene));
            if !path.is_file() {
                return Ok(Vec::new());
            }
            let ps = path.to_string_lossy();
            let ws = load_collect_gene_mask_workspace(
                &ps,
                gene.as_str(),
                obs_names,
                cluster_keys,
                cell_include_mask,
                None,
            )?;
            let rows = collect_interactions_mask_from_workspace(&ws, mode);
            row_counts.fetch_add(rows.len(), std::sync::atomic::Ordering::Relaxed);
            Ok(rows)
        })
        .collect();

    let total_rows = row_counts.load(std::sync::atomic::Ordering::Relaxed);
    let mut merged = Vec::with_capacity(total_rows);
    for chunk in chunks {
        merged.extend(chunk?);
    }
    merged.par_sort_unstable_by(|a, b| {
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
    let (mapping, _) =
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

#[cfg(test)]
mod collect_interactions_all_cell_types_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn one_gene_cluster_keyed_mean_per_cell_type() {
        let dir = std::env::temp_dir().join(format!(
            "betadata_collect_all_ct_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("TG_betadata.feather");
        let cols = vec!["beta0".into(), "beta_MOD".into()];
        let m = array![[0.0f64, 10.0], [0.0, 30.0]];
        write_betadata_feather(
            path.to_str().unwrap(),
            "Cluster",
            &["cA".into(), "cB".into()],
            &cols,
            &m,
        )
        .unwrap();

        let obs_names = vec!["o1".into(), "o2".into(), "o3".into(), "o4".into()];
        let cluster_keys = vec!["cA".into(), "cA".into(), "cB".into(), "cB".into()];
        let labels: Vec<String> = vec!["T1".into(), "T1".into(), "T2".into(), "T2".into()];
        let unique = unique_sorted_cell_types(&labels);
        let unique_arcs: Vec<Arc<str>> = unique.iter().map(|s| Arc::from(s.as_str())).collect();

        let rows = betadata_collect_interactions_all_cell_types_one_gene(
            path.to_str().unwrap(),
            "TG",
            &obs_names,
            &cluster_keys,
            &labels,
            &unique_arcs,
            BetadataCollectAggregate::Mean,
        )
        .unwrap();

        assert_eq!(rows.len(), 2);
        let t1: Vec<_> = rows.iter().filter(|r| r.cell_type == "T1").collect();
        assert_eq!(t1.len(), 1);
        assert!((t1[0].beta - 10.0).abs() < 1e-9);
        assert_eq!(t1[0].target_gene, "TG");
        assert_eq!(t1[0].interaction, "beta_MOD");
        assert_eq!(t1[0].interaction_type, "tf");
        let t2: Vec<_> = rows.iter().filter(|r| r.cell_type == "T2").collect();
        assert_eq!(t2.len(), 1);
        assert!((t2[0].beta - 30.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn full_collect_per_cluster_obs_partition() {
        let dir = std::env::temp_dir().join(format!(
            "betadata_collect_cluster_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("TG_betadata.feather");
        let cols = vec!["beta0".into(), "beta_MOD".into()];
        let m = array![[0.0f64, 10.0], [0.0, 30.0]];
        write_betadata_feather(
            path.to_str().unwrap(),
            "Cluster",
            &["cA".into(), "cB".into()],
            &cols,
            &m,
        )
        .unwrap();

        let obs_names = vec!["o1".into(), "o2".into(), "o3".into(), "o4".into()];
        let cluster_keys = vec!["cA".into(), "cA".into(), "cB".into(), "cB".into()];
        let labels = vec!["T1".into(), "T1".into(), "T2".into(), "T2".into()];
        let cluster_obs = vec!["C1".into(), "C1".into(), "C2".into(), "C2".into()];

        let rows = betadata_collect_interactions_all_cell_types_full(
            dir.to_str().unwrap(),
            &obs_names,
            &cluster_keys,
            &labels,
            None,
            Some(cluster_obs.as_slice()),
        )
        .unwrap();

        assert!(rows.iter().all(|r| r.cluster.is_some()));
        let c1 = rows
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C1") && r.interaction == "beta_MOD")
            .unwrap();
        assert!((c1.aggregates.mean.unwrap() - 10.0).abs() < 1e-9);
        let c2 = rows
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C2") && r.interaction == "beta_MOD")
            .unwrap();
        assert!((c2.aggregates.mean.unwrap() - 30.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same cell type in two obs clusters must not blend β across clusters.
    #[test]
    fn full_collect_cluster_partition_isolates_obs_clusters() {
        let dir = std::env::temp_dir().join(format!(
            "betadata_collect_cluster_iso_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("TG_betadata.feather");
        let cols = vec!["beta0".into(), "beta_MOD".into()];
        let m = array![[0.0f64, 10.0], [0.0, 30.0]];
        write_betadata_feather(
            path.to_str().unwrap(),
            "Cluster",
            &["cA".into(), "cB".into()],
            &cols,
            &m,
        )
        .unwrap();

        let obs_names = vec!["o1".into(), "o2".into(), "o3".into(), "o4".into()];
        let cluster_keys = vec!["cA".into(), "cA".into(), "cB".into(), "cB".into()];
        let labels = vec!["T1".into(), "T1".into(), "T1".into(), "T1".into()];
        let cluster_obs = vec!["C1".into(), "C1".into(), "C2".into(), "C2".into()];

        let rows = betadata_collect_interactions_all_cell_types_full(
            dir.to_str().unwrap(),
            &obs_names,
            &cluster_keys,
            &labels,
            None,
            Some(cluster_obs.as_slice()),
        )
        .unwrap();

        let t1: Vec<_> = rows
            .iter()
            .filter(|r| r.cell_type == "T1" && r.interaction == "beta_MOD")
            .collect();
        assert_eq!(t1.len(), 2);
        let c1 = t1
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C1"))
            .unwrap();
        let c2 = t1
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C2"))
            .unwrap();
        assert!((c1.aggregates.mean.unwrap() - 10.0).abs() < 1e-9);
        assert!((c2.aggregates.mean.unwrap() - 30.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mask_collect_per_cluster_obs_partition() {
        let dir = std::env::temp_dir().join(format!(
            "betadata_collect_mask_cluster_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("TG_betadata.feather");
        let cols = vec!["beta0".into(), "beta_MOD".into()];
        let m = array![[0.0f64, 10.0], [0.0, 30.0]];
        write_betadata_feather(
            path.to_str().unwrap(),
            "Cluster",
            &["cA".into(), "cB".into()],
            &cols,
            &m,
        )
        .unwrap();

        let obs_names = vec!["o1".into(), "o2".into(), "o3".into(), "o4".into()];
        let cluster_keys = vec!["cA".into(), "cA".into(), "cB".into(), "cB".into()];
        let mask = vec![true, false, false, true];
        let cluster_obs = vec!["C1".into(), "C1".into(), "C2".into(), "C2".into()];

        let rows = betadata_collect_interactions_parallel_full(
            dir.to_str().unwrap(),
            &["TG".into()],
            &obs_names,
            &cluster_keys,
            &mask,
            Some(cluster_obs.as_slice()),
        )
        .unwrap();

        assert_eq!(rows.len(), 2);
        let c1 = rows
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C1"))
            .unwrap();
        let c2 = rows
            .iter()
            .find(|r| r.cluster.as_deref() == Some("C2"))
            .unwrap();
        assert!((c1.aggregates.mean.unwrap() - 10.0).abs() < 1e-9);
        assert!((c2.aggregates.mean.unwrap() - 30.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn materialize_coef_columns_aligns_with_feather_row_indices() {
        let dir = std::env::temp_dir().join(format!(
            "betadata_collect_nulls_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("TG_betadata.feather");
        let cols = vec!["beta0".into(), "beta_MOD".into()];
        let m = array![[0.0f64, 10.0], [0.0, 30.0]];
        write_betadata_feather(
            path.to_str().unwrap(),
            "Cluster",
            &["cA".into(), "cB".into()],
            &cols,
            &m,
        )
        .unwrap();

        let df = read_betadata_feather_df(path.to_str().unwrap()).unwrap();
        let all_names: Vec<String> = df
            .get_columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        let label_idx = betadata_feather_label_column_index(&all_names);
        let coefs = materialize_coef_columns(&df, &all_names, label_idx).unwrap();
        let beta_mod = coefs.iter().find(|c| c.name.as_ref() == "beta_MOD").unwrap();
        assert_eq!(beta_mod.values.len(), df.height());
        assert!((beta_mod.values[0] - 10.0).abs() < 1e-9);
        assert!((beta_mod.values[1] - 30.0).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }
}
