use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use polars::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

/// Named matrix for gene expression or weighted ligand data.
/// Wraps a dense 2D array with gene-name → column-index lookup.
pub struct GeneMatrix {
    pub data: Array2<f64>,
    pub col_names: Vec<String>,
    col_map: HashMap<String, usize>,
}

impl GeneMatrix {
    pub fn new(data: Array2<f64>, col_names: Vec<String>) -> Self {
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

    pub fn col(&self, name: &str) -> Option<ArrayView1<'_, f64>> {
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

    pub intercepts: Array1<f64>,
    pub tf_betas: Array2<f64>,
    pub lr_betas: Array2<f64>,
    pub tfl_betas: Array2<f64>,

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
}

/// Write betadata as Feather-compatible Arrow IPC (LZ4). `id_col` is `Cluster` (seed-only) or `CellID` (per-cell CNN).
pub fn write_betadata_feather(
    path: &str,
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
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    let mut w = IpcWriter::new(f).with_compression(Some(IpcCompression::LZ4));
    w.finish(&mut df).with_context(|| format!("write IPC {}", path))?;
    Ok(())
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

        let label_col_idx = all_col_names.iter().position(|c| {
            c == "Cluster"
                || c == "CellID"
                || c.starts_with("__index")
                || c == "cell_id"
                || c == "index"
                || c == "obs_names"
        });

        let (row_labels, data_col_names) = if let Some(idx) = label_col_idx {
            let label_name = &all_col_names[idx];
            let label_casted = df.column(label_name)?.cast(&DataType::String)?;
            let labels: Vec<String> = label_casted
                .str()?
                .into_no_null_iter()
                .map(|s| s.to_string())
                .collect();
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
        let mut raw = Array2::zeros((n_rows, n_cols));

        for (j, col_name) in data_col_names.iter().enumerate() {
            let casted = df.column(col_name)?.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            for (i, val) in ca.into_iter().enumerate() {
                raw[[i, j]] = val.unwrap_or(0.0);
            }
        }

        let gene_name = Self::extract_gene_name(path);
        Self::from_raw(gene_name, row_labels, data_col_names, raw)
    }

    /// Construct directly from typed arrays (useful for tests and programmatic construction).
    /// Starts with an identity cell→beta mapping (n_cells == n_beta_rows).
    pub fn from_parts(
        gene_name: String,
        row_labels: Vec<String>,
        intercepts: Array1<f64>,
        tf_betas: Array2<f64>,
        tfs: Vec<String>,
        lr_betas: Array2<f64>,
        ligands: Vec<String>,
        receptors: Vec<String>,
        tfl_betas: Array2<f64>,
        tfl_ligands: Vec<String>,
        tfl_regulators: Vec<String>,
    ) -> Self {
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
        }
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
    pub fn compute_cell_mapping(
        row_labels: &[String],
        obs_names: &[String],
        clusters: &[usize],
    ) -> Vec<usize> {
        let row_map: HashMap<&str, usize> = row_labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        // Try cluster-based mapping (seed-only: row_labels are "0", "1", …)
        let cluster_mapping: Option<Vec<usize>> = clusters
            .iter()
            .map(|c| {
                let key = c.to_string();
                row_map.get(key.as_str()).copied()
            })
            .collect();

        if let Some(mapping) = cluster_mapping {
            return mapping;
        }

        // Fall back to obs_name matching (CNN: row_labels are cell IDs)
        obs_names
            .iter()
            .map(|name| row_map.get(name.as_str()).copied().unwrap_or(0))
            .collect()
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
        data: Array2<f64>,
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
        })
    }

    fn extract_cols(data: &Array2<f64>, indices: &[usize], n_rows: usize) -> Array2<f64> {
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
        scale_factor: f64,
        beta_cap: Option<f64>,
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
        let mut result = vec![0.0f64; n * n_out];

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

                if gex > 0.0 {
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
    /// then expand every frame to cell level using the given obs_names + clusters.
    pub fn from_directory(
        dir: &str,
        obs_names: &[String],
        clusters: &[usize],
        gene2index: Option<&HashMap<String, usize>>,
    ) -> Result<Self> {
        let dir_path = Path::new(dir);
        anyhow::ensure!(dir_path.exists(), "Directory {} does not exist", dir);

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

        let pb = indicatif::ProgressBar::new(paths.len() as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Reading betadata")?
                .progress_chars("#>-"),
        );

        let mut frames: Vec<BetaFrame> = paths
            .par_iter()
            .filter_map(|path| {
                let result = BetaFrame::from_path(path);
                pb.inc(1);
                match result {
                    Ok(frame) => Some(frame),
                    Err(e) => {
                        eprintln!("Warning: failed to load {}: {}", path, e);
                        None
                    }
                }
            })
            .collect();

        pb.finish_with_message("Done loading betadata");

        // Expand all frames to cell level. Compute the mapping once per unique
        // set of row_labels and share via Arc to avoid duplicating per gene.
        let cell_labels = Arc::new(obs_names.to_vec());
        let mut last_row_labels: Option<Vec<String>> = None;
        let mut last_mapping: Option<Arc<Vec<usize>>> = None;

        let mut data = HashMap::new();
        let mut ligands_set = HashSet::new();
        let mut receptors_set = HashSet::new();
        let mut tfl_ligands_set = HashSet::new();
        let mut tfs_set = HashSet::new();

        for mut frame in frames.drain(..) {
            ligands_set.extend(frame.ligands.iter().cloned());
            receptors_set.extend(frame.receptors.iter().cloned());
            tfl_ligands_set.extend(frame.tfl_ligands.iter().cloned());
            tfs_set.extend(frame.tfs.iter().cloned());

            // Reuse the mapping Arc when row_labels haven't changed
            let mapping = if last_row_labels.as_ref() == Some(&frame.row_labels) {
                last_mapping.as_ref().unwrap().clone()
            } else {
                let m = Arc::new(BetaFrame::compute_cell_mapping(
                    &frame.row_labels,
                    obs_names,
                    clusters,
                ));
                last_row_labels = Some(frame.row_labels.clone());
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
