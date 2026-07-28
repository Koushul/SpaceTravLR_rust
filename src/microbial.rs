//! Bacterial secretion sources → host receptor (BR) modulators.
//!
//! External sender loci (colonies / bins) emit signal amounts \(A_{bk}\); receivers
//! get Gaussian-weighted fields \(\widetilde S\), then features \(x=\widetilde S\cdot R\)
//! with Lasso group id **4**.

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// One resolved bact→host pair used as a modulator column `signal$receptor`.
#[derive(Debug, Clone)]
pub struct BrPair {
    pub signal_id: String,
    pub receptor: String,
    pub radius_um: f64,
    pub pair_name: String,
}

/// Preloaded sender table + interaction DB for training workers.
#[derive(Debug, Clone)]
pub struct MicrobialContext {
    pub sender_xy: Array2<f64>,
    pub amounts: Array2<f64>,
    pub signal_names: Vec<String>,
    pub signal_index: HashMap<String, usize>,
    pub br_pairs: Vec<BrPair>,
    pub scale_factor: f64,
    /// Soft known signal ids (for betadata type classification).
    pub known_signals: HashSet<String>,
}

impl MicrobialContext {
    pub fn n_senders(&self) -> usize {
        self.sender_xy.nrows()
    }

    pub fn pair_names(&self) -> Vec<String> {
        self.br_pairs.iter().map(|p| p.pair_name.clone()).collect()
    }

    pub fn pairs_for_target(&self, target: &str) -> Vec<BrPair> {
        self.br_pairs
            .iter()
            .filter(|p| p.receptor != target && p.signal_id != target)
            .cloned()
            .collect()
    }

    pub fn is_bact_pair_name(name: &str, known: &HashSet<String>) -> bool {
        let body = name.strip_prefix("beta_").unwrap_or(name);
        if let Some((sig, _)) = body.split_once('$') {
            return known.contains(sig);
        }
        false
    }
}

#[derive(Debug, Clone)]
pub struct MicrobialConfig {
    pub enabled: bool,
    pub sender_table: Option<String>,
    pub interactions: String,
    pub scale_factor: f64,
    pub radius_um_override: Option<f64>,
    /// Truncate Gaussian at this multiple of radius (speed). Default 3.
    pub dmax_factor: f64,
}

impl Default for MicrobialConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sender_table: None,
            interactions: "data/microbial/bact_host_interactions.v0.csv".into(),
            scale_factor: 1.0,
            radius_um_override: None,
            dmax_factor: 3.0,
        }
    }
}

/// Load interactions CSV and sender parquet; resolve receptors against `var_names`.
pub fn load_microbial_context(
    cfg: &MicrobialConfig,
    var_names: &[String],
    config_dir: Option<&Path>,
) -> Result<Option<Arc<MicrobialContext>>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let sender_path = cfg
        .sender_table
        .as_ref()
        .ok_or_else(|| anyhow!("[microbial].enabled requires sender_table"))?;
    let sender_path = resolve_path(sender_path, config_dir);
    let inter_path = resolve_path(&cfg.interactions, config_dir);

    let (sender_xy, amounts, signal_names) = load_sender_table(&sender_path)?;
    let mut signal_index = HashMap::new();
    for (i, s) in signal_names.iter().enumerate() {
        signal_index.insert(s.clone(), i);
    }

    let var_set: HashSet<String> = var_names.iter().cloned().collect();
    let var_lower: HashMap<String, String> = var_names
        .iter()
        .map(|g| (g.to_ascii_lowercase(), g.clone()))
        .collect();

    let inter = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(inter_path.clone()))?
        .finish()
        .with_context(|| format!("read interactions {}", inter_path.display()))?;

    let sig_col = inter.column("signal_id")?.str()?;
    let rec_col = inter.column("receptor")?.str()?;
    let rad_col = inter.column("default_radius_um")?.f64()?;

    let mut known_signals: HashSet<String> = HashSet::new();
    let mut br_pairs = Vec::new();
    let mut seen_pairs: HashSet<String> = HashSet::new();

    for i in 0..inter.height() {
        let Some(sig) = sig_col.get(i) else { continue };
        let Some(rec_raw) = rec_col.get(i) else { continue };
        known_signals.insert(sig.to_string());
        if !signal_index.contains_key(sig) {
            continue;
        }
        let Some(rec) = var_lower.get(&rec_raw.to_ascii_lowercase()) else {
            continue;
        };
        if !var_set.contains(rec) {
            continue;
        }
        let radius = cfg
            .radius_um_override
            .unwrap_or_else(|| rad_col.get(i).unwrap_or(40.0));
        let pair_name = format!("{sig}${rec}");
        if !seen_pairs.insert(pair_name.clone()) {
            continue;
        }
        br_pairs.push(BrPair {
            signal_id: sig.to_string(),
            receptor: rec.clone(),
            radius_um: radius,
            pair_name,
        });
    }

    if br_pairs.is_empty() {
        bail!(
            "microbial enabled but no BR pairs resolved (senders have {} signals; check receptors in var)",
            signal_names.len()
        );
    }

    Ok(Some(Arc::new(MicrobialContext {
        sender_xy,
        amounts,
        signal_names,
        signal_index,
        br_pairs,
        scale_factor: cfg.scale_factor,
        known_signals,
    })))
}

fn resolve_path(p: &str, config_dir: Option<&Path>) -> std::path::PathBuf {
    let path = crate::config::expand_user_path(p.trim());
    let pb = std::path::PathBuf::from(&path);
    if pb.is_absolute() || pb.exists() {
        return pb;
    }
    if let Some(dir) = config_dir {
        let cand = dir.join(&pb);
        if cand.exists() {
            return cand;
        }
    }
    // try repo-relative from cwd
    pb
}

/// Sender parquet: columns `x`, `y`, plus one float column per signal.
pub fn load_sender_table(path: &Path) -> Result<(Array2<f64>, Array2<f64>, Vec<String>)> {
    let path_s = path.to_string_lossy().into_owned();
    let df = LazyFrame::scan_parquet(
        polars_utils::plpath::PlPath::from_string(path_s.clone()),
        ScanArgsParquet::default(),
    )
    .with_context(|| format!("scan sender parquet {}", path.display()))?
    .collect()
    .with_context(|| format!("read sender parquet {}", path.display()))?;

    let x = col_f64(&df, "x")?;
    let y = col_f64(&df, "y")?;
    anyhow::ensure!(x.len() == y.len(), "x/y length mismatch");
    let n = x.len();
    let mut sender_xy = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        sender_xy[[i, 0]] = x[i];
        sender_xy[[i, 1]] = y[i];
    }

    // Metadata / id columns — never treat as microbial signal amounts.
    let skip: HashSet<&str> = [
        "x",
        "y",
        "sender_id",
        "key",
        "bact_label",
        "dominant_genus_umi",
        "label",
        "cluster",
        "genus",
    ]
    .into_iter()
    .collect();
    let mut signal_names = Vec::new();
    for name in df.get_column_names() {
        let s = name.to_string();
        if skip.contains(s.as_str()) {
            continue;
        }
        // only numeric signal columns
        if df.column(&s).ok().and_then(|c| c.f64().ok()).is_some()
            || df.column(&s).ok().and_then(|c| c.cast(&DataType::Float64).ok()).is_some()
        {
            signal_names.push(s);
        }
    }
    signal_names.sort();
    anyhow::ensure!(!signal_names.is_empty(), "no signal columns in {}", path.display());

    let mut amounts = Array2::<f64>::zeros((n, signal_names.len()));
    for (k, sig) in signal_names.iter().enumerate() {
        let col = df
            .column(sig)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        anyhow::ensure!(col.len() == n, "signal {sig} length mismatch");
        for i in 0..n {
            amounts[[i, k]] = col[i];
        }
    }

    Ok((sender_xy, amounts, signal_names))
}

fn col_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let c = df
        .column(name)?
        .cast(&DataType::Float64)
        .with_context(|| format!("column {name}"))?;
    Ok(c.f64()?.into_no_null_iter().collect())
}

/// Received microbial signals at receiver positions from external senders.
/// Uses hard cutoff at `dmax_factor * radius` per column (channels may share radius groups).
pub fn calculate_received_from_senders(
    receiver_xy: &Array2<f64>,
    sender_xy: &Array2<f64>,
    amounts: &Array2<f64>,
    radius_per_channel: &[f64],
    scale_factor: f64,
    dmax_factor: f64,
) -> Array2<f64> {
    let n_recv = receiver_xy.nrows();
    let n_send = sender_xy.nrows();
    let n_chan = amounts.ncols();
    let mut out = Array2::<f64>::zeros((n_recv, n_chan));
    if n_recv == 0 || n_send == 0 || n_chan == 0 {
        return out;
    }
    assert_eq!(radius_per_channel.len(), n_chan);

    // Group channels by radius to reuse neighbor lists
    let mut radius_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (k, &r) in radius_per_channel.iter().enumerate() {
        let key = r.to_bits();
        radius_groups.entry(key).or_default().push(k);
    }

    for (r_bits, chans) in radius_groups {
        let radius = f64::from_bits(r_bits);
        if !(radius.is_finite() && radius > 0.0) {
            continue;
        }
        let dmax = dmax_factor * radius;
        let dmax2 = dmax * dmax;
        let inv_2r2 = -1.0 / (2.0 * radius * radius);

        let rows: Vec<Vec<f64>> = (0..n_recv)
            .into_par_iter()
            .map(|i| {
                let xi = receiver_xy[[i, 0]];
                let yi = receiver_xy[[i, 1]];
                let mut acc = vec![0.0f64; chans.len()];
                for j in 0..n_send {
                    let dx = xi - sender_xy[[j, 0]];
                    let dy = yi - sender_xy[[j, 1]];
                    let d2 = dx * dx + dy * dy;
                    if d2 > dmax2 {
                        continue;
                    }
                    let w = scale_factor * (d2 * inv_2r2).exp();
                    for (t, &k) in chans.iter().enumerate() {
                        acc[t] += w * amounts[[j, k]];
                    }
                }
                acc
            })
            .collect();

        for (i, acc) in rows.into_iter().enumerate() {
            for (t, &k) in chans.iter().enumerate() {
                out[[i, k]] = acc[t];
            }
        }
    }
    out
}

/// Median-normalize positive columns of a received-signal matrix (in place).
fn median_normalize_positive_columns(received: &mut Array2<f64>) {
    let n_sig = received.ncols();
    for t in 0..n_sig {
        let col = received.column(t);
        let mut pos: Vec<f64> = col.iter().copied().filter(|&v| v > 0.0).collect();
        if !pos.is_empty() {
            pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = pos[pos.len() / 2];
            if med > 1e-12 {
                received.column_mut(t).mapv_inplace(|v| v / med);
            }
        }
    }
}

/// Precompute median-normalized received fields for every signal in `ctx` at receiver sites.
/// Columns align with `ctx.signal_names` / `ctx.signal_index`.
pub fn precompute_all_signal_received(
    ctx: &MicrobialContext,
    receiver_xy: &Array2<f64>,
    dmax_factor: f64,
) -> Array2<f64> {
    let n_sig = ctx.signal_names.len();
    let mut radii = vec![40.0_f64; n_sig];
    for p in &ctx.br_pairs {
        if let Some(&k) = ctx.signal_index.get(&p.signal_id) {
            radii[k] = p.radius_um;
        }
    }
    let mut received = calculate_received_from_senders(
        receiver_xy,
        &ctx.sender_xy,
        &ctx.amounts,
        &radii,
        ctx.scale_factor,
        dmax_factor,
    );
    median_normalize_positive_columns(&mut received);
    received
}

/// Build BR feature matrix (n_cells × n_pairs) = received(S) * receptor_expr.
///
/// When `cached_received` is `Some`, it must be `n_cells × n_signals` aligned with
/// [`MicrobialContext::signal_names`] (from [`precompute_all_signal_received`]); the expensive
/// Gaussian field is skipped.
pub fn build_br_features(
    ctx: &MicrobialContext,
    pairs: &[BrPair],
    receiver_xy: &Array2<f64>,
    receptor_expr: &HashMap<String, Array1<f64>>,
    dmax_factor: f64,
    cached_received: Option<&Array2<f64>>,
) -> Array2<f64> {
    let n = receiver_xy.nrows();
    if pairs.is_empty() {
        return Array2::zeros((n, 0));
    }

    let owned_received;
    let (local_index, received): (HashMap<String, usize>, &Array2<f64>) =
        if let Some(full) = cached_received {
            assert_eq!(full.nrows(), n);
            assert_eq!(full.ncols(), ctx.signal_names.len());
            let mut local_index = HashMap::new();
            for p in pairs {
                if let Some(&k) = ctx.signal_index.get(&p.signal_id) {
                    local_index.entry(p.signal_id.clone()).or_insert(k);
                }
            }
            (local_index, full)
        } else {
            let mut sig_cols: Vec<usize> = Vec::new();
            let mut sig_radii: Vec<f64> = Vec::new();
            let mut local_index: HashMap<String, usize> = HashMap::new();
            for p in pairs {
                if let Some(&k) = ctx.signal_index.get(&p.signal_id) {
                    if !local_index.contains_key(&p.signal_id) {
                        local_index.insert(p.signal_id.clone(), sig_cols.len());
                        sig_cols.push(k);
                        sig_radii.push(p.radius_um);
                    }
                }
            }

            let n_sig = sig_cols.len();
            let mut amounts = Array2::<f64>::zeros((ctx.n_senders(), n_sig));
            for (t, &k) in sig_cols.iter().enumerate() {
                amounts.column_mut(t).assign(&ctx.amounts.column(k));
            }

            let mut received = calculate_received_from_senders(
                receiver_xy,
                &ctx.sender_xy,
                &amounts,
                &sig_radii,
                ctx.scale_factor,
                dmax_factor,
            );
            median_normalize_positive_columns(&mut received);
            owned_received = received;
            (local_index, &owned_received)
        };

    let mut out = Array2::<f64>::zeros((n, pairs.len()));
    for (j, p) in pairs.iter().enumerate() {
        let Some(&t) = local_index.get(&p.signal_id) else {
            continue;
        };
        let Some(rec) = receptor_expr.get(&p.receptor) else {
            continue;
        };
        let s = received.column(t);
        for i in 0..n {
            out[[i, j]] = s[i] * rec[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn external_senders_peak_near_source() {
        let sender_xy = array![[0.0, 0.0]];
        let amounts = array![[10.0]];
        let receivers = array![[0.0, 0.0], [50.0, 0.0], [200.0, 0.0]];
        let got = calculate_received_from_senders(
            &receivers,
            &sender_xy,
            &amounts,
            &[40.0],
            1.0,
            3.0,
        );
        assert!(got[[0, 0]] > got[[1, 0]]);
        assert!(got[[1, 0]] > got[[2, 0]]);
    }

    #[test]
    fn bact_pair_classifier() {
        let mut known = HashSet::new();
        known.insert("Lps".into());
        assert!(MicrobialContext::is_bact_pair_name("beta_Lps$Tlr4", &known));
        assert!(!MicrobialContext::is_bact_pair_name("beta_Tgfa$Erbb2", &known));
        assert!(!MicrobialContext::is_bact_pair_name("beta_Stat3", &known));
    }

    #[test]
    fn cached_received_matches_uncached_br_features() {
        let ctx = MicrobialContext {
            sender_xy: array![[0.0, 0.0]],
            amounts: array![[10.0]],
            signal_names: vec!["Lps".into()],
            signal_index: {
                let mut m = HashMap::new();
                m.insert("Lps".into(), 0);
                m
            },
            br_pairs: vec![BrPair {
                signal_id: "Lps".into(),
                receptor: "Tlr4".into(),
                radius_um: 40.0,
                pair_name: "Lps$Tlr4".into(),
            }],
            scale_factor: 1.0,
            known_signals: HashSet::from(["Lps".into()]),
        };
        let receivers = array![[0.0, 0.0], [50.0, 0.0]];
        let mut rec_map = HashMap::new();
        rec_map.insert("Tlr4".into(), array![1.0, 2.0]);
        let pairs = ctx.br_pairs.clone();
        let uncached = build_br_features(&ctx, &pairs, &receivers, &rec_map, 3.0, None);
        let cached = precompute_all_signal_received(&ctx, &receivers, 3.0);
        let with_cache = build_br_features(&ctx, &pairs, &receivers, &rec_map, 3.0, Some(&cached));
        assert_eq!(uncached.nrows(), with_cache.nrows());
        assert_eq!(uncached.ncols(), with_cache.ncols());
        for i in 0..uncached.nrows() {
            for j in 0..uncached.ncols() {
                assert!((uncached[[i, j]] - with_cache[[i, j]]).abs() < 1e-9);
            }
        }
    }
}
