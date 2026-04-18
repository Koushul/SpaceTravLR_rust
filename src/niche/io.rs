//! Write per-cell niche labels + embeddings as Feather and CSV.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::*;

/// Cells × `{cell_id, niche, embedding_*}`.
pub struct NicheLabels<'a> {
    pub obs_names: &'a [String],
    pub labels: &'a [usize],
    pub embeddings: &'a Array2<f32>,
}

impl<'a> NicheLabels<'a> {
    pub fn into_dataframe(&self) -> Result<DataFrame> {
        let n = self.obs_names.len();
        anyhow::ensure!(self.labels.len() == n, "labels length mismatch");
        anyhow::ensure!(
            self.embeddings.nrows() == n,
            "embeddings rows mismatch"
        );
        let cell_id = Series::new("CellID".into(), self.obs_names.to_vec());
        let niche = Series::new(
            "niche".into(),
            self.labels.iter().map(|&l| l as i64).collect::<Vec<i64>>(),
        );
        let mut series = vec![cell_id, niche];
        let dim = self.embeddings.ncols();
        for d in 0..dim {
            let col: Vec<f32> = (0..n).map(|i| self.embeddings[[i, d]]).collect();
            series.push(Series::new(format!("z{:02}", d).into(), col));
        }
        let columns: Vec<Column> = series.into_iter().map(|s| s.into_column()).collect();
        let df = DataFrame::new(columns)?;
        Ok(df)
    }
}

pub fn write_niche_labels_feather(path: &Path, labels: NicheLabels<'_>) -> Result<()> {
    let mut df = labels.into_dataframe()?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    IpcWriter::new(file)
        .with_compression(Some(IpcCompression::LZ4))
        .finish(&mut df)
        .context("write IPC feather")?;
    Ok(())
}

pub fn write_niche_labels_csv(path: &Path, labels: NicheLabels<'_>) -> Result<()> {
    let n = labels.obs_names.len();
    let dim = labels.embeddings.ncols();
    let mut out = File::create(path).with_context(|| format!("create {}", path.display()))?;
    write!(out, "CellID,niche")?;
    for d in 0..dim {
        write!(out, ",z{:02}", d)?;
    }
    writeln!(out)?;
    for i in 0..n {
        write!(out, "{},{}", labels.obs_names[i], labels.labels[i])?;
        for d in 0..dim {
            write!(out, ",{}", labels.embeddings[[i, d]])?;
        }
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_csv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("niche.csv");
        let obs = vec!["cell_a".to_string(), "cell_b".to_string()];
        let labels = vec![0usize, 1];
        let emb = Array2::<f32>::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        write_niche_labels_csv(
            &path,
            NicheLabels {
                obs_names: &obs,
                labels: &labels,
                embeddings: &emb,
            },
        )
        .unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("CellID,niche,z00,z01,z02"));
        assert!(s.contains("cell_a,0,0.1,0.2,0.3"));
        assert!(s.contains("cell_b,1,0.4,0.5,0.6"));
    }
}
