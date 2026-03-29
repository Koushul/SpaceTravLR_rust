use std::path::{Path, PathBuf};

pub fn list_betadata_target_genes(dir: &str) -> anyhow::Result<Vec<String>> {
    let dir_path = Path::new(dir);
    anyhow::ensure!(dir_path.is_dir(), "betadata directory does not exist: {}", dir);
    let mut genes = Vec::new();
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(s) = name.to_str() else {
            continue;
        };
        if let Some(stem) = s.strip_suffix("_betadata.feather") {
            genes.push(stem.to_string());
        }
    }
    genes.sort();
    Ok(genes)
}

pub fn betadata_feather_path(dir: &str, gene: &str) -> PathBuf {
    Path::new(dir).join(format!("{}_betadata.feather", gene))
}
