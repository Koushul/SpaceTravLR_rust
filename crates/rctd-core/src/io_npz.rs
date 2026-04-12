use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{Array1, Array2};
use ndarray_npy::{NpzReader, ReadNpzError};

use crate::likelihood_tables::build_x_vals;

pub type QMatrixMap = HashMap<String, Array2<f64>>;

#[derive(Debug, thiserror::Error)]
pub enum IoNpzError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Npz(#[from] ReadNpzError),
}

pub fn load_q_matrices_npz(path: &Path) -> Result<(QMatrixMap, Array1<f64>), IoNpzError> {
    let f = File::open(path)?;
    let mut npz = NpzReader::new(f)?;
    let names = npz.names()?;
    let mut map = HashMap::new();
    for name in names {
        if name == "X_vals" {
            continue;
        }
        let arr: Array2<f64> = npz.by_name(&name)?;
        map.insert(name, arr);
    }
    let x_vals: Array1<f64> = match npz.by_name("X_vals") {
        Ok(x) => x,
        Err(_) => build_x_vals(),
    };
    Ok((map, x_vals))
}
