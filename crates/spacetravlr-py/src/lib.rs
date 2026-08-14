//! PyO3 bindings for [`spacetravlr_transition`] (velocyto / Cartography UMAP quiver).

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spacetravlr_transition::{
    NullSubtractMode, TransitionGrid, TransitionUmapParams, col_delta_cor,
    col_delta_cor_partial, compute_umap_transition_grid, pearson_vel_vs_expr_delta,
    round_delta_inplace, umap_grid_axes, umap_knn_indices,
};

fn arr2_from_py(a: PyReadonlyArray2<'_, f64>) -> PyResult<Array2<f64>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2-D array"));
    }
    let v = a.as_slice()?.to_vec();
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn umap_from_py(a: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let shape = a.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err("umap must be (n_cells, 2)"));
    }
    let s = a.as_slice()?;
    let mut out = Vec::with_capacity(shape[0]);
    for i in 0..shape[0] {
        out.push([s[i * 2], s[i * 2 + 1]]);
    }
    Ok(out)
}

fn params_from_kwargs(
    n_neighbors: usize,
    temperature: f64,
    remove_null: bool,
    unit_directions: bool,
    grid_scale: f64,
    vector_scale: f64,
    delta_rescale: f64,
    magnitude_threshold: f64,
    use_full_graph: bool,
    full_graph_max_cells: usize,
    null_subtract_mode: &str,
) -> PyResult<TransitionUmapParams> {
    let mode = match null_subtract_mode {
        "clip_renorm" | "clip" => NullSubtractMode::ClipRenorm,
        "raw" | "python" | "spaceoracle" => NullSubtractMode::Raw,
        other => {
            return Err(PyValueError::new_err(format!(
                "null_subtract_mode must be 'clip_renorm' or 'raw', got {other:?}"
            )));
        }
    };
    Ok(TransitionUmapParams {
        n_neighbors,
        temperature,
        remove_null,
        unit_directions,
        grid_scale,
        vector_scale,
        delta_rescale,
        magnitude_threshold,
        use_full_graph,
        full_graph_max_cells,
        null_subtract_mode: mode,
    })
}

fn grid_to_dict<'py>(py: Python<'py>, g: &TransitionGrid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let nx = g.grid_x.len();
    let ny = g.grid_y.len();
    d.set_item("grid_x", PyArray1::from_slice(py, &g.grid_x))?;
    d.set_item("grid_y", PyArray1::from_slice(py, &g.grid_y))?;
    let mut u = vec![0.0_f64; nx * ny];
    let mut v = vec![0.0_f64; nx * ny];
    for (i, w) in g.vectors.iter().enumerate() {
        u[i] = w[0];
        v[i] = w[1];
    }
    d.set_item("u", PyArray1::from_vec(py, u))?;
    d.set_item("v", PyArray1::from_vec(py, v))?;
    d.set_item("nx", nx)?;
    d.set_item("ny", ny)?;
    let n = g.cell_vectors.len();
    let mut cu = vec![0.0_f64; n];
    let mut cv = vec![0.0_f64; n];
    for (i, w) in g.cell_vectors.iter().enumerate() {
        cu[i] = w[0];
        cv[i] = w[1];
    }
    d.set_item("cell_u", PyArray1::from_vec(py, cu))?;
    d.set_item("cell_v", PyArray1::from_vec(py, cv))?;
    // meshgrid points in cartography order (ix outer, iy inner)
    let mut gx = Vec::with_capacity(nx * ny);
    let mut gy = Vec::with_capacity(nx * ny);
    for ix in 0..nx {
        for iy in 0..ny {
            gx.push(g.grid_x[ix]);
            gy.push(g.grid_y[iy]);
        }
    }
    d.set_item("grid_points_x", PyArray1::from_vec(py, gx))?;
    d.set_item("grid_points_y", PyArray1::from_vec(py, gy))?;
    Ok(d)
}

/// Pearson corr(vel_i, expr_j - expr_i); NaN/zero-var → 1.0 (velocyto).
#[pyfunction]
fn pearson_velocity_vs_expr_delta(
    expr_i: Vec<f64>,
    expr_j: Vec<f64>,
    vel_i: Vec<f64>,
) -> PyResult<f64> {
    if expr_i.len() != expr_j.len() || expr_i.len() != vel_i.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    Ok(pearson_vel_vs_expr_delta(&expr_i, &expr_j, &vel_i))
}

/// Dense `colDeltaCor`: out[i,j] = corr(delta[i], expr[j]-expr[i]).
#[pyfunction]
fn col_delta_cor_py<'py>(
    py: Python<'py>,
    expr: PyReadonlyArray2<'_, f64>,
    delta: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let e = arr2_from_py(expr)?;
    let d = arr2_from_py(delta)?;
    if e.dim() != d.dim() {
        return Err(PyValueError::new_err("expr/delta shape mismatch"));
    }
    let out = col_delta_cor(&e, &d);
    Ok(PyArray2::from_vec2(py, &out.outer_iter().map(|r| r.to_vec()).collect::<Vec<_>>())?)
}

/// Partial `colDeltaCor` for neighbor lists (list of list of int indices).
#[pyfunction]
fn col_delta_cor_partial_py<'py>(
    py: Python<'py>,
    expr: PyReadonlyArray2<'_, f64>,
    delta: PyReadonlyArray2<'_, f64>,
    neighbors: Vec<Vec<usize>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let e = arr2_from_py(expr)?;
    let d = arr2_from_py(delta)?;
    if e.nrows() != neighbors.len() {
        return Err(PyValueError::new_err("neighbors length must equal n_cells"));
    }
    let out = col_delta_cor_partial(&e, &d, &neighbors);
    Ok(PyArray2::from_vec2(py, &out.outer_iter().map(|r| r.to_vec()).collect::<Vec<_>>())?)
}

/// UMAP KNN indices (self excluded), shape (n, k) jagged as list of lists.
#[pyfunction]
fn umap_knn(umap: PyReadonlyArray2<'_, f64>, k: usize) -> PyResult<Vec<Vec<usize>>> {
    let u = umap_from_py(umap)?;
    Ok(umap_knn_indices(&u, k))
}

/// Adaptive grid axes matching `get_grid_layout` + cartography scale.
#[pyfunction]
fn umap_grid_axes_py<'py>(
    py: Python<'py>,
    umap: PyReadonlyArray2<'_, f64>,
    grid_scale: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let u = umap_from_py(umap)?;
    let (gx, gy) = umap_grid_axes(&u, grid_scale);
    Ok((PyArray1::from_vec(py, gx), PyArray1::from_vec(py, gy)))
}

/// Full Cartography transition vector field on UMAP.
///
/// `expr` / `delta`: (n_cells, n_genes). `umap`: (n_cells, 2).
/// Returns a dict with grid_x/y, u/v, cell_u/v, grid_points_*.
#[pyfunction]
#[pyo3(signature = (
    expr,
    delta,
    umap,
    n_neighbors=150,
    temperature=0.05,
    remove_null=true,
    unit_directions=false,
    grid_scale=1.0,
    vector_scale=0.85,
    delta_rescale=1.0,
    magnitude_threshold=0.0,
    use_full_graph=false,
    full_graph_max_cells=4096,
    null_subtract_mode="raw",
    round_delta=true,
))]
fn compute_transition_grid<'py>(
    py: Python<'py>,
    expr: PyReadonlyArray2<'_, f64>,
    delta: PyReadonlyArray2<'_, f64>,
    umap: PyReadonlyArray2<'_, f64>,
    n_neighbors: usize,
    temperature: f64,
    remove_null: bool,
    unit_directions: bool,
    grid_scale: f64,
    vector_scale: f64,
    delta_rescale: f64,
    magnitude_threshold: f64,
    use_full_graph: bool,
    full_graph_max_cells: usize,
    null_subtract_mode: &str,
    round_delta: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let e = arr2_from_py(expr)?;
    let d = arr2_from_py(delta)?;
    let u = umap_from_py(umap)?;
    if e.nrows() != d.nrows() || e.nrows() != u.len() {
        return Err(PyValueError::new_err("n_cells mismatch across expr/delta/umap"));
    }
    if e.ncols() != d.ncols() {
        return Err(PyValueError::new_err("n_genes mismatch"));
    }
    let _ = round_delta;
    let params = params_from_kwargs(
        n_neighbors,
        temperature,
        remove_null,
        unit_directions,
        grid_scale,
        vector_scale,
        delta_rescale,
        magnitude_threshold,
        use_full_graph,
        full_graph_max_cells,
        null_subtract_mode,
    )?;
    let g = py.allow_threads(|| compute_umap_transition_grid(&e, &d, &u, &params));
    grid_to_dict(py, &g)
}

/// Round delta in-place like cartography `.round(3)`.
#[pyfunction]
fn round_delta_py(mut delta: numpy::PyReadwriteArray2<'_, f64>, decimals: i32) -> PyResult<()> {
    let shape = delta.shape().to_vec();
    let slice = delta.as_slice_mut()?;
    let mut arr = Array2::from_shape_vec((shape[0], shape[1]), slice.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    round_delta_inplace(&mut arr, decimals);
    for (dst, src) in slice.iter_mut().zip(arr.iter()) {
        *dst = *src;
    }
    Ok(())
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pearson_velocity_vs_expr_delta, m)?)?;
    m.add_function(wrap_pyfunction!(col_delta_cor_py, m)?)?;
    m.add_function(wrap_pyfunction!(col_delta_cor_partial_py, m)?)?;
    m.add_function(wrap_pyfunction!(umap_knn, m)?)?;
    m.add_function(wrap_pyfunction!(umap_grid_axes_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_transition_grid, m)?)?;
    m.add_function(wrap_pyfunction!(round_delta_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
