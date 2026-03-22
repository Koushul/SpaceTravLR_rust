use rayon::prelude::*;
use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4, Axis};
use crate::lasso::{GroupLasso, GroupLassoParams};
use crate::model::{CellularNicheNetwork, CellularNicheNetworkConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;

#[inline]
pub(crate) fn finite_or_zero_f64(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

#[inline]
fn finite_or_zero_f32(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

fn min_max_finite_col(col: ndarray::ArrayView1<f64>) -> (f32, f32) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in col.iter() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        (0.0, 0.0)
    } else {
        (lo as f32, hi as f32)
    }
}

pub struct ClusteredGCNNWR<B: AutodiffBackend> {
    pub params: GroupLassoParams,
    pub spatial_dim: usize,
    pub models: HashMap<usize, CellularNicheNetwork<B>>,
    pub r2_scores: HashMap<usize, f64>,
    pub lasso_coefficients: HashMap<usize, Array2<f64>>,
    pub lasso_intercepts: HashMap<usize, f64>,
    pub group_reg_vec: Option<Vec<f64>>,
}

impl<B: AutodiffBackend> ClusteredGCNNWR<B> {
    pub fn new(params: GroupLassoParams, spatial_dim: usize) -> Self {
        Self {
            params,
            spatial_dim,
            models: HashMap::new(),
            r2_scores: HashMap::new(),
            lasso_coefficients: HashMap::new(),
            lasso_intercepts: HashMap::new(),
            group_reg_vec: None,
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        device: &B::Device,
        epochs: usize,
        seed_only: bool,
    ) {
        let n_samples = x.nrows();
        let unique_clusters: Vec<usize> = (0..num_clusters).collect();

        let spatial_features = crate::estimator::create_spatial_features(xy, clusters, num_clusters, 100.0);
        let spatial_maps = crate::estimator::xyc2spatial_fast(xy, clusters, num_clusters, self.spatial_dim, self.spatial_dim);

        let fitted_results: Vec<(usize, CellularNicheNetwork<B>, f64, Array2<f64>, f64)> = unique_clusters
            .into_iter()
            .filter_map(|c_id| {
                let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
                if indices.is_empty() { return None; }

                let x_c = x.select(Axis(0), &indices);
                let y_c = y.select(Axis(0), &indices).insert_axis(Axis(1)); 
                
                let mut lasso = if let Some(regs) = &self.group_reg_vec {
                    GroupLasso::new_with_regs(self.params.clone(), regs.clone())
                } else {
                    GroupLasso::new(self.params.clone())
                };
                
                if let Err(e) = lasso.fit(&x_c, &y_c, None) {
                    match e {
                        crate::lasso::GroupLassoError::ConvergenceWarning => {}
                        _ => {
                            println!("⚠️ Lasso fit error for cluster {}: {:?}", c_id, e);
                            return None;
                        }
                    }
                }
                
                let fitted = lasso.fitted.as_ref().unwrap();
                let lasso_coef = fitted.coef.mapv(finite_or_zero_f64);
                let intercept = finite_or_zero_f64(fitted.intercept[[0, 0]]);
                
                let y_pred_lasso = lasso.predict(&x_c).unwrap();
                let y_c_flat = y_c.column(0);
                let y_pred_flat = y_pred_lasso.column(0);
                let ss_res: f64 = y_c_flat.iter().zip(y_pred_flat.iter()).map(|(yi, yhat)| (yi - yhat).powi(2)).sum();
                let y_mean = y_c_flat.mean().unwrap_or(0.0);
                let ss_tot: f64 = y_c_flat.iter().map(|yi| (yi - y_mean).powi(2)).sum();
                let r2 = finite_or_zero_f64(if ss_tot > 0.0 {
                    1.0 - (ss_res / ss_tot)
                } else {
                    0.0
                });

                let mut anchors_vec = vec![finite_or_zero_f32(intercept as f32)];
                anchors_vec.extend(
                    lasso_coef
                        .column(0)
                        .iter()
                        .map(|&v| finite_or_zero_f32(v as f32)),
                );
                
                let anchors_tensor = Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::new(anchors_vec.clone(), [lasso_coef.nrows() + 1]),
                    device,
                );

                let config = CellularNicheNetworkConfig {
                    n_modulators: lasso_coef.nrows(),
                    n_clusters: num_clusters,
                };
                let mut model = config.init::<B>(device, anchors_tensor);
                
                if seed_only {
                    return Some((c_id, model, r2, lasso_coef, intercept));
                }
                
                let cluster_n = indices.len();
                let x_tensor = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(
                        x_c
                            .iter()
                            .map(|&v| finite_or_zero_f32(v as f32))
                            .collect(),
                        [cluster_n, lasso_coef.nrows()],
                    ),
                    device,
                );
                let y_tensor = Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::new(
                        y_c
                            .iter()
                            .map(|&v| finite_or_zero_f32(v as f32))
                            .collect(),
                        [cluster_n],
                    ),
                    device,
                );
                let sf_c = spatial_features.select(Axis(0), &indices);
                let sf_tensor = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(
                        sf_c
                            .iter()
                            .map(|&v| finite_or_zero_f32(v as f32))
                            .collect(),
                        [cluster_n, num_clusters],
                    ),
                    device,
                );
                let sm_c = spatial_maps.select(Axis(0), &indices);
                let sm_tensor = Tensor::<B, 4>::from_data(
                    burn::tensor::TensorData::new(
                        sm_c.iter().cloned().map(finite_or_zero_f32).collect(),
                        [cluster_n, num_clusters, self.spatial_dim, self.spatial_dim],
                    ),
                    device,
                );

                let mut optim = AdamConfig::new().init::<B, CellularNicheNetwork<B>>();
                for _epoch in 0..epochs {
                    let y_pred = model.forward(sm_tensor.clone(), x_tensor.clone(), sf_tensor.clone());
                    let loss = burn::nn::loss::MseLoss::new().forward(y_pred, y_tensor.clone(), burn::nn::loss::Reduction::Mean);
                    
                    let grads = loss.backward();
                    let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                    model = optim.step(1e-3, model, grads);
                }

                Some((c_id, model, r2, lasso_coef, intercept))
            })
            .collect();

        for (id, model, r2, coef, intercept) in fitted_results {
            self.models.insert(id, model);
            self.r2_scores.insert(id, r2);
            self.lasso_coefficients.insert(id, coef);
            self.lasso_intercepts.insert(id, intercept);
        }
    }

    pub fn predict_betas(
        &self,
        x: &Array2<f64>,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        device: &B::Device,
    ) -> Array2<f64> {
        let n_samples = xy.nrows();
        let n_modulators = x.ncols();
        let spatial_features = crate::estimator::create_spatial_features(xy, clusters, num_clusters, 100.0);
        let spatial_maps = crate::estimator::xyc2spatial_fast(xy, clusters, num_clusters, self.spatial_dim, self.spatial_dim);
        
        let mut all_betas = Array2::<f64>::zeros((n_samples, n_modulators + 1));
        
        for c_id in 0..num_clusters {
            let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
            if indices.is_empty() { continue; }

            if let Some(model) = self.models.get(&c_id) {
                let cluster_n = indices.len();
                let sf_c = spatial_features.select(Axis(0), &indices);
                let sf_tensor = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(
                        sf_c
                            .iter()
                            .map(|&v| finite_or_zero_f32(v as f32))
                            .collect(),
                        [cluster_n, num_clusters],
                    ),
                    device,
                );
                let sm_c = spatial_maps.select(Axis(0), &indices);
                let sm_tensor = Tensor::<B, 4>::from_data(
                    burn::tensor::TensorData::new(
                        sm_c.iter().cloned().map(finite_or_zero_f32).collect(),
                        [cluster_n, num_clusters, self.spatial_dim, self.spatial_dim],
                    ),
                    device,
                );
                
                let betas_tensor = model.get_betas(sm_tensor, sf_tensor);
                let betas_data = betas_tensor.into_data();
                let betas_v: &[f32] = betas_data.as_slice::<f32>().unwrap();
                
                let n_betas = n_modulators + 1;
                for (i, idx) in indices.iter().enumerate() {
                    for j in 0..n_betas {
                        let v = betas_v[i * n_betas + j];
                        all_betas[[*idx, j]] = finite_or_zero_f32(v) as f64;
                    }
                }
            } else if let Some(lasso_coef) = self.lasso_coefficients.get(&c_id) {
                let intercept = finite_or_zero_f64(
                    self.lasso_intercepts.get(&c_id).copied().unwrap_or(0.0),
                );
                let coef_col = lasso_coef.column(0);
                for &idx in &indices {
                    all_betas[[idx, 0]] = intercept;
                    for (j, &v) in coef_col.iter().enumerate() {
                        all_betas[[idx, j + 1]] = finite_or_zero_f64(v);
                    }
                }
            }
        }
        all_betas.mapv(finite_or_zero_f64)
    }
}

pub fn xyc2spatial_fast(
    xy: &Array2<f64>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    m: usize,
    n: usize,
) -> Array4<f32> {
    let num_cells = xy.nrows();
    let x_col = xy.column(0);
    let y_col = xy.column(1);

    let (xmin, xmax) = min_max_finite_col(x_col);
    let (ymin, ymax) = min_max_finite_col(y_col);

    let span_x = (xmax - xmin).max(1e-6);
    let span_y = (ymax - ymin).max(1e-6);
    let cell_width = span_x / n as f32;
    let cell_height = span_y / m as f32;

    let cx_grid: Vec<f32> = (0..n).map(|j| xmin + (j as f32 + 0.5) * cell_width).collect();
    let cy_grid: Vec<f32> = (0..m).map(|i| ymax - (i as f32 + 0.5) * cell_height).collect();

    let mut spatial_maps = Array4::<f32>::zeros((num_cells, num_clusters, m, n));

    spatial_maps.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(s, mut cell_maps)| {
            let cluster_s = clusters[s];
            if cluster_s < num_clusters {
                let x_s = xy[[s, 0]] as f32;
                let y_s = xy[[s, 1]] as f32;
                if !x_s.is_finite() || !y_s.is_finite() {
                    return;
                }

                let mut channel_map = cell_maps.index_axis_mut(Axis(0), cluster_s);

                for i in 0..m {
                    let gy = cy_grid[i];
                    if !gy.is_finite() {
                        continue;
                    }
                    let dy2 = (y_s - gy).powi(2);
                    for j in 0..n {
                        let gx = cx_grid[j];
                        if !gx.is_finite() {
                            continue;
                        }
                        let dx2 = (x_s - gx).powi(2);
                        let d = (dx2 + dy2).sqrt().max(1e-6);
                        channel_map[[i, j]] = 1.0 / d;
                    }
                }
            }
        });

    spatial_maps
}

pub fn create_spatial_features(
    xy: &Array2<f64>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    radius: f64,
) -> Array2<f64> {
    let n = xy.nrows();
    let mut result = Array2::zeros((n, num_clusters));
    let r2 = radius * radius;

    result.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let xi = xy[[i, 0]];
            let yi = xy[[i, 1]];
            if !xi.is_finite() || !yi.is_finite() {
                return;
            }
            for j in 0..n {
                let xj = xy[[j, 0]];
                let yj = xy[[j, 1]];
                if !xj.is_finite() || !yj.is_finite() {
                    continue;
                }
                let dx = xi - xj;
                let dy = yi - yj;
                if dx * dx + dy * dy <= r2 {
                    let c = clusters[j];
                    if c < num_clusters {
                        row[c] += 1.0;
                    }
                }
            }
        });
    result
}
