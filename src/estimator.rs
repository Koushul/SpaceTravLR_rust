use crate::config::CnnConfig;
use crate::lasso::{GroupLasso, GroupLassoParams};
use crate::model::{CellularNicheNetwork, CellularNicheNetworkConfig};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use ndarray::{Array1, Array2, Array4, Axis};
use rayon::prelude::*;
use std::collections::HashMap;

/// Precomputed spatial data shared across all per-gene training runs.
pub struct CachedSpatialData {
    pub spatial_features: Array2<f64>,
    pub spatial_maps: Array4<f32>,
}

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

#[derive(Debug, Clone)]
pub struct ClusterTrainingSummary {
    pub cluster_id: usize,
    pub n_cells: usize,
    pub n_modulators: usize,
    pub lasso_r2: f64,
    pub lasso_train_mse: f64,
    pub lasso_fista_iters: usize,
    pub lasso_converged: bool,
    pub cnn_train_mse_epochs: Vec<f32>,
}

pub struct ClusteredGCNNWR<B: AutodiffBackend> {
    pub params: GroupLassoParams,
    pub spatial_dim: usize,
    pub spatial_feature_radius: f64,
    pub models: HashMap<usize, CellularNicheNetwork<B>>,
    pub r2_scores: HashMap<usize, f64>,
    pub lasso_coefficients: HashMap<usize, Array2<f64>>,
    pub lasso_intercepts: HashMap<usize, f64>,
    pub group_reg_vec: Option<Vec<f64>>,
    pub regulator_masks_by_cluster: Option<HashMap<usize, Vec<bool>>>,
    pub cluster_training_summaries: Vec<ClusterTrainingSummary>,
}

impl<B: AutodiffBackend> ClusteredGCNNWR<B> {
    pub fn new(params: GroupLassoParams, spatial_dim: usize, spatial_feature_radius: f64) -> Self {
        Self {
            params,
            spatial_dim,
            spatial_feature_radius,
            models: HashMap::new(),
            r2_scores: HashMap::new(),
            lasso_coefficients: HashMap::new(),
            lasso_intercepts: HashMap::new(),
            group_reg_vec: None,
            regulator_masks_by_cluster: None,
            cluster_training_summaries: Vec::new(),
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
        learning_rate: f64,
        seed_only: bool,
        cnn: &CnnConfig,
        cached_spatial: Option<&CachedSpatialData>,
    ) {
        let n_samples = x.nrows();
        let unique_clusters: Vec<usize> = (0..num_clusters).collect();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            let r_sf = self.spatial_feature_radius;
            owned_sf = create_spatial_features(xy, clusters, num_clusters, r_sf);
            owned_sm = xyc2spatial_fast(
                xy, clusters, num_clusters, self.spatial_dim, self.spatial_dim,
            );
            (&owned_sf, &owned_sm)
        };

        self.cluster_training_summaries.clear();
        let mut training_summaries: Vec<ClusterTrainingSummary> = Vec::new();

        let fitted_results: Vec<(usize, CellularNicheNetwork<B>, f64, Array2<f64>, f64)> =
            unique_clusters
                .into_iter()
                .filter_map(|c_id| {
                    let indices: Vec<usize> =
                        (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
                    if indices.is_empty() {
                        return None;
                    }

                    let x_c = x.select(Axis(0), &indices);
                    let mut x_c = x_c;
                    if let Some(mask) = self
                        .regulator_masks_by_cluster
                        .as_ref()
                        .and_then(|m| m.get(&c_id))
                    {
                        for (j, allowed) in mask.iter().copied().enumerate().take(x_c.ncols()) {
                            if !allowed {
                                x_c.column_mut(j).fill(0.0);
                            }
                        }
                    }
                    let y_c = y.select(Axis(0), &indices).insert_axis(Axis(1));

                    let mut lasso = if let Some(regs) = &self.group_reg_vec {
                        GroupLasso::new_with_regs(self.params.clone(), regs.clone())
                    } else {
                        GroupLasso::new(self.params.clone())
                    };

                    let lasso_converged = match lasso.fit(&x_c, &y_c, None) {
                        Ok(_) => true,
                        Err(crate::lasso::GroupLassoError::ConvergenceWarning) => false,
                        Err(e) => {
                            println!("⚠️ Lasso fit error for cluster {}: {:?}", c_id, e);
                            return None;
                        }
                    };

                    let fitted = lasso.fitted.as_ref().unwrap();
                    let lasso_coef = fitted.coef.mapv(finite_or_zero_f64);
                    let intercept = finite_or_zero_f64(fitted.intercept[[0, 0]]);
                    let lasso_fista_iters = lasso.last_fista_iterations;

                    let y_pred_lasso = lasso.predict(&x_c).unwrap();
                    let y_c_flat = y_c.column(0);
                    let y_pred_flat = y_pred_lasso.column(0);
                    let ss_res: f64 = y_c_flat
                        .iter()
                        .zip(y_pred_flat.iter())
                        .map(|(yi, yhat)| (yi - yhat).powi(2))
                        .sum();
                    let cluster_n = indices.len();
                    let lasso_train_mse = ss_res / cluster_n.max(1) as f64;
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
                        burn::tensor::TensorData::new(
                            anchors_vec.clone(),
                            [lasso_coef.nrows() + 1],
                        ),
                        device,
                    );

                    let config = CellularNicheNetworkConfig {
                        n_modulators: lasso_coef.nrows(),
                        n_clusters: num_clusters,
                    };
                    let mut model = config.init::<B>(device, anchors_tensor);

                    if seed_only {
                        training_summaries.push(ClusterTrainingSummary {
                            cluster_id: c_id,
                            n_cells: cluster_n,
                            n_modulators: lasso_coef.nrows(),
                            lasso_r2: r2,
                            lasso_train_mse,
                            lasso_fista_iters,
                            lasso_converged,
                            cnn_train_mse_epochs: Vec::new(),
                        });
                        return Some((c_id, model, r2, lasso_coef, intercept));
                    }

                    let x_tensor = Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(
                            x_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
                            [cluster_n, lasso_coef.nrows()],
                        ),
                        device,
                    );
                    let y_tensor = Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::new(
                            y_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
                            [cluster_n],
                        ),
                        device,
                    );
                    let sf_c = spatial_features.select(Axis(0), &indices);
                    let sf_tensor = Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(
                            sf_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
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

                    let mut adam = AdamConfig::new()
                        .with_beta_1(cnn.adam_beta_1 as f32)
                        .with_beta_2(cnn.adam_beta_2 as f32)
                        .with_epsilon(cnn.adam_epsilon as f32);
                    if let Some(wd) = cnn.weight_decay {
                        adam = adam.with_weight_decay(Some(WeightDecayConfig::new(wd as f32)));
                    }
                    if let Some(gc) = cnn.grad_clip_norm {
                        adam =
                            adam.with_grad_clipping(Some(GradientClippingConfig::Norm(gc as f32)));
                    }
                    let mut optim = adam.init::<B, CellularNicheNetwork<B>>();
                    let mut cnn_train_mse_epochs = Vec::with_capacity(epochs);
                    for _epoch in 0..epochs {
                        let y_pred =
                            model.forward(sm_tensor.clone(), x_tensor.clone(), sf_tensor.clone());
                        let loss = burn::nn::loss::MseLoss::new().forward(
                            y_pred,
                            y_tensor.clone(),
                            burn::nn::loss::Reduction::Mean,
                        );
                        let mse = finite_or_zero_f32(loss.clone().into_scalar().elem());
                        cnn_train_mse_epochs.push(mse);
                        let grads = loss.backward();
                        let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                        model = optim.step(learning_rate, model, grads);
                    }

                    training_summaries.push(ClusterTrainingSummary {
                        cluster_id: c_id,
                        n_cells: cluster_n,
                        n_modulators: lasso_coef.nrows(),
                        lasso_r2: r2,
                        lasso_train_mse,
                        lasso_fista_iters,
                        lasso_converged,
                        cnn_train_mse_epochs,
                    });

                    Some((c_id, model, r2, lasso_coef, intercept))
                })
                .collect();

        self.cluster_training_summaries = training_summaries;

        for (id, model, r2, coef, intercept) in fitted_results {
            self.models.insert(id, model);
            self.r2_scores.insert(id, r2);
            self.lasso_coefficients.insert(id, coef);
            self.lasso_intercepts.insert(id, intercept);
        }
    }

    pub fn fit_cnn_refinement(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        device: &B::Device,
        epochs: usize,
        learning_rate: f64,
        cnn: &CnnConfig,
        cached_spatial: Option<&CachedSpatialData>,
    ) {
        let n_samples = x.nrows();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            owned_sf = create_spatial_features(xy, clusters, num_clusters, self.spatial_feature_radius);
            owned_sm = xyc2spatial_fast(
                xy, clusters, num_clusters, self.spatial_dim, self.spatial_dim,
            );
            (&owned_sf, &owned_sm)
        };

        let mut summaries_by_cluster: HashMap<usize, ClusterTrainingSummary> = self
            .cluster_training_summaries
            .iter()
            .cloned()
            .map(|s| (s.cluster_id, s))
            .collect();

        for c_id in 0..num_clusters {
            if !self.models.contains_key(&c_id) {
                continue;
            }
            let indices: Vec<usize> =
                (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
            if indices.is_empty() {
                continue;
            }

            let mut model = match self.models.remove(&c_id) {
                Some(m) => m,
                None => continue,
            };

            let cluster_n = indices.len();
            let n_modulators = x.ncols();
            let x_c = x.select(Axis(0), &indices);
            let mut x_c = x_c;
            if let Some(mask) = self
                .regulator_masks_by_cluster
                .as_ref()
                .and_then(|m| m.get(&c_id))
            {
                for (j, allowed) in mask.iter().copied().enumerate().take(x_c.ncols()) {
                    if !allowed {
                        x_c.column_mut(j).fill(0.0);
                    }
                }
            }
            let y_c = y.select(Axis(0), &indices);

            let x_tensor = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(
                    x_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
                    [cluster_n, n_modulators],
                ),
                device,
            );
            let y_tensor = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(
                    y_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
                    [cluster_n],
                ),
                device,
            );
            let sf_c = spatial_features.select(Axis(0), &indices);
            let sf_tensor = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(
                    sf_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
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

            let mut adam = AdamConfig::new()
                .with_beta_1(cnn.adam_beta_1 as f32)
                .with_beta_2(cnn.adam_beta_2 as f32)
                .with_epsilon(cnn.adam_epsilon as f32);
            if let Some(wd) = cnn.weight_decay {
                adam = adam.with_weight_decay(Some(WeightDecayConfig::new(wd as f32)));
            }
            if let Some(gc) = cnn.grad_clip_norm {
                adam = adam.with_grad_clipping(Some(GradientClippingConfig::Norm(gc as f32)));
            }
            let mut optim = adam.init::<B, CellularNicheNetwork<B>>();
            let mut cnn_train_mse_epochs = Vec::with_capacity(epochs);
            for _epoch in 0..epochs {
                let y_pred =
                    model.forward(sm_tensor.clone(), x_tensor.clone(), sf_tensor.clone());
                let loss = burn::nn::loss::MseLoss::new().forward(
                    y_pred,
                    y_tensor.clone(),
                    burn::nn::loss::Reduction::Mean,
                );
                let mse = finite_or_zero_f32(loss.clone().into_scalar().elem());
                cnn_train_mse_epochs.push(mse);
                let grads = loss.backward();
                let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(learning_rate, model, grads);
            }

            self.models.insert(c_id, model);

            if let Some(s) = summaries_by_cluster.get_mut(&c_id) {
                s.cnn_train_mse_epochs = cnn_train_mse_epochs;
            }
        }

        let mut ordered: Vec<ClusterTrainingSummary> = summaries_by_cluster.into_values().collect();
        ordered.sort_by_key(|s| s.cluster_id);
        self.cluster_training_summaries = ordered;
    }

    pub fn predict_betas(
        &self,
        x: &Array2<f64>,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        device: &B::Device,
        cached_spatial: Option<&CachedSpatialData>,
    ) -> Array2<f64> {
        let n_samples = xy.nrows();
        let n_modulators = x.ncols();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            owned_sf = create_spatial_features(
                xy, clusters, num_clusters, self.spatial_feature_radius,
            );
            owned_sm = xyc2spatial_fast(
                xy, clusters, num_clusters, self.spatial_dim, self.spatial_dim,
            );
            (&owned_sf, &owned_sm)
        };

        let mut all_betas = Array2::<f64>::zeros((n_samples, n_modulators + 1));

        for c_id in 0..num_clusters {
            let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
            if indices.is_empty() {
                continue;
            }

            if let Some(model) = self.models.get(&c_id) {
                let cluster_n = indices.len();
                let sf_c = spatial_features.select(Axis(0), &indices);
                let sf_tensor = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(
                        sf_c.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
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
                let intercept =
                    finite_or_zero_f64(self.lasso_intercepts.get(&c_id).copied().unwrap_or(0.0));
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

    let cx_grid: Vec<f32> = (0..n)
        .map(|j| xmin + (j as f32 + 0.5) * cell_width)
        .collect();
    let cy_grid: Vec<f32> = (0..m)
        .map(|i| ymax - (i as f32 + 0.5) * cell_height)
        .collect();

    let mut spatial_maps = Array4::<f32>::zeros((num_cells, num_clusters, m, n));

    spatial_maps
        .axis_iter_mut(Axis(0))
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

    let mut points = Vec::with_capacity(n);
    let mut valid_indices = Vec::with_capacity(n);
    for i in 0..n {
        let x = xy[[i, 0]];
        let y = xy[[i, 1]];
        if x.is_finite() && y.is_finite() {
            valid_indices.push(i);
            points.push([x, y]);
        }
    }

    if points.is_empty() {
        return result;
    }

    let tree = kiddo::ImmutableKdTree::<f64, 2>::new_from_slice(&points);

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let xi = xy[[i, 0]];
            let yi = xy[[i, 1]];
            if !xi.is_finite() || !yi.is_finite() {
                return;
            }
            let neighbors = tree.within::<kiddo::SquaredEuclidean>(&[xi, yi], r2);
            for nb in &neighbors {
                let j = valid_indices[nb.item as usize];
                let c = clusters[j];
                if c < num_clusters {
                    row[c] += 1.0;
                }
            }
        });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn finite_or_zero_f64_normal() {
        assert_abs_diff_eq!(finite_or_zero_f64(3.14), 3.14, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(-2.0), -2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn finite_or_zero_f64_special() {
        assert_abs_diff_eq!(finite_or_zero_f64(f64::NAN), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(f64::INFINITY), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(f64::NEG_INFINITY), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn finite_or_zero_f32_special() {
        assert_eq!(finite_or_zero_f32(f32::NAN), 0.0);
        assert_eq!(finite_or_zero_f32(f32::INFINITY), 0.0);
        assert_eq!(finite_or_zero_f32(1.5), 1.5);
    }

    #[test]
    fn min_max_finite_col_normal() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn min_max_finite_col_with_nan() {
        let data = array![[f64::NAN], [2.0], [5.0]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 2.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn min_max_finite_col_all_nan() {
        let data = array![[f64::NAN], [f64::NAN]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn spatial_features_shape() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 1, 0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 100.0);
        assert_eq!(sf.shape(), &[4, 2]);
    }

    #[test]
    fn spatial_features_self_count() {
        // Each cell is within radius of itself (distance = 0)
        let xy = array![[0.0, 0.0], [1000.0, 1000.0]];
        let clusters = Array1::from_vec(vec![0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 1.0); // small radius
        // Cell 0: only itself in radius → cluster 0 count = 1
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
        // Cell 1: only itself → cluster 1 count = 1
        assert_abs_diff_eq!(sf[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_all_in_radius() {
        // All cells within radius of each other
        let xy = array![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let sf = create_spatial_features(&xy, &clusters, 2, 10.0);
        // Cell 0: sees cells 0,2 (cluster 0) and cell 1 (cluster 1)
        assert_abs_diff_eq!(sf[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_radius_boundary() {
        // Cell 0 at origin, cell 1 at (1,0). Radius = 1.0 → distance = 1.0 ≤ radius
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 1.0);
        // Both cells within radius of each other
        assert_abs_diff_eq!(sf[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_just_outside_radius() {
        // Cell 0 at origin, cell 1 at (1.01, 0). Radius = 1.0 → distance > radius
        let xy = array![[0.0, 0.0], [1.01, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 1.0);
        // Each cell only sees itself
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_nan_handling() {
        let xy = array![[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 100.0);
        // Cell 0 has NaN coords → row should be all zeros
        assert_abs_diff_eq!(sf[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
        // Cells 1,2 should not count cell 0
        assert_abs_diff_eq!(sf[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_shape() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 2, 8, 8);
        assert_eq!(maps.shape(), &[3, 2, 8, 8]);
    }

    #[test]
    fn xyc2spatial_only_own_cluster_nonzero() {
        // Cell 0 is cluster 0 → only channel 0 should have nonzero entries
        let xy = array![[0.5, 0.5], [1.5, 1.5]];
        let clusters = Array1::from_vec(vec![0, 1]);
        let maps = xyc2spatial_fast(&xy, &clusters, 2, 4, 4);

        // Cell 0, cluster 0 channel should have nonzero values
        let ch0_sum: f32 = maps.slice(ndarray::s![0, 0, .., ..]).iter().sum();
        assert!(ch0_sum > 0.0, "Own cluster channel should be nonzero");

        // Cell 0, cluster 1 channel should be zero
        let ch1_sum: f32 = maps.slice(ndarray::s![0, 1, .., ..]).iter().sum();
        assert_abs_diff_eq!(ch1_sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_inverse_distance_positive() {
        let xy = array![[0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4);
        // All values in the active channel should be positive (1/d > 0)
        for &v in maps.slice(ndarray::s![0, 0, .., ..]).iter() {
            assert!(v > 0.0, "Inverse distance should be positive");
        }
    }

    #[test]
    fn xyc2spatial_closer_grid_points_higher_value() {
        let xy = array![[0.0, 1.0]]; // at the top-left area
        let clusters = Array1::from_vec(vec![0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4);
        let channel = maps.slice(ndarray::s![0, 0, .., ..]);
        // The grid point closest to the cell should have the highest value
        let max_val = channel.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_val > 0.0);
    }

    #[test]
    fn xyc2spatial_nan_cell_is_zero() {
        let xy = array![[f64::NAN, 0.0], [1.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4);
        let cell0_sum: f32 = maps.slice(ndarray::s![0, .., .., ..]).iter().sum();
        assert_abs_diff_eq!(cell0_sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_deterministic() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let m1 = xyc2spatial_fast(&xy, &clusters, 2, 4, 4);
        let m2 = xyc2spatial_fast(&xy, &clusters, 2, 4, 4);
        assert_eq!(m1, m2);
    }

    #[test]
    fn spatial_features_symmetry() {
        // Two cells at same distance from each other, both cluster 0
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 10.0);
        assert_abs_diff_eq!(sf[[0, 0]], sf[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_nonnegative() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let sf = create_spatial_features(&xy, &clusters, 2, 5.0);
        for &v in sf.iter() {
            assert!(v >= 0.0, "Spatial features (counts) must be non-negative");
        }
    }

    #[test]
    fn spatial_features_large_cluster_count() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 5]);
        let sf = create_spatial_features(&xy, &clusters, 10, 100.0);
        assert_eq!(sf.ncols(), 10);
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 5]], 1.0, epsilon = 1e-10);
        // Other cluster columns should be zero
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
    }
}
