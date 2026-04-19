use crate::config::CnnOutputActivation;
use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PRelu, PReluConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

/// Initial slope used for every PReLU site, matching the Python reference
/// (`nn.PReLU(init=0.1)`). The slope itself is a learnable parameter, so this
/// only fixes the starting value.
pub(crate) const PRELU_INIT_ALPHA: f64 = 0.1;

/// BatchNorm `epsilon` added to variance before `sqrt` (`std = sqrt(var + eps)`).
///
/// PyTorch’s default (`1e-5`, see `python_reference_cnn.py::BN_EPS`) is fine on CPU/CUDA.
/// On **WebGPU f32**, full-cluster batches (thousands of cells × spatial map) can push some
/// channel variances very small mid-training; the BatchNorm backward then scales like `~1/std`
/// and can overflow to **NaN** weights. A larger `eps` caps that ratio (same idea as using a
/// less aggressive normalization). Architecture stays identical to the Python reference; only
/// numerical stability for the GPU trainer differs.
pub const CNN_BATCH_NORM_EPS: f64 = 1e-3;

pub const CNN_SPP_FLAT_DIM: usize = 64 * (1 + 4 + 16);

pub(crate) fn cnn_output_activation_tag(a: CnnOutputActivation) -> u8 {
    match a {
        CnnOutputActivation::Identity => 0,
        CnnOutputActivation::Sigmoid => 1,
        CnnOutputActivation::Tanh => 2,
        CnnOutputActivation::SigmoidX2 => 3,
    }
}

pub(crate) fn apply_cnn_output_activation<B: Backend>(tag: u8, out: Tensor<B, 2>) -> Tensor<B, 2> {
    match tag {
        0 => out,
        2 => burn::tensor::activation::tanh(out),
        3 => burn::tensor::activation::sigmoid(out) * 2.0,
        _ => burn::tensor::activation::sigmoid(out),
    }
}

#[derive(Module, Debug)]
pub struct CellularNicheNetwork<B: Backend> {
    pub(crate) conv_layers: VisionEncoder<B>,
    pub(crate) spatial_features_mlp: SpatialMLP<B>,
    pub(crate) mlp: HeadMLP<B>,
    pub anchors: Tensor<B, 1>,
    pub(crate) output_activation: u8,
}

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    pub(crate) conv1: Conv2d<B>,
    pub(crate) bn1: BatchNorm<B, 2>,
    pub(crate) prelu1: PRelu<B>,
    pub(crate) conv2: Conv2d<B>,
    pub(crate) bn2: BatchNorm<B, 2>,
    pub(crate) prelu2: PRelu<B>,
    pub(crate) conv3: Conv2d<B>,
    pub(crate) bn3: BatchNorm<B, 2>,
    pub(crate) prelu3: PRelu<B>,
    pub(crate) spp_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct SpatialMLP<B: Backend> {
    pub(crate) l1: Linear<B>,
    pub(crate) prelu1: PRelu<B>,
    pub(crate) l2: Linear<B>,
    pub(crate) prelu2: PRelu<B>,
    pub(crate) l3: Linear<B>,
}

#[derive(Module, Debug)]
pub struct HeadMLP<B: Backend> {
    pub(crate) l1: Linear<B>,
    pub(crate) prelu: PRelu<B>,
    pub(crate) l2: Linear<B>,
}

#[derive(burn::config::Config)]
pub struct CellularNicheNetworkConfig {
    pub n_modulators: usize,
    pub n_clusters: usize,
    pub vision_in_channels: usize,
}

impl CellularNicheNetworkConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        anchors: Tensor<B, 1>,
        output_activation: CnnOutputActivation,
    ) -> CellularNicheNetwork<B> {
        let dim = self.n_modulators + 1;
        let in_ch = self.vision_in_channels.max(1);
        let prelu = || {
            PReluConfig::new()
                .with_num_parameters(1)
                .with_alpha(PRELU_INIT_ALPHA)
                .init(device)
        };

        CellularNicheNetwork {
            conv_layers: VisionEncoder {
                conv1: Conv2dConfig::new([in_ch, 16], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn1: BatchNormConfig::new(16)
                    .with_epsilon(CNN_BATCH_NORM_EPS)
                    .init(device),
                prelu1: prelu(),
                conv2: Conv2dConfig::new([16, 32], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn2: BatchNormConfig::new(32)
                    .with_epsilon(CNN_BATCH_NORM_EPS)
                    .init(device),
                prelu2: prelu(),
                conv3: Conv2dConfig::new([32, 64], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn3: BatchNormConfig::new(64)
                    .with_epsilon(CNN_BATCH_NORM_EPS)
                    .init(device),
                prelu3: prelu(),
                spp_proj: LinearConfig::new(CNN_SPP_FLAT_DIM, 64).init(device),
            },
            spatial_features_mlp: SpatialMLP {
                l1: LinearConfig::new(self.n_clusters, 16).init(device),
                prelu1: prelu(),
                l2: LinearConfig::new(16, 32).init(device),
                prelu2: prelu(),
                l3: LinearConfig::new(32, 64).init(device),
            },
            mlp: HeadMLP {
                l1: LinearConfig::new(64, 64).init(device),
                prelu: prelu(),
                l2: LinearConfig::new(64, dim).init(device),
            },
            anchors,
            output_activation: cnn_output_activation_tag(output_activation),
        }
    }
}

impl<B: Backend> CellularNicheNetwork<B> {
    /// `spatial_maps` must be `[batch, 1, H, W]` — one inverse-distance map for the cluster being
    /// trained (see `spatial_maps_for_cluster_cnn` in `estimator.rs`). Neighbor-count context for
    /// all clusters stays in `spatial_features` `[batch, n_clusters]`. Lasso intercept + coefficients
    /// seed `anchors`; the CNN applies `output_activation` to the last linear output, then scales by anchors.
    ///
    /// Layer-for-layer port of the reference Python `CellularNicheNetwork.get_betas`:
    /// 3× (Conv2d same-pad → BatchNorm2d → PReLU → MaxPool2d 2×2) → AdaptiveAvgPool2d(1) →
    /// flatten, summed with a 3-layer PReLU MLP over `spatial_features`, head MLP, output
    /// activation, scaled by `anchors`. Each PReLU is a learnable single-parameter module
    /// initialised at 0.1 (same as `nn.PReLU(init=0.1)`).
    ///
    /// Note: the Python source wraps each `nn.Conv2d` in `weight_norm`, which burn 0.16 does
    /// not ship. `weight_norm` is a reparameterisation `W = g · v / ‖v‖` with the same function
    /// space as a plain `Conv2d`, so the inference graph here is equivalent; only the
    /// optimiser-level conditioning differs.
    pub fn get_betas(
        &self,
        spatial_maps: Tensor<B, 4>,
        spatial_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, _channels, _h, _w] = spatial_maps.dims();

        let x = self.conv_layers.conv1.forward(spatial_maps);
        let x = self.conv_layers.bn1.forward(x);
        let x = self.conv_layers.prelu1.forward(x);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.conv_layers.conv2.forward(x);
        let x = self.conv_layers.bn2.forward(x);
        let x = self.conv_layers.prelu2.forward(x);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.conv_layers.conv3.forward(x);
        let x = self.conv_layers.bn3.forward(x);
        let x = self.conv_layers.prelu3.forward(x);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let p1 = burn::tensor::module::adaptive_avg_pool2d(x.clone(), [1, 1]);
        let p2 = burn::tensor::module::adaptive_avg_pool2d(x.clone(), [2, 2]);
        let p3 = burn::tensor::module::adaptive_avg_pool2d(x, [4, 4]);
        let f1 = p1.reshape([batch, 64]);
        let f2 = p2.reshape([batch, 64 * 4]);
        let f3 = p3.reshape([batch, 64 * 16]);
        let spp = Tensor::cat(vec![f1, f2, f3], 1);
        let x = self.conv_layers.spp_proj.forward(spp);

        let s = self.spatial_features_mlp.l1.forward(spatial_features);
        let s = self.spatial_features_mlp.prelu1.forward(s);
        let s = self.spatial_features_mlp.l2.forward(s);
        let s = self.spatial_features_mlp.prelu2.forward(s);
        let s = self.spatial_features_mlp.l3.forward(s);

        let out = x.add(s);
        let out = self.mlp.l1.forward(out);
        let out = self.mlp.prelu.forward(out);
        let out = self.mlp.l2.forward(out);
        let betas = apply_cnn_output_activation(self.output_activation, out);

        betas.mul(self.anchors.clone().unsqueeze_dim(0))
    }

    pub fn linear_readout_y(betas: Tensor<B, 2>, inputs_x: Tensor<B, 2>) -> Tensor<B, 1> {
        let dims = betas.dims();
        let batch = dims[0];
        let n_betas = dims[1];

        let beta0 = betas.clone().slice([0..batch, 0..1]).reshape([batch]);
        let beta_rest = betas.slice([0..batch, 1..n_betas]);

        let y_interaction = (beta_rest * inputs_x).sum_dim(1).reshape([batch]);
        beta0.add(y_interaction)
    }

    pub fn forward(
        &self,
        spatial_maps: Tensor<B, 4>,
        inputs_x: Tensor<B, 2>,
        spatial_features: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let betas = self.get_betas(spatial_maps, spatial_features);
        Self::linear_readout_y(betas, inputs_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32, i32>;

    #[test]
    fn prelu_modules_initialize_at_python_default_slope() {
        let device = Default::default();
        let cfg = CellularNicheNetworkConfig {
            n_modulators: 2,
            n_clusters: 2,
            vision_in_channels: 1,
        };
        let anchors = Tensor::<TestBackend, 1>::ones([3], &device);
        let net = cfg.init::<TestBackend>(&device, anchors, CnnOutputActivation::Sigmoid);

        for alpha in [
            net.conv_layers.prelu1.alpha.val(),
            net.conv_layers.prelu2.alpha.val(),
            net.conv_layers.prelu3.alpha.val(),
            net.spatial_features_mlp.prelu1.alpha.val(),
            net.spatial_features_mlp.prelu2.alpha.val(),
            net.mlp.prelu.alpha.val(),
        ] {
            assert_eq!(alpha.dims(), [1]);
            let v = alpha.into_data().as_slice::<f32>().unwrap()[0];
            assert!(
                (v - PRELU_INIT_ALPHA as f32).abs() < 1e-6,
                "PReLU alpha init expected {} got {v}",
                PRELU_INIT_ALPHA
            );
        }
    }

    #[test]
    fn get_betas_each_output_activation_is_finite_and_shaped() {
        let device = Default::default();
        let anchors = Tensor::<TestBackend, 1>::from_floats([0.5, 1.0, -0.25], &device);
        let cfg = CellularNicheNetworkConfig {
            n_modulators: 2,
            n_clusters: 2,
            vision_in_channels: 1,
        };
        let batch = 2usize;
        let h = 32usize;
        let sm = Tensor::<TestBackend, 4>::zeros([batch, 1, h, h], &device);
        let sf = Tensor::<TestBackend, 2>::zeros([batch, 2], &device);

        for act in [
            CnnOutputActivation::Identity,
            CnnOutputActivation::Sigmoid,
            CnnOutputActivation::Tanh,
            CnnOutputActivation::SigmoidX2,
        ] {
            let net = cfg.init::<TestBackend>(&device, anchors.clone(), act);
            let betas = net.get_betas(sm.clone(), sf.clone());
            assert_eq!(betas.dims(), [batch, 3]);
            let sl = betas.into_data().as_slice::<f32>().unwrap().to_vec();
            assert!(
                sl.iter().all(|x| x.is_finite()),
                "activation {:?} produced non-finite",
                act
            );
        }
    }

    #[test]
    fn sigmoid_activation_matches_tag_default_path() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [-1.0, 2.0]], &device);
        let s = apply_cnn_output_activation(1, t.clone());
        let e = burn::tensor::activation::sigmoid(t);
        let a: Vec<f32> = s.into_data().as_slice::<f32>().unwrap().to_vec();
        let b: Vec<f32> = e.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn sigmoid_x2_is_twice_sigmoid() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [-1.0, 2.0]], &device);
        let s = apply_cnn_output_activation(3, t.clone());
        let e = burn::tensor::activation::sigmoid(t) * 2.0;
        let a: Vec<f32> = s.into_data().as_slice::<f32>().unwrap().to_vec();
        let b: Vec<f32> = e.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn identity_tag_is_passthrough() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [-1.0, 2.0]], &device);
        let s = apply_cnn_output_activation(0, t.clone());
        let a: Vec<f32> = s.into_data().as_slice::<f32>().unwrap().to_vec();
        let b: Vec<f32> = t.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn tanh_tag_matches_burn_tanh() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [-1.0, 2.0]], &device);
        let s = apply_cnn_output_activation(2, t.clone());
        let e = burn::tensor::activation::tanh(t);
        let a: Vec<f32> = s.into_data().as_slice::<f32>().unwrap().to_vec();
        let b: Vec<f32> = e.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(a, b);
    }
}
