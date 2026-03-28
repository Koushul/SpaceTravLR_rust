use crate::config::CnnOutputActivation;
use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

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
    pub(crate) conv2: Conv2d<B>,
    pub(crate) bn2: BatchNorm<B, 2>,
    pub(crate) conv3: Conv2d<B>,
    pub(crate) bn3: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
pub struct SpatialMLP<B: Backend> {
    pub(crate) l1: Linear<B>,
    pub(crate) l2: Linear<B>,
    pub(crate) l3: Linear<B>,
}

#[derive(Module, Debug)]
pub struct HeadMLP<B: Backend> {
    pub(crate) l1: Linear<B>,
    pub(crate) l2: Linear<B>,
}

#[derive(burn::config::Config)]
pub struct CellularNicheNetworkConfig {
    pub n_modulators: usize,
    pub n_clusters: usize,
}

impl CellularNicheNetworkConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        anchors: Tensor<B, 1>,
        output_activation: CnnOutputActivation,
    ) -> CellularNicheNetwork<B> {
        let dim = self.n_modulators + 1;

        CellularNicheNetwork {
            conv_layers: VisionEncoder {
                conv1: Conv2dConfig::new([1, 16], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn1: BatchNormConfig::new(16).init(device),
                conv2: Conv2dConfig::new([16, 32], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn2: BatchNormConfig::new(32).init(device),
                conv3: Conv2dConfig::new([32, 64], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                bn3: BatchNormConfig::new(64).init(device),
            },
            spatial_features_mlp: SpatialMLP {
                l1: LinearConfig::new(self.n_clusters, 16).init(device),
                l2: LinearConfig::new(16, 32).init(device),
                l3: LinearConfig::new(32, 64).init(device),
            },
            mlp: HeadMLP {
                l1: LinearConfig::new(64, 64).init(device),
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
    pub fn get_betas(
        &self,
        spatial_maps: Tensor<B, 4>,
        spatial_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, channels, _h, _w] = spatial_maps.dims();
        debug_assert_eq!(
            channels,
            1,
            "VisionEncoder conv1 expects 1 input channel (per-cluster niche map)"
        );
        let device = &spatial_maps.device();

        let x = self.conv_layers.conv1.forward(spatial_maps);
        let x = self.conv_layers.bn1.forward(x);
        let x = burn::tensor::activation::prelu(x, Tensor::zeros([1], device) + 0.1);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.conv_layers.conv2.forward(x);
        let x = self.conv_layers.bn2.forward(x);
        let x = burn::tensor::activation::prelu(x, Tensor::zeros([1], device) + 0.1);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.conv_layers.conv3.forward(x);
        let x = self.conv_layers.bn3.forward(x);
        let x = burn::tensor::activation::prelu(x, Tensor::zeros([1], device) + 0.1);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = burn::tensor::module::adaptive_avg_pool2d(x, [1, 1]);
        let x = x.reshape([batch, 64]);

        let s = self.spatial_features_mlp.l1.forward(spatial_features);
        let s = burn::tensor::activation::prelu(s, Tensor::zeros([1], device) + 0.1);
        let s = self.spatial_features_mlp.l2.forward(s);
        let s = burn::tensor::activation::prelu(s, Tensor::zeros([1], device) + 0.1);
        let s = self.spatial_features_mlp.l3.forward(s);

        let out = x.add(s);
        let out = self.mlp.l1.forward(out);
        let out = burn::tensor::activation::prelu(out, Tensor::zeros([1], device) + 0.1);
        let out = self.mlp.l2.forward(out);
        let betas = apply_cnn_output_activation(self.output_activation, out);

        betas.mul(self.anchors.clone().unsqueeze_dim(0))
    }

    pub fn forward(
        &self,
        spatial_maps: Tensor<B, 4>,
        inputs_x: Tensor<B, 2>,
        spatial_features: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let betas = self.get_betas(spatial_maps, spatial_features);

        let dims = betas.dims();
        let batch = dims[0];
        let n_betas = dims[1];

        let beta0 = betas.clone().slice([0..batch, 0..1]).reshape([batch]);
        let beta_rest = betas.slice([0..batch, 1..n_betas]);

        let y_interaction = (beta_rest * inputs_x).sum_dim(1).reshape([batch]);
        beta0.add(y_interaction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32, i32>;

    #[test]
    fn get_betas_each_output_activation_is_finite_and_shaped() {
        let device = Default::default();
        let anchors = Tensor::<TestBackend, 1>::from_floats([0.5, 1.0, -0.25], &device);
        let cfg = CellularNicheNetworkConfig {
            n_modulators: 2,
            n_clusters: 2,
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
