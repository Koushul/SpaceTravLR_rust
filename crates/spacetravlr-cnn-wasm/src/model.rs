use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PRelu, PReluConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

pub const PRELU_INIT_ALPHA: f64 = 0.1;
pub const CNN_BATCH_NORM_EPS: f64 = 1e-3;
pub const CNN_SPP_FLAT_DIM: usize = 64 * (1 + 4 + 16);

pub fn apply_cnn_output_activation<B: Backend>(tag: u8, out: Tensor<B, 2>) -> Tensor<B, 2> {
    match tag {
        0 => out,
        2 => burn::tensor::activation::tanh(out),
        3 => burn::tensor::activation::sigmoid(out) * 2.0,
        _ => burn::tensor::activation::sigmoid(out),
    }
}

#[derive(Module, Debug)]
pub struct CellularNicheNetwork<B: Backend> {
    pub conv_layers: VisionEncoder<B>,
    pub spatial_features_mlp: SpatialMLP<B>,
    pub mlp: HeadMLP<B>,
    pub anchors: Tensor<B, 1>,
    pub anchors_row: Tensor<B, 2>,
    pub output_activation: u8,
}

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B, 2>,
    pub prelu1: PRelu<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm<B, 2>,
    pub prelu2: PRelu<B>,
    pub conv3: Conv2d<B>,
    pub bn3: BatchNorm<B, 2>,
    pub prelu3: PRelu<B>,
    pub spp_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct SpatialMLP<B: Backend> {
    pub l1: Linear<B>,
    pub prelu1: PRelu<B>,
    pub l2: Linear<B>,
    pub prelu2: PRelu<B>,
    pub l3: Linear<B>,
}

#[derive(Module, Debug)]
pub struct HeadMLP<B: Backend> {
    pub l1: Linear<B>,
    pub prelu: PRelu<B>,
    pub l2: Linear<B>,
}

#[derive(Config)]
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
        output_activation: u8,
    ) -> CellularNicheNetwork<B> {
        let dim = self.n_modulators + 1;
        let in_ch = self.vision_in_channels.max(1);
        let prelu = || {
            PReluConfig::new()
                .with_num_parameters(1)
                .with_alpha(PRELU_INIT_ALPHA)
                .init(device)
        };
        let n_anchor = anchors.dims()[0];
        let anchors_row = anchors.clone().reshape([1, n_anchor]);
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
            anchors_row,
            output_activation,
        }
    }
}

impl<B: Backend> CellularNicheNetwork<B> {
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
        betas.mul(self.anchors_row.clone())
    }

    pub fn linear_readout_y(betas: Tensor<B, 2>, inputs_x: Tensor<B, 2>) -> Tensor<B, 1> {
        let dims = betas.dims();
        let batch = dims[0];
        let n_betas = dims[1];
        if n_betas <= 1 {
            return betas.reshape([batch]);
        }
        let mut parts = betas
            .split_with_sizes(vec![1, n_betas.saturating_sub(1)], 1)
            .into_iter();
        let beta0 = parts.next().expect("split column 0").reshape([batch]);
        let beta_rest = parts.next().expect("split rest");
        let y_interaction = (beta_rest * inputs_x).sum_dim(1).reshape([batch]);
        beta0.add(y_interaction)
    }
}
