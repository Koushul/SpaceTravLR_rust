use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct CellularNicheNetwork<B: Backend> {
    conv_layers: VisionEncoder<B>,
    spatial_features_mlp: SpatialMLP<B>,
    mlp: HeadMLP<B>,
    pub anchors: Tensor<B, 1>,
}

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
}

#[derive(Module, Debug)]
pub struct SpatialMLP<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

#[derive(Module, Debug)]
pub struct HeadMLP<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
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
        }
    }
}

impl<B: Backend> CellularNicheNetwork<B> {
    pub fn get_betas(
        &self,
        spatial_maps: Tensor<B, 4>,
        spatial_features: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, _c, _h, _w] = spatial_maps.dims();
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
        let betas = burn::tensor::activation::sigmoid(out);

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
