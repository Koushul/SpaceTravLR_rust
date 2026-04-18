//! `NicheEncoder` — a small CNN that consumes per-cell `(n_targets ×
//! n_modulators)` splash images and produces a `D`-dim embedding plus three
//! auxiliary heads (functional, recon, projection-for-contrastive).

use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

/// Configuration for [`NicheEncoder`].
#[derive(burn::config::Config)]
pub struct NicheEncoderConfig {
    pub in_height: usize,
    pub in_width: usize,
    /// Embedding dimension. Default 32.
    #[config(default = 32)]
    pub embedding_dim: usize,
    /// Number of unsupervised programs the functional head predicts.
    #[config(default = 16)]
    pub n_programs: usize,
    /// Recon head reconstructs an `(in_height/down × in_width/down)` low-rank
    /// summary of the input. Default `down=4`.
    #[config(default = 4)]
    pub recon_down: usize,
    #[config(default = 32)]
    pub conv1_channels: usize,
    #[config(default = 64)]
    pub conv2_channels: usize,
    #[config(default = 64)]
    pub conv3_channels: usize,
    /// Internal MLP hidden width.
    #[config(default = 128)]
    pub mlp_hidden: usize,
    /// Projection head dim (for contrastive loss). Default 16.
    #[config(default = 16)]
    pub projection_dim: usize,
}

/// Internal vision backbone — three conv blocks with BN + PReLU + 2x2 maxpool,
/// then adaptive avg pool to `(pool_h × pool_w)`.
#[derive(Module, Debug)]
pub struct NicheBackbone<B: Backend> {
    pub(crate) conv1: Conv2d<B>,
    pub(crate) bn1: BatchNorm<B, 2>,
    pub(crate) conv2: Conv2d<B>,
    pub(crate) bn2: BatchNorm<B, 2>,
    pub(crate) conv3: Conv2d<B>,
    pub(crate) bn3: BatchNorm<B, 2>,
    pub(crate) pool_h: usize,
    pub(crate) pool_w: usize,
    pub(crate) out_channels: usize,
}

#[derive(Module, Debug)]
pub struct NicheHeads<B: Backend> {
    pub(crate) embed_l1: Linear<B>,
    pub(crate) embed_l2: Linear<B>,
    pub(crate) func_l1: Linear<B>,
    pub(crate) func_l2: Linear<B>,
    pub(crate) recon_l1: Linear<B>,
    pub(crate) recon_l2: Linear<B>,
    pub(crate) proj_l1: Linear<B>,
    pub(crate) proj_l2: Linear<B>,
    pub(crate) recon_h: usize,
    pub(crate) recon_w: usize,
}

#[derive(Module, Debug)]
pub struct NicheEncoder<B: Backend> {
    pub(crate) backbone: NicheBackbone<B>,
    pub(crate) heads: NicheHeads<B>,
    pub(crate) embedding_dim: usize,
}

/// Output of one [`NicheEncoder::forward`] call.
pub struct NicheForward<B: Backend> {
    pub embedding: Tensor<B, 2>,
    pub functional: Tensor<B, 2>,
    pub recon: Tensor<B, 4>,
    pub projection: Tensor<B, 2>,
}

impl NicheEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NicheEncoder<B> {
        let pool_h = (self.in_height / 2).max(1);
        let pool_w = (self.in_width / 2).max(1);
        let pool_h = pool_h.min(8).max(2);
        let pool_w = pool_w.min(8).max(2);
        let out_channels = self.conv3_channels;
        let backbone = NicheBackbone {
            conv1: Conv2dConfig::new([1, self.conv1_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.conv1_channels).init(device),
            conv2: Conv2dConfig::new([self.conv1_channels, self.conv2_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.conv2_channels).init(device),
            conv3: Conv2dConfig::new([self.conv2_channels, self.conv3_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn3: BatchNormConfig::new(self.conv3_channels).init(device),
            pool_h,
            pool_w,
            out_channels,
        };

        let flat = out_channels * pool_h * pool_w;
        let recon_h = (self.in_height / self.recon_down.max(1)).max(2);
        let recon_w = (self.in_width / self.recon_down.max(1)).max(2);
        let heads = NicheHeads {
            embed_l1: LinearConfig::new(flat, self.mlp_hidden).init(device),
            embed_l2: LinearConfig::new(self.mlp_hidden, self.embedding_dim).init(device),
            func_l1: LinearConfig::new(self.embedding_dim, self.mlp_hidden).init(device),
            func_l2: LinearConfig::new(self.mlp_hidden, self.n_programs).init(device),
            recon_l1: LinearConfig::new(self.embedding_dim, self.mlp_hidden).init(device),
            recon_l2: LinearConfig::new(self.mlp_hidden, recon_h * recon_w).init(device),
            proj_l1: LinearConfig::new(self.embedding_dim, self.mlp_hidden).init(device),
            proj_l2: LinearConfig::new(self.mlp_hidden, self.projection_dim).init(device),
            recon_h,
            recon_w,
        };

        NicheEncoder {
            backbone,
            heads,
            embedding_dim: self.embedding_dim,
        }
    }
}

fn prelu_block<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let device = x.device();
    burn::tensor::activation::prelu(x, Tensor::zeros([1], &device) + 0.1)
}

fn prelu_2d<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = x.device();
    burn::tensor::activation::prelu(x, Tensor::zeros([1], &device) + 0.1)
}

impl<B: Backend> NicheEncoder<B> {
    /// Compute only the embedding (no aux heads). Faster path used for k-means.
    pub fn embed(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let pooled = self.run_backbone(images);
        let h = self.heads.embed_l1.forward(pooled);
        let h = prelu_2d(h);
        self.heads.embed_l2.forward(h)
    }

    /// Full forward pass: embedding + functional + recon + projection heads.
    pub fn forward(&self, images: Tensor<B, 4>) -> NicheForward<B> {
        let pooled = self.run_backbone(images);
        let embed = self.heads.embed_l1.forward(pooled);
        let embed = prelu_2d(embed);
        let embed = self.heads.embed_l2.forward(embed);

        let f = self.heads.func_l1.forward(embed.clone());
        let f = prelu_2d(f);
        let f = self.heads.func_l2.forward(f);

        let r = self.heads.recon_l1.forward(embed.clone());
        let r = prelu_2d(r);
        let r = self.heads.recon_l2.forward(r);
        let dims = embed.dims();
        let recon = r.reshape([dims[0], 1, self.heads.recon_h, self.heads.recon_w]);

        let p = self.heads.proj_l1.forward(embed.clone());
        let p = prelu_2d(p);
        let p = self.heads.proj_l2.forward(p);

        NicheForward {
            embedding: embed,
            functional: f,
            recon,
            projection: p,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn run_backbone(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let dims = images.dims();
        let batch = dims[0];

        let x = self.backbone.conv1.forward(images);
        let x = self.backbone.bn1.forward(x);
        let x = prelu_block(x);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.backbone.conv2.forward(x);
        let x = self.backbone.bn2.forward(x);
        let x = prelu_block(x);
        let x = burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1]);

        let x = self.backbone.conv3.forward(x);
        let x = self.backbone.bn3.forward(x);
        let x = prelu_block(x);

        let x = burn::tensor::module::adaptive_avg_pool2d(
            x,
            [self.backbone.pool_h, self.backbone.pool_w],
        );
        x.reshape([
            batch,
            self.backbone.out_channels * self.backbone.pool_h * self.backbone.pool_w,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32, i32>;

    #[test]
    fn forward_shapes_are_consistent() {
        let device = Default::default();
        let cfg = NicheEncoderConfig::new(16, 24)
            .with_embedding_dim(8)
            .with_n_programs(4)
            .with_recon_down(4)
            .with_conv1_channels(8)
            .with_conv2_channels(16)
            .with_conv3_channels(16)
            .with_mlp_hidden(32)
            .with_projection_dim(4);
        let net = cfg.init::<B>(&device);
        let batch = 5;
        let x = Tensor::<B, 4>::zeros([batch, 1, 16, 24], &device);
        let out = net.forward(x);
        assert_eq!(out.embedding.dims(), [batch, 8]);
        assert_eq!(out.functional.dims(), [batch, 4]);
        assert_eq!(out.projection.dims(), [batch, 4]);
        let recon_dims = out.recon.dims();
        assert_eq!(recon_dims[0], batch);
        assert_eq!(recon_dims[1], 1);
    }
}
