//! Proves the fixed-seq / fixed-seq-pool heads are trainable end-to-end: a few
//! Adam steps on a tiny fixed batch must drive the cross-entropy loss down,
//! which can only happen if gradients flow through the temporal blocks.

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use capburn_core::{CaptchaModelConfig, IMG_HEIGHT, IMG_WIDTH};

type AB = Autodiff<NdArray<f32, i64>>;

const BATCH: usize = 6;
const NUM_CHARS: usize = 5;

/// Deterministic pseudo-random image batch — distinct per sample so the model
/// can memorize the mapping. No RNG (tests must stay reproducible).
fn synthetic_images(device: &burn::backend::ndarray::NdArrayDevice) -> Tensor<AB, 4> {
    let mut data = Vec::with_capacity(BATCH * IMG_HEIGHT * IMG_WIDTH);
    for b in 0..BATCH {
        for p in 0..(IMG_HEIGHT * IMG_WIDTH) {
            // A cheap hash-like value in [0, 1), varying by sample and pixel.
            let v = (((b * 2_654_435_761 + p * 40_503) % 997) as f32) / 997.0;
            data.push(v);
        }
    }
    Tensor::from_data(
        TensorData::new(data, [BATCH, 1, IMG_HEIGHT, IMG_WIDTH]),
        device,
    )
}

/// Sample `i` gets the constant label `i % 10` repeated across all positions.
fn synthetic_targets(device: &burn::backend::ndarray::NdArrayDevice) -> Tensor<AB, 2, Int> {
    let mut data = Vec::with_capacity(BATCH * NUM_CHARS);
    for b in 0..BATCH {
        for _ in 0..NUM_CHARS {
            data.push((b % 10) as i64);
        }
    }
    Tensor::from_data(TensorData::new(data, [BATCH, NUM_CHARS]), device)
}

fn overfits(arch: &str) -> (f32, f32) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    AB::seed(&device, 7);
    let config =
        CaptchaModelConfig::new("0123456789".into(), NUM_CHARS).with_arch(arch.to_string());
    config.validate().unwrap();
    let mut model = config.init::<AB>(&device);

    let images = synthetic_images(&device);
    let targets = synthetic_targets(&device);
    let ce = CrossEntropyLossConfig::new().init(&device);
    let mut optim = AdamConfig::new().init();

    let loss_value = |model: &_| -> f32 {
        let logits: Tensor<AB, 3> =
            capburn_core::CaptchaModel::<AB>::forward_fixed_logits(model, images.clone());
        let [b, n, c] = logits.dims();
        let loss = ce.forward(logits.reshape([b * n, c]), targets.clone().reshape([b * n]));
        loss.into_scalar()
    };

    let initial = loss_value(&model);
    for _ in 0..80 {
        let logits = model.forward_fixed_logits(images.clone());
        let [b, n, c] = logits.dims();
        let loss = ce.forward(logits.reshape([b * n, c]), targets.clone().reshape([b * n]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }
    let final_loss = loss_value(&model);
    (initial, final_loss)
}

#[test]
fn fixed_seq_is_trainable() {
    let (initial, final_loss) = overfits("fixed-seq");
    assert!(
        initial.is_finite() && final_loss.is_finite(),
        "loss went NaN"
    );
    assert!(
        final_loss < initial * 0.5,
        "fixed-seq did not learn: loss {initial:.4} -> {final_loss:.4}"
    );
}

#[test]
fn fixed_seq_pool_is_trainable() {
    let (initial, final_loss) = overfits("fixed-seq-pool");
    assert!(
        initial.is_finite() && final_loss.is_finite(),
        "loss went NaN"
    );
    assert!(
        final_loss < initial * 0.5,
        "fixed-seq-pool did not learn: loss {initial:.4} -> {final_loss:.4}"
    );
}
