//! Manual training loop for both heads (fixed / CTC).
//!
//! A hand-written loop (instead of Burn's `Learner`) because the two objectives
//! need different batching and full-captcha accuracy is measured by decoding the
//! predicted sequence. The loop also implements a quality gate: only the epoch
//! with the best validation accuracy is written to disk.

use crate::data::{Dataset, build_images, build_images_aug, build_targets};
use burn::config::Config;
use burn::module::AutodiffModule;
use burn::nn::loss::{CTCLossConfig, CrossEntropyLossConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use capburn_core::{
    Arch, CaptchaModel, CaptchaModelConfig, Charset, fixed_decode_indices, greedy_decode_indices,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: CaptchaModelConfig,
    /// Minimum label length accepted from the dataset (maximum is `model.num_chars`).
    #[config(default = 1)]
    pub min_chars: usize,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 5.0e-4)]
    pub learning_rate: f64,
    /// Apply random affine + photometric augmentation to training images.
    #[config(default = true)]
    pub augment: bool,
}

pub fn run<B: AutodiffBackend>(
    artifact_dir: &str,
    data_dir: &str,
    device: B::Device,
    config: TrainingConfig,
) {
    config
        .model
        .validate()
        .unwrap_or_else(|e| panic!("invalid model config: {e}"));
    B::seed(&device, config.seed);

    let arch = Arch::parse(&config.model.arch).expect("invalid arch");
    let charset = Charset::from_chars(&config.model.charset);
    let min_len = config.min_chars;
    let max_len = config.model.num_chars;

    // The fixed head has exactly `num_chars` output slots and cross-entropy per
    // slot, so it requires a single length. A range needs the CTC head.
    assert!(
        !(arch == Arch::Fixed && min_len != max_len),
        "--arch fixed needs a single length (--num-chars N), got {min_len}..={max_len} — use --arch ctc for variable length"
    );
    // CTC target length cannot exceed the fixed sequence length the backbone emits.
    assert!(
        !(arch == Arch::Ctc && max_len > capburn_core::CTC_TIME_STEPS),
        "--num-chars max {max_len} exceeds the CTC sequence length {} — captchas this long are not supported",
        capburn_core::CTC_TIME_STEPS
    );

    println!(
        "Arch: {}  |  charset: {} chars ({})  |  length: {}",
        arch.as_str(),
        charset.len(),
        charset.describe_families(),
        if min_len == max_len {
            format!("fixed {max_len}")
        } else {
            format!("variable {min_len}..={max_len}")
        }
    );

    print!("Loading and decoding images... ");
    let mut dataset =
        Dataset::from_folder(data_dir, &charset, min_len, max_len).expect("read dataset");
    println!("{} examples", dataset.len());
    assert!(
        dataset.len() > 1,
        "No suitable examples found in {data_dir} — check --charset and --num-chars"
    );
    dataset.shuffle(config.seed);
    let (train, valid) = dataset.split(0.9);
    println!(
        "Train: {} examples, Valid: {} examples",
        train.len(),
        valid.len()
    );

    // Save the configs only after the dataset is read and valid.
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("config save");
    config
        .model
        .save(format!("{artifact_dir}/model.json"))
        .expect("model config save");

    let mut model = config.model.init::<B>(&device);
    let blank = model.blank();
    let ctc = CTCLossConfig::new().with_blank(blank).init();
    let ce = CrossEntropyLossConfig::new().init(&device);
    let mut optim = AdamConfig::new().init();

    let mut best_acc = -1.0f32;
    for epoch in 1..=config.num_epochs {
        let mut train = train.clone();
        train.shuffle(config.seed.wrapping_add(epoch as u64));

        let mut aug_rng =
            StdRng::seed_from_u64(config.seed ^ (epoch as u64).wrapping_mul(0x9E3779B9));
        let mut running_loss = 0.0f64;
        let mut batches = 0usize;
        for batch in train.examples().chunks(config.batch_size) {
            let images = if config.augment {
                build_images_aug::<B>(batch, &device, &mut aug_rng)
            } else {
                build_images::<B>(batch, &device)
            };
            let (targets, target_lengths) = build_targets::<B>(batch, &device);

            let loss = match arch {
                Arch::Fixed => {
                    // All labels have length num_chars, so targets is [B, N].
                    let logits = model.forward_fixed(images); // [B, N, C]
                    let [b, n, c] = logits.dims();
                    let logits_flat = logits.reshape([b * n, c]);
                    let targets_flat = targets.reshape([b * n]);
                    ce.forward(logits_flat, targets_flat)
                }
                Arch::Ctc => {
                    let log_probs = model.forward_ctc(images); // [T, B, C]
                    let time = log_probs.dims()[0];
                    let input_lengths = Tensor::<B, 1, Int>::from_data(
                        TensorData::new(vec![time as i64; batch.len()], [batch.len()]),
                        &device,
                    );
                    ctc.forward(log_probs, targets, input_lengths, target_lengths)
                        .mean()
                }
            };

            let loss_value: f32 = loss.clone().into_scalar().elem();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);

            running_loss += loss_value as f64;
            batches += 1;
        }

        let (acc, char_acc) = evaluate::<B>(&model, &valid, config.batch_size, &device);
        let avg_loss = running_loss / batches.max(1) as f64;
        let flag = if acc > best_acc { " *best" } else { "" };
        println!(
            "Epoch {epoch:>3}/{}  loss {avg_loss:.4}  val char-acc {char_acc:.1}%  full-captcha {acc:.2}%{flag}",
            config.num_epochs
        );
        // Flush so headless logs (redirected stdout is block-buffered) stay live.
        let _ = std::io::Write::flush(&mut std::io::stdout());

        if acc > best_acc {
            best_acc = acc;
            model
                .valid()
                .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
                .expect("save model");
        }
    }

    println!(
        "Done. Best validation accuracy: {best_acc:.2}%. Model saved to {artifact_dir}/model.mpk (+ model.json)"
    );
}

/// Full-captcha accuracy: fraction of validation examples whose decoded sequence
/// exactly matches the label. Runs on the inner (eval) backend so BatchNorm uses
/// running stats and dropout is disabled.
fn evaluate<B: AutodiffBackend>(
    model: &CaptchaModel<B>,
    valid: &Dataset,
    batch_size: usize,
    device: &B::Device,
) -> (f32, f32) {
    let eval_model = model.valid();
    let arch = model.arch();
    let blank = model.blank();
    let mut full_correct = 0usize;
    let mut total = 0usize;
    let mut char_correct = 0usize;
    let mut char_total = 0usize;
    for batch in valid.examples().chunks(batch_size) {
        let images = build_images::<B::InnerBackend>(batch, device);
        let decoded = match arch {
            Arch::Fixed => fixed_decode_indices(eval_model.forward_fixed(images)),
            Arch::Ctc => greedy_decode_indices(eval_model.forward_ctc(images), blank),
        };
        for (ex, seq) in batch.iter().zip(decoded) {
            if seq == ex.labels {
                full_correct += 1;
            }
            total += 1;
            // Per-character accuracy (position-aligned up to the shorter length).
            for (a, b) in seq.iter().zip(ex.labels.iter()) {
                if a == b {
                    char_correct += 1;
                }
            }
            char_total += ex.labels.len();
        }
    }
    let full = if total == 0 {
        0.0
    } else {
        full_correct as f32 / total as f32 * 100.0
    };
    let chars = if char_total == 0 {
        0.0
    } else {
        char_correct as f32 / char_total as f32 * 100.0
    };
    (full, chars)
}
