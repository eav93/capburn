//! Training dataset: labels from file names, images preloaded into memory,
//! manual batching into tensors for the CTC training loop.

use burn::prelude::*;
use capburn_core::Charset;
use capburn_core::image_ops::{
    IMG_HEIGHT, IMG_WIDTH, PreprocessMode, load_image_as_floats_with_mode,
};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::path::Path;

/// One preloaded example: grayscale image floats and label class indices.
#[derive(Clone)]
pub struct Example {
    pub image: Vec<f32>,
    pub labels: Vec<usize>,
}

#[derive(Clone)]
pub struct Dataset {
    examples: Vec<Example>,
}

/// Training augmentation strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AugmentProfile {
    Light,
    Medium,
    Strong,
}

impl AugmentProfile {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "light" => Ok(Self::Light),
            "medium" => Ok(Self::Medium),
            "strong" => Ok(Self::Strong),
            other => Err(format!(
                "unknown augment profile {other:?} (expected 'light', 'medium', 'strong' or 'off')"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Medium => "medium",
            Self::Strong => "strong",
        }
    }

    fn params(self) -> AugmentParams {
        match self {
            Self::Light => AugmentParams {
                max_angle: 0.06,
                min_scale: 0.95,
                max_scale: 1.05,
                max_tx: 2.0,
                max_ty: 1.0,
                min_contrast: 0.9,
                max_contrast: 1.1,
                max_brightness: 0.06,
                max_noise: 0.015,
            },
            Self::Medium => AugmentParams {
                max_angle: 0.10,
                min_scale: 0.9,
                max_scale: 1.1,
                max_tx: 3.0,
                max_ty: 2.0,
                min_contrast: 0.8,
                max_contrast: 1.2,
                max_brightness: 0.1,
                max_noise: 0.05,
            },
            Self::Strong => AugmentParams {
                max_angle: 0.14,
                min_scale: 0.85,
                max_scale: 1.15,
                max_tx: 5.0,
                max_ty: 3.0,
                min_contrast: 0.75,
                max_contrast: 1.25,
                max_brightness: 0.12,
                max_noise: 0.07,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct AugmentParams {
    max_angle: f32,
    min_scale: f32,
    max_scale: f32,
    max_tx: f32,
    max_ty: f32,
    min_contrast: f32,
    max_contrast: f32,
    max_brightness: f32,
    max_noise: f32,
}

/// Extract the label from a file name — the part before the first `_`
/// (e.g. `12345_abcdef.png` → `12345`).
pub fn label_from_stem(stem: &str) -> &str {
    stem.split('_').next().unwrap_or(stem)
}

impl Dataset {
    /// Load a dataset from a folder, decoding all images into memory. Only files
    /// whose label length is in `min_len..=max_len` and whose characters are all
    /// in `charset` are included.
    pub fn from_folder<P: AsRef<Path>>(
        folder: P,
        charset: &Charset,
        min_len: usize,
        max_len: usize,
        preprocess: PreprocessMode,
    ) -> std::io::Result<Self> {
        let mut examples = Vec::new();
        let mut skipped = 0usize;
        let mut decode_failed = 0usize;
        for entry in std::fs::read_dir(folder)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let label_part = label_from_stem(stem);
            let len = label_part.chars().count();
            if len < min_len || len > max_len {
                skipped += 1;
                continue;
            }
            let mut labels = Vec::with_capacity(len);
            let mut ok = true;
            for c in label_part.chars() {
                match charset.index_of(c) {
                    Some(idx) => labels.push(idx),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                skipped += 1;
                continue;
            }
            // Skip unreadable/corrupt files (e.g. a `12345.txt` whose stem looks
            // like a label) instead of killing the whole training run.
            let image = match load_image_as_floats_with_mode(&path, preprocess) {
                Ok(img) => img,
                Err(e) => {
                    decode_failed += 1;
                    if decode_failed <= 5 {
                        eprintln!("Skipping {}: {e}", path.display());
                    }
                    continue;
                }
            };
            examples.push(Example { image, labels });
        }
        if skipped > 0 {
            println!(
                "Skipped {skipped} files (label length outside {min_len}..={max_len} or a character outside the charset)"
            );
        }
        if decode_failed > 0 {
            println!("Skipped {decode_failed} files that could not be decoded as images");
        }
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn examples(&self) -> &[Example] {
        &self.examples
    }

    /// Deterministic seeded shuffle — reproducible train/valid split and epochs.
    pub fn shuffle(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        self.examples.shuffle(&mut rng);
    }

    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let len = self.examples.len();
        let n_train = ((len as f32) * train_ratio).round() as usize;
        // Keep at least one example on each side when the dataset has ≥2, so the
        // validation split (and the quality gate) is never empty.
        let n_train = if len >= 2 {
            n_train.clamp(1, len - 1)
        } else {
            n_train.min(len)
        };
        let (train, val) = self.examples.split_at(n_train);
        (
            Self {
                examples: train.to_vec(),
            },
            Self {
                examples: val.to_vec(),
            },
        )
    }
}

/// Build the image tensor `[B, 1, H, W]` for a slice of examples.
pub fn build_images<B: Backend>(batch: &[Example], device: &B::Device) -> Tensor<B, 4> {
    let bsz = batch.len();
    let mut data = Vec::with_capacity(bsz * IMG_HEIGHT * IMG_WIDTH);
    for ex in batch {
        data.extend_from_slice(&ex.image);
    }
    Tensor::from_data(
        TensorData::new(data, [bsz, 1, IMG_HEIGHT, IMG_WIDTH]),
        device,
    )
}

/// Build an augmented image tensor for training — random affine warp
/// (rotation/scale/translation) plus brightness/contrast/noise, applied per
/// image. Regularizes training on small captcha datasets.
pub fn build_images_aug<B: Backend>(
    batch: &[Example],
    device: &B::Device,
    rng: &mut StdRng,
    profile: AugmentProfile,
) -> Tensor<B, 4> {
    let bsz = batch.len();
    let mut data = Vec::with_capacity(bsz * IMG_HEIGHT * IMG_WIDTH);
    for ex in batch {
        augment_into(&ex.image, &mut data, rng, profile);
    }
    Tensor::from_data(
        TensorData::new(data, [bsz, 1, IMG_HEIGHT, IMG_WIDTH]),
        device,
    )
}

/// Apply a random affine warp and photometric jitter to one grayscale image
/// (`IMG_HEIGHT * IMG_WIDTH` floats), appending the result to `out`. Border
/// pixels are replicated, so no fixed background color is assumed.
fn augment_into(image: &[f32], out: &mut Vec<f32>, rng: &mut StdRng, profile: AugmentProfile) {
    let params = profile.params();
    let (h, w) = (IMG_HEIGHT as f32, IMG_WIDTH as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);

    let angle = rng.random_range(-params.max_angle..params.max_angle);
    let scale = rng.random_range(params.min_scale..params.max_scale);
    // Keep horizontal translation small: the fixed head assigns each width slot
    // to a position, so large shifts would move characters across slot borders.
    let tx = rng.random_range(-params.max_tx..params.max_tx);
    let ty = rng.random_range(-params.max_ty..params.max_ty);
    let (sin, cos) = angle.sin_cos();

    let contrast = rng.random_range(params.min_contrast..params.max_contrast);
    let brightness = rng.random_range(-params.max_brightness..params.max_brightness);
    let noise = rng.random_range(0.0f32..params.max_noise);

    out.reserve(IMG_HEIGHT * IMG_WIDTH);
    for oy in 0..IMG_HEIGHT {
        for ox in 0..IMG_WIDTH {
            // Map output pixel back to the source (inverse rotation + scale).
            let dx = ox as f32 - cx - tx;
            let dy = oy as f32 - cy - ty;
            let sx = (cos * dx + sin * dy) / scale + cx;
            let sy = (-sin * dx + cos * dy) / scale + cy;
            // Nearest neighbour with border replication.
            let ix = (sx.round() as i32).clamp(0, IMG_WIDTH as i32 - 1) as usize;
            let iy = (sy.round() as i32).clamp(0, IMG_HEIGHT as i32 - 1) as usize;
            let mut v = image[iy * IMG_WIDTH + ix];
            // Photometric jitter.
            v = (v - 0.5) * contrast + 0.5 + brightness;
            if noise > 0.0 {
                v += rng.random_range(-noise..noise);
            }
            // Pixels are produced in row-major order, so pushing keeps layout.
            out.push(v.clamp(0.0, 1.0));
        }
    }
}

/// Build the CTC target tensors: padded targets `[B, S]` and target lengths
/// `[B]`. Padding value is 0 (ignored via the per-sample lengths).
pub fn build_targets<B: Backend>(
    batch: &[Example],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let bsz = batch.len();
    let max_len = batch
        .iter()
        .map(|e| e.labels.len())
        .max()
        .unwrap_or(1)
        .max(1);

    let mut target_data = vec![0i64; bsz * max_len];
    let mut lengths = Vec::with_capacity(bsz);
    for (i, ex) in batch.iter().enumerate() {
        for (j, &idx) in ex.labels.iter().enumerate() {
            target_data[i * max_len + j] = idx as i64;
        }
        lengths.push(ex.labels.len() as i64);
    }

    let targets = Tensor::from_data(TensorData::new(target_data, [bsz, max_len]), device);
    let target_lengths = Tensor::from_data(TensorData::new(lengths, [bsz]), device);
    (targets, target_lengths)
}

#[cfg(test)]
mod tests {
    use super::label_from_stem;

    #[test]
    fn label_before_underscore() {
        assert_eq!(label_from_stem("12345"), "12345");
        assert_eq!(label_from_stem("12345_a1b2c3"), "12345");
        assert_eq!(label_from_stem("абв_hash"), "абв");
        assert_eq!(label_from_stem("_x"), "");
    }
}
