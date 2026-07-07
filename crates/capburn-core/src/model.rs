//! Captcha recognizer with a shared CNN backbone and two selectable heads.
//!
//! * `fixed` — positional classifier: the CNN feature map is pooled into exactly
//!   `num_chars` width slots and each slot is classified independently with
//!   cross-entropy. Best for fixed-length captchas — trains stably, handles
//!   repeated characters (e.g. `00075`), and is fast at inference.
//! * `ctc`   — a per-width-column classifier trained with CTC loss. Handles
//!   variable length and shifting character positions without per-slot labels.
//!
//! Both heads share the same convolutional backbone. No RNN is used: recurrent
//! layers are prohibitively slow on the fusion-less wgpu backend used for
//! headless training.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear,
    LinearConfig, PaddingConfig2d,
};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;

/// Feature channels produced by the CNN backbone.
const CNN_OUT: usize = 128;

/// Recognition head / training objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Positional classifier + cross-entropy (fixed length).
    Fixed,
    /// Per-column classifier + CTC loss (variable length).
    Ctc,
}

impl Arch {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "fixed" => Ok(Arch::Fixed),
            "ctc" => Ok(Arch::Ctc),
            other => Err(format!(
                "unknown arch {other:?} (expected 'fixed' or 'ctc')"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Arch::Fixed => "fixed",
            Arch::Ctc => "ctc",
        }
    }
}

#[derive(Config, Debug)]
pub struct CaptchaModelConfig {
    /// Expanded charset string. Class indices `0..len` map to these characters.
    pub charset: String,
    /// Number of characters: the fixed length (`fixed`), or the maximum length
    /// (`ctc`, informational + dataset filtering).
    pub num_chars: usize,
    /// Recognition head: `"fixed"` or `"ctc"`.
    #[config(default = "String::from(\"fixed\")")]
    pub arch: String,
    #[config(default = 0.2)]
    pub dropout: f64,
}

/// A conv + batch-norm block; pooling is applied separately.
#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ConvBlock<B> {
    fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn: BatchNormConfig::new(out_ch).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>, act: &LeakyRelu) -> Tensor<B, 4> {
        act.forward(self.bn.forward(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct CaptchaModel<B: Backend> {
    b1: ConvBlock<B>,
    b2: ConvBlock<B>,
    b3: ConvBlock<B>,
    b4: ConvBlock<B>,
    b5: ConvBlock<B>,
    pool2x2: MaxPool2d,
    pool2x1: MaxPool2d,
    activation: LeakyRelu,
    /// Pools the width into exactly `num_chars` slots (fixed head only).
    slot_pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc: Linear<B>,
    #[module(skip)]
    arch: Arch,
    #[module(skip)]
    num_chars: usize,
    #[module(skip)]
    num_classes: usize,
}

impl CaptchaModelConfig {
    /// Validate the config: known arch, non-empty charset with no duplicates,
    /// positive length, dropout in range.
    pub fn validate(&self) -> Result<(), String> {
        Arch::parse(&self.arch)?;
        if self.num_chars == 0 {
            return Err("num_chars must be > 0".into());
        }
        let chars: Vec<char> = self.charset.chars().collect();
        if chars.is_empty() {
            return Err("charset is empty".into());
        }
        let mut seen = std::collections::HashSet::new();
        for c in &chars {
            if !seen.insert(*c) {
                return Err(format!("charset contains duplicate character {c:?}"));
            }
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(format!("dropout must be in [0, 1), got {}", self.dropout));
        }
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CaptchaModel<B> {
        let arch = Arch::parse(&self.arch).expect("invalid arch");
        let charset_len = self.charset.chars().count();
        // CTC needs an extra blank class (last index); the fixed head does not.
        let num_classes = match arch {
            Arch::Fixed => charset_len,
            Arch::Ctc => charset_len + 1,
        };
        CaptchaModel {
            b1: ConvBlock::new(1, 32, device),
            b2: ConvBlock::new(32, 64, device),
            b3: ConvBlock::new(64, CNN_OUT, device),
            b4: ConvBlock::new(CNN_OUT, CNN_OUT, device),
            b5: ConvBlock::new(CNN_OUT, CNN_OUT, device),
            pool2x2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            pool2x1: MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init(),
            activation: LeakyReluConfig::new().init(),
            slot_pool: AdaptiveAvgPool2dConfig::new([1, self.num_chars]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            fc: LinearConfig::new(CNN_OUT, num_classes).init(device),
            arch,
            num_chars: self.num_chars,
            num_classes,
        }
    }
}

impl<B: Backend> CaptchaModel<B> {
    pub fn arch(&self) -> Arch {
        self.arch
    }

    pub fn num_chars(&self) -> usize {
        self.num_chars
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// CTC blank class index (last class). Only meaningful for `Arch::Ctc`.
    pub fn blank(&self) -> usize {
        self.num_classes - 1
    }

    /// Shared CNN backbone. Input `[B, 1, 32, 128]` → `[B, CNN_OUT, 1, 32]`
    /// where the height is collapsed to 1 and the width becomes the sequence.
    fn backbone(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.b1.forward(x, &self.activation);
        let x = self.pool2x2.forward(x); // [B, 32, 16, 64]
        let x = self.b2.forward(x, &self.activation);
        let x = self.pool2x2.forward(x); // [B, 64, 8, 32]
        let x = self.b3.forward(x, &self.activation);
        let x = self.pool2x1.forward(x); // [B, 128, 4, 32]
        let x = self.b4.forward(x, &self.activation);
        let x = self.pool2x1.forward(x); // [B, 128, 2, 32]
        let x = self.b5.forward(x, &self.activation);
        self.pool2x1.forward(x) // [B, 128, 1, 32]
    }

    /// Fixed head: logits of shape `[B, num_chars, num_classes]` (no softmax;
    /// cross-entropy applies it).
    pub fn forward_fixed(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let feat = self.backbone(x); // [B, C, 1, W']
        let pooled = self.slot_pool.forward(feat); // [B, C, 1, num_chars]
        let [batch, channels, _h, slots] = pooled.dims();
        // [B, C, 1, N] → [B, C, N] → [B, N, C]
        let seq = pooled.reshape([batch, channels, slots]).swap_dims(1, 2);
        let seq = self.dropout.forward(seq);
        self.fc.forward(seq) // [B, N, num_classes]
    }

    /// CTC head: log-probabilities of shape `[T, B, num_classes]` (time-major).
    pub fn forward_ctc(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let feat = self.backbone(x); // [B, C, 1, W']
        let [batch, channels, _h, width] = feat.dims();
        let seq = feat.reshape([batch, channels, width]).swap_dims(1, 2); // [B, W', C]
        let seq = self.dropout.forward(seq);
        let logits = self.fc.forward(seq); // [B, W', num_classes]
        log_softmax(logits, 2).swap_dims(0, 1) // [T, B, num_classes]
    }
}

/// Greedy CTC decode: per batch element, collapse repeated classes and drop the
/// blank, returning the predicted class-index sequence.
pub fn greedy_decode_indices<B: Backend>(log_probs: Tensor<B, 3>, blank: usize) -> Vec<Vec<usize>> {
    let [time, batch, _classes] = log_probs.dims();
    // Read via float so the code is backend-agnostic (wgpu Int is i32,
    // ndarray Int is i64); class indices are small and exact in f32.
    let flat: Vec<f32> = log_probs
        .argmax(2)
        .float()
        .into_data()
        .to_vec()
        .expect("convert argmax to vec");

    let mut out = vec![Vec::new(); batch];
    for (bi, seq) in out.iter_mut().enumerate() {
        let mut prev = usize::MAX;
        for ti in 0..time {
            // Row-major [T, B, 1] → element (ti, bi) is at ti * batch + bi.
            let cls = flat[ti * batch + bi] as usize;
            if cls != prev && cls != blank {
                seq.push(cls);
            }
            prev = cls;
        }
    }
    out
}

/// Argmax decode of the fixed head: `[B, N, num_classes]` logits → per-example
/// class-index sequence of length `N`.
pub fn fixed_decode_indices<B: Backend>(logits: Tensor<B, 3>) -> Vec<Vec<usize>> {
    let [batch, slots, _classes] = logits.dims();
    let flat: Vec<f32> = logits
        .argmax(2)
        .float()
        .into_data()
        .to_vec()
        .expect("convert argmax to vec");
    let mut out = vec![Vec::with_capacity(slots); batch];
    for (bi, seq) in out.iter_mut().enumerate() {
        for si in 0..slots {
            seq.push(flat[bi * slots + si] as usize);
        }
    }
    out
}
