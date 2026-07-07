//! Core of the captcha recognizer.
//!
//! This crate does not depend on a specific Burn backend: the trainer uses it
//! with `Autodiff<Wgpu>` (or CUDA/CPU), and the PHP extension uses it with
//! `NdArray` (CPU). The network definition, charset and image preprocessing are
//! shared.

pub mod charset;
pub mod image_ops;
pub mod inference;
pub mod model;

pub use charset::Charset;
pub use image_ops::{
    IMG_HEIGHT, IMG_WIDTH, image_to_floats, load_image_as_floats, load_image_from_bytes,
};
pub use inference::{CpuBackend, CpuRecognizer, Recognizer};
pub use model::{
    Arch, CTC_TIME_STEPS, CaptchaModel, CaptchaModelConfig, fixed_decode_indices,
    greedy_decode_indices,
};
