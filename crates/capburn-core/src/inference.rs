//! Load a trained model and recognize captchas. Backend-independent: PHP uses
//! `NdArray` (CPU), the trainer can also verify on its own backend.

use crate::charset::Charset;
use crate::image_ops::{
    InputSize, PreprocessMode, image_to_floats_with_mode_and_size,
    load_image_from_bytes_with_mode_and_size,
};
use crate::model::CaptchaModel;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;

/// A ready-to-use recognizer: model, charset and character count.
pub struct Recognizer<B: Backend> {
    model: CaptchaModel<B>,
    charset: Charset,
    num_chars: usize,
    preprocess: PreprocessMode,
    input_size: InputSize,
    device: B::Device,
}

impl<B: Backend> Recognizer<B> {
    /// Load a model from an artifacts folder. Expects `model.json` (config with
    /// the charset) and `model.mpk` (weights, CompactRecorder format).
    pub fn load<P: AsRef<Path>>(artifact_dir: P, device: B::Device) -> Result<Self, String> {
        let dir = artifact_dir.as_ref();
        let cfg_path = dir.join("model.json");
        // Lenient load: fills defaults for fields added in newer versions, and
        // rejects a model newer than this build supports with a clear message.
        let config = crate::model::load_model_config(&cfg_path)?;
        config
            .validate()
            .map_err(|e| format!("invalid model config in {}: {e}", cfg_path.display()))?;

        let charset = Charset::from_chars(&config.charset);
        let num_chars = config.num_chars;
        let preprocess = PreprocessMode::parse(&config.preprocess)
            .map_err(|e| format!("invalid model preprocess in {}: {e}", cfg_path.display()))?;
        let input_size = config.input_size();

        let model: CaptchaModel<B> = config.init(&device);
        let record = CompactRecorder::new()
            .load(dir.join("model"), &device)
            .map_err(|e| format!("cannot load model weights: {e}"))?;
        let model = model.load_record(record);

        Ok(Self {
            model,
            charset,
            num_chars,
            preprocess,
            input_size,
            device,
        })
    }

    /// Number of characters the model outputs.
    pub fn num_chars(&self) -> usize {
        self.num_chars
    }

    fn recognize_floats(&self, data: Vec<f32>) -> String {
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(
                data,
                [1usize, 1, self.input_size.height, self.input_size.width],
            ),
            &self.device,
        );
        let decoded = match self.model.arch() {
            crate::Arch::Fixed
            | crate::Arch::FixedGlobal
            | crate::Arch::FixedSeq
            | crate::Arch::FixedSeqPool => {
                let logits = self.model.forward_fixed_logits(input); // [1, N, C]
                crate::model::fixed_decode_indices(logits)
            }
            crate::Arch::Ctc => {
                let log_probs = self.model.forward_ctc(input); // [T, 1, C]
                crate::model::greedy_decode_indices(log_probs, self.model.blank())
            }
        };
        // Single image → single sequence. try_char_at drops any out-of-range
        // index rather than crashing the process (matters for the PHP extension).
        decoded
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|i| self.charset.try_char_at(i))
            .collect()
    }

    /// Recognize a captcha from a file.
    pub fn recognize_path<P: AsRef<Path>>(&self, path: P) -> Result<String, String> {
        let data = crate::image_ops::load_image_as_floats_with_mode_and_size(
            path,
            self.preprocess,
            self.input_size,
        )?;
        Ok(self.recognize_floats(data))
    }

    /// Recognize a captcha from in-memory image bytes (PNG/JPEG/…).
    pub fn recognize_bytes(&self, bytes: &[u8]) -> Result<String, String> {
        let data =
            load_image_from_bytes_with_mode_and_size(bytes, self.preprocess, self.input_size)?;
        Ok(self.recognize_floats(data))
    }

    /// Recognize an already-decoded image.
    pub fn recognize_image(&self, img: &image::DynamicImage) -> String {
        self.recognize_floats(image_to_floats_with_mode_and_size(
            img,
            self.preprocess,
            self.input_size,
        ))
    }
}

/// CPU backend (ndarray) — used by the PHP extension and the trainer's infer.
pub type CpuBackend = burn::backend::NdArray<f32, i64>;
/// CPU recognizer.
pub type CpuRecognizer = Recognizer<CpuBackend>;

impl Recognizer<CpuBackend> {
    /// Load a model for CPU inference. Hides the backend choice so callers (e.g.
    /// the PHP extension) do not need to know about Burn.
    pub fn load_cpu<P: AsRef<Path>>(artifact_dir: P) -> Result<Self, String> {
        Self::load(artifact_dir, burn::backend::ndarray::NdArrayDevice::Cpu)
    }
}
