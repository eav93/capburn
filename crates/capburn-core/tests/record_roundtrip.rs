//! Save/load roundtrip for every architecture, plus forward-shape checks.

use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use capburn_core::{CaptchaModelConfig, IMG_HEIGHT, IMG_WIDTH};

type B = burn::backend::NdArray<f32, i64>;

fn dummy_input(device: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B, 4> {
    Tensor::zeros([2, 1, IMG_HEIGHT, IMG_WIDTH], device)
}

#[test]
fn roundtrip_all_archs() {
    let device = Default::default();
    let dir = std::env::temp_dir().join("capburn-roundtrip-test");
    std::fs::create_dir_all(&dir).unwrap();

    for arch in [
        "fixed",
        "fixed-global",
        "fixed-seq",
        "fixed-seq-pool",
        "ctc",
    ] {
        let config = CaptchaModelConfig::new("0123456789".into(), 5).with_arch(arch.to_string());
        config.validate().unwrap();
        let model = config.init::<B>(&device);

        // Forward must produce [B, N, C] for fixed heads, [T, B, C] for ctc.
        if arch == "ctc" {
            let out = model.forward_ctc(dummy_input(&device));
            assert_eq!(out.dims()[1], 2, "{arch}: batch dim");
        } else {
            let out = model.forward_fixed_logits(dummy_input(&device));
            assert_eq!(out.dims(), [2, 5, 10], "{arch}: logits shape");
        }

        let path = dir.join(format!("model-{arch}"));
        model
            .clone()
            .save_file(path.clone(), &CompactRecorder::new())
            .unwrap();
        let loaded = config.init::<B>(&device).load_record(
            burn::record::Recorder::<B>::load(&CompactRecorder::new(), path, &device)
                .unwrap_or_else(|e| panic!("{arch}: load failed: {e}")),
        );
        // Loaded model must still run forward.
        if arch == "ctc" {
            loaded.forward_ctc(dummy_input(&device));
        } else {
            loaded.forward_fixed_logits(dummy_input(&device));
        }
    }
}
