use burn::backend::Autodiff;
use capburn_core::{Arch, CTC_TIME_STEPS, CaptchaModelConfig, Charset, PreprocessMode, Recognizer};
use clap::{Parser, Subcommand, ValueEnum};

mod data;
mod detect;
mod training;

use data::AugmentProfile;
use training::TrainingConfig;

#[derive(Parser)]
#[command(name = "capburn", about = "Train and run a captcha recognizer on Burn")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    /// GPU via wgpu: Metal (macOS), Vulkan/DX12 (Linux/Windows). Default.
    Wgpu,
    /// CPU (ndarray). Slow, but works without a GPU.
    Cpu,
    /// NVIDIA CUDA. Available only when built with `--features cuda`.
    Cuda,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train a model: scans a folder for images, reads labels from file names.
    Train {
        /// Root folder containing datasets.
        #[arg(long, default_value = "./data")]
        data: String,
        /// Dataset subfolder name inside `--data` (e.g. `digits5`).
        /// If omitted, `--data` itself is used.
        #[arg(long)]
        dataset: Option<String>,
        /// Where to save the model and training logs.
        #[arg(long, default_value = "./artifacts")]
        out: String,
        /// Charset: `auto` (default, detected from the dataset),
        /// `digits`, `lower`, `upper`, `letters`, `cyrillic`, `cyrillic_upper`,
        /// a `+`-combination (`cyrillic+digits`), or an explicit list of
        /// characters (`ABCDEF0123456789`).
        #[arg(long, default_value = "auto")]
        charset: String,
        /// Captcha length: a fixed number (`5`) or a range for variable length
        /// (`4-7`). Detected from the dataset if omitted.
        #[arg(long)]
        num_chars: Option<String>,
        /// Recognition head: `auto` (fixed for a single length, ctc for a
        /// range), `fixed`, `fixed-global`, or `ctc`.
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Image preprocessing: `auto`, `stretch` (resize exactly), or `fit`
        /// (preserve aspect ratio and pad).
        #[arg(long, default_value = "auto")]
        preprocess: String,
        /// Compute backend.
        #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
        backend: BackendKind,
        #[arg(long, default_value_t = 30)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 5.0e-4)]
        lr: f64,
        /// Dropout probability used by the recognition head.
        #[arg(long, default_value_t = 0.2)]
        dropout: f64,
        /// Augmentation strength: `auto`, `light`, `medium`, `strong`, or `off`.
        #[arg(long, default_value = "auto")]
        augment: String,
        /// Disable training-image augmentation (same as `--augment off`).
        #[arg(long)]
        no_augment: bool,
    },
    /// Recognize a single captcha (CPU inference).
    Infer {
        image: String,
        #[arg(long, default_value = "./artifacts")]
        artifacts: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Train {
            data,
            dataset,
            out,
            charset,
            num_chars,
            arch,
            preprocess,
            backend,
            epochs,
            batch_size,
            lr,
            dropout,
            augment,
            no_augment,
        } => {
            let data_dir = match &dataset {
                Some(name) => format!("{data}/{name}"),
                None => data.clone(),
            };

            // Clean CLI validation (no panic/backtrace).
            let fail = |msg: String| -> ! {
                eprintln!("Error: {msg}");
                std::process::exit(1);
            };

            let charset_override = if charset.eq_ignore_ascii_case("auto") {
                None
            } else {
                Some(Charset::from_spec(&charset).expect("invalid charset spec"))
            };
            let length_override = num_chars
                .as_deref()
                .map(|s| detect::parse_length_spec(s).expect("invalid --num-chars"));

            let (charset, length) =
                detect::detect_format(&data_dir, charset_override, length_override)
                    .expect("cannot read dataset folder");

            let image_scan = if preprocess.eq_ignore_ascii_case("auto") {
                let scan = detect::scan_images(&data_dir).expect("cannot scan dataset images");
                print!(
                    "Image scan: sampled {}/{}, {} decoded",
                    scan.sampled, scan.total, scan.decoded
                );
                if scan.decode_failed > 0 {
                    print!(", {} decode failed", scan.decode_failed);
                }
                print!(", dims:");
                for ((w, h), count) in scan.dim_hist.iter().take(5) {
                    print!(" {w}x{h}×{count}");
                }
                if scan.dim_hist.len() > 5 {
                    print!(" …");
                }
                println!(
                    ", mode {}x{}×{}, aspect {:.2}, color {:.1}%, ink {:.1}%",
                    scan.mode_dim.0,
                    scan.mode_dim.1,
                    scan.mode_dim_count,
                    scan.avg_aspect,
                    scan.color_fraction * 100.0,
                    scan.ink_fraction * 100.0
                );
                if scan.decode_failed > 0 {
                    let fail_ratio = scan.decode_failed as f32 / scan.sampled.max(1) as f32;
                    if fail_ratio > 0.05 {
                        eprintln!(
                            "Warning: {:.1}% of sampled images failed to decode; training will skip bad files",
                            fail_ratio * 100.0
                        );
                    }
                }
                Some(scan)
            } else {
                None
            };

            let preprocess_mode = if preprocess.eq_ignore_ascii_case("auto") {
                image_scan
                    .as_ref()
                    .map(detect::recommend_preprocess)
                    .unwrap_or(PreprocessMode::Stretch)
            } else {
                PreprocessMode::parse(&preprocess).unwrap_or_else(|e| fail(e))
            };

            // Auto: fixed length → positional head; variable length → CTC.
            // Use `--arch ctc` explicitly to A/B alignment-free training on a
            // fixed-length dataset.
            let arch = if arch.eq_ignore_ascii_case("auto") {
                if length.min_chars != length.max_chars {
                    "ctc".to_string()
                } else {
                    "fixed".to_string()
                }
            } else {
                arch
            };

            let (augment_enabled, augment_profile) =
                if no_augment || augment.eq_ignore_ascii_case("off") {
                    (false, AugmentProfile::Medium)
                } else if augment.eq_ignore_ascii_case("auto") {
                    (true, detect::recommend_augment())
                } else {
                    (
                        true,
                        AugmentProfile::parse(&augment).unwrap_or_else(|e| fail(e)),
                    )
                };

            println!(
                "Auto params: arch={}, preprocess={}, augment={}",
                arch,
                preprocess_mode.as_str(),
                if augment_enabled {
                    augment_profile.as_str()
                } else {
                    "off"
                }
            );

            let arch_kind = Arch::parse(&arch).unwrap_or_else(|e| fail(e));
            if epochs == 0 {
                fail("--epochs must be > 0".into());
            }
            if batch_size == 0 {
                fail("--batch-size must be > 0".into());
            }
            if !(lr.is_finite() && lr > 0.0) {
                fail(format!("--lr must be a positive finite number, got {lr}"));
            }
            if !(dropout.is_finite() && (0.0..1.0).contains(&dropout)) {
                fail(format!(
                    "--dropout must be a finite number in [0, 1), got {dropout}"
                ));
            }
            if arch_kind != Arch::Ctc && length.min_chars != length.max_chars {
                fail(format!(
                    "--arch {} needs a single length, got {}..={} — use --arch ctc for a range",
                    arch_kind.as_str(),
                    length.min_chars,
                    length.max_chars
                ));
            }
            // CTC needs a blank between adjacent equal characters, so the worst
            // case (all-repeats, e.g. "00000") needs 2*len-1 time steps.
            let ctc_worst_case_steps = length.max_chars.saturating_mul(2).saturating_sub(1);
            if arch_kind == Arch::Ctc && ctc_worst_case_steps > CTC_TIME_STEPS {
                fail(format!(
                    "--num-chars max {} is too long for the CTC sequence length {} (needs 2*len-1 ≤ {})",
                    length.max_chars, CTC_TIME_STEPS, CTC_TIME_STEPS
                ));
            }

            let model_cfg = CaptchaModelConfig::new(charset.as_chars(), length.max_chars)
                .with_arch(arch)
                .with_preprocess(preprocess_mode.as_str().to_string())
                .with_dropout(dropout);
            let cfg = TrainingConfig::new(model_cfg)
                .with_min_chars(length.min_chars)
                .with_num_epochs(epochs)
                .with_batch_size(batch_size)
                .with_learning_rate(lr)
                .with_augment(augment_enabled)
                .with_augment_profile(augment_profile.as_str().to_string());

            if std::path::Path::new(&out).join("model.mpk").exists() {
                eprintln!(
                    "Warning: {out}/model.mpk already exists and will be overwritten by this training run"
                );
            }

            run_training(backend, &out, &data_dir, cfg);
        }
        Cmd::Infer { image, artifacts } => {
            use burn::backend::NdArray;
            use burn::backend::ndarray::NdArrayDevice;
            let device = NdArrayDevice::Cpu;
            let recognizer = Recognizer::<NdArray<f32, i64>>::load(&artifacts, device)
                .expect("failed to load model");
            let result = recognizer
                .recognize_path(&image)
                .expect("failed to recognize image");
            println!("Predicted: {result}");
        }
    }
}

/// Run training on the selected backend.
fn run_training(backend: BackendKind, out: &str, data_dir: &str, cfg: TrainingConfig) {
    match backend {
        BackendKind::Wgpu => {
            use burn::backend::Wgpu;
            use burn::backend::wgpu::WgpuDevice;
            training::run::<Autodiff<Wgpu<f32, i32>>>(out, data_dir, WgpuDevice::default(), cfg);
        }
        BackendKind::Cpu => {
            use burn::backend::NdArray;
            use burn::backend::ndarray::NdArrayDevice;
            training::run::<Autodiff<NdArray<f32, i64>>>(out, data_dir, NdArrayDevice::Cpu, cfg);
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use burn::backend::Cuda;
                use burn::backend::cuda::CudaDevice;
                training::run::<Autodiff<Cuda<f32, i32>>>(
                    out,
                    data_dir,
                    CudaDevice::default(),
                    cfg,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (out, data_dir, cfg);
                eprintln!(
                    "CUDA backend not compiled. Rebuild with:\n  \
                     cargo build --release -p capburn-train --features cuda"
                );
                std::process::exit(1);
            }
        }
    }
}
