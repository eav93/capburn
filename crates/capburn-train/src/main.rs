use burn::backend::Autodiff;
use capburn_core::{CaptchaModelConfig, Charset, Recognizer};
use clap::{Parser, Subcommand, ValueEnum};

mod data;
mod detect;
mod training;

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
        /// range), `fixed` (positional classifier, best for fixed length),
        /// or `ctc` (variable length / shifting positions).
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Compute backend.
        #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
        backend: BackendKind,
        #[arg(long, default_value_t = 30)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 5.0e-4)]
        lr: f64,
        /// Disable training-image augmentation (affine + photometric jitter).
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
            backend,
            epochs,
            batch_size,
            lr,
            no_augment,
        } => {
            let data_dir = match &dataset {
                Some(name) => format!("{data}/{name}"),
                None => data.clone(),
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

            // Auto: a single length → positional `fixed` head (stronger, faster);
            // a length range → `ctc` for variable length.
            let arch = if arch.eq_ignore_ascii_case("auto") {
                if length.min_chars == length.max_chars {
                    "fixed".to_string()
                } else {
                    "ctc".to_string()
                }
            } else {
                arch
            };

            let model_cfg =
                CaptchaModelConfig::new(charset.as_chars(), length.max_chars).with_arch(arch);
            let cfg = TrainingConfig::new(model_cfg)
                .with_min_chars(length.min_chars)
                .with_num_epochs(epochs)
                .with_batch_size(batch_size)
                .with_learning_rate(lr)
                .with_augment(!no_augment);

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
