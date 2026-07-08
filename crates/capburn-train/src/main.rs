use burn::backend::Autodiff;
use capburn_core::{Arch, CaptchaModelConfig, Charset, InputSize, PreprocessMode, Recognizer};
use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

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
        /// range), `fixed`, `fixed-global`, `fixed-seq`, `fixed-seq-pool`, or `ctc`.
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Image preprocessing: `auto`, `stretch` (resize exactly), or `fit`
        /// (preserve aspect ratio and pad).
        #[arg(long, default_value = "auto")]
        preprocess: String,
        /// Model input size: `auto` (dominant dataset image size) or `WIDTHxHEIGHT`.
        #[arg(long, default_value = "auto")]
        input_size: String,
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
    /// Evaluate a saved model on a labeled dataset folder. Reports accuracy on
    /// the whole folder and, when the artifacts contain `config.json`, on the
    /// reconstructed training validation split (the honest held-out metric).
    Eval {
        /// Root folder containing datasets.
        #[arg(long, default_value = "./data")]
        data: String,
        /// Dataset subfolder name inside `--data`. If omitted, `--data` itself is used.
        #[arg(long)]
        dataset: Option<String>,
        /// Saved model folder containing `model.json` and `model.mpk`.
        #[arg(long, default_value = "./artifacts")]
        artifacts: String,
        /// Evaluate at most this many files.
        #[arg(long)]
        limit: Option<usize>,
        /// Number of top confusions to print per position.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Print progress every N processed files; use 0 to disable.
        #[arg(long, default_value_t = 100)]
        progress_every: usize,
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
            input_size,
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

            let image_scan = if preprocess.eq_ignore_ascii_case("auto")
                || input_size.eq_ignore_ascii_case("auto")
            {
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

            let input_size = if input_size.eq_ignore_ascii_case("auto") {
                image_scan
                    .as_ref()
                    .map(detect::recommend_input_size)
                    .unwrap_or_else(InputSize::default)
            } else {
                InputSize::parse(&input_size).unwrap_or_else(|e| fail(e))
            };
            input_size.validate().unwrap_or_else(|e| fail(e));

            let preprocess_mode = if preprocess.eq_ignore_ascii_case("auto") {
                image_scan
                    .as_ref()
                    .map(|scan| detect::recommend_preprocess(scan, input_size))
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
                "Auto params: arch={}, input-size={}, preprocess={}, augment={}",
                arch,
                input_size.as_string(),
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
            let ctc_time_steps = input_size.ctc_time_steps();
            if arch_kind == Arch::Ctc && ctc_worst_case_steps > ctc_time_steps {
                fail(format!(
                    "--num-chars max {} is too long for the CTC sequence length {} (needs 2*len-1 ≤ {})",
                    length.max_chars, ctc_time_steps, ctc_time_steps
                ));
            }

            let model_cfg = CaptchaModelConfig::new(charset.as_chars(), length.max_chars)
                .with_arch(arch)
                .with_preprocess(preprocess_mode.as_str().to_string())
                .with_input_width(input_size.width)
                .with_input_height(input_size.height)
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
        Cmd::Eval {
            data,
            dataset,
            artifacts,
            limit,
            top,
            progress_every,
        } => {
            let data_dir = match &dataset {
                Some(name) => format!("{data}/{name}"),
                None => data.clone(),
            };
            run_eval(&artifacts, &data_dir, limit, top, progress_every);
        }
    }
}

fn run_eval(
    artifacts: &str,
    data_dir: &str,
    limit: Option<usize>,
    top: usize,
    progress_every: usize,
) {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    let device = NdArrayDevice::Cpu;
    let recognizer =
        Recognizer::<NdArray<f32, i64>>::load(artifacts, device).expect("failed to load model");

    let mut files: Vec<PathBuf> = std::fs::read_dir(data_dir)
        .unwrap_or_else(|e| panic!("cannot read dataset folder {data_dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(data::label_from_stem)
                .is_some_and(|label| !label.is_empty())
        })
        .collect();
    files.sort();
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    let planned = files.len();
    eprintln!("Evaluating {planned} files from {data_dir} using {artifacts}...");
    let _ = std::io::stderr().flush();

    // Best-effort reconstruction of the training validation split, so the
    // held-out accuracy can be reported separately from the full-dataset one.
    let valid_set = load_validation_split(artifacts, data_dir);

    let mut total = 0usize;
    let mut seen = 0usize;
    let mut full_correct = 0usize;
    let mut char_correct = 0usize;
    let mut char_total = 0usize;
    let mut decode_failed = 0usize;
    let mut length_mismatch = 0usize;
    let mut valid_total = 0usize;
    let mut valid_full_correct = 0usize;
    let mut valid_char_correct = 0usize;
    let mut valid_char_total = 0usize;
    let mut pos_correct = Vec::<usize>::new();
    let mut pos_total = Vec::<usize>::new();
    let mut pos_confusions = Vec::<HashMap<String, usize>>::new();
    let mut pred_lengths = HashMap::<usize, usize>::new();
    let mut label_chars = HashMap::<char, usize>::new();
    let mut pred_chars = HashMap::<char, usize>::new();

    for path in files {
        seen += 1;
        let label = label_for_path(&path);
        let pred = match recognizer.recognize_path(&path) {
            Ok(pred) => pred,
            Err(e) => {
                decode_failed += 1;
                if decode_failed <= 5 {
                    eprintln!("Skipping {}: {e}", path.display());
                }
                continue;
            }
        };

        let label: Vec<char> = label.chars().collect();
        let pred: Vec<char> = pred.chars().collect();
        total += 1;
        let full_ok = pred == label;
        if full_ok {
            full_correct += 1;
        }
        *pred_lengths.entry(pred.len()).or_insert(0) += 1;

        for c in &label {
            *label_chars.entry(*c).or_insert(0) += 1;
        }
        for c in &pred {
            *pred_chars.entry(*c).or_insert(0) += 1;
        }

        // Positional matches over the overlap; the denominator is the longer of
        // the two, so both missing and extra characters count as errors.
        let matched = pred
            .iter()
            .zip(label.iter())
            .filter(|(a, b)| a == b)
            .count();
        char_correct += matched;
        char_total += label.len().max(pred.len());

        let in_valid = valid_set.as_ref().is_some_and(|set| set.contains(&path));
        if in_valid {
            valid_total += 1;
            if full_ok {
                valid_full_correct += 1;
            }
            valid_char_correct += matched;
            valid_char_total += label.len().max(pred.len());
        }

        // Per-position stats are positional, so they are only meaningful when
        // the prediction has the label's length (always true for fixed heads).
        if pred.len() == label.len() {
            ensure_positions(
                &mut pos_correct,
                &mut pos_total,
                &mut pos_confusions,
                label.len(),
            );
            for (idx, (expected, actual)) in label.iter().zip(pred.iter()).enumerate() {
                pos_total[idx] += 1;
                if actual == expected {
                    pos_correct[idx] += 1;
                } else {
                    let key = format!("{expected}->{actual}");
                    *pos_confusions[idx].entry(key).or_insert(0) += 1;
                }
            }
        } else {
            length_mismatch += 1;
        }
        print_eval_progress(seen, planned, progress_every, full_correct, total);
    }

    println!("Evaluated: {total} examples");
    if decode_failed > 0 {
        println!("Skipped decode failures: {decode_failed}");
    }
    println!("Full-captcha accuracy: {:.2}%", pct(full_correct, total));
    println!("Character accuracy: {:.2}%", pct(char_correct, char_total));
    match &valid_set {
        Some(set) if valid_total > 0 => {
            println!(
                "Held-out (training validation split, {valid_total}/{} files): full-captcha {:.2}%, char {:.2}%",
                set.len(),
                pct(valid_full_correct, valid_total),
                pct(valid_char_correct, valid_char_total)
            );
            if decode_failed > 0 {
                println!(
                    "  note: decode failures can shift the reconstructed split — treat as approximate"
                );
            }
        }
        Some(_) => println!("Held-out: no validation-split files were evaluated (raise --limit)"),
        None => println!(
            "Held-out: cannot reconstruct the training split (no readable config.json in {artifacts})"
        ),
    }
    println!("Prediction lengths: {}", format_counts(&pred_lengths));
    println!("Label chars: {}", format_counts(&label_chars));
    println!("Pred chars:  {}", format_counts(&pred_chars));
    if length_mismatch > 0 {
        println!("Length mismatches: {length_mismatch} (excluded from per-position stats below)");
    }
    println!("Position accuracy:");
    for idx in 0..pos_total.len() {
        println!(
            "  pos {}: {:.2}%  top errors: {}",
            idx + 1,
            pct(pos_correct[idx], pos_total[idx]),
            format_top_counts(&pos_confusions[idx], top)
        );
    }
}

/// Replay the training dataset pipeline (same filtering in `read_dir` order,
/// seeded shuffle, 0.9/0.1 split) over file paths to recover which files were
/// the validation split. Best effort: returns `None` without `config.json`, and
/// files that failed to decode during training shift the permutation slightly.
fn load_validation_split(
    artifacts: &str,
    data_dir: &str,
) -> Option<std::collections::HashSet<PathBuf>> {
    use burn::config::Config;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    let config = training::TrainingConfig::load(format!("{artifacts}/config.json")).ok()?;
    let charset: std::collections::HashSet<char> = config.model.charset.chars().collect();
    let min_len = config.min_chars;
    let max_len = config.model.num_chars;

    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(data_dir).ok()? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let label = data::label_from_stem(stem);
        let len = label.chars().count();
        if len < min_len || len > max_len || !label.chars().all(|c| charset.contains(&c)) {
            continue;
        }
        candidates.push(path);
    }
    if candidates.len() < 2 {
        return None;
    }
    // Mirrors Dataset::shuffle + Dataset::split(0.9) in the training run.
    let mut rng = StdRng::seed_from_u64(config.seed);
    candidates.shuffle(&mut rng);
    let n_train = ((candidates.len() as f32) * 0.9).round() as usize;
    let n_train = n_train.clamp(1, candidates.len() - 1);
    Some(candidates.split_off(n_train).into_iter().collect())
}

fn print_eval_progress(
    seen: usize,
    planned: usize,
    progress_every: usize,
    full_correct: usize,
    total: usize,
) {
    if progress_every == 0 || seen == 0 || !seen.is_multiple_of(progress_every) {
        return;
    }
    eprintln!(
        "Eval progress: {seen}/{planned} files, usable {total}, full {:.2}%",
        pct(full_correct, total)
    );
    let _ = std::io::stderr().flush();
}

fn label_for_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("file name must be valid UTF-8");
    data::label_from_stem(stem).to_string()
}

fn ensure_positions(
    pos_correct: &mut Vec<usize>,
    pos_total: &mut Vec<usize>,
    pos_confusions: &mut Vec<HashMap<String, usize>>,
    len: usize,
) {
    while pos_correct.len() < len {
        pos_correct.push(0);
        pos_total.push(0);
        pos_confusions.push(HashMap::new());
    }
}

fn pct(correct: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32 * 100.0
    }
}

fn format_counts<K: Ord + std::fmt::Display>(counts: &HashMap<K, usize>) -> String {
    let mut pairs: Vec<_> = counts.iter().collect();
    pairs.sort_by_key(|&(k, _)| k);
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_top_counts(counts: &HashMap<String, usize>, top: usize) -> String {
    let mut pairs: Vec<_> = counts.iter().collect();
    pairs.sort_by(|(ka, va), (kb, vb)| vb.cmp(va).then_with(|| ka.cmp(kb)));
    let text = pairs
        .into_iter()
        .take(top)
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(", ");
    if text.is_empty() { "-".into() } else { text }
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
