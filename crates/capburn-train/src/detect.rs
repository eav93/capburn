//! Auto-detect the captcha format from a dataset: labels, image geometry and
//! conservative training defaults.

use crate::data::{AugmentProfile, label_from_stem};
use capburn_core::{Charset, InputSize, PreprocessMode, inspect_image};
use std::collections::HashMap;
use std::path::Path;

const IMAGE_SCAN_LIMIT: usize = 256;

/// Result of scanning a dataset folder.
pub struct Scan {
    /// All unique characters seen in "plausible" labels.
    pub observed: Vec<char>,
    /// Most frequent label length (mode) — the `num_chars` candidate.
    pub length_mode: usize,
    /// Label length distribution: length → number of files.
    pub length_hist: Vec<(usize, usize)>,
    /// How many files were analyzed.
    pub considered: usize,
}

/// Result of scanning decoded images.
pub struct ImageScan {
    /// How many plausible files were available for image scan.
    pub total: usize,
    /// How many files were sampled for image scan.
    pub sampled: usize,
    /// How many images were decoded and inspected.
    pub decoded: usize,
    /// How many plausible image files failed to decode.
    pub decode_failed: usize,
    /// Image dimension distribution: (width, height) → number of files.
    pub dim_hist: Vec<((u32, u32), usize)>,
    /// Dominant image dimensions.
    pub mode_dim: (u32, u32),
    /// Number of images with the dominant dimensions.
    pub mode_dim_count: usize,
    /// Dominant raw source aspect ratio.
    pub mode_aspect: f32,
    /// Average raw source aspect ratio.
    pub avg_aspect: f32,
    /// Average fraction of visibly colored pixels.
    pub color_fraction: f32,
    /// Average fraction of non-background pixels.
    pub ink_fraction: f32,
}

/// Whether a label is "plausible": non-empty and consisting of letters/digits
/// only (skips files like `README`, `thumbs.db`, hashes with symbols).
fn is_plausible_label(label: &str) -> bool {
    !label.is_empty() && label.chars().all(|c| c.is_alphanumeric())
}

/// Scan a folder and collect statistics about labels from file names.
pub fn scan_folder<P: AsRef<Path>>(folder: P) -> std::io::Result<Scan> {
    let mut observed: Vec<char> = Vec::new();
    let mut lengths: HashMap<usize, usize> = HashMap::new();
    let mut considered = 0usize;

    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let label = label_from_stem(stem);
        if !is_plausible_label(label) {
            continue;
        }
        considered += 1;
        *lengths.entry(label.chars().count()).or_insert(0) += 1;
        for c in label.chars() {
            if !observed.contains(&c) {
                observed.push(c);
            }
        }
    }

    let mut length_hist: Vec<(usize, usize)> = lengths.into_iter().collect();
    length_hist.sort_by_key(|(len, _)| *len);

    // Deterministic mode: highest frequency, ties broken toward the smaller
    // length (length_hist is sorted ascending by length; iterating reversed,
    // max_by_key returns the last max, i.e. the smallest length on a tie).
    let length_mode = length_hist
        .iter()
        .rev()
        .max_by_key(|(_, count)| *count)
        .map(|(len, _)| *len)
        .unwrap_or(0);

    Ok(Scan {
        observed,
        length_mode,
        length_hist,
        considered,
    })
}

/// Decode images and collect cheap visual statistics for auto-configuration.
pub fn scan_images<P: AsRef<Path>>(folder: P) -> std::io::Result<ImageScan> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if !is_plausible_label(label_from_stem(stem)) {
                continue;
            }
            paths.push(path);
        }
    }
    paths.sort();
    let total = paths.len();
    let sample_len = total.min(IMAGE_SCAN_LIMIT);

    let mut dims: HashMap<(u32, u32), usize> = HashMap::new();
    let mut decoded = 0usize;
    let mut decode_failed = 0usize;
    let mut aspect_sum = 0.0f32;
    let mut color_sum = 0.0f32;
    let mut ink_sum = 0.0f32;
    for i in 0..sample_len {
        let idx = if sample_len == total {
            i
        } else {
            i * total / sample_len
        };
        let path = &paths[idx];
        match inspect_image(path) {
            Ok(info) => {
                decoded += 1;
                *dims.entry((info.width, info.height)).or_insert(0) += 1;
                aspect_sum += info.width as f32 / info.height.max(1) as f32;
                color_sum += info.color_fraction;
                ink_sum += info.ink_fraction;
            }
            Err(_) => decode_failed += 1,
        }
    }

    let mut dim_hist: Vec<((u32, u32), usize)> = dims.into_iter().collect();
    dim_hist.sort_by_key(|((w, h), _)| (*w, *h));
    let (mode_dim, mode_dim_count) = dim_hist
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(dim, count)| (*dim, *count))
        .unwrap_or(((0, 0), 0));
    let mode_aspect = mode_dim.0 as f32 / mode_dim.1.max(1) as f32;

    Ok(ImageScan {
        total,
        sampled: sample_len,
        decoded,
        decode_failed,
        dim_hist,
        mode_dim,
        mode_dim_count,
        mode_aspect,
        avg_aspect: aspect_sum / decoded.max(1) as f32,
        color_fraction: color_sum / decoded.max(1) as f32,
        ink_fraction: ink_sum / decoded.max(1) as f32,
    })
}

/// Recommend preprocessing from source geometry. Historical exact resize is the
/// default; `fit` is selected only for extreme aspect-ratio mismatch.
pub fn recommend_preprocess(scan: &ImageScan, target_size: InputSize) -> PreprocessMode {
    if scan.decoded == 0 {
        return PreprocessMode::Stretch;
    }
    let target_aspect = target_size.width as f32 / target_size.height as f32;
    let ratio = scan.mode_aspect / target_aspect;
    if !(0.67..=1.50).contains(&ratio) {
        PreprocessMode::Fit
    } else {
        PreprocessMode::Stretch
    }
}

/// Recommend model input size from the dominant source image dimensions. Keeps
/// normal captcha sizes close to native resolution, but caps very large images
/// to avoid exploding memory and fixed-global parameter count.
pub fn recommend_input_size(scan: &ImageScan) -> InputSize {
    if scan.mode_dim.0 == 0 || scan.mode_dim.1 == 0 {
        return InputSize::default();
    }

    const MAX_AUTO_PIXELS: usize = 32_768;
    let mut width = scan.mode_dim.0 as usize;
    let mut height = (scan.mode_dim.1 as usize).max(32);
    let pixels = width.saturating_mul(height);
    if pixels > MAX_AUTO_PIXELS {
        let scale = (MAX_AUTO_PIXELS as f32 / pixels as f32).sqrt();
        width = ((width as f32 * scale).round() as usize).max(32);
        height = ((height as f32 * scale).round() as usize).max(32);
    }

    width = round_up_to_multiple(width, 4).min(512);
    height = height.min(256);
    InputSize::new(width, height)
}

fn round_up_to_multiple(value: usize, step: usize) -> usize {
    value.div_ceil(step) * step
}

/// Conservative default until per-dataset A/B says otherwise.
pub fn recommend_augment() -> AugmentProfile {
    AugmentProfile::Medium
}

/// Chosen captcha length range. CTC handles variable length natively, so this is
/// only used to filter which files enter the dataset and to set the maximum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LengthSpec {
    pub min_chars: usize,
    pub max_chars: usize,
}

/// Parse a `--num-chars` value: `5` (fixed) or `4-7` (min-max range).
pub fn parse_length_spec(s: &str) -> Result<LengthSpec, String> {
    let s = s.trim();
    if let Some((lo, hi)) = s.split_once('-') {
        let lo: usize = lo
            .trim()
            .parse()
            .map_err(|_| format!("invalid range: {s:?}"))?;
        let hi: usize = hi
            .trim()
            .parse()
            .map_err(|_| format!("invalid range: {s:?}"))?;
        if lo == 0 || hi == 0 || lo > hi {
            return Err(format!(
                "invalid range: {s:?} (expected MIN-MAX, 1 ≤ MIN ≤ MAX)"
            ));
        }
        Ok(LengthSpec {
            min_chars: lo,
            max_chars: hi,
        })
    } else {
        let n: usize = s.parse().map_err(|_| format!("invalid length: {s:?}"))?;
        if n == 0 {
            return Err("length must be > 0".into());
        }
        Ok(LengthSpec {
            min_chars: n,
            max_chars: n,
        })
    }
}

/// Detect the charset and captcha length from a dataset folder.
///
/// `charset_override` — explicit charset from the CLI (when not "auto");
/// `length_override` — explicit length/range from the CLI (when set).
pub fn detect_format<P: AsRef<Path>>(
    folder: P,
    charset_override: Option<Charset>,
    length_override: Option<LengthSpec>,
) -> std::io::Result<(Charset, LengthSpec)> {
    let scan = scan_folder(&folder)?;
    if scan.considered == 0 {
        eprintln!("Warning: no files with recognizable labels found in the folder");
    }

    let charset = match charset_override {
        Some(cs) => cs,
        None => Charset::from_observed(scan.observed.iter().copied())
            .expect("could not auto-detect charset — set --charset manually"),
    };

    let length = length_override.unwrap_or_else(|| detect_length(&scan));
    assert!(
        length.max_chars > 0,
        "could not determine captcha length (no labeled files in {}) — set --num-chars",
        folder.as_ref().display()
    );

    println!("Auto-detected format from {} files:", scan.considered);
    print!("  label lengths:");
    for (len, count) in &scan.length_hist {
        print!(" {len}×{count}");
    }
    println!();
    let mode = if length.min_chars == length.max_chars {
        format!("fixed {}", length.max_chars)
    } else {
        format!("variable {}..={}", length.min_chars, length.max_chars)
    };
    println!(
        "  chosen: length = {mode}, charset = {} chars ({})",
        charset.len(),
        charset.describe_families()
    );

    Ok((charset, length))
}

/// Determine the length range from the scanned histogram. Lengths covering at
/// least 5% of the labeled files count; the range spans the smallest to the
/// largest such length.
fn detect_length(scan: &Scan) -> LengthSpec {
    let threshold = (scan.considered as f64 * 0.05).ceil() as usize;
    let significant: Vec<usize> = scan
        .length_hist
        .iter()
        .filter(|(_, count)| *count >= threshold.max(1))
        .map(|(len, _)| *len)
        .collect();
    match (significant.iter().min(), significant.iter().max()) {
        (Some(&lo), Some(&hi)) => LengthSpec {
            min_chars: lo,
            max_chars: hi,
        },
        _ => LengthSpec {
            min_chars: scan.length_mode,
            max_chars: scan.length_mode,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_length_mode_and_families() {
        let dir = std::env::temp_dir().join("capburn_detect_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // three length-5 labels and one length-4 → mode 5; chars: digits + cyrillic
        for name in ["12345_a", "67890_b", "1а2б3", "12ц4"] {
            std::fs::write(dir.join(format!("{name}.png")), b"x").unwrap();
        }

        let scan = scan_folder(&dir).unwrap();
        assert_eq!(scan.length_mode, 5);
        assert_eq!(scan.considered, 4);

        let cs = Charset::from_observed(scan.observed.iter().copied()).unwrap();
        assert!(cs.index_of('0').is_some(), "digits present");
        assert!(cs.index_of('а').is_some(), "cyrillic present");
        assert!(cs.index_of('A').is_none(), "no latin");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_length_spec_fixed_and_range() {
        assert_eq!(
            parse_length_spec("5").unwrap(),
            LengthSpec {
                min_chars: 5,
                max_chars: 5
            }
        );
        assert_eq!(
            parse_length_spec("4-7").unwrap(),
            LengthSpec {
                min_chars: 4,
                max_chars: 7
            }
        );
        assert!(parse_length_spec("0").is_err());
        assert!(parse_length_spec("7-4").is_err());
        assert!(parse_length_spec("x").is_err());
    }
}
