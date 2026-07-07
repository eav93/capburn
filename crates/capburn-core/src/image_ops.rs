//! Image preprocessing: convert to a fixed-size grayscale tensor.

use image::imageops::{self, FilterType};
use image::{DynamicImage, GrayImage, ImageReader, Limits, Luma};
use std::io::Cursor;
use std::path::Path;

// Small input keeps training and CPU inference fast; captcha characters stay
// legible at this size. The width becomes the sequence axis after the CNN.
pub const IMG_WIDTH: usize = 128;
pub const IMG_HEIGHT: usize = 32;

/// How source images are mapped to the fixed model input size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreprocessMode {
    /// Resize exactly to 128x32. This is the historical behavior and keeps old
    /// models compatible.
    Stretch,
    /// Preserve source aspect ratio and center-pad to 128x32.
    Fit,
}

impl PreprocessMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "stretch" => Ok(Self::Stretch),
            "fit" => Ok(Self::Fit),
            other => Err(format!(
                "unknown preprocess {other:?} (expected 'stretch' or 'fit')"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stretch => "stretch",
            Self::Fit => "fit",
        }
    }
}

/// Cheap image statistics used by the trainer's auto-configuration.
#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// Fraction of sampled pixels whose RGB channels differ noticeably.
    pub color_fraction: f32,
    /// Fraction of sampled pixels that differ from the estimated background.
    pub ink_fraction: f32,
}

/// Upper bound on decoded image dimensions — guards against "decode bombs"
/// (a tiny file that expands into a huge buffer) when accepting untrusted
/// input in the PHP extension.
const MAX_IMAGE_DIM: u32 = 8192;
const MAX_IMAGE_ALLOC: u64 = 128 * 1024 * 1024;

fn safe_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC);
    limits
}

/// Turn a decoded image into a `[0.0, 1.0]` luminance vector of length
/// `IMG_HEIGHT * IMG_WIDTH` (single channel, resized exactly).
pub fn image_to_floats(img: &DynamicImage) -> Vec<f32> {
    image_to_floats_with_mode(img, PreprocessMode::Stretch)
}

/// Turn a decoded image into model input using the selected preprocessing mode.
pub fn image_to_floats_with_mode(img: &DynamicImage, mode: PreprocessMode) -> Vec<f32> {
    let gray = match mode {
        PreprocessMode::Stretch => img
            .resize_exact(IMG_WIDTH as u32, IMG_HEIGHT as u32, FilterType::Triangle)
            .to_luma8(),
        PreprocessMode::Fit => resize_fit(img),
    };
    gray.as_raw().iter().map(|p| *p as f32 / 255.0).collect()
}

/// Load an image from disk and convert it to a luminance vector. Applies the
/// same size limits as the in-memory decoder to guard against decode bombs.
pub fn load_image_as_floats<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, String> {
    load_image_as_floats_with_mode(path, PreprocessMode::Stretch)
}

/// Load an image from disk and convert it with the selected preprocessing mode.
pub fn load_image_as_floats_with_mode<P: AsRef<Path>>(
    path: P,
    mode: PreprocessMode,
) -> Result<Vec<f32>, String> {
    let mut reader = ImageReader::open(path.as_ref())
        .map_err(|e| format!("cannot open {}: {e}", path.as_ref().display()))?
        .with_guessed_format()
        .map_err(|e| format!("cannot guess image format: {e}"))?;
    reader.limits(safe_limits());
    let img = reader
        .decode()
        .map_err(|e| format!("cannot decode image: {e}"))?;
    Ok(image_to_floats_with_mode(&img, mode))
}

/// Decode an image from in-memory bytes (for the PHP extension) and convert it
/// to a luminance vector. Applies size limits to guard against decode bombs.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    load_image_from_bytes_with_mode(bytes, PreprocessMode::Stretch)
}

/// Decode an image from in-memory bytes and convert it with the selected mode.
pub fn load_image_from_bytes_with_mode(
    bytes: &[u8],
    mode: PreprocessMode,
) -> Result<Vec<f32>, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("cannot guess image format: {e}"))?;
    reader.limits(safe_limits());
    let img = reader
        .decode()
        .map_err(|e| format!("cannot decode image from bytes: {e}"))?;
    Ok(image_to_floats_with_mode(&img, mode))
}

/// Inspect a decoded image without applying the model preprocessing.
pub fn inspect_image<P: AsRef<Path>>(path: P) -> Result<ImageInfo, String> {
    let mut reader = ImageReader::open(path.as_ref())
        .map_err(|e| format!("cannot open {}: {e}", path.as_ref().display()))?
        .with_guessed_format()
        .map_err(|e| format!("cannot guess image format: {e}"))?;
    reader.limits(safe_limits());
    let img = reader
        .decode()
        .map_err(|e| format!("cannot decode image: {e}"))?;

    let rgb = img.to_rgb8();
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let bg = estimate_background(&gray) as i16;
    let mut sampled = 0usize;
    let mut color = 0usize;
    let mut ink = 0usize;
    let stride = ((width as usize * height as usize) / 8192).max(1);
    for (i, (rgb_px, gray_px)) in rgb.pixels().zip(gray.pixels()).enumerate() {
        if i % stride != 0 {
            continue;
        }
        sampled += 1;
        let [r, g, b] = rgb_px.0;
        let lo = r.min(g).min(b);
        let hi = r.max(g).max(b);
        if hi.saturating_sub(lo) > 18 {
            color += 1;
        }
        if (gray_px.0[0] as i16 - bg).abs() > 28 {
            ink += 1;
        }
    }

    Ok(ImageInfo {
        width,
        height,
        color_fraction: color as f32 / sampled.max(1) as f32,
        ink_fraction: ink as f32 / sampled.max(1) as f32,
    })
}

fn resize_fit(img: &DynamicImage) -> GrayImage {
    let gray = img.to_luma8();
    let (src_w, src_h) = gray.dimensions();
    let scale =
        (IMG_WIDTH as f32 / src_w.max(1) as f32).min(IMG_HEIGHT as f32 / src_h.max(1) as f32);
    let new_w = ((src_w as f32 * scale).round() as u32).clamp(1, IMG_WIDTH as u32);
    let new_h = ((src_h as f32 * scale).round() as u32).clamp(1, IMG_HEIGHT as u32);

    let resized = imageops::resize(&gray, new_w, new_h, FilterType::Triangle);
    let bg = estimate_background(&gray);
    let mut out = GrayImage::from_pixel(IMG_WIDTH as u32, IMG_HEIGHT as u32, Luma([bg]));
    let x = (IMG_WIDTH as u32 - new_w) / 2;
    let y = (IMG_HEIGHT as u32 - new_h) / 2;
    imageops::replace(&mut out, &resized, x.into(), y.into());
    out
}

fn estimate_background(gray: &GrayImage) -> u8 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 255;
    }

    let mut sum = 0u64;
    let mut count = 0u64;
    for x in 0..w {
        sum += gray.get_pixel(x, 0).0[0] as u64;
        sum += gray.get_pixel(x, h - 1).0[0] as u64;
        count += 2;
    }
    for y in 1..h.saturating_sub(1) {
        sum += gray.get_pixel(0, y).0[0] as u64;
        sum += gray.get_pixel(w - 1, y).0[0] as u64;
        count += 2;
    }
    (sum / count.max(1)) as u8
}
