//! Image preprocessing: convert to a fixed-size grayscale tensor.

use image::imageops::FilterType;
use image::{DynamicImage, ImageReader, Limits};
use std::io::Cursor;
use std::path::Path;

// Small input keeps training and CPU inference fast; captcha characters stay
// legible at this size. The width becomes the sequence axis after the CNN.
pub const IMG_WIDTH: usize = 128;
pub const IMG_HEIGHT: usize = 32;

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
    let gray = img
        .resize_exact(IMG_WIDTH as u32, IMG_HEIGHT as u32, FilterType::Triangle)
        .to_luma8();
    gray.as_raw().iter().map(|p| *p as f32 / 255.0).collect()
}

/// Load an image from disk and convert it to a luminance vector. Applies the
/// same size limits as the in-memory decoder to guard against decode bombs.
pub fn load_image_as_floats<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, String> {
    let mut reader = ImageReader::open(path.as_ref())
        .map_err(|e| format!("cannot open {}: {e}", path.as_ref().display()))?
        .with_guessed_format()
        .map_err(|e| format!("cannot guess image format: {e}"))?;
    reader.limits(safe_limits());
    let img = reader
        .decode()
        .map_err(|e| format!("cannot decode image: {e}"))?;
    Ok(image_to_floats(&img))
}

/// Decode an image from in-memory bytes (for the PHP extension) and convert it
/// to a luminance vector. Applies size limits to guard against decode bombs.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("cannot guess image format: {e}"))?;
    reader.limits(safe_limits());
    let img = reader
        .decode()
        .map_err(|e| format!("cannot decode image from bytes: {e}"))?;
    Ok(image_to_floats(&img))
}
