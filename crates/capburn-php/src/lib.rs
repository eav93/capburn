//! capburn-php — PHP extension for captcha recognition.
//!
//! The extension loads a model trained by the `capburn` trainer and runs
//! inference on the CPU (ndarray backend). No PHP (Composer) wrapper is needed —
//! the extension registers a ready-to-use class.
//!
//! Registered class:
//! * `Capburn\Ext\Recognizer` — loads a model and recognizes captchas.

// PHP extensions on Windows use the `vectorcall` ABI; the ext-php-rs macros
// expand `extern "vectorcall"`, so the unstable feature must be enabled here
// too. Linux and macOS build on stable.
#![cfg_attr(windows, feature(abi_vectorcall))]
#![allow(clippy::new_without_default)]

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use capburn_core::CpuRecognizer;
use ext_php_rs::exception::PhpException;
use ext_php_rs::prelude::*;
use ext_php_rs::types::Zval;
use ext_php_rs::zend::ModuleEntry;
use ext_php_rs::{info_table_end, info_table_row, info_table_start};

/// Upper bound on the input image size (in bytes). A captcha is a small image;
/// this limit rejects attempts to feed a huge input to a web server.
const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;

/// Captcha recognizer: holds the loaded model and charset.
#[php_class]
#[php(name = "Capburn\\Ext\\Recognizer")]
pub struct Recognizer {
    inner: CpuRecognizer,
}

#[php_impl]
impl Recognizer {
    /// Load a model from an artifacts folder (`model.json` + `model.mpk`).
    ///
    /// @param string $artifactsDir path to the folder with the trained model.
    pub fn __construct(artifacts_dir: &str) -> PhpResult<Self> {
        let inner = CpuRecognizer::load_cpu(artifacts_dir).map_err(PhpException::default)?;
        Ok(Self { inner })
    }

    /// Recognize a captcha from a file. Returns the character string.
    pub fn recognize(&self, image_path: &str) -> PhpResult<String> {
        self.inner
            .recognize_path(image_path)
            .map_err(PhpException::default)
    }

    /// Recognize a captcha from a binary string containing the image bytes
    /// (e.g. the result of `file_get_contents()` or an HTTP response body).
    pub fn recognize_bytes(&self, data: &Zval) -> PhpResult<String> {
        let bytes = data
            .zend_str()
            .ok_or_else(|| PhpException::default("expected a string of image bytes".into()))?
            .as_bytes();
        if bytes.len() > MAX_INPUT_BYTES {
            return Err(PhpException::default(format!(
                "image too large: {} bytes (max {MAX_INPUT_BYTES})",
                bytes.len()
            )));
        }
        self.inner
            .recognize_bytes(bytes)
            .map_err(PhpException::default)
    }

    /// Recognize a captcha from a base64 string. Both raw base64 and a data-URL
    /// like `data:image/png;base64,iVBORw0...` are supported.
    pub fn recognize_base64(&self, data: &str) -> PhpResult<String> {
        let payload = match data.split_once(";base64,") {
            Some((_, b64)) => b64,
            None => data.trim(),
        };
        let payload = payload.trim();
        // Estimate the decoded size (~3/4 of the base64 length) to reject a
        // huge input before actually decoding it.
        if payload.len() / 4 * 3 > MAX_INPUT_BYTES {
            return Err(PhpException::default(format!(
                "image too large (max {MAX_INPUT_BYTES} bytes)"
            )));
        }
        let bytes = STANDARD
            .decode(payload)
            .map_err(|e| PhpException::default(format!("invalid base64: {e}")))?;
        self.inner
            .recognize_bytes(&bytes)
            .map_err(PhpException::default)
    }

    /// Captcha length (number of characters) the model outputs.
    pub fn num_chars(&self) -> i64 {
        self.inner.num_chars() as i64
    }

    /// Extension build version (release tag, or `0.0.0-dev` for a local build).
    pub fn extension_version() -> &'static str {
        env!("CAPBURN_PHP_BUILD_VERSION")
    }
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module.info_function(module_info).class::<Recognizer>()
}

#[unsafe(no_mangle)]
pub extern "C" fn module_info(_module: *mut ModuleEntry) {
    info_table_start!();
    info_table_row!("capburn-php", "enabled");
    info_table_row!("Extension version", env!("CAPBURN_PHP_BUILD_VERSION"));
    info_table_row!("Backend", "ndarray (CPU)");
    info_table_end!();
}
