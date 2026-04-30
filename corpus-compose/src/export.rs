//! Output format helpers (PNG, WAV, HTML, JSON manifest).

use std::path::Path;

use image::{DynamicImage, ImageFormat};

use crate::Result;

/// Image output format.
#[derive(Debug, Clone, Copy)]
pub enum ImageFormatChoice {
    Png,
    Jpeg,
}

impl ImageFormatChoice {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().trim_start_matches('.') {
            "jpg" | "jpeg" => Self::Jpeg,
            _ => Self::Png,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }

    pub fn image_format(&self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
        }
    }
}

/// Save an image to disk in the requested format. JPEGs get a flattened RGB
/// representation since they have no alpha channel.
pub fn save_image(img: &DynamicImage, path: &Path, format: ImageFormatChoice) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match format {
        ImageFormatChoice::Jpeg => {
            let rgb = img.to_rgb8();
            rgb.save_with_format(path, ImageFormat::Jpeg)?;
        }
        ImageFormatChoice::Png => {
            img.save_with_format(path, ImageFormat::Png)?;
        }
    }
    Ok(())
}

/// Write a JSON manifest to disk (pretty-printed).
pub fn write_manifest_json<T: serde::Serialize>(value: &T, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = serde_json::to_string_pretty(value)?;
    std::fs::write(path, s)?;
    Ok(())
}
