use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::{DynamicImage, RgbaImage};
use tracing::{info, warn};

/// Configuration for moodboard generation.
#[derive(Debug, Clone)]
pub struct MoodboardConfig {
    /// Number of columns in the grid.
    pub columns: u32,
    /// Tile size in pixels (each image is resized to fit this square).
    pub tile_size: u32,
    /// Gap between tiles in pixels.
    pub gap: u32,
    /// Background color (R, G, B).
    pub bg_color: [u8; 3],
    /// Output format (png, jpg).
    pub format: String,
}

impl Default for MoodboardConfig {
    fn default() -> Self {
        Self {
            columns: 5,
            tile_size: 300,
            gap: 4,
            bg_color: [14, 14, 16], // matches UI dark bg
            format: "png".to_string(),
        }
    }
}

/// Result of a moodboard generation.
#[derive(Debug)]
pub struct MoodboardResult {
    pub output_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub images_used: usize,
}

/// An image to include in the moodboard.
#[derive(Debug, Clone)]
pub struct MoodboardEntry {
    /// Path to the image file.
    pub path: String,
    /// Optional label (filename, score, etc.)
    pub label: Option<String>,
}

/// Generate a moodboard/contact sheet from a list of images.
pub fn generate(
    seed: Option<&MoodboardEntry>,
    entries: &[MoodboardEntry],
    config: &MoodboardConfig,
    output_dir: &Path,
) -> Result<MoodboardResult> {
    if entries.is_empty() && seed.is_none() {
        bail!("No images provided");
    }

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create output dir: {}", output_dir.display()))?;

    // Build the full list: seed first (if any), then entries
    let mut all_entries: Vec<&MoodboardEntry> = Vec::new();
    if let Some(s) = seed {
        all_entries.push(s);
    }
    all_entries.extend(entries.iter());

    let total = all_entries.len() as u32;
    let cols = config.columns;
    let rows = total.div_ceil(cols);
    let tile = config.tile_size;
    let gap = config.gap;

    let canvas_w = cols * tile + (cols + 1) * gap;
    let canvas_h = rows * tile + (rows + 1) * gap;

    info!("Generating moodboard: {total} images, {cols}x{rows} grid, {canvas_w}x{canvas_h}px");

    let mut canvas = RgbaImage::from_pixel(
        canvas_w,
        canvas_h,
        image::Rgba([config.bg_color[0], config.bg_color[1], config.bg_color[2], 255]),
    );

    let mut images_used = 0;
    for (i, entry) in all_entries.iter().enumerate() {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        let x = gap + col * (tile + gap);
        let y = gap + row * (tile + gap);

        match load_and_fit(&entry.path, tile) {
            Ok(fitted) => {
                // Center the fitted image within the tile
                let (fw, fh) = fitted.dimensions();
                let offset_x = (tile - fw) / 2;
                let offset_y = (tile - fh) / 2;
                image::imageops::overlay(&mut canvas, &fitted, (x + offset_x) as i64, (y + offset_y) as i64);
                images_used += 1;
            }
            Err(e) => {
                warn!("Cannot load {}: {e}", entry.path);
            }
        }
    }

    let output_path = output_dir.join(format!("moodboard.{}", config.format));
    let dynamic = DynamicImage::ImageRgba8(canvas);

    match config.format.as_str() {
        "jpg" | "jpeg" => {
            let rgb = dynamic.to_rgb8();
            rgb.save(&output_path)
                .with_context(|| format!("Cannot save {}", output_path.display()))?;
        }
        _ => {
            dynamic
                .save(&output_path)
                .with_context(|| format!("Cannot save {}", output_path.display()))?;
        }
    }

    info!("Moodboard saved: {} ({images_used} images)", output_path.display());

    Ok(MoodboardResult {
        output_path,
        width: canvas_w,
        height: canvas_h,
        images_used,
    })
}

/// Load an image and resize it to fit within a square, preserving aspect ratio.
fn load_and_fit(path: &str, max_size: u32) -> Result<RgbaImage> {
    let img = image::open(Path::new(path))
        .with_context(|| format!("Cannot open: {path}"))?;

    let resized = img.thumbnail(max_size, max_size);
    Ok(resized.to_rgba8())
}
