use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use image::GenericImageView;
use tracing::warn;

#[derive(Debug, Default)]
pub struct ImageAnalysis {
    /// Average brightness (0.0-1.0, perceptual luminance)
    pub brightness: Option<f64>,
    /// Dominant color as hex string (e.g. "#4a7c3f")
    pub dominant_color: Option<String>,
    /// Dominant color hue in degrees (0-360)
    pub dominant_hue: Option<f64>,
    /// Dominant color saturation (0.0-1.0)
    pub dominant_saturation: Option<f64>,
    /// Color temperature estimate: negative=cool/blue, positive=warm/yellow
    pub color_temperature: Option<f64>,
    /// Aspect ratio as a float (width/height)
    pub aspect_ratio: Option<f64>,
    /// Width in pixels
    pub width: Option<u32>,
    /// Height in pixels
    pub height: Option<u32>,
    /// Number of distinct color clusters (palette complexity)
    pub palette_complexity: Option<f64>,
}

/// Convert RGB to HSL. Returns (hue: 0-360, saturation: 0-1, lightness: 0-1).
fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < f64::EPSILON {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < f64::EPSILON {
        let mut h = (g - b) / d;
        if g < b {
            h += 6.0;
        }
        h
    } else if (max - g).abs() < f64::EPSILON {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    (h * 60.0, s, l)
}

/// Perceptual luminance (ITU-R BT.709).
fn luminance(r: u8, g: u8, b: u8) -> f64 {
    0.2126 * (r as f64 / 255.0) + 0.7152 * (g as f64 / 255.0) + 0.0722 * (b as f64 / 255.0)
}

/// Estimate color temperature from RGB.
/// Positive = warm (reds/yellows), negative = cool (blues).
/// Range roughly -1.0 to 1.0.
fn color_temp(avg_r: f64, avg_g: f64, avg_b: f64) -> f64 {
    // Simple heuristic: warm if red > blue, cool if blue > red
    let total = avg_r + avg_g + avg_b;
    if total < 1.0 {
        return 0.0;
    }
    (avg_r - avg_b) / total
}

/// Quantize an RGB color to a reduced palette bucket for clustering.
fn quantize(r: u8, g: u8, b: u8, bits: u8) -> (u8, u8, u8) {
    let shift = 8 - bits;
    let mask = !((1u8 << shift) - 1);
    (r & mask, g & mask, b & mask)
}

type Rgb = (u8, u8, u8);
/// Running sums: (r_sum, g_sum, b_sum, count)
type ColorAccum = (u64, u64, u64, u64);

/// Simple dominant color via histogram bucketing.
/// Faster than real k-means for our throughput needs.
fn find_dominant_color(pixels: &[Rgb]) -> Option<Rgb> {
    let mut buckets: HashMap<Rgb, ColorAccum> = HashMap::new();

    for &(r, g, b) in pixels {
        let key = quantize(r, g, b, 4); // 16 levels per channel = 4096 buckets
        let entry = buckets.entry(key).or_default();
        entry.0 += r as u64;
        entry.1 += g as u64;
        entry.2 += b as u64;
        entry.3 += 1;
    }

    buckets
        .values()
        .max_by_key(|v| v.3)
        .map(|v| {
            let count = v.3;
            ((v.0 / count) as u8, (v.1 / count) as u8, (v.2 / count) as u8)
        })
}

/// Count distinct color buckets as a measure of palette complexity.
fn palette_complexity(pixels: &[(u8, u8, u8)]) -> f64 {
    let mut buckets: std::collections::HashSet<(u8, u8, u8)> = std::collections::HashSet::new();
    for &(r, g, b) in pixels {
        buckets.insert(quantize(r, g, b, 3)); // 8 levels = 512 possible buckets
    }
    buckets.len() as f64
}

/// Analyze an image file. Resizes to a small thumbnail for fast processing.
pub fn analyze(path: &str) -> ImageAnalysis {
    match analyze_inner(path) {
        Ok(a) => a,
        Err(e) => {
            warn!("Image analysis failed for {path}: {e}");
            ImageAnalysis::default()
        }
    }
}

fn analyze_inner(path: &str) -> Result<ImageAnalysis> {
    let img = image::open(Path::new(path))
        .with_context(|| format!("Cannot open image: {path}"))?;

    let (orig_w, orig_h) = img.dimensions();

    // Resize to max 256px on longest side for fast analysis
    let thumb = img.thumbnail(256, 256);
    let (tw, th) = thumb.dimensions();

    let rgb = thumb.to_rgb8();
    let mut pixels: Vec<(u8, u8, u8)> = Vec::with_capacity((tw * th) as usize);
    let mut lum_sum = 0.0;
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;

    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        pixels.push((r, g, b));
        lum_sum += luminance(r, g, b);
        r_sum += r as f64;
        g_sum += g as f64;
        b_sum += b as f64;
    }

    let n = pixels.len() as f64;
    if n == 0.0 {
        return Ok(ImageAnalysis::default());
    }

    let brightness = lum_sum / n;
    let temperature = color_temp(r_sum / n, g_sum / n, b_sum / n);
    let aspect = orig_w as f64 / orig_h as f64;
    let complexity = palette_complexity(&pixels);

    let (dominant_color, dominant_hue, dominant_sat) = match find_dominant_color(&pixels) {
        Some((r, g, b)) => {
            let hex = format!("#{r:02x}{g:02x}{b:02x}");
            let (h, s, _) = rgb_to_hsl(r, g, b);
            (Some(hex), Some(h), Some(s))
        }
        None => (None, None, None),
    };

    Ok(ImageAnalysis {
        brightness: Some(brightness),
        dominant_color,
        dominant_hue,
        dominant_saturation: dominant_sat,
        color_temperature: Some(temperature),
        aspect_ratio: Some(aspect),
        width: Some(orig_w),
        height: Some(orig_h),
        palette_complexity: Some(complexity),
    })
}
