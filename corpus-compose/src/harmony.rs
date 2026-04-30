//! Tonal/color/audio harmonic consistency checks.
//!
//! Lightweight scoring functions that quantify how well a set of members
//! "go together" so the strategy layer can re-rank candidates. Real palette
//! and key analysis lives in corpus-enrich; here we just score what is
//! already there (or approximate from the rendered images).

use image::RgbImage;

/// Mean RGB of an image.
pub fn mean_rgb(img: &RgbImage) -> [f64; 3] {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return [0.0; 3];
    }
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    // Sample at most 10000 pixels to keep this cheap on full-res inputs.
    let step = ((w * h) / 10_000).max(1);
    let mut idx: u32 = 0;
    for p in img.pixels() {
        if idx.is_multiple_of(step) {
            sum[0] += p[0] as u64;
            sum[1] += p[1] as u64;
            sum[2] += p[2] as u64;
            count += 1;
        }
        idx = idx.wrapping_add(1);
    }
    if count == 0 {
        return [0.0; 3];
    }
    [
        sum[0] as f64 / count as f64,
        sum[1] as f64 / count as f64,
        sum[2] as f64 / count as f64,
    ]
}

/// Approximate ΔE in RGB space (not Lab; cheap and order-preserving for
/// palette-coherence scoring).
pub fn rgb_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Score palette coherence in [0, 1] across a set of mean-RGB samples.
/// 1.0 = identical, 0.0 = maximally divergent.
pub fn palette_coherence(samples: &[[f64; 3]]) -> f64 {
    if samples.len() < 2 {
        return 1.0;
    }
    // Centroid
    let mut centroid = [0.0; 3];
    for s in samples {
        centroid[0] += s[0];
        centroid[1] += s[1];
        centroid[2] += s[2];
    }
    let n = samples.len() as f64;
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    // Mean distance from centroid, normalized by max possible (~441 in RGB).
    let mut dist_sum = 0.0;
    for s in samples {
        dist_sum += rgb_distance(*s, centroid);
    }
    let mean_dist = dist_sum / n;
    let max_dist = 441.673; // sqrt(255^2 * 3)
    (1.0 - (mean_dist / max_dist)).clamp(0.0, 1.0)
}

/// BPM compatibility in [0, 1]. Two BPMs are 1.0 when equal, fall off with
/// half-time and double-time tolerance.
pub fn bpm_compatibility(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.5;
    }
    let candidates = [b, b * 2.0, b / 2.0];
    let best = candidates
        .iter()
        .map(|&c| (a - c).abs())
        .fold(f64::MAX, f64::min);
    let max_tolerance = 20.0;
    (1.0 - (best / max_tolerance)).clamp(0.0, 1.0)
}
