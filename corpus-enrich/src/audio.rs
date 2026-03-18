use std::process::Command;

use anyhow::{Context, Result};
use tracing::warn;

/// Result of analyzing a single audio file.
#[derive(Debug, Default)]
pub struct AudioAnalysis {
    pub bpm: Option<f64>,
    pub musical_key: Option<String>,
    pub key_numeric: Option<f64>,
    pub spectral_centroid: Option<f64>,
    pub duration: Option<f64>,
}

/// Circle-of-fifths encoding: maps key names to a numeric position 0..23.
/// Even numbers are major keys, odd are minor, arranged by fifths.
const KEY_MAP: &[(&str, f64)] = &[
    ("C", 0.0),
    ("Am", 1.0),
    ("G", 2.0),
    ("Em", 3.0),
    ("D", 4.0),
    ("Bm", 5.0),
    ("A", 6.0),
    ("F#m", 7.0),
    ("E", 8.0),
    ("C#m", 9.0),
    ("B", 10.0),
    ("G#m", 11.0),
    ("F#", 12.0),
    ("Ebm", 13.0),
    ("Db", 14.0),
    ("Bbm", 15.0),
    ("Ab", 16.0),
    ("Fm", 17.0),
    ("Eb", 18.0),
    ("Cm", 19.0),
    ("Bb", 20.0),
    ("Gm", 21.0),
    ("F", 22.0),
    ("Dm", 23.0),
];

/// Normalize common key name variants to our canonical form.
fn normalize_key(raw: &str) -> String {
    let s = raw.trim();
    // Handle various notations: "A minor" -> "Am", "F# minor" -> "F#m", "C major" -> "C"
    let s = s
        .replace(" minor", "m")
        .replace(" major", "")
        .replace(" min", "m")
        .replace(" maj", "");
    s.trim().to_string()
}

pub fn key_to_numeric(key: &str) -> Option<f64> {
    let normalized = normalize_key(key);
    KEY_MAP.iter().find(|(k, _)| *k == normalized).map(|(_, v)| *v)
}

/// Distance on the circle of fifths (0..12 range, wrapping).
pub fn key_distance(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    diff.min(24.0 - diff)
}

/// Extract BPM using `aubio tempo`.
fn extract_bpm(path: &str) -> Result<Option<f64>> {
    let output = Command::new("aubio")
        .args(["tempo", "-i", path])
        .output()
        .context("Failed to run aubio. Is it installed? (brew install aubio)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("aubio tempo failed for {path}: {stderr}");
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // aubio tempo outputs one BPM value per line; we take the last line
    // or the line that looks like a single float
    for line in stdout.lines().rev() {
        // aubio outputs "125.41 bpm" or just "125.41"
        let trimmed = line.trim().trim_end_matches("bpm").trim();
        if let Ok(bpm) = trimmed.parse::<f64>()
            && bpm > 0.0
            && bpm < 500.0
        {
            return Ok(Some(bpm));
        }
    }

    Ok(None)
}

/// Check once whether keyfinder-cli is available.
fn keyfinder_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let ok = Command::new("keyfinder-cli").arg("--help").output().is_ok();
        if !ok {
            tracing::warn!("keyfinder-cli not found — skipping key detection. Install libkeyfinder to enable.");
        }
        ok
    })
}

/// Extract musical key using `keyfinder-cli`.
fn extract_key(path: &str) -> Result<Option<(String, f64)>> {
    if !keyfinder_available() {
        return Ok(None);
    }

    let output = Command::new("keyfinder-cli")
        .arg(path)
        .output()
        .context("Failed to run keyfinder-cli")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("keyfinder-cli failed for {path}: {stderr}");
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let key_str = stdout.trim();
    if key_str.is_empty() || key_str.to_lowercase() == "silence" {
        return Ok(None);
    }

    let normalized = normalize_key(key_str);
    let numeric = key_to_numeric(&normalized);

    match numeric {
        Some(n) => Ok(Some((normalized, n))),
        None => {
            warn!("Unknown key: {key_str} (normalized: {normalized})");
            Ok(None)
        }
    }
}

/// Extract duration using ffprobe.
fn extract_duration(path: &str) -> Result<Option<f64>> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.trim().parse::<f64>().ok())
}

/// Extract spectral centroid using aubio mfcc (approximation via aubio's spectral features).
/// Falls back to a simpler ffprobe-based estimation if aubio doesn't provide it directly.
fn extract_spectral_centroid(path: &str) -> Result<Option<f64>> {
    // Use aubio's spectral descriptor: `aubio melbands` gives us frequency distribution
    // For a simple first pass, we use ffprobe to get the audio stats
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output();

    let sample_rate = match output {
        Ok(ref o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<f64>()
                .unwrap_or(44100.0)
        }
        _ => 44100.0,
    };

    // Use ffmpeg's astats filter to get a spectral estimate
    let output = Command::new("ffmpeg")
        .args([
            "-i", path,
            "-af", "aspectralstats=measure=centroid,ametadata=print:key=lavfi.aspectralstats.1.centroid:file=-",
            "-f", "null",
            "-t", "30", // analyze first 30 seconds
            "-",
        ])
        .output();

    match output {
        Ok(ref o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            // The metadata is printed to the file specified (stdout in this case)
            let combined = format!("{stdout}{stderr}");

            let mut sum = 0.0;
            let mut count = 0u64;
            for line in combined.lines() {
                if (line.contains("centroid=") || line.contains("centroid:"))
                    && let Some(val_str) = line.split('=').next_back().or_else(|| line.split(':').next_back())
                    && let Ok(val) = val_str.trim().parse::<f64>()
                {
                    // The value is normalized 0..1, scale by Nyquist
                    let hz = if val <= 1.0 { val * sample_rate / 2.0 } else { val };
                    if hz > 0.0 && hz < sample_rate / 2.0 {
                        sum += hz;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                Ok(Some(sum / count as f64))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Analyze a single audio file, extracting all available properties.
pub fn analyze(path: &str) -> AudioAnalysis {
    let mut result = AudioAnalysis::default();

    match extract_bpm(path) {
        Ok(bpm) => result.bpm = bpm,
        Err(e) => warn!("BPM extraction error for {path}: {e}"),
    }

    match extract_key(path) {
        Ok(Some((key, num))) => {
            result.musical_key = Some(key);
            result.key_numeric = Some(num);
        }
        Ok(None) => {}
        Err(e) => warn!("Key extraction error for {path}: {e}"),
    }

    match extract_spectral_centroid(path) {
        Ok(centroid) => result.spectral_centroid = centroid,
        Err(e) => warn!("Spectral centroid error for {path}: {e}"),
    }

    match extract_duration(path) {
        Ok(dur) => result.duration = dur,
        Err(e) => warn!("Duration extraction error for {path}: {e}"),
    }

    result
}
