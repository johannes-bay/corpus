use std::process::Command;

use anyhow::Result;
use tracing::warn;

#[derive(Debug, Default)]
pub struct VideoAnalysis {
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Width in pixels
    pub width: Option<i64>,
    /// Height in pixels
    pub height: Option<i64>,
    /// Frames per second
    pub fps: Option<f64>,
    /// Video codec name
    pub video_codec: Option<String>,
    /// Audio codec name (if audio track present)
    pub audio_codec: Option<String>,
    /// Total bitrate in kbps
    pub bitrate_kbps: Option<i64>,
    /// Whether the video has an audio track
    pub has_audio: Option<bool>,
    /// Aspect ratio as float
    pub aspect_ratio: Option<f64>,
}

/// Analyze a video file using ffprobe.
pub fn analyze(path: &str) -> VideoAnalysis {
    match analyze_inner(path) {
        Ok(a) => a,
        Err(e) => {
            warn!("Video analysis failed for {path}: {e}");
            VideoAnalysis::default()
        }
    }
}

fn analyze_inner(path: &str) -> Result<VideoAnalysis> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()?;

    if !output.status.success() {
        anyhow::bail!("ffprobe failed");
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    let mut analysis = VideoAnalysis::default();

    // Format-level info
    if let Some(format) = json.get("format") {
        analysis.duration = format
            .get("duration")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok());
        analysis.bitrate_kbps = format
            .get("bit_rate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .map(|b| b / 1000);
    }

    // Stream-level info
    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
        let mut found_audio = false;

        for stream in streams {
            let codec_type = stream
                .get("codec_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            match codec_type {
                "video" if analysis.video_codec.is_none() => {
                    analysis.video_codec = stream
                        .get("codec_name")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    analysis.width = stream
                        .get("width")
                        .and_then(|v| v.as_i64());
                    analysis.height = stream
                        .get("height")
                        .and_then(|v| v.as_i64());

                    // Parse frame rate from "25/1" or "30000/1001" format
                    if let Some(fps_str) = stream
                        .get("r_frame_rate")
                        .and_then(|v| v.as_str())
                    {
                        analysis.fps = parse_fps(fps_str);
                    }

                    if let (Some(w), Some(h)) = (analysis.width, analysis.height)
                        && h > 0
                    {
                        analysis.aspect_ratio = Some(w as f64 / h as f64);
                    }
                }
                "audio" if !found_audio => {
                    found_audio = true;
                    analysis.audio_codec = stream
                        .get("codec_name")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                }
                _ => {}
            }
        }

        analysis.has_audio = Some(found_audio);
    }

    Ok(analysis)
}

fn parse_fps(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().ok()?;
        let d: f64 = den.parse().ok()?;
        if d > 0.0 {
            return Some(n / d);
        }
    }
    s.parse().ok()
}
