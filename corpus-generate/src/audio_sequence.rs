use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

/// A clip to include in the generated sequence.
#[derive(Debug, Clone)]
pub struct SequenceClip {
    /// Path to the source audio file.
    pub path: String,
    /// Original BPM of the clip (if known).
    pub bpm: Option<f64>,
    /// Duration in seconds (if known).
    pub duration: Option<f64>,
    /// Start offset in seconds within the source file.
    pub start: f64,
    /// Length to use in seconds (0 = entire file).
    pub length: f64,
}

/// Configuration for audio sequence generation.
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Target BPM — all clips will be time-stretched to this tempo.
    /// If None, uses the seed's BPM.
    pub target_bpm: Option<f64>,
    /// Crossfade duration in seconds between clips.
    pub crossfade_secs: f64,
    /// Maximum duration per clip in seconds (0 = no limit).
    pub max_clip_secs: f64,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Output format (wav, mp3, flac).
    pub format: String,
    /// Fade in/out at start and end in seconds.
    pub fade_secs: f64,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            target_bpm: None,
            crossfade_secs: 2.0,
            max_clip_secs: 30.0,
            sample_rate: 44100,
            format: "wav".to_string(),
            fade_secs: 1.0,
        }
    }
}

/// Result of a sequence generation.
#[derive(Debug)]
pub struct SequenceResult {
    pub output_path: PathBuf,
    pub total_duration: f64,
    pub clips_used: usize,
}

/// Generate an audio sequence from a list of clips.
///
/// Workflow:
/// 1. For each clip, extract the segment and time-stretch to target BPM
/// 2. Concatenate with crossfades using ffmpeg
/// 3. Apply fade in/out to the final output
pub fn generate(
    clips: &[SequenceClip],
    config: &SequenceConfig,
    output_dir: &Path,
) -> Result<SequenceResult> {
    if clips.is_empty() {
        bail!("No clips provided");
    }

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create output dir: {}", output_dir.display()))?;

    let target_bpm = config.target_bpm.unwrap_or_else(|| {
        clips[0].bpm.unwrap_or(120.0)
    });

    info!("Generating sequence: {} clips, target BPM {target_bpm:.0}", clips.len());

    // Step 1: Prepare individual clips (extract segment + tempo adjust)
    let mut prepared: Vec<PathBuf> = Vec::new();
    for (i, clip) in clips.iter().enumerate() {
        let clip_path = output_dir.join(format!("_clip_{i:03}.wav"));
        prepare_clip(clip, target_bpm, config, &clip_path)?;
        if clip_path.exists() {
            prepared.push(clip_path);
        } else {
            warn!("Skipping clip {i}: preparation failed for {}", clip.path);
        }
    }

    if prepared.is_empty() {
        bail!("No clips were successfully prepared");
    }

    // Step 2: Concatenate with crossfades
    let output_path = output_dir.join(format!("sequence.{}", config.format));

    if prepared.len() == 1 {
        // Single clip — just copy/convert
        std::fs::copy(&prepared[0], &output_path)?;
    } else {
        concatenate_with_crossfade(&prepared, config.crossfade_secs, config.fade_secs, &output_path)?;
    }

    // Get duration of the final output
    let total_duration = probe_duration(&output_path).unwrap_or(0.0);

    // Clean up temp clips
    for p in &prepared {
        let _ = std::fs::remove_file(p);
    }

    info!("Sequence generated: {} ({total_duration:.1}s)", output_path.display());

    Ok(SequenceResult {
        output_path,
        total_duration,
        clips_used: prepared.len(),
    })
}

/// Prepare a single clip: extract segment and time-stretch to target BPM.
fn prepare_clip(
    clip: &SequenceClip,
    target_bpm: f64,
    config: &SequenceConfig,
    output: &Path,
) -> Result<()> {
    let src = &clip.path;
    if !Path::new(src).exists() {
        warn!("Source file does not exist: {src}");
        return Ok(());
    }

    let mut args: Vec<String> = vec![
        "-y".into(),
        "-i".into(),
        src.into(),
    ];

    // Seek to start offset
    if clip.start > 0.0 {
        args.extend(["-ss".into(), format!("{:.3}", clip.start)]);
    }

    // Limit duration
    let max_len = if clip.length > 0.0 {
        clip.length
    } else if config.max_clip_secs > 0.0 {
        config.max_clip_secs
    } else {
        0.0
    };
    if max_len > 0.0 {
        args.extend(["-t".into(), format!("{max_len:.3}")]);
    }

    // Build audio filter chain
    let mut filters: Vec<String> = Vec::new();

    // Time-stretch if we know the source BPM
    if let Some(src_bpm) = clip.bpm
        && src_bpm > 0.0 && (src_bpm - target_bpm).abs() > 0.5
    {
        let ratio = target_bpm / src_bpm;
        let tempo_filters = atempo_chain(ratio);
        filters.push(tempo_filters);
    }

    // Resample to target sample rate
    filters.push(format!("aresample={}", config.sample_rate));

    if !filters.is_empty() {
        args.extend(["-af".into(), filters.join(",")]);
    }

    // Output as WAV for lossless intermediate
    args.extend([
        "-ar".into(), config.sample_rate.to_string(),
        "-ac".into(), "2".into(),
        output.to_string_lossy().into_owned(),
    ]);

    let status = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .with_context(|| "ffmpeg not found")?;

    if !status.success() {
        warn!("ffmpeg clip preparation failed for {src}");
    }

    Ok(())
}

/// Generate an atempo filter chain for ratios outside 0.5-2.0.
fn atempo_chain(ratio: f64) -> String {
    if (0.5..=2.0).contains(&ratio) {
        return format!("atempo={ratio:.6}");
    }

    let mut remaining = ratio;
    let mut parts = Vec::new();

    while remaining > 2.0 {
        parts.push("atempo=2.0".to_string());
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        parts.push("atempo=0.5".to_string());
        remaining /= 0.5;
    }
    parts.push(format!("atempo={remaining:.6}"));
    parts.join(",")
}

/// Concatenate clips with crossfade using ffmpeg's acrossfade filter.
fn concatenate_with_crossfade(
    clips: &[PathBuf],
    crossfade_secs: f64,
    fade_secs: f64,
    output: &Path,
) -> Result<()> {
    if clips.len() < 2 {
        bail!("Need at least 2 clips for crossfade");
    }

    // For simplicity with many clips, use a concat approach with acrossfade
    // between each pair. Build a filter graph.
    let mut args: Vec<String> = Vec::new();

    // Add all input files
    for clip in clips {
        args.extend(["-i".into(), clip.to_string_lossy().into_owned()]);
    }

    // Build the filter graph
    let n = clips.len();
    let mut filter = String::new();

    // First crossfade: [0] and [1]
    filter.push_str(&format!(
        "[0:a][1:a]acrossfade=d={crossfade_secs}:c1=tri:c2=tri[a01]"
    ));

    // Chain subsequent crossfades
    let mut prev = "a01".to_string();
    for i in 2..n {
        let next = format!("a{:02}", i);
        filter.push_str(&format!(
            ";[{prev}][{i}:a]acrossfade=d={crossfade_secs}:c1=tri:c2=tri[{next}]"
        ));
        prev = next;
    }

    // Apply fade in/out to the final result
    if fade_secs > 0.0 {
        let faded = format!("{prev}_faded");
        // We need to know the duration for fade out, use afade with type=out
        filter.push_str(&format!(
            ";[{prev}]afade=t=in:st=0:d={fade_secs},afade=t=out:st=-{fade_secs}:d={fade_secs}[{faded}]"
        ));
        prev = faded;
    }

    args.extend([
        "-filter_complex".into(),
        filter,
        "-map".into(),
        format!("[{prev}]"),
        "-y".into(),
        output.to_string_lossy().into_owned(),
    ]);

    let result = Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .with_context(|| "ffmpeg not found")?;

    if !result.status.success() {
        // Fall back to simple concatenation without crossfade
        warn!("Crossfade failed, falling back to simple concatenation");
        return concatenate_simple(clips, output);
    }

    Ok(())
}

/// Simple concatenation fallback using ffmpeg concat demuxer.
fn concatenate_simple(clips: &[PathBuf], output: &Path) -> Result<()> {
    let list_path = output.with_extension("txt");
    let mut list_content = String::new();
    for clip in clips {
        list_content.push_str(&format!("file '{}'\n", clip.display()));
    }
    std::fs::write(&list_path, &list_content)?;

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", &list_path.to_string_lossy(),
            "-c", "copy",
            &output.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;

    let _ = std::fs::remove_file(&list_path);

    if !status.success() {
        bail!("ffmpeg concatenation failed");
    }

    Ok(())
}

/// Get audio duration via ffprobe.
fn probe_duration(path: &Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            &path.to_string_lossy(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()
}
