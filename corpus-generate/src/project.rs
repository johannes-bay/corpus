use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;
use tracing::info;

use corpus_associate::matcher::ScoredMatch;
use corpus_db::queries;

use crate::audio_sequence::{self, SequenceClip, SequenceConfig};
use crate::moodboard::{self, MoodboardConfig, MoodboardEntry};

/// Configuration for project generation.
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// Whether to generate an audio sequence from audio matches.
    pub generate_audio: bool,
    /// Whether to generate a moodboard from image matches.
    pub generate_moodboard: bool,
    /// Whether to symlink matched files into the project folder.
    pub link_sources: bool,
    /// Audio sequence config.
    pub audio_config: SequenceConfig,
    /// Moodboard config.
    pub moodboard_config: MoodboardConfig,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            generate_audio: true,
            generate_moodboard: true,
            link_sources: true,
            audio_config: SequenceConfig::default(),
            moodboard_config: MoodboardConfig::default(),
        }
    }
}

/// Result of a project generation.
#[derive(Debug)]
pub struct ProjectResult {
    pub project_dir: PathBuf,
    pub audio_output: Option<PathBuf>,
    pub moodboard_output: Option<PathBuf>,
    pub manifest_path: PathBuf,
    pub sources_linked: usize,
}

const AUDIO_EXTS: &[&str] = &[".wav", ".mp3", ".m4a", ".aif", ".aiff", ".flac", ".ogg"];
const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"];

fn is_audio(ext: &Option<String>) -> bool {
    ext.as_deref()
        .map(|e| AUDIO_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_image(ext: &Option<String>) -> bool {
    ext.as_deref()
        .map(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Generate a full project from compose results.
///
/// Creates a structured folder:
/// ```text
/// project_dir/
/// ├── manifest.json        # Full compose results with metadata
/// ├── sequence.wav         # Generated audio sequence (if audio matches)
/// ├── moodboard.png        # Generated moodboard (if image matches)
/// └── sources/             # Symlinks to matched source files
///     ├── 01_filename.wav
///     ├── 02_filename.jpg
///     └── ...
/// ```
pub fn generate(
    conn: &Connection,
    seed_path: &str,
    matches: &[ScoredMatch],
    config: &ProjectConfig,
    output_dir: &Path,
) -> Result<ProjectResult> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create: {}", output_dir.display()))?;

    let seed_file = queries::get_file(conn, seed_path)?;

    // Separate matches by media type
    let audio_matches: Vec<&ScoredMatch> = matches
        .iter()
        .filter(|m| is_audio(&m.file.extension))
        .collect();
    let image_matches: Vec<&ScoredMatch> = matches
        .iter()
        .filter(|m| is_image(&m.file.extension))
        .collect();

    info!(
        "Project: {} audio, {} image matches",
        audio_matches.len(),
        image_matches.len()
    );

    // Generate audio sequence
    let mut audio_output = None;
    if config.generate_audio && !audio_matches.is_empty() {
        let clips: Vec<SequenceClip> = audio_matches
            .iter()
            .map(|m| {
                let bpm = m
                    .axis_scores
                    .iter()
                    .find(|(name, _)| name == "bpm")
                    .and_then(|_| {
                        // Look up the actual BPM from the match's properties
                        queries::get_properties(conn, &m.file.path)
                            .ok()
                            .and_then(|props| {
                                props
                                    .iter()
                                    .find(|p| p.domain == "audio" && p.key == "bpm")
                                    .and_then(|p| p.value_num)
                            })
                    });
                let duration = queries::get_properties(conn, &m.file.path)
                    .ok()
                    .and_then(|props| {
                        props
                            .iter()
                            .find(|p| p.domain == "audio" && p.key == "duration")
                            .and_then(|p| p.value_num)
                    });
                SequenceClip {
                    path: m.file.path.clone(),
                    bpm,
                    duration,
                    start: 0.0,
                    length: 0.0, // use max_clip_secs from config
                }
            })
            .collect();

        match audio_sequence::generate(&clips, &config.audio_config, output_dir) {
            Ok(result) => {
                audio_output = Some(result.output_path);
            }
            Err(e) => {
                tracing::warn!("Audio sequence generation failed: {e}");
            }
        }
    }

    // Generate moodboard
    let mut moodboard_output = None;
    if config.generate_moodboard && !image_matches.is_empty() {
        let seed_entry = seed_file.as_ref().and_then(|f| {
            if is_image(&f.extension) {
                Some(MoodboardEntry {
                    path: f.path.clone(),
                    label: Some("SEED".to_string()),
                })
            } else {
                None
            }
        });

        let entries: Vec<MoodboardEntry> = image_matches
            .iter()
            .enumerate()
            .map(|(i, m)| MoodboardEntry {
                path: m.file.path.clone(),
                label: Some(format!("#{} {:.2}", i + 1, m.total_score)),
            })
            .collect();

        match moodboard::generate(
            seed_entry.as_ref(),
            &entries,
            &config.moodboard_config,
            output_dir,
        ) {
            Ok(result) => {
                moodboard_output = Some(result.output_path);
            }
            Err(e) => {
                tracing::warn!("Moodboard generation failed: {e}");
            }
        }
    }

    // Symlink source files
    let mut sources_linked = 0;
    if config.link_sources {
        let sources_dir = output_dir.join("sources");
        std::fs::create_dir_all(&sources_dir)?;

        for (i, m) in matches.iter().enumerate() {
            let src = Path::new(&m.file.path);
            if !src.exists() {
                continue;
            }
            let link_path = sources_dir.join(format!("{:02}_{}", i + 1, sanitize_filename(&m.file.filename)));

            if std::os::unix::fs::symlink(src, &link_path).is_ok() {
                sources_linked += 1;
            }
        }
    }

    // Write manifest
    let manifest_path = output_dir.join("manifest.json");
    let manifest = build_manifest(seed_path, &seed_file, matches);
    std::fs::write(&manifest_path, manifest)?;

    info!("Project generated at {}", output_dir.display());

    Ok(ProjectResult {
        project_dir: output_dir.to_path_buf(),
        audio_output,
        moodboard_output,
        manifest_path,
        sources_linked,
    })
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

fn build_manifest(
    seed_path: &str,
    seed_file: &Option<corpus_db::models::FileEntry>,
    matches: &[ScoredMatch],
) -> String {
    let seed_info = seed_file.as_ref().map(|f| {
        serde_json::json!({
            "path": f.path,
            "filename": f.filename,
            "extension": f.extension,
            "size_bytes": f.size_bytes,
        })
    });

    let match_entries: Vec<serde_json::Value> = matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            serde_json::json!({
                "rank": i + 1,
                "path": m.file.path,
                "filename": m.file.filename,
                "score": m.total_score,
                "axes": m.axis_scores.iter()
                    .map(|(name, score)| serde_json::json!({"name": name, "score": score}))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    let manifest = serde_json::json!({
        "seed": seed_path,
        "seed_info": seed_info,
        "matches": match_entries,
        "generated_at": chrono_now(),
    });

    serde_json::to_string_pretty(&manifest).unwrap_or_default()
}

/// Simple ISO 8601 timestamp without pulling in chrono.
fn chrono_now() -> String {
    let output = std::process::Command::new("date")
        .args(["+%Y-%m-%dT%H:%M:%S"])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
