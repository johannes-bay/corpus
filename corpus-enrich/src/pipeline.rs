use std::sync::Mutex;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rusqlite::Connection;
use tracing::info;

use corpus_db::models::FileEntry;
use corpus_db::queries;

use crate::audio;
use crate::image as img;

const AUDIO_EXTENSIONS: &[&str] = &[".wav", ".mp3", ".m4a", ".aif", ".aiff", ".flac", ".ogg"];

/// Run the kf-worker subprocess for a single file.
/// Returns Some((key_name, cof_position)) on success, None on failure/silence.
/// Segfaults in libkeyfinder only kill the child process.
fn run_kf_worker(
    worker_bin: &std::path::Path,
    path: &str,
    max_seconds: &str,
) -> Option<(String, f64)> {
    let output = std::process::Command::new(worker_bin)
        .args([path, max_seconds])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    if line.is_empty() {
        return None;
    }

    let mut parts = line.split('\t');
    let name = parts.next()?.to_string();
    let cof: f64 = parts.next()?.parse().ok()?;

    Some((name, cof))
}

fn make_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

/// Run audio enrichment (BPM, spectral, duration) on all files that haven't been processed yet.
pub fn enrich_audio(conn: &Mutex<Connection>, concurrency: usize) -> Result<()> {
    let files: Vec<FileEntry> = {
        let db = conn.lock().unwrap();
        queries::get_files_without_property(&db, "audio", "bpm", AUDIO_EXTENSIONS)?
    };

    let total = files.len();
    if total == 0 {
        info!("All audio files already enriched.");
        return Ok(());
    }

    info!("Found {total} audio files to enrich");

    // Pre-load existing audio_meta durations so we can skip ffprobe when possible
    let existing_durations: std::collections::HashMap<String, f64> = {
        let db = conn.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT path, duration_secs FROM audio_meta WHERE duration_secs IS NOT NULL AND duration_secs > 0",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        rows.filter_map(|r| r.ok()).collect()
    };

    let pb = make_progress_bar(total as u64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()?;

    let batch_size = 100;
    for chunk in files.chunks(batch_size) {
        let results: Vec<(String, audio::AudioAnalysis, Option<f64>)> = pool.install(|| {
            chunk
                .par_iter()
                .map(|file| {
                    pb.set_message(file.filename.clone());
                    let analysis = audio::analyze(&file.path);
                    let existing_dur = existing_durations.get(&file.path).copied();
                    pb.inc(1);
                    (file.path.clone(), analysis, existing_dur)
                })
                .collect()
        });

        let db = conn.lock().unwrap();
        let tx = db.unchecked_transaction()?;
        for (path, analysis, existing_dur) in &results {
            if let Some(bpm) = analysis.bpm {
                queries::set_property(&tx, path, "audio", "bpm", Some(bpm), None)?;
            } else {
                queries::set_property(&tx, path, "audio", "bpm", Some(-1.0), Some("undetected"))?;
            }

            if let Some(ref key) = analysis.musical_key {
                queries::set_property(
                    &tx,
                    path,
                    "audio",
                    "musical_key",
                    analysis.key_numeric,
                    Some(key),
                )?;
            }

            if let Some(centroid) = analysis.spectral_centroid {
                queries::set_property(&tx, path, "audio", "spectral_centroid", Some(centroid), None)?;
            }

            let duration = existing_dur.or(analysis.duration);
            if let Some(dur) = duration {
                queries::set_property(&tx, path, "audio", "duration", Some(dur), None)?;
            }
        }
        tx.commit()?;
    }

    pb.finish_with_message("done");
    info!("Audio enrichment complete");
    Ok(())
}

/// Run musical key detection on all audio files that don't have a key yet.
/// Uses libkeyfinder via symphonia decoding — no shell-outs.
pub fn enrich_keys(conn: &Mutex<Connection>, concurrency: usize, max_seconds: f64) -> Result<()> {
    let files: Vec<FileEntry> = {
        let db = conn.lock().unwrap();
        queries::get_files_without_property(&db, "audio", "musical_key", AUDIO_EXTENSIONS)?
    };

    let total = files.len();
    if total == 0 {
        info!("All audio files already have key detection.");
        return Ok(());
    }

    info!("Found {total} audio files for key detection (analyzing up to {max_seconds}s each)");

    let pb = make_progress_bar(total as u64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()?;

    // Resolve the kf-worker binary path (next to the current executable)
    let worker_bin = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("kf-worker")))
        .unwrap_or_else(|| std::path::PathBuf::from("kf-worker"));

    if !worker_bin.exists() {
        anyhow::bail!(
            "kf-worker binary not found at {}. Build with `cargo build --release`.",
            worker_bin.display()
        );
    }

    let max_sec_str = max_seconds.to_string();

    let batch_size = 100;
    for chunk in files.chunks(batch_size) {
        let results: Vec<(String, Option<(String, f64)>)> = pool.install(|| {
            chunk
                .par_iter()
                .map(|file| {
                    pb.set_message(file.filename.clone());
                    let result = run_kf_worker(&worker_bin, &file.path, &max_sec_str);
                    pb.inc(1);
                    (file.path.clone(), result)
                })
                .collect()
        });

        let db = conn.lock().unwrap();
        let tx = db.unchecked_transaction()?;
        for (path, key_result) in &results {
            match key_result {
                Some((name, cof)) => {
                    queries::set_property(
                        &tx,
                        path,
                        "audio",
                        "musical_key",
                        Some(*cof),
                        Some(name),
                    )?;
                }
                None => {
                    queries::set_property(
                        &tx,
                        path,
                        "audio",
                        "musical_key",
                        Some(-1.0),
                        Some("undetected"),
                    )?;
                }
            }
        }
        tx.commit()?;
    }

    pb.finish_with_message("done");
    info!("Key detection complete");
    Ok(())
}

const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp",
];

/// Run image enrichment on all image files that haven't been processed yet.
pub fn enrich_images(conn: &Mutex<Connection>, concurrency: usize) -> Result<()> {
    let files: Vec<FileEntry> = {
        let db = conn.lock().unwrap();
        queries::get_files_without_property(&db, "image", "brightness", IMAGE_EXTENSIONS)?
    };

    let total = files.len();
    if total == 0 {
        info!("All image files already enriched.");
        return Ok(());
    }

    info!("Found {total} image files to enrich");

    let pb = make_progress_bar(total as u64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()?;

    let batch_size = 100;
    for chunk in files.chunks(batch_size) {
        let results: Vec<(String, img::ImageAnalysis)> = pool.install(|| {
            chunk
                .par_iter()
                .map(|file| {
                    pb.set_message(file.filename.clone());
                    let analysis = img::analyze(&file.path);
                    pb.inc(1);
                    (file.path.clone(), analysis)
                })
                .collect()
        });

        let db = conn.lock().unwrap();
        let tx = db.unchecked_transaction()?;
        for (path, a) in &results {
            if let Some(v) = a.brightness {
                queries::set_property(&tx, path, "image", "brightness", Some(v), None)?;
            } else {
                // Sentinel for files we couldn't process
                queries::set_property(&tx, path, "image", "brightness", Some(-1.0), Some("failed"))?;
            }

            if let Some(ref hex) = a.dominant_color {
                queries::set_property(&tx, path, "image", "dominant_color", None, Some(hex))?;
            }
            if let Some(hue) = a.dominant_hue {
                queries::set_property(&tx, path, "image", "dominant_hue", Some(hue), None)?;
            }
            if let Some(sat) = a.dominant_saturation {
                queries::set_property(&tx, path, "image", "dominant_saturation", Some(sat), None)?;
            }
            if let Some(temp) = a.color_temperature {
                queries::set_property(&tx, path, "image", "color_temperature", Some(temp), None)?;
            }
            if let Some(ar) = a.aspect_ratio {
                queries::set_property(&tx, path, "image", "aspect_ratio", Some(ar), None)?;
            }
            if let Some(w) = a.width {
                queries::set_property(&tx, path, "image", "width", Some(w as f64), None)?;
            }
            if let Some(h) = a.height {
                queries::set_property(&tx, path, "image", "height", Some(h as f64), None)?;
            }
            if let Some(c) = a.palette_complexity {
                queries::set_property(&tx, path, "image", "palette_complexity", Some(c), None)?;
            }
        }
        tx.commit()?;
    }

    pb.finish_with_message("done");
    info!("Image enrichment complete");
    Ok(())
}

/// Print enrichment statistics.
pub fn print_stats(conn: &Connection) -> Result<()> {
    let mut total_audio = 0i64;
    for ext in AUDIO_EXTENSIONS {
        total_audio += queries::count_files_by_ext(conn, ext)?;
    }

    let bpm_count = queries::count_enriched(conn, "audio", "bpm")?;
    let key_count = queries::count_enriched(conn, "audio", "musical_key")?;
    let centroid_count = queries::count_enriched(conn, "audio", "spectral_centroid")?;
    let duration_count = queries::count_enriched(conn, "audio", "duration")?;

    println!("Enrichment Statistics");
    println!("=====================");
    println!("Audio files total:       {total_audio}");
    println!("  BPM detected:          {bpm_count}");
    println!("  Musical key detected:  {key_count}");
    println!("  Spectral centroid:     {centroid_count}");
    println!("  Duration:              {duration_count}");

    if total_audio > 0 {
        let pct = (bpm_count as f64 / total_audio as f64) * 100.0;
        println!("  Progress:              {pct:.1}%");
    }

    println!();

    let mut total_images = 0i64;
    for ext in IMAGE_EXTENSIONS {
        total_images += queries::count_files_by_ext(conn, ext)?;
    }

    let bright_count = queries::count_enriched(conn, "image", "brightness")?;
    let color_count = queries::count_enriched(conn, "image", "dominant_color")?;
    let temp_count = queries::count_enriched(conn, "image", "color_temperature")?;

    println!("Image files total:       {total_images}");
    println!("  Brightness:            {bright_count}");
    println!("  Dominant color:        {color_count}");
    println!("  Color temperature:     {temp_count}");

    if total_images > 0 {
        let pct = (bright_count as f64 / total_images as f64) * 100.0;
        println!("  Progress:              {pct:.1}%");
    }

    Ok(())
}
