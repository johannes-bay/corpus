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
use crate::text;
use crate::video;

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

const DOCUMENT_EXTENSIONS: &[&str] = &[
    ".pdf", ".txt", ".rtf", ".doc", ".docx", ".pages", ".epub", ".mobi", ".md",
];

const VIDEO_EXTENSIONS: &[&str] = &[".mov", ".mp4", ".avi", ".mkv", ".wmv", ".webm", ".m4v"];

/// Run text/document enrichment on all document files that haven't been processed yet.
pub fn enrich_documents(conn: &Mutex<Connection>, concurrency: usize) -> Result<()> {
    let files: Vec<FileEntry> = {
        let db = conn.lock().unwrap();
        queries::get_files_without_property(&db, "text", "word_count", DOCUMENT_EXTENSIONS)?
    };

    let total = files.len();
    if total == 0 {
        info!("All document files already enriched.");
        return Ok(());
    }

    info!("Found {total} document files to enrich");

    let pb = make_progress_bar(total as u64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()?;

    // Pre-load existing document_meta for page counts / titles / authors
    type DocMeta = (Option<i64>, Option<String>, Option<String>);
    let existing_meta: std::collections::HashMap<String, DocMeta> = {
        let db = conn.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT path, page_count, title, author FROM document_meta",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;
        rows.filter_map(|r| r.ok())
            .map(|(path, pages, title, author)| (path, (pages, title, author)))
            .collect()
    };

    let batch_size = 100;
    for chunk in files.chunks(batch_size) {
        let results: Vec<(String, text::TextAnalysis)> = pool.install(|| {
            chunk
                .par_iter()
                .filter_map(|file| {
                    pb.set_message(file.filename.clone());
                    let ext = file.extension.as_deref().unwrap_or("");
                    let path = file.path.clone();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        text::analyze(&path, ext)
                    }));
                    pb.inc(1);
                    match result {
                        Ok(analysis) => Some((path, analysis)),
                        Err(_) => {
                            tracing::warn!("Panic during text analysis of {path}");
                            None
                        }
                    }
                })
                .collect()
        });

        let db = conn.lock().unwrap();
        let tx = db.unchecked_transaction()?;
        for (path, a) in &results {
            // Word count (primary enrichment marker)
            let wc = a.word_count.unwrap_or(0);
            queries::set_property(&tx, path, "text", "word_count", Some(wc as f64), None)?;

            if let Some(cc) = a.char_count {
                queries::set_property(&tx, path, "text", "char_count", Some(cc as f64), None)?;
            }

            // Page count: prefer our extraction, fall back to existing metadata
            let page_count = a.page_count.or_else(|| {
                existing_meta.get(path.as_str()).and_then(|(p, _, _)| *p)
            });
            if let Some(pc) = page_count {
                queries::set_property(&tx, path, "text", "page_count", Some(pc as f64), None)?;
            }

            if let Some(ref lang) = a.language {
                queries::set_property(&tx, path, "text", "language", None, Some(lang))?;
            }

            // Title/author: prefer extraction, fall back to existing metadata
            let title = a.title.as_deref().or_else(|| {
                existing_meta.get(path.as_str()).and_then(|(_, t, _)| t.as_deref())
            });
            if let Some(t) = title {
                queries::set_property(&tx, path, "text", "title", None, Some(t))?;
            }

            let author = a.author.as_deref().or_else(|| {
                existing_meta.get(path.as_str()).and_then(|(_, _, au)| au.as_deref())
            });
            if let Some(au) = author {
                queries::set_property(&tx, path, "text", "author", None, Some(au))?;
            }
        }
        tx.commit()?;
    }

    pb.finish_with_message("done");
    info!("Document enrichment complete");
    Ok(())
}

/// Run video enrichment on all video files that haven't been processed yet.
pub fn enrich_videos(conn: &Mutex<Connection>, concurrency: usize) -> Result<()> {
    let files: Vec<FileEntry> = {
        let db = conn.lock().unwrap();
        queries::get_files_without_property(&db, "video", "duration", VIDEO_EXTENSIONS)?
    };

    let total = files.len();
    if total == 0 {
        info!("All video files already enriched.");
        return Ok(());
    }

    info!("Found {total} video files to enrich");

    let pb = make_progress_bar(total as u64);

    // Pre-load existing video_meta for duration/resolution
    type VidMeta = (Option<f64>, Option<i64>, Option<i64>);
    let existing_meta: std::collections::HashMap<String, VidMeta> = {
        let db = conn.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT path, duration_secs, width, height FROM video_meta",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<f64>>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<i64>>(3)?,
            ))
        })?;
        rows.filter_map(|r| r.ok())
            .map(|(path, dur, w, h)| (path, (dur, w, h)))
            .collect()
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()?;

    let batch_size = 50;
    for chunk in files.chunks(batch_size) {
        let results: Vec<(String, video::VideoAnalysis)> = pool.install(|| {
            chunk
                .par_iter()
                .map(|file| {
                    pb.set_message(file.filename.clone());
                    let analysis = video::analyze(&file.path);
                    pb.inc(1);
                    (file.path.clone(), analysis)
                })
                .collect()
        });

        let db = conn.lock().unwrap();
        let tx = db.unchecked_transaction()?;
        for (path, a) in &results {
            let existing = existing_meta.get(path.as_str());

            // Duration: prefer ffprobe, fall back to existing metadata
            let duration = a.duration.or_else(|| existing.and_then(|(d, _, _)| *d));
            if let Some(dur) = duration {
                queries::set_property(&tx, path, "video", "duration", Some(dur), None)?;
            } else {
                queries::set_property(&tx, path, "video", "duration", Some(-1.0), Some("undetected"))?;
            }

            // Resolution
            let width = a.width.or_else(|| existing.and_then(|(_, w, _)| *w));
            let height = a.height.or_else(|| existing.and_then(|(_, _, h)| *h));
            if let Some(w) = width {
                queries::set_property(&tx, path, "video", "width", Some(w as f64), None)?;
            }
            if let Some(h) = height {
                queries::set_property(&tx, path, "video", "height", Some(h as f64), None)?;
            }

            if let Some(fps) = a.fps {
                queries::set_property(&tx, path, "video", "fps", Some(fps), None)?;
            }

            if let Some(ref codec) = a.video_codec {
                queries::set_property(&tx, path, "video", "video_codec", None, Some(codec))?;
            }

            if let Some(has_audio) = a.has_audio {
                queries::set_property(&tx, path, "video", "has_audio", Some(if has_audio { 1.0 } else { 0.0 }), None)?;
            }

            if let Some(ar) = a.aspect_ratio {
                queries::set_property(&tx, path, "video", "aspect_ratio", Some(ar), None)?;
            }

            if let Some(br) = a.bitrate_kbps {
                queries::set_property(&tx, path, "video", "bitrate_kbps", Some(br as f64), None)?;
            }
        }
        tx.commit()?;
    }

    pb.finish_with_message("done");
    info!("Video enrichment complete");
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

    println!();

    let mut total_docs = 0i64;
    for ext in DOCUMENT_EXTENSIONS {
        total_docs += queries::count_files_by_ext(conn, ext)?;
    }

    let word_count = queries::count_enriched(conn, "text", "word_count")?;
    let lang_count = queries::count_enriched(conn, "text", "language")?;
    let page_count = queries::count_enriched(conn, "text", "page_count")?;

    println!("Document files total:    {total_docs}");
    println!("  Word count:            {word_count}");
    println!("  Language detected:     {lang_count}");
    println!("  Page count:            {page_count}");

    if total_docs > 0 {
        let pct = (word_count as f64 / total_docs as f64) * 100.0;
        println!("  Progress:              {pct:.1}%");
    }

    println!();

    let mut total_videos = 0i64;
    for ext in VIDEO_EXTENSIONS {
        total_videos += queries::count_files_by_ext(conn, ext)?;
    }

    let vid_dur = queries::count_enriched(conn, "video", "duration")?;
    let vid_res = queries::count_enriched(conn, "video", "width")?;
    let vid_fps = queries::count_enriched(conn, "video", "fps")?;

    println!("Video files total:       {total_videos}");
    println!("  Duration:              {vid_dur}");
    println!("  Resolution:            {vid_res}");
    println!("  FPS:                   {vid_fps}");

    if total_videos > 0 {
        let pct = (vid_dur as f64 / total_videos as f64) * 100.0;
        println!("  Progress:              {pct:.1}%");
    }

    Ok(())
}
