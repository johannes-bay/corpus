use std::sync::Mutex;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "corpus", about = "Local multimodal creative composition engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run enrichment pipelines on indexed files
    Enrich {
        /// Path to the index database
        #[arg(long)]
        db: String,
        /// Domain to enrich (audio, image)
        #[arg(long, default_value = "audio")]
        domain: String,
        /// Number of parallel workers
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
    /// Run musical key detection on audio files (uses libkeyfinder)
    EnrichKeys {
        /// Path to the index database
        #[arg(long)]
        db: String,
        /// Number of parallel workers
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Max seconds of audio to analyze per file (0 = entire file)
        #[arg(long, default_value_t = 60.0)]
        max_seconds: f64,
    },
    /// Find files that compose well with a seed file
    Compose {
        /// Path to the index database
        #[arg(long)]
        db: String,
        /// Seed file path (as stored in the database)
        #[arg(long)]
        seed: String,
        /// Axis weights as key=value pairs (e.g. bpm=0.8,key=0.9,spectral=0.3,temporal=0.5,provenance=0.2)
        #[arg(long)]
        axes: String,
        /// Number of results to return
        #[arg(long, default_value_t = 10)]
        count: usize,
    },
    /// Show all properties and metadata for a file
    Inspect {
        /// Path to the index database
        #[arg(long)]
        db: String,
        /// File path to inspect (as stored in the database)
        #[arg(long)]
        path: String,
    },
    /// Show enrichment progress statistics
    Stats {
        /// Path to the index database
        #[arg(long)]
        db: String,
    },
    /// Launch the web UI
    Serve {
        /// Path to the index database
        #[arg(long)]
        db: String,
        /// Port to listen on
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
    /// List all registered scoring axes with descriptions
    Axes,
}

fn parse_axes<'a>(
    input: &str,
    registry: &'a corpus_associate::AxisRegistry,
) -> Result<Vec<corpus_associate::WeightedAxis<'a>>> {
    let mut axes = Vec::new();

    for pair in input.split(',') {
        let parts: Vec<&str> = pair.split('=').collect();
        if parts.len() != 2 {
            bail!("Invalid axis format: {pair}. Expected key=weight (e.g. bpm=0.8)");
        }

        let weight: f64 = parts[1]
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid weight: {}", parts[1]))?;

        let axis = registry.get(parts[0]).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown axis: {}. Available: {}",
                parts[0],
                registry.names().join(", ")
            )
        })?;

        axes.push(corpus_associate::WeightedAxis { axis, weight });
    }

    Ok(axes)
}

fn print_opt(label: &str, val: &Option<String>) {
    if let Some(v) = val
        && !v.is_empty()
    {
        println!("  {label}: {v}");
    }
}

fn print_opt_num<T: std::fmt::Display>(label: &str, val: &Option<T>) {
    if let Some(v) = val {
        println!("  {label}: {v}");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Enrich {
            db,
            domain,
            concurrency,
        } => {
            let conn = corpus_db::open_db(&db)?;
            match domain.as_str() {
                "audio" => {
                    let conn = Mutex::new(conn);
                    corpus_enrich::pipeline::enrich_audio(&conn, concurrency)?;
                }
                "image" => {
                    let conn = Mutex::new(conn);
                    corpus_enrich::pipeline::enrich_images(&conn, concurrency)?;
                }
                other => bail!("Unknown domain: {other}. Available: audio, image"),
            }
        }
        Commands::EnrichKeys {
            db,
            concurrency,
            max_seconds,
        } => {
            let conn = corpus_db::open_db(&db)?;
            let conn = Mutex::new(conn);
            corpus_enrich::pipeline::enrich_keys(&conn, concurrency, max_seconds)?;
        }
        Commands::Compose {
            db,
            seed,
            axes,
            count,
        } => {
            let conn = corpus_db::open_db(&db)?;
            let registry = corpus_associate::AxisRegistry::new();
            let weighted_axes = parse_axes(&axes, &registry)?;
            let matches =
                corpus_associate::matcher::find_matches(&conn, &seed, &weighted_axes, count)?;

            if matches.is_empty() {
                println!("No matches found. Has the seed file been enriched?");
                return Ok(());
            }

            println!("Top {count} matches for: {seed}\n");
            for (i, m) in matches.iter().enumerate() {
                println!(
                    "Match #{}: {} (score: {:.2})",
                    i + 1,
                    m.file.filename,
                    m.total_score,
                );
                print!("{}", m.explanation);
                println!();
            }
        }
        Commands::Inspect { db, path } => {
            let conn = corpus_db::open_db(&db)?;

            let file = corpus_db::queries::get_file(&conn, &path)?;
            match file {
                Some(f) => {
                    println!("File: {}", f.filename);
                    println!("Path: {}", f.path);
                    println!("Size: {} bytes", f.size_bytes);
                    print_opt("Modified", &f.modified_date);
                    print_opt("Extension", &f.extension);
                    println!("Parent: {}", f.parent_folder);
                }
                None => {
                    bail!("File not found in database: {path}");
                }
            }

            // Show hash if available
            if let Some(hash) = corpus_db::queries::get_hash(&conn, &path)? {
                println!("\n--- Hash ---");
                println!("  MD5: {hash}");
            }

            // Show typed metadata from original index
            if let Some(m) = corpus_db::queries::get_audio_meta(&conn, &path)? {
                println!("\n--- Audio Metadata ---");
                print_opt_num("Duration", &m.duration_secs.map(|d| format!("{d:.1}s")));
                print_opt_num("Sample rate", &m.sample_rate.map(|r| format!("{r} Hz")));
                print_opt_num("Bit depth", &m.bit_depth);
                print_opt_num("Channels", &m.channels);
                print_opt("Artist", &m.artist);
                print_opt("Album", &m.album);
                print_opt("Title", &m.title);
                print_opt("Genre", &m.genre);
                print_opt("Year", &m.year);
                print_opt_num("BPM", &m.bpm);
                print_opt("Codec", &m.codec);
                print_opt("Bitrate", &m.bitrate);
                print_opt("File type", &m.file_type);
            }

            if let Some(m) = corpus_db::queries::get_photo_meta(&conn, &path)? {
                println!("\n--- Photo Metadata ---");
                print_opt_num(
                    "Dimensions",
                    &Some(format!(
                        "{}x{}",
                        m.width.unwrap_or(0),
                        m.height.unwrap_or(0)
                    )),
                );
                print_opt("Camera", &m.camera_model);
                print_opt("Make", &m.camera_make);
                print_opt("Lens", &m.lens);
                print_opt("Focal length", &m.focal_length);
                print_opt("Aperture", &m.aperture);
                print_opt("Shutter speed", &m.shutter_speed);
                print_opt_num("ISO", &m.iso);
                print_opt("Date taken", &m.date_taken);
                if m.gps_lat.is_some() {
                    println!(
                        "  GPS: {:.6}, {:.6}",
                        m.gps_lat.unwrap_or(0.0),
                        m.gps_lon.unwrap_or(0.0)
                    );
                }
                print_opt("Color space", &m.color_space);
            }

            if let Some(m) = corpus_db::queries::get_video_meta(&conn, &path)? {
                println!("\n--- Video Metadata ---");
                print_opt_num("Duration", &m.duration_secs.map(|d| format!("{d:.1}s")));
                print_opt_num(
                    "Dimensions",
                    &Some(format!(
                        "{}x{}",
                        m.width.unwrap_or(0),
                        m.height.unwrap_or(0)
                    )),
                );
                print_opt("Framerate", &m.framerate);
                print_opt("Video codec", &m.video_codec);
                print_opt("Audio codec", &m.audio_codec);
                print_opt_num("Bitrate", &m.bitrate_kbps.map(|b| format!("{b} kbps")));
                print_opt("Creation date", &m.creation_date);
            }

            if let Some(m) = corpus_db::queries::get_document_meta(&conn, &path)? {
                println!("\n--- Document Metadata ---");
                print_opt_num("Pages", &m.page_count);
                print_opt("Title", &m.title);
                print_opt("Author", &m.author);
                print_opt("Creator", &m.creator);
                print_opt("Created", &m.creation_date);
                print_opt("File type", &m.file_type);
            }

            if let Some(m) = corpus_db::queries::get_font_meta(&conn, &path)? {
                println!("\n--- Font Metadata ---");
                print_opt("Family", &m.font_family);
                print_opt("Style", &m.font_style);
                print_opt("Version", &m.font_version);
                print_opt("File type", &m.file_type);
            }

            // Show enriched properties
            let props = corpus_db::queries::get_properties(&conn, &path)?;
            if !props.is_empty() {
                println!("\n--- Enriched Properties ---");
                for p in &props {
                    let val = match (&p.value_num, &p.value_txt) {
                        (Some(n), Some(t)) => format!("{n} ({t})"),
                        (Some(n), None) => format!("{n}"),
                        (None, Some(t)) => t.clone(),
                        (None, None) => "(empty)".to_string(),
                    };
                    println!("  [{}.{}] = {val}", p.domain, p.key);
                }
            }
        }
        Commands::Stats { db } => {
            let conn = corpus_db::open_db(&db)?;
            corpus_enrich::pipeline::print_stats(&conn)?;
        }
        Commands::Serve { db, port } => {
            let conn = corpus_db::open_db(&db)?;
            corpus_ui::server::run(conn, port).await?;
        }
        Commands::Axes => {
            let registry = corpus_associate::AxisRegistry::new();
            println!("Registered scoring axes:\n");
            for axis in registry.list() {
                println!("  {:<14} {}", axis.name(), axis.description());
            }
            println!();
            println!("Usage: corpus compose --axes bpm=0.8,key=0.9,spectral=0.3,temporal=0.5,provenance=0.2");
        }
    }

    Ok(())
}
