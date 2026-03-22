use std::collections::HashMap;

use anyhow::{bail, Result};
use rusqlite::Connection;

use corpus_db::models::FileEntry;
use corpus_db::queries;

use crate::axes::{Axis, AxisRegistry, ScoringContext, SegmentVector};
use crate::explain::MatchExplanation;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A named axis with an associated weight for the weighted-sum scoring.
pub struct WeightedAxis<'a> {
    pub axis: &'a dyn Axis,
    pub weight: f64,
}

/// A scored candidate file.
#[derive(Debug, Clone)]
pub struct ScoredMatch {
    pub file: FileEntry,
    pub total_score: f64,
    pub axis_scores: Vec<(String, f64)>,
    pub explanation: MatchExplanation,
}

// ---------------------------------------------------------------------------
// Segment axis metadata — maps axis names to (segment_type, emb_model)
// ---------------------------------------------------------------------------

/// Check if an axis name is a segment axis and return its (segment_type, emb_model).
fn segment_axis_info(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        "objects" => Some(("region", "clip:ViT-B-32")),
        "vocals" => Some(("stem", "clap:HTSAT-tiny")),
        "scenes" => Some(("scene", "clip:ViT-B-32")),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Building ScoringContexts
// ---------------------------------------------------------------------------

fn build_context(conn: &Connection, file: FileEntry, seg_keys: &[String]) -> Result<ScoringContext> {
    let all_props = queries::get_properties(conn, &file.path)?;
    let mut properties = HashMap::new();
    for p in all_props {
        let lookup = format!("{}.{}", p.domain, p.key);
        properties.insert(lookup, p);
    }

    // Load all embeddings for this file
    let emb_rows = queries::get_embeddings(conn, &file.path)?;
    let mut embeddings = HashMap::new();
    for e in emb_rows {
        if !e.vector.is_empty() {
            embeddings.insert(e.model, e.vector);
        }
    }

    // Load segment embeddings for requested segment types
    let mut segment_embeddings: HashMap<String, Vec<SegmentVector>> = HashMap::new();
    if !seg_keys.is_empty() {
        // Parse seg_keys like "region:clip:ViT-B-32" into (segment_type, emb_model)
        let mut types_models: Vec<(&str, &str)> = Vec::new();
        for key in seg_keys {
            if let Some((seg_type, rest)) = key.split_once(':') {
                types_models.push((seg_type, rest));
            }
        }

        for (seg_type, emb_model) in &types_models {
            let segments = queries::get_segments_by_type(conn, &file.path, seg_type)?;
            if segments.is_empty() {
                continue;
            }
            let seg_ids: Vec<String> = segments.iter().map(|s| s.id.clone()).collect();
            let seg_embs = queries::get_segment_embeddings(conn, &seg_ids, emb_model)?;

            // Map segment_id -> embedding vector
            let emb_map: HashMap<String, Vec<f32>> = seg_embs
                .into_iter()
                .map(|e| (e.segment_id, e.vector))
                .collect();

            let key = format!("{seg_type}:{emb_model}");
            let vectors: Vec<SegmentVector> = segments
                .into_iter()
                .filter_map(|s| {
                    let vector = emb_map.get(&s.id)?.clone();
                    if vector.is_empty() {
                        return None;
                    }
                    Some(SegmentVector {
                        segment_id: s.id,
                        label: s.label,
                        area_frac: s.area_frac,
                        vector,
                    })
                })
                .collect();

            if !vectors.is_empty() {
                segment_embeddings.insert(key, vectors);
            }
        }
    }

    Ok(ScoringContext { file, properties, embeddings, segment_embeddings })
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Find the top N matches for a given seed file.
pub fn find_matches(
    conn: &Connection,
    seed_path: &str,
    axes: &[WeightedAxis<'_>],
    count: usize,
) -> Result<Vec<ScoredMatch>> {
    // Verify seed exists
    let seed_file = queries::get_file(conn, seed_path)?;
    let Some(seed_file) = seed_file else {
        bail!("Seed file not found in database: {seed_path}");
    };

    // Determine which segment types we need
    let seg_keys: Vec<String> = axes
        .iter()
        .filter_map(|wa| {
            let (seg_type, emb_model) = segment_axis_info(wa.axis.name())?;
            Some(format!("{seg_type}:{emb_model}"))
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let seed_ctx = build_context(conn, seed_file, &seg_keys)?;

    // --- Gather candidate files ---
    let mut candidate_files: HashMap<String, FileEntry> = HashMap::new();

    // Determine which domains the seed has properties in
    let seed_domains: Vec<String> = seed_ctx
        .properties
        .values()
        .map(|p| p.domain.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // BPM pre-filter for audio seeds
    let has_bpm_axis = axes.iter().any(|wa| wa.axis.name() == "bpm");
    if has_bpm_axis
        && let Some(seed_bpm) = seed_ctx.audio_num("bpm")
        && seed_bpm > 0.0
    {
        let bpm_range = 40.0;
        let results = queries::find_by_property_range(
            conn, "audio", "bpm",
            seed_bpm - bpm_range, seed_bpm + bpm_range,
        )?;
        for (file, _val) in results {
            if file.path != seed_path {
                candidate_files.entry(file.path.clone()).or_insert(file);
            }
        }
    }

    // Brightness pre-filter for image seeds
    let has_brightness_axis = axes.iter().any(|wa| wa.axis.name() == "brightness");
    if has_brightness_axis
        && let Some(seed_bright) = seed_ctx.image_num("brightness")
        && seed_bright >= 0.0
    {
        let bright_range = 0.5;
        let results = queries::find_by_property_range(
            conn, "image", "brightness",
            (seed_bright - bright_range).max(0.0),
            (seed_bright + bright_range).min(1.0),
        )?;
        for (file, _val) in results {
            if file.path != seed_path {
                candidate_files.entry(file.path.clone()).or_insert(file);
            }
        }
    }

    // File-level embedding candidate gathering (visual/sonic)
    let embedding_models: Vec<&str> = axes.iter().filter_map(|wa| {
        match wa.axis.name() {
            "visual" => Some("clip:ViT-B-32"),
            "sonic" => Some("clap:HTSAT-tiny"),
            _ => None,
        }
    }).collect();

    for model in &embedding_models {
        if seed_ctx.embeddings.contains_key(*model) {
            let paths = queries::find_paths_with_embedding(conn, model)?;
            for path in paths {
                if path != seed_path && !candidate_files.contains_key(&path) {
                    if let Ok(Some(file)) = queries::get_file(conn, &path) {
                        candidate_files.insert(path, file);
                    }
                }
            }
        }
    }

    // Segment-level candidate gathering (objects/vocals/scenes)
    for wa in axes {
        if let Some((seg_type, emb_model)) = segment_axis_info(wa.axis.name()) {
            let seg_key = format!("{seg_type}:{emb_model}");
            if seed_ctx.segment_embeddings.contains_key(&seg_key) {
                let paths = queries::find_paths_with_segment_embeddings(conn, seg_type, emb_model)?;
                for path in paths {
                    if path != seed_path && !candidate_files.contains_key(&path) {
                        if let Ok(Some(file)) = queries::get_file(conn, &path) {
                            candidate_files.insert(path, file);
                        }
                    }
                }
            }
        }
    }

    // Fallback: gather all enriched files from seed's domains
    if candidate_files.is_empty() {
        for domain in &seed_domains {
            let files = queries::find_files_by_domain(conn, domain)?;
            for file in files {
                if file.path != seed_path {
                    candidate_files.entry(file.path.clone()).or_insert(file);
                }
            }
        }
    }

    // --- Score candidates ---
    let total_weight: f64 = axes.iter().map(|a| a.weight).sum();
    if total_weight == 0.0 {
        bail!("Total axis weight is zero");
    }

    let mut scored: Vec<ScoredMatch> = Vec::with_capacity(candidate_files.len());

    for (_path, file) in candidate_files {
        let cand_ctx = match build_context(conn, file, &seg_keys) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut weighted_sum = 0.0;
        let mut axis_scores = Vec::new();
        let mut axis_details = Vec::new();

        for wa in axes {
            let s = wa.axis.score(&seed_ctx, &cand_ctx);
            let detail = wa.axis.explain(&seed_ctx, &cand_ctx);
            weighted_sum += s * wa.weight;
            axis_scores.push((wa.axis.name().to_string(), s));
            axis_details.push((wa.axis.name().to_string(), s, detail));
        }

        let total_score = weighted_sum / total_weight;

        scored.push(ScoredMatch {
            file: cand_ctx.file.clone(),
            total_score,
            axis_scores,
            explanation: MatchExplanation::new(
                &cand_ctx.file.filename,
                total_score,
                axis_details,
            ),
        });
    }

    // Sort descending by score
    scored.sort_by(|a, b| {
        b.total_score
            .partial_cmp(&a.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(count);

    Ok(scored)
}

/// Convenience wrapper that accepts axis names + weights as string pairs.
pub fn find_matches_by_name(
    conn: &Connection,
    registry: &AxisRegistry,
    seed_path: &str,
    axis_weights: &[(&str, f64)],
    count: usize,
) -> Result<Vec<ScoredMatch>> {
    let mut weighted: Vec<WeightedAxis<'_>> = Vec::new();
    for (name, weight) in axis_weights {
        let axis = registry.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown axis: {name}. Available: {}",
                registry.names().join(", ")
            )
        })?;
        weighted.push(WeightedAxis {
            axis,
            weight: *weight,
        });
    }
    find_matches(conn, seed_path, &weighted, count)
}
