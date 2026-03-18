use std::collections::HashMap;

use anyhow::{bail, Result};
use rusqlite::Connection;

use corpus_db::models::FileEntry;
use corpus_db::queries;

use crate::axes::{Axis, AxisRegistry, ScoringContext};
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
// Building ScoringContexts
// ---------------------------------------------------------------------------

fn build_context(conn: &Connection, file: FileEntry) -> Result<ScoringContext> {
    let all_props = queries::get_properties(conn, &file.path)?;
    let mut properties = HashMap::new();
    for p in all_props {
        if p.domain == "audio" {
            properties.insert(p.key.clone(), p);
        }
    }
    Ok(ScoringContext { file, properties })
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Find the top N matches for a given seed file.
///
/// The caller passes `WeightedAxis` references obtained from an `AxisRegistry`.
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

    let seed_ctx = build_context(conn, seed_file)?;

    // --- Gather candidate files ---
    // Start with BPM pre-filter if the seed has a valid BPM and a bpm axis is
    // requested. Then fill in any files we haven't seen yet for other axes.
    let mut candidate_files: HashMap<String, FileEntry> = HashMap::new();

    let has_bpm_axis = axes.iter().any(|wa| wa.axis.name() == "bpm");
    if has_bpm_axis
        && let Some(seed_bpm) = seed_ctx.audio_num("bpm")
        && seed_bpm > 0.0
    {
        let bpm_range = 40.0;
        let results = queries::find_by_property_range(
            conn,
            "audio",
            "bpm",
            seed_bpm - bpm_range,
            seed_bpm + bpm_range,
        )?;
        for (file, _val) in results {
            if file.path == seed_path {
                continue;
            }
            candidate_files.entry(file.path.clone()).or_insert(file);
        }
    }

    // If no candidates were gathered via BPM (or BPM axis is absent), fall back
    // to fetching all audio files (those with at least one enriched property).
    if candidate_files.is_empty() {
        // Grab files that have any audio-domain property
        let results = queries::find_by_property_range(
            conn,
            "audio",
            "bpm",
            -1e10,
            1e10,
        )?;
        for (file, _val) in results {
            if file.path == seed_path {
                continue;
            }
            candidate_files.entry(file.path.clone()).or_insert(file);
        }
    }

    // --- Score candidates ---
    let total_weight: f64 = axes.iter().map(|a| a.weight).sum();
    if total_weight == 0.0 {
        bail!("Total axis weight is zero");
    }

    let mut scored: Vec<ScoredMatch> = Vec::with_capacity(candidate_files.len());

    for (_path, file) in candidate_files {
        let cand_ctx = match build_context(conn, file) {
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

/// Convenience wrapper that accepts axis names + weights as string pairs
/// and resolves them through the registry. Returns an error for unknown names.
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
