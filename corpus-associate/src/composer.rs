use std::collections::HashMap;

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

use corpus_db::models::FileEntry;
use corpus_db::queries;

use crate::axes::key_compatibility;
use crate::concept::ConceptMatch;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// What kind of output to compose.
#[derive(Debug, Clone, Serialize)]
pub enum CompositionMode {
    Audio,
    Image,
    Mixed,
    Auto,
}

/// Options for composition assembly.
#[derive(Debug, Clone)]
pub struct ComposeOpts {
    pub mode: CompositionMode,
    pub max_items: usize,
    /// For audio: prefer Camelot-compatible key ordering.
    pub key_chain: bool,
    /// For audio: prefer contrast in stem character between consecutive tracks.
    pub stem_contrast: bool,
    /// For images: target column count for layout.
    pub image_columns: u32,
}

impl Default for ComposeOpts {
    fn default() -> Self {
        Self {
            mode: CompositionMode::Auto,
            max_items: 30,
            key_chain: true,
            stem_contrast: true,
            image_columns: 5,
        }
    }
}

/// A planned item in the composition, with role and ordering metadata.
#[derive(Debug, Clone, Serialize)]
pub struct CompositionItem {
    pub file: FileEntry,
    pub concept_score: f64,
    pub role: String,
    pub position: usize,
    pub notes: Vec<String>,
}

/// The assembled composition plan.
#[derive(Debug, Clone, Serialize)]
pub struct CompositionPlan {
    pub concept: String,
    pub audio_items: Vec<CompositionItem>,
    pub image_items: Vec<CompositionItem>,
    pub other_items: Vec<CompositionItem>,
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Media type classification
// ---------------------------------------------------------------------------

const AUDIO_EXTS: &[&str] = &[".wav", ".mp3", ".m4a", ".aif", ".aiff", ".flac", ".ogg"];
const IMAGE_EXTS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp",
];

fn is_audio(ext: &Option<String>) -> bool {
    ext.as_deref()
        .is_some_and(|e| AUDIO_EXTS.contains(&e.to_lowercase().as_str()))
}

fn is_image(ext: &Option<String>) -> bool {
    ext.as_deref()
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
}

// ---------------------------------------------------------------------------
// Audio: Camelot key-chain ordering
// ---------------------------------------------------------------------------

struct AudioCandidate {
    file: FileEntry,
    concept_score: f64,
    key_name: Option<String>,
    bpm: Option<f64>,
    dominant_stem: Option<String>,
}

/// Load audio metadata needed for composition ordering.
fn load_audio_candidate(conn: &Connection, file: &FileEntry, concept_score: f64) -> AudioCandidate {
    let props = queries::get_properties(conn, &file.path).unwrap_or_default();
    let key_name = props
        .iter()
        .find(|p| p.domain == "audio" && p.key == "musical_key")
        .and_then(|p| p.value_txt.clone());
    let bpm = props
        .iter()
        .find(|p| p.domain == "audio" && p.key == "bpm")
        .and_then(|p| p.value_num)
        .filter(|&v| v > 0.0);

    // Find dominant stem from stem awareness scores
    let dominant_stem = props
        .iter()
        .filter(|p| p.domain == "stems")
        .max_by(|a, b| {
            a.value_num
                .unwrap_or(0.0)
                .partial_cmp(&b.value_num.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.key.clone());

    AudioCandidate {
        file: file.clone(),
        concept_score,
        key_name,
        bpm,
        dominant_stem,
    }
}

/// Order audio tracks by Camelot key-chain: start from the highest-scored
/// track, then greedily pick the next harmonically adjacent track.
fn order_by_key_chain(candidates: &mut [AudioCandidate]) -> Vec<usize> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let mut used = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Start with highest concept score
    let start = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.concept_score
                .partial_cmp(&b.concept_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    used[start] = true;
    order.push(start);

    // Greedily chain by key compatibility
    for _ in 1..n {
        let current = *order.last().unwrap();
        let current_key = candidates[current].key_name.as_deref();

        let mut best_idx = None;
        let mut best_compat = -1.0_f64;

        for (j, cand) in candidates.iter().enumerate() {
            if used[j] {
                continue;
            }
            let compat = match (current_key, cand.key_name.as_deref()) {
                (Some(ck), Some(nk)) => key_compatibility(ck, nk).0,
                // No key info: fall back to concept score as tiebreaker
                _ => 0.5 + cand.concept_score * 0.1,
            };
            if compat > best_compat {
                best_compat = compat;
                best_idx = Some(j);
            }
        }

        if let Some(idx) = best_idx {
            used[idx] = true;
            order.push(idx);
        }
    }

    order
}

/// Insert contrast items when consecutive tracks share the same dominant stem.
fn apply_stem_contrast(candidates: &[AudioCandidate], order: &mut Vec<usize>) {
    if order.len() < 3 {
        return;
    }

    let mut i = 1;
    while i < order.len() {
        let prev_stem = &candidates[order[i - 1]].dominant_stem;
        let curr_stem = &candidates[order[i]].dominant_stem;

        // If both have the same dominant stem, try to find a contrasting track to swap in
        if prev_stem.is_some() && prev_stem == curr_stem {
            // Look ahead for a track with a different dominant stem
            for j in (i + 1)..order.len() {
                if candidates[order[j]].dominant_stem.as_ref() != prev_stem.as_ref() {
                    order.swap(i, j);
                    break;
                }
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Image: provenance-clustered ordering
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Core entry point
// ---------------------------------------------------------------------------

/// Compose a plan from concept graph results.
/// Separates matches by media type, orders audio by Camelot key-chain,
/// clusters images by provenance, and assigns roles.
pub fn compose(
    conn: &Connection,
    concept: &str,
    matches: &[ConceptMatch],
    opts: &ComposeOpts,
) -> Result<CompositionPlan> {
    let _mode = match opts.mode {
        CompositionMode::Auto => {
            let has_audio = matches.iter().any(|m| is_audio(&m.file.extension));
            let has_image = matches.iter().any(|m| is_image(&m.file.extension));
            match (has_audio, has_image) {
                (true, true) => CompositionMode::Mixed,
                (true, false) => CompositionMode::Audio,
                (false, true) => CompositionMode::Image,
                (false, false) => CompositionMode::Mixed,
            }
        }
        ref m => m.clone(),
    };

    // Separate by media type
    let audio_matches: Vec<&ConceptMatch> = matches
        .iter()
        .filter(|m| is_audio(&m.file.extension))
        .take(opts.max_items)
        .collect();
    let image_matches: Vec<&ConceptMatch> = matches
        .iter()
        .filter(|m| is_image(&m.file.extension))
        .take(opts.max_items)
        .collect();
    let other_matches: Vec<&ConceptMatch> = matches
        .iter()
        .filter(|m| !is_audio(&m.file.extension) && !is_image(&m.file.extension))
        .take(opts.max_items)
        .collect();

    // --- Audio composition ---
    let audio_items = if !audio_matches.is_empty() {
        let mut candidates: Vec<AudioCandidate> = audio_matches
            .iter()
            .map(|m| load_audio_candidate(conn, &m.file, m.concept_score))
            .collect();

        let mut order = if opts.key_chain {
            order_by_key_chain(&mut candidates)
        } else {
            (0..candidates.len()).collect()
        };

        if opts.stem_contrast {
            apply_stem_contrast(&candidates, &mut order);
        }

        order
            .iter()
            .enumerate()
            .map(|(pos, &idx)| {
                let c = &candidates[idx];
                let mut notes = Vec::new();
                if let Some(ref key) = c.key_name {
                    notes.push(format!("key: {key}"));
                }
                if let Some(bpm) = c.bpm {
                    notes.push(format!("bpm: {bpm:.0}"));
                }
                if let Some(ref stem) = c.dominant_stem {
                    notes.push(format!("dominant: {stem}"));
                }

                // Role assignment
                let role = if pos == 0 {
                    "anchor"
                } else if pos < 3 && c.concept_score > 0.5 {
                    "anchor"
                } else if c.key_name.is_some() && pos > 0 {
                    // Check if this was placed here for key compatibility
                    let prev_key = candidates[order[pos - 1]].key_name.as_deref();
                    if let (Some(pk), Some(ck)) = (prev_key, c.key_name.as_deref()) {
                        let (compat, _) = key_compatibility(pk, ck);
                        if compat >= 0.85 {
                            notes.push(format!("bridge from {pk}"));
                            "bridge"
                        } else {
                            "companion"
                        }
                    } else {
                        "companion"
                    }
                } else {
                    "companion"
                };

                CompositionItem {
                    file: c.file.clone(),
                    concept_score: c.concept_score,
                    role: role.to_string(),
                    position: pos,
                    notes,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // --- Image composition ---
    let image_items = if !image_matches.is_empty() {
        // Group by provenance, sort by hue within clusters
        let mut clusters: HashMap<&str, Vec<(usize, f64)>> = HashMap::new();
        for (i, m) in image_matches.iter().enumerate() {
            let folder = m.file.parent_folder.as_str();
            let hue = queries::get_properties(conn, &m.file.path)
                .unwrap_or_default()
                .iter()
                .find(|p| p.domain == "image" && p.key == "dominant_hue")
                .and_then(|p| p.value_num)
                .unwrap_or(0.0);
            clusters.entry(folder).or_default().push((i, hue));
        }

        // Sort clusters by max concept score
        let mut cluster_list: Vec<(&str, Vec<(usize, f64)>)> =
            clusters.into_iter().collect();
        cluster_list.sort_by(|a, b| {
            let max_a = a.1.iter().map(|&(i, _)| image_matches[i].concept_score).fold(0.0_f64, f64::max);
            let max_b = b.1.iter().map(|&(i, _)| image_matches[i].concept_score).fold(0.0_f64, f64::max);
            max_b.partial_cmp(&max_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Within each cluster, sort by hue
        let mut ordered_indices = Vec::new();
        for (_, items) in &mut cluster_list {
            items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(i, _) in items.iter() {
                ordered_indices.push(i);
            }
        }

        ordered_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| {
                let m = image_matches[idx];
                let role = if pos < 3 && m.concept_score > 0.5 {
                    "anchor"
                } else {
                    "companion"
                };
                CompositionItem {
                    file: m.file.clone(),
                    concept_score: m.concept_score,
                    role: role.to_string(),
                    position: pos,
                    notes: Vec::new(),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // --- Other items (documents, videos, etc.) ---
    let other_items: Vec<CompositionItem> = other_matches
        .iter()
        .enumerate()
        .map(|(pos, m)| CompositionItem {
            file: m.file.clone(),
            concept_score: m.concept_score,
            role: "companion".to_string(),
            position: pos,
            notes: Vec::new(),
        })
        .collect();

    let summary = format!(
        "Concept \"{concept}\": {} audio (key-chained), {} images (provenance-clustered), {} other",
        audio_items.len(),
        image_items.len(),
        other_items.len(),
    );

    Ok(CompositionPlan {
        concept: concept.to_string(),
        audio_items,
        image_items,
        other_items,
        summary,
    })
}
