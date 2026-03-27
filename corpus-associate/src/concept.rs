use std::collections::HashMap;

use anyhow::Result;
use rusqlite::Connection;

use corpus_db::models::{FileEntry, Segment};
use corpus_db::queries;

use crate::axes::cosine_similarity;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConceptSource {
    pub edge_type: String,
    pub via_path: Option<String>,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct ConceptMatch {
    pub file: FileEntry,
    pub concept_score: f64,
    pub sources: Vec<ConceptSource>,
}

pub struct ConceptQueryOpts {
    pub max_results: usize,
    pub max_depth: usize,
    pub score_floor: f64,
    pub fan_out: usize,
    /// Additional anchors from visual embedding search (path, score).
    pub visual_anchors: Vec<(String, f64)>,
    /// Negative query text — files matching this get penalized.
    pub negative_query: Option<String>,
    /// Stem filter: only keep results with stem scores above these thresholds.
    /// E.g., [("vocals", 0.5)] keeps only files with vocals score >= 0.5.
    pub stem_filter: Vec<(String, f64)>,
    /// Whether to enable stem-based anchoring (default: true).
    pub stem_anchoring: bool,
}

impl Default for ConceptQueryOpts {
    fn default() -> Self {
        Self {
            max_results: 50,
            max_depth: 2,
            score_floor: 0.1,
            fan_out: 50,
            visual_anchors: Vec::new(),
            negative_query: None,
            stem_filter: Vec::new(),
            stem_anchoring: true,
        }
    }
}

// Base edge decay constants. The actual propagated score is further divided
// by the fan-out count (number of siblings this edge produces), so the
// constants don't need to be tuned per-archive. A co-folder edge to 1 of 3
// siblings carries 10x more weight than 1 of 30.
const DECAY_CO_FOLDER: f64 = 0.6;
const DECAY_CO_PROJECT: f64 = 0.5;
const DECAY_CO_ALBUM: f64 = 0.7;
const DECAY_CO_ARTIST: f64 = 0.5;
const DECAY_NEIGHBOR: f64 = 0.35;
const DECAY_CO_SESSION: f64 = 0.65;
const SESSION_WINDOW_SECS: i64 = 3600; // 1 hour
// CONVERGENCE_BONUS removed — asymptotic accumulation handles convergence naturally

/// Stem names and their synonyms for concept-to-stem matching.
const STEM_SYNONYMS: &[(&str, &[&str])] = &[
    ("vocals", &["vocals", "singing", "voice", "vocal", "singer", "rap", "rapper", "choir"]),
    ("drums", &["drums", "percussion", "drum", "beat", "rhythm", "drummer", "snare", "kick"]),
    ("bass", &["bass", "bassline", "sub", "low-end", "bassist"]),
    ("guitar", &["guitar", "guitarist", "riff", "strum", "acoustic guitar", "electric guitar"]),
    ("synth", &["synth", "synthesizer", "electronic", "pad", "arpeggio"]),
    ("piano", &["piano", "keyboard", "keys", "pianist", "grand piano"]),
    ("strings", &["strings", "violin", "cello", "viola", "orchestra", "orchestral"]),
    ("ambient", &["ambient", "atmospheric", "texture", "drone", "soundscape", "atmosphere"]),
];

/// Check if query text matches any stem concept.
/// Returns (canonical_stem_name, match_quality) where quality is 1.0 for exact
/// word match and 0.7 for substring match.
fn match_stem_concept(query: &str) -> Option<(&'static str, f64)> {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    for &(stem_name, synonyms) in STEM_SYNONYMS {
        for &syn in synonyms {
            if words.iter().any(|w| *w == syn) {
                return Some((stem_name, 1.0));
            }
            if q.contains(syn) {
                return Some((stem_name, 0.7));
            }
        }
    }
    None
}

/// Topology-based intentionality multiplier.
/// Files in deliberately curated contexts get boosted;
/// files in system/accidental contexts get dampened.
fn topology_weight(conn: &Connection, path: &str) -> f64 {
    // Look up topology for any file in the same folder
    let folder = path.rsplit_once('/').map(|(f, _)| f).unwrap_or(path);
    let result = conn.query_row(
        "SELECT value_txt FROM properties WHERE domain = 'topology' AND key = 'type' AND path IN (SELECT path FROM files WHERE parent_folder = ?1 LIMIT 1)",
        [folder],
        |row| row.get::<_, String>(0),
    );
    match result.as_deref() {
        Ok("production-session") => 1.4,
        Ok("curated-images") => 1.3,
        Ok("reference-collection") => 1.3,
        Ok("work-project") => 1.2,
        Ok("design-project") => 1.2,
        Ok("personal-photos") => 1.1,
        Ok("document-collection") => 1.1,
        Ok("video-project") => 1.1,
        Ok("music-collection") => 1.0,
        Ok("album-rip") => 1.0,
        Ok("camera-dump") => 0.9,
        Ok("export-output") => 0.8,
        Ok("sample-library") => 0.8,
        Ok("archive") => 0.7,
        Ok("download-accumulation") => 0.6,
        Ok("batch-export") => 0.5,
        Ok("system-artifact") => 0.3,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Internal state for graph traversal
// ---------------------------------------------------------------------------

struct CandidateInfo {
    file: FileEntry,
    score: f64,
    sources: Vec<ConceptSource>,
    edge_types: std::collections::HashSet<String>,
    /// Count per edge-type category (co-folder, co-session, etc.)
    /// Used to dampen repeated same-type edges.
    edge_type_counts: HashMap<String, usize>,
}

/// Generalize edge type to its category for dampening.
/// "neighbor:clip:ViT-B-32+seg(0.80)" → "neighbor"
/// "anchor:stem:drums" → "anchor"
/// "co-folder" → "co-folder"
fn edge_category(edge_type: &str) -> &str {
    edge_type.split(':').next().unwrap_or(edge_type)
}

impl CandidateInfo {
    fn add_source(&mut self, edge_type: &str, via_path: Option<String>, score: f64) {
        // Dampen repeated same-category edges: the Nth edge of the same type
        // contributes 1/N as much. First anchor/co-folder/neighbor is full strength;
        // second is half; third is a third; etc.
        let category = edge_category(edge_type).to_string();
        let count = self.edge_type_counts.entry(category).or_insert(0);
        *count += 1;
        let dampened = score / (*count as f64);

        // Asymptotic accumulation with dampened input
        self.score = self.score + dampened * (1.0 - self.score);
        self.edge_types.insert(edge_type.to_string());
        self.sources.push(ConceptSource {
            edge_type: edge_type.to_string(),
            via_path,
            weight: score,
        });
    }

    fn final_score(&self) -> f64 {
        self.score.min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Segment-level similarity
// ---------------------------------------------------------------------------

/// Compute segment-level similarity boost between two files.
/// Loads image region segments for both, computes area-weighted best-match
/// cosine similarity. Returns 0.0 if either has no segments.
fn segment_similarity_boost(
    conn: &Connection,
    path_a: &str,
    path_b: &str,
    seg_cache: &mut HashMap<String, Vec<(Segment, Vec<f32>)>>,
    seg_model: &str,
) -> f64 {
    let model = seg_model.to_string();
    let segs_a = seg_cache
        .entry(format!("{path_a}:{model}"))
        .or_insert_with(|| {
            queries::get_segments_with_embeddings(conn, path_a, "region", &model)
                .unwrap_or_default()
        })
        .clone();
    let segs_b = seg_cache
        .entry(format!("{path_b}:{model}"))
        .or_insert_with(|| {
            queries::get_segments_with_embeddings(conn, path_b, "region", &model)
                .unwrap_or_default()
        })
        .clone();

    if segs_a.is_empty() || segs_b.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for (seg, vec_a) in &segs_a {
        let area = seg.area_frac.unwrap_or(1.0 / segs_a.len() as f64);
        let best = segs_b
            .iter()
            .map(|(_, vec_b)| cosine_similarity(vec_a, vec_b))
            .fold(0.0_f64, f64::max);
        weighted_sum += best * area;
        weight_total += area;
    }
    if weight_total < 1e-12 {
        return 0.0;
    }
    weighted_sum / weight_total
}

// ---------------------------------------------------------------------------
// Concept query
// ---------------------------------------------------------------------------

pub fn concept_query(
    conn: &Connection,
    query_text: &str,
    opts: &ConceptQueryOpts,
) -> Result<Vec<ConceptMatch>> {
    let mut candidates: HashMap<String, CandidateInfo> = HashMap::new();

    // ---- Phase 1: Anchor Discovery ----

    // 1a. FTS search across filenames, metadata, text chunks
    let fts_hits = queries::fts_search(conn, query_text, 500)?;

    for hit in &fts_hits {
        let score = match hit.source_type.as_str() {
            "filename" => 1.0,
            "caption" => 0.95,
            "title" | "album" => 0.9,
            "artist" => 0.8,
            "folder" => 0.7,
            "genre" => 0.6,
            "chunk" => 0.5,
            "author" => 0.6,
            _ => 0.4,
        };
        // Boost by FTS rank (more negative = better match in FTS5)
        let rank_boost = (1.0 + hit.rank.abs().min(10.0) / 10.0).recip();
        let final_anchor_score = score * (0.5 + 0.5 * rank_boost);

        if let Ok(Some(file)) = queries::get_file(conn, &hit.path) {
            let entry = candidates.entry(hit.path.clone()).or_insert_with(|| CandidateInfo {
                file,
                score: 0.0,
                sources: Vec::new(),
                edge_types: std::collections::HashSet::new(),
                edge_type_counts: HashMap::new(),
            });
            entry.add_source(
                &format!("anchor:{}", hit.source_type),
                None,
                final_anchor_score,
            );
        }
    }

    // 1b. Stem awareness anchors — if query matches a stem concept,
    // inject files with high stem scores as anchors
    let mut stem_anchor_count = 0usize;
    if opts.stem_anchoring
        && let Some((stem_name, match_quality)) = match_stem_concept(query_text)
    {
        if let Ok(stem_files) = queries::find_files_by_stem_score(conn, stem_name, 0.5, 200) {
            for (file, stem_score) in stem_files {
                let anchor_score = stem_score * match_quality * 0.85;
                let entry = candidates.entry(file.path.clone()).or_insert_with(|| CandidateInfo {
                    file,
                    score: 0.0,
                    sources: Vec::new(),
                    edge_types: std::collections::HashSet::new(),
                edge_type_counts: HashMap::new(),
                });
                entry.add_source(
                    &format!("anchor:stem:{stem_name}"),
                    None,
                    anchor_score,
                );
                stem_anchor_count += 1;
            }
        }
    }

    // 1c. Visual anchors from embedding search (CLIP/SigLIP text→image)
    for (path, sim) in &opts.visual_anchors {
        if let Ok(Some(file)) = queries::get_file(conn, path) {
            let entry = candidates.entry(path.clone()).or_insert_with(|| CandidateInfo {
                file,
                score: 0.0,
                sources: Vec::new(),
                edge_types: std::collections::HashSet::new(),
                edge_type_counts: HashMap::new(),
            });
            // Visual anchors scored lower than FTS (one signal among many)
            entry.add_source("anchor:visual", None, sim * 0.6);
        }
    }

    tracing::info!("Phase 1: {} anchors ({} FTS + {} stem + {} visual)", candidates.len(), fts_hits.len(), stem_anchor_count, opts.visual_anchors.len());

    // ---- Phase 2: Edge Traversal ----

    // Discover available models from the database
    let neighbor_models = queries::available_neighbor_models(conn).unwrap_or_default();
    let seg_emb_model = queries::best_segment_emb_model(conn, "region")
        .unwrap_or_default()
        .unwrap_or_else(|| "clip:ViT-B-32".to_string());

    // Segment cache to avoid redundant DB loads during segment similarity boost
    let mut seg_cache: HashMap<String, Vec<(Segment, Vec<f32>)>> = HashMap::new();

    for _depth in 0..opts.max_depth {
        // Snapshot current frontier (paths + scores above floor)
        let frontier: Vec<(String, f64)> = candidates
            .iter()
            .filter(|(_, c)| c.score >= opts.score_floor)
            .map(|(p, c)| (p.clone(), c.score))
            .collect();

        for (path, parent_score) in &frontier {
            // Co-folder edge (fan-out normalized)
            if let Ok(siblings) = queries::get_folder_siblings(conn, path, 10) {
                let fan = (siblings.len() as f64).max(1.0).sqrt();
                let propagated = parent_score * DECAY_CO_FOLDER / fan;
                if propagated >= opts.score_floor {
                    for sib in siblings {
                        let entry = candidates.entry(sib.path.clone()).or_insert_with(|| CandidateInfo {
                            file: sib,
                            score: 0.0,
                            sources: Vec::new(),
                            edge_types: std::collections::HashSet::new(),
                            edge_type_counts: HashMap::new(),
                        });
                        entry.add_source("co-folder", Some(path.clone()), propagated);
                    }
                }
            }

            // Co-session edge (temporal proximity within same folder)
            if let Ok(session_sibs) = queries::get_session_siblings(
                conn, path, SESSION_WINDOW_SECS, 15
            ) {
                let fan = (session_sibs.len() as f64).max(1.0).sqrt();
                let propagated = parent_score * DECAY_CO_SESSION / fan;
                if propagated >= opts.score_floor {
                    for sib in session_sibs {
                        let entry = candidates.entry(sib.path.clone()).or_insert_with(|| CandidateInfo {
                            file: sib,
                            score: 0.0,
                            sources: Vec::new(),
                            edge_types: std::collections::HashSet::new(),
                            edge_type_counts: HashMap::new(),
                        });
                        entry.add_source("co-session", Some(path.clone()), propagated);
                    }
                }
            }

            // Co-project edge (fan-out normalized)
            if let Ok(Some(project)) = queries::get_file_project(conn, path) {
                if let Ok(project_files) = queries::get_project_files(conn, &project.id, opts.fan_out) {
                    let fan = (project_files.len() as f64).max(1.0).sqrt();
                    let propagated = parent_score * DECAY_CO_PROJECT / fan;
                    if propagated >= opts.score_floor {
                        for pf in project_files {
                            if pf.path == *path {
                                continue;
                            }
                            let entry = candidates.entry(pf.path.clone()).or_insert_with(|| CandidateInfo {
                                file: pf,
                                score: 0.0,
                                sources: Vec::new(),
                                edge_types: std::collections::HashSet::new(),
                                edge_type_counts: HashMap::new(),
                            });
                            entry.add_source("co-project", Some(path.clone()), propagated);
                        }
                    }
                }
            }

            // Co-album edge (fan-out normalized)
            if let Ok(Some(meta)) = queries::get_audio_meta(conn, path) {
                if let Some(ref album) = meta.album {
                    if let Ok(album_files) = queries::get_files_by_album(conn, album, 5) {
                        let fan = (album_files.len() as f64).max(1.0).sqrt();
                        let propagated = parent_score * DECAY_CO_ALBUM / fan;
                        if propagated >= opts.score_floor {
                            for af in album_files {
                                if af.path == *path {
                                    continue;
                                }
                                let entry = candidates.entry(af.path.clone()).or_insert_with(|| CandidateInfo {
                                    file: af,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                    edge_type_counts: HashMap::new(),
                                });
                                entry.add_source("co-album", Some(path.clone()), propagated);
                            }
                        }
                    }
                }
                if let Some(ref artist) = meta.artist {
                    if let Ok(artist_files) = queries::get_files_by_artist(conn, artist, 5) {
                        let fan = (artist_files.len() as f64).max(1.0).sqrt();
                        let propagated = parent_score * DECAY_CO_ARTIST / fan;
                        if propagated >= opts.score_floor {
                            for af in artist_files {
                                if af.path == *path {
                                    continue;
                                }
                                let entry = candidates.entry(af.path.clone()).or_insert_with(|| CandidateInfo {
                                    file: af,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                    edge_type_counts: HashMap::new(),
                                });
                                entry.add_source("co-artist", Some(path.clone()), propagated);
                            }
                        }
                    }
                }
            }

            // Embedding neighbor edges — traverse all available neighbor models
            let is_visual_model = |m: &str| m.starts_with("clip:") || m.starts_with("siglip:");
            for model in &neighbor_models {
                if let Ok(neighbors) = queries::get_neighbors(conn, path, model, 20) {
                    for nb in neighbors {
                        let propagated = parent_score * DECAY_NEIGHBOR * nb.similarity;
                        if propagated >= opts.score_floor {
                            if let Ok(Some(file)) = queries::get_file(conn, &nb.path_b) {
                                // Segment-level boost for visual neighbors:
                                // if both files have image region segments,
                                // boost score based on segment-level similarity
                                let seg_boost = if is_visual_model(model) {
                                    segment_similarity_boost(
                                        conn, path, &nb.path_b,
                                        &mut seg_cache, &seg_emb_model,
                                    )
                                } else {
                                    0.0
                                };
                                let boosted = propagated * (1.0 + seg_boost * 0.5);

                                let entry = candidates.entry(nb.path_b.clone()).or_insert_with(|| CandidateInfo {
                                    file,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                    edge_type_counts: HashMap::new(),
                                });
                                let edge_label = if seg_boost > 0.1 {
                                    format!("neighbor:{model}+seg({seg_boost:.2})")
                                } else {
                                    format!("neighbor:{model}")
                                };
                                entry.add_source(
                                    &edge_label,
                                    Some(path.clone()),
                                    boosted,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    tracing::info!("Phase 2: {} total candidates after traversal", candidates.len());

    // ---- Phase 2b: Negative query penalty ----

    if let Some(ref neg_query) = opts.negative_query {
        if let Ok(neg_hits) = queries::fts_search(conn, neg_query, 200) {
            for hit in &neg_hits {
                if let Some(entry) = candidates.get_mut(&hit.path) {
                    entry.score *= 0.3;
                    entry.sources.push(ConceptSource {
                        edge_type: "penalty:negative".to_string(),
                        via_path: None,
                        weight: -0.7,
                    });
                }
            }
            tracing::info!("Negative penalty applied to {} candidates", neg_hits.len());
        }
    }

    // ---- Phase 3: Apply topology weighting and rank ----

    let mut results: Vec<ConceptMatch> = candidates
        .into_values()
        .map(|c| {
            let base_score = c.final_score();
            let topo_w = topology_weight(conn, &c.file.path);
            let score = (base_score * topo_w).min(1.0);
            ConceptMatch {
                file: c.file,
                concept_score: score,
                sources: c.sources,
            }
        })
        .collect();

    // Apply stem filter if specified
    if !opts.stem_filter.is_empty() {
        results.retain(|m| {
            let props = queries::get_properties(conn, &m.file.path).unwrap_or_default();
            for (stem_name, min_score) in &opts.stem_filter {
                let stem_val = props
                    .iter()
                    .find(|p| p.domain == "stems" && p.key == *stem_name)
                    .and_then(|p| p.value_num)
                    .unwrap_or(0.0);
                if stem_val < *min_score {
                    return false;
                }
            }
            true
        });
    }

    results.sort_by(|a, b| {
        b.concept_score
            .partial_cmp(&a.concept_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(opts.max_results);

    Ok(results)
}
