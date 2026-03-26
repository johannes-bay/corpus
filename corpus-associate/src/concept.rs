use std::collections::HashMap;

use anyhow::Result;
use rusqlite::Connection;

use corpus_db::models::FileEntry;
use corpus_db::queries;

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
}

impl Default for ConceptQueryOpts {
    fn default() -> Self {
        Self {
            max_results: 50,
            max_depth: 2,
            score_floor: 0.05,
            fan_out: 50,
        }
    }
}

// Edge decay constants
const DECAY_CO_FOLDER: f64 = 0.6;
const DECAY_CO_PROJECT: f64 = 0.5;
const DECAY_CO_ALBUM: f64 = 0.7;
const DECAY_CO_ARTIST: f64 = 0.5;
const DECAY_NEIGHBOR: f64 = 0.35;
const CONVERGENCE_BONUS: f64 = 0.1;

// ---------------------------------------------------------------------------
// Internal state for graph traversal
// ---------------------------------------------------------------------------

struct CandidateInfo {
    file: FileEntry,
    score: f64,
    sources: Vec<ConceptSource>,
    edge_types: std::collections::HashSet<String>,
}

impl CandidateInfo {
    fn add_source(&mut self, edge_type: &str, via_path: Option<String>, score: f64) {
        if score > self.score {
            self.score = score;
        }
        self.edge_types.insert(edge_type.to_string());
        self.sources.push(ConceptSource {
            edge_type: edge_type.to_string(),
            via_path,
            weight: score,
        });
    }

    fn final_score(&self) -> f64 {
        let convergence = 1.0 + CONVERGENCE_BONUS * (self.edge_types.len() as f64 - 1.0).max(0.0);
        (self.score * convergence).min(1.0)
    }
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
            "folder" => 0.7,
            "title" | "album" => 0.9,
            "artist" => 0.8,
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
            });
            entry.add_source(
                &format!("anchor:{}", hit.source_type),
                None,
                final_anchor_score,
            );
        }
    }

    tracing::info!("Phase 1: {} anchors from FTS", candidates.len());

    // ---- Phase 2: Edge Traversal ----

    for _depth in 0..opts.max_depth {
        // Snapshot current frontier (paths + scores above floor)
        let frontier: Vec<(String, f64)> = candidates
            .iter()
            .filter(|(_, c)| c.score >= opts.score_floor)
            .map(|(p, c)| (p.clone(), c.score))
            .collect();

        for (path, parent_score) in &frontier {
            // Co-folder edge (capped)
            if let Ok(siblings) = queries::get_folder_siblings(conn, path, 10) {
                let propagated = parent_score * DECAY_CO_FOLDER;
                if propagated >= opts.score_floor {
                    for sib in siblings {
                        let entry = candidates.entry(sib.path.clone()).or_insert_with(|| CandidateInfo {
                            file: sib,
                            score: 0.0,
                            sources: Vec::new(),
                            edge_types: std::collections::HashSet::new(),
                        });
                        entry.add_source("co-folder", Some(path.clone()), propagated);
                    }
                }
            }

            // Co-project edge
            if let Ok(Some(project)) = queries::get_file_project(conn, path) {
                let propagated = parent_score * DECAY_CO_PROJECT;
                if propagated >= opts.score_floor {
                    if let Ok(project_files) = queries::get_project_files(conn, &project.id, opts.fan_out) {
                        for pf in project_files {
                            if pf.path == *path {
                                continue;
                            }
                            let entry = candidates.entry(pf.path.clone()).or_insert_with(|| CandidateInfo {
                                file: pf,
                                score: 0.0,
                                sources: Vec::new(),
                                edge_types: std::collections::HashSet::new(),
                            });
                            entry.add_source("co-project", Some(path.clone()), propagated);
                        }
                    }
                }
            }

            // Co-album edge (capped tight to avoid discography flooding)
            if let Ok(Some(meta)) = queries::get_audio_meta(conn, path) {
                if let Some(ref album) = meta.album {
                    let propagated = parent_score * DECAY_CO_ALBUM;
                    if propagated >= opts.score_floor {
                        if let Ok(album_files) = queries::get_files_by_album(conn, album, 5) {
                            for af in album_files {
                                if af.path == *path {
                                    continue;
                                }
                                let entry = candidates.entry(af.path.clone()).or_insert_with(|| CandidateInfo {
                                    file: af,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                });
                                entry.add_source("co-album", Some(path.clone()), propagated);
                            }
                        }
                    }
                }
                if let Some(ref artist) = meta.artist {
                    let propagated = parent_score * DECAY_CO_ARTIST;
                    if propagated >= opts.score_floor {
                        if let Ok(artist_files) = queries::get_files_by_artist(conn, artist, 5) {
                            for af in artist_files {
                                if af.path == *path {
                                    continue;
                                }
                                let entry = candidates.entry(af.path.clone()).or_insert_with(|| CandidateInfo {
                                    file: af,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                });
                                entry.add_source("co-artist", Some(path.clone()), propagated);
                            }
                        }
                    }
                }
            }

            // Embedding neighbor edges
            for model in &["clip:ViT-B-32", "clap:HTSAT-tiny"] {
                if let Ok(neighbors) = queries::get_neighbors(conn, path, model, 20) {
                    for nb in neighbors {
                        let propagated = parent_score * DECAY_NEIGHBOR * nb.similarity;
                        if propagated >= opts.score_floor {
                            if let Ok(Some(file)) = queries::get_file(conn, &nb.path_b) {
                                let entry = candidates.entry(nb.path_b.clone()).or_insert_with(|| CandidateInfo {
                                    file,
                                    score: 0.0,
                                    sources: Vec::new(),
                                    edge_types: std::collections::HashSet::new(),
                                });
                                entry.add_source(
                                    &format!("neighbor:{model}"),
                                    Some(path.clone()),
                                    propagated,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    tracing::info!("Phase 2: {} total candidates after traversal", candidates.len());

    // ---- Phase 3: Rank and return ----

    let mut results: Vec<ConceptMatch> = candidates
        .into_values()
        .map(|c| {
            let score = c.final_score();
            ConceptMatch {
                file: c.file,
                concept_score: score,
                sources: c.sources,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.concept_score
            .partial_cmp(&a.concept_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(opts.max_results);

    Ok(results)
}
