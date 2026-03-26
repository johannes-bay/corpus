use std::collections::HashMap;
use std::path::Path;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use corpus_associate::matcher;
use corpus_associate::{AxisRegistry, WeightedAxis};
use corpus_db::queries;

use crate::server::AppState;

// ---------------------------------------------------------------------------
// Static files (embedded)
// ---------------------------------------------------------------------------

const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_JS: &str = include_str!("../static/app.js");
const STYLE_CSS: &str = include_str!("../static/style.css");
const GRAPH_HTML: &str = include_str!("../static/graph.html");

pub async fn graph_page() -> impl IntoResponse {
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/html; charset=utf-8")], GRAPH_HTML)
}

pub async fn static_files(uri: axum::http::Uri) -> impl IntoResponse {
    let path = uri.path();
    match path {
        "/app.js" => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/javascript")],
            APP_JS,
        ),
        "/style.css" => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/css")],
            STYLE_CSS,
        ),
        "/graph" => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            GRAPH_HTML,
        ),
        _ => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            INDEX_HTML,
        ),
    }
}

// ---------------------------------------------------------------------------
// GET /api/stats
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct StatsResponse {
    total_files: i64,
    audio_files: i64,
    image_files: i64,
    enrichment: HashMap<String, i64>,
}

const AUDIO_EXTENSIONS: &[&str] = &[".wav", ".mp3", ".m4a", ".aif", ".aiff", ".flac", ".ogg"];
const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp",
];

pub async fn stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, AppError> {
    let db = state.db.lock().map_err(|e| AppError(format!("db lock: {e}")))?;

    let total_files: i64 = db
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap_or(0);

    let mut audio_files = 0i64;
    for ext in AUDIO_EXTENSIONS {
        audio_files += queries::count_files_by_ext(&db, ext).unwrap_or(0);
    }

    let mut image_files = 0i64;
    for ext in IMAGE_EXTENSIONS {
        image_files += queries::count_files_by_ext(&db, ext).unwrap_or(0);
    }

    let mut enrichment = HashMap::new();
    for (domain, key) in &[
        ("audio", "bpm"),
        ("audio", "musical_key"),
        ("audio", "spectral_centroid"),
        ("audio", "duration"),
        ("image", "brightness"),
        ("image", "dominant_color"),
        ("image", "color_temperature"),
    ] {
        let count = queries::count_enriched(&db, domain, key).unwrap_or(0);
        enrichment.insert(format!("{domain}.{key}"), count);
    }

    // Embedding counts
    for model in &["clip:ViT-B-32", "clap:HTSAT-tiny"] {
        let count = queries::count_embeddings(&db, model).unwrap_or(0);
        if count > 0 {
            enrichment.insert(format!("embedding:{model}"), count);
        }
    }

    Ok(Json(StatsResponse {
        total_files,
        audio_files,
        image_files,
        enrichment,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/file?path=...
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct FileQuery {
    path: String,
}

#[derive(Serialize)]
pub struct FileInfoResponse {
    file: Option<corpus_db::models::FileEntry>,
    properties: Vec<corpus_db::models::Property>,
    audio_meta: Option<corpus_db::models::AudioMeta>,
    photo_meta: Option<corpus_db::models::PhotoMeta>,
    video_meta: Option<corpus_db::models::VideoMeta>,
    document_meta: Option<corpus_db::models::DocumentMeta>,
    font_meta: Option<corpus_db::models::FontMeta>,
}

pub async fn file_info(
    State(state): State<AppState>,
    Query(q): Query<FileQuery>,
) -> Result<Json<FileInfoResponse>, AppError> {
    let db = state.db.lock().map_err(|e| AppError(format!("db lock: {e}")))?;

    let file = queries::get_file(&db, &q.path).unwrap_or(None);
    let properties = queries::get_properties(&db, &q.path).unwrap_or_default();
    let audio_meta = queries::get_audio_meta(&db, &q.path).unwrap_or(None);
    let photo_meta = queries::get_photo_meta(&db, &q.path).unwrap_or(None);
    let video_meta = queries::get_video_meta(&db, &q.path).unwrap_or(None);
    let document_meta = queries::get_document_meta(&db, &q.path).unwrap_or(None);
    let font_meta = queries::get_font_meta(&db, &q.path).unwrap_or(None);

    Ok(Json(FileInfoResponse {
        file,
        properties,
        audio_meta,
        photo_meta,
        video_meta,
        document_meta,
        font_meta,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/search?q=...&ext=...&limit=20
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SearchQuery {
    q: Option<String>,
    ext: Option<String>,
    limit: Option<usize>,
}

pub async fn search(
    State(state): State<AppState>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<Vec<corpus_db::models::FileEntry>>, AppError> {
    let db = state.db.lock().map_err(|e| AppError(format!("db lock: {e}")))?;
    let limit = q.limit.unwrap_or(20).min(200);

    let mut sql = String::from(
        "SELECT id, path, filename, extension, size_bytes, modified_date, parent_folder FROM files WHERE 1=1",
    );
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut idx = 1;

    if let Some(ref term) = q.q
        && !term.is_empty()
    {
        sql.push_str(&format!(" AND filename LIKE ?{idx}"));
        params_vec.push(Box::new(format!("%{term}%")));
        idx += 1;
    }

    if let Some(ref ext) = q.ext
        && !ext.is_empty()
    {
        let ext_val = if ext.starts_with('.') {
            ext.clone()
        } else {
            format!(".{ext}")
        };
        sql.push_str(&format!(" AND extension = ?{idx}"));
        params_vec.push(Box::new(ext_val));
        idx += 1;
    }

    let _ = idx; // suppress unused warning

    sql.push_str(&format!(" ORDER BY filename LIMIT {limit}"));

    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();

    let mut stmt = db
        .prepare(&sql)
        .map_err(|e| AppError(format!("sql prepare: {e}")))?;

    let rows = stmt
        .query_map(&*params_refs, |row| {
            Ok(corpus_db::models::FileEntry {
                id: row.get("id")?,
                path: row.get("path")?,
                filename: row.get("filename")?,
                extension: row.get("extension")?,
                size_bytes: row.get("size_bytes")?,
                modified_date: row.get("modified_date")?,
                parent_folder: row.get("parent_folder")?,
            })
        })
        .map_err(|e| AppError(format!("query: {e}")))?;

    let results: Vec<_> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(results))
}

// ---------------------------------------------------------------------------
// POST /api/compose
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ComposeRequest {
    seed: String,
    axes: HashMap<String, f64>,
    count: Option<usize>,
}

#[derive(Serialize)]
pub struct ComposeResponse {
    seed: Option<corpus_db::models::FileEntry>,
    seed_properties: Vec<corpus_db::models::Property>,
    matches: Vec<MatchResult>,
}

#[derive(Serialize)]
pub struct MatchResult {
    file: corpus_db::models::FileEntry,
    score: f64,
    axes_detail: Vec<AxisScore>,
    explanation: Vec<String>,
}

#[derive(Serialize)]
pub struct AxisScore {
    name: String,
    score: f64,
}

/// Build WeightedAxis list from a name->weight map using the registry.
fn build_weighted_axes<'a>(
    axes_map: &HashMap<String, f64>,
    registry: &'a AxisRegistry,
) -> Vec<WeightedAxis<'a>> {
    let mut weighted = Vec::new();
    for (name, weight) in axes_map {
        if let Some(axis) = registry.get(name) {
            weighted.push(WeightedAxis {
                axis,
                weight: *weight,
            });
        }
    }
    weighted
}

pub async fn compose(
    State(state): State<AppState>,
    Json(req): Json<ComposeRequest>,
) -> Result<Json<ComposeResponse>, AppError> {
    let db = state.db.lock().map_err(|e| AppError(format!("db lock: {e}")))?;
    let count = req.count.unwrap_or(10).min(100);

    let seed = queries::get_file(&db, &req.seed).unwrap_or(None);
    let seed_properties = queries::get_properties(&db, &req.seed).unwrap_or_default();

    let registry = AxisRegistry::new();
    let weighted_axes = build_weighted_axes(&req.axes, &registry);
    if weighted_axes.is_empty() {
        return Ok(Json(ComposeResponse {
            seed,
            seed_properties,
            matches: Vec::new(),
        }));
    }

    let scored = matcher::find_matches(&db, &req.seed, &weighted_axes, count)
        .unwrap_or_default();

    let matches = scored
        .into_iter()
        .map(|m| MatchResult {
            file: m.file,
            score: m.total_score,
            axes_detail: m
                .axis_scores
                .iter()
                .map(|(n, s)| AxisScore {
                    name: n.clone(),
                    score: *s,
                })
                .collect(),
            explanation: m
                .explanation
                .axis_details
                .iter()
                .map(|(name, score, detail)| format!("{name}: {score:.2} ({detail})"))
                .collect(),
        })
        .collect();

    Ok(Json(ComposeResponse {
        seed,
        seed_properties,
        matches,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/file/preview?path=...
// ---------------------------------------------------------------------------

fn content_type_for_ext(ext: &str) -> &'static str {
    match ext.to_lowercase().as_str() {
        ".mp3" => "audio/mpeg",
        ".wav" => "audio/wav",
        ".m4a" => "audio/mp4",
        ".aac" => "audio/aac",
        ".ogg" => "audio/ogg",
        ".flac" => "audio/flac",
        ".aif" | ".aiff" => "audio/aiff",
        ".jpg" | ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".tif" | ".tiff" => "image/tiff",
        ".svg" => "image/svg+xml",
        ".mp4" => "video/mp4",
        ".mov" => "video/quicktime",
        ".avi" => "video/x-msvideo",
        ".webm" => "video/webm",
        ".pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

/// Parse an HTTP Range header value like "bytes=0-1023".
/// Returns (start, optional_end).
fn parse_range(range_header: &str, file_size: u64) -> Option<(u64, u64)> {
    let range_str = range_header.strip_prefix("bytes=")?;
    let (start_str, end_str) = range_str.split_once('-')?;

    let start: u64 = if start_str.is_empty() {
        // suffix range like "bytes=-500"
        let suffix: u64 = end_str.parse().ok()?;
        file_size.saturating_sub(suffix)
    } else {
        start_str.parse().ok()?
    };

    let end: u64 = if end_str.is_empty() {
        file_size - 1
    } else {
        end_str.parse().ok()?
    };

    if start > end || start >= file_size {
        return None;
    }

    let end = end.min(file_size - 1);
    Some((start, end))
}

pub async fn file_preview(
    State(state): State<AppState>,
    Query(q): Query<FileQuery>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    // Verify the file exists in the database
    {
        let db = state
            .db
            .lock()
            .map_err(|e| AppError(format!("db lock: {e}")))?;
        let file = queries::get_file(&db, &q.path).unwrap_or(None);
        if file.is_none() {
            return Ok((StatusCode::NOT_FOUND, "File not in database").into_response());
        }
    }

    let file_path = Path::new(&q.path);
    if !file_path.exists() {
        return Ok((StatusCode::NOT_FOUND, "File not found on disk").into_response());
    }

    let metadata = tokio::fs::metadata(&q.path)
        .await
        .map_err(|e| AppError(format!("metadata: {e}")))?;
    let file_size = metadata.len();

    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();
    let content_type = content_type_for_ext(&ext);

    // Check for Range header
    if let Some(range_val) = headers.get(header::RANGE) {
        let range_str = range_val
            .to_str()
            .map_err(|e| AppError(format!("range header: {e}")))?;

        if let Some((start, end)) = parse_range(range_str, file_size) {
            let length = end - start + 1;

            let file = tokio::fs::File::open(&q.path)
                .await
                .map_err(|e| AppError(format!("open: {e}")))?;

            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let mut file = file;
            file.seek(std::io::SeekFrom::Start(start))
                .await
                .map_err(|e| AppError(format!("seek: {e}")))?;
            let mut buf = vec![0u8; length as usize];
            file.read_exact(&mut buf)
                .await
                .map_err(|e| AppError(format!("read: {e}")))?;

            return Ok(Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(header::CONTENT_TYPE, content_type)
                .header(header::CONTENT_LENGTH, length.to_string())
                .header(
                    header::CONTENT_RANGE,
                    format!("bytes {start}-{end}/{file_size}"),
                )
                .header(header::ACCEPT_RANGES, "bytes")
                .body(Body::from(buf))
                .unwrap());
        }
    }

    // Full file response
    let bytes = tokio::fs::read(&q.path)
        .await
        .map_err(|e| AppError(format!("read file: {e}")))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, file_size.to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .body(Body::from(bytes))
        .unwrap())
}

// ---------------------------------------------------------------------------
// POST /api/concept
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ConceptRequest {
    query: String,
    count: Option<usize>,
    max_depth: Option<usize>,
}

#[derive(Serialize)]
pub struct ConceptResultItem {
    file: corpus_db::models::FileEntry,
    score: f64,
    sources: Vec<ConceptSourceInfo>,
}

#[derive(Serialize)]
pub struct ConceptSourceInfo {
    edge_type: String,
    via: Option<String>,
    weight: f64,
}

pub async fn concept_search(
    State(state): State<AppState>,
    Json(req): Json<ConceptRequest>,
) -> Result<Json<Vec<ConceptResultItem>>, AppError> {
    let db = state.db.lock().map_err(|e| AppError(format!("db lock: {e}")))?;
    let count = req.count.unwrap_or(200).min(500);

    let opts = corpus_associate::concept::ConceptQueryOpts {
        max_results: count,
        max_depth: req.max_depth.unwrap_or(2),
        ..Default::default()
    };

    let results = corpus_associate::concept::concept_query(&db, &req.query, &opts)
        .unwrap_or_default();

    let items = results
        .into_iter()
        .map(|m| ConceptResultItem {
            file: m.file,
            score: m.concept_score,
            sources: m
                .sources
                .iter()
                .map(|s| ConceptSourceInfo {
                    edge_type: s.edge_type.clone(),
                    via: s.via_path.as_ref().map(|p| {
                        p.split('/').last().unwrap_or(p).to_string()
                    }),
                    weight: s.weight,
                })
                .collect(),
        })
        .collect();

    Ok(Json(items))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0 })),
        )
            .into_response()
    }
}
