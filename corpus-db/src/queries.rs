use anyhow::Result;
use rusqlite::{params, Connection};

use crate::models::{AudioMeta, DocumentMeta, Embedding, FileEntry, FontMeta, FtsHit, Neighbor, PhotoMeta, Project, Property, Segment, SegmentEmbedding, VideoMeta};

fn row_to_file(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileEntry> {
    Ok(FileEntry {
        id: row.get("id")?,
        path: row.get("path")?,
        filename: row.get("filename")?,
        extension: row.get("extension")?,
        size_bytes: row.get("size_bytes")?,
        modified_date: row.get("modified_date")?,
        parent_folder: row.get("parent_folder")?,
    })
}

// ---------------------------------------------------------------------------
// File queries
// ---------------------------------------------------------------------------

/// Get all files matching a given extension (e.g. ".mp3").
pub fn get_files_by_ext(conn: &Connection, ext: &str) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, filename, extension, size_bytes, modified_date, parent_folder
         FROM files WHERE extension = ?1",
    )?;
    let rows = stmt.query_map([ext], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get files of the given extension(s) that don't yet have a specific enriched property.
/// Used for resumable enrichment pipelines.
pub fn get_files_without_property(
    conn: &Connection,
    domain: &str,
    key: &str,
    extensions: &[&str],
) -> Result<Vec<FileEntry>> {
    let placeholders: String = extensions.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM files f
         WHERE f.extension IN ({placeholders})
           AND NOT EXISTS (
               SELECT 1 FROM properties p
               WHERE p.path = f.path AND p.domain = ?{d} AND p.key = ?{k}
           )",
        d = extensions.len() + 1,
        k = extensions.len() + 2,
    );

    let mut stmt = conn.prepare(&sql)?;

    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for ext in extensions {
        param_values.push(Box::new(ext.to_string()));
    }
    param_values.push(Box::new(domain.to_string()));
    param_values.push(Box::new(key.to_string()));

    let params: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

    let rows = stmt.query_map(&*params, row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get a single file by path.
pub fn get_file(conn: &Connection, path: &str) -> Result<Option<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, filename, extension, size_bytes, modified_date, parent_folder
         FROM files WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], row_to_file);
    match result {
        Ok(f) => Ok(Some(f)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Count files by extension.
pub fn count_files_by_ext(conn: &Connection, ext: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE extension = ?1",
        [ext],
        |row| row.get(0),
    )?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Typed metadata queries (read-only on original tables)
// ---------------------------------------------------------------------------

pub fn get_audio_meta(conn: &Connection, path: &str) -> Result<Option<AudioMeta>> {
    let mut stmt = conn.prepare(
        "SELECT path, duration_secs, sample_rate, bit_depth, channels,
                artist, album, title, genre, year, bpm, codec, bitrate, file_type
         FROM audio_meta WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], |row| {
        Ok(AudioMeta {
            path: row.get("path")?,
            duration_secs: row.get("duration_secs")?,
            sample_rate: row.get("sample_rate")?,
            bit_depth: row.get("bit_depth")?,
            channels: row.get("channels")?,
            artist: row.get("artist")?,
            album: row.get("album")?,
            title: row.get("title")?,
            genre: row.get("genre")?,
            year: row.get("year")?,
            bpm: row.get("bpm")?,
            codec: row.get("codec")?,
            bitrate: row.get("bitrate")?,
            file_type: row.get("file_type")?,
        })
    });
    match result {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn get_photo_meta(conn: &Connection, path: &str) -> Result<Option<PhotoMeta>> {
    let mut stmt = conn.prepare(
        "SELECT path, width, height, camera_make, camera_model, lens,
                focal_length, aperture, shutter_speed, iso, date_taken,
                gps_lat, gps_lon, color_space, file_type
         FROM photo_meta WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], |row| {
        Ok(PhotoMeta {
            path: row.get("path")?,
            width: row.get("width")?,
            height: row.get("height")?,
            camera_make: row.get("camera_make")?,
            camera_model: row.get("camera_model")?,
            lens: row.get("lens")?,
            focal_length: row.get("focal_length")?,
            aperture: row.get("aperture")?,
            shutter_speed: row.get("shutter_speed")?,
            iso: row.get("iso")?,
            date_taken: row.get("date_taken")?,
            gps_lat: row.get("gps_lat")?,
            gps_lon: row.get("gps_lon")?,
            color_space: row.get("color_space")?,
            file_type: row.get("file_type")?,
        })
    });
    match result {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn get_video_meta(conn: &Connection, path: &str) -> Result<Option<VideoMeta>> {
    let mut stmt = conn.prepare(
        "SELECT path, duration_secs, width, height, framerate,
                video_codec, audio_codec, bitrate_kbps, file_type, creation_date
         FROM video_meta WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], |row| {
        Ok(VideoMeta {
            path: row.get("path")?,
            duration_secs: row.get("duration_secs")?,
            width: row.get("width")?,
            height: row.get("height")?,
            framerate: row.get("framerate")?,
            video_codec: row.get("video_codec")?,
            audio_codec: row.get("audio_codec")?,
            bitrate_kbps: row.get("bitrate_kbps")?,
            file_type: row.get("file_type")?,
            creation_date: row.get("creation_date")?,
        })
    });
    match result {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn get_document_meta(conn: &Connection, path: &str) -> Result<Option<DocumentMeta>> {
    let mut stmt = conn.prepare(
        "SELECT path, page_count, title, author, creator, creation_date, file_type
         FROM document_meta WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], |row| {
        Ok(DocumentMeta {
            path: row.get("path")?,
            page_count: row.get("page_count")?,
            title: row.get("title")?,
            author: row.get("author")?,
            creator: row.get("creator")?,
            creation_date: row.get("creation_date")?,
            file_type: row.get("file_type")?,
        })
    });
    match result {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn get_font_meta(conn: &Connection, path: &str) -> Result<Option<FontMeta>> {
    let mut stmt = conn.prepare(
        "SELECT path, font_family, font_style, font_version, file_type
         FROM font_meta WHERE path = ?1",
    )?;
    let result = stmt.query_row([path], |row| {
        Ok(FontMeta {
            path: row.get("path")?,
            font_family: row.get("font_family")?,
            font_style: row.get("font_style")?,
            font_version: row.get("font_version")?,
            file_type: row.get("file_type")?,
        })
    });
    match result {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Get the MD5 hash for a file from the duplicates table.
pub fn get_hash(conn: &Connection, path: &str) -> Result<Option<String>> {
    let result = conn.query_row(
        "SELECT md5_hash FROM duplicates WHERE path = ?1",
        [path],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(h) => Ok(Some(h)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ---------------------------------------------------------------------------
// Enriched properties (corpus-added tables)
// ---------------------------------------------------------------------------

/// Upsert a property value for a file.
pub fn set_property(
    conn: &Connection,
    path: &str,
    domain: &str,
    key: &str,
    value_num: Option<f64>,
    value_txt: Option<&str>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO properties (path, domain, key, value_num, value_txt)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path, domain, key) DO UPDATE SET
           value_num = excluded.value_num,
           value_txt = excluded.value_txt",
        params![path, domain, key, value_num, value_txt],
    )?;
    Ok(())
}

/// Get all enriched properties for a given file.
pub fn get_properties(conn: &Connection, path: &str) -> Result<Vec<Property>> {
    let mut stmt = conn.prepare(
        "SELECT path, domain, key, value_num, value_txt FROM properties WHERE path = ?1",
    )?;
    let rows = stmt.query_map([path], |row| {
        Ok(Property {
            path: row.get("path")?,
            domain: row.get("domain")?,
            key: row.get("key")?,
            value_num: row.get("value_num")?,
            value_txt: row.get("value_txt")?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Find files whose numeric property falls within [min, max].
pub fn find_by_property_range(
    conn: &Connection,
    domain: &str,
    key: &str,
    min: f64,
    max: f64,
) -> Result<Vec<(FileEntry, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder,
                p.value_num
         FROM properties p
         JOIN files f ON f.path = p.path
         WHERE p.domain = ?1 AND p.key = ?2
           AND p.value_num >= ?3 AND p.value_num <= ?4",
    )?;
    let rows = stmt.query_map(params![domain, key, min, max], |row| {
        let file = FileEntry {
            id: row.get("id")?,
            path: row.get("path")?,
            filename: row.get("filename")?,
            extension: row.get("extension")?,
            size_bytes: row.get("size_bytes")?,
            modified_date: row.get("modified_date")?,
            parent_folder: row.get("parent_folder")?,
        };
        let val: f64 = row.get("value_num")?;
        Ok((file, val))
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Find files with high stem awareness scores for a given stem name.
/// Returns (FileEntry, score) pairs sorted by score descending.
pub fn find_files_by_stem_score(
    conn: &Connection,
    stem_name: &str,
    min_score: f64,
    limit: usize,
) -> Result<Vec<(FileEntry, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes,
                f.modified_date, f.parent_folder, p.value_num
         FROM properties p
         JOIN files f ON f.path = p.path
         WHERE p.domain = 'stems' AND p.key = ?1 AND p.value_num >= ?2
         ORDER BY p.value_num DESC LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![stem_name, min_score, limit as i64], |row| {
        let file = row_to_file(row)?;
        let val: f64 = row.get("value_num")?;
        Ok((file, val))
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Find all files that have at least one property in a given domain.
pub fn find_files_by_domain(conn: &Connection, domain: &str) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM properties p
         JOIN files f ON f.path = p.path
         WHERE p.domain = ?1",
    )?;
    let rows = stmt.query_map([domain], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Count how many files have a given enriched property.
pub fn count_enriched(conn: &Connection, domain: &str, key: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM properties WHERE domain = ?1 AND key = ?2",
        params![domain, key],
        |row| row.get(0),
    )?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Embedding queries
// ---------------------------------------------------------------------------

/// Get a specific embedding for a file and model.
pub fn get_embedding(conn: &Connection, path: &str, model: &str) -> Result<Option<Embedding>> {
    let mut stmt = conn.prepare(
        "SELECT path, model, vector, dim FROM embeddings WHERE path = ?1 AND model = ?2",
    )?;
    let result = stmt.query_row(params![path, model], |row| {
        let vector_blob: Vec<u8> = row.get("vector")?;
        let dim: usize = row.get::<_, i64>("dim")? as usize;
        let vector = bytes_to_f32(&vector_blob, dim);
        Ok(Embedding {
            path: row.get("path")?,
            model: row.get("model")?,
            vector,
            dim,
        })
    });
    match result {
        Ok(e) => Ok(Some(e)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Get all embeddings for a file (all models).
pub fn get_embeddings(conn: &Connection, path: &str) -> Result<Vec<Embedding>> {
    let mut stmt = conn.prepare(
        "SELECT path, model, vector, dim FROM embeddings WHERE path = ?1",
    )?;
    let rows = stmt.query_map([path], |row| {
        let vector_blob: Vec<u8> = row.get("vector")?;
        let dim: usize = row.get::<_, i64>("dim")? as usize;
        let vector = bytes_to_f32(&vector_blob, dim);
        Ok(Embedding {
            path: row.get("path")?,
            model: row.get("model")?,
            vector,
            dim,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Count embeddings for a given model.
pub fn count_embeddings(conn: &Connection, model: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM embeddings WHERE model = ?1",
        [model],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Find all file paths that have an embedding for a given model.
pub fn find_paths_with_embedding(conn: &Connection, model: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT path FROM embeddings WHERE model = ?1",
    )?;
    let rows = stmt.query_map([model], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// Segment queries
// ---------------------------------------------------------------------------

/// Get all segments for a file.
pub fn get_segments(conn: &Connection, path: &str) -> Result<Vec<Segment>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, segment_type, segment_key, label,
                bbox_x, bbox_y, bbox_w, bbox_h,
                time_start, time_end, confidence, area_frac, model
         FROM segments WHERE path = ?1",
    )?;
    let rows = stmt.query_map([path], |row| {
        Ok(Segment {
            id: row.get("id")?,
            path: row.get("path")?,
            segment_type: row.get("segment_type")?,
            segment_key: row.get("segment_key")?,
            label: row.get("label")?,
            bbox_x: row.get("bbox_x")?,
            bbox_y: row.get("bbox_y")?,
            bbox_w: row.get("bbox_w")?,
            bbox_h: row.get("bbox_h")?,
            time_start: row.get("time_start")?,
            time_end: row.get("time_end")?,
            confidence: row.get("confidence")?,
            area_frac: row.get("area_frac")?,
            model: row.get("model")?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get segments of a specific type for a file.
pub fn get_segments_by_type(conn: &Connection, path: &str, segment_type: &str) -> Result<Vec<Segment>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, segment_type, segment_key, label,
                bbox_x, bbox_y, bbox_w, bbox_h,
                time_start, time_end, confidence, area_frac, model
         FROM segments WHERE path = ?1 AND segment_type = ?2",
    )?;
    let rows = stmt.query_map(params![path, segment_type], |row| {
        Ok(Segment {
            id: row.get("id")?,
            path: row.get("path")?,
            segment_type: row.get("segment_type")?,
            segment_key: row.get("segment_key")?,
            label: row.get("label")?,
            bbox_x: row.get("bbox_x")?,
            bbox_y: row.get("bbox_y")?,
            bbox_w: row.get("bbox_w")?,
            bbox_h: row.get("bbox_h")?,
            time_start: row.get("time_start")?,
            time_end: row.get("time_end")?,
            confidence: row.get("confidence")?,
            area_frac: row.get("area_frac")?,
            model: row.get("model")?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get segment embeddings for a list of segment IDs.
pub fn get_segment_embeddings(conn: &Connection, segment_ids: &[String], model: &str) -> Result<Vec<SegmentEmbedding>> {
    if segment_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders: String = segment_ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT segment_id, model, vector, dim FROM segment_embeddings WHERE segment_id IN ({placeholders}) AND model = ?{}",
        segment_ids.len() + 1
    );
    let mut stmt = conn.prepare(&sql)?;

    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for id in segment_ids {
        param_values.push(Box::new(id.clone()));
    }
    param_values.push(Box::new(model.to_string()));
    let params: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

    let rows = stmt.query_map(&*params, |row| {
        let vector_blob: Vec<u8> = row.get("vector")?;
        let dim: usize = row.get::<_, i64>("dim")? as usize;
        let vector = bytes_to_f32(&vector_blob, dim);
        Ok(SegmentEmbedding {
            segment_id: row.get("segment_id")?,
            model: row.get("model")?,
            vector,
            dim,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Load segments of a given type for a file, paired with their embedding vectors.
/// Convenience wrapper combining get_segments_by_type + get_segment_embeddings.
pub fn get_segments_with_embeddings(
    conn: &Connection,
    path: &str,
    segment_type: &str,
    emb_model: &str,
) -> Result<Vec<(Segment, Vec<f32>)>> {
    let segments = get_segments_by_type(conn, path, segment_type)?;
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    let seg_ids: Vec<String> = segments.iter().map(|s| s.id.clone()).collect();
    let embs = get_segment_embeddings(conn, &seg_ids, emb_model)?;
    let emb_map: std::collections::HashMap<String, Vec<f32>> = embs
        .into_iter()
        .map(|e| (e.segment_id, e.vector))
        .collect();
    let result = segments
        .into_iter()
        .filter_map(|s| {
            let vec = emb_map.get(&s.id)?.clone();
            if vec.is_empty() {
                return None;
            }
            Some((s, vec))
        })
        .collect();
    Ok(result)
}

/// Find all file paths that have segments of a given type with embeddings for a model.
pub fn find_paths_with_segment_embeddings(
    conn: &Connection,
    segment_type: &str,
    emb_model: &str,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT s.path
         FROM segments s
         JOIN segment_embeddings se ON se.segment_id = s.id
         WHERE s.segment_type = ?1 AND se.model = ?2",
    )?;
    let rows = stmt.query_map(params![segment_type, emb_model], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Count segments by type.
pub fn count_segments(conn: &Connection, segment_type: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM segments WHERE segment_type = ?1",
        [segment_type],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Count segment embeddings by model.
pub fn count_segment_embeddings(conn: &Connection, model: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM segment_embeddings WHERE model = ?1",
        [model],
        |row| row.get(0),
    )?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/// Find the best available segment embedding model for a given segment type.
/// Prefers larger models (ViT-L-14 > ViT-B-32, SO400M > smaller).
/// Returns None if no segment embeddings exist for this type.
pub fn best_segment_emb_model(conn: &Connection, segment_type: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT se.model
         FROM segment_embeddings se
         JOIN segments s ON s.id = se.segment_id
         WHERE s.segment_type = ?1
         ORDER BY se.model DESC",
    )?;
    let rows = stmt.query_map([segment_type], |row| row.get::<_, String>(0))?;
    let models: Vec<String> = rows.filter_map(|r| r.ok()).collect();

    // Prefer larger/newer models
    let preference = ["clip:ViT-L-14", "siglip:SO400M-384", "clip:ViT-B-32"];
    for pref in preference {
        if models.iter().any(|m| m == pref) {
            return Ok(Some(pref.to_string()));
        }
    }
    Ok(models.into_iter().next())
}

/// Find all distinct embedding models available in the embeddings table.
pub fn available_embedding_models(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT DISTINCT model FROM embeddings ORDER BY model")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Find all distinct neighbor graph models.
pub fn available_neighbor_models(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT DISTINCT model FROM neighbors ORDER BY model")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// Full-text search queries
// ---------------------------------------------------------------------------

/// Search the FTS index for a query string. Returns matching file paths with source info.
pub fn fts_search(conn: &Connection, query: &str, limit: usize) -> Result<Vec<FtsHit>> {
    let mut stmt = conn.prepare(
        "SELECT path, source_type, source_key, rank
         FROM corpus_fts WHERE content MATCH ?1
         ORDER BY rank LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![query, limit as i64], |row| {
        Ok(FtsHit {
            path: row.get("path")?,
            source_type: row.get("source_type")?,
            source_key: row.get("source_key")?,
            rank: row.get("rank")?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Count rows in the FTS index.
pub fn count_fts(conn: &Connection) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM corpus_fts",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Neighbor graph queries
// ---------------------------------------------------------------------------

/// Get the top-K nearest neighbors for a file and model.
pub fn get_neighbors(conn: &Connection, path: &str, model: &str, limit: usize) -> Result<Vec<Neighbor>> {
    let mut stmt = conn.prepare(
        "SELECT path_a, path_b, model, similarity FROM neighbors
         WHERE path_a = ?1 AND model = ?2
         ORDER BY similarity DESC LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![path, model, limit as i64], |row| {
        Ok(Neighbor {
            path_a: row.get("path_a")?,
            path_b: row.get("path_b")?,
            model: row.get("model")?,
            similarity: row.get("similarity")?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// Project queries
// ---------------------------------------------------------------------------

/// Get the project a file belongs to.
pub fn get_file_project(conn: &Connection, path: &str) -> Result<Option<Project>> {
    let result = conn.query_row(
        "SELECT p.id, p.name, p.project_root, p.file_count, p.date_range
         FROM file_projects fp
         JOIN projects p ON p.id = fp.project_id
         WHERE fp.path = ?1",
        [path],
        |row| {
            Ok(Project {
                id: row.get("id")?,
                name: row.get("name")?,
                project_root: row.get("project_root")?,
                file_count: row.get("file_count")?,
                date_range: row.get("date_range")?,
            })
        },
    );
    match result {
        Ok(p) => Ok(Some(p)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Get all files in a project.
pub fn get_project_files(conn: &Connection, project_id: &str, limit: usize) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM file_projects fp
         JOIN files f ON f.path = fp.path
         WHERE fp.project_id = ?1 LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![project_id, limit as i64], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get files sharing the same parent folder.
pub fn get_folder_siblings(conn: &Connection, path: &str, limit: usize) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM files f
         WHERE f.parent_folder = (SELECT parent_folder FROM files WHERE path = ?1)
         AND f.path != ?1 LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![path, limit as i64], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get files in the same folder modified within a time window of the given file.
/// Returns files sorted by temporal proximity (closest first).
pub fn get_session_siblings(
    conn: &Connection,
    path: &str,
    time_window_secs: i64,
    limit: usize,
) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT f2.id, f2.path, f2.filename, f2.extension, f2.size_bytes,
                f2.modified_date, f2.parent_folder
         FROM files f1
         JOIN files f2 ON f2.parent_folder = f1.parent_folder
         WHERE f1.path = ?1
           AND f2.path != ?1
           AND f1.modified_date IS NOT NULL
           AND f2.modified_date IS NOT NULL
           AND ABS(
               CAST(strftime('%s', f2.modified_date) AS INTEGER)
             - CAST(strftime('%s', f1.modified_date) AS INTEGER)
           ) <= ?2
         ORDER BY ABS(
               CAST(strftime('%s', f2.modified_date) AS INTEGER)
             - CAST(strftime('%s', f1.modified_date) AS INTEGER)
           ) ASC
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![path, time_window_secs, limit as i64], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get audio files by artist.
pub fn get_files_by_artist(conn: &Connection, artist: &str, limit: usize) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM audio_meta am
         JOIN files f ON f.path = am.path
         WHERE am.artist = ?1 LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![artist, limit as i64], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get audio files by album.
pub fn get_files_by_album(conn: &Connection, album: &str, limit: usize) -> Result<Vec<FileEntry>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.path, f.filename, f.extension, f.size_bytes, f.modified_date, f.parent_folder
         FROM audio_meta am
         JOIN files f ON f.path = am.path
         WHERE am.album = ?1 LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![album, limit as i64], row_to_file)?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decode a little-endian f32 blob into a Vec<f32>.
fn bytes_to_f32(blob: &[u8], dim: usize) -> Vec<f32> {
    let expected_bytes = dim * 4;
    if blob.len() < expected_bytes {
        return Vec::new();
    }
    (0..dim)
        .map(|i| {
            let offset = i * 4;
            f32::from_le_bytes([
                blob[offset],
                blob[offset + 1],
                blob[offset + 2],
                blob[offset + 3],
            ])
        })
        .collect()
}
