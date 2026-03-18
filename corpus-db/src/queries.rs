use anyhow::Result;
use rusqlite::{params, Connection};

use crate::models::{AudioMeta, DocumentMeta, FileEntry, FontMeta, PhotoMeta, Property, VideoMeta};

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

/// Count how many files have a given enriched property.
pub fn count_enriched(conn: &Connection, domain: &str, key: &str) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM properties WHERE domain = ?1 AND key = ?2",
        params![domain, key],
        |row| row.get(0),
    )?;
    Ok(count)
}
