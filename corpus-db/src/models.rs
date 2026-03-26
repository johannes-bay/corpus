use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub id: i64,
    pub path: String,
    pub filename: String,
    pub extension: Option<String>,
    pub size_bytes: i64,
    pub modified_date: Option<String>,
    pub parent_folder: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMeta {
    pub path: String,
    pub duration_secs: Option<f64>,
    pub sample_rate: Option<i64>,
    pub bit_depth: Option<i64>,
    pub channels: Option<i64>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub title: Option<String>,
    pub genre: Option<String>,
    pub year: Option<String>,
    pub bpm: Option<f64>,
    pub codec: Option<String>,
    pub bitrate: Option<String>,
    pub file_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoMeta {
    pub path: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens: Option<String>,
    pub focal_length: Option<String>,
    pub aperture: Option<String>,
    pub shutter_speed: Option<String>,
    pub iso: Option<i64>,
    pub date_taken: Option<String>,
    pub gps_lat: Option<f64>,
    pub gps_lon: Option<f64>,
    pub color_space: Option<String>,
    pub file_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMeta {
    pub path: String,
    pub duration_secs: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub framerate: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub bitrate_kbps: Option<i64>,
    pub file_type: Option<String>,
    pub creation_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMeta {
    pub path: String,
    pub page_count: Option<i64>,
    pub title: Option<String>,
    pub author: Option<String>,
    pub creator: Option<String>,
    pub creation_date: Option<String>,
    pub file_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontMeta {
    pub path: String,
    pub font_family: Option<String>,
    pub font_style: Option<String>,
    pub font_version: Option<String>,
    pub file_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub path: String,
    pub domain: String,
    pub key: String,
    pub value_num: Option<f64>,
    pub value_txt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub path: String,
    pub model: String,
    pub vector: Vec<f32>,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composition {
    pub id: String,
    pub name: String,
    pub seed_path: String,
    pub axes: String,
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionItem {
    pub composition_id: String,
    pub path: String,
    pub role: String,
    pub score: Option<f64>,
    pub axes_detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub path: String,
    pub segment_type: String,
    pub segment_key: String,
    pub label: Option<String>,
    pub bbox_x: Option<f64>,
    pub bbox_y: Option<f64>,
    pub bbox_w: Option<f64>,
    pub bbox_h: Option<f64>,
    pub time_start: Option<f64>,
    pub time_end: Option<f64>,
    pub confidence: Option<f64>,
    pub area_frac: Option<f64>,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SegmentEmbedding {
    pub segment_id: String,
    pub model: String,
    pub vector: Vec<f32>,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsHit {
    pub path: String,
    pub source_type: String,
    pub source_key: String,
    pub rank: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub path_a: String,
    pub path_b: String,
    pub model: String,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub project_root: String,
    pub file_count: i64,
    pub date_range: Option<String>,
}
