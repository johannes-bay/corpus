//! API endpoints for the new corpus-compose pipeline (visual collage,
//! audio mix). These endpoints are stubs in Phase 1 — they wire the
//! composition crate to the web UI but do not persist compositions yet.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use corpus_compose::strategy::{
    AudioMixMode, AudioMixStrategy, CollageStrategy, EdgeBlend, LayoutMode,
};
use corpus_compose::{
    ArtifactRef, Composition, CompositionMember, CompositionStrategy, MemberRole, Renderer,
};

use crate::api::AppError;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// In-process composition store
// ---------------------------------------------------------------------------

/// Simple in-memory store for compositions during a session.
/// Persistence to the corpus-db `compositions` table is left for Phase 2.
#[derive(Default, Clone)]
pub struct CompositionStore {
    inner: Arc<Mutex<HashMap<String, Composition>>>,
}

impl CompositionStore {
    pub fn insert(&self, composition: Composition) {
        if let Ok(mut g) = self.inner.lock() {
            g.insert(composition.id.clone(), composition);
        }
    }

    pub fn get(&self, id: &str) -> Option<Composition> {
        self.inner.lock().ok().and_then(|g| g.get(id).cloned())
    }
}

// ---------------------------------------------------------------------------
// Output directory
// ---------------------------------------------------------------------------

fn output_dir() -> PathBuf {
    std::env::temp_dir().join("corpus-compose")
}

// ---------------------------------------------------------------------------
// POST /api/compose/visual
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct VisualComposeRequest {
    pub seed: String,
    /// Strategy is always "collage" in Phase 1.
    #[serde(default)]
    pub strategy: Option<String>,
    /// "role-based" | "grid" | "balanced". Default role-based.
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default)]
    pub tonal_consistency: Option<f64>,
    /// Member paths (typically supplied by the matcher result).
    #[serde(default)]
    pub members: Vec<VisualMemberRef>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

#[derive(Deserialize)]
pub struct VisualMemberRef {
    pub path: String,
    /// Optional role string; falls back to auto-cycling.
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

#[derive(Serialize)]
pub struct ComposeVisualResponse {
    pub id: String,
    pub manifest: corpus_compose::CompositionManifest,
    pub preview_url: String,
    pub render_url: String,
}

pub async fn compose_visual(
    State(state): State<AppState>,
    Json(req): Json<VisualComposeRequest>,
) -> Result<Json<ComposeVisualResponse>, AppError> {
    let layout = match req.layout.as_deref() {
        Some("grid") => LayoutMode::Grid { cols: 3, rows: 2 },
        Some("balanced") => LayoutMode::Balanced,
        _ => LayoutMode::RoleBased,
    };
    let canvas = (req.width.unwrap_or(1920), req.height.unwrap_or(1080));
    let strategy = CompositionStrategy::Collage(CollageStrategy {
        layout,
        tonal_consistency: req.tonal_consistency.unwrap_or(0.5),
        edge_blending: EdgeBlend::Feathered(8),
        canvas_size: canvas,
    });

    let mut composition = Composition::new(
        format!("collage-{}", short_ts()),
        ArtifactRef::file(req.seed.clone(), "image"),
        strategy,
    );
    composition.members.push(CompositionMember::new(
        ArtifactRef::file(req.seed.clone(), "image"),
        MemberRole::Seed,
        1.0,
        "seed",
    ));
    let role_cycle = [
        MemberRole::Background,
        MemberRole::Structure,
        MemberRole::Subject,
        MemberRole::Accent,
        MemberRole::Texture,
    ];
    for (i, m) in req.members.iter().enumerate() {
        let role = m
            .role
            .as_deref()
            .and_then(parse_role)
            .unwrap_or(role_cycle[i % role_cycle.len()]);
        composition.members.push(CompositionMember::new(
            ArtifactRef::file(m.path.clone(), "image"),
            role,
            m.score.unwrap_or(0.0),
            String::new(),
        ));
    }

    let manifest = corpus_compose::ManifestRenderer::new().build(&composition);
    state.compositions.insert(composition.clone());

    let id = composition.id.clone();
    Ok(Json(ComposeVisualResponse {
        manifest,
        preview_url: format!("/api/compose/{id}/preview"),
        render_url: format!("/api/compose/{id}/render"),
        id,
    }))
}

// ---------------------------------------------------------------------------
// POST /api/compose/audio
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct AudioComposeRequest {
    pub seed: String,
    /// "sequence" | "layer" | "duck-and-layer"
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub crossfade_ms: Option<u32>,
    #[serde(default)]
    pub max_clip_secs: Option<f64>,
    #[serde(default)]
    pub members: Vec<AudioMemberRef>,
}

#[derive(Deserialize)]
pub struct AudioMemberRef {
    pub path: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

pub async fn compose_audio(
    State(state): State<AppState>,
    Json(req): Json<AudioComposeRequest>,
) -> Result<Json<ComposeVisualResponse>, AppError> {
    let mode = match req.strategy.as_deref() {
        Some("layer") => AudioMixMode::Layer,
        Some("duck-and-layer") | Some("duck") => AudioMixMode::DuckAndLayer,
        _ => AudioMixMode::Sequence,
    };

    let strategy = CompositionStrategy::AudioMix(AudioMixStrategy {
        mode,
        crossfade_ms: req.crossfade_ms.unwrap_or(2000),
        max_clip_secs: req.max_clip_secs.unwrap_or(20.0),
        ..Default::default()
    });

    let mut composition = Composition::new(
        format!("audio-{}", short_ts()),
        ArtifactRef::file(req.seed.clone(), "audio"),
        strategy,
    );
    composition.members.push(CompositionMember::new(
        ArtifactRef::file(req.seed.clone(), "audio"),
        MemberRole::Seed,
        1.0,
        "seed",
    ));
    let role_cycle = [
        MemberRole::Background,
        MemberRole::Subject,
        MemberRole::Accent,
        MemberRole::Texture,
    ];
    for (i, m) in req.members.iter().enumerate() {
        let role = m
            .role
            .as_deref()
            .and_then(parse_role)
            .unwrap_or(role_cycle[i % role_cycle.len()]);
        composition.members.push(CompositionMember::new(
            ArtifactRef::file(m.path.clone(), "audio"),
            role,
            m.score.unwrap_or(0.0),
            String::new(),
        ));
    }

    let manifest = corpus_compose::ManifestRenderer::new().build(&composition);
    state.compositions.insert(composition.clone());

    let id = composition.id.clone();
    Ok(Json(ComposeVisualResponse {
        manifest,
        preview_url: format!("/api/compose/{id}/preview"),
        render_url: format!("/api/compose/{id}/render"),
        id,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/compose/:id/preview
// ---------------------------------------------------------------------------

pub async fn compose_preview(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Response, AppError> {
    let comp = state
        .compositions
        .get(&id)
        .ok_or_else(|| AppError("composition not found".to_string()))?;

    let preview = match &comp.strategy {
        CompositionStrategy::Collage(_) => corpus_compose::CollageRenderer::new()
            .preview(&comp)
            .map_err(|e| AppError(format!("preview: {e}")))?,
        CompositionStrategy::AudioMix(_) => corpus_compose::AudioRenderer::new()
            .preview(&comp)
            .map_err(|e| AppError(format!("preview: {e}")))?,
        _ => corpus_compose::ManifestRenderer::new()
            .preview(&comp)
            .map_err(|e| AppError(format!("preview: {e}")))?,
    };

    let response = match preview {
        corpus_compose::PreviewData::Image(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg")],
            bytes,
        )
            .into_response(),
        corpus_compose::PreviewData::AudioWaveform(env) => Json(serde_json::json!({
            "kind": "waveform",
            "envelope": env,
        }))
        .into_response(),
        corpus_compose::PreviewData::TextSnippet(s)
        | corpus_compose::PreviewData::Manifest(s) => Json(serde_json::json!({
            "kind": "text",
            "content": s,
        }))
        .into_response(),
        corpus_compose::PreviewData::Html(s) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            s,
        )
            .into_response(),
    };
    Ok(response)
}

// ---------------------------------------------------------------------------
// GET /api/compose/:id/render
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct RenderResponse {
    pub id: String,
    pub output_path: String,
    pub kind: String,
}

pub async fn compose_render(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json<RenderResponse>, AppError> {
    let comp = state
        .compositions
        .get(&id)
        .ok_or_else(|| AppError("composition not found".to_string()))?;
    let dir = output_dir();
    let (path, kind): (PathBuf, &str) = match &comp.strategy {
        CompositionStrategy::Collage(_) => {
            let p = corpus_compose::CollageRenderer::new()
                .render(&comp, &dir)
                .map_err(|e| AppError(format!("render: {e}")))?;
            (p, "image")
        }
        CompositionStrategy::AudioMix(_) => {
            let p = corpus_compose::AudioRenderer::new()
                .render(&comp, &dir)
                .map_err(|e| AppError(format!("render: {e}")))?;
            (p, "audio")
        }
        _ => {
            corpus_compose::ManifestRenderer::new()
                .render(&comp, &dir)
                .map_err(|e| AppError(format!("render: {e}")))?;
            (dir.join(format!("{}.json", comp.id)), "manifest")
        }
    };
    Ok(Json(RenderResponse {
        id,
        output_path: path.to_string_lossy().into_owned(),
        kind: kind.to_string(),
    }))
}

// ---------------------------------------------------------------------------
// PUT /api/compose/:id/adjust
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct AdjustRequest {
    /// Replace the strategy entirely. JSON-serialized strategy struct.
    #[serde(default)]
    pub strategy: Option<serde_json::Value>,
    /// Pin a member to a slot index — placement is locked.
    #[serde(default)]
    pub pin_member: Option<usize>,
    /// Add a member by path with role.
    #[serde(default)]
    pub add_member: Option<AdjustAddMember>,
}

#[derive(Deserialize)]
pub struct AdjustAddMember {
    pub path: String,
    pub role: Option<String>,
    pub domain: Option<String>,
}

pub async fn compose_adjust(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    Query(_q): Query<HashMap<String, String>>,
    Json(req): Json<AdjustRequest>,
) -> Result<Json<ComposeVisualResponse>, AppError> {
    let mut comp = state
        .compositions
        .get(&id)
        .ok_or_else(|| AppError("composition not found".to_string()))?;

    if let Some(strat) = req.strategy {
        let parsed: CompositionStrategy = serde_json::from_value(strat)
            .map_err(|e| AppError(format!("invalid strategy: {e}")))?;
        comp.strategy = parsed;
    }
    if let Some(idx) = req.pin_member
        && idx < comp.members.len()
    {
        // No-op marker: leaves the member as-is. Once we have slot pinning
        // semantics in the renderer, a slot_override would be set here.
        let _ = &comp.members[idx];
    }
    if let Some(add) = req.add_member {
        let role = add
            .role
            .as_deref()
            .and_then(parse_role)
            .unwrap_or(MemberRole::Accent);
        let domain = add.domain.unwrap_or_else(|| "image".to_string());
        comp.members.push(CompositionMember::new(
            ArtifactRef::file(add.path, domain),
            role,
            0.0,
            "added via adjust".to_string(),
        ));
    }

    let manifest = corpus_compose::ManifestRenderer::new().build(&comp);
    state.compositions.insert(comp.clone());

    Ok(Json(ComposeVisualResponse {
        manifest,
        preview_url: format!("/api/compose/{id}/preview"),
        render_url: format!("/api/compose/{id}/render"),
        id: comp.id,
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_role(s: &str) -> Option<MemberRole> {
    match s.to_ascii_lowercase().as_str() {
        "seed" => Some(MemberRole::Seed),
        "background" | "bg" => Some(MemberRole::Background),
        "structure" => Some(MemberRole::Structure),
        "subject" => Some(MemberRole::Subject),
        "accent" => Some(MemberRole::Accent),
        "bridge" => Some(MemberRole::Bridge),
        "texture" => Some(MemberRole::Texture),
        _ => None,
    }
}

fn short_ts() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs:x}")
}
