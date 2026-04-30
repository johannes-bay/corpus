//! Composition strategies define the arrangement logic.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::MemberRole;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionStrategy {
    /// Visual: arrange elements in a spatial layout.
    Collage(CollageStrategy),
    /// Audio: layer or sequence tracks.
    AudioMix(AudioMixStrategy),
    /// Temporal: arrange elements on a timeline.
    Timeline(TimelineStrategy),
    /// Document: weave text passages together.
    TextWeave(TextWeaveStrategy),
    /// Cross-modal: combine across modalities into an HTML/interactive artifact.
    CrossModal(CrossModalStrategy),
}

impl CompositionStrategy {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Collage(_) => "collage",
            Self::AudioMix(_) => "audio_mix",
            Self::Timeline(_) => "timeline",
            Self::TextWeave(_) => "text_weave",
            Self::CrossModal(_) => "cross_modal",
        }
    }
}

// ---------------------------------------------------------------------------
// Collage (visual)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollageStrategy {
    pub layout: LayoutMode,
    /// 0=anything goes, 1=strict palette matching.
    pub tonal_consistency: f64,
    pub edge_blending: EdgeBlend,
    pub canvas_size: (u32, u32),
}

impl Default for CollageStrategy {
    fn default() -> Self {
        Self {
            layout: LayoutMode::RoleBased,
            tonal_consistency: 0.5,
            edge_blending: EdgeBlend::Feathered(8),
            canvas_size: (1920, 1080),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutMode {
    /// Elements fill predefined slots based on their role.
    /// Background → full canvas, Structure → edges, Subject → focal area.
    RoleBased,
    /// Grid arrangement, each element gets a cell.
    Grid { cols: u32, rows: u32 },
    /// Free placement based on visual balance.
    Balanced,
    /// User-defined slot positions.
    Custom(Vec<SlotDefinition>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotDefinition {
    pub role: MemberRole,
    /// x, y, width, height all in [0.0, 1.0] of the canvas.
    pub region: NormalizedRect,
    /// Layering order (higher = on top).
    pub z_order: i32,
    pub opacity: f64,
    pub blend_mode: BlendMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalizedRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl NormalizedRect {
    pub fn full() -> Self {
        Self { x: 0.0, y: 0.0, w: 1.0, h: 1.0 }
    }

    /// Resolve to integer pixel bounds clamped within the canvas.
    pub fn to_pixels(&self, canvas_w: u32, canvas_h: u32) -> (u32, u32, u32, u32) {
        let cw = canvas_w as f64;
        let ch = canvas_h as f64;
        let x = (self.x.clamp(0.0, 1.0) * cw).round() as u32;
        let y = (self.y.clamp(0.0, 1.0) * ch).round() as u32;
        let w = (self.w.clamp(0.0, 1.0) * cw).round() as u32;
        let h = (self.h.clamp(0.0, 1.0) * ch).round() as u32;
        (x, y, w.max(1).min(canvas_w.saturating_sub(x).max(1)), h.max(1).min(canvas_h.saturating_sub(y).max(1)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    /// Darken: good for silhouettes over sky.
    Multiply,
    /// Lighten: good for light elements over dark.
    Screen,
    /// Contrast enhancement.
    Overlay,
    /// Subtle tinting.
    SoftLight,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EdgeBlend {
    /// Clean edges (graphic, collage-like).
    HardCut,
    /// Soft edge in pixels.
    Feathered(u32),
    /// Use element mask for natural edges.
    MaskBased,
}

// ---------------------------------------------------------------------------
// Audio mix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMixStrategy {
    pub mode: AudioMixMode,
    pub crossfade_ms: u32,
    /// Auto-transpose to match seed key (Phase 2).
    pub key_transpose: bool,
    /// Auto-stretch to match seed BPM (Phase 2).
    pub bpm_timestretch: bool,
    pub stem_roles: HashMap<MemberRole, StemSettings>,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Maximum seconds taken from each member when sequencing.
    pub max_clip_secs: f64,
}

impl Default for AudioMixStrategy {
    fn default() -> Self {
        Self {
            mode: AudioMixMode::Sequence,
            crossfade_ms: 2000,
            key_transpose: false,
            bpm_timestretch: false,
            stem_roles: HashMap::new(),
            sample_rate: 44100,
            max_clip_secs: 30.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioMixMode {
    /// Layer stems simultaneously (ambient pad + vocal + rhythm).
    Layer,
    /// Sequence one after another with crossfades.
    Sequence,
    /// Layer with automatic ducking (subject louder, background quieter).
    DuckAndLayer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemSettings {
    /// Volume relative to seed.
    pub gain_db: f64,
    /// -1.0 left to 1.0 right.
    pub pan: f64,
    pub eq: Option<EqPreset>,
}

impl Default for StemSettings {
    fn default() -> Self {
        Self { gain_db: 0.0, pan: 0.0, eq: None }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EqPreset {
    /// Cut below Hz — good for layering, reduces mud.
    HighPass(f64),
    /// Cut above Hz — good for background/texture.
    LowPass(f64),
    /// Keep range — isolate a frequency band.
    BandPass(f64, f64),
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineStrategy {
    pub total_duration_secs: f64,
    /// Align cuts to audio beats if audio is present.
    pub beat_sync: bool,
    pub pacing: PacingCurve,
}

impl Default for TimelineStrategy {
    fn default() -> Self {
        Self {
            total_duration_secs: 60.0,
            beat_sync: false,
            pacing: PacingCurve::Constant(4.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PacingCurve {
    /// Even cut length in seconds.
    Constant(f64),
    /// Start slow, speed up.
    Accelerating,
    /// Slow-fast-slow-fast cycle.
    Breathing,
    /// Cut faster when audio energy is high.
    MatchAudioEnergy,
}

// ---------------------------------------------------------------------------
// Text weave
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextWeaveStrategy {
    pub mode: TextWeaveMode,
    pub max_words: usize,
}

impl Default for TextWeaveStrategy {
    fn default() -> Self {
        Self { mode: TextWeaveMode::ThematicFlow, max_words: 800 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TextWeaveMode {
    /// Interleave passages from different sources.
    Interleave,
    /// Arrange by thematic similarity (smooth reading flow).
    ThematicFlow,
    /// Juxtapose contrasting passages.
    Juxtapose,
}

// ---------------------------------------------------------------------------
// Cross-modal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalStrategy {
    /// Output an interactive HTML page that presents all modalities together.
    pub layout: CrossModalLayout,
}

impl Default for CrossModalStrategy {
    fn default() -> Self {
        Self { layout: CrossModalLayout::FocalPoint }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CrossModalLayout {
    /// Central image, audio player, text sidebar.
    FocalPoint,
    /// Grid of all media types.
    Gallery,
    /// Scrolling story: image-text-image-audio-text flow.
    Narrative,
}
