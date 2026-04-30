//! Corpus composition engine.
//!
//! Sits between the association engine (which proposes candidates) and the UI
//! (which lets you steer). Given a set of matched artifacts/elements and a
//! composition strategy, this crate produces a concrete output — a manifest,
//! a collage, an audio mix, a timeline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod export;
pub mod harmony;
pub mod layout;
pub mod renderers;
pub mod strategy;

pub use renderers::{audio::AudioRenderer, image::CollageRenderer, manifest::ManifestRenderer};
pub use strategy::*;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ComposeError {
    #[error("composition has no members")]
    Empty,
    #[error("strategy {strategy} cannot render members of domain {domain}")]
    DomainMismatch { strategy: String, domain: String },
    #[error("missing seed in composition")]
    MissingSeed,
    #[error("invalid slot index {0}")]
    InvalidSlot(usize),
    #[error("source file missing on disk: {0}")]
    SourceMissing(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("audio decode error: {0}")]
    Audio(String),
    #[error("hound error: {0}")]
    Hound(#[from] hound::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ComposeError>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A composition is a set of artifacts with roles and relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composition {
    pub id: String,
    pub name: String,
    pub seed: ArtifactRef,
    pub members: Vec<CompositionMember>,
    pub relationships: Vec<Relationship>,
    pub strategy: CompositionStrategy,
    pub parameters: CompositionParameters,
}

impl Composition {
    /// Create a new empty composition with the given seed.
    pub fn new(name: impl Into<String>, seed: ArtifactRef, strategy: CompositionStrategy) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            seed,
            members: Vec::new(),
            relationships: Vec::new(),
            strategy,
            parameters: CompositionParameters::default(),
        }
    }

    pub fn with_member(mut self, member: CompositionMember) -> Self {
        self.members.push(member);
        self
    }

    /// Filter members by role.
    pub fn members_with_role(&self, role: MemberRole) -> impl Iterator<Item = &CompositionMember> {
        self.members.iter().filter(move |m| m.role == role)
    }
}

/// Reference to a source artifact (file) or sub-element (segment) within an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub path: String,
    /// If referring to a sub-element (e.g. an image segment, an audio stem),
    /// the segment id. None means the whole file.
    pub element_id: Option<String>,
    /// Domain hint: "audio", "image", "text", "video".
    pub domain: String,
}

impl ArtifactRef {
    pub fn file(path: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            element_id: None,
            domain: domain.into(),
        }
    }

    pub fn element(
        path: impl Into<String>,
        element_id: impl Into<String>,
        domain: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            element_id: Some(element_id.into()),
            domain: domain.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionMember {
    pub artifact: ArtifactRef,
    pub role: MemberRole,
    pub match_score: f64,
    pub match_explanation: String,
    /// Optional pinned slot — placement override that bypasses the strategy
    /// default for this role.
    pub slot_override: Option<SlotDefinition>,
}

impl CompositionMember {
    pub fn new(
        artifact: ArtifactRef,
        role: MemberRole,
        match_score: f64,
        match_explanation: impl Into<String>,
    ) -> Self {
        Self {
            artifact,
            role,
            match_score,
            match_explanation: match_explanation.into(),
            slot_override: None,
        }
    }
}

/// What role a member plays in the composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemberRole {
    /// The original seed artifact.
    Seed,
    /// Fills space, sets tone (sky, ambient pad, background texture).
    Background,
    /// Provides visual/temporal framework (lines, grid, rhythm).
    Structure,
    /// Focal point (a bird, a vocal, a key passage).
    Subject,
    /// Color/tonal pop, counterpoint.
    Accent,
    /// Connects two other elements (transition, gradient, crossfade).
    Bridge,
    /// Adds surface detail (grain, noise, subtle pattern).
    Texture,
}

impl MemberRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Seed => "Seed",
            Self::Background => "Background",
            Self::Structure => "Structure",
            Self::Subject => "Subject",
            Self::Accent => "Accent",
            Self::Bridge => "Bridge",
            Self::Texture => "Texture",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Index into composition.members.
    pub from: usize,
    pub to: usize,
    pub kind: RelationshipKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipKind {
    /// These share a quality (same key, similar palette).
    Harmonizes,
    /// Deliberate tension (warm vs cool, fast vs slow).
    Contrasts,
    /// One provides context for the other.
    Supports,
    /// One leads into the other temporally.
    Transitions,
}

/// Free-form parameters that don't fit into the strategy enum.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompositionParameters {
    pub notes: Vec<String>,
    pub extra: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Renderer trait
// ---------------------------------------------------------------------------

/// A renderer turns a `Composition` into a concrete artifact.
///
/// All renderers should implement at least `render` (the full output) and
/// `preview` (a fast/cheap version for the UI).
pub trait Renderer {
    type Output;

    /// Validate that the composition can be rendered with this strategy.
    fn validate(&self, composition: &Composition) -> Result<()>;

    /// Render the composition into a concrete output.
    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<Self::Output>;

    /// Preview: cheaper/faster version for the UI (thumbnails, short clips).
    fn preview(&self, composition: &Composition) -> Result<PreviewData>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreviewData {
    /// JPEG/PNG thumbnail bytes of the composition.
    Image(Vec<u8>),
    /// Amplitude envelope for display.
    AudioWaveform(Vec<f64>),
    /// First ~200 chars of composed text.
    TextSnippet(String),
    /// Rendered HTML preview.
    Html(String),
    /// Just a manifest summary.
    Manifest(String),
}

#[derive(Debug, Clone)]
pub enum ComposedOutput {
    ImageFile(PathBuf),
    AudioFile(PathBuf),
    VideoFile(PathBuf),
    HtmlFile(PathBuf),
    Manifest(Box<CompositionManifest>),
}

/// A manifest describes the composition without rendering it. This is the
/// minimum viable output and is always available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionManifest {
    pub composition: Composition,
    pub output_format: String,
    /// Human-readable assembly instructions.
    pub instructions: Vec<String>,
}
