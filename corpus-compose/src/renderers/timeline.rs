//! Video timeline renderer (Phase 2 — needs ffmpeg orchestration).
//!
//! Phase 1: validates and falls back to a manifest description.

use std::path::{Path, PathBuf};

use crate::renderers::manifest::ManifestRenderer;
use crate::strategy::CompositionStrategy;
use crate::{ComposeError, Composition, PreviewData, Renderer, Result};

#[derive(Debug, Default, Clone)]
pub struct TimelineRenderer;

impl Renderer for TimelineRenderer {
    type Output = PathBuf;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        match &composition.strategy {
            CompositionStrategy::Timeline(_) => Ok(()),
            other => Err(ComposeError::DomainMismatch {
                strategy: other.kind().to_string(),
                domain: "video".to_string(),
            }),
        }
    }

    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<PathBuf> {
        self.validate(composition)?;
        // Phase 2 will call ffmpeg here. For now, write a manifest stub.
        let manifest = ManifestRenderer::new();
        manifest.render(composition, output_dir)?;
        let path = output_dir.join(format!("{}.txt", composition.id));
        Ok(path)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        ManifestRenderer::new().preview(composition)
    }
}
