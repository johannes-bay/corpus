//! Text/document weaving renderer (Phase 2 — needs full text extraction).
//!
//! Phase 1: validates and writes a manifest stub.

use std::path::{Path, PathBuf};

use crate::renderers::manifest::ManifestRenderer;
use crate::strategy::CompositionStrategy;
use crate::{ComposeError, Composition, PreviewData, Renderer, Result};

#[derive(Debug, Default, Clone)]
pub struct DocumentRenderer;

impl Renderer for DocumentRenderer {
    type Output = PathBuf;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        match &composition.strategy {
            CompositionStrategy::TextWeave(_) => Ok(()),
            other => Err(ComposeError::DomainMismatch {
                strategy: other.kind().to_string(),
                domain: "text".to_string(),
            }),
        }
    }

    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<PathBuf> {
        self.validate(composition)?;
        ManifestRenderer::new().render(composition, output_dir)?;
        let path = output_dir.join(format!("{}.txt", composition.id));
        Ok(path)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        Ok(PreviewData::TextSnippet(
            ManifestRenderer::new().render_text(composition),
        ))
    }
}
