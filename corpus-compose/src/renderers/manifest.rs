//! Manifest renderer — always available, no rendering dependencies.
//!
//! Outputs a JSON manifest plus a human-readable description of what the
//! composition would look like.

use std::fmt::Write;
use std::path::Path;

use crate::export::write_manifest_json;
use crate::layout::resolve_slots;
use crate::strategy::{CompositionStrategy, LayoutMode};
use crate::{
    ComposeError, Composition, CompositionManifest, MemberRole, PreviewData, Renderer, Result,
};

/// Manifest renderer.
#[derive(Debug, Default, Clone)]
pub struct ManifestRenderer {
    pub include_human_readable: bool,
}

impl ManifestRenderer {
    pub fn new() -> Self {
        Self { include_human_readable: true }
    }

    /// Build the manifest in-memory (no file I/O).
    pub fn build(&self, composition: &Composition) -> CompositionManifest {
        CompositionManifest {
            composition: composition.clone(),
            output_format: format!("{}.json", composition.strategy.kind()),
            instructions: human_readable_lines(composition),
        }
    }

    /// Render the human-readable description as a single multi-line string.
    pub fn render_text(&self, composition: &Composition) -> String {
        human_readable_lines(composition).join("\n")
    }
}

impl Renderer for ManifestRenderer {
    type Output = CompositionManifest;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        Ok(())
    }

    fn render(
        &self,
        composition: &Composition,
        output_dir: &Path,
    ) -> Result<CompositionManifest> {
        self.validate(composition)?;
        std::fs::create_dir_all(output_dir)?;

        let manifest = self.build(composition);

        let json_path = output_dir.join(format!("{}.json", composition.id));
        write_manifest_json(&manifest, &json_path)?;

        if self.include_human_readable {
            let txt_path = output_dir.join(format!("{}.txt", composition.id));
            std::fs::write(&txt_path, self.render_text(composition))?;
        }

        Ok(manifest)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        Ok(PreviewData::Manifest(self.render_text(composition)))
    }
}

// ---------------------------------------------------------------------------
// Human-readable description
// ---------------------------------------------------------------------------

fn human_readable_lines(composition: &Composition) -> Vec<String> {
    let mut lines = Vec::new();

    let _ = writeln_capture(
        &mut lines,
        format!(
            "Composition: \"{}\" (seed: {})",
            composition.name,
            short_path(&composition.seed.path),
        ),
    );
    lines.push(String::new());

    match &composition.strategy {
        CompositionStrategy::Collage(coll) => {
            describe_visual(composition, &coll.layout, &mut lines);
        }
        CompositionStrategy::AudioMix(mix) => {
            describe_audio(composition, mix, &mut lines);
        }
        CompositionStrategy::Timeline(tl) => {
            let _ = writeln_capture(
                &mut lines,
                format!(
                    "Timeline: {:.1}s total, pacing = {:?}",
                    tl.total_duration_secs, tl.pacing
                ),
            );
            describe_member_list(composition, &mut lines);
        }
        CompositionStrategy::TextWeave(tw) => {
            let _ = writeln_capture(
                &mut lines,
                format!("Text weave: mode = {:?}, max_words = {}", tw.mode, tw.max_words),
            );
            describe_member_list(composition, &mut lines);
        }
        CompositionStrategy::CrossModal(cm) => {
            let _ = writeln_capture(
                &mut lines,
                format!("Cross-modal: layout = {:?}", cm.layout),
            );
            describe_member_list(composition, &mut lines);
        }
    }

    if !composition.relationships.is_empty() {
        lines.push(String::new());
        lines.push("Relationships:".to_string());
        for r in &composition.relationships {
            let from = composition
                .members
                .get(r.from)
                .map(|m| m.artifact.path.as_str())
                .unwrap_or("?");
            let to = composition
                .members
                .get(r.to)
                .map(|m| m.artifact.path.as_str())
                .unwrap_or("?");
            let _ = writeln_capture(
                &mut lines,
                format!("  {} {:?} {}", short_path(from), r.kind, short_path(to)),
            );
        }
    }

    lines
}

fn describe_visual(
    composition: &Composition,
    layout: &LayoutMode,
    lines: &mut Vec<String>,
) {
    let slots = resolve_slots(&composition.members, layout);
    for (i, (member, slot)) in composition.members.iter().zip(slots.iter()).enumerate() {
        let _ = writeln_capture(
            lines,
            format!(
                "Slot {} [{}]: {}",
                i + 1,
                role_label(member.role),
                short_path(&member.artifact.path),
            ),
        );
        let (px, py, pw, ph) = (slot.region.x, slot.region.y, slot.region.w, slot.region.h);
        let _ = writeln_capture(
            lines,
            format!(
                "  → Region: {:.0}%×{:.0}% at ({:.0}%, {:.0}%) | Opacity: {:.2} | Blend: {:?}",
                pw * 100.0,
                ph * 100.0,
                px * 100.0,
                py * 100.0,
                slot.opacity,
                slot.blend_mode,
            ),
        );
        if !member.match_explanation.is_empty() {
            let _ = writeln_capture(
                lines,
                format!("  → Matched: {} (score {:.2})", member.match_explanation, member.match_score),
            );
        }
        lines.push(String::new());
    }
}

fn describe_audio(
    composition: &Composition,
    mix: &crate::strategy::AudioMixStrategy,
    lines: &mut Vec<String>,
) {
    let _ = writeln_capture(
        lines,
        format!(
            "Audio mix: mode = {:?}, crossfade = {}ms, sample_rate = {}",
            mix.mode, mix.crossfade_ms, mix.sample_rate
        ),
    );
    lines.push(String::new());
    for (i, m) in composition.members.iter().enumerate() {
        let stem = mix.stem_roles.get(&m.role);
        let stem_desc = stem
            .map(|s| format!(" gain {:+.1}dB pan {:+.2}", s.gain_db, s.pan))
            .unwrap_or_default();
        let _ = writeln_capture(
            lines,
            format!(
                "Track {} [{}]: {}{}",
                i + 1,
                role_label(m.role),
                short_path(&m.artifact.path),
                stem_desc,
            ),
        );
        if !m.match_explanation.is_empty() {
            let _ = writeln_capture(
                lines,
                format!("  → Matched: {} (score {:.2})", m.match_explanation, m.match_score),
            );
        }
    }
}

fn describe_member_list(composition: &Composition, lines: &mut Vec<String>) {
    for (i, m) in composition.members.iter().enumerate() {
        let _ = writeln_capture(
            lines,
            format!(
                "  {}. [{}] {} (score {:.2})",
                i + 1,
                role_label(m.role),
                short_path(&m.artifact.path),
                m.match_score,
            ),
        );
    }
}

fn role_label(role: MemberRole) -> &'static str {
    role.as_str()
}

fn short_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string())
}

fn writeln_capture(lines: &mut Vec<String>, line: String) -> Result<()> {
    let mut s = String::new();
    let _ = write!(s, "{line}");
    lines.push(s);
    Ok(())
}
