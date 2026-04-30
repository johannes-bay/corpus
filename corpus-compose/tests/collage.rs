//! Integration test: end-to-end image collage rendering.
//!
//! Generates two synthetic JPEG inputs, composes a collage, and verifies the
//! output PNG is a valid image with the expected dimensions.

use std::path::Path;

use image::{DynamicImage, RgbImage};

use corpus_compose::strategy::{
    CollageStrategy, CompositionStrategy, EdgeBlend, LayoutMode,
};
use corpus_compose::{
    ArtifactRef, CollageRenderer, Composition, CompositionMember, ManifestRenderer, MemberRole,
    Renderer,
};

fn synth_image(rgb: [u8; 3], w: u32, h: u32, path: &Path) {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel_mut(x, y);
            p[0] = rgb[0];
            p[1] = rgb[1];
            p[2] = rgb[2];
        }
    }
    DynamicImage::ImageRgb8(img)
        .save(path)
        .expect("write synth image");
}

#[test]
fn collage_renders_from_two_images() {
    let tmp = tempdir().expect("tempdir");

    let bg_path = tmp.join("bg.jpg");
    let subj_path = tmp.join("subj.jpg");
    synth_image([35, 60, 90], 800, 600, &bg_path);
    synth_image([220, 200, 110], 600, 400, &subj_path);

    let strategy = CompositionStrategy::Collage(CollageStrategy {
        layout: LayoutMode::RoleBased,
        tonal_consistency: 0.5,
        edge_blending: EdgeBlend::Feathered(8),
        canvas_size: (640, 360),
    });
    let mut comp = Composition::new(
        "test-collage",
        ArtifactRef::file(bg_path.to_string_lossy().to_string(), "image"),
        strategy,
    );
    comp.members.push(CompositionMember::new(
        ArtifactRef::file(bg_path.to_string_lossy().to_string(), "image"),
        MemberRole::Background,
        0.95,
        "synthetic background",
    ));
    comp.members.push(CompositionMember::new(
        ArtifactRef::file(subj_path.to_string_lossy().to_string(), "image"),
        MemberRole::Subject,
        0.88,
        "synthetic subject",
    ));

    // Manifest first (always available)
    let manifest = ManifestRenderer::new()
        .render(&comp, &tmp)
        .expect("manifest render");
    assert_eq!(manifest.composition.members.len(), 2);
    assert!(!manifest.instructions.is_empty());

    // Then the actual collage
    let renderer = CollageRenderer::new();
    let out = renderer.render(&comp, &tmp).expect("collage render");
    assert!(out.exists(), "output file should exist: {}", out.display());

    let img = image::open(&out).expect("decode output");
    assert_eq!(img.width(), 640);
    assert_eq!(img.height(), 360);

    // Preview should yield a JPEG
    let preview = renderer.preview(&comp).expect("preview");
    if let corpus_compose::PreviewData::Image(bytes) = preview {
        assert!(bytes.len() > 100, "preview image should not be empty");
    } else {
        panic!("expected Image preview");
    }
}

#[test]
fn manifest_describes_audio_strategy_without_rendering() {
    use corpus_compose::strategy::{AudioMixMode, AudioMixStrategy};

    let strategy = CompositionStrategy::AudioMix(AudioMixStrategy {
        mode: AudioMixMode::Sequence,
        crossfade_ms: 1500,
        ..Default::default()
    });
    let mut comp = Composition::new(
        "audio-stub",
        ArtifactRef::file("/no/such/seed.wav".to_string(), "audio"),
        strategy,
    );
    comp.members.push(CompositionMember::new(
        ArtifactRef::file("/no/such/seed.wav".to_string(), "audio"),
        MemberRole::Seed,
        1.0,
        "seed",
    ));
    comp.members.push(CompositionMember::new(
        ArtifactRef::file("/no/such/match.wav".to_string(), "audio"),
        MemberRole::Subject,
        0.5,
        "test match",
    ));

    let text = ManifestRenderer::new().render_text(&comp);
    assert!(text.contains("Audio mix"));
    assert!(text.contains("seed.wav"));
    assert!(text.contains("match.wav"));
}

// ---------------------------------------------------------------------------
// Tiny tempdir helper without pulling in the `tempfile` crate.
// ---------------------------------------------------------------------------

struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for TempDir {
    type Target = std::path::Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

fn tempdir() -> std::io::Result<TempDir> {
    let mut path = std::env::temp_dir();
    let suffix: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64);
    path.push(format!("corpus-compose-test-{:x}", suffix));
    std::fs::create_dir_all(&path)?;
    Ok(TempDir(path))
}
