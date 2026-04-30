//! Image collage renderer.
//!
//! Phase 1 capability:
//!   - Load source images
//!   - Crop to element bounding boxes (from segments)
//!   - Resize to fit slot regions on the canvas
//!   - Composite with Normal and Multiply blend modes
//!   - Apply feathered edges if requested
//!   - Output as PNG/JPEG

use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{DynamicImage, GenericImage, GenericImageView, Rgba, RgbaImage};
use tracing::warn;

use crate::export::{save_image, ImageFormatChoice};
use crate::layout::resolve_slots;
use crate::strategy::{
    BlendMode, CollageStrategy, CompositionStrategy, EdgeBlend, SlotDefinition,
};
use crate::{
    ComposeError, Composition, CompositionMember, MemberRole, PreviewData, Renderer, Result,
};

/// Image collage renderer.
#[derive(Debug, Clone)]
pub struct CollageRenderer {
    pub format: ImageFormatChoice,
    pub preview_max_dim: u32,
    pub jpeg_quality: u8,
}

impl Default for CollageRenderer {
    fn default() -> Self {
        Self {
            format: ImageFormatChoice::Png,
            preview_max_dim: 512,
            jpeg_quality: 88,
        }
    }
}

impl CollageRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: ImageFormatChoice) -> Self {
        self.format = format;
        self
    }

    fn collage_strategy<'a>(&self, comp: &'a Composition) -> Result<&'a CollageStrategy> {
        match &comp.strategy {
            CompositionStrategy::Collage(c) => Ok(c),
            other => Err(ComposeError::DomainMismatch {
                strategy: other.kind().to_string(),
                domain: "image".to_string(),
            }),
        }
    }

    fn render_canvas(&self, composition: &Composition) -> Result<RgbaImage> {
        let strat = self.collage_strategy(composition)?;
        let (cw, ch) = strat.canvas_size;

        let mut canvas: RgbaImage = RgbaImage::from_pixel(cw, ch, Rgba([14, 14, 16, 255]));

        let slots = resolve_slots(&composition.members, &strat.layout);
        // Sort by z_order so lower layers go down first.
        let mut indices: Vec<usize> = (0..composition.members.len()).collect();
        indices.sort_by_key(|&i| slots[i].z_order);

        for &i in &indices {
            let member = &composition.members[i];
            let slot = &slots[i];
            if let Err(e) = composite_member(&mut canvas, member, slot, strat) {
                warn!("compose: failed to composite {}: {e}", member.artifact.path);
            }
        }

        Ok(canvas)
    }
}

impl Renderer for CollageRenderer {
    type Output = PathBuf;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        let _ = self.collage_strategy(composition)?;
        Ok(())
    }

    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<PathBuf> {
        self.validate(composition)?;
        std::fs::create_dir_all(output_dir)?;

        let canvas = self.render_canvas(composition)?;
        let dynamic = DynamicImage::ImageRgba8(canvas);
        let path = output_dir.join(format!("{}.{}", composition.id, self.format.extension()));
        save_image(&dynamic, &path, self.format)?;
        Ok(path)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        self.validate(composition)?;
        let canvas = self.render_canvas(composition)?;
        let (w, h) = canvas.dimensions();
        let max_dim = self.preview_max_dim.max(64);
        let scale = (max_dim as f32 / w.max(h) as f32).min(1.0);
        let preview = if scale < 1.0 {
            let nw = ((w as f32) * scale).round() as u32;
            let nh = ((h as f32) * scale).round() as u32;
            DynamicImage::ImageRgba8(canvas).resize(nw.max(1), nh.max(1), FilterType::Lanczos3)
        } else {
            DynamicImage::ImageRgba8(canvas)
        };

        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        preview
            .to_rgb8()
            .write_to(&mut cursor, image::ImageFormat::Jpeg)?;
        Ok(PreviewData::Image(buf))
    }
}

// ---------------------------------------------------------------------------
// Compositing
// ---------------------------------------------------------------------------

fn composite_member(
    canvas: &mut RgbaImage,
    member: &CompositionMember,
    slot: &SlotDefinition,
    strat: &CollageStrategy,
) -> Result<()> {
    let path = &member.artifact.path;
    if !Path::new(path).exists() {
        return Err(ComposeError::SourceMissing(path.clone()));
    }

    let img = image::open(path)?;

    // Crop if there is an element_id with a segment bbox stored in the
    // explanation (Phase 1 has no DB lookup here — element_id is informational).
    // For now we just use the whole image; layout slots already shape placement.
    let cropped = img;

    let (cw, ch) = canvas.dimensions();
    let (sx, sy, sw, sh) = slot.region.to_pixels(cw, ch);

    // Resize to slot dimensions, preserving aspect by cover-fitting.
    let resized = cover_fit(&cropped, sw, sh);

    // Convert to RGBA.
    let rgba = resized.to_rgba8();

    // Apply edge blending (alpha modulation).
    let masked = apply_edge_blend(rgba, strat.edge_blending);

    // Apply opacity scaling.
    let masked = scale_alpha(masked, slot.opacity);

    // Apply blend mode.
    blend_into(canvas, &masked, sx, sy, slot.blend_mode);

    Ok(())
}

/// Resize while preserving aspect ratio so the result fully covers (target_w,
/// target_h), then center-crop to those exact dimensions.
fn cover_fit(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    if target_w == 0 || target_h == 0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }
    let scale = (target_w as f32 / w as f32).max(target_h as f32 / h as f32);
    let new_w = ((w as f32) * scale).round().max(target_w as f32) as u32;
    let new_h = ((h as f32) * scale).round().max(target_h as f32) as u32;
    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

    let crop_x = (new_w.saturating_sub(target_w)) / 2;
    let crop_y = (new_h.saturating_sub(target_h)) / 2;
    resized.crop_imm(crop_x, crop_y, target_w, target_h)
}

fn apply_edge_blend(mut img: RgbaImage, edge: EdgeBlend) -> RgbaImage {
    match edge {
        EdgeBlend::HardCut | EdgeBlend::MaskBased => img,
        EdgeBlend::Feathered(radius) => {
            if radius == 0 {
                return img;
            }
            let (w, h) = img.dimensions();
            let r = radius.min(w / 2).min(h / 2).max(1);
            for y in 0..h {
                for x in 0..w {
                    let dx = x.min(w - 1 - x);
                    let dy = y.min(h - 1 - y);
                    let d = dx.min(dy);
                    if d < r {
                        let factor = d as f32 / r as f32;
                        let p = img.get_pixel_mut(x, y);
                        let new_a = (p[3] as f32 * factor).clamp(0.0, 255.0) as u8;
                        p[3] = new_a;
                    }
                }
            }
            img
        }
    }
}

fn scale_alpha(mut img: RgbaImage, opacity: f64) -> RgbaImage {
    let opacity = opacity.clamp(0.0, 1.0) as f32;
    if (opacity - 1.0).abs() < f32::EPSILON {
        return img;
    }
    for p in img.pixels_mut() {
        p[3] = (p[3] as f32 * opacity).clamp(0.0, 255.0) as u8;
    }
    img
}

fn blend_into(
    canvas: &mut RgbaImage,
    overlay: &RgbaImage,
    dest_x: u32,
    dest_y: u32,
    mode: BlendMode,
) {
    let (cw, ch) = canvas.dimensions();
    let (ow, oh) = overlay.dimensions();
    for y in 0..oh {
        let cy = dest_y + y;
        if cy >= ch {
            break;
        }
        for x in 0..ow {
            let cx = dest_x + x;
            if cx >= cw {
                break;
            }
            let src = *overlay.get_pixel(x, y);
            let dst = canvas.get_pixel(cx, cy);
            let blended = blend_pixels(*dst, src, mode);
            unsafe {
                canvas.unsafe_put_pixel(cx, cy, blended);
            }
        }
    }
}

fn blend_pixels(dst: Rgba<u8>, src: Rgba<u8>, mode: BlendMode) -> Rgba<u8> {
    let alpha = src[3] as f32 / 255.0;
    if alpha == 0.0 {
        return dst;
    }
    let dr = dst[0] as f32;
    let dg = dst[1] as f32;
    let db = dst[2] as f32;
    let sr = src[0] as f32;
    let sg = src[1] as f32;
    let sb = src[2] as f32;

    let (br, bg, bb) = match mode {
        BlendMode::Normal => (sr, sg, sb),
        BlendMode::Multiply => (
            (dr * sr) / 255.0,
            (dg * sg) / 255.0,
            (db * sb) / 255.0,
        ),
        BlendMode::Screen => (
            255.0 - ((255.0 - dr) * (255.0 - sr)) / 255.0,
            255.0 - ((255.0 - dg) * (255.0 - sg)) / 255.0,
            255.0 - ((255.0 - db) * (255.0 - sb)) / 255.0,
        ),
        BlendMode::Overlay => (
            overlay_channel(dr, sr),
            overlay_channel(dg, sg),
            overlay_channel(db, sb),
        ),
        BlendMode::SoftLight => (
            soft_light_channel(dr, sr),
            soft_light_channel(dg, sg),
            soft_light_channel(db, sb),
        ),
    };

    // Standard "over" alpha compositing of the blended foreground onto dst.
    let inv = 1.0 - alpha;
    let r = (br * alpha + dr * inv).clamp(0.0, 255.0) as u8;
    let g = (bg * alpha + dg * inv).clamp(0.0, 255.0) as u8;
    let b = (bb * alpha + db * inv).clamp(0.0, 255.0) as u8;

    let a_dst = dst[3] as f32 / 255.0;
    let a_out = (alpha + a_dst * (1.0 - alpha)).clamp(0.0, 1.0);
    Rgba([r, g, b, (a_out * 255.0).round() as u8])
}

fn overlay_channel(d: f32, s: f32) -> f32 {
    if d < 128.0 {
        2.0 * d * s / 255.0
    } else {
        255.0 - 2.0 * (255.0 - d) * (255.0 - s) / 255.0
    }
}

fn soft_light_channel(d: f32, s: f32) -> f32 {
    let dn = d / 255.0;
    let sn = s / 255.0;
    let res = if sn < 0.5 {
        dn - (1.0 - 2.0 * sn) * dn * (1.0 - dn)
    } else {
        dn + (2.0 * sn - 1.0) * (g_w3c(dn) - dn)
    };
    res.clamp(0.0, 1.0) * 255.0
}

fn g_w3c(d: f32) -> f32 {
    if d <= 0.25 {
        ((16.0 * d - 12.0) * d + 4.0) * d
    } else {
        d.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Convenience: build a quick collage composition from a seed + member paths.
// Useful for callers without the association engine context (CLI smoke tests).
// ---------------------------------------------------------------------------

/// Build a default Collage composition from a seed image plus auxiliary paths.
/// The first path is treated as Background, the rest cycle through Structure,
/// Subject, Accent, Texture roles.
pub fn build_default_collage(
    name: impl Into<String>,
    seed_path: &str,
    member_paths: &[String],
) -> Composition {
    use crate::{ArtifactRef, CompositionMember};

    let seed_ref = ArtifactRef::file(seed_path, "image");
    let strategy = CompositionStrategy::Collage(CollageStrategy::default());

    let mut comp = Composition::new(name, seed_ref.clone(), strategy);

    comp.members.push(CompositionMember::new(
        seed_ref,
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

    for (i, path) in member_paths.iter().enumerate() {
        let role = role_cycle[i % role_cycle.len()];
        comp.members.push(CompositionMember::new(
            ArtifactRef::file(path, "image"),
            role,
            0.0,
            "auto-assigned role",
        ));
    }

    comp
}
