//! Spatial layout engine for visual compositions.
//!
//! Resolves a layout mode + a list of members into concrete slot definitions.

use crate::strategy::{BlendMode, LayoutMode, NormalizedRect, SlotDefinition};
use crate::{CompositionMember, MemberRole};

/// Default slot for a given role under `LayoutMode::RoleBased`.
fn default_slot_for_role(role: MemberRole) -> SlotDefinition {
    match role {
        MemberRole::Background | MemberRole::Seed => SlotDefinition {
            role,
            region: NormalizedRect::full(),
            z_order: 0,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        },
        MemberRole::Texture => SlotDefinition {
            role,
            region: NormalizedRect::full(),
            z_order: 1,
            opacity: 0.35,
            blend_mode: BlendMode::Overlay,
        },
        MemberRole::Structure => SlotDefinition {
            role,
            region: NormalizedRect { x: 0.0, y: 0.0, w: 1.0, h: 0.4 },
            z_order: 2,
            opacity: 0.9,
            blend_mode: BlendMode::Multiply,
        },
        MemberRole::Subject => SlotDefinition {
            role,
            region: NormalizedRect { x: 0.3, y: 0.25, w: 0.5, h: 0.5 },
            z_order: 3,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        },
        MemberRole::Accent => SlotDefinition {
            role,
            region: NormalizedRect { x: 0.05, y: 0.6, w: 0.25, h: 0.3 },
            z_order: 4,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        },
        MemberRole::Bridge => SlotDefinition {
            role,
            region: NormalizedRect { x: 0.0, y: 0.4, w: 1.0, h: 0.2 },
            z_order: 1,
            opacity: 0.7,
            blend_mode: BlendMode::SoftLight,
        },
    }
}

/// Resolve concrete slot definitions for each member, given a layout mode.
///
/// Output is parallel to the input `members` slice — index i in the result
/// corresponds to member i.
pub fn resolve_slots(
    members: &[CompositionMember],
    layout: &LayoutMode,
) -> Vec<SlotDefinition> {
    match layout {
        LayoutMode::RoleBased => members
            .iter()
            .map(|m| {
                m.slot_override
                    .clone()
                    .unwrap_or_else(|| default_slot_for_role(m.role))
            })
            .collect(),

        LayoutMode::Grid { cols, rows } => {
            let cols = (*cols).max(1);
            let rows = (*rows).max(1);
            let cell_w = 1.0 / cols as f64;
            let cell_h = 1.0 / rows as f64;
            let total = (cols * rows) as usize;

            members
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    if let Some(over) = &m.slot_override {
                        return over.clone();
                    }
                    let idx = i % total;
                    let col = (idx as u32) % cols;
                    let row = (idx as u32) / cols;
                    SlotDefinition {
                        role: m.role,
                        region: NormalizedRect {
                            x: col as f64 * cell_w,
                            y: row as f64 * cell_h,
                            w: cell_w,
                            h: cell_h,
                        },
                        z_order: i as i32,
                        opacity: 1.0,
                        blend_mode: BlendMode::Normal,
                    }
                })
                .collect()
        }

        LayoutMode::Balanced => {
            // Simple balance: split into rows of up to 3 cells. Honors role
            // for z_order so background sits at the bottom.
            let n = members.len().max(1);
            let cols = (n as f64).sqrt().ceil() as u32;
            let cols = cols.max(1);
            let rows = (n as u32).div_ceil(cols);
            let cell_w = 1.0 / cols as f64;
            let cell_h = 1.0 / rows as f64;

            members
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    if let Some(over) = &m.slot_override {
                        return over.clone();
                    }
                    let col = (i as u32) % cols;
                    let row = (i as u32) / cols;
                    let z = match m.role {
                        MemberRole::Background | MemberRole::Seed => 0,
                        MemberRole::Texture => 1,
                        MemberRole::Bridge => 2,
                        MemberRole::Structure => 3,
                        MemberRole::Subject => 4,
                        MemberRole::Accent => 5,
                    };
                    SlotDefinition {
                        role: m.role,
                        region: NormalizedRect {
                            x: col as f64 * cell_w,
                            y: row as f64 * cell_h,
                            w: cell_w,
                            h: cell_h,
                        },
                        z_order: z,
                        opacity: 1.0,
                        blend_mode: BlendMode::Normal,
                    }
                })
                .collect()
        }

        LayoutMode::Custom(slots) => members
            .iter()
            .enumerate()
            .map(|(i, m)| {
                if let Some(over) = &m.slot_override {
                    over.clone()
                } else if let Some(s) = slots.get(i) {
                    s.clone()
                } else {
                    default_slot_for_role(m.role)
                }
            })
            .collect(),
    }
}
