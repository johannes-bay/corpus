#!/usr/bin/env python3
"""Concept-driven composition using concept graph + contrastive CLIP + CLAP.

Usage:
    python concept_compose.py --db JB5TB_Corpus.db --concepts "arm,ball" --out output/

Combines:
  1. Concept graph (FTS anchors + relational edge traversal)
  2. Contrastive CLIP segment search (positive - negative prompts)
  3. CLAP audio search via text descriptions
  4. Quality-filtered text overlays from document chunks
"""

import argparse
import sqlite3
import struct
import os
import random
import re
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bytes_to_f32(blob, dim):
    return [struct.unpack_from('<f', blob, i * 4)[0] for i in range(dim)]

def cosine_batch(query, matrix):
    q = np.array(query, dtype=np.float32)
    dots = matrix @ q
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    return dots / (norms + 1e-12)

# ---------------------------------------------------------------------------
# Contrastive concept embedding
# ---------------------------------------------------------------------------

# For common concepts, define positive and negative prompts to sharpen retrieval.
# Negatives suppress visually similar but semantically different results.
CONTRASTIVE_PROMPTS = {
    "arm": {
        "positive": [
            "a human arm reaching out", "arm with hand and fingers",
            "person extending their arm", "arm throwing something",
            "muscular arm flexing", "arm in motion gesture",
        ],
        "negative": [
            "torso body", "leg thigh", "plain skin texture",
            "wooden surface", "blurry beige rectangle", "tan wall",
        ],
    },
    "ball": {
        "positive": [
            "a round sports ball", "soccer ball football",
            "basketball orange ball", "tennis ball green",
            "ball in the air flying", "globe sphere round object",
        ],
        "negative": [
            "circle icon flat", "button UI element", "dot pattern",
            "ring loop", "wheel tire", "clock face", "flat round logo",
        ],
    },
    "face": {
        "positive": [
            "human face portrait", "person's face closeup",
            "face with eyes nose mouth", "smiling face",
        ],
        "negative": [
            "mask", "emoji", "cartoon face", "clock face",
            "building facade", "surface texture",
        ],
    },
}

def build_concept_vector(clip_model, tokenizer, device, concept_name):
    """Build a contrastive concept embedding vector."""

    if concept_name in CONTRASTIVE_PROMPTS:
        prompts = CONTRASTIVE_PROMPTS[concept_name]
        pos_texts = prompts["positive"]
        neg_texts = prompts["negative"]
    else:
        # Generic: use the concept name with descriptive context
        pos_texts = [
            f"a {concept_name}",
            f"photo of {concept_name}",
            f"{concept_name} in real life",
            f"clear image of {concept_name}",
        ]
        neg_texts = [
            "blurry background", "plain texture", "abstract pattern",
            "UI icon flat design", "text document page",
        ]

    with torch.no_grad():
        pos_toks = tokenizer(pos_texts).to(device)
        pos_feats = clip_model.encode_text(pos_toks)
        pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)
        pos_avg = pos_feats.cpu().numpy().mean(axis=0)

        neg_toks = tokenizer(neg_texts).to(device)
        neg_feats = clip_model.encode_text(neg_toks)
        neg_feats = neg_feats / neg_feats.norm(dim=-1, keepdim=True)
        neg_avg = neg_feats.cpu().numpy().mean(axis=0)

    # Subtract negative direction
    vec = pos_avg - 0.3 * neg_avg
    return vec / np.linalg.norm(vec)

# ---------------------------------------------------------------------------
# Text quality filter
# ---------------------------------------------------------------------------

# Patterns to reject from rendered text
_REJECT_PATTERNS = [
    re.compile(r'(?:EMPLOYMENT|CONTRACT|SALARY|EMPLOYER|INVOICE|LICENSE)', re.I),
    re.compile(r'\\[a-z]\d'),  # RTF codes
    re.compile(r'[0-9a-f]{8,}', re.I),  # long hex strings
    re.compile(r'http[s]?://'),
    re.compile(r'@gmail|@yahoo|@hotmail'),
    re.compile(r'(?:Apache|GPL|MIT) License', re.I),
    re.compile(r'^\s*[0-9\s\.\,\-]+\s*$'),  # pure numbers
]

def is_good_text(label: str) -> bool:
    """Filter text chunks for composition rendering quality."""
    if not label or len(label) < 30:
        return False
    # Alpha ratio
    alpha = sum(1 for c in label if c.isalpha() or c.isspace())
    if alpha / len(label) < 0.75:
        return False
    # Real words
    words = [w for w in label.split() if len(w) >= 3 and any(c.isalpha() for c in w)]
    if len(words) < 6:
        return False
    # Reject patterns
    for pat in _REJECT_PATTERNS:
        if pat.search(label):
            return False
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Concept-driven composition")
    parser.add_argument("--db", required=True)
    parser.add_argument("--concepts", required=True, help="Comma-separated concepts")
    parser.add_argument("--out", default="output/concept")
    parser.add_argument("--width", type=int, default=3000)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--max-segments", type=int, default=300)
    args = parser.parse_args()

    concepts = [c.strip() for c in args.concepts.split(",")]
    os.makedirs(args.out, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load CLIP
    print("Loading CLIP...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()

    # Build contrastive vectors
    concept_vecs = {}
    for c in concepts:
        concept_vecs[c] = build_concept_vector(clip_model, tokenizer, device, c)
        print(f"  Concept '{c}' embedded")

    # Load segments
    print("Loading segments...")
    conn = sqlite3.connect(args.db)
    seg_rows = conn.execute('''
        SELECT s.id, s.path, s.bbox_x, s.bbox_y, s.bbox_w, s.bbox_h, s.area_frac,
               se.vector, se.dim
        FROM segments s JOIN segment_embeddings se ON se.segment_id = s.id
        WHERE s.segment_type = 'region' AND se.model = 'clip:ViT-B-32'
        AND s.bbox_x IS NOT NULL AND s.area_frac > 0.03
    ''').fetchall()

    seg_info, seg_matrix = [], []
    for sid, path, bx, by, bw, bh, area, vb, dim in seg_rows:
        seg_info.append((sid, path, bx, by, bw, bh, area))
        seg_matrix.append(bytes_to_f32(vb, dim))
    seg_matrix = np.array(seg_matrix, dtype=np.float32)
    print(f"  {len(seg_matrix)} segments loaded")

    # Score segments per concept
    concept_scores = {}
    for c, vec in concept_vecs.items():
        concept_scores[c] = cosine_batch(vec, seg_matrix)

    # Concept graph — FTS anchors
    print("Concept graph...")
    fts_terms = concepts + [f"{c}s" for c in concepts]  # add plurals
    concept_files = {}
    for term in fts_terms:
        try:
            rows = conn.execute(
                "SELECT path, source_type FROM corpus_fts WHERE content MATCH ? ORDER BY rank LIMIT 100",
                (term,)
            ).fetchall()
            for path, src in rows:
                score = {'filename': 1.0, 'title': 0.9, 'artist': 0.8, 'album': 0.8,
                         'folder': 0.6, 'chunk': 0.5, 'genre': 0.5}.get(src, 0.4)
                if path not in concept_files or concept_files[path] < score:
                    concept_files[path] = score
        except Exception:
            pass

    # Edge traversal
    for path in list(concept_files.keys())[:50]:
        score = concept_files[path]
        try:
            siblings = conn.execute(
                "SELECT path FROM files WHERE parent_folder = (SELECT parent_folder FROM files WHERE path = ?) AND path != ? LIMIT 20",
                (path, path)).fetchall()
            for sib in siblings:
                prop = score * 0.5
                if sib[0] not in concept_files or concept_files[sib[0]] < prop:
                    concept_files[sib[0]] = prop
        except Exception:
            pass
    print(f"  {len(concept_files)} concept graph files")

    # Boost segments from concept graph files
    for i, (sid, path, *_) in enumerate(seg_info):
        if path in concept_files:
            for c in concept_scores:
                concept_scores[c][i] = concept_scores[c][i] * 1.5 + concept_files[path] * 0.2

    # Text matches — quality filtered
    print("Finding text...")
    blended = sum(concept_vecs.values())
    blended = blended / np.linalg.norm(blended)

    text_rows = conn.execute(
        "SELECT s.label, se.vector, se.dim FROM segment_embeddings se "
        "JOIN segments s ON s.id=se.segment_id WHERE s.segment_type='chunk' "
        "AND se.model='clip:ViT-B-32' AND s.label IS NOT NULL "
        "AND LENGTH(s.label)>30 ORDER BY RANDOM() LIMIT 8000"
    ).fetchall()

    text_labels, text_vecs_list = [], []
    for label, vb, dim in text_rows:
        if is_good_text(label):
            text_labels.append(label[:120])
            text_vecs_list.append(bytes_to_f32(vb, dim))

    text_lines = []
    if text_vecs_list:
        tmat = np.array(text_vecs_list, dtype=np.float32)
        tscores = cosine_batch(blended, tmat)
        for i in np.argsort(tscores)[::-1][:12]:
            text_lines.append(text_labels[i])
    print(f"  {len(text_lines)} quality text matches")

    conn.close()

    # === COMPOSITING ===
    print("Compositing...")
    W, H = args.width, args.height
    canvas = Image.new('RGBA', (W, H), (6, 4, 10, 255))
    random.seed(42)

    placed = 0
    sources = set()
    n_concepts = len(concepts)

    def crop_seg(path, bx, by, bw, bh):
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert('RGBA')
        w, h = img.size
        x1, y1 = int(bx * w), int(by * h)
        x2, y2 = min(int((bx + bw) * w), w), min(int((by + bh) * h), h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return img.crop((x1, y1, x2, y2))

    def place(canvas, crop, cx, cy, size, alpha, angle=0):
        asp = crop.width / crop.height
        cw = max(30, int(size * min(asp, 3)))
        ch = max(30, int(size / max(asp, 0.33)))
        crop = crop.resize((cw, ch), Image.LANCZOS)
        if angle:
            crop = crop.rotate(angle, expand=True, resample=Image.BICUBIC)
        r, g, b, a = crop.split()
        a = a.point(lambda x: min(x, alpha))
        crop = Image.merge('RGBA', (r, g, b, a))
        canvas.paste(crop, (cx - crop.width // 2, cy - crop.height // 2), crop)

    # Assign spatial zones per concept (spread across canvas)
    zones = {}
    for i, c in enumerate(concepts):
        cx = W * (i + 0.5) / n_concepts
        zones[c] = cx

    # Background layer — blended top segments
    combined = sum(concept_scores.values()) / n_concepts
    for idx in np.argsort(combined)[::-1][:20]:
        sid, path, bx, by, bw, bh, area = seg_info[idx]
        try:
            crop = crop_seg(path, bx, by, bw, bh)
            if not crop:
                continue
            place(canvas, crop, random.randint(0, W), random.randint(0, H),
                  random.randint(400, 700), random.randint(20, 50), random.uniform(-20, 20))
            placed += 1
            sources.add(path)
        except Exception:
            continue

    # Per-concept layers
    per_concept = args.max_segments // n_concepts
    for c_name, scores in concept_scores.items():
        cx_zone = zones[c_name]
        for idx in np.argsort(scores)[::-1][:per_concept]:
            sid, path, bx, by, bw, bh, area = seg_info[idx]
            try:
                crop = crop_seg(path, bx, by, bw, bh)
                if not crop:
                    continue
                cx = int(random.gauss(cx_zone, W * 0.18))
                cy = int(random.gauss(H * 0.5, H * 0.35))
                size = int(50 + scores[idx] * 450)
                alpha = int(80 + scores[idx] * 200)
                place(canvas, crop, cx, cy, size, min(alpha, 240), random.uniform(-30, 30))
                placed += 1
                sources.add(path)
            except Exception:
                continue

    print(f"Placed {placed} elements from {len(sources)} sources")

    # Text overlay
    draw = ImageDraw.Draw(canvas)
    try:
        fonts = {s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)
                 for s in [32, 24, 18, 14, 160]}
    except Exception:
        fonts = {s: ImageFont.load_default() for s in [32, 24, 18, 14, 160]}

    # Faint concept labels
    for i, c in enumerate(concepts):
        x = int(zones[c]) - 200
        draw.text((max(40, x), 50), c.upper(), fill=(255, 255, 255, 20), font=fonts[160])

    # Filtered text fragments
    fsizes = [24, 18, 14, 14, 18, 14, 14, 14, 18, 14, 14, 14]
    for i, line in enumerate(text_lines):
        x = random.randint(30, W - 500)
        y = random.randint(50, H - 70)
        f = fonts[fsizes[i % len(fsizes)]]
        bbox = draw.textbbox((x, y), line, font=f)
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                       fill=(0, 0, 0, 130))
        draw.text((x, y), line, fill=(210, 215, 225, 210), font=f)

    # Bottom bar
    draw.line([(30, H - 45), (W - 30, H - 45)], fill=(124, 58, 237, 180), width=2)
    draw.text((30, H - 38),
              f'CORPUS — concepts: "{", ".join(concepts)}" | {placed} elements, {len(sources)} sources | concept graph + contrastive CLIP',
              fill=(120, 120, 150, 200), font=fonts[14])

    final = Image.new('RGB', (W, H), (6, 4, 10))
    final.paste(canvas, mask=canvas.split()[3])
    outpath = os.path.join(args.out, f'{"_".join(concepts)}.png')
    final.save(outpath, quality=95)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
