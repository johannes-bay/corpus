#!/usr/bin/env python3
"""Segment images with SAM2 and embed each region with CLIP.

Usage:
    python segment_images.py --db path/to/JB5TB_Corpus.db [--max-masks 20] [--min-area 0.01]

For each image:
  1. SAM2 automatic mask generation → object masks
  2. Crop each mask's bounding box region
  3. Embed each crop with CLIP ViT-B-32
  4. Store segments + segment_embeddings in SQLite

Resumable: skips images that already have segments.
"""

import argparse
import hashlib
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
SAM_MODEL = "sam2:large"
CLIP_MODEL = "clip:ViT-B-32"


def segment_id(path: str, segment_type: str, segment_key: str) -> str:
    """Deterministic segment ID."""
    raw = f"{path}\0{segment_type}\0{segment_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def get_pending_paths(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in IMAGE_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT DISTINCT s.path FROM segments s WHERE s.model = ?
        )
    """
    paths = [r[0] for r in conn.execute(query, (SAM_MODEL,)).fetchall()]
    conn.close()
    return paths


def save_batch(db_path: str, segments: list, seg_embeddings: list):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO segments
           (id, path, segment_type, segment_key, label, bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, area_frac, model, mask_rle)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        segments,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO segment_embeddings (segment_id, model, vector, dim)
           VALUES (?, ?, ?, ?)""",
        seg_embeddings,
    )
    conn.commit()
    conn.close()


def mask_to_rle(mask: np.ndarray) -> bytes:
    """Simple run-length encoding of a binary mask."""
    flat = mask.flatten().astype(np.uint8)
    runs = []
    i = 0
    while i < len(flat):
        val = flat[i]
        run_len = 1
        while i + run_len < len(flat) and flat[i + run_len] == val and run_len < 255:
            run_len += 1
        runs.append(val)
        runs.append(run_len)
        i += run_len
    return bytes(runs)


def main():
    parser = argparse.ArgumentParser(description="Segment images with SAM2 + CLIP")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--max-masks", type=int, default=10, help="Max masks per image")
    parser.add_argument("--min-area", type=float, default=0.01, help="Min mask area fraction")
    parser.add_argument("--max-area", type=float, default=0.95, help="Max mask area fraction")
    parser.add_argument("--batch-size", type=int, default=64, help="CLIP batch size")
    parser.add_argument("--max-dim", type=int, default=1024, help="Max image dimension for SAM2 (resize larger)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SAM2
    print("Loading SAM2...")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt", device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=8,          # 64 points: fast, still catches main objects
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
    )
    print("SAM2 loaded.")

    # Load CLIP
    print("Loading CLIP...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device,
    )
    clip_model.eval()
    print("CLIP loaded.")

    # Get pending files
    pending = get_pending_paths(args.db)
    total = len(pending)
    print(f"Images to segment: {total}")
    if total == 0:
        print("All images already segmented.")
        return

    done = 0
    failed = 0
    total_segments = 0
    start_time = time.time()

    for path in pending:
        if not Path(path).exists():
            failed += 1
            continue

        try:
            # Load image
            img = Image.open(path).convert("RGB")
            orig_w, orig_h = img.size

            # Resize for SAM2 if too large (saves VRAM)
            max_dim = args.max_dim
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                sam_img = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                sam_img = img
                scale = 1.0

            img_np = np.array(sam_img)
            h, w = img_np.shape[:2]
            total_pixels = h * w

            # Generate masks on (possibly resized) image
            masks = mask_generator.generate(img_np)

            # Filter by area
            masks = [m for m in masks
                     if args.min_area <= m['area'] / total_pixels <= args.max_area]

            # Sort by predicted IoU, take top N
            masks.sort(key=lambda m: m.get('predicted_iou', 0), reverse=True)
            masks = masks[:args.max_masks]

            if not masks:
                done += 1
                # Store empty marker so we don't retry
                sid = segment_id(path, "region", "none")
                save_batch(args.db,
                    [(sid, path, "region", "none", None, None, None, None, None, None, None, SAM_MODEL, None)],
                    [])
                continue

            # Crop regions and prepare for CLIP
            crops = []
            seg_rows = []
            for i, mask_data in enumerate(masks):
                seg_key = f"mask_{i}"
                sid = segment_id(path, "region", seg_key)

                bbox = mask_data['bbox']  # [x, y, w, h] in pixels (of resized image)
                bx, by, bw, bh = bbox
                # Normalize to 0-1 (resolution-independent)
                nx, ny, nw, nh = bx / w, by / h, bw / w, bh / h

                area_frac = mask_data['area'] / total_pixels
                confidence = mask_data.get('predicted_iou', 0.0)

                # RLE encode mask
                rle = mask_to_rle(mask_data['segmentation'])

                # Crop from ORIGINAL image for best CLIP quality
                ox1 = int(nx * orig_w)
                oy1 = int(ny * orig_h)
                ox2 = min(int((nx + nw) * orig_w), orig_w)
                oy2 = min(int((ny + nh) * orig_h), orig_h)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue

                crop = img.crop((ox1, oy1, ox2, oy2))
                crops.append((sid, crop))

                seg_rows.append((
                    sid, path, "region", seg_key, None,
                    nx, ny, nw, nh,
                    confidence, area_frac, SAM_MODEL, rle,
                ))

            # Batch embed crops with CLIP
            seg_emb_rows = []
            for batch_start in range(0, len(crops), args.batch_size):
                batch = crops[batch_start:batch_start + args.batch_size]
                tensors = torch.stack([clip_preprocess(c) for _, c in batch]).to(device)
                with torch.no_grad():
                    features = clip_model.encode_image(tensors)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.cpu().numpy()

                for j, (sid, _) in enumerate(batch):
                    vec_bytes = embed_to_bytes(features_np[j])
                    dim = features_np.shape[1]
                    seg_emb_rows.append((sid, CLIP_MODEL, vec_bytes, dim))

            # Save to DB
            save_batch(args.db, seg_rows, seg_emb_rows)
            total_segments += len(seg_rows)
            done += 1

        except Exception as e:
            failed += 1
            if done < 5:
                print(f"\nError on {path}: {e}")

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - done - failed
        eta = remaining / rate if rate > 0 else 0
        print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} img/s | {total_segments} segs | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} images, {total_segments} segments, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
