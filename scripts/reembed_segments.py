#!/usr/bin/env python3
"""Re-embed existing image segments with a better CLIP model.

Usage:
    python reembed_segments.py --db JB5TB_Corpus.db [--model ViT-L-14] [--batch-size 32]

Reads existing segment bounding boxes, crops from original images,
and embeds with a larger/better CLIP model. Stores alongside existing
embeddings with a new model tag.

Resumable: skips segments that already have the target model embedding.
"""

import argparse
import hashlib
import sqlite3
import struct
import time
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

def embed_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def main():
    parser = argparse.ArgumentParser(description="Re-embed segments with better CLIP")
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="ViT-L-14", help="CLIP model variant")
    parser.add_argument("--pretrained", default="datacomp_xl_s13b_b90k", help="Pretrained weights")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_tag = f"clip:{args.model}"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model} ({args.pretrained})")
    print(f"Tag: {model_tag}")

    # Load CLIP
    print("Loading CLIP...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    clip_model.eval()
    print("CLIP loaded.")

    # Get segments that need re-embedding
    conn = sqlite3.connect(args.db)
    total_segs = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE segment_type = 'region' AND bbox_x IS NOT NULL"
    ).fetchone()[0]
    already_done = conn.execute(
        "SELECT COUNT(*) FROM segment_embeddings WHERE model = ?", (model_tag,)
    ).fetchone()[0]
    print(f"Total region segments: {total_segs}")
    print(f"Already embedded with {model_tag}: {already_done}")

    # Get pending segments
    rows = conn.execute('''
        SELECT s.id, s.path, s.bbox_x, s.bbox_y, s.bbox_w, s.bbox_h
        FROM segments s
        WHERE s.segment_type = 'region' AND s.bbox_x IS NOT NULL
        AND s.id NOT IN (SELECT segment_id FROM segment_embeddings WHERE model = ?)
    ''', (model_tag,)).fetchall()
    conn.close()

    pending = len(rows)
    print(f"Pending: {pending}")
    if pending == 0:
        print("All segments already embedded.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_ids = []
    batch_crops = []

    def flush_batch():
        nonlocal batch_ids, batch_crops, done
        if not batch_crops:
            return
        try:
            tensors = torch.stack(batch_crops).to(device)
            with torch.no_grad():
                features = clip_model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy()

            conn2 = sqlite3.connect(args.db)
            conn2.execute("PRAGMA journal_mode=WAL")
            emb_rows = []
            for i, seg_id in enumerate(batch_ids):
                vec_bytes = embed_to_bytes(features_np[i])
                dim = features_np.shape[1]
                emb_rows.append((seg_id, model_tag, vec_bytes, dim))
            conn2.executemany(
                "INSERT OR REPLACE INTO segment_embeddings (segment_id, model, vector, dim) VALUES (?, ?, ?, ?)",
                emb_rows,
            )
            conn2.commit()
            conn2.close()
            done += len(batch_ids)
        except Exception as e:
            failed_count = len(batch_ids)
            if done < 10:
                print(f"\nBatch error: {e}")
        batch_ids = []
        batch_crops = []

    for seg_id, path, bx, by, bw, bh in rows:
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            img = Image.open(path).convert('RGB')
            w, h = img.size
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = min(int((bx + bw) * w), w), min(int((by + bh) * h), h)
            if x2 <= x1 or y2 <= y1:
                failed += 1
                continue
            crop = img.crop((x1, y1, x2, y2))
            tensor = clip_preprocess(crop)
            batch_ids.append(seg_id)
            batch_crops.append(tensor)

            if len(batch_crops) >= args.batch_size:
                flush_batch()

        except Exception:
            failed += 1

        total_processed = done + failed
        if total_processed % 100 == 0 and total_processed > 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = pending - total_processed
            eta = remaining / rate if rate > 0 else 0
            print(f"\r  {done}/{pending} ({done*100//pending}%) | {rate:.1f} segs/s | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    # Final flush
    flush_batch()

    elapsed = time.time() - start_time
    print(f"\nDone: {done} embedded, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
