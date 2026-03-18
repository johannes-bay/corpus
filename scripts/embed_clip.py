#!/usr/bin/env python3
"""Generate CLIP embeddings for all image files in the corpus database.

Usage:
    python embed_clip.py --db path/to/JB5TB_Corpus.db [--batch-size 64] [--model ViT-B-32]

Stores embeddings in the `embeddings` table (path, model, vector, dim).
Resumable: skips files that already have an embedding for the given model.
"""

import argparse
import sqlite3
import struct
import sys
import time
from pathlib import Path

import torch
import open_clip
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_pending_files(db_path: str, model_name: str) -> list[str]:
    """Get image file paths that don't yet have an embedding."""
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in IMAGE_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT e.path FROM embeddings e WHERE e.model = ?
        )
    """
    cursor = conn.execute(query, (model_name,))
    paths = [row[0] for row in cursor.fetchall()]
    conn.close()
    return paths


def embed_to_bytes(embedding: torch.Tensor) -> bytes:
    """Convert a 1-D float32 tensor to little-endian bytes."""
    arr = embedding.cpu().float().numpy()
    return struct.pack(f'<{len(arr)}f', *arr)


def save_batch(db_path: str, model_name: str, results: list[tuple[str, bytes, int]]):
    """Save a batch of embeddings to the database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO embeddings (path, model, vector, dim)
           VALUES (?, ?, ?, ?)""",
        [(path, model_name, vec_bytes, dim) for path, vec_bytes, dim in results],
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for images")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="Pretrained weights")
    args = parser.parse_args()

    model_tag = f"clip:{args.model}"
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model} ({args.pretrained})")

    # Load model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()

    # Get pending files
    pending = get_pending_files(args.db, model_tag)
    total = len(pending)
    print(f"Files to embed: {total}")

    if total == 0:
        print("All images already embedded.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch_paths = pending[batch_start:batch_start + batch_size]
        images = []
        valid_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img)
                images.append(img_tensor)
                valid_paths.append(path)
            except Exception:
                failed += 1
                continue

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        # Save batch
        results = []
        dim = features.shape[1]
        for i, path in enumerate(valid_paths):
            vec_bytes = embed_to_bytes(features[i])
            results.append((path, vec_bytes, dim))

        save_batch(args.db, model_tag, results)
        done += len(valid_paths)

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done - failed) / rate if rate > 0 else 0
        print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} img/s | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} embedded, {failed} failed, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
