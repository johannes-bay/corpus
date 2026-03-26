#!/usr/bin/env python3
"""Embed whole images with SigLIP-SO400M via MLX on Apple Silicon.

Usage:
    python embed_siglip_mlx.py --db path/to/JB5TB_Corpus.db [--batch-size 16]

Stores embeddings in the embeddings table with model tag 'siglip:SO400M-384'.
Resumable: skips images that already have this model's embedding.
"""

import argparse
import sqlite3
import struct
import time
import os
from pathlib import Path

import numpy as np
import mlx.core as mx
from PIL import Image

MODEL_TAG = "siglip:SO400M-384"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}


def preprocess_image(img):
    """Resize to 384x384 and normalize for SigLIP."""
    img = img.convert('RGB').resize((384, 384), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return ((arr - mean) / std).transpose(2, 0, 1)  # HWC -> CHW


def embed_to_bytes(embedding) -> bytes:
    arr = np.array(embedding, dtype=np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def get_pending_paths(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in IMAGE_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT e.path FROM embeddings e WHERE e.model = ?
        )
    """
    paths = [r[0] for r in conn.execute(query, (MODEL_TAG,)).fetchall()]
    conn.close()
    return paths


def save_batch(db_path: str, results: list[tuple[str, bytes, int]]):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (path, model, vector, dim) VALUES (?, ?, ?, ?)",
        [(path, MODEL_TAG, vec, dim) for path, vec, dim in results],
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Embed images with SigLIP via MLX")
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print(f"Model: {MODEL_TAG}")
    print("Loading SigLIP via MLX...")
    from mlx_embeddings.utils import load
    model, _ = load('mlx-community/siglip-so400m-patch14-384')
    print("Loaded.")

    pending = get_pending_paths(args.db)
    total = len(pending)
    print(f"Images to embed: {total}")
    if total == 0:
        print("All images already embedded.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_paths = []
    batch_tensors = []

    def flush():
        nonlocal batch_paths, batch_tensors, done
        if not batch_tensors:
            return
        try:
            stacked = mx.array(np.stack(batch_tensors))
            features = model.get_image_features(stacked)
            # Normalize
            norms = mx.sqrt(mx.sum(features * features, axis=-1, keepdims=True))
            features = features / (norms + 1e-12)
            mx.eval(features)
            features_np = np.array(features)

            results = []
            dim = features_np.shape[1]
            for i, path in enumerate(batch_paths):
                vec_bytes = embed_to_bytes(features_np[i])
                results.append((path, vec_bytes, dim))
            save_batch(args.db, results)
            done += len(batch_paths)
        except Exception as e:
            if done < 5:
                print(f"\nBatch error: {e}")
        batch_paths = []
        batch_tensors = []

    for path in pending:
        if not Path(path).exists():
            failed += 1
            continue
        try:
            img = Image.open(path).convert('RGB')
            tensor = preprocess_image(img)
            batch_paths.append(path)
            batch_tensors.append(tensor)
            if len(batch_tensors) >= args.batch_size:
                flush()
        except Exception:
            failed += 1

        total_processed = done + failed
        if total_processed % 100 == 0 and total_processed > 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - total_processed
            eta = remaining / rate if rate > 0 else 0
            print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} img/s | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    flush()
    elapsed = time.time() - start_time
    print(f"\nDone: {done} embedded, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
