#!/usr/bin/env python3
"""Re-embed video scene keyframes with SigLIP via MLX.

Usage:
    python embed_siglip_video_mlx.py --db path/to/JB5TB_Corpus.db

Re-embeds existing scene segments (keyframes extracted by segment_video.py)
with SigLIP. Reads the original video, extracts the keyframe at midpoint,
and embeds with SigLIP.

Stores in segment_embeddings with model tag 'siglip:SO400M-384'.
Resumable.
"""

import argparse
import sqlite3
import struct
import subprocess
import tempfile
import time
import os
import numpy as np
import mlx.core as mx
from PIL import Image

MODEL_TAG = "siglip:SO400M-384"


def preprocess_image(img):
    img = img.convert('RGB').resize((384, 384), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return ((arr - mean) / std).transpose(2, 0, 1)


def embed_to_bytes(embedding) -> bytes:
    arr = np.array(embedding, dtype=np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def extract_frame(video_path, timestamp, output_path):
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
             '-vframes', '1', '-q:v', '2', output_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("Loading SigLIP via MLX...")
    from mlx_embeddings.utils import load
    model, _ = load('mlx-community/siglip-so400m-patch14-384')
    print("Loaded.")

    conn = sqlite3.connect(args.db)
    # Get scene segments not yet embedded with SigLIP
    rows = conn.execute('''
        SELECT s.id, s.path, s.time_start, s.time_end
        FROM segments s
        WHERE s.segment_type = 'scene'
        AND s.id NOT IN (SELECT segment_id FROM segment_embeddings WHERE model = ?)
    ''', (MODEL_TAG,)).fetchall()
    conn.close()

    total = len(rows)
    print(f"Scene keyframes to embed: {total}")
    if total == 0:
        print("All done.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_ids = []
    batch_tensors = []

    def flush():
        nonlocal batch_ids, batch_tensors, done
        if not batch_tensors:
            return
        try:
            stacked = mx.array(np.stack(batch_tensors))
            features = model.get_image_features(stacked)
            norms = mx.sqrt(mx.sum(features * features, axis=-1, keepdims=True))
            features = features / (norms + 1e-12)
            mx.eval(features)
            features_np = np.array(features)

            conn2 = sqlite3.connect(args.db)
            conn2.execute("PRAGMA journal_mode=WAL")
            dim = features_np.shape[1]
            emb_rows = [(batch_ids[i], MODEL_TAG, embed_to_bytes(features_np[i]), dim)
                        for i in range(len(batch_ids))]
            conn2.executemany(
                "INSERT OR REPLACE INTO segment_embeddings (segment_id, model, vector, dim) VALUES (?,?,?,?)",
                emb_rows,
            )
            conn2.commit()
            conn2.close()
            done += len(batch_ids)
        except Exception as e:
            if done < 5:
                print(f"\nBatch error: {e}")
        batch_ids = []
        batch_tensors = []

    with tempfile.TemporaryDirectory(prefix="siglip_vid_") as tmp:
        for seg_id, path, t_start, t_end in rows:
            if not os.path.exists(path):
                failed += 1
                continue
            try:
                midpoint = ((t_start or 0) + (t_end or 0)) / 2
                frame_path = os.path.join(tmp, f"{seg_id}.jpg")
                if extract_frame(path, midpoint, frame_path):
                    img = Image.open(frame_path).convert('RGB')
                    batch_ids.append(seg_id)
                    batch_tensors.append(preprocess_image(img))
                    os.remove(frame_path)
                    if len(batch_tensors) >= args.batch_size:
                        flush()
                else:
                    failed += 1
            except Exception:
                failed += 1

            total_processed = done + failed
            if total_processed % 100 == 0 and total_processed > 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - total_processed) / rate if rate > 0 else 0
                print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} scenes/s | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    flush()
    elapsed = time.time() - start_time
    print(f"\nDone: {done} scenes in {elapsed:.1f}s, {failed} failed")


if __name__ == "__main__":
    main()
