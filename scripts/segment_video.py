#!/usr/bin/env python3
"""Segment videos into scenes and embed keyframes with CLIP.

Usage:
    python segment_video.py --db path/to/JB5TB_Corpus.db [--threshold 0.4]

For each video:
  1. Detect scene boundaries via ffmpeg scene filter
  2. Extract keyframe (midpoint) from each scene
  3. Embed keyframe with CLIP ViT-B-32
  4. Store segments + segment_embeddings in SQLite

Resumable: skips videos that already have scene segments.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}
SCENE_MODEL = "scene:ffmpeg"
CLIP_MODEL = "clip:ViT-B-32"


def segment_id(path: str, segment_type: str, segment_key: str) -> str:
    raw = f"{path}\0{segment_type}\0{segment_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def get_pending_paths(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in VIDEO_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT DISTINCT s.path FROM segments s WHERE s.model = ?
        )
    """
    paths = [r[0] for r in conn.execute(query, (SCENE_MODEL,)).fetchall()]
    conn.close()
    return paths


def save_batch(db_path: str, segments: list, seg_embeddings: list):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO segments
           (id, path, segment_type, segment_key, label,
            time_start, time_end, confidence, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        segments,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO segment_embeddings (segment_id, model, vector, dim)
           VALUES (?, ?, ?, ?)""",
        seg_embeddings,
    )
    conn.commit()
    conn.close()


def detect_scenes(video_path: str, threshold: float = 0.4) -> list[float]:
    """Detect scene change timestamps using ffmpeg."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0

    if duration <= 0:
        return []

    # Use ffmpeg scene detection
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Parse timestamps from showinfo output
        timestamps = [0.0]  # Start at beginning
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                try:
                    pts = line.split('pts_time:')[1].split()[0]
                    timestamps.append(float(pts))
                except (ValueError, IndexError):
                    pass
        timestamps.append(duration)
        return sorted(set(timestamps))
    except Exception:
        return [0.0, duration] if duration > 0 else []


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at the given timestamp."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1", "-q:v", "2",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Segment videos into scenes + CLIP")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--threshold", type=float, default=0.4, help="Scene detection threshold")
    parser.add_argument("--max-scenes", type=int, default=50, help="Max scenes per video")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CLIP
    print("Loading CLIP...")
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device,
    )
    clip_model.eval()
    print("CLIP loaded.")

    pending = get_pending_paths(args.db)
    total = len(pending)
    print(f"Videos to segment: {total}")
    if total == 0:
        print("All videos already segmented.")
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
            # Detect scenes
            timestamps = detect_scenes(path, args.threshold)
            if len(timestamps) < 2:
                done += 1
                continue

            # Build scene intervals
            scenes = []
            for i in range(len(timestamps) - 1):
                t_start = timestamps[i]
                t_end = timestamps[i + 1]
                if t_end - t_start < 0.5:
                    continue
                midpoint = (t_start + t_end) / 2
                scenes.append((t_start, t_end, midpoint))

            scenes = scenes[:args.max_scenes]
            if not scenes:
                done += 1
                continue

            # Extract keyframes and embed
            seg_rows = []
            seg_emb_rows = []
            crops = []

            with tempfile.TemporaryDirectory(prefix="scene_") as tmp_dir:
                for i, (t_start, t_end, midpoint) in enumerate(scenes):
                    seg_key = f"scene_{i:03d}"
                    sid = segment_id(path, "scene", seg_key)
                    frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")

                    if extract_frame(path, midpoint, frame_path):
                        try:
                            img = Image.open(frame_path).convert("RGB")
                            crops.append((sid, img, t_start, t_end))
                        except Exception:
                            pass

                # Batch CLIP embed
                if crops:
                    tensors = torch.stack([clip_preprocess(c) for _, c, _, _ in crops]).to(device)
                    with torch.no_grad():
                        features = clip_model.encode_image(tensors)
                        features = features / features.norm(dim=-1, keepdim=True)
                        features_np = features.cpu().numpy()

                    for j, (sid, _, t_start, t_end) in enumerate(crops):
                        seg_key = f"scene_{j:03d}"
                        confidence = 1.0 / len(crops)  # equal weight per scene

                        seg_rows.append((
                            sid, path, "scene", seg_key, f"scene {j}",
                            t_start, t_end, confidence, SCENE_MODEL,
                        ))

                        vec_bytes = embed_to_bytes(features_np[j])
                        seg_emb_rows.append((sid, CLIP_MODEL, vec_bytes, features_np.shape[1]))

            if seg_rows:
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
        print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.2f} vid/s | {total_segments} scenes | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} videos, {total_segments} scenes, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
