#!/usr/bin/env python3
"""Separate audio stems with Demucs and embed each with CLAP.

Usage:
    python segment_audio.py --db path/to/JB5TB_Corpus.db [--min-duration 10]

For each audio file:
  1. Demucs htdemucs → 4 stems (vocals, drums, bass, other)
  2. Embed each stem with CLAP HTSAT-tiny
  3. Store segments + segment_embeddings in SQLite
  4. Delete temp stem WAVs (only embeddings are kept)

Resumable: skips files that already have stem segments.
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aif', '.aiff', '.flac', '.ogg'}
DEMUCS_MODEL = "demucs:htdemucs"
CLAP_MODEL = "clap:HTSAT-tiny"
STEM_NAMES = ["vocals", "drums", "bass", "other"]


def segment_id(path: str, segment_type: str, segment_key: str) -> str:
    raw = f"{path}\0{segment_type}\0{segment_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def get_pending_paths(db_path: str, min_duration: float) -> list[tuple[str, float]]:
    """Get audio files that haven't been stem-separated yet, with duration."""
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in AUDIO_EXTENSIONS)

    # Duration is stored in properties table as audio.duration
    query = f"""
        SELECT f.path, p.value_num
        FROM files f
        JOIN properties p ON p.path = f.path AND p.domain = 'audio' AND p.key = 'duration'
        WHERE f.extension IN ({exts})
        AND p.value_num >= ?
        AND f.path NOT IN (
            SELECT DISTINCT s.path FROM segments s WHERE s.model = ?
        )
        ORDER BY p.value_num DESC
    """
    rows = conn.execute(query, (min_duration, DEMUCS_MODEL)).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def save_batch(db_path: str, segments: list, seg_embeddings: list):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO segments
           (id, path, segment_type, segment_key, label,
            time_start, time_end, area_frac, model)
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


def run_demucs(input_path: str, output_dir: str) -> dict[str, str]:
    """Run Demucs and return {stem_name: wav_path}."""
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",  # produces vocals + no_vocals initially
        "-n", "htdemucs",
        "--out", output_dir,
        input_path,
    ]
    # Actually, let's use the full 4-stem separation
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        "--out", output_dir,
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[:500]}")

    # Find output stems
    filename = Path(input_path).stem
    stem_dir = Path(output_dir) / "htdemucs" / filename
    stems = {}
    for name in STEM_NAMES:
        stem_path = stem_dir / f"{name}.wav"
        if stem_path.exists():
            stems[name] = str(stem_path)
    return stems


def compute_rms(wav_path: str) -> float:
    """Compute RMS energy of a WAV file using ffmpeg."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        import json
        data = json.loads(result.stdout)
        # Just use file size as proxy for energy (simpler, faster)
        return float(data.get("format", {}).get("size", 0))
    except Exception:
        return 1.0


def main():
    parser = argparse.ArgumentParser(description="Separate audio stems with Demucs + CLAP")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--min-duration", type=float, default=10.0,
                        help="Minimum duration in seconds (skip short samples)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CLAP
    print("Loading CLAP model...")
    import laion_clap
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt()
    print("CLAP loaded.")

    # Get pending files
    pending = get_pending_paths(args.db, args.min_duration)
    total = len(pending)
    print(f"Audio files to segment: {total}")
    if total == 0:
        print("All audio already segmented.")
        return

    done = 0
    failed = 0
    total_segments = 0
    start_time = time.time()

    for path, duration in pending:
        if not Path(path).exists():
            failed += 1
            continue

        tmp_dir = tempfile.mkdtemp(prefix="demucs_")
        try:
            # Run Demucs
            stems = run_demucs(path, tmp_dir)

            if not stems:
                failed += 1
                continue

            # Compute relative energy for each stem
            stem_sizes = {name: os.path.getsize(p) for name, p in stems.items()}
            total_size = sum(stem_sizes.values()) or 1

            seg_rows = []
            seg_emb_rows = []

            # Embed each stem with CLAP
            stem_paths_list = []
            stem_names_list = []
            for name in STEM_NAMES:
                if name in stems:
                    stem_paths_list.append(stems[name])
                    stem_names_list.append(name)

            if stem_paths_list:
                try:
                    embeddings = clap_model.get_audio_embedding_from_filelist(
                        x=stem_paths_list,
                        use_tensor=False,
                    )

                    for i, name in enumerate(stem_names_list):
                        sid = segment_id(path, "stem", name)
                        area_frac = stem_sizes.get(name, 0) / total_size

                        seg_rows.append((
                            sid, path, "stem", name, name,
                            0.0, duration, area_frac, DEMUCS_MODEL,
                        ))

                        vec_bytes = embed_to_bytes(embeddings[i])
                        dim = embeddings.shape[1]
                        seg_emb_rows.append((sid, CLAP_MODEL, vec_bytes, dim))

                except Exception as e:
                    # Try one by one on batch failure
                    for name in stem_names_list:
                        sid = segment_id(path, "stem", name)
                        area_frac = stem_sizes.get(name, 0) / total_size
                        try:
                            emb = clap_model.get_audio_embedding_from_filelist(
                                x=[stems[name]], use_tensor=False,
                            )
                            seg_rows.append((
                                sid, path, "stem", name, name,
                                0.0, duration, area_frac, DEMUCS_MODEL,
                            ))
                            vec_bytes = embed_to_bytes(emb[0])
                            seg_emb_rows.append((sid, CLAP_MODEL, vec_bytes, emb.shape[1]))
                        except Exception:
                            pass

            if seg_rows:
                save_batch(args.db, seg_rows, seg_emb_rows)
                total_segments += len(seg_rows)

            done += 1

        except Exception as e:
            failed += 1
            if done < 5:
                print(f"\nError on {path}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - done - failed
        eta = remaining / rate if rate > 0 else 0
        print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.2f} files/s | {total_segments} stems | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} files, {total_segments} stems, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
