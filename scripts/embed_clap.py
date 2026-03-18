#!/usr/bin/env python3
"""Generate CLAP embeddings for all audio files in the corpus database.

Usage:
    python embed_clap.py --db path/to/JB5TB_Corpus.db [--batch-size 32]

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
import numpy as np

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aif', '.aiff', '.flac', '.ogg'}
MODEL_TAG = "clap:HTSAT-tiny"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_pending_files(db_path: str) -> list[str]:
    """Get audio file paths that don't yet have an embedding."""
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in AUDIO_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT e.path FROM embeddings e WHERE e.model = ?
        )
    """
    cursor = conn.execute(query, (MODEL_TAG,))
    paths = [row[0] for row in cursor.fetchall()]
    conn.close()
    return paths


def embed_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a 1-D float32 numpy array to little-endian bytes."""
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def save_batch(db_path: str, results: list[tuple[str, bytes, int]]):
    """Save a batch of embeddings to the database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO embeddings (path, model, vector, dim)
           VALUES (?, ?, ?, ?)""",
        [(path, MODEL_TAG, vec_bytes, dim) for path, vec_bytes, dim in results],
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for audio")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: HTSAT-tiny (CLAP)")

    # Load model
    print("Loading CLAP model...")
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    model.load_ckpt()  # downloads default checkpoint if needed
    print("CLAP model loaded.")

    # Get pending files
    pending = get_pending_files(args.db)
    total = len(pending)
    print(f"Files to embed: {total}")

    if total == 0:
        print("All audio already embedded.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch_paths = pending[batch_start:batch_start + batch_size]

        # Filter to files that exist
        valid_paths = [p for p in batch_paths if Path(p).exists()]
        skipped = len(batch_paths) - len(valid_paths)
        failed += skipped

        if not valid_paths:
            continue

        try:
            # CLAP's get_audio_embedding_from_filelist handles loading internally
            embeddings = model.get_audio_embedding_from_filelist(
                x=valid_paths,
                use_tensor=False,  # returns numpy
            )

            results = []
            dim = embeddings.shape[1]
            for i, path in enumerate(valid_paths):
                vec_bytes = embed_to_bytes(embeddings[i])
                results.append((path, vec_bytes, dim))

            save_batch(args.db, results)
            done += len(valid_paths)

        except Exception as e:
            # If whole batch fails, try one by one
            for path in valid_paths:
                try:
                    emb = model.get_audio_embedding_from_filelist(
                        x=[path],
                        use_tensor=False,
                    )
                    dim = emb.shape[1]
                    vec_bytes = embed_to_bytes(emb[0])
                    save_batch(args.db, [(path, vec_bytes, dim)])
                    done += 1
                except Exception:
                    failed += 1

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - done - failed
        eta = remaining / rate if rate > 0 else 0
        print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} files/s | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} embedded, {failed} failed, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
