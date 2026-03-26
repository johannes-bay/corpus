#!/usr/bin/env python3
"""Pre-compute embedding neighbor graph for fast concept graph traversal.

Usage:
    python build_neighbors.py --db JB5TB_Corpus.db [--k 20]

For each file with a CLIP or CLAP embedding, finds the top-K most similar
files and stores in the neighbors table. Pure matrix math, no GPU needed.
"""

import argparse
import sqlite3
import struct
import time
import numpy as np


def bytes_to_f32(blob, dim):
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def main():
    parser = argparse.ArgumentParser(description="Build embedding neighbor graph")
    parser.add_argument("--db", required=True)
    parser.add_argument("--k", type=int, default=20, help="Neighbors per file")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size for similarity computation")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    for model in ["clip:ViT-B-32", "clap:HTSAT-tiny"]:
        print(f"\n=== {model} ===")

        # Check existing
        existing = conn.execute(
            "SELECT COUNT(*) FROM neighbors WHERE model = ?", (model,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  Already have {existing} neighbor rows, skipping. Delete to rebuild.")
            continue

        # Load all embeddings for this model
        print("  Loading embeddings...")
        t0 = time.time()
        rows = conn.execute(
            "SELECT path, vector, dim FROM embeddings WHERE model = ?", (model,)
        ).fetchall()
        print(f"  {len(rows)} embeddings loaded in {time.time()-t0:.1f}s")

        if len(rows) < 2:
            print("  Not enough embeddings, skipping.")
            continue

        paths = []
        vecs = []
        for path, blob, dim in rows:
            paths.append(path)
            vecs.append(bytes_to_f32(blob, dim))

        matrix = np.stack(vecs).astype(np.float32)
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1
        matrix = matrix / norms

        N = len(paths)
        K = min(args.k, N - 1)
        print(f"  Computing top-{K} neighbors for {N} files...")

        t0 = time.time()
        neighbor_rows = []
        batch_size = args.batch

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # Batch similarity: (batch_size x dim) @ (dim x N) = (batch_size x N)
            sims = matrix[start:end] @ matrix.T

            for i in range(end - start):
                global_i = start + i
                row_sims = sims[i]
                # Zero out self
                row_sims[global_i] = -1
                # Top K
                top_k = np.argpartition(row_sims, -K)[-K:]
                top_k = top_k[np.argsort(row_sims[top_k])[::-1]]

                for j in top_k:
                    sim = float(row_sims[j])
                    if sim > 0.1:  # skip very weak neighbors
                        neighbor_rows.append((paths[global_i], paths[j], model, sim))

            elapsed = time.time() - t0
            done = end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (N - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{N} ({done*100//N}%) | {rate:.0f} files/s | ETA {eta:.0f}s | {len(neighbor_rows)} edges", end="", flush=True)

        print(f"\n  Writing {len(neighbor_rows)} neighbor edges...")
        t0 = time.time()
        conn.executemany(
            "INSERT OR REPLACE INTO neighbors (path_a, path_b, model, similarity) VALUES (?, ?, ?, ?)",
            neighbor_rows,
        )
        conn.commit()
        print(f"  Written in {time.time()-t0:.1f}s")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
