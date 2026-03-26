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

    # ---- Cross-modal neighbors: text chunks → images ----
    # Both use CLIP embeddings in the same vector space.
    # For each text chunk, find the top-K most similar images.

    cross_model = "clip:ViT-B-32"  # shared vector space
    cross_tag = "cross:text→image"

    existing_cross = conn.execute(
        "SELECT COUNT(*) FROM neighbors WHERE model = ?", (cross_tag,)
    ).fetchone()[0]

    if existing_cross > 0:
        print(f"\n=== Cross-modal ({cross_tag}) ===")
        print(f"  Already have {existing_cross} edges, skipping.")
    else:
        print(f"\n=== Cross-modal: text chunks → images ({cross_model}) ===")

        # Load image embeddings
        print("  Loading image embeddings...")
        img_rows = conn.execute('''
            SELECT e.path, e.vector, e.dim FROM embeddings e
            JOIN files f ON f.path = e.path
            WHERE e.model = ? AND f.extension IN ('.jpg','.jpeg','.png','.gif','.webp','.tif','.tiff','.bmp')
        ''', (cross_model,)).fetchall()

        img_paths = []
        img_vecs = []
        for path, blob, dim in img_rows:
            img_paths.append(path)
            img_vecs.append(bytes_to_f32(blob, dim))

        if not img_vecs:
            print("  No image embeddings found, skipping.")
        else:
            img_matrix = np.stack(img_vecs).astype(np.float32)
            img_norms = np.linalg.norm(img_matrix, axis=1, keepdims=True)
            img_norms[img_norms < 1e-12] = 1
            img_matrix = img_matrix / img_norms
            print(f"  {len(img_paths)} images")

            # Load text chunk embeddings
            print("  Loading text chunk embeddings...")
            txt_rows = conn.execute('''
                SELECT s.path, se.vector, se.dim FROM segment_embeddings se
                JOIN segments s ON s.id = se.segment_id
                WHERE s.segment_type = 'chunk' AND se.model = ?
            ''', (cross_model,)).fetchall()

            txt_paths = []
            txt_vecs = []
            for path, blob, dim in txt_rows:
                txt_paths.append(path)
                txt_vecs.append(bytes_to_f32(blob, dim))

            if not txt_vecs:
                print("  No text chunk embeddings found, skipping.")
            else:
                txt_matrix = np.stack(txt_vecs).astype(np.float32)
                txt_norms = np.linalg.norm(txt_matrix, axis=1, keepdims=True)
                txt_norms[txt_norms < 1e-12] = 1
                txt_matrix = txt_matrix / txt_norms
                print(f"  {len(txt_paths)} text chunks")

                # For each text chunk's parent file, find top-K similar images
                # Group chunks by parent file first to avoid duplicates
                file_vecs = {}  # path -> averaged embedding
                for i, path in enumerate(txt_paths):
                    if path not in file_vecs:
                        file_vecs[path] = []
                    file_vecs[path].append(txt_vecs[i])

                # Average chunks per file
                file_paths = []
                file_avg_vecs = []
                for path, vecs_list in file_vecs.items():
                    avg = np.mean(vecs_list, axis=0)
                    avg = avg / (np.linalg.norm(avg) + 1e-12)
                    file_paths.append(path)
                    file_avg_vecs.append(avg)

                file_matrix = np.stack(file_avg_vecs).astype(np.float32)
                print(f"  {len(file_paths)} unique text files")

                K_cross = min(args.k, len(img_paths) - 1)
                print(f"  Computing top-{K_cross} image neighbors for each text file...")

                t0 = time.time()
                cross_rows = []
                batch_size = args.batch
                N_txt = len(file_paths)

                for start in range(0, N_txt, batch_size):
                    end = min(start + batch_size, N_txt)
                    # (batch x dim) @ (dim x N_img) = (batch x N_img)
                    sims = file_matrix[start:end] @ img_matrix.T

                    for i in range(end - start):
                        row_sims = sims[i]
                        top_k = np.argpartition(row_sims, -K_cross)[-K_cross:]
                        top_k = top_k[np.argsort(row_sims[top_k])[::-1]]

                        txt_path = file_paths[start + i]
                        for j in top_k:
                            sim = float(row_sims[j])
                            if sim > 0.15:
                                # text file → image (cross-modal)
                                cross_rows.append((txt_path, img_paths[j], cross_tag, sim))
                                # image → text file (reverse)
                                cross_rows.append((img_paths[j], txt_path, cross_tag, sim))

                    elapsed = time.time() - t0
                    done = end
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (N_txt - done) / rate if rate > 0 else 0
                    print(f"\r  {done}/{N_txt} ({done*100//N_txt}%) | {rate:.0f} files/s | ETA {eta:.0f}s | {len(cross_rows)} edges", end="", flush=True)

                print(f"\n  Writing {len(cross_rows)} cross-modal edges...")
                t0 = time.time()
                conn.executemany(
                    "INSERT OR REPLACE INTO neighbors (path_a, path_b, model, similarity) VALUES (?, ?, ?, ?)",
                    cross_rows,
                )
                conn.commit()
                print(f"  Written in {time.time()-t0:.1f}s")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
