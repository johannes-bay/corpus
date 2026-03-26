#!/usr/bin/env python3
"""Re-embed text chunks with SigLIP text encoder via MLX.

Usage:
    python embed_siglip_text_mlx.py --db path/to/JB5TB_Corpus.db

Stores in segment_embeddings with model tag 'siglip:SO400M-384'.
Cross-modal with the SigLIP image embeddings (same vector space).
Resumable.
"""

import argparse
import sqlite3
import struct
import time
import numpy as np
import mlx.core as mx

MODEL_TAG = "siglip:SO400M-384"


def embed_to_bytes(embedding) -> bytes:
    arr = np.array(embedding, dtype=np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Loading SigLIP via MLX...")
    from mlx_embeddings.utils import load
    model, processor = load('mlx-community/siglip-so400m-patch14-384')
    tokenizer = processor.tokenizer
    print("Loaded.")

    conn = sqlite3.connect(args.db)

    # Get text chunks not yet embedded with SigLIP
    rows = conn.execute('''
        SELECT s.id, s.label FROM segments s
        WHERE s.segment_type = 'chunk' AND s.label IS NOT NULL AND LENGTH(s.label) > 20
        AND s.id NOT IN (SELECT segment_id FROM segment_embeddings WHERE model = ?)
    ''', (MODEL_TAG,)).fetchall()
    conn.close()

    total = len(rows)
    print(f"Text chunks to embed: {total}")
    if total == 0:
        print("All done.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch_ids = []
    batch_texts = []

    def flush():
        nonlocal batch_ids, batch_texts, done
        if not batch_texts:
            return
        try:
            # Tokenize
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=64, return_tensors='np')
            input_ids = mx.array(encoded['input_ids'])

            features = model.get_text_features(input_ids)
            norms = mx.sqrt(mx.sum(features * features, axis=-1, keepdims=True))
            features = features / (norms + 1e-12)
            mx.eval(features)
            features_np = np.array(features)

            conn2 = sqlite3.connect(args.db)
            conn2.execute("PRAGMA journal_mode=WAL")
            emb_rows = []
            dim = features_np.shape[1]
            for i, seg_id in enumerate(batch_ids):
                vec_bytes = embed_to_bytes(features_np[i])
                emb_rows.append((seg_id, MODEL_TAG, vec_bytes, dim))
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
        batch_texts = []

    for seg_id, label in rows:
        batch_ids.append(seg_id)
        batch_texts.append(label[:200])  # truncate long chunks
        if len(batch_texts) >= args.batch_size:
            flush()

        total_processed = done + failed
        if total_processed % 500 == 0 and total_processed > 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - total_processed) / rate if rate > 0 else 0
            print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} chunks/s | ETA {eta:.0f}s", end="", flush=True)

    flush()
    elapsed = time.time() - start_time
    print(f"\nDone: {done} chunks in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
