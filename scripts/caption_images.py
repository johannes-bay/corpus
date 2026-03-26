#!/usr/bin/env python3
"""Generate natural language captions for all images using Qwen2.5-VL-7B via MLX.

Usage:
    python caption_images.py --db path/to/JB5TB_Corpus.db [--max-tokens 100]

Stores captions as properties (domain='caption', key='description').
Also indexes in FTS for concept graph discovery.
Resumable: skips images that already have a caption.
"""

import argparse
import sqlite3
import time
import os
from pathlib import Path

import mlx.core as mx

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
PROMPT = "Describe this image in one detailed sentence. Focus on the main subjects, objects, setting, and activity."


def get_pending_paths(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in IMAGE_EXTENSIONS)
    query = f"""
        SELECT f.path FROM files f
        WHERE f.extension IN ({exts})
        AND f.path NOT IN (
            SELECT path FROM properties WHERE domain = 'caption' AND key = 'description'
        )
        ORDER BY f.size_bytes DESC
    """
    paths = [r[0] for r in conn.execute(query).fetchall()]
    conn.close()
    return paths


def save_caption(db_path: str, path: str, caption: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
           VALUES (?, 'caption', 'description', NULL, ?)""",
        (path, caption),
    )
    # Also add to FTS index for concept graph discovery
    conn.execute(
        """INSERT INTO corpus_fts (path, source_type, source_key, content)
           VALUES (?, 'caption', 'caption', ?)""",
        (path, caption),
    )
    conn.commit()
    conn.close()


def save_batch_captions(db_path: str, results: list[tuple[str, str]]):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
           VALUES (?, 'caption', 'description', NULL, ?)""",
        results,
    )
    conn.executemany(
        """INSERT INTO corpus_fts (path, source_type, source_key, content)
           VALUES (?, 'caption', 'caption', ?)""",
        results,
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Caption images with Qwen2.5-VL-7B")
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--batch-save", type=int, default=10, help="Save to DB every N images")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading VLM...")
    from mlx_vlm import load, generate
    t0 = time.time()
    model, processor = load(args.model)
    print(f"Loaded in {time.time()-t0:.1f}s")

    pending = get_pending_paths(args.db)
    total = len(pending)
    print(f"Images to caption: {total}")
    if total == 0:
        print("All images already captioned.")
        return

    done = 0
    failed = 0
    start_time = time.time()
    batch = []

    for path in pending:
        if not Path(path).exists():
            failed += 1
            continue

        try:
            result = generate(
                model, processor, PROMPT, [path],
                max_tokens=args.max_tokens, verbose=False, temperature=0.1,
            )
            mx.clear_cache()

            caption = str(result).strip()
            if not caption or len(caption) < 10:
                failed += 1
                continue

            batch.append((path, caption))
            done += 1

            if len(batch) >= args.batch_save:
                save_batch_captions(args.db, batch)
                batch = []

        except Exception as e:
            failed += 1
            if done < 5:
                print(f"\nError on {path}: {e}")

        total_processed = done + failed
        if total_processed % 50 == 0 and total_processed > 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - total_processed
            eta = remaining / rate if rate > 0 else 0
            eta_h = eta / 3600
            print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.2f} img/s | ETA {eta_h:.1f}h | {failed} failed", end="", flush=True)

    # Final flush
    if batch:
        save_batch_captions(args.db, batch)

    elapsed = time.time() - start_time
    print(f"\nDone: {done} captioned, {failed} failed, {elapsed:.1f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
