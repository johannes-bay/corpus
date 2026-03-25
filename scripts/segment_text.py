#!/usr/bin/env python3
"""Extract text from documents, chunk it, and embed with CLIP text encoder.

Usage:
    python segment_text.py --db path/to/JB5TB_Corpus.db [--chunk-size 200]

For each document:
  1. Extract text (pdftotext, textutil, or direct read)
  2. Split into chunks (~200 words each)
  3. Embed each chunk with CLIP ViT-B-32 text encoder
  4. Store segments (type='chunk') + segment_embeddings in SQLite

Because CLIP text and image embeddings share the same vector space,
text chunks can be matched against image regions and vice versa.

Resumable: skips documents that already have chunk segments.
"""

import argparse
import hashlib
import sqlite3
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

DOC_EXTENSIONS = {'.pdf', '.txt', '.rtf', '.md', '.docx', '.doc', '.pages', '.epub'}
SEGMENT_MODEL = "text:clip-chunk"
CLIP_MODEL = "clip:ViT-B-32"


def segment_id(path: str, segment_type: str, segment_key: str) -> str:
    raw = f"{path}\0{segment_type}\0{segment_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed_to_bytes(embedding: np.ndarray) -> bytes:
    arr = embedding.astype(np.float32)
    return struct.pack(f'<{len(arr)}f', *arr)


def get_pending_paths(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    exts = ','.join(f"'{e}'" for e in DOC_EXTENSIONS)
    # Only process docs that have been enriched (have word_count)
    query = f"""
        SELECT f.path FROM files f
        JOIN properties p ON p.path = f.path AND p.domain = 'text' AND p.key = 'word_count'
        WHERE f.extension IN ({exts})
        AND p.value_num > 50
        AND f.path NOT IN (
            SELECT DISTINCT s.path FROM segments s WHERE s.model = ?
        )
    """
    paths = [r[0] for r in conn.execute(query, (SEGMENT_MODEL,)).fetchall()]
    conn.close()
    return paths


def save_batch(db_path: str, segments: list, seg_embeddings: list):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        """INSERT OR REPLACE INTO segments
           (id, path, segment_type, segment_key, label, area_frac, model)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        segments,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO segment_embeddings (segment_id, model, vector, dim)
           VALUES (?, ?, ?, ?)""",
        seg_embeddings,
    )
    conn.commit()
    conn.close()


def extract_text(path: str, ext: str) -> str:
    """Extract text from a document file."""
    try:
        if ext == '.pdf':
            result = subprocess.run(
                ['pdftotext', '-l', '10', path, '-'],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        elif ext in ('.txt', '.rtf', '.md'):
            return Path(path).read_text(errors='replace')[:100_000]
        elif ext in ('.docx', '.doc', '.pages', '.epub'):
            result = subprocess.run(
                ['textutil', '-convert', 'txt', '-stdout', path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
    except Exception:
        pass
    return ""


def chunk_text(text: str, chunk_words: int = 200, overlap_words: int = 30) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []

    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_words])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_words - overlap_words

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Segment text with CLIP text encoder")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
    parser.add_argument("--max-chunks", type=int, default=20, help="Max chunks per document")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP text batch size")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CLIP text encoder
    print("Loading CLIP...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device,
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    print("CLIP loaded.")

    pending = get_pending_paths(args.db)
    total = len(pending)
    print(f"Documents to segment: {total}")
    if total == 0:
        print("All documents already segmented.")
        return

    done = 0
    failed = 0
    total_segments = 0
    start_time = time.time()

    # Process in batches — collect chunks across documents, then embed in bulk
    chunk_buffer = []  # (seg_row_tuple, chunk_text)
    EMBED_BATCH = args.batch_size

    def flush_buffer():
        nonlocal chunk_buffer, total_segments
        if not chunk_buffer:
            return

        # Embed all chunks
        texts = [text for _, text in chunk_buffer]
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features_np = features.cpu().numpy()

        seg_rows = []
        seg_emb_rows = []
        for i, (row, _) in enumerate(chunk_buffer):
            seg_rows.append(row)
            vec_bytes = embed_to_bytes(features_np[i])
            dim = features_np.shape[1]
            seg_emb_rows.append((row[0], CLIP_MODEL, vec_bytes, dim))  # row[0] = segment_id

        save_batch(args.db, seg_rows, seg_emb_rows)
        total_segments += len(seg_rows)
        chunk_buffer = []

    for path in pending:
        if not Path(path).exists():
            failed += 1
            continue

        try:
            ext = Path(path).suffix.lower()
            text = extract_text(path, ext)

            if not text or len(text.split()) < 20:
                done += 1
                # Store marker so we don't retry
                sid = segment_id(path, "chunk", "empty")
                save_batch(args.db,
                    [(sid, path, "chunk", "empty", None, None, SEGMENT_MODEL)],
                    [])
                continue

            chunks = chunk_text(text, args.chunk_size)[:args.max_chunks]

            total_words = len(text.split())
            for i, chunk in enumerate(chunks):
                sid = segment_id(path, "chunk", f"chunk_{i}")
                chunk_words = len(chunk.split())
                area_frac = chunk_words / total_words if total_words > 0 else 0

                # Truncate label to first 100 chars for display
                label = chunk[:100].replace('\n', ' ').strip()

                row = (sid, path, "chunk", f"chunk_{i}", label, area_frac, SEGMENT_MODEL)
                chunk_buffer.append((row, chunk))

                if len(chunk_buffer) >= EMBED_BATCH:
                    flush_buffer()

            done += 1

        except Exception as e:
            failed += 1
            if done < 5:
                print(f"\nError on {path}: {e}")

        if done % 50 == 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = total - done - failed
            eta = remaining / rate if rate > 0 else 0
            print(f"\r  {done}/{total} ({done*100//total}%) | {rate:.1f} docs/s | {total_segments} chunks | ETA {eta:.0f}s | {failed} failed", end="", flush=True)

    # Flush remaining
    flush_buffer()

    elapsed = time.time() - start_time
    print(f"\nDone: {done} docs, {total_segments} chunks, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
