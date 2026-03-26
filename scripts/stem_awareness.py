#!/usr/bin/env python3
"""Score all CLAP-embedded audio against stem/instrument concepts.

Usage:
    python stem_awareness.py --db path/to/JB5TB_Corpus.db

For each audio file with a CLAP embedding, compute cosine similarity
against text descriptions of common stems/instruments. Stores results
as properties (domain='stems', key='vocals', value_num=0.85, etc.).

This is pure vector math — no audio processing needed. Runs in minutes
on 124K files.
"""

import argparse
import sqlite3
import struct
import time

import numpy as np

CLAP_MODEL = "clap:HTSAT-tiny"

# Stem/instrument concepts — each gets a group of prompts averaged together
STEM_CONCEPTS = {
    "vocals": [
        "singing voice", "vocal melody", "human voice singing",
        "rapper rapping", "spoken word", "choir vocals",
    ],
    "drums": [
        "drum beat", "percussion rhythm", "kick snare hi-hat",
        "drum loop", "drumming",
    ],
    "bass": [
        "bass guitar", "bass line", "deep bass notes",
        "sub bass", "bass synth",
    ],
    "guitar": [
        "electric guitar", "acoustic guitar", "guitar riff",
        "guitar strumming", "guitar solo",
    ],
    "synth": [
        "synthesizer", "electronic synth pad", "synth melody",
        "analog synthesizer", "synth arpeggio",
    ],
    "piano": [
        "piano", "piano melody", "piano chords",
        "keyboard playing", "grand piano",
    ],
    "strings": [
        "violin", "orchestra strings", "string section",
        "cello", "string quartet",
    ],
    "ambient": [
        "ambient texture", "atmospheric sound", "ambient pad",
        "reverb wash", "ambient noise",
    ],
}


def bytes_to_f32(blob, dim):
    return [struct.unpack_from('<f', blob, i * 4)[0] for i in range(dim)]


def cosine_batch(query, matrix):
    q = np.array(query, dtype=np.float32)
    dots = matrix @ q
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    return dots / (norms + 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Score audio against stem concepts via CLAP")
    parser.add_argument("--db", required=True, help="Path to corpus database")
    args = parser.parse_args()

    print("Loading CLAP text encoder...")
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap.load_ckpt()
    print("CLAP loaded.")

    # Embed stem concept prompts
    print("Embedding stem concepts...")
    concept_vecs = {}
    for stem_name, prompts in STEM_CONCEPTS.items():
        embs = clap.get_text_embedding(prompts, use_tensor=False)
        avg = embs.mean(axis=0)
        concept_vecs[stem_name] = avg / np.linalg.norm(avg)
    print(f"Concepts: {list(concept_vecs.keys())}")

    # Load all CLAP audio embeddings
    print("Loading CLAP embeddings...")
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT path, vector, dim FROM embeddings WHERE model = ?",
        (CLAP_MODEL,),
    ).fetchall()
    print(f"Audio files with CLAP: {len(rows)}")

    if not rows:
        print("No CLAP embeddings found.")
        return

    paths = []
    matrix = []
    for path, vec_blob, dim in rows:
        paths.append(path)
        matrix.append(bytes_to_f32(vec_blob, dim))
    matrix = np.array(matrix, dtype=np.float32)

    # Check which files already have stem awareness
    existing = set()
    for row in conn.execute(
        "SELECT DISTINCT path FROM properties WHERE domain = 'stems'"
    ).fetchall():
        existing.add(row[0])

    pending_mask = [p not in existing for p in paths]
    pending_count = sum(pending_mask)
    print(f"Already scored: {len(existing)}, pending: {pending_count}")

    if pending_count == 0:
        print("All files already scored.")
        return

    # Score all files against all concepts
    print("Scoring...")
    t0 = time.time()

    all_scores = {}
    for stem_name, vec in concept_vecs.items():
        scores = cosine_batch(vec, matrix)
        all_scores[stem_name] = scores
        top_idx = np.argsort(scores)[::-1][:3]
        print(f"  {stem_name}: top={scores[top_idx[0]]:.3f} ({paths[top_idx[0]].split('/')[-1]})")

    elapsed = time.time() - t0
    print(f"Scoring done in {elapsed:.1f}s")

    # Save to properties table
    print("Saving to database...")
    conn.execute("PRAGMA journal_mode=WAL")

    batch = []
    saved = 0
    for i, path in enumerate(paths):
        if not pending_mask[i]:
            continue
        for stem_name, scores in all_scores.items():
            score = float(scores[i])
            # Normalize to 0-1 range (CLAP cosine sim is typically 0.1-0.5)
            # Map 0.15-0.45 range to 0-1 for more useful scoring
            normalized = max(0.0, min(1.0, (score - 0.15) / 0.30))
            batch.append((path, 'stems', stem_name, normalized, None))

        if len(batch) >= 5000:
            conn.executemany(
                """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
                   VALUES (?, ?, ?, ?, ?)""",
                batch,
            )
            conn.commit()
            saved += len(batch)
            batch = []
            print(f"\r  {saved} properties saved...", end="", flush=True)

    if batch:
        conn.executemany(
            """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
               VALUES (?, ?, ?, ?, ?)""",
            batch,
        )
        conn.commit()
        saved += len(batch)

    conn.close()
    print(f"\nDone: {saved} properties saved for {pending_count} files")
    print(f"Stems per file: {len(STEM_CONCEPTS)}")


if __name__ == "__main__":
    main()
