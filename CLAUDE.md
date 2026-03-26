# Corpus

Local-first multimodal creative composition engine. Takes a concept or seed artifact and composes a constellation of related material from a personal archive.

## What it is

Corpus indexes, segments, and semantically understands 542,403 files (2.3TB) across audio, images, video, text, and fonts. Given a concept like "empire" or a seed file, it discovers related material through a concept graph (relational edges + semantic embeddings) and composes new multimodal outputs.

## Architecture

Rust workspace with 7 crates + Python ML scripts:

```
corpus-db/        SQLite schema, models, queries (sole data store, WAL mode)
corpus-keyfinder/ Audio key detection via Symphonia
corpus-enrich/    Enrichment pipelines (audio, image, text, video analysis)
corpus-associate/ Scoring axes, concept graph, matcher
corpus-generate/  Output generation (audio sequences, moodboards, projects)
corpus-ui/        Axum web server, REST API, WebSocket, graph UI
corpus-cli/       CLI entry point (serve, enrich, compose, concept)
scripts/          Python ML pipelines (CLIP, CLAP, SAM2, Demucs, etc.)
```

## Key concepts

- **Axes**: Scoring dimensions (bpm, key, visual, sonic, objects, concepts, etc.). 17 total.
- **Segments**: Sub-file decomposition. SAM2 regions for images, Demucs stems for audio, scenes for video, text chunks for documents.
- **Concept graph**: Spreading activation from text query → FTS anchors → edge traversal (co-folder, co-album, co-artist, co-project, embedding neighbors).
- **Embeddings**: CLIP (ViT-B-32 + ViT-L-14) for images/text, CLAP for audio. Stored as blobs in SQLite.
- **Contrastive prompts**: Positive minus negative prompt embeddings for sharper concept retrieval.

## Data state

- 302K audio: BPM, key, spectral, duration, CLAP embeddings (124K), stem awareness (8 instrument scores per file)
- 97K images: brightness, color, CLIP embeddings, 687K SAM2 segments with CLIP embeddings (ViT-L-14 re-embedding in progress)
- 16K documents: text extracted, 88K chunks with CLIP text embeddings
- 3.7K videos: 16.5K scene segments with CLIP keyframe embeddings
- 1.8M FTS index rows across filenames, folders, metadata, text content

## How the vision evolved

**Phase 1 — File-seeded axis matching**: Seed a file, find similar files via weighted scoring axes (BPM proximity, key compatibility, spectral similarity). Audio-first.

**Phase 2 — Cross-modal enrichment**: Extended to images (brightness, color, aspect), text (word count, language), video (duration, resolution). Added domain-agnostic property system.

**Phase 3 — Whole-file embeddings**: CLIP for images, CLAP for audio. Cosine similarity axes ("visual", "sonic"). File-to-file semantic matching.

**Phase 4 — Sub-file segmentation**: SAM2 for image regions, text chunking, video scene detection. The insight: "find images containing similar objects" not just "similar images." Per-segment embeddings enable object-level matching.

**Phase 5 — Concept graph**: The shift from file-seeded to concept-seeded composition. Type a word, spreading activation discovers related material through relational edges (same folder, same album, same artist) not just embedding similarity. FTS anchors + edge traversal + convergence scoring.

**Phase 6 — Intrinsic understanding** (current): The realization that text→embedding matching is shallow ("what does the word look like/sound like"). Real concept matching comes from contextual/relational paths. Stem awareness (scoring existing CLAP embeddings against instrument concepts) instead of pre-separating all audio. On-demand Demucs separation only when composing.

## Running

```bash
# Build
cargo build

# Serve web UI
cargo run --bin corpus-cli -- serve --db JB5TB_Corpus.db --port 3000

# Graph UI at http://localhost:3000/graph
# Compose UI at http://localhost:3000

# Enrichment
cargo run --bin corpus-cli -- enrich --db JB5TB_Corpus.db --domain audio

# Python pipelines (in .venv)
python scripts/embed_clip.py --db JB5TB_Corpus.db
python scripts/segment_images.py --db JB5TB_Corpus.db
python scripts/stem_awareness.py --db JB5TB_Corpus.db
python scripts/segment_text.py --db JB5TB_Corpus.db
```

## Remote compute

The M1 Max 64GB MacBook (`freelancedmy@macbookpro.ip` / `192.168.1.251`) runs heavy ML jobs. Scripts + DB are at `~/corpus/`. Crop server at port 4000 serves segment previews.

## Design principles

- SQLite is the sole data store. No external vector DB.
- Read-only on source files. Never modify/move/delete originals.
- Resumable pipelines. All scripts skip already-processed files.
- Embeddings stored as little-endian f32 blobs.
- Additive migrations only.
