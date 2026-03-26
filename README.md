# Corpus

Local-first multimodal creative composition engine.

Corpus takes a concept — a word, a phrase, a file — and composes a constellation of related material from a personal archive. It discovers connections through relational context, not just keyword search or embedding similarity.

## What it does

Given a concept like "empire", Corpus:

1. **Finds anchors** — files with "empire" in their name, metadata, folder path, or document text
2. **Traverses edges** — spreads outward through relational paths: same folder, same album, same artist, same project, embedding similarity
3. **Decomposes** — understands files at the sub-object level: individual regions in images, stems in audio, scenes in video, concepts in text
4. **Composes** — assembles the discovered material into multimodal outputs: visual collages, audio sequences, project folders

## Architecture

Rust workspace with Python ML pipelines. SQLite is the sole data store.

```
corpus-db/          Schema, models, queries
corpus-keyfinder/   Audio key detection
corpus-enrich/      Enrichment pipelines (BPM, color, text analysis, etc.)
corpus-associate/   Scoring axes, concept graph, matcher
corpus-generate/    Output generation (audio sequences, moodboards)
corpus-ui/          Web server + graph visualization UI
corpus-cli/         CLI entry point
scripts/            Python ML pipelines (CLIP, CLAP, SAM2, Demucs, etc.)
```

## The concept graph

The core idea: embedding similarity ("what does this word look like/sound like") is shallow. Real concept matching comes from contextual paths.

A track relates to "empire" not because it *sounds* like the word, but because:
- It was in a project about empire
- Its metadata references it
- It sits next to files about empire
- Documents in the same folder discuss empire
- It shares sonic qualities with tracks that are explicitly about empire

The concept graph discovers these paths through spreading activation across relational edges, with FTS text anchors as starting points.

## Axes

17 scoring dimensions that can be weighted and combined:

| Domain | Axes |
|--------|------|
| Audio | `bpm`, `key`, `spectral`, `sonic`, `vocals` |
| Image | `brightness`, `hue`, `color_temp`, `palette`, `aspect`, `visual`, `objects` |
| Video | `scenes`, `duration` |
| Text | `concepts` |
| Cross-modal | `temporal`, `provenance` |

## Segmentation

Files are decomposed into sub-elements, each individually embedded:

- **Images** — SAM2 automatic mask generation → per-region CLIP embeddings
- **Audio** — Stem awareness scoring (vocals, drums, bass, guitar, synth, piano, strings, ambient). On-demand Demucs separation when composing.
- **Video** — Scene detection → keyframe CLIP embeddings
- **Text** — Chunk extraction → CLIP text embeddings (cross-modal with images)

## Pipeline registry

Every processing step is tracked in a `pipelines` table with model name, version, parameters, and progress. When better models emerge, add a new pipeline — old embeddings persist alongside new ones.

## Running

```bash
cargo build
cargo run --bin corpus-cli -- serve --db your_corpus.db --port 3000
```

Graph visualization at `http://localhost:3000/graph`. Type a concept and watch the graph traverse.

## Processing your own archive

Corpus expects a SQLite database with a `files` table (path, filename, extension, size_bytes, modified_date, parent_folder). The enrichment and embedding pipelines build on top of that.

```bash
# Enrich
cargo run --bin corpus-cli -- enrich --db your_corpus.db --domain audio

# Embed (Python, needs torch + open_clip)
python scripts/embed_clip.py --db your_corpus.db
python scripts/embed_clap.py --db your_corpus.db

# Segment
python scripts/segment_images.py --db your_corpus.db
python scripts/segment_text.py --db your_corpus.db
python scripts/segment_video.py --db your_corpus.db

# Stem awareness (fast, no GPU separation needed)
python scripts/stem_awareness.py --db your_corpus.db
```

All pipelines are resumable and skip already-processed files.

## Requirements

- Rust (nightly, for let-chains)
- Python 3.10+ with: `torch`, `open-clip-torch`, `laion-clap`, `sam2`, `Pillow`
- `ffmpeg`, `ffprobe`, `pdftotext` (poppler)
- macOS (uses `textutil` for document extraction, MPS for GPU acceleration)

## License

MIT
