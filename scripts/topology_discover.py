#!/usr/bin/env python3
"""Discover natural topologies in the archive's folder structure.

Computes a rich feature vector per folder and clusters them to reveal
how files accumulate — creative sessions, dumps, sample libraries,
system artifacts, research phases, etc.

Usage:
    python topology_discover.py --db path/to/JB5TB_Corpus.db
"""

import argparse
import sqlite3
import json
import re
import math
from collections import Counter, defaultdict
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--min-files", type=int, default=3, help="Min files per folder to analyze")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Load all files grouped by folder
    print("Loading files...")
    folders = defaultdict(list)
    for row in conn.execute(
        "SELECT path, filename, extension, size_bytes, modified_date, parent_folder FROM files"
    ):
        folders[row[5]].append({
            "path": row[0], "filename": row[1], "extension": row[2],
            "size": row[3], "modified": row[4], "folder": row[5],
        })

    print(f"{len(folders)} folders, analyzing {sum(1 for f in folders.values() if len(f) >= args.min_files)} with {args.min_files}+ files")

    # Compute feature vectors
    profiles = []
    for folder, files in folders.items():
        if len(files) < args.min_files:
            continue

        profile = compute_folder_profile(folder, files)
        profiles.append(profile)

    # Classify into topology types using heuristic rules
    topology_counts = Counter()
    for p in profiles:
        topo = classify_topology(p)
        p["topology"] = topo
        topology_counts[topo] += 1

    print(f"\n=== Topology Distribution ({len(profiles)} folders) ===\n")
    for topo, count in topology_counts.most_common():
        pct = count * 100 / len(profiles)
        print(f"  {topo:30} {count:5} ({pct:.1f}%)")

    # Show examples of each topology
    print(f"\n=== Examples ===\n")
    shown = set()
    for topo, _ in topology_counts.most_common():
        examples = [p for p in profiles if p["topology"] == topo]
        examples.sort(key=lambda p: p["file_count"], reverse=True)
        print(f"--- {topo} ---")
        for p in examples[:3]:
            short = '/'.join(p["folder"].split('/')[-3:])
            print(f"  .../{short}")
            print(f"    {p['file_count']} files, {p['ext_diversity']} types, span {p['temporal_span_days']:.0f} days")
            print(f"    Top exts: {p['top_extensions']}")
            print(f"    Signals: naming={p['naming_pattern']}, burst={p['is_burst']}, sequential={p['sequential_ratio']:.2f}")
            print()
        shown.add(topo)

    # Save to DB
    print("Saving topology profiles to properties...")
    conn.execute("PRAGMA journal_mode=WAL")
    saved = 0
    for p in profiles:
        # Store as folder-level properties using the folder path as key
        # Use a representative file from the folder
        for f in folders[p["folder"]][:1]:
            conn.execute(
                """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
                   VALUES (?, 'topology', 'type', NULL, ?)""",
                (f["path"], p["topology"])
            )
            conn.execute(
                """INSERT OR REPLACE INTO properties (path, domain, key, value_num, value_txt)
                   VALUES (?, 'topology', 'profile', NULL, ?)""",
                (f["path"], json.dumps({
                    "folder": p["folder"],
                    "file_count": p["file_count"],
                    "topology": p["topology"],
                    "ext_diversity": p["ext_diversity"],
                    "temporal_span_days": p["temporal_span_days"],
                    "naming_pattern": p["naming_pattern"],
                    "depth": p["depth"],
                }))
            )
            saved += 1

    conn.commit()
    conn.close()
    print(f"Saved {saved} topology profiles")


def compute_folder_profile(folder, files):
    """Compute a rich feature vector for a folder."""
    n = len(files)
    parts = [p for p in folder.split('/') if p]

    # Extension diversity
    exts = Counter(f["extension"] for f in files)
    ext_diversity = len(exts)
    top_exts = ', '.join(f"{e}:{c}" for e, c in exts.most_common(3))
    dominant_ext_ratio = exts.most_common(1)[0][1] / n if exts else 0
    is_mono_type = dominant_ext_ratio > 0.9

    # Temporal analysis
    dates = []
    for f in files:
        if f["modified"]:
            try:
                dt = datetime.strptime(f["modified"][:19], "%Y-%m-%d %H:%M:%S")
                dates.append(dt)
            except (ValueError, TypeError):
                pass

    if len(dates) >= 2:
        dates.sort()
        span = (dates[-1] - dates[0]).total_seconds()
        temporal_span_days = span / 86400
        # Burst detection: most files created within a short window
        # Check if 80% of files are within 1 hour
        if len(dates) >= 5:
            windows = []
            for i in range(len(dates) - int(len(dates) * 0.8)):
                window = (dates[i + int(len(dates) * 0.8) - 1] - dates[i]).total_seconds()
                windows.append(window)
            min_window = min(windows) if windows else span
            is_burst = min_window < 3600  # 80% within 1 hour
        else:
            is_burst = span < 3600
    else:
        temporal_span_days = 0
        is_burst = False

    # Naming patterns
    filenames = [f["filename"] for f in files]

    # Sequential naming (IMG_0001, DM-SNA 0068, etc)
    sequential_pattern = re.compile(r'[\d]{3,}')
    sequential_count = sum(1 for fn in filenames if sequential_pattern.search(fn))
    sequential_ratio = sequential_count / n

    # UUID naming (system generated)
    uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}', re.I)
    uuid_count = sum(1 for fn in filenames if uuid_pattern.search(fn))
    uuid_ratio = uuid_count / n

    # Common prefix (batch export, same tool)
    if n >= 3:
        prefix_len = 0
        ref = filenames[0]
        for i in range(min(20, len(ref))):
            if all(fn[i:i+1] == ref[i:i+1] for fn in filenames[:min(20, n)] if len(fn) > i):
                prefix_len = i + 1
            else:
                break
        common_prefix_ratio = prefix_len / max(len(ref), 1)
    else:
        common_prefix_ratio = 0

    # Detect naming pattern
    if uuid_ratio > 0.5:
        naming_pattern = "system-generated"
    elif sequential_ratio > 0.7 and common_prefix_ratio > 0.3:
        naming_pattern = "sequential-batch"
    elif sequential_ratio > 0.5:
        naming_pattern = "sequential"
    elif common_prefix_ratio > 0.5:
        naming_pattern = "common-prefix"
    else:
        naming_pattern = "mixed"

    # Size analysis
    sizes = [f["size"] for f in files if f["size"]]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    size_variance = (sum((s - avg_size)**2 for s in sizes) / len(sizes))**0.5 if len(sizes) > 1 else 0
    size_uniformity = 1 - min(size_variance / (avg_size + 1), 1)

    # Path signals
    depth = len(parts)
    path_lower = folder.lower()

    has_work_signal = any(w in path_lower for w in ['work', 'project', 'job', 'client', 'freelance'])
    has_personal_signal = any(w in path_lower for w in ['personal', 'photo', 'camera', 'vacation', 'trip'])
    has_system_signal = any(w in path_lower for w in ['library', 'cache', 'derivatives', 'resources', '.app', 'journal'])
    has_sample_signal = any(w in path_lower for w in ['sample', 'loop', 'preset', 'patch', 'drum', 'synth'])
    has_reference_signal = any(w in path_lower for w in ['reference', 'ref', 'inspiration', 'moodboard'])
    has_download_signal = any(w in path_lower for w in ['download', 'inbox', 'desktop'])
    has_archive_signal = any(w in path_lower for w in ['archive', 'backup', 'old', 'legacy'])
    has_export_signal = any(w in path_lower for w in ['export', 'render', 'output', 'bounced', 'bounce', 'final'])

    # Media type composition
    audio_exts = {'.wav', '.mp3', '.m4a', '.aif', '.aiff', '.flac', '.ogg'}
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.tif', '.tiff', '.bmp'}
    doc_exts = {'.pdf', '.txt', '.docx', '.doc', '.md', '.rtf'}
    video_exts = {'.mp4', '.mov', '.avi', '.mkv'}
    design_exts = {'.psd', '.ai', '.indd', '.sketch', '.fig', '.xd'}
    code_exts = {'.js', '.py', '.rs', '.html', '.css', '.json', '.yaml', '.toml'}
    working_exts = {'.asd', '.als', '.logicx', '.ptx'}  # DAW project files

    media_mix = {
        "audio": sum(1 for f in files if f["extension"] in audio_exts) / n,
        "image": sum(1 for f in files if f["extension"] in image_exts) / n,
        "document": sum(1 for f in files if f["extension"] in doc_exts) / n,
        "video": sum(1 for f in files if f["extension"] in video_exts) / n,
        "design": sum(1 for f in files if f["extension"] in design_exts) / n,
        "code": sum(1 for f in files if f["extension"] in code_exts) / n,
        "working": sum(1 for f in files if f["extension"] in working_exts) / n,
    }
    dominant_media = max(media_mix, key=media_mix.get)

    return {
        "folder": folder,
        "file_count": n,
        "depth": depth,
        "ext_diversity": ext_diversity,
        "top_extensions": top_exts,
        "dominant_ext_ratio": dominant_ext_ratio,
        "is_mono_type": is_mono_type,
        "temporal_span_days": temporal_span_days,
        "is_burst": is_burst,
        "naming_pattern": naming_pattern,
        "sequential_ratio": sequential_ratio,
        "uuid_ratio": uuid_ratio,
        "common_prefix_ratio": common_prefix_ratio,
        "avg_size": avg_size,
        "size_uniformity": size_uniformity,
        "has_work_signal": has_work_signal,
        "has_personal_signal": has_personal_signal,
        "has_system_signal": has_system_signal,
        "has_sample_signal": has_sample_signal,
        "has_reference_signal": has_reference_signal,
        "has_download_signal": has_download_signal,
        "has_archive_signal": has_archive_signal,
        "has_export_signal": has_export_signal,
        "media_mix": media_mix,
        "dominant_media": dominant_media,
    }


def classify_topology(p):
    """Heuristic classification of folder topology."""

    # System artifacts
    if p["has_system_signal"] or p["uuid_ratio"] > 0.7:
        return "system-artifact"

    # Batch export (burst of mono-type files)
    if p["is_burst"] and p["is_mono_type"] and p["file_count"] > 10:
        return "batch-export"

    # Sample library (lots of uniform audio, sequential naming)
    if p["has_sample_signal"] and p["media_mix"]["audio"] > 0.8:
        return "sample-library"
    if p["is_mono_type"] and p["media_mix"]["audio"] > 0.9 and p["sequential_ratio"] > 0.5 and p["file_count"] > 20:
        return "sample-library"

    # Camera dump (sequential images, burst or short span)
    if p["media_mix"]["image"] > 0.8 and p["sequential_ratio"] > 0.5 and p["naming_pattern"] in ("sequential", "sequential-batch"):
        return "camera-dump"

    # Reference/moodboard (images, mixed naming)
    if p["has_reference_signal"]:
        return "reference-collection"
    if p["media_mix"]["image"] > 0.7 and p["naming_pattern"] == "mixed" and not p["sequential_ratio"] > 0.5:
        return "curated-images"

    # Working project (mixed types including DAW/design files)
    if p["media_mix"]["working"] > 0.05:
        return "production-session"
    if p["has_work_signal"] and p["ext_diversity"] >= 3:
        return "work-project"

    # Export/render folder
    if p["has_export_signal"]:
        return "export-output"

    # Archive
    if p["has_archive_signal"]:
        return "archive"

    # Download accumulation
    if p["has_download_signal"]:
        return "download-accumulation"

    # Academic/research (documents)
    if p["media_mix"]["document"] > 0.5 and p["file_count"] > 3:
        return "document-collection"

    # Music collection (audio, mixed naming, not samples)
    if p["media_mix"]["audio"] > 0.7 and p["naming_pattern"] == "mixed":
        return "music-collection"
    if p["media_mix"]["audio"] > 0.7 and p["sequential_ratio"] > 0.3:
        return "album-rip"

    # Personal photos (images with date-based folders)
    if p["has_personal_signal"] and p["media_mix"]["image"] > 0.5:
        return "personal-photos"

    # Video project
    if p["media_mix"]["video"] > 0.3 and p["ext_diversity"] >= 2:
        return "video-project"

    # Code/config
    if p["media_mix"]["code"] > 0.3:
        return "code-project"

    # Design project
    if p["media_mix"]["design"] > 0.1:
        return "design-project"

    # Mixed project (diverse types, some structure)
    if p["ext_diversity"] >= 4:
        return "mixed-project"

    # Single-type collection
    if p["is_mono_type"]:
        return "single-type-collection"

    return "uncategorized"


if __name__ == "__main__":
    main()
