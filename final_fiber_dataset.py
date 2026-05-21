#!/usr/bin/env python3
"""
Final 15-fiber recognition dataset utilities.

Discovers challenge labels from video filename stems (not hard-coded A-Z).
Supports domain/fiber layout under a configurable data root.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2

from unified_dataset import _fast_frame_count

VIDEO_EXTS = ("*.avi", "*.AVI", "*.mp4", "*.MP4", "*.mkv", "*.mov", "*.MOV")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}


def _list_video_files(fiber_path: str) -> List[str]:
    """List video files only; ignore dotfiles and non-video names."""
    found: set = set()
    for ext in VIDEO_EXTS:
        for f in glob.glob(os.path.join(fiber_path, ext)):
            base = os.path.basename(f)
            if base.startswith("."):
                continue
            if os.path.splitext(base)[1].lower() not in VIDEO_SUFFIXES:
                continue
            found.add(os.path.normpath(f))
    return sorted(found)

DEFAULT_DOMAINS = ["GreenAndRed", "RedChange"]

# Fixed challenge vocabulary for final 15-fiber training (do not add typos as separate classes).
CANONICAL_LABELS = ["1", "2", "3", "a", "b", "c", "boy", "girl"]

# Map known filename typos to canonical labels without renaming raw videos.
FILENAME_LABEL_ALIASES = {
    "gril": "girl",
}

DOMAIN_SLUGS = {
    "GreenAndRed": "red_green_fixed",
    "RedChange": "red_green_dynamic",
}


def resolve_data_root(data_root: str, project_root: Optional[str] = None) -> str:
    """
    Resolve dataset root when user passes '(5.20)' or 'recognition_dataset'.

    Tries, in order:
      1. data_root if it already contains domain folders
      2. project_root/recognition_dataset/data_root
      3. project_root/recognition_dataset
      4. project_root/data_root
    """
    data_root = data_root.strip()
    candidates = [os.path.abspath(data_root)]
    if project_root:
        candidates.extend([
            os.path.join(project_root, "recognition_dataset", data_root),
            os.path.join(project_root, "recognition_dataset"),
            os.path.join(project_root, data_root),
        ])
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if _has_domain_folders(path):
            return path
    return candidates[0]


def _has_domain_folders(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in DEFAULT_DOMAINS:
        if os.path.isdir(os.path.join(path, name)):
            return True
    return False


def extract_label_from_filename(filename: str) -> Optional[str]:
    """
    Infer class label from video filename stem.

    Examples: 1.avi -> '1', a.avi -> 'a', boy.avi -> 'boy'
    Strips trailing '(n)' duplicate markers.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"\(\d+\)$", "", base).strip()
    if not base:
        return None
    return normalize_label(base)


def normalize_label(label: str) -> Optional[str]:
    """Apply filename typo aliases; return None if label is empty."""
    if not label:
        return None
    return FILENAME_LABEL_ALIASES.get(label, label)


def build_final_label_map() -> dict:
    """Return the fixed 8-class label map for final fiber training."""
    return build_label_map(CANONICAL_LABELS)


def find_typo_video_files(data_root: str, domains: List[str], fibers: List[str]) -> List[dict]:
    """List videos whose raw stem differs from normalized label (e.g. gril.avi -> girl)."""
    found = []
    for fiber in fibers:
        for domain_folder in domains:
            fiber_path = os.path.join(data_root, domain_folder, fiber)
            if not os.path.isdir(fiber_path):
                continue
            for vpath in _list_video_files(fiber_path):
                raw = os.path.splitext(os.path.basename(vpath))[0]
                raw = re.sub(r"\(\d+\)$", "", raw).strip()
                norm = normalize_label(raw)
                if raw != norm:
                    found.append({
                        "path": vpath,
                        "fiber": fiber,
                        "domain": domain_folder,
                        "raw_label": raw,
                        "canonical_label": norm,
                    })
    return found


def label_sort_key(label: str) -> Tuple:
    """Deterministic order: digits, single letters, boy/girl, then alpha."""
    if label.isdigit():
        return (0, int(label), label)
    if len(label) == 1 and label.isalpha():
        return (1, label.lower(), label)
    if label in ("boy", "girl"):
        return (2, 0 if label == "boy" else 1, label)
    return (3, label.lower(), label)


def sort_labels(labels: List[str]) -> List[str]:
    return sorted(set(labels), key=label_sort_key)


def build_label_map(labels: List[str]) -> dict:
    ordered = sort_labels(labels)
    label_to_index = {lb: i for i, lb in enumerate(ordered)}
    index_to_label = {str(i): lb for i, lb in enumerate(ordered)}
    return {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "labels": ordered,
        "num_classes": len(ordered),
    }


def discover_fiber_videos(
    data_root: str,
    fiber: str,
    domains: List[str],
    label_map: Optional[dict] = None,
) -> List[dict]:
    """Collect videos for one fiber across domains."""
    videos: List[dict] = []
    label_to_index = (label_map or {}).get("label_to_index", {})

    for domain_folder in domains:
        domain_path = os.path.join(data_root, domain_folder)
        if not os.path.isdir(domain_path):
            continue
        fiber_path = os.path.join(domain_path, fiber)
        if not os.path.isdir(fiber_path):
            continue

        domain_slug = DOMAIN_SLUGS.get(domain_folder, domain_folder)
        vfiles = _list_video_files(fiber_path)

        for vpath in vfiles:
            label_name = extract_label_from_filename(vpath)
            if label_name is None:
                continue
            if label_name.startswith("."):
                continue
            if label_map and label_name not in label_to_index:
                print(f"  [WARNING] Skipping unknown label '{label_name}': {vpath}")
                continue
            label_idx = label_to_index.get(label_name)
            if label_idx is None and label_map is None:
                label_idx = -1
            video_id = f"{domain_slug}/{fiber}/{os.path.basename(vpath)}"
            videos.append({
                "path": vpath,
                "label_name": label_name,
                "letter": label_name,
                "label": label_idx,
                "domain": domain_slug,
                "domain_folder": domain_folder,
                "fiber": fiber,
                "video_id": video_id,
                "filename": os.path.basename(vpath),
            })

    return videos


def discover_all_videos(
    data_root: str,
    domains: List[str],
    fibers: List[str],
) -> List[dict]:
    """Scan all fibers; labels assigned after global label map is built."""
    raw: List[dict] = []
    for fiber in fibers:
        fiber_path_exists = any(
            os.path.isdir(os.path.join(data_root, d, fiber)) for d in domains
        )
        if not fiber_path_exists:
            continue
        for domain_folder in domains:
            fiber_path = os.path.join(data_root, domain_folder, fiber)
            if not os.path.isdir(fiber_path):
                continue
            domain_slug = DOMAIN_SLUGS.get(domain_folder, domain_folder)
            vfiles = _list_video_files(fiber_path)
            for vpath in vfiles:
                label_name = extract_label_from_filename(vpath)
                if label_name is None:
                    continue
                if label_name.startswith("."):
                    continue
                video_id = f"{domain_slug}/{fiber}/{os.path.basename(vpath)}"
                raw.append({
                    "path": vpath,
                    "label_name": label_name,
                    "letter": label_name,
                    "label": -1,
                    "domain": domain_slug,
                    "domain_folder": domain_folder,
                    "fiber": fiber,
                    "video_id": video_id,
                    "filename": os.path.basename(vpath),
                })
    return raw


def apply_label_map(videos: List[dict], label_map: dict) -> List[dict]:
    l2i = label_map["label_to_index"]
    out = []
    for v in videos:
        name = v["label_name"]
        if name not in l2i:
            continue
        v = dict(v)
        v["label"] = l2i[name]
        v["letter"] = name
        out.append(v)
    return out


def probe_video(path: str) -> dict:
    """Read metadata; mark readable=False on failure."""
    info = {
        "readable": False,
        "n_frames": 0,
        "fps": None,
        "width": None,
        "height": None,
        "error": None,
    }
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            info["error"] = "cannot open"
            return info
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        info["fps"] = round(fps, 3) if fps and fps > 0 else None
        cap.release()
        n = _fast_frame_count(path)
        info["n_frames"] = n
        if n <= 0:
            info["error"] = "zero frames"
            return info
        info["readable"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


def estimate_clips(n_frames: int, clip_len: int, stride: int) -> int:
    if n_frames < clip_len:
        return 1 if n_frames > 0 else 0
    return max(1, (n_frames - clip_len) // stride + 1)


def audit_dataset(
    data_root: str,
    domains: List[str],
    fibers: List[str],
    clip_len: int = 16,
    stride: int = 8,
) -> dict:
    """Full dataset audit for markdown report and label map."""
    all_videos = discover_all_videos(data_root, domains, fibers)
    typo_files = find_typo_video_files(data_root, domains, fibers)
    label_map = build_final_label_map()
    all_labels = label_map["labels"]
    all_videos = apply_label_map(all_videos, label_map)

    per_fiber_domain: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    broken: List[dict] = []
    class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for v in all_videos:
        probe = probe_video(v["path"])
        v["probe"] = probe
        key = (v["fiber"], v["domain_folder"])
        per_fiber_domain[v["fiber"]][v["domain_folder"]] = per_fiber_domain[v["fiber"]].get(
            v["domain_folder"], {"videos": [], "missing": False}
        )
        if not probe["readable"]:
            broken.append({
                "path": v["path"],
                "fiber": v["fiber"],
                "domain": v["domain_folder"],
                "label": v["label_name"],
                "error": probe.get("error"),
            })

        class_counts[v["fiber"]][v["label_name"]] += 1

    fiber_label_sets = {
        f: sort_labels([v["label_name"] for v in all_videos if v["fiber"] == f])
        for f in fibers
    }
    global_set = set(all_labels)
    inconsistent_fibers = {
        f: sorted(global_set - set(lbs)) + sorted(set(lbs) - global_set)
        for f, lbs in fiber_label_sets.items()
        if set(lbs) != global_set
    }

    clip_estimates = {}
    for v in all_videos:
        n = v["probe"].get("n_frames", 0)
        est = estimate_clips(n, clip_len, stride)
        clip_estimates[v["video_id"]] = est * 3 if n > 0 else 0

    return {
        "data_root": data_root,
        "domains": domains,
        "fibers": fibers,
        "label_map": label_map,
        "typo_files": typo_files,
        "all_labels": all_labels,
        "num_classes": len(all_labels),
        "total_videos": len(all_videos),
        "videos": all_videos,
        "broken_videos": broken,
        "class_counts_per_fiber": dict(class_counts),
        "fiber_label_sets": fiber_label_sets,
        "inconsistent_fibers": inconsistent_fibers,
        "clip_estimates": clip_estimates,
        "per_fiber_domain": per_fiber_domain,
    }


def save_label_map(label_map: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)


def format_audit_markdown(audit: dict, clip_len: int, stride: int) -> str:
    lines = [
        "# Final 15-Fiber Dataset Audit",
        "",
        f"- **Data root:** `{audit['data_root']}`",
        f"- **Domains:** {', '.join(audit['domains'])}",
        f"- **Fibers:** {', '.join(audit['fibers'])}",
        f"- **Total videos:** {audit['total_videos']}",
        f"- **Number of classes:** {audit['num_classes']}",
        f"- **Labels (ordered):** {', '.join(audit['all_labels'])}",
        "",
        "## Label map",
        "",
        "| Index | Label |",
        "|------:|-------|",
    ]
    for i, lb in enumerate(audit["all_labels"]):
        lines.append(f"| {i} | {lb} |")

    lines.extend(["", "## Videos per fiber and domain", ""])
    for fiber in audit["fibers"]:
        lines.append(f"### {fiber}")
        for domain in audit["domains"]:
            vids = [v for v in audit["videos"] if v["fiber"] == fiber and v["domain_folder"] == domain]
            lines.append(f"- **{domain}:** {len(vids)} videos")
            if not vids and not os.path.isdir(os.path.join(audit["data_root"], domain, fiber)):
                lines.append("  - *folder missing*")
            for v in sorted(vids, key=lambda x: label_sort_key(x["label_name"])):
                p = v["probe"]
                est = estimate_clips(p.get("n_frames", 0), clip_len, stride)
                lines.append(
                    f"  - `{v['filename']}` label=`{v['label_name']}` "
                    f"frames={p.get('n_frames', '?')} fps={p.get('fps')} "
                    f"size={p.get('width')}x{p.get('height')} "
                    f"readable={p.get('readable')} "
                    f"~clips/video(deploy)={est * 3}"
                )
        lines.append("")

    lines.extend(["## Class counts per fiber (videos)", ""])
    for fiber, counts in sorted(audit["class_counts_per_fiber"].items()):
        parts = ", ".join(f"{k}:{counts[k]}" for k in sort_labels(list(counts.keys())))
        lines.append(f"- **{fiber}:** {parts}")

    if audit["inconsistent_fibers"]:
        lines.extend(["", "## Label set inconsistencies", ""])
        for fiber, diff in audit["inconsistent_fibers"].items():
            if diff:
                lines.append(f"- **{fiber}:** differs from global set ({diff})")
    else:
        lines.extend(["", "## Label set inconsistencies", "", "All fibers share the same class set."])

    if audit["broken_videos"]:
        lines.extend(["", "## Broken or unreadable videos", ""])
        for b in audit["broken_videos"]:
            lines.append(f"- `{b['path']}` ({b['fiber']}/{b['domain']}) — {b['error']}")
    else:
        lines.extend(["", "## Broken or unreadable videos", "", "None detected."])

    if audit.get("typo_files"):
        lines.extend(["", "## Filename typo aliases (raw file unchanged)", ""])
        for t in audit["typo_files"]:
            lines.append(
                f"- `{t['path']}`: raw `{t['raw_label']}` -> canonical `{t['canonical_label']}`"
            )

    lines.extend([
        "",
        "## Split strategy (training)",
        "",
        "Default final training uses **uniform_temporal**: all clips from each video "
        "are listed in time order, clip indices are shuffled with a fixed seed, then "
        "partitioned 70% train / 15% val / 15% test so splits sample the full timeline.",
        "",
        "Legacy **contiguous_temporal** (first 70% / next 15% / last 15% of frames) "
        "remains available via `--split_strategy contiguous_temporal`.",
        "",
        "Clips use disjoint frame ranges; the same video may contribute clips to "
        "multiple splits. Per-fiber `split_report.json` records clip indices per video.",
    ])
    return "\n".join(lines) + "\n"
