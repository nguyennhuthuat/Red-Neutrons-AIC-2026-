
import os
import sys
import json
import argparse 
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not installed. Run `pip install faiss-cpu` or `pip install faiss-gpu`. ")

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "hcmc2023":{
        "raw":             ROOT / "data" / "hcmc2023", 
        "processed":       ROOT / "data" / "processed_hcmc2023", 
        "clip_model":      "ViT-B-16-quickgelu", 
        "clip_pretrained": "openai", 
        "kaggle_slug":     "mkimwp/hcmc-ai-challenge-2023",
    }, 
    "hcmc2026":{
        "raw":             ROOT / "data" / "hcmc2026", 
        "processed":       ROOT / "data" / "processed_hcmc2026",
        "clip_model":      "ViT-B-32-quickgelu", 
        "clip_pretrained": "openai", 
        "kaggle_slug":     None,
    }
}

""" Download dataset 2023"""


def find_map_keyframe_csvs(root: Path) -> dict[str, Path]:
    result = {}
    for csv in root.rglob("*.csv"):
        parts_lower = str(csv).lower()
        if "map" in parts_lower and "keyframe" in parts_lower:
            result[csv.stem] = csv
    print(f"[discover] Found {len(result)} map-keyframes CSVs")
    return result


def find_feature_files(root: Path) -> dict[str, Path]:
    """Return {video_id: npy_path} for all CLIP feature files."""
    result = {}
    for npy in root.rglob("*.npy"):
        result[npy.stem] = npy
    print(f"[discover] Found {len(result)} .npy feature files")
    return result


def find_keyframe_dirs(root: Path) -> dict[str, Path]:
    result = {}
    for d in root.rglob("*"):
        if d.is_dir() and "_V" in d.name:
            # cheap check: does it contain at least one image?
            has_img = next(d.glob("*.jpg"), None) or next(d.glob("*.png"), None)
            if has_img:
                result[d.name] = d
    print(f"[discover] Found keyframe image dirs for {len(result)} videos")
    return result


def keyframe_image_path(kf_dir: Path | None, n: int) -> str:
    if kf_dir is None:
        return ""
    for pad in (3, 4, 5):
        for ext in (".jpg", ".png"):
            p = kf_dir / f"{n:0{pad}d}{ext}"
            if p.exists():
                return str(p)
    return ""


def build_artifacts(cfg: dict):
    data_dir = cfg["raw"]
    out_dir = cfg["processed"]
    csvs = find_map_keyframe_csvs(data_dir)
    feats = find_feature_files(data_dir)
    kf_dirs = find_keyframe_dirs(data_dir)

    if not csvs or not feats:
        sys.exit("[build] Missing CSVs or feature files. Check dataset layout "
                 "with explore_data.py first.")

    common = sorted(set(csvs) & set(feats))
    only_csv = set(csvs) - set(feats)
    only_npy = set(feats) - set(csvs)
    if only_csv:
        print(f"[build] WARNING: {len(only_csv)} videos have CSV but no features "
              f"(e.g. {sorted(only_csv)[:3]})")
    if only_npy:
        print(f"[build] WARNING: {len(only_npy)} videos have features but no CSV "
              f"(e.g. {sorted(only_npy)[:3]})")
    print(f"[build] Processing {len(common)} videos with both CSV + features")

    meta_frames = []
    feat_blocks = []
    dim = None

    for i, vid in enumerate(common):
        df = pd.read_csv(csvs[vid])
        arr = np.load(feats[vid])

        if arr.ndim != 2:
            print(f"[build] SKIP {vid}: unexpected feature shape {arr.shape}")
            continue
        if dim is None:
            dim = arr.shape[1]
            print(f"[build] Feature dim = {dim}")
        elif arr.shape[1] != dim:
            print(f"[build] SKIP {vid}: dim {arr.shape[1]} != {dim}")
            continue

        # Row-count sanity check: CSV rows must align with feature rows.
        if len(df) != arr.shape[0]:
            k = min(len(df), arr.shape[0])
            print(f"[build] WARNING {vid}: {len(df)} CSV rows vs "
                  f"{arr.shape[0]} feature rows -> truncating to {k}")
            df = df.iloc[:k]
            arr = arr[:k]

        df = df.copy()
        df["video_id"] = vid
        kd = kf_dirs.get(vid)
        df["image_path"] = [keyframe_image_path(kd, int(n)) for n in df["n"]]

        meta_frames.append(df)
        feat_blocks.append(arr.astype(np.float32))

        if (i + 1) % 100 == 0:
            print(f"[build] ... {i + 1}/{len(common)} videos")

    metadata = pd.concat(meta_frames, ignore_index=True)

    for col in ["n", "pts_time", "fps", "frame_idx"]:
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

    # Khôi phục frame_idx bị thiếu: frame_idx ≈ round(pts_time * fps).
    missing = metadata["frame_idx"].isna()
    if missing.any():
        print(f"[build] Vá {int(missing.sum())} frame_idx rỗng từ pts_time*fps")
        metadata.loc[missing, "frame_idx"] = (
            metadata.loc[missing, "pts_time"] * metadata.loc[missing, "fps"]
        ).round()

    metadata["n"] = metadata["n"].astype("int64")
    metadata["frame_idx"] = metadata["frame_idx"].astype("int64")

    features = np.vstack(feat_blocks)
    assert len(metadata) == features.shape[0], "metadata/features row mismatch"

    # L2-normalize so inner product == cosine similarity.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(out_dir / "metadata.parquet", index=False)
    np.save(out_dir / "features.npy", features)
    print(f"[build] Saved metadata.parquet ({len(metadata)} rows) "
          f"and features.npy {features.shape}")

    if faiss is not None:
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        faiss.write_index(index, str(out_dir / "faiss.index"))
        print(f"[build] Saved faiss.index ({index.ntotal} vectors)")

    # Save a small manifest so teammates can verify their build matches.
    manifest = {
        "dataset": cfg["raw"].name,
        "clip_model": cfg["clip_model"], 
        "clip_pretrained": cfg["clip_pretrained"], 
        "num_videos": len(common),
        "num_keyframes": int(len(metadata)),
        "feature_dim": int(dim),
        "columns": list(metadata.columns),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[build] Manifest: {manifest}")


if __name__ == "__main__": 
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices= list(DATASETS), default="hcmc2026")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    print(f"[main] Dataset: {args.dataset} -> {cfg['processed']}")

    build_artifacts(cfg)