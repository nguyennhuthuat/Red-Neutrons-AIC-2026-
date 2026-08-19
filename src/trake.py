
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "hcmc2026"
PROC = ROOT / "data" / "processed_hcmc2026"


@lru_cache(maxsize=1)
def _video_index() -> dict[str, Path]:
    """video_id -> đường dẫn mp4. Quét một lần rồi nhớ, 873 file nằm rải 14 gói."""
    return {p.stem: p for p in RAW.rglob("*.mp4")}


def video_path(video_id: str) -> Path:
    p = _video_index().get(video_id)
    if p is None:
        raise FileNotFoundError(
            f"không thấy {video_id}.mp4 trong {RAW} — TRAKE bắt buộc cần video gốc")
    return p


def read_frames(video_id: str, frame_ids) -> tuple[list[int], list]:
    import cv2
    from PIL import Image

    want = sorted(set(int(f) for f in frame_ids))
    cap = cv2.VideoCapture(str(video_path(video_id)))
    if not cap.isOpened():
        raise RuntimeError(f"không mở được video {video_id}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    got_id, got_img, pos = [], [], -1
    try:
        for f in want:
            if f < 0 or (total and f >= total):
                continue
            if pos < 0 or f - pos > 30:        # nhảy xa thì seek, gần thì đọc lướt
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                pos = f
            else:
                while pos < f:
                    cap.grab()                  # grab: giải mã bỏ, rẻ hơn read
                    pos += 1
            ok, bgr = cap.read()
            pos += 1
            if not ok:
                continue
            got_id.append(f)
            got_img.append(Image.fromarray(bgr[:, :, ::-1]))
    finally:
        cap.release()
    return got_id, got_img


def _encode_images(images, model_name: str, pretrained: str) -> np.ndarray:
    import open_clip
    import torch
    key = (model_name, pretrained)
    if key not in _MODELS:
        m, _, prep = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        m.eval()
        _MODELS[key] = (m, prep, open_clip.get_tokenizer(model_name))
    model, prep, _ = _MODELS[key]
    with torch.no_grad():
        v = model.encode_image(torch.stack([prep(i) for i in images])).float()
        v /= v.norm(dim=-1, keepdim=True)
    return v.numpy()


def encode_text(text: str, model_name: str, pretrained: str) -> np.ndarray:
    import open_clip
    import torch
    key = (model_name, pretrained)
    if key not in _MODELS:
        m, _, prep = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        m.eval()
        _MODELS[key] = (m, prep, open_clip.get_tokenizer(model_name))
    model, _, tokz = _MODELS[key]
    with torch.no_grad():
        v = model.encode_text(tokz([text])).float()
        v /= v.norm(dim=-1, keepdim=True)
    return v.numpy()[0]


_MODELS: dict = {}

ALIGN_MODEL = ("ViT-B-16-SigLIP2-256", "webli")

NEO_MODEL = ("ViT-B-16-SigLIP2-384", "webli")
NEO_FEATURES = (ROOT / "data" / "kernel_out"
                / "features_ViT-B-16-SigLIP2-384__webli.npy")


@lru_cache(maxsize=1)
def _feat_neo():
    """Vector của encoder chọn mỏ neo. Lùi về encoder chính nếu thiếu tệp."""
    if NEO_FEATURES.exists():
        return np.load(NEO_FEATURES, mmap_mode="r")
    return _corpus()[1]


def align_moment(video_id: str, text_vec: np.ndarray, anchor_frame: int, *,
                 model_name: str, pretrained: str,
                 window: int = 60, coarse_step: int = 5, fine_span: int = 6,
                 verbose: bool = False) -> tuple[int, float, int]:
    n_enc = 0

    coarse = list(range(anchor_frame - window, anchor_frame + window + 1, coarse_step))
    ids, imgs = read_frames(video_id, coarse)
    if not ids:
        return anchor_frame, float("-inf"), 0
    sims = _encode_images(imgs, model_name, pretrained) @ text_vec
    n_enc += len(imgs)
    best = int(ids[int(np.argmax(sims))])
    if verbose:
        print(f"    thô: {len(ids)} khung -> frame {best} ({sims.max():.4f})")

    fine = [f for f in range(best - fine_span, best + fine_span + 1) if f not in set(ids)]
    if fine:
        ids2, imgs2 = read_frames(video_id, fine)
        if ids2:
            s2 = _encode_images(imgs2, model_name, pretrained) @ text_vec
            n_enc += len(imgs2)
            ids = list(ids) + list(ids2)
            sims = np.concatenate([sims, s2])

    k = int(np.argmax(sims))
    if verbose:
        print(f"    tinh: +{n_enc - len(coarse)} khung -> frame {ids[k]} ({sims[k]:.4f})")
    return int(ids[k]), float(sims[k]), n_enc


def choose_anchors(scores: np.ndarray, frames=None) -> list[int]:
    S = np.asarray(scores, dtype=np.float64)
    N, K = S.shape
    dp = np.full((N, K), -np.inf)
    par = np.zeros((N, K), dtype=np.int32)
    dp[0] = S[0]
    ar = np.arange(K)
    for j in range(1, N):
        prev = dp[j - 1]
        run = np.maximum.accumulate(prev)
        arg = np.maximum.accumulate(np.where(prev == run, ar, 0))
        # mốc j phải đứng SAU hẳn mốc j-1 nên chỉ nhìn tiền tố tới k-1
        dp[j] = S[j] + np.concatenate([[-np.inf], run[:-1]])
        par[j] = np.concatenate([[0], arg[:-1]])
    k = int(np.argmax(dp[N - 1]))
    out = [k]
    for j in range(N - 1, 0, -1):
        k = int(par[j][k])
        out.append(k)
    out = out[::-1]
    return [int(frames[i]) for i in out] if frames is not None else out


def align_sequence(video_id: str, texts, anchors, *, model_name: str | None = None,
                   pretrained: str | None = None, **kw) -> list[dict]:
    if model_name is None:
        model_name, pretrained = ALIGN_MODEL
    out = []
    for t, a in zip(texts, anchors):
        vec = encode_text(t, model_name, pretrained)
        f, s, n = align_moment(video_id, vec, int(a), model_name=model_name,
                               pretrained=pretrained, **kw)
        out.append({"text": t, "anchor": int(a), "frame_idx": f, "score": s, "n_encode": n})
    return out


@lru_cache(maxsize=1)
def _corpus():
    """(metadata, features mmap, manifest) của kho keyframe. Nạp một lần."""
    import json
    import pandas as pd
    return (pd.read_parquet(PROC / "metadata.parquet"),
            np.load(PROC / "features.npy", mmap_mode="r"),
            json.loads((PROC / "manifest.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _nhom_video() -> dict:
    """video_id -> chỉ số hàng trong metadata, đã xếp theo thời gian."""
    meta, _, _ = _corpus()
    v = meta.video_id.to_numpy()
    fr = meta.frame_idx.to_numpy()
    out = {}
    for k in dict.fromkeys(v):
        ix = np.where(v == k)[0]
        out[k] = ix[np.argsort(fr[ix])]
    return out


def rank_videos_scores(S: np.ndarray, top_k: int = 10, nen: int = 50,
                       kem_neo: bool = False):
    S = np.asarray(S)
    nhom = _nhom_video()
    ung_vien = nhom.keys()
    if nen:
        tho = {v: S[:, ix].max() for v, ix in nhom.items()}
        ung_vien = sorted(tho, key=tho.get, reverse=True)[:nen]

    diem, neo = {}, {}
    for v in ung_vien:
        s = S[:, nhom[v]]
        if s.shape[1] < s.shape[0]:        # video ít keyframe hơn số mốc
            continue
        loc = choose_anchors(s)
        diem[v] = float(sum(s[j, loc[j]] for j in range(s.shape[0])))
        # Hàng metadata của từng mốc — để giao diện bày cả chuỗi, không chỉ 1 ảnh.
        neo[v] = [int(nhom[v][k]) for k in loc]
    xh = sorted(diem.items(), key=lambda kv: -kv[1])[:top_k]
    return [(v, d, neo[v]) for v, d in xh] if kem_neo else xh


def rank_videos(texts, top_k: int = 10, nen: int = 50) -> list[tuple[str, float]]:
    meta, feat, man = _corpus()
    T = np.vstack([encode_text(t, man["clip_model"], man["clip_pretrained"])
                   for t in texts])
    return rank_videos_scores(T @ np.asarray(feat, dtype=np.float32).T,
                              top_k=top_k, nen=nen)


# ĐỪNG cộng ensemble vào bước này: 50% -> 42% (phụ lục B, mục trake.py).
def keyframe_anchors(video_id: str, texts) -> list[int]:
    meta, _, _ = _corpus()
    sub = meta[meta.video_id == video_id]
    if sub.empty:
        raise ValueError(f"không có keyframe nào của {video_id} trong kho")
    idx = sub.index.to_numpy()
    frames = sub.frame_idx.to_numpy()
    o = np.argsort(frames)                       # ứng viên phải xếp theo thời gian
    idx, frames = idx[o], frames[o]

    feat = _feat_neo()
    T = np.vstack([encode_text(t, *NEO_MODEL) for t in texts])
    S = T @ np.asarray(feat[idx], dtype=np.float32).T
    return choose_anchors(S, frames=frames)


def locate_sequence(video_id: str, texts, **kw) -> list[dict]:
    return align_sequence(video_id, list(texts),
                          keyframe_anchors(video_id, texts), **kw)


def submission_row(video_id: str, moments) -> tuple:
    return (video_id, *[int(m["frame_idx"]) for m in moments])
