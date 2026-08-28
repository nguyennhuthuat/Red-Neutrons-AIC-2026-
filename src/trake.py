
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


def align_scores(video_id: str, text_vec: np.ndarray, anchor_frame: int, *,
                 model_name: str, pretrained: str,
                 window: int = 60, coarse_step: int = 5, fine_span: int = 6,
                 verbose: bool = False) -> tuple[list[int], np.ndarray, int]:
    """Điểm của MỌI khung đã mã hoá quanh mỏ neo, không chỉ khung thắng."""
    n_enc = 0

    coarse = list(range(anchor_frame - window, anchor_frame + window + 1, coarse_step))
    ids, imgs = read_frames(video_id, coarse)
    if not ids:
        return [], np.zeros(0), 0
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

    if verbose:
        k = int(np.argmax(sims))
        print(f"    tinh: +{n_enc - len(coarse)} khung -> frame {ids[k]} ({sims[k]:.4f})")
    return [int(i) for i in ids], np.asarray(sims, dtype=np.float64), n_enc


def align_moment(video_id: str, text_vec: np.ndarray, anchor_frame: int,
                 **kw) -> tuple[int, float, int]:
    ids, sims, n_enc = align_scores(video_id, text_vec, anchor_frame, **kw)
    if not ids:
        return int(anchor_frame), float("-inf"), 0
    k = int(np.argmax(sims))
    return ids[k], float(sims[k]), n_enc


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
        ids, sims, n = align_scores(video_id, vec, int(a), model_name=model_name,
                                    pretrained=pretrained, **kw)
        if not ids:
            out.append({"text": t, "anchor": int(a), "frame_idx": int(a),
                        "score": float("-inf"), "n_encode": 0, "cands": []})
            continue
        o = np.argsort(-sims)
        k = int(o[0])
        # Giữ cả bảng điểm: sinh dòng dự phòng về sau KHÔNG tốn thêm mã hoá.
        out.append({"text": t, "anchor": int(a), "frame_idx": ids[k],
                    "score": float(sims[k]), "n_encode": n,
                    "cands": [(ids[i], float(sims[i])) for i in o]})
    return out


def _thua(cands, per_moment: int, spread: int):
    """Chọn ứng viên theo điểm nhưng ép cách nhau `spread` frame — dò tinh cho
    hàng chục khung sát nhau, giữ nguyên thì 16 dòng đều nằm trong ±6 frame."""
    giu = []
    for f, sc in cands:
        if all(abs(f - g) >= spread for g, _ in giu):
            giu.append((f, sc))
            if len(giu) >= per_moment:
                break
    return giu or list(cands[:per_moment])


def candidate_rows(moments, limit: int = 100, per_moment: int = 16,
                   spread: int = 10, beam: int = 4000) -> list[list[int]]:
    """Tới `limit` chuỗi frame tăng dần, xếp theo tổng điểm. Không mã hoá thêm."""
    cols = []
    for m in moments:
        c = m.get("cands") or [(int(m["frame_idx"]), 0.0)]
        cols.append(sorted(_thua(c, per_moment, spread), key=lambda x: x[0]))
    if not cols:
        return []

    duong = [([f], sc) for f, sc in cols[0]]
    for c in cols[1:]:
        moi = [(d + [f], t + sc) for d, t in duong for f, sc in c if f > d[-1]]
        if not moi:
            return []
        moi.sort(key=lambda x: -x[1])
        duong = moi[:beam]
    duong.sort(key=lambda x: -x[1])
    return [d for d, _ in duong[:limit]]


def rai_luoi(frames, limit: int = 100, buoc: int = 20) -> list[list[int]]:
    """Dòng dự phòng quanh MỘT chuỗi đã có, không cần mô hình lẫn bảng điểm.

    Mỗi dòng chỉ đẩy một mốc đi ±k·bước. Đo 21/08 trên 8 chuỗi eval: ngang
    ngửa cách xếp theo điểm (±10 frame: 0,2004 so với 0,2033)."""
    import itertools
    goc = [int(f) for f in frames]
    ra, thay = [list(goc)], {tuple(goc)}
    for k in itertools.count(1):
        them = False
        for d in (buoc * k, -buoc * k):
            for j in range(len(goc)):
                r = list(goc)
                r[j] += d
                if r[j] < 0 or any(b <= a for a, b in zip(r, r[1:])):
                    continue
                them = True
                if tuple(r) not in thay:
                    thay.add(tuple(r))
                    ra.append(r)
                    if len(ra) >= limit:
                        return ra
        if not them or k > 200:
            return ra


def submission_rows(video_moments, limit: int = 100, **kw) -> list[tuple]:
    """Gộp nhiều video đã căn thành danh sách dòng nộp, video tốt nhất trước.

    Video đầu giữ phần lớn chỗ (sai video là mất trắng, mà bước 1 chọn đúng
    video 8/8), nhưng video dự phòng vẫn phải có chỗ thật sự."""
    vm = list(video_moments)
    if not vm:
        return []
    du = limit // (2 * len(vm))
    phan = [limit - du * (len(vm) - 1)] + [du] * (len(vm) - 1)

    ra = []
    for (vid, moments), n in zip(vm, phan):
        ra.extend((vid, *f) for f in candidate_rows(moments, limit=n, **kw))
    if len(ra) < limit:                       # video nào không đủ dòng thì bù
        for vid, moments in vm:
            for f in candidate_rows(moments, limit=limit, **kw):
                if len(ra) >= limit:
                    break
                if (vid, *f) not in ra:
                    ra.append((vid, *f))
    return ra[:limit]


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
