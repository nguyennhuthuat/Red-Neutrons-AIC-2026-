"""Căn đúng khoảnh khắc trong video, cho dạng truy vấn TRAKE.

Cửa sổ đáp án mỗi mốc thường dưới 10 frame còn keyframe cách nhau ~90 frame, nên
bắt buộc giải mã video gốc. Giải mã rẻ (~494 fps) nhưng mã hoá đắt, nên chạy thô
rồi tinh: keyframe làm mốc → quét thưa bước `coarse_step` → quét dày ±`fine_span`.

Cơ sở đo cho từng hằng số (bán kính 60, encoder nhỏ cho bước căn tinh, ràng buộc
thứ tự thời gian): xem docs/bao_cao_he_thong.tex, mục "Dạng TRAKE".
"""

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
    """Đọc đúng những frame được yêu cầu. Trả (frame_id thật đọc được, ảnh PIL).

    Đọc TUẦN TỰ và chỉ seek khi bước nhảy lớn: seek mất 0,12 s còn đọc một frame
    chỉ mất 0,002 s, nên nhảy lung tung sẽ chậm gấp 60 lần đọc thẳng.
    """
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

# Encoder cho bước CĂN TINH — cố ý khác encoder tìm kiếm. Căn tinh chỉ chọn 1
# trong ~28 khung gần nhau nên model nhỏ đủ dùng và nhanh gấp 8,8 lần.
ALIGN_MODEL = ("ViT-B-16-SigLIP2-256", "webli")


def align_moment(video_id: str, text_vec: np.ndarray, anchor_frame: int, *,
                 model_name: str, pretrained: str,
                 window: int = 60, coarse_step: int = 5, fine_span: int = 6,
                 verbose: bool = False) -> tuple[int, float, int]:
    """Tìm frame khớp `text_vec` nhất quanh `anchor_frame`.

    Trả (frame_id tốt nhất, điểm, số lần mã hoá đã tốn). `window` tính bằng FRAME.

    ⚠️ window=60 là con số ĐÃ ĐO trên hai nền encoder khác nhau, đừng nới rộng
    cho "chắc ăn" — ±120 phá mất mốc đang đúng.
    """
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
    """Chọn mỏ neo cho từng mốc, ép đúng thứ tự thời gian.

    `scores`: ma trận (N mốc, K ứng viên), ứng viên xếp sẵn theo thời gian tăng.
    Trả list N chỉ số; truyền `frames` thì trả thẳng frame_idx.

    Quy hoạch động O(N·K): dp[j][k] = S[j][k] + max(dp[j-1][k' < k]), lấy tiền tố
    cực đại bằng `maximum.accumulate` nên không phải O(N·K²).

    ⚠️ Đã thử và BỎ: z-score theo từng mốc (đường đi không đổi), phạt λ cho
    khoảng cách giữa hai mốc (lỗ đều, không có vùng phẳng ⇒ khớp nhiễu).
    """
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
    """Căn cả một chuỗi mốc. `anchors` là frame_idx thô của từng mốc.

    Giữ cả điểm trong kết quả để người thi thấy mốc nào đáng ngờ. Mặc định dùng
    ALIGN_MODEL.
    """
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


def rank_videos_scores(S: np.ndarray, top_k: int = 10,
                       nen: int = 50) -> list[tuple[str, float]]:
    """Giai đoạn 1a từ ma trận điểm sẵn có. `S`: (n_mốc, n_khung toàn corpus).

    Chấm mỗi video bằng TỔNG điểm của đường đi đã ép thứ tự thời gian, thay vì
    bằng điểm cao nhất mà một mốc bất kỳ đạt được. Luật cũ cho một video thắng chỉ
    nhờ tình cờ chứa MỘT khung giống MỘT mốc; luật này bắt video phải chứa được cả
    chuỗi. Không thêm tham số nào — ràng buộc thứ tự là thứ đề bài cho sẵn.

    Đo trên bộ 5 chuỗi: luật cũ đúng 4/5, luật này đúng 5/5, kéo điểm TRAKE
    đầu-cuối từ 0,3467 lên 0,3967. ⚠️ Chênh lệch đó là MỘT chuỗi trên bộ 5 chuỗi —
    cơ chế vững nhưng bộ đo quá nhỏ để khẳng định độ lớn.

    `nen` = lọc thô bằng điểm cao nhất trước khi chạy quy hoạch động, vì chạy QHĐ
    cho cả 873 video mất 8,6 giây. Luật cũ xếp video đúng ở hạng 1-2 trên cả 5
    chuỗi nên 50 là biên rất rộng; đặt 0 để tắt lọc.

    Nhận ma trận điểm thay vì tự mã hoá câu để BÊN GỌI DÙNG LẠI ENCODER CỦA MÌNH:
    giao diện đã giữ sẵn một bản SigLIP2-L, và nạp bản thứ hai (~1,7 GB) làm cạn
    bộ nhớ ảo của Windows. Đã dính thật, chỉ lộ khi chạy giao diện chứ không lộ ở
    phép đo nào.
    """
    S = np.asarray(S)
    nhom = _nhom_video()
    ung_vien = nhom.keys()
    if nen:
        tho = {v: S[:, ix].max() for v, ix in nhom.items()}
        ung_vien = sorted(tho, key=tho.get, reverse=True)[:nen]

    diem = {}
    for v in ung_vien:
        s = S[:, nhom[v]]
        if s.shape[1] < s.shape[0]:        # video ít keyframe hơn số mốc
            continue
        loc = choose_anchors(s)
        diem[v] = float(sum(s[j, loc[j]] for j in range(s.shape[0])))
    return sorted(diem.items(), key=lambda kv: -kv[1])[:top_k]


def rank_videos(texts, top_k: int = 10, nen: int = 50) -> list[tuple[str, float]]:
    """Bản tiện dụng: tự mã hoá câu rồi gọi `rank_videos_scores`.

    Dùng cho script đo. Giao diện thì nên tự dựng ma trận điểm bằng encoder đã
    nạp sẵn của nó rồi gọi thẳng `rank_videos_scores`.
    """
    meta, feat, man = _corpus()
    T = np.vstack([encode_text(t, man["clip_model"], man["clip_pretrained"])
                   for t in texts])
    return rank_videos_scores(T @ np.asarray(feat, dtype=np.float32).T,
                              top_k=top_k, nen=nen)


def keyframe_anchors(video_id: str, texts) -> list[int]:
    """Giai đoạn 1b: chọn mỏ neo (frame_idx) cho từng mốc trong một video.

    Chấm keyframe bằng ENCODER TÌM KIẾM trong manifest (không phải ALIGN_MODEL),
    rồi ép thứ tự thời gian.

    ⚠️ ĐỪNG cộng `ensemble` vào đây dù nó ăn ở KIS và Q&A — đo trên 26 mốc TRAKE
    thì nó LỖ (trong ±60: 50% → 46% → 42% khi w tăng). Ensemble giúp truy xuất
    toàn corpus nhưng hại phân biệt tinh trong một video đã biết. Nó vẫn dùng
    được ở bước chọn VIDEO, chỉ cấm ở bước chọn MỎ NEO này.
    """
    meta, feat, man = _corpus()
    sub = meta[meta.video_id == video_id]
    if sub.empty:
        raise ValueError(f"không có keyframe nào của {video_id} trong kho")
    idx = sub.index.to_numpy()
    frames = sub.frame_idx.to_numpy()
    o = np.argsort(frames)                       # ứng viên phải xếp theo thời gian
    idx, frames = idx[o], frames[o]

    T = np.vstack([encode_text(t, man["clip_model"], man["clip_pretrained"])
                   for t in texts])
    S = T @ np.asarray(feat[idx], dtype=np.float32).T
    return choose_anchors(S, frames=frames)


def locate_sequence(video_id: str, texts, **kw) -> list[dict]:
    """Chạy trọn TRAKE trên một video: chọn mỏ neo (có ép thứ tự) rồi căn tinh.

    Đây là đường chạy nên dùng; `align_sequence` là tầng dưới, đòi tự đưa mỏ neo.
    """
    return align_sequence(video_id, list(texts),
                          keyframe_anchors(video_id, texts), **kw)


def submission_row(video_id: str, moments) -> tuple:
    """Dựng một dòng nộp TRAKE: `(video_id, frame_id₁, …, frame_idₙ)`.

    Thứ tự mốc phải giữ nguyên thứ tự đề bài — chấm theo từng giai đoạn, đảo thứ
    tự là mất điểm dù tìm đúng cả n khoảnh khắc.
    """
    return (video_id, *[int(m["frame_idx"]) for m in moments])
