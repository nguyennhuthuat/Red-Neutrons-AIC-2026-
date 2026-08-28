# -*- coding: utf-8 -*-
"""Gộp kênh lời nói vào CLIP có lãi không? Đo trên đủ bộ 81 câu, thang chính thức.

Đây là con số mà kế hoạch Elasticsearch ghi là "quyết định" nhưng chưa ai đo:
RRF giữa bảng xếp hạng CLIP và một kênh từ vựng trên toàn kho. Kênh chữ đồ hoạ
đã thử và thua (mục sec:ocrchu); lời nói thì cùng loại văn với câu truy vấn nên
đáng thử lại.

CLIP nhận câu tiếng Anh (đúng cấu hình đang chạy), kênh lời nói nhận tiếng Việt
vì bản chép là tiếng Việt.

  python eval/danh_gia_asr_tim.py --w 0 0.1 0.25 0.5 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import asr_tim  # noqa: E402
import corpus  # noqa: E402

KS = (1, 5, 20, 50, 100)
TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"
RRF_K = 60


def diem_clip(texts, w_phu: float) -> np.ndarray:
    import open_clip
    import torch

    F = np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy")
    m, _, _ = open_clip.create_model_and_transforms(TEN, pretrained=PRE)
    tok = open_clip.get_tokenizer(TEN)
    m.eval()
    with torch.no_grad():
        q = m.encode_text(tok(list(texts))).float().numpy()
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    S = F @ q.T
    if w_phu > 0:
        import ensemble as ens
        S = np.stack([ens.ghep(S[:, i], t, w=w_phu)
                      for i, t in enumerate(texts)], axis=1)
    return S


def thu_hang(diem: np.ndarray) -> np.ndarray:
    o = np.argsort(-diem, kind="stable")
    h = np.empty(len(diem), dtype=np.int64)
    h[o] = np.arange(1, len(diem) + 1)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="eval/queries_hcmc2026.csv")
    ap.add_argument("--ensemble", type=float, default=0.5)
    ap.add_argument("--w", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 1.0],
                    help="trọng số kênh lời nói trong RRF")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-cau", default=None,
                    help="ép cột cho CLIP; mặc định text_en nếu có")
    ap.add_argument("--cau", default="text_vi",
                    help="cột dùng cho kênh lời nói")
    args = ap.parse_args()

    q = pd.read_csv(ROOT / args.bo)
    if args.limit:
        q = q.head(args.limit)
    meta = corpus.load_metadata()
    print(f"{len(q)} câu · {len(meta):,} khung")

    cot = args.clip_cau or ("text_en" if "text_en" in q.columns
                            else "text_vi")
    print(f"CLIP nhận cột {cot}")
    S = diem_clip(q[cot].tolist(), args.ensemble)
    A = np.stack([asr_tim.diem_khung(t) for t in q[args.cau]], axis=1)
    print(f"kênh lời nói: {(A > 0).sum(0).mean():.0f} khung có điểm mỗi câu")

    dung = ((meta.video_id.to_numpy()[:, None] == q.video_id.to_numpy())
            & (meta.frame_idx.to_numpy()[:, None] >= q.frame_idx_min.to_numpy())
            & (meta.frame_idx.to_numpy()[:, None] <= q.frame_idx_max.to_numpy()))

    def cham(hs):
        h = np.array(hs)
        r = {k: float((h <= k).mean()) for k in KS}
        return r, float(np.mean(list(r.values())))

    bang = {}
    for w in args.w:
        hs = []
        for i in range(len(q)):
            h_clip = thu_hang(S[:, i])
            if w <= 0:
                tong = -h_clip.astype(np.float64)
            else:
                h_asr = thu_hang(A[:, i])
                tong = 1.0 / (RRF_K + h_clip) + w / (RRF_K + h_asr)
            hs.append(int(thu_hang(-tong if w <= 0 else tong)[dung[:, i]].min())
                      if w > 0 else int(h_clip[dung[:, i]].min()))
        bang[w] = cham(hs)

    hs_asr = [int(thu_hang(A[:, i])[dung[:, i]].min()) for i in range(len(q))]
    bang["chỉ lời nói"] = cham(hs_asr)

    print(f"\n{'cấu hình':<16}" + "".join(f"  R@{k:<4}" for k in KS) + "   FINAL")
    for ten, (r, f) in bang.items():
        nhan = "CLIP một mình" if ten == 0.0 else (
            ten if isinstance(ten, str) else f"RRF w={ten}")
        print(f"{str(nhan):<16}" + "".join(f"  {r[k]:.3f}" for k in KS)
              + f"   {f:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
