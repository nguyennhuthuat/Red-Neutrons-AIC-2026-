# -*- coding: utf-8 -*-
"""Đo ĐỘ PHỦ của rổ: khung đáp án có nằm trong 20 ảnh đưa cho VLM không.

Khác hẳn evaluate.py. Đó đo THỨ HẠNG trên danh sách nộp 100 dòng, nên đa dạng
hoá theo video làm tụt điểm. Tệp này đo XÁC SUẤT CHỨA trong một rổ nhỏ cố định
— mục tiêu khác, và tối ưu cho nó có thể ngược dấu. Không gọi API lần nào.

  python eval/danh_gia_phuro.py --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import corpus  # noqa: E402
from evaluate import load_clip, encode_batch, translate  # noqa: E402

TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"
BO = ["queries_qa_hcmc2026.csv", "queries_qa_tenrieng.csv", "queries_qa_chunho.csv"]


def ro_thuan(thu_tu, vid, top):
    """Cách hiện tại: lấy thẳng top-N toàn kho."""
    return list(thu_tu[:top])


def ro_theo_video(thu_tu, vid, top, moi_video):
    """Chia đều: mỗi video nhiều nhất `moi_video` khung, quét theo thứ hạng."""
    dem, ra = {}, []
    for i in thu_tu:
        v = vid[i]
        if dem.get(v, 0) >= moi_video:
            continue
        dem[v] = dem.get(v, 0) + 1
        ra.append(int(i))
        if len(ra) >= top:
            break
    return ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--translate", default="google")
    args = ap.parse_args()

    meta = corpus.load_metadata()
    vid = meta.video_id.to_numpy()
    X = np.asarray(np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy",
                           mmap_mode="r"))
    model, tok = load_clip(TEN, PRE)

    cach = [("top-N thuần (hiện tại)", None)] + [
        (f"≤{k} khung/video", k) for k in (1, 2, 3, 4, 5, 8, 10)]
    bang = {ten: [] for ten, _ in cach}

    for bo in BO:
        d = pd.read_csv(ROOT / "eval" / bo)
        truy, _ = translate(d["text_vi"].tolist(), args.translate)
        Q = encode_batch(truy, model, tok)
        S = X @ Q.T
        print(f"\n[{bo}] {len(d)} câu · rổ {args.top} ảnh")
        for ten, k in cach:
            co = 0
            for j, r in enumerate(d.itertuples()):
                thu = np.argsort(-S[:, j])[:2000]
                ro = (ro_thuan(thu, vid, args.top) if k is None
                      else ro_theo_video(thu, vid, args.top, k))
                co += any(str(vid[i]) == str(r.video_id)
                          and int(r.frame_idx_min) <= int(meta.frame_idx.iat[i])
                          <= int(r.frame_idx_max) for i in ro)
            bang[ten].append((co, len(d)))
            print(f"  {ten:26s} {co:3d}/{len(d)} = {co / len(d):.3f}")

    print(f"\n=== GỘP cả ba bộ ({sum(n for _, n in bang[cach[0][0]])} câu) ===")
    for ten, _ in cach:
        c = sum(x for x, _ in bang[ten])
        n = sum(y for _, y in bang[ten])
        print(f"  {ten:26s} {c:3d}/{n} = {c / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
