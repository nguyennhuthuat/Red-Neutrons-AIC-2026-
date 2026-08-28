# -*- coding: utf-8 -*-
"""Kênh chữ trên khung có bù được chỗ CLIP mù không?

Đo trên đúng những câu thuộc nhóm đã lập chỉ mục, ba cấu hình cạnh nhau: CLIP
một mình, chữ một mình, và gộp hai bảng xếp hạng bằng RRF. Câu truy vấn dùng
`text_vi` -- đúng thứ người thi gõ, không phải đáp án.

  python eval/danh_gia_ocr_chu.py --bo eval/queries_qa_chunho.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import corpus  # noqa: E402
import ocr_chu  # noqa: E402

KS = (1, 5, 20, 100)
TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"


def diem_clip(texts, w_phu: float):
    """Điểm CLIP trên toàn kho cho từng câu, kèm encoder phụ nếu bật."""
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


def hang_ocr(cau: str, khoa, n_kf: int) -> np.ndarray:
    """Điểm chữ dàn về mọi khung của kho; khung chưa quét thì bằng 0."""
    v = np.zeros(n_kf, dtype=np.float32)
    kq = ocr_chu.tim(cau, limit=2000)
    if len(kq):
        vi = [khoa.get((r.video_id, int(r.n))) for r in kq.itertuples()]
        for j, d in zip(vi, kq.diem):
            if j is not None:
                v[j] = float(d)
    return v


def rrf(*hang, k: int = 60):
    """Gộp thứ hạng thay vì gộp điểm: hai thang điểm không so được với nhau."""
    return sum(1.0 / (k + h) for h in hang)


def thu_hang(diem: np.ndarray) -> np.ndarray:
    """Thứ hạng 1-based theo điểm giảm dần."""
    o = np.argsort(-diem, kind="stable")
    h = np.empty(len(diem), dtype=np.int64)
    h[o] = np.arange(1, len(diem) + 1)
    return h



def do_dungchu(so_cau: int, moi_cau: int, hat: int, nhom=None):
    """Nhãn đúng do CÁCH DỰNG: lấy từ CÓ THẬT trên khung làm truy vấn.

    Đây là cách người thi thật sự dùng kênh chữ -- gõ chữ họ đoán có trên màn
    hình, không gõ mô tả cảnh. Đo bằng mô tả thì kênh này thua đứt, vì lời văn
    và chữ chồng trên khung gần như không chung từ nào.
    """
    import math
    import random

    b = ocr_chu.bang()
    kho = ocr_chu._kho()
    n = len(b)
    # Bỏ từ có mặt ở hơn 15% kho: tên đài, tên chương trình, chữ luôn nằm đó.
    df = {}
    rng = random.Random(hat)
    hop = b.ky_tu >= 30
    if nhom:
        hop &= b.video_id.str[:3].isin(list(nhom))
    ung = b.index[hop].tolist()
    rng.shuffle(ung)

    hang_khung, hang_video, hang_chu, bo = [], [], [], 0
    for i in ung:
        tu = [w for w in dict.fromkeys(kho[i].split()) if len(w) >= 4]
        for w in tu:
            if w not in df:
                df[w] = int(kho.str.contains(w, regex=False).sum())
        hiem = sorted((w for w in tu if df[w] / n <= 0.15),
                      key=lambda w: df[w])[:moi_cau]
        if len(hiem) < moi_cau:
            bo += 1
            continue
        kq = ocr_chu.tim(" ".join(hiem), limit=100)
        vt = {int(x.Index): k for k, x in enumerate(kq.itertuples(), 1)}
        hang_khung.append(vt.get(int(i), 10 ** 9))
        vid = b.video_id[i]
        hang_video.append(next((k for k, x in enumerate(kq.itertuples(), 1)
                                if x.video_id == vid), 10 ** 9))
        # Khung mang ĐÚNG cùng chuỗi chữ thì không truy vấn chữ nào tách nổi.
        hang_chu.append(next((k for k, x in enumerate(kq.itertuples(), 1)
                              if kho[int(x.Index)] == kho[i]), 10 ** 9))
        if len(hang_khung) >= so_cau:
            break

    print(f"{len(hang_khung)} truy vấn · {moi_cau} từ mỗi câu · "
          f"bỏ {bo} khung không đủ từ hiếm")
    print(f"{'muc tieu':<12}" + "".join(f"  R@{k:<5}" for k in KS))
    for ten, hs in (("đúng khung", hang_khung), ("khung cùng chữ", hang_chu),
                    ("đúng video", hang_video)):
        h = np.array(hs)
        print(f"{ten:<12}" + "".join(f"  {(h <= k).mean():.3f} " for k in KS))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="eval/queries_qa_chunho.csv")
    ap.add_argument("--ensemble", type=float, default=0.5)
    ap.add_argument("--cach", default="gop", choices=["gop", "dungchu"])
    ap.add_argument("--so-cau", type=int, default=60)
    ap.add_argument("--moi-cau", type=int, default=3)
    ap.add_argument("--hat", type=int, default=2026)
    ap.add_argument("--nhom", nargs="+", default=None)
    args = ap.parse_args()

    b = ocr_chu.bang()
    if b is None or not len(b):
        print("chưa có chỉ mục chữ — chạy src/ocr_chu.py trước")
        return 1
    nhom_co = sorted(b.video_id.str[:3].unique())
    print(f"chỉ mục chữ: {len(b):,} khung · nhóm {' '.join(nhom_co)}")
    if args.cach == "dungchu":
        return do_dungchu(args.so_cau, args.moi_cau, args.hat,
                          args.nhom)

    q = pd.read_csv(ROOT / args.bo)
    q = q[q.video_id.str[:3].isin(nhom_co)].reset_index(drop=True)
    print(f"bộ đo: {len(q)} câu thuộc các nhóm ấy\n")
    if not len(q):
        return 1

    meta = corpus.load_metadata()
    khoa = {(v, int(n)): i for i, (v, n)
            in enumerate(zip(meta.video_id, meta.n))}
    S = diem_clip(q.text_vi.tolist(), args.ensemble)

    dung = ((meta.video_id.to_numpy()[:, None] == q.video_id.to_numpy())
            & (meta.frame_idx.to_numpy()[:, None] >= q.frame_idx_min.to_numpy())
            & (meta.frame_idx.to_numpy()[:, None] <= q.frame_idx_max.to_numpy()))

    ket = {"CLIP": [], "chữ": [], "gộp RRF": []}
    for i, r in enumerate(q.itertuples()):
        v_ocr = hang_ocr(r.text_vi, khoa, len(meta))
        h_clip, h_ocr = thu_hang(S[:, i]), thu_hang(v_ocr)
        cot = dung[:, i]
        ket["CLIP"].append(int(h_clip[cot].min()))
        ket["chữ"].append(int(h_ocr[cot].min()) if v_ocr.any() else 10 ** 9)
        ket["gộp RRF"].append(int(thu_hang(rrf(h_clip, h_ocr))[cot].min()))
        print(f"  {r.query_id} {r.video_id:<9} CLIP {ket['CLIP'][-1]:>7,} · "
              f"chữ {ket['chữ'][-1]:>7,} · gộp {ket['gộp RRF'][-1]:>7,}  "
              f"| {str(r.answer)[:22]}")

    print(f"\n{'cách':<10}" + "".join(f"  R@{k:<5}" for k in KS))
    for ten, hs in ket.items():
        h = np.array(hs)
        print(f"{ten:<10}" + "".join(f"  {(h <= k).mean():.3f} " for k in KS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
