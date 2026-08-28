# -*- coding: utf-8 -*-
"""Đã biết đúng video rồi, chọn đúng KHUNG được bao nhiêu?

Đây là khe đã đo của hệ thống: CLIP tìm đúng video 0,988 mà đúng khung chỉ 0,654.
Tab Q&A vốn đã tách làm hai bước nên "biết đúng video" gần như miễn phí trong
phòng thi. Câu hỏi còn lại là ở bước hai, kênh chữ có bù được cho CLIP không --
gộp toàn kho thì đã thua (mục sec:ocrchu), nhưng trong một video thì nhiễu của
872 video kia không còn.

  python eval/danh_gia_trongvideo.py --bo eval/queries_qa_chunho.csv
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

KS = (1, 3, 5, 10)
TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"
RRF_K = 10


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


def hang(diem: np.ndarray) -> np.ndarray:
    o = np.argsort(-diem, kind="stable")
    h = np.empty(len(diem), dtype=np.int64)
    h[o] = np.arange(1, len(diem) + 1)
    return h


def loc_trung(thu_tu, X, tau: float):
    """Giữ theo thứ hạng, bỏ khung quá giống một khung đã giữ.

    Trong một video, một phần ba cặp keyframe kề nhau có cos > 0,95 -- rổ 5 ảnh
    dễ thành 5 lần cùng một khoảnh khắc. Lọc trùng đổi chúng lấy 5 khoảnh khắc
    khác nhau, mà đúng khoảnh khắc mới là thứ câu hỏi Q&A phân biệt.
    """
    if tau >= 1.0:
        return [int(i) for i in thu_tu]
    giu = []
    for i in thu_tu:
        if giu and float(np.max(X[giu] @ X[i])) > tau:
            continue
        giu.append(int(i))
    return giu


def do_loctrung(bo: str, w_phu: float, taus):
    b = pd.read_csv(ROOT / bo)
    meta = corpus.load_metadata()
    F = np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy",
                mmap_mode="r")
    S = diem_clip(b.text_vi.tolist(), w_phu)
    print(f"{len(b)} câu · lọc trùng trong video")

    ket, con = {}, {}
    for tau in taus:
        hs, ti = [], []
        for i, r in enumerate(b.itertuples()):
            trong = np.flatnonzero(meta.video_id.to_numpy() == r.video_id)
            X = np.asarray(F[trong], dtype=np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            thu_tu = np.argsort(-S[trong, i], kind="stable")
            giu = loc_trung(thu_tu, X, tau)
            fi = meta.frame_idx.to_numpy()[trong]
            hs.append(next((k for k, j in enumerate(giu, 1)
                            if r.frame_idx_min <= fi[j] <= r.frame_idx_max),
                           10 ** 9))
            ti.append(len(giu) / len(trong))
        ket[tau] = np.array(hs)
        con[tau] = float(np.mean(ti))

    print()
    print(f"{'nguong':<10}" + "".join(f"  R@{k:<3}" for k in KS) + "   con lai")
    for tau, h in ket.items():
        ten = "tat" if tau >= 1.0 else f"{tau}"
        print(f"{ten:<10}" + "".join(f"  {(h <= k).mean():.3f}" for k in KS)
              + f"   {con[tau]:.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="eval/queries_qa_chunho.csv")
    ap.add_argument("--ensemble", type=float, default=0.5)
    ap.add_argument("--cau", default=None,
                    help="cột cho kênh chữ; mặc định dùng answer nếu có")
    ap.add_argument("--w", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--loc-trung", type=float, nargs="*", default=None,
                    metavar="TAU", help="đo lọc trùng thay vì gộp chữ")
    args = ap.parse_args()

    if args.loc_trung is not None:
        return do_loctrung(args.bo, args.ensemble,
                           args.loc_trung or [1.0, 0.98, 0.95, 0.92, 0.88])
    b = ocr_chu.bang()
    nhom = set(b.video_id.str[:3]) if b is not None else set()
    q = pd.read_csv(ROOT / args.bo)
    q = q[q.video_id.str[:3].isin(nhom)].reset_index(drop=True)
    if not len(q):
        print("không câu nào thuộc nhóm đã lập chỉ mục")
        return 1
    cot_chu = args.cau or ("answer" if "answer" in q.columns else "text_vi")
    print(f"{len(q)} câu · kênh chữ nhận cột {cot_chu}")

    meta = corpus.load_metadata()
    S = diem_clip(q.text_vi.tolist(), args.ensemble)
    chu_khoa = {(v, int(n)): i for i, (v, n)
                in enumerate(zip(b.video_id, b.n))}

    ket = {"CLIP trong video": []}
    for w in args.w:
        ket[f"+ chữ w={w}"] = []

    for i, r in enumerate(q.itertuples()):
        trong = np.flatnonzero(meta.video_id.to_numpy() == r.video_id)
        dung = ((meta.frame_idx.to_numpy()[trong] >= r.frame_idx_min)
                & (meta.frame_idx.to_numpy()[trong] <= r.frame_idx_max))
        h_clip = hang(S[trong, i])
        ket["CLIP trong video"].append(int(h_clip[dung].min()))

        # Điểm chữ lấy từ chỉ mục toàn kho rồi cắt về đúng video, để trọng số
        # nghịch tần suất vẫn tính trên cả kho chứ không trên vài trăm khung.
        kq = ocr_chu.tim(str(getattr(r, cot_chu)), limit=100000)
        diem = np.zeros(len(trong), dtype=np.float32)
        if len(kq):
            vt = {int(khoa): d for khoa, d in
                  zip(kq.index, kq.diem)}
            for k, j in enumerate(trong):
                idx = chu_khoa.get((meta.video_id.iat[j], int(meta.n.iat[j])))
                if idx is not None and idx in vt:
                    diem[k] = vt[idx]
        h_chu = hang(diem)
        for w in args.w:
            tong = 1.0 / (RRF_K + h_clip) + w / (RRF_K + h_chu)
            ket[f"+ chữ w={w}"].append(int(hang(tong)[dung].min()))

    print(f"\n{'cách':<20}" + "".join(f"  R@{k:<3}" for k in KS))
    for ten, hs in ket.items():
        h = np.array(hs)
        print(f"{ten:<20}" + "".join(f"  {(h <= k).mean():.3f}" for k in KS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
