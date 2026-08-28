# -*- coding: utf-8 -*-
"""Chèn keyframe LÁNG GIỀNG vào danh sách nộp có lãi không?

Chẩn đoán dẫn tới phép thử này: trong 33/81 câu CLIP chọn sai khung dù đã cho
đúng video, 70% chọn một khung có cos > 0,90 với khung đúng và 27% chọn đúng
khung kề ngay bên. Nó không nhìn nhầm chỗ, nó không phân biệt nổi hai khung
trông y hệt. Vậy thì mọi cách mô tả nội dung đều bó tay, còn thứ rẻ tiền chưa
thử là mua thêm vé số ngay cạnh chỗ đã đoán.

Danh sách nộp có 100 dòng, nên đây là bài toán chia chỗ: mỗi láng giềng chèn vào
đẩy một khung xếp hạng thấp hơn ra.

  python eval/danh_gia_langgieng.py --ban-kinh 0 1 2
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

KS = (1, 5, 20, 50, 100)
TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"


def diem_clip(texts, w_phu: float) -> np.ndarray:
    import open_clip
    import torch

    F = np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy")
    m, _, _ = open_clip.create_model_and_transforms(TEN, pretrained=PRE)
    tok = open_clip.get_tokenizer(TEN)
    m.eval()
    with torch.no_grad():
        v = m.encode_text(tok(list(texts))).float().numpy()
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    S = F @ v.T
    if w_phu > 0:
        import ensemble as ens
        S = np.stack([ens.ghep(S[:, i], t, w=w_phu)
                      for i, t in enumerate(texts)], axis=1)
    return S


def xen_langgieng(thu_tu, vid, ban_kinh: int, sau: int = 100):
    """Duyệt theo thứ hạng, mỗi khung kéo theo láng giềng cùng video của nó.

    Láng giềng là hàng liền kề trong metadata, đã sắp theo video rồi tới n, nên
    chỉ cần cộng trừ chỉ số và kiểm cùng video.
    """
    # Khung được xếp hạng phải giữ đúng chỗ của nó; láng giềng chỉ xếp sau.
    lech = [0] + [d for k in range(1, ban_kinh + 1) for d in (k, -k)]
    ra, da = [], set()
    for i in thu_tu:
        for d in lech:
            j = int(i) + d
            if j < 0 or j >= len(vid) or vid[j] != vid[int(i)] or j in da:
                continue
            da.add(j)
            ra.append(j)
            if len(ra) >= sau:
                return ra
    return ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="eval/queries_hcmc2026.csv")
    ap.add_argument("--ensemble", type=float, default=0.5)
    ap.add_argument("--ban-kinh", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cot", default=None)
    args = ap.parse_args()

    q = pd.read_csv(ROOT / args.bo)
    meta = corpus.load_metadata()
    cot = args.cot or ("text_en" if "text_en" in q.columns else "text_vi")
    print(f"{len(q)} câu · CLIP nhận cột {cot}")
    S = diem_clip(q[cot].tolist(), args.ensemble)
    vid = meta.video_id.to_numpy()
    fi = meta.frame_idx.to_numpy()

    sau = max(KS)
    print(f"\n{'ban kinh':<12}" + "".join(f"  R@{k:<4}" for k in KS) + "   FINAL")
    for bk in args.ban_kinh:
        hs = []
        for i, r in enumerate(q.itertuples()):
            thu_tu = np.argsort(-S[:, i], kind="stable")[:sau * (2 * bk + 1)]
            ds = xen_langgieng(thu_tu, vid, bk, sau)
            h = next((k for k, j in enumerate(ds, 1)
                      if vid[j] == r.video_id
                      and r.frame_idx_min <= fi[j] <= r.frame_idx_max), 10 ** 9)
            hs.append(h)
        h = np.array(hs)
        r = {k: float((h <= k).mean()) for k in KS}
        print(f"{bk:<12}" + "".join(f"  {r[k]:.3f}" for k in KS)
              + f"   {np.mean(list(r.values())):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
