# -*- coding: utf-8 -*-
"""Chấm theo THỨ TỰ có thật sự hơn nhồi-một-vector không? Đo mà không cần nhãn.

Không bộ đo nào có câu KIS nhiều cảnh (0/81, mục sec:chuoicanh), nên không đo
end-to-end được. Nhưng câu hỏi cốt lõi đo được mà chẳng cần nhãn nào: khi truy
vấn tả HAI khoảnh khắc khác nhau, gộp chúng thành một vector có dở hơn chấm
theo thứ tự không?

Dựng bài toán bằng chính đặc trưng ảnh: lấy hai khung A < B trong cùng video,
đủ khác nhau để là hai cảnh. Đưa cho hệ thống "hai cảnh" ấy rồi xem nó có tìm
lại được A không. Đáp án đúng biết trước theo cách dựng, nên không cần chấm tay.

Số tuyệt đối ở đây LẠC QUAN (vector ảnh tả cảnh chuẩn hơn câu chữ nhiều). Thứ
đáng tin là phần SO SÁNH giữa hai cách, vì cả hai ăn cùng một đầu vào.

  python eval/danh_gia_chuoi.py --so-cap 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chuoi  # noqa: E402
import corpus  # noqa: E402

KS = (1, 5, 10, 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-cap", type=int, default=300)
    ap.add_argument("--cach-min", type=float, default=5.0)
    ap.add_argument("--cach-max", type=float, default=60.0)
    ap.add_argument("--khac", type=float, default=0.85,
                    help="cos tối đa giữa hai khung — trên ngưỡng là cùng một cảnh")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = corpus.load_metadata()
    X = np.asarray(np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy",
                           mmap_mode="r"))
    vid = m.video_id.to_numpy()
    pts = m.pts_time.to_numpy().astype(np.float32)
    rng = np.random.default_rng(args.seed)

    dau = {}
    for i, v in enumerate(vid):
        dau.setdefault(v, []).append(i)
    vids = [v for v, ix in dau.items() if len(ix) >= 8]

    cap = []
    while len(cap) < args.so_cap:
        v = vids[rng.integers(len(vids))]
        ix = np.array(dau[v])
        a = int(ix[rng.integers(len(ix))])
        sau = ix[(pts[ix] > pts[a] + args.cach_min)
                 & (pts[ix] < pts[a] + args.cach_max)]
        if not len(sau):
            continue
        b = int(sau[rng.integers(len(sau))])
        if float(X[a] @ X[b]) > args.khac:
            continue                      # hai khung quá giống nhau: cùng một cảnh
        cap.append((a, b))

    print(f"[chuỗi] {len(cap)} cặp khung · cách {args.cach_min:.0f}-"
          f"{args.cach_max:.0f}s · cos < {args.khac}")
    hang = {"nhồi 1 vector": [], "chấm theo THỨ TỰ": []}
    for a, b in cap:
        gop = X[a] + X[b]
        gop /= np.linalg.norm(gop) + 1e-9
        s_gop = X @ gop
        S = np.stack([X @ X[a], X @ X[b]])
        s_thu = chuoi.diem_chuoi(S, vid, pts)
        for ten, s in (("nhồi 1 vector", s_gop), ("chấm theo THỨ TỰ", s_thu)):
            hang[ten].append(int(np.where(np.argsort(-s) == a)[0][0]) + 1)

    print(f"\n  {'cách chấm':20s} " + " ".join(f"R@{k:<3d}" for k in KS)
          + "  hạng trung vị")
    for ten, hs in hang.items():
        h = np.array(hs)
        print(f"  {ten:20s} " + " ".join(f"{(h <= k).mean():.3f}" for k in KS)
              + f"     {int(np.median(h))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
