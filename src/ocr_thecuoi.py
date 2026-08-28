# -*- coding: utf-8 -*-
"""Lập chỉ mục chữ cho THẺ TÓM TẮT cuối mỗi video nấu ăn L26.

Quét OCR cả corpus là 140 giờ nên không làm được. Nhưng L26 có một quy luật cứng:
đo trên 40 video ngẫu nhiên, 40/40 keyframe CUỐI là bảng nguyên liệu. Chỉ quét
đúng khung ấy thì 498 video mất chừng 23 phút, mà lại phủ đúng chỗ CLIP mù —
"bảng nguyên liệu trên nền gỗ" là mô tả khớp hàng trăm video như nhau.

  python src/ocr_thecuoi.py            # quét và ghi parquet
  python src/ocr_thecuoi.py --thu 40   # chỉ thử 40 video
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import corpus  # noqa: E402

RA = ROOT / "data" / "processed_hcmc2026" / "ocr_thecuoi.parquet"
NHOM = ("L26",)
TIN = 0.6


def quet(nhom=NHOM, thu: int = 0, moi_video: int = 1, ra: Path = RA) -> int:
    from rapidocr_onnxruntime import RapidOCR

    meta = pd.read_parquet(ROOT / "data" / "processed_hcmc2026"
                           / "metadata.parquet")
    meta["image_path"] = corpus.resolve_paths(meta["image_path"])
    meta = meta[meta.video_id.str[:3].isin(nhom)]

    # Lấy `moi_video` keyframe cuối của từng video.
    cuoi = (meta.sort_values(["video_id", "n"])
                .groupby("video_id").tail(moi_video))
    if thu:
        vids = sorted(cuoi.video_id.unique())[:thu]
        cuoi = cuoi[cuoi.video_id.isin(vids)]

    ocr = RapidOCR(det_limit_side_len=736)
    dong, t0 = [], time.time()
    for i, r in enumerate(cuoi.itertuples(), 1):
        try:
            kq, _ = ocr(r.image_path)
        except Exception as e:
            print(f"  {r.video_id}/{r.frame_idx}: {type(e).__name__}", flush=True)
            continue
        mau = [c for c in (kq or []) if float(c[2]) >= TIN]
        mau.sort(key=lambda c: (min(p[1] for p in c[0]),
                                min(p[0] for p in c[0])))
        dong.append({"video_id": r.video_id, "n": int(r.n),
                     "frame_idx": int(r.frame_idx),
                     "text": "\n".join(c[1] for c in mau),
                     "n_vung": len(mau)})
        if i % 25 == 0:
            giay = time.time() - t0
            print(f"  {i}/{len(cuoi)} · {giay:.0f}s · còn "
                  f"{giay / i * (len(cuoi) - i) / 60:.0f} phút", flush=True)
            pd.DataFrame(dong).to_parquet(ra, index=False)

    pd.DataFrame(dong).to_parquet(ra, index=False)
    print(f"\n{len(dong)} thẻ · {time.time() - t0:.0f}s · {ra.name}")
    return len(dong)


_BANG = None


def bang():
    """Bảng chữ thẻ cuối, nạp một lần. Trả None nếu chưa lập chỉ mục."""
    global _BANG
    if _BANG is None:
        if not RA.exists():
            return None
        _BANG = pd.read_parquet(RA)
    return _BANG


def _bo_dau(s):
    import re
    import unicodedata

    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s.replace("đ", "d")).strip()


_KHO = _KHO_LIEN = None


def _kho():
    """Chữ của mọi thẻ, đã bỏ dấu. OCR hay dính chữ nên tra bằng chuỗi con."""
    global _KHO
    if _KHO is None:
        b = bang()
        _KHO = None if b is None else b.text.map(_bo_dau)
    return _KHO


def _kho_lien():
    """Bản bỏ hết khoảng trắng, để tra những mảnh OCR dán liền."""
    global _KHO_LIEN
    if _KHO_LIEN is None:
        k = _kho()
        _KHO_LIEN = None if k is None else k.str.replace(" ", "", regex=False)
    return _KHO_LIEN


def _manh_lien(tu):
    """Các mảnh dán liền đáng tra: cả câu, rồi từng cặp từ kề nhau.

    OCR trả "CHANG4" cho chữ "CHẶNG 4" trên màn hình. Tra theo từng từ thì mảnh
    phân biệt được rơi đúng vào chỗ bị cắt đôi: "chang" trúng hàng trăm khung,
    "4" trúng gần hết kho, mà "chang4" thì gần như chỉ trúng một chỗ.
    """
    ra = []
    if len(tu) > 1:
        ra.append("".join(tu))
        ra += ["".join(tu[i:i + 2]) for i in range(len(tu) - 1)]
    return [m for m in dict.fromkeys(ra) if len(m) >= 4]


def tim(tu_khoa: str, limit: int = 20):
    """Tìm thẻ tóm tắt chứa từ khoá, chấm theo nghịch tần suất tài liệu.

    Đếm số từ khớp thì hỏng: thẻ nào cũng kèm credit giống hệt nhau nên những từ
    ấy cộng đều cho mọi thẻ. Từ càng hiếm trong kho càng đáng điểm.
    """
    import math

    b = bang()
    kho, lien = _kho(), _kho_lien()
    if b is None or kho is None or not str(tu_khoa).strip():
        return pd.DataFrame()
    tu_tho = _bo_dau(tu_khoa).split()
    tu = [w for w in tu_tho if len(w) > 1]
    if not tu_tho:
        return pd.DataFrame()

    n = len(b)
    diem = 0
    for w, kho_w in ([(w, kho) for w in dict.fromkeys(tu)]
                     + [(m, lien) for m in _manh_lien(tu_tho)]):
        co = kho_w.str.contains(w, regex=False)
        df = int(co.sum())
        if df == 0:
            continue
        diem = diem + co.astype(float) * math.log(n / df)
    if not hasattr(diem, "__len__"):
        return pd.DataFrame()
    ra = b.assign(diem=diem.round(3))
    return ra[ra.diem > 0].sort_values("diem", ascending=False).head(limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu", type=int, default=0, help="chỉ quét N video đầu")
    ap.add_argument("--moi-video", type=int, default=1,
                    help="số keyframe cuối lấy mỗi video")
    args = ap.parse_args()
    raise SystemExit(0 if quet(thu=args.thu, moi_video=args.moi_video) else 1)
