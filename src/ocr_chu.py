# -*- coding: utf-8 -*-
"""Chỉ mục chữ cho phần kho đáng quét, chọn theo nhóm chương trình.

Kết luận cũ "không lập chỉ mục OCR nổi" dựa trên hai con số đều sai. Thứ nhất,
ép mỗi thợ dùng đúng một luồng ONNX rồi chạy 8 thợ thì nhanh gấp 3,9 lần so với
một tiến trình ôm cả 22 lõi -- 0,46 s/khung thay vì 1,80. Thứ hai, chữ trên
khung không phải logo đài mà là nội dung: bản tin in nguyên dòng tin chạy, thể
thao in chặng và vận tốc, ôn thi in cả đề bài.

Mật độ chữ chênh nhau rất xa giữa các nhóm (94% ở L22 xuống 1% ở L24) nên quét
theo thứ tự mật độ: bốn nhóm đầu là 32% kho mà chỉ tốn 7,3 giờ.

  python src/ocr_chu.py --nhom L23           # 18 phút
  python src/ocr_chu.py --nhom L21 L22 L23   # 2,5 giờ
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import corpus  # noqa: E402
from ocr_thecuoi import _bo_dau, _manh_lien  # noqa: E402

RA = ROOT / "data" / "processed_hcmc2026" / "ocr_chu.parquet"
TIN = 0.6
SIDE = 736

# Xếp theo mật độ chữ đo trên mẫu 900 khung trải đều 10 nhóm.
UU_TIEN = ("L22", "L21", "L25", "L23", "L26", "L30", "L28", "L27", "L29", "L24")

_ocr = None


def _quet_mot(cong):
    """Một khung trong một thợ riêng. Mỗi thợ đúng 1 luồng ONNX, nếu không 8 thợ
    giành nhau 22 lõi và chậm hơn cả chạy một mình."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR(det_limit_side_len=SIDE, intra_op_num_threads=1,
                        inter_op_num_threads=1)
    vid, n, fi, duong = cong
    try:
        kq, _ = _ocr(duong)
    except Exception:
        return None
    mau = [c for c in (kq or []) if float(c[2]) >= TIN]
    mau.sort(key=lambda c: (min(p[1] for p in c[0]), min(p[0] for p in c[0])))
    chu = "\n".join(c[1] for c in mau)
    return {"video_id": vid, "n": int(n), "frame_idx": int(fi),
            "text": chu, "n_vung": len(mau), "ky_tu": len(chu)}


def quet(nhom, tho: int = 8, thu: int = 0, ra: Path = RA) -> int:
    """Quét các nhóm đã chọn, bỏ qua khung đã có trong chỉ mục để chạy tiếp được."""
    meta = corpus.load_metadata()
    meta["image_path"] = corpus.resolve_paths(meta["image_path"])
    meta = meta[meta.video_id.str[:3].isin(list(nhom))]
    if thu:
        meta = meta.head(thu)

    cu = pd.read_parquet(ra) if ra.exists() else None
    if cu is not None and len(cu):
        xong = set(zip(cu.video_id, cu.n))
        meta = meta[[(v, int(n)) not in xong
                     for v, n in zip(meta.video_id, meta.n)]]
    if not len(meta):
        print("không còn khung nào cần quét")
        return 0

    cong = [(r.video_id, r.n, r.frame_idx, r.image_path)
            for r in meta.itertuples()]
    print(f"{len(cong):,} khung · {tho} thợ · ước "
          f"{len(cong) * 0.46 / 3600:.1f} giờ", flush=True)

    dong, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=tho) as ex:
        for i, r in enumerate(ex.map(_quet_mot, cong, chunksize=8), 1):
            if r is not None:
                dong.append(r)
            if i % 2000 == 0:
                giay = time.time() - t0
                print(f"  {i:,}/{len(cong):,} · {giay / 60:.0f} phút · còn "
                      f"{giay / i * (len(cong) - i) / 60:.0f} phút", flush=True)
                _ghi(cu, dong, ra)

    _ghi(cu, dong, ra)
    print(f"\n{len(dong):,} khung mới · {(time.time() - t0) / 60:.0f} phút · "
          f"{ra.name}")
    return len(dong)


def _ghi(cu, dong, ra: Path):
    d = pd.DataFrame(dong)
    if cu is not None and len(cu):
        d = pd.concat([cu, d], ignore_index=True)
    d.to_parquet(ra, index=False)


_BANG = _KHO = _KHO_LIEN = None


def bang():
    """Bảng chữ đã quét, nạp một lần. Trả None nếu chưa lập chỉ mục."""
    global _BANG
    if _BANG is None:
        if not RA.exists():
            return None
        _BANG = pd.read_parquet(RA)
    return _BANG


def _kho():
    global _KHO
    if _KHO is None:
        b = bang()
        _KHO = None if b is None else b.text.map(_bo_dau)
    return _KHO


def _kho_lien():
    """Bản bỏ hết khoảng trắng. OCR hay dán chữ liền: chữ trên màn hình là
    "CHẶNG 4" mà máy trả về "CHANG4", nên tra theo từng từ thì mảnh phân biệt
    được lại rơi đúng vào chỗ bị cắt đôi."""
    global _KHO_LIEN
    if _KHO_LIEN is None:
        k = _kho()
        _KHO_LIEN = None if k is None else k.str.replace(" ", "", regex=False)
    return _KHO_LIEN


def tim(tu_khoa: str, limit: int = 50, nhom=None):
    """Tìm khung chứa từ khoá, chấm theo nghịch tần suất tài liệu.

    Đếm từ khớp thì hỏng: mọi khung bản tin đều mang tên đài và giờ phát nên
    những từ ấy cộng đều cho tất cả. Từ càng hiếm trong kho càng đáng điểm.
    """
    import math

    b, kho = bang(), _kho()
    if b is None or kho is None or not str(tu_khoa).strip():
        return pd.DataFrame()
    tu_tho = _bo_dau(tu_khoa).split()
    tu = [w for w in tu_tho if len(w) > 1]
    if not tu_tho:
        return pd.DataFrame()

    lien = _kho_lien()
    if nhom is None:
        bb, kk, ll = b, kho, lien
    else:
        loc = b.video_id.str[:3].isin(list(nhom))
        bb, kk, ll = b[loc], kho[loc], lien[loc]
    n = max(1, len(bb))
    diem = 0
    for w, kho_w in ([(w, kk) for w in dict.fromkeys(tu)]
                     + [(m, ll) for m in _manh_lien(tu_tho)]):
        co = kho_w.str.contains(w, regex=False)
        df = int(co.sum())
        if df == 0:
            continue
        diem = diem + co.astype(float) * math.log(n / df)
    if not hasattr(diem, "__len__"):
        return pd.DataFrame()
    ra = bb.assign(diem=diem.round(3))
    return ra[ra.diem > 0].sort_values("diem", ascending=False).head(limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhom", nargs="+", default=["L23"])
    ap.add_argument("--tho", type=int, default=8)
    ap.add_argument("--thu", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(0 if quet(args.nhom, tho=args.tho, thu=args.thu) else 1)
