# -*- coding: utf-8 -*-
"""Đo kênh THẺ CUỐI: gõ tên nguyên liệu, chỉ mục có ra đúng video không.

Bộ đo Q&A chữ nhỏ KHÔNG đo được kênh này — câu hỏi ở đó cố tình không chứa từ
khoá nguyên liệu để khỏi lộ đáp án. Nên phải có phép đo riêng, dựng sao cho
không tự khen: lấy k tên nguyên liệu CÓ THẬT trong thẻ của một video, ghép thành
truy vấn, xem chỉ mục có đưa đúng video ấy lên đầu không. Nhãn đúng do CÁCH DỰNG,
không do mô hình nào phán.

  python eval/danh_gia_thecuoi.py            # 40 truy vấn, 3 nguyên liệu
  python eval/danh_gia_thecuoi.py --tu 1     # chỉ 1 nguyên liệu, khó hơn hẳn
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ocr_thecuoi as tc  # noqa: E402

# Credit của nhà sản xuất lặp ở gần như mọi thẻ (97-122/125 khi đo), còn tên
# nguyên liệu chỉ ở 1-10 thẻ. Hai tín hiệu này tách được chúng mà không chỉnh tay.
CREDIT = ("cong ty", "congty", "chuong trinh", "chuongtrinh", "truyen hinh",
          "truyenhinh", "thuc hien", "thuchien", "ban quyen", "banquyen",
          "thanh pho", "thanhpho", "tham gia", "thamgia", "phoi hop",
          "phoihop", "nguyen lieu", "nguyenlieu", "golden scale", "hkfilm",
          "ajinomoto", "hakuhodo", "kich ban", "kichban")


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s.replace("đ", "d")).strip()


def ung_vien(text: str) -> set[str]:
    """Dòng không chứa số, dài vừa phải, không phải credit."""
    ra = set()
    for dong in str(text).splitlines():
        d = bo_dau(dong)
        if (5 <= len(d) <= 30 and not any(c.isdigit() for c in d)
                and not any(k in d for k in CREDIT)):
            ra.add(d)
    return ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="số truy vấn")
    ap.add_argument("--tu", type=int, default=3, help="số nguyên liệu mỗi truy vấn")
    ap.add_argument("--tran-df", type=float, default=0.15,
                    help="bỏ từ có mặt ở quá tỉ lệ này của kho")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    b = tc.bang()
    if b is None:
        print("chưa có chỉ mục — chạy `python src/ocr_thecuoi.py` trước")
        return 1
    n = len(b)
    rng = random.Random(args.seed)

    df: Counter[str] = Counter()
    uv_the: dict[str, set[str]] = {}
    for r in b.itertuples():
        uv = ung_vien(r.text)
        uv_the[r.video_id] = uv
        df.update(uv)

    du = [(v, sorted(u for u in uv if df[u] <= args.tran_df * n))
          for v, uv in uv_the.items()]
    du = [(v, nl) for v, nl in du if len(nl) >= args.tu]
    print(f"[thẻ cuối] {n} thẻ · {len(du)} thẻ có đủ {args.tu} nguyên liệu "
          f"riêng biệt\n")
    if not du:
        return 1

    mau = rng.sample(du, min(args.n, len(du)))
    h1 = h5 = truot = 0
    for i, (vid, nl) in enumerate(mau, 1):
        tv = " ".join(rng.sample(nl, args.tu))
        kq = tc.tim(tv, limit=5)
        vs = list(kq.video_id) if len(kq) else []
        h = vs.index(vid) + 1 if vid in vs else 0
        h1 += h == 1
        h5 += 1 <= h <= 5
        truot += h == 0
        if i <= 8 or h != 1:
            print(f"{i:>3} {vid} hạng {h or '>5'}  ← {tv[:64]}")

    m = len(mau)
    print(f"\n{'=' * 58}\n{m} truy vấn · {args.tu} nguyên liệu mỗi câu")
    print(f"  hạng 1 : {h1}/{m} = {h1 / m:.3f}")
    print(f"  top-5  : {h5}/{m} = {h5 / m:.3f}")
    print(f"  trượt  : {truot}/{m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
