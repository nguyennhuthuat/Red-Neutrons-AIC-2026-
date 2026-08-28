# -*- coding: utf-8 -*-
"""Đếm lặp: độ ĐỒNG THUẬN giữa các lần chạy có thay được confidence tự khai không?

confidence do VLM tự khai đã đo được là vô dụng (10/10 ở câu bịa, 0/10 ở câu
rổ hạng 1 — mục sec:haicakho). Tệp này kiểm một tín hiệu thay thế, đo được chứ
không tự khai: chạy cùng một câu đếm nhiều lần rồi xem chúng có khớp nhau không.

  python eval/danh_gia_dem_lap.py --lan 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import qa  # noqa: E402
from danh_gia_qa import trung  # noqa: E402

# Câu đếm KHÓ: vài chục ký hiệu nhỏ, không có nhãn khách quan nào.
# Chỉ dùng để đo độ ổn định, không chấm đúng/sai.
KHO = [
    ("map4", "L21_V006", 7, "chấm màu vàng nhạt ứng với mức 4.0-4.5 trên bản đồ"),
    ("mapall", "L21_V006", 7, "chấm tròn màu đánh dấu tâm chấn động đất trên bản đồ"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", type=int, default=3)
    ap.add_argument("--ra", default="")
    args = ap.parse_args()

    m = pd.read_parquet(ROOT / "data" / "processed_hcmc2026" / "metadata.parquet")
    ghi = []

    d = pd.read_csv(ROOT / "eval" / "queries_qa_dem.csv")
    print(f"[đếm DỄ] {len(d)} câu · {args.lan} lần mỗi câu")
    for r in d.itertuples():
        p = str(m[(m.video_id == r.video_id) & (m.n == int(r.n))].image_path.iat[0])
        vat = r.question_vi
        g = qa.dem_lap(vat, p, lan=args.lan, desc=r.text_vi)
        if not g:
            continue
        ok = trung(g["answer"], r.answer)
        print(f"  {r.query_id} {'✅' if ok else '❌'} lần chạy {g['cac_lan']} "
              f"→ {g['count']} · chuẩn '{r.answer}' · "
              f"{'ĐỒNG THUẬN' if g['on_dinh'] else 'lệch nhau'}")
        ghi.append({"id": r.query_id, "loai": "dễ", "cac_lan": g["cac_lan"],
                    "on_dinh": g["on_dinh"], "dung": ok})

    print(f"\n[đếm KHÓ] {len(KHO)} câu · không có nhãn, chỉ đo ổn định")
    for ten, v, n, vat in KHO:
        p = str(m[(m.video_id == v) & (m.n == n)].image_path.iat[0])
        g = qa.dem_lap(vat, p, lan=args.lan, desc="Bản đồ phân bố chấn tâm động đất")
        if not g:
            continue
        print(f"  {ten:7s} lần chạy {g['cac_lan']} → {g['count']} · "
              f"{'ĐỒNG THUẬN' if g['on_dinh'] else 'lệch nhau'}")
        ghi.append({"id": ten, "loai": "khó", "cac_lan": g["cac_lan"],
                    "on_dinh": g["on_dinh"], "dung": None})

    de = [x for x in ghi if x["loai"] == "dễ"]
    kho = [x for x in ghi if x["loai"] == "khó"]
    print(f"\n  đồng thuận ở câu DỄ : {sum(x['on_dinh'] for x in de)}/{len(de)}")
    print(f"  đồng thuận ở câu KHÓ: {sum(x['on_dinh'] for x in kho)}/{len(kho)}")
    dt = [x for x in de if x["on_dinh"]]
    if dt:
        print(f"  trong câu DỄ có đồng thuận, đúng: "
              f"{sum(x['dung'] for x in dt)}/{len(dt)}")
    if args.ra:
        Path(args.ra).write_text(json.dumps(ghi, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
