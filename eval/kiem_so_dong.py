# -*- coding: utf-8 -*-
"""Số dòng ghi trong phụ lục A phải khớp tệp thật.

Phụ lục điểm mặt từng tệp kèm số dòng. Mỗi lần sửa mã mà quên sửa con số ấy là
báo cáo tự nói dối một chút, nên để máy soi thay vì để mắt người.

  python eval/kiem_so_dong.py         # soi
  python eval/kiem_so_dong.py --sua   # soi rồi vá luôn
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs" / "bao_cao_he_thong.tex"
BS = chr(92)
MAU = re.compile(BS + BS + r"texttt\{([^}]+\.py)\} \((\d+) dòng\)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sua", action="store_true")
    args = ap.parse_args()

    d = D.read_text(encoding="utf-8")
    lech, thieu = [], []
    for m in MAU.finditer(d):
        duong = m.group(1).replace(BS, "")
        f = ROOT / duong
        if not f.exists():
            thieu.append(duong)
            continue
        that = len(f.read_text(encoding="utf-8").splitlines())
        if int(m.group(2)) != that:
            lech.append((duong, int(m.group(2)), that, m.group(0)))

    for duong, ghi, that, _ in lech:
        print(f"  🔴 {duong:<28} ghi {ghi:>5} · thật {that:>5}")
    for duong in thieu:
        print(f"  🔴 {duong:<28} không có tệp này")

    if args.sua and lech:
        for duong, ghi, that, cu in lech:
            d = d.replace(cu, cu.replace(f"({ghi} dòng)", f"({that} dòng)"))
        D.write_text(d, encoding="utf-8")
        print(f"\n✅ vá {len(lech)} chỗ")
        return 0

    print("✅ số dòng khớp hết" if not (lech or thieu)
          else f"{len(lech) + len(thieu)} chỗ lệch")
    return 1 if (lech or thieu) else 0


if __name__ == "__main__":
    sys.exit(main())
