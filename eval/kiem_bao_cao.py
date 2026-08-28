# -*- coding: utf-8 -*-
"""Soi báo cáo LaTeX: nhãn khớp tham chiếu, môi trường cân, hàng bảng đủ dấu."""
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "docs" / "bao_cao_he_thong.tex"
BS = chr(92)
NGAT = ("midrule", "toprule", "bottomrule", "addlinespace", "cmidrule")


def hang_thieu_dau(d: str) -> list[str]:
    """Hàng bảng được phép xuống dòng giữa chừng, nên phải gom tới khi gặp dấu
    kết hàng rồi mới xét, chứ không xét từng dòng một."""
    ra, trong, gom, dau = [], False, "", 0
    for i, dong in enumerate(d.splitlines(), 1):
        t = dong.strip()
        if t.startswith(BS + "begin{tabular}"):
            trong, gom = True, ""
            continue
        het = t.startswith(BS + "end{tabular}")
        if het or any(t.startswith(BS + k) for k in NGAT):
            if "&" in gom:
                ra.append(f"dòng {dau}: hàng bảng thiếu dấu xuống hàng")
            gom = ""
            if het:
                trong = False
            continue
        if not trong or t.startswith("%") or not t:
            continue
        if not gom:
            dau = i
        gom += " " + t
        if t.endswith(BS * 2):
            gom = ""
    return ra


def main() -> int:
    d = D.read_text(encoding="utf-8")
    loi = []

    lab = set(re.findall(BS + BS + r"label\{([^}]+)\}", d))
    ref = set(re.findall(BS + BS + r"ref\{([^}]+)\}", d))
    print(f"nhãn {len(lab)} · tham chiếu {len(ref)}")
    if ref - lab:
        loi.append(f"tham chiếu treo: {sorted(ref - lab)}")
    if lab - ref:
        loi.append(f"nhãn không ai trỏ tới: {sorted(lab - ref)}")

    for moi in ("tabular", "center", "itemize", "enumerate"):
        b, e = d.count(BS + "begin{" + moi + "}"), d.count(BS + "end{" + moi + "}")
        print(f"  {moi:<10} {b:>4} / {e:<4} {'cân' if b == e else 'LỆCH'}")
        if b != e:
            loi.append(f"{moi} lệch {b} mở / {e} đóng")

    if "@@" in d:
        loi.append("còn ký hiệu vá @@ chưa thay")
    loi += hang_thieu_dau(d)

    print()
    for x in loi:
        print("  🔴", x)
    print("✅ báo cáo sạch" if not loi else f"{len(loi)} lỗi")
    return 1 if loi else 0


if __name__ == "__main__":
    sys.exit(main())
