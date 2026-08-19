# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui" / "search_ui.py"
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

SO_CAU = int(sys.argv[1]) if len(sys.argv) > 1 else 3
q = pd.read_csv(ROOT / "eval" / "queries_hcmc2026.csv").head(SO_CAU)


def cat(s, p):
    w = str(s).split()
    return " ".join(w[:max(1, round(len(w) * p))])


def buoc(at, nhan, viec):
    t0 = time.perf_counter()
    at = viec(at)
    dt = time.perf_counter() - t0
    loi = at.exception[0].value if at.exception else None
    print(f"    {'🔴' if loi else '  '} {nhan:38}{dt:>7.1f}s"
          + (f"  — {loi}" if loi else ""))
    return at, dt, loi


print(f"diễn tập {len(q)} truy vấn · nạp giao diện lần đầu…")
t0 = time.perf_counter()
at = AppTest.from_file(str(UI), default_timeout=900).run()
print(f"  nạp lần đầu (chỉ một lần cho cả buổi): {time.perf_counter()-t0:.0f}s\n")

tong, loi_tong = [], 0
for r in q.itertuples():
    print(f"  {r.query_id}")
    t_cau = 0.0
    for phut, phan in ((1, 0.2), (2, 0.4), (3, 0.6)):
        def tim(a, txt=cat(r.text_vi, phan)):
            a.text_input[0].set_value(txt)
            return a.run()
        at, dt, e = buoc(at, f"phút {phut}: dán gợi ý {int(phan*100)}% rồi tìm", tim)
        t_cau += dt
        loi_tong += bool(e)

        if phut == 2:
            # Đánh dấu 3 khung đầu — người thi tick trên lưới kết quả.
            def danh_dau(a):
                n = 0
                for c in a.checkbox:
                    if "Đúng hướng" in (c.label or "") and n < 3:
                        c.set_value(True)
                        n += 1
                return a.run()
            at, dt, e = buoc(at, "đánh dấu 3 khung rồi tìm lại", danh_dau)
            t_cau += dt
            loi_tong += bool(e)

    def luu(a):
        for b in a.button:
            if "Lưu vào kho" in (b.label or ""):
                b.click()
                return a.run()
        return a
    at, dt, e = buoc(at, "lưu vào kho bài nộp", luu)
    t_cau += dt
    loi_tong += bool(e)

    # Dọn đánh dấu trước câu sau, đúng như người thi phải làm.
    def don(a):
        for b in a.button:
            if (b.label or "").startswith("Xoá hết"):
                b.click()
                return a.run()
        return a
    at, _, _ = buoc(at, "xoá đánh dấu cho câu sau", don)

    print(f"    ── tổng thời gian MÁY cho câu này: {t_cau:.1f}s "
          f"({'ĐẠT' if t_cau < 60 else 'QUÁ CHẬM'}, ngân sách 5 phút)\n")
    tong.append(t_cau)

print(f"trung bình {sum(tong)/len(tong):.1f}s máy mỗi truy vấn "
      f"(chậm nhất {max(tong):.1f}s)")
print(f"{'🔴 có ' + str(loi_tong) + ' bước hỏng' if loi_tong else '✅ không bước nào hỏng'}")
print("\nGhi chú: đây là thời gian MÁY. Người còn phải đọc gợi ý, quét 50 ô,")
print("mở video ra xem — nên phần máy càng nhỏ càng tốt.")
sys.exit(1 if loi_tong else 0)
