# -*- coding: utf-8 -*-
"""Soi lại khung sắp nộp: chi tiết nào trong đề KHÔNG nhìn thấy trong ảnh.

Nguồn lỗi lớn nhất đo được ở phòng thi không phải mô hình mà là người chọn
nhầm khung: soi tay 18 khung KIS của đề thật vòng 2 thì 5 khung sai hẳn và 3
khung đúng đoạn nhưng sai khoảnh khắc --- 8/18.

Đo trên đúng 18 khung ấy, một lời gọi VLM:
  * làm CHUÔNG BÁO thì bắt đủ 8/8 khung có vấn đề, nhưng báo nhầm 5/10 khung
    tốt --- nửa số khung đúng bị hét lên, không dùng làm cổng chặn được;
  * làm DANH SÁCH CHI TIẾT thì chính xác: nó chỉ ra p2-25 mặc áo tối chứ không
    phải sơ mi kẻ sọc, p2-18 không có đèn đếm ngược, p2-10 là bắp non chứ
    không phải bông hẹ --- toàn thứ người soi dễ lướt qua.

Nên tệp này trả về DANH SÁCH, còn phán quyết tổng chỉ là gợi ý phụ.
"""
from __future__ import annotations

import os
import pathlib
import re

MAX_CHI_TIET = 12

NHAC = """Bạn đang kiểm tra một bài nộp cho cuộc thi tìm kiếm video.

Dưới đây là CÂU MÔ TẢ cảnh cần tìm, và MỘT KHUNG HÌNH mà đội định nộp.
Hãy kiểm tra khung hình này có đúng là cảnh được mô tả không.

Cách làm:
1. Liệt kê từng chi tiết cụ thể mà câu mô tả nêu ra (vật, màu, hành động, chữ).
2. Với mỗi chi tiết, ghi CO nếu nhìn thấy trong ảnh, KHONG nếu không thấy.
3. Kết luận bằng đúng một trong ba từ ở dòng cuối, viết hoa:
   DUNG   - mọi chi tiết chính đều thấy, đây đúng là cảnh được tả
   THIEU  - đúng bối cảnh/đoạn phim nhưng thiếu hành động hoặc chi tiết chính
   SAI    - khung này không phải cảnh được tả

Đừng suy đoán từ bối cảnh chung. Chỉ tính chi tiết THẬT SỰ nhìn thấy.

CÂU MÔ TẢ:
"""

# Đầu dòng có thể là gạch, chấm tròn, HOẶC số thứ tự — model đổi kiểu tuỳ lúc.
# Cùng lớp lỗi đã dính ở goiten._tach: chỉ bắt gạch thì mất 8/18 phán quyết.
_DONG = re.compile(
    r"^\s*(?:[-*•]|\d+\s*[.)])\s*(.+?)\s*:\s*(CO|KHONG)\b\s*(.*)$", re.I)
_KL = re.compile(r"\b(DUNG|THIEU|SAI)\b")


def _tach(tho: str) -> tuple[list[tuple[str, bool, str]], str]:
    """Văn bản model trả về -> (danh sách chi tiết, phán quyết)."""
    muc = []
    for d in str(tho or "").splitlines():
        m = _DONG.match(d.replace("*", ""))
        if not m:
            continue
        ten = " ".join(m.group(1).split())
        ly_do = " ".join(m.group(3).split()).strip("()")
        muc.append((ten, m.group(2).upper() == "CO", ly_do))
        if len(muc) >= MAX_CHI_TIET:
            break
    hit = _KL.findall(str(tho or "").upper())
    return muc, (hit[-1] if hit else "")


def soi(cau: str, anh: str | pathlib.Path) -> dict:
    """Chi tiết nào của câu KHÔNG thấy trong khung này.

    Trả {"muc": [(chi tiết, có/không, lý do)], "phan_quyet": str, "tho": str}.
    Rỗng khi không gọi được --- gọi ở đây tốn một lời gọi VLM có ảnh.
    """
    p = pathlib.Path(anh)
    if not str(cau or "").strip() or not p.is_file():
        return {"muc": [], "phan_quyet": "", "tho": ""}

    import rerank
    from google.genai import types
    cl = rerank._CLIENTS
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    data = p.read_bytes()
    for _ in range(max(cl.n, 1)):
        c = cl.current()
        if c is None:
            return {"muc": [], "phan_quyet": "", "tho": ""}
        try:
            r = c.models.generate_content(
                model=model,
                contents=[NHAC + " ".join(str(cau).split()),
                          types.Part.from_bytes(data=data,
                                                mime_type="image/jpeg")])
            tho = (r.text or "").strip()
            muc, pq = _tach(tho)
            return {"muc": muc, "phan_quyet": pq, "tho": tho}
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                cl.retire()
                continue
            return {"muc": [], "phan_quyet": "", "tho": f"lỗi: {e}"}
    return {"muc": [], "phan_quyet": "", "tho": ""}


def thieu(kq: dict) -> list[tuple[str, str]]:
    """Chỉ những chi tiết KHÔNG thấy, kèm lý do — thứ đáng liếc trước khi nộp."""
    return [(t, ly) for t, co, ly in (kq or {}).get("muc", []) if not co]


# Từ nửa số chi tiết trở lên không thấy thì đáng dừng lại xem. Quét 0,34 -> 0,70
# trên 18 khung đã soi tay: mọi ngưỡng trong 0,34-0,50 cho cùng kết quả, nên
# 0,5 nằm giữa vùng phẳng chứ không phải điểm dò khít.
NGUONG = 0.5


def dang_ngo(kq: dict, nguong: float = NGUONG) -> bool:
    """Khung này có đáng dừng lại soi kỹ không?

    ĐẾM đầu ra của model thắng PHÁN QUYẾT của chính nó: trên 18 khung đã soi
    tay, ngưỡng này bắt đủ 8/8 khung có vấn đề mà chỉ báo nhầm 2/10 khung tốt,
    còn phán quyết ``THIEU/SAI`` của model báo nhầm tới 5/10.
    """
    muc = (kq or {}).get("muc", [])
    if not muc:
        return False
    return sum(1 for _, co, _ in muc if not co) / len(muc) >= nguong
