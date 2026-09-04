# -*- coding: utf-8 -*-
"""Gọi tên cái mà đề chỉ TẢ, rồi tra lại bằng tên ấy.

Ban tổ chức cố tình không gọi tên vật --- đó chính là cái làm nó thành bài tìm
kiếm. p2-22 viết "nguyên liệu hải sản màu trắng được khứa vuông góc, cắt thành
que", còn thẻ nguyên liệu của đúng video ghi "Mucong tuoi: 150g". Kho đánh chỉ
mục theo TÊN, đề chỉ có phần TẢ, và hai đầu không gặp nhau.

Đo trên 30 câu đề thật: tên đoán cứu được ĐÚNG MỘT câu (p2-22, từ không kênh
nào thấy lên hạng 18) --- nhưng trên trung bình nó THUA cả câu gốc ở cả hai
kênh (thẻ hạng-1 1/14 -> 0/14, lời nói 3/14 -> 1/14). Nên đây là kênh gọi khi
cần, KHÔNG phải kênh thay thế, và tuyệt đối không trộn vào điểm.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
KHO = ROOT / "data" / "kernel_out" / "goiten.json"
MAX_TEN = 6

NHAC = """Đây là câu mô tả một cảnh trong video, dùng để tìm lại đúng cảnh đó.
Người ra đề CỐ TÌNH không gọi tên vật, chỉ tả hình dáng, màu sắc, cách chế biến.

Hãy đoán TÊN TIẾNG VIỆT CỤ THỂ của những vật/nguyên liệu/món ăn mà câu đang tả
nhưng không gọi tên. Ví dụ "nguyên liệu hải sản màu trắng được khứa theo đường
vuông góc rồi cắt thành que" -> "mực ống, mực lá".

Quy tắc:
- Chỉ trả tên, cách nhau bằng dấu phẩy. Không giải thích, không đánh số.
- Tối đa 6 tên, xếp tên chắc chắn nhất lên đầu.
- KHÔNG lặp lại từ đã có sẵn trong câu.
- Nếu câu đã gọi tên rõ hết rồi thì trả đúng một dấu gạch ngang: -

Câu: """


def _doc_kho() -> dict:
    if KHO.exists():
        try:
            return json.loads(KHO.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _ghi_kho(k: dict) -> None:
    KHO.parent.mkdir(parents=True, exist_ok=True)
    KHO.write_text(json.dumps(k, ensure_ascii=False, indent=1), encoding="utf-8")


_SO = re.compile(r"^\s*\d+\s*[.)]\s*")


def _tach(kq: str) -> list[str]:
    """Chuỗi model trả về -> danh sách tên sạch.

    Tách TRƯỚC rồi mới gộp khoảng trắng: gộp trước thì dấu xuống dòng biến mất
    và một danh sách đánh số dồn thành một chuỗi rác.
    """
    kq = str(kq or "")
    if not kq.strip() or kq.strip() in ("-", "—"):
        return []
    ra = []
    for t in re.split(r"[,;\n]", kq):
        t = " ".join(_SO.sub("", t).split()).strip(" .-–—*•\"'")
        if t and len(t) > 1 and t not in ra:
            ra.append(t)
    return ra[:MAX_TEN]


def de_xuat(cau: str, *, dung_kho: bool = True) -> list[str]:
    """Tên có thể của những vật mà câu chỉ tả. [] nếu đề đã gọi tên rõ.

    Tốn MỘT lời gọi văn bản thuần cho mỗi câu mới, không đụng hạn mức ảnh của
    tầng VLM. Kết quả ghi ra đĩa nên hỏi lại cùng một câu thì miễn phí.
    """
    cau = " ".join(str(cau or "").split())
    if not cau:
        return []
    kho = _doc_kho() if dung_kho else {}
    if cau in kho:
        return _tach(kho[cau])

    import rerank
    cl = rerank._CLIENTS
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    kq = ""
    for _ in range(max(cl.n, 1)):
        c = cl.current()
        if c is None:
            return []
        try:
            r = c.models.generate_content(model=model, contents=NHAC + cau)
            kq = (r.text or "").strip()
            break
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                cl.retire()
                continue
            return []
    kho[cau] = kq
    try:
        _ghi_kho(kho)
    except Exception:      # ổ chỉ đọc thì thôi, đừng làm hỏng lượt tìm
        pass
    return _tach(kq)


def tra(ten: list[str], lay: int = 200) -> dict:
    """Tra danh sách tên qua kênh thẻ nguyên liệu và kênh lời nói.

    Trả {"the": [(video_id, điểm)...], "asr": [...]}. KHÔNG trộn điểm --- đo
    được là tên đoán thua cả câu gốc trên trung bình, nó chỉ đáng làm ý kiến
    thứ hai cho những câu mà mọi kênh khác đã trắng tay.
    """
    ra = {"the": [], "asr": []}
    if not ten:
        return ra
    cau = ", ".join(ten)
    try:
        import ocr_thecuoi
        d = ocr_thecuoi.tim(cau, lay)
        if d is not None and len(d) and "video_id" in d:
            g = d.groupby("video_id", sort=False)["diem"].max()
            ra["the"] = [(v, float(x))
                         for v, x in g.sort_values(ascending=False).head(8).items()]
    except Exception:
        pass
    try:
        import asr_tim
        d = asr_tim.tim(cau, lay)
        if d is not None and len(d) and "video_id" in d:
            g = d.groupby("video_id", sort=False)["diem"].max()
            ra["asr"] = [(v, float(x))
                         for v, x in g.sort_values(ascending=False).head(8).items()]
    except Exception:
        pass
    return ra


if __name__ == "__main__":
    import sys
    cau = " ".join(sys.argv[1:]) or (
        "Trong video nấu ăn, một loại nguyên liệu hải sản màu trắng được khứa "
        "theo những đường thẳng vuông góc nhau, trên cả 2 bề mặt.")
    ten = de_xuat(cau)
    print("tên đoán:", ", ".join(ten) or "(đề đã gọi tên rõ)")
    for kenh, ds in tra(ten).items():
        print(f"  {kenh}: {[v for v, _ in ds][:5]}")
