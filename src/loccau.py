# -*- coding: utf-8 -*-
"""Lọc nhiễu câu hỏi BTC và cắt cho vừa cửa sổ 64 token của SigLIP2.

Đo 29/08: 12/18 câu KIS đề thật vòng 1 dùng HẾT 64 token, tức là bị tokenizer
cắt cụt trong im lặng — phần đuôi câu chưa bao giờ tới được encoder. Bộ đo 81
câu không thấy lỗi này vì trung vị chỉ 31 token, tràn 0/81.

Đo 30/08 trên 30 câu đề vòng 2: 17/30 tràn, mà lọc chữ dẫn chuyện chỉ cứu
được 1 câu — đề vòng 2 dài vì NỘI DUNG thật chứ không vì văn vẻ. Đường ra là
tách cảnh (xem chuoi.py), nên hàm ở đây phải GIỮ dấu xuống dòng của BTC.

Ba việc, theo đúng thứ tự đó:
  1. bỏ chữ nói về VIỆC QUAY chứ không nói về cái NHÌN THẤY;
  2. tách câu hỏi ra khỏi phần mô tả (câu Q&A);
  3. đếm token để bày ra chỗ câu bị cắt.
"""
from __future__ import annotations

import re

NGAN_SACH = 64          # context_length của ViT-L-16-SigLIP2-512
CHUA = 2                # chừa chỗ cho <eos> và sai số khi dịch

# Khung dẫn chuyện. Chỉ khớp ở ĐẦU một câu: giữa câu thì "cảnh" thường là
# danh từ thật ("cảnh quay trực diện"), bỏ đi là mất nghĩa.
KHUNG_DAU = [
    r"đoạn\s+(?:clip|video|phim)\s+(?:ngắn\s+)?(?:cần\s+tìm\s+)?"
    r"(?:được\s+)?(?:mô\s+tả|tường\s+thuật|ghi\s+lại|nói\s+về|về|là|có|gồm)?\s*"
    r"(?:cảnh(?:\s+quay)?|phân\s+cảnh|khoảnh\s+khắc)?\s*",
    r"tìm\s+(?:chính\s+xác\s+)?(?:đoạn\s+(?:clip|video)|phân\s+cảnh|khoảnh\s+khắc)"
    r"\s*(?:ngắn\s+)?(?:với|có|là)?\s*",
    r"trong\s+(?:đoạn\s+)?(?:clip|video|phim)"
    r"(?:\s+(?:hướng\s+dẫn\s+)?(?:nấu\s+ăn|dạy\s+học))?\s*"
    r"(?:có\s+là|có|là)?\s*,?\s*",
    r"(?:hình\s+ảnh|cảnh\s+quay|khung\s+hình)\s+(?:cho\s+thấy|ghi\s+lại)\s*",
    r"phân\s+cảnh\s+(?:bắt\s+đầu|tiếp\s+theo|cuối\s+cùng)\s*"
    r"(?:cho\s+thấy|là|có|với)?\s*",
    # Vòng 2 mở câu bằng "Cảnh quay một tô cháo…", "Cảnh phim lần lượt…".
    r"cảnh\s+(?:quay|phim)\s+(?:lần\s+lượt\s+)?(?:giới\s+thiệu\s+)?",
]
# Cụm nối giữa câu: chỉ là cách nói "trong ảnh còn có", bỏ được nguyên cụm.
KHUNG_GIUA = [
    r"xuất\s+hiện\s+trong\s+khung\s+hình\s+(?:còn\s+)?(?:có|là)\s*",
    r"trong\s+khung\s+hình\s+(?:gồm\s+)?(?:có|là)\s*",
    r"khung\s+hình\s+(?:gồm\s+)?(?:có|là)\s*",
    r"trong\s+cảnh\s+(?:gồm\s+)?(?:có|là)\s*",
    r"(?:đoạn\s+)?(?:clip|video)\s+(?:do|của)\s+[^,.;]{0,40}?ghi\s+lại\s+cho\s+thấy\s*",
]
# Mệnh đề cảm thán / kiến thức ngoài hình: cắt tới hết mệnh đề.
# CỐ Ý không có "tạo thành": "đổ bóng lên tường, TẠO THÀNH hình chân dung" là
# thứ nhìn thấy được, bỏ đi là mất hẳn nội dung câu.
CAM_THAN = [
    r",?\s*tạo\s+(?:cảm\s+giác|không\s+khí|điểm\s+nhấn|vẻ|nên\s+vẻ)[^.;]*",
    r",?\s*để\s+tăng\s+(?:phần|thêm)\s+[^.;]*",
    r",?\s*(?:trông|tỏ\s+vẻ|ra\s+vẻ)\s+(?:rất\s+)?"
    r"(?:thích\s+thú|vui\s+vẻ|hạnh\s+phúc|thoải\s+mái)[^.;]*",
    r"\s*đây\s+là\s+(?:loài|loại|món|nơi|địa\s+danh)[^.;]*(?:thường\s+thấy|nổi"
    r"\s+tiếng|đặc\s+trưng)[^.;]*\.?",
    r",?\s*(?:mang\s+lại|gợi)\s+(?:cảm\s+giác|không\s+khí)[^.;]*",
]
# Ngoặc đơn KHÔNG tả cái nhìn thấy: người ra đề tự phân vân, hoặc chú metadata
# hành chính. CỐ Ý chừa ngoặc chú thích đồng nghĩa ("nấm mèo (mộc nhĩ)") và
# ngoặc đánh nhãn ("loài cây (II)") — hai thứ đó là nội dung.
NGOAC = [
    r"\s*\([^)]*\?\s*\)",                              # (hay rồng/sư tử?)
    r"\s*\([^)]*(?:cũ|trước\s+ngày|sau\s+ngày|nay\s+là)[^)]*\)",
]

_DAU = re.compile(r"^\s*(?:" + "|".join(KHUNG_DAU) + r")", re.I)
_GIUA = re.compile("|".join(KHUNG_GIUA), re.I)
_CT = re.compile("|".join(CAM_THAN), re.I)
_NG = re.compile("|".join(NGOAC), re.I)
_CAU = re.compile(r"(?<=[.!?;])\s+|\n+")
_CAU_DONG = re.compile(r"(?<=[.!?;])\s+")
# Sau khi gỡ khung, câu hay mở đầu bằng chữ nối trơ trọi.
_NOI_THUA = re.compile(r"^\s*(?:là|có|gồm|với|và|còn|thì)\s+", re.I)

# Câu HỎI ở cuối đề Q&A: 10/10 câu vòng 2 có, và không câu nào tả cái nhìn
# thấy được. Với khâu TÌM thì đó là ~12 token nhiễu; với khâu TRẢ LỜI thì
# đó là toàn bộ đề bài — nên tách đôi chứ không vứt.
# Chỉ cắt khi câu KẾT bằng '?', hoặc mở bằng "Hỏi"/"Hãy cho biết". Bắt theo đại
# từ nghi vấn không thôi thì nuốt nhầm câu tả: "…để xem hôm nay nấu món gì."
# (p2-30) là một CẢNH thật, không phải câu hỏi.
_HOI = re.compile(r"^(?:\s*(?:hỏi|hãy\s+cho\s+biết)\b.*|.*\?)\s*$", re.I)
# Đường lùi: câu CUỐI có đại từ nghi vấn. Chỉ dùng khi luật hẹp trắng tay, và
# chỉ lấy MỘT câu — bắt tham thì nuốt câu tả ("…để xem hôm nay nấu món gì.").
_HOI_RONG = re.compile(
    r"\b(?:số|con|loài|món|màu|chữ|tên|người|cái|bao\s+nhiêu)\s+"
    r"(?:nào|gì|mấy)\b|\bbao\s+nhiêu\b|\bmấy\s+(?:giờ|cái|con|người)\b", re.I)


def _sach(c: str) -> str:
    """Gỡ khung dẫn chuyện khỏi MỘT câu.

    Giữ lại dấu ':' cuối câu: đó là dấu hiệu dòng tiêu đề ("Trên slide bao
    gồm:"), và tầng tách cảnh dựa vào nó để ghép tiêu đề vào dòng sau.
    """
    ket = ":" if c.rstrip().endswith(":") else ""
    c = _NG.sub("", c)
    c = _GIUA.sub(" ", c)
    c = _DAU.sub("", c)
    c = _CT.sub("", c)
    c = _NOI_THUA.sub("", c)
    return " ".join(c.split()).strip(" ,;:.") + ket


def bo_nhieu(text: str) -> str:
    """Bỏ chữ nói về việc quay, giữ nguyên chữ nói về cái nhìn thấy.

    GIỮ dấu xuống dòng: BTC xuống dòng ở đúng chỗ đổi cảnh, và đó là tín hiệu
    tách cảnh chắc hơn dấu chấm nhiều. Gộp hết về một dòng là tự tay xoá nó.
    """
    ra_dong = []
    for dong in str(text or "").split("\n"):
        cau = [_sach(c) for c in _CAU_DONG.split(dong) if c.strip()]
        cau = [c for c in cau if len(c) >= 3]
        if cau:
            ra_dong.append(". ".join(cau))
    return "\n".join(ra_dong) or " ".join(str(text or "").split())


def tach_hoi(text: str) -> tuple[str, str]:
    """Tách đề Q&A thành (phần MÔ TẢ, câu HỎI).

    BTC phát một khối liền: tả cảnh rồi mới hỏi. Khâu tìm chỉ cần phần tả,
    khâu trả lời chỉ cần câu hỏi. Không tách được thì trả câu hỏi rỗng.
    """
    dong = [d for d in str(text or "").split("\n") if d.strip()]
    if not dong:
        return "", ""
    cau = [c.strip() for c in _CAU_DONG.split(dong[-1]) if c.strip()]
    hoi = []
    while cau and _HOI.match(cau[-1]):
        hoi.insert(0, cau.pop())
    if not hoi and len(cau) > 1 and _HOI_RONG.search(cau[-1]):
        hoi = [cau.pop()]
    if not hoi:
        return " ".join(str(text or "").split()), ""
    dong[-1] = " ".join(cau)
    mo_ta = "\n".join(d for d in dong if d.strip())
    return mo_ta.strip(), " ".join(hoi)


def tach_cau(text: str) -> list[str]:
    """Cắt theo dấu câu và xuống dòng — đơn vị nhỏ nhất còn tự đứng được."""
    return [" ".join(c.split()).strip(" ,;:.")
            for c in _CAU.split(str(text or "")) if c.strip()]


def gom_vua_khung(cau: list[str], dem, ngan_sach: int = NGAN_SACH - CHUA
                  ) -> list[str]:
    """Gộp các câu liền kề thành mảnh lớn nhất còn vừa cửa sổ token.

    Gộp tham lam theo THỨ TỰ, không xếp lại: hai câu liền nhau thường tả cùng
    một cảnh, tách ra là mất quan hệ giữa chúng.
    """
    manh, hien = [], ""
    for c in cau:
        thu = f"{hien}. {c}" if hien else c
        if hien and dem(thu) > ngan_sach:
            manh.append(hien)
            hien = c
        else:
            hien = thu
    if hien:
        manh.append(hien)
    # Một câu đơn lẻ mà vẫn tràn thì đành để tokenizer cắt — nhưng đã báo ở UI.
    return manh or [""]


def dem(text: str, tokenizer) -> int:
    """Số token thật sự đi vào encoder (đã bỏ phần đệm 0)."""
    return int((tokenizer([str(text or "")])[0] != 0).sum())


def phan_giu_lai(text: str, tokenizer) -> str | None:
    """Phần câu encoder THẬT SỰ đọc được; None nếu không bị cắt gì.

    Có để người thi nhìn tận mắt chỗ câu mình bị cắt, thay vì đoán.
    """
    ids = tokenizer([str(text or "")])[0]
    if int((ids != 0).sum()) < NGAN_SACH:
        return None
    tk = getattr(tokenizer, "tokenizer", None)
    if tk is None or not hasattr(tk, "decode"):
        return ""
    try:
        return tk.decode([int(i) for i in ids if int(i) != 0]).replace("<eos>", "")
    except Exception:
        return ""
