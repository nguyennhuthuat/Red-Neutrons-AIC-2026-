
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

_CACHE: dict[str, str] = {}

# Kho dịch trên ĐĨA. Trước đây kho chỉ nằm trong RAM: khởi động lại giao
# diện là mất sạch, mỗi câu đã dịch rồi vẫn tiêu lượt lần nữa.
TEP_KHO = pathlib.Path(os.environ.get(
    "DICH_KHO",
    str(pathlib.Path(__file__).resolve().parent.parent
        / "data" / "kernel_out" / "dich_cache.json")))


def _chuan_hoa(t) -> str:
    """Khoá kho: gộp mọi khoảng trắng, để câu chỉ khác nhau ở dấu cách
    không tiêu hai lượt."""
    return " ".join(str(t or "").split())


def nap_kho() -> int:
    try:
        d = json.loads(TEP_KHO.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(d, dict):
        return 0
    n = 0
    for k, v in d.items():
        if isinstance(k, str) and isinstance(v, str) and k and v:
            _CACHE.setdefault(_chuan_hoa(k), v)
            n += 1
    return n


def ghi_kho() -> None:
    """Trộn với những gì tiến trình khác đã ghi, rồi thay tệp nguyên khối."""
    try:
        try:
            cu = json.loads(TEP_KHO.read_text(encoding="utf-8"))
            if isinstance(cu, dict):
                for k, v in cu.items():
                    if isinstance(k, str) and isinstance(v, str):
                        _CACHE.setdefault(k, v)
        except Exception:
            pass
        TEP_KHO.parent.mkdir(parents=True, exist_ok=True)
        tam = TEP_KHO.with_suffix(".tmp")
        tam.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=0),
                       encoding="utf-8")
        tam.replace(TEP_KHO)
    except Exception:
        pass


nap_kho()


# Endpoint Google chặn theo IP khi gọi dồn — hỏng 6/6 ngày 28/08 và vẫn hỏng
# (HTTP 429) sáng 28/08. Nên phải có đường lùi.
#
# ĐO NGÀY 28/08, TRÊN ĐÚNG BỘ 81 QUERY, và kết quả LẬT NGƯỢC lựa chọn hôm qua:
#     Google (mạng)        0,7827   (đo 19/08, lúc endpoint còn sống)
#     tiếng Việt thô       0,7160   ← không dịch gì cả
#     opus ngoại tuyến     0,5926   ← THẤP HƠN CẢ KHÔNG DỊCH, mất 0,123
# Hôm qua tôi chọn opus vì suy ra "tiếng Anh thô vẫn hơn tiếng Việt", lấy con số
# 0,780 của Google gán cho opus. Sai: 0,780 là chất lượng bản dịch Google, không
# phải của mọi bản dịch. Opus dịch hỏng nghĩa ("đoàn múa lân" -> "a procession",
# "xe cộ mắc kẹt" -> "it's stuck in traffic") nên encoder tìm sai hẳn.
# Vì vậy opus MẶC ĐỊNH TẮT. Chuỗi lùi đúng là: mạng -> Gemini -> tiếng Việt thô.
# Gemini có 7 khoá còn sống và KIS vốn không tiêu API, nên đây là chỗ tiêu hợp lý.
MO_HINH_LUI = os.environ.get("DICH_NGOAI_TUYEN", "")   # "" = tắt, xem đo ở trên
NGUON_CUOI = "chưa dịch lần nào"   # "mạng"|"gemini"|"ngoại tuyến"|"hỏng"
_LUI = None
# Cầu dao: endpoint chết thì mỗi truy vấn phí ~6 giây chờ vô ích. Hỏng liên
# tiếp đủ số lần thì thôi gọi mạng, đi thẳng mô hình trên máy.
HONG_TOI_DA = 2
_hong_lien_tiep = 0
# Cầu dao mở VĨNH VIỄN là quá tay: một cú 429 thoáng qua thì mất Google cả
# buổi thi, mà Google 0,7827 còn Gemini 0,7556. Nghỉ rồi thử lại một lượt.
NGHI_GIAY = float(os.environ.get("DICH_NGHI_GIAY", "180"))
_mo_lai_luc = 0.0


def mach_mang_con_song() -> bool:
    return _hong_lien_tiep < HONG_TOI_DA or time.time() >= _mo_lai_luc


def _ngat_mach() -> None:
    global _hong_lien_tiep, _mo_lai_luc
    _hong_lien_tiep += 1
    if _hong_lien_tiep >= HONG_TOI_DA:
        _mo_lai_luc = time.time() + NGHI_GIAY


def dong_lai_cau_dao() -> None:
    """Bật lại đường mạng (dùng khi mạng đã hồi)."""
    global _hong_lien_tiep, _mo_lai_luc
    _hong_lien_tiep = 0
    _mo_lai_luc = 0.0


def _nap_lui():
    """Nạp mô hình dịch ngoại tuyến, một lần rồi giữ. ~20 s lần đầu."""
    global _LUI
    if _LUI is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        kw = {"src_lang": "vi_VN"} if "vinai" in MO_HINH_LUI else {}
        tok = AutoTokenizer.from_pretrained(MO_HINH_LUI, **kw)
        mod = AutoModelForSeq2SeqLM.from_pretrained(MO_HINH_LUI)
        mod.eval()
        _LUI = (tok, mod)
    return _LUI


def dich_ngoai_tuyen(text: str) -> str | None:
    """Dịch bằng mô hình trên máy. Không cần mạng, không cần khoá API."""
    try:
        import torch
        tok, mod = _nap_lui()
        pre = "vi: " if "envit5" in MO_HINH_LUI else ""
        x = tok([pre + text], return_tensors="pt", padding=True)
        gen = {"max_new_tokens": 200}
        if "vinai" in MO_HINH_LUI:
            gen["decoder_start_token_id"] = tok.convert_tokens_to_ids("en_XX")
        with torch.no_grad():
            y = mod.generate(**x, **gen)
        out = tok.batch_decode(y, skip_special_tokens=True)[0].strip()
        return out[3:].strip() if out.lower().startswith("en:") else out or None
    except Exception:
        return None


NHAC_DICH = (
    "Dịch câu mô tả cảnh sau sang tiếng Anh, giữ nguyên mọi chi tiết nhìn thấy "
    "được (màu sắc, số lượng, vị trí, trang phục, vật thể). Chỉ trả về câu "
    "tiếng Anh, không giải thích, không thêm gì khác." + chr(10) * 2)


def dich_gemini(text: str) -> str | None:
    """Đường lùi CHÍNH khi mạng Google chết. Dùng chung bể khoá với khâu Q&A."""
    try:
        import rerank
        clients = rerank._CLIENTS
        if not clients:
            return None
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        for _ in range(clients.n):
            c = clients.current()
            if c is None:
                return None
            try:
                r = c.models.generate_content(
                    model=model, contents=NHAC_DICH + text)
                out = (r.text or "").strip().strip('"')
                return out.split(chr(10))[0].strip() or None
            except Exception as err:
                # Hết hạn mức thì đổi khoá; khoá hỏng cũng bỏ. Lỗi khác thì thôi.
                if rerank._is_quota_error(err) or rerank._is_auth_error(err):
                    clients.retire(verbose=False)
                    continue
                return None
    except Exception:
        return None
    return None


def _qua_han(err) -> bool:
    """429/403 = đang bị chặn. Gọi lại đúng lúc ấy chỉ kéo dài lệnh chặn."""
    return isinstance(err, urllib.error.HTTPError) and err.code in (429, 403)


def _goi_mang(q: str, *, timeout: float, retries: int) -> str | None:
    """Một lượt gọi endpoint. Trả nguyên văn bản đã nối, KHÔNG cắt khoảng trắng
    hai đầu — dich_nhieu còn phải tách lại theo dấu xuống dòng."""
    global _hong_lien_tiep
    url = ("https://translate.googleapis.com/translate_a/single"
           "?client=gtx&sl=vi&tl=en&dt=t&q=" + urllib.parse.quote(q))
    for attempt in range(retries if mach_mang_con_song() else 0):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = "".join(seg[0] for seg in data[0])
            if out.strip():
                globals()["NGUON_CUOI"] = "mạng"
                _hong_lien_tiep = 0
                return out
            break
        except Exception as err:
            if _qua_han(err):
                _ngat_mach()
                return None
            if attempt < retries - 1:
                time.sleep(1.0)
    if mach_mang_con_song():
        _ngat_mach()
    return None


def dich_nhieu(texts, *, timeout: float = 10.0, retries: int = 3) -> list[str | None]:
    """Dịch cả danh sách bằng MỘT lượt gọi, nối bằng dấu xuống dòng.

    Google cắt đoạn theo CÂU nên số đoạn trả về không khớp số dòng vào; nhưng nó
    giữ nguyên dấu xuống dòng, nên nối hết rồi tách lại thì khớp. Không khớp thì
    lùi về dịch từng câu — thà tốn lượt còn hơn ghép nhầm cảnh này vào cảnh kia.
    """
    ts = [_chuan_hoa(t) for t in texts]
    ra: list[str | None] = [_CACHE.get(t) if t else None for t in ts]
    con = [i for i, t in enumerate(ts) if t and ra[i] is None]
    if len(con) > 1 and mach_mang_con_song():
        goc = [ts[i] for i in con]
        gop = _goi_mang(chr(10).join(goc), timeout=timeout, retries=retries)
        phan = gop.split(chr(10)) if gop is not None else []
        if len(phan) == len(goc):
            for i, p in zip(con, phan):
                p = p.strip()
                if p:
                    _CACHE[ts[i]] = p
                    ra[i] = p
            ghi_kho()
            con = [i for i in con if ra[i] is None]
    for i in con:
        ra[i] = to_english(ts[i], timeout=timeout, retries=retries)
    return ra


def to_english(text: str, *, timeout: float = 10.0, retries: int = 3) -> str | None:
    text = _chuan_hoa(text)
    if not text:
        return None
    if text in _CACHE:
        globals()["NGUON_CUOI"] = "kho"
        return _CACHE[text]

    out = _goi_mang(text, timeout=timeout, retries=retries)
    if out and out.strip():
        _CACHE[text] = out.strip()
        ghi_kho()
        return _CACHE[text]

    ra = dich_gemini(text)
    if ra:
        globals()["NGUON_CUOI"] = "gemini"
        _CACHE[text] = ra
        ghi_kho()
        return ra

    # Chỉ chạy khi người dùng CỐ Ý bật DICH_NGOAI_TUYEN (máy không có mạng lẫn
    # khoá API). Mặc định tắt vì đo được 0,5926, thua cả tiếng Việt thô 0,7160.
    if MO_HINH_LUI:
        ra = dich_ngoai_tuyen(text)
        if ra:
            globals()["NGUON_CUOI"] = "ngoại tuyến"
            _CACHE[text] = ra
            ghi_kho()
            return ra

    # Cả hai đường chết: trả None để bên gọi dùng thẳng tiếng Việt (0,7160) —
    # vẫn hơn bất kỳ bản dịch hỏng nào.
    globals()["NGUON_CUOI"] = "hỏng"
    return None


def to_english_batch(texts, *, sleep: float = 0.25,
                     goi: int = 40) -> list[str | None]:
    """Dịch danh sách câu. Gộp từng gói `goi` câu vào một lượt gọi."""
    out: list[str | None] = []
    for d in range(0, len(texts), max(1, goi)):
        out.extend(dich_nhieu(texts[d:d + max(1, goi)]))
        if d + goi < len(texts):
            time.sleep(sleep)
    return out
