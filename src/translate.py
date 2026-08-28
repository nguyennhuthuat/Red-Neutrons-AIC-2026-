
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

_CACHE: dict[str, str] = {}


# Endpoint Google chặn theo IP khi gọi dồn — đã hỏng 6/6 ngay trước giờ thi
# ngày 28/08. Nên phải có đường lùi KHÔNG cần mạng. Ba mô hình dịch đều đã nằm
# sẵn trong cache HuggingFace trên ổ D; vinai cho bản dịch sát nghĩa nhất
# ("con lân" -> "unicorn", trong khi envit5 ra "the yellow one").
# CHỌN MÔ HÌNH NHẸ NHẤT, CỐ Ý. Đo 28/08 ngay trước giờ thi:
#   vinai (mBART 2,4 GB) dịch sát nghĩa nhất -> segfault KHÔNG ĐỀU TAY
#     khi nạp chung tiến trình với SigLIP2
#   envit5 (T5 1,2 GB)   -> có lượt treo quá 120 giây
#   opus  (Marian 72M)   -> nạp 2 s, dịch 0,4 s, chưa thấy sự cố
# Bản dịch của opus thô hơn, nhưng đường lùi này chỉ chạy khi mạng đã
# chết, và tiếng Anh thô vẫn hơn tiếng Việt: 0,780 so với 0,704.
MO_HINH_LUI = os.environ.get("DICH_NGOAI_TUYEN",
                             "Helsinki-NLP/opus-mt-vi-en")
NGUON_CUOI = "chưa dịch lần nào"      # "mạng" | "ngoại tuyến" | "hỏng"
_LUI = None
# Cầu dao: endpoint chết thì mỗi truy vấn phí ~6 giây chờ vô ích. Hỏng liên
# tiếp đủ số lần thì thôi gọi mạng, đi thẳng mô hình trên máy.
HONG_TOI_DA = 2
_hong_lien_tiep = 0


def mach_mang_con_song() -> bool:
    return _hong_lien_tiep < HONG_TOI_DA


def dong_lai_cau_dao() -> None:
    """Bật lại đường mạng (dùng khi mạng đã hồi)."""
    global _hong_lien_tiep
    _hong_lien_tiep = 0


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


def to_english(text: str, *, timeout: float = 10.0, retries: int = 3) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    if text in _CACHE:
        return _CACHE[text]

    global _hong_lien_tiep
    url = ("https://translate.googleapis.com/translate_a/single"
           "?client=gtx&sl=vi&tl=en&dt=t&q=" + urllib.parse.quote(text))
    for attempt in range(retries if mach_mang_con_song() else 0):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = "".join(seg[0] for seg in data[0]).strip()
            if out:
                _CACHE[text] = out
                globals()["NGUON_CUOI"] = "mạng"
                _hong_lien_tiep = 0
                return out
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    if mach_mang_con_song():
        _hong_lien_tiep += 1
    # Mạng hỏng thì lùi về mô hình trên máy chứ KHÔNG trả về câu tiếng Việt thô:
    # SigLIP2 đọc tiếng Việt được 0,7037 so với 0,7802 khi dịch (mục sec:tiengviet2).
    ra = dich_ngoai_tuyen(text)
    if ra:
        globals()["NGUON_CUOI"] = "ngoại tuyến"
        _CACHE[text] = ra
        return ra
    globals()["NGUON_CUOI"] = "hỏng"
    return None


def to_english_batch(texts, *, sleep: float = 0.25) -> list[str | None]:
    """Dịch danh sách câu. `sleep` để né rate limit của endpoint."""
    out: list[str | None] = []
    for t in texts:
        out.append(to_english(t))
        time.sleep(sleep)
    return out
