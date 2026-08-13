"""
translate.py — dịch câu truy vấn Việt → Anh.

VÌ SAO VẪN GIỮ dù encoder mới đã đọc được tiếng Việt (đo trên 81 query 2026,
ViT-L-16-SigLIP2-384):

    tiếng Việt thô          0.7037
    tiếng Việt + dịch       0.7210    <- +0.017
    tiếng Anh viết tay      0.7802

Với encoder ViT-B-32 cũ thì tiếng Việt thô cho **0.0000 tuyệt đối** (0/81 lọt cả
top-1000), nên dịch từng là điều kiện SỐNG CÒN. SigLIP2 huấn luyện trên WebLI đa
ngôn ngữ nên đọc thẳng được, và dịch tụt xuống thành "tối ưu cộng thêm".

Vẫn đáng giữ vì +0.017 là thật mà giá gần như bằng 0: endpoint không chính thức
của Google Translate, không cần API key, ~0,25 s/câu.

⚠️ Đây là endpoint KHÔNG chính thức — có thể chết bất cứ lúc nào. Mọi chỗ gọi
phải chịu được `None` (dịch hỏng) bằng cách dùng lại câu tiếng Việt gốc: với
SigLIP2 thì câu gốc vẫn cho 0.7037, tức đường lùi rất êm chứ không về 0 như xưa.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

_CACHE: dict[str, str] = {}


def to_english(text: str, *, timeout: float = 10.0, retries: int = 3) -> str | None:
    """Dịch một câu. Trả None nếu hỏng — KHÔNG trả câu gốc.

    Trả None thay vì im lặng trả lại câu gốc là cố ý: bên gọi cần PHÂN BIỆT
    "đã dịch" với "dịch hỏng, đang dùng câu gốc". Đã có lần cache nhầm câu gốc
    làm bản dịch, khiến điểm về 0 mà không một dòng cảnh báo nào.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text in _CACHE:
        return _CACHE[text]

    url = ("https://translate.googleapis.com/translate_a/single"
           "?client=gtx&sl=vi&tl=en&dt=t&q=" + urllib.parse.quote(text))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = "".join(seg[0] for seg in data[0]).strip()
            if out:
                _CACHE[text] = out
                return out
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    return None


def to_english_batch(texts, *, sleep: float = 0.25) -> list[str | None]:
    """Dịch danh sách câu. `sleep` để né rate limit của endpoint."""
    out: list[str | None] = []
    for t in texts:
        out.append(to_english(t))
        time.sleep(sleep)
    return out
