
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

_CACHE: dict[str, str] = {}


def to_english(text: str, *, timeout: float = 10.0, retries: int = 3) -> str | None:
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
