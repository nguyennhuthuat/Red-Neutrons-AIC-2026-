
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "eval" / ".cache"

VLM_BATCH = 20        # ảnh mỗi lời gọi; rẻ hơn hẳn gọi lẻ mà model vẫn phân biệt
VLM_PX = int(os.environ.get("VLM_PX", 224))
VLM_TIMEOUT_MS = int(os.environ.get("VLM_TIMEOUT_MS", 90_000))

VLM_CHO_429 = float(os.environ.get("VLM_CHO_429", 30))

PROMPT = """Bạn đang chấm điểm cho một hệ thống tìm kiếm video.
Câu truy vấn: "{qt}"

Dưới đây là {k} ảnh, đánh số 1..{k} theo đúng thứ tự gửi.
Với MỖI ảnh, chấm 0-10 mức độ ảnh đó khớp câu truy vấn:
  10 = khớp mọi chi tiết được nêu
  5  = đúng chủ đề chung nhưng sai chi tiết
  0  = không liên quan
Chú ý các chi tiết cụ thể (màu sắc, số lượng, vị trí, hành động), đừng chỉ nhìn chủ đề.

CHỈ trả về JSON, không giải thích: {{"scores": [<{k} số nguyên>]}}"""


# ─────────────────────────────── tiện ích ───────────────────────────────

def _thumb(path: str | Path, px: int = VLM_PX) -> bytes:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return (a - a.mean()) / (a.std() + 1e-9)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


class _Cache:
    """Cache trên đĩa cho các lời gọi VLM (đắt và có hạn mức)."""

    def __init__(self, name: str = "vlm_rerank.json"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / name
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def _key(query: str, paths, px: int, model: str) -> str:
    h = hashlib.sha1("|".join(str(p) for p in paths).encode("utf-8")).hexdigest()[:16]
    return f"{model}|{px}|{hashlib.sha1(query.encode()).hexdigest()[:12]}|{h}"


# ───────────────────────── nhiều khoá API ─────────────────────────

def api_keys() -> list[str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    out, seen = [], set()
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"] + \
                [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]:
        v = (os.environ.get(name) or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _is_auth_error(e: Exception) -> bool:
    """401/403 = khoá sai hoặc chưa bật API ⇒ bỏ ngay, thử lại là phí."""
    s = f"{type(e).__name__} {e}"
    return "401" in s or "403" in s or "UNAUTHENTICATED" in s or "PERMISSION_DENIED" in s


def _is_quota_error(e: Exception) -> bool:
    """429 = hết hạn mức ⇒ đổi khoá. Lỗi khác (503, mạng) thì thử lại."""
    s = f"{type(e).__name__} {e}"
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


class _Clients:
    """Giữ client cho từng khoá và nhớ khoá nào đã cạn, khỏi thử lại vô ích."""

    def __init__(self):
        self._keys = api_keys()
        self._made: dict[int, object] = {}
        self._dead: set[int] = set()
        self.i = 0

    def __bool__(self):
        return bool(self._keys)

    @property
    def n(self):
        return len(self._keys)

    def current(self):
        from google import genai
        while self.i < len(self._keys) and self.i in self._dead:
            self.i += 1
        if self.i >= len(self._keys):
            return None
        if self.i not in self._made:
            self._made[self.i] = genai.Client(
                api_key=self._keys[self.i],
                http_options={"timeout": VLM_TIMEOUT_MS})
        return self._made[self.i]

    def retire(self, verbose: bool = True):
        """Đánh dấu khoá hiện tại đã cạn và chuyển sang khoá kế."""
        self._dead.add(self.i)
        old = self.i
        self.i += 1
        if verbose:
            left = len(self._keys) - len(self._dead)
            print(f"    [API] khoá #{old + 1} hết hạn mức -> chuyển khoá "
                  f"#{self.i + 1} ({left} khoá còn dùng được)", flush=True)
        return self.current()


# ─────────────────────────────── kênh VLM ───────────────────────────────

_CLIENTS = _Clients()


def vlm_scores(query_vi: str, image_paths, *, model: str | None = None,
               px: int = VLM_PX, batch: int = VLM_BATCH, sleep: float = 1.5,
               cache: _Cache | None = None, verbose: bool = False) -> np.ndarray:
    from google.genai import types

    clients = _CLIENTS
    if not clients:
        raise RuntimeError("Không thấy GEMINI_API_KEY trong .env hay biến môi trường")

    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    cache = cache if cache is not None else _Cache()
    paths = list(image_paths)
    out: list[float] = []

    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ck = _key(query_vi, chunk, px, model)
        hit = cache.get(ck)
        if hit is not None:
            out.extend(hit)
            continue

        parts = [types.Part.from_text(text=PROMPT.format(qt=query_vi, k=len(chunk)))]
        parts += [types.Part.from_bytes(data=_thumb(p, px), mime_type="image/jpeg")
                  for p in chunk]
        got = None
        quota_hits = 0
        for attempt in range(5):
            client = clients.current()
            if client is None:                      # mọi khoá đã cạn
                break
            try:
                resp = client.models.generate_content(
                    model=model, contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.0, response_mime_type="application/json"))
                sc = json.loads(resp.text)["scores"]
                if len(sc) != len(chunk):
                    raise ValueError(f"trả {len(sc)} điểm thay vì {len(chunk)}")
                got = [float(x) for x in sc]
                break
            except Exception as e:
                if _is_auth_error(e):
                    print(f"    [API] khoá #{clients.i + 1} KHÔNG HỢP LỆ "
                          f"({type(e).__name__} 401/403) — bỏ qua khoá này", flush=True)
                    if clients.retire(verbose=False) is None:
                        break
                    continue
                if _is_quota_error(e):
                    quota_hits += 1
                    if quota_hits == 1 and VLM_CHO_429 > 0:
                        print(f"    [API] 429 — chờ {VLM_CHO_429:.0f}s xem có phải "
                              f"chặn theo phút", flush=True)
                        time.sleep(VLM_CHO_429)
                        continue
                    if VLM_CHO_429 <= 0:
                        break
                    if clients.retire() is None:
                        break
                    quota_hits = 0          # khoá mới, đếm lại từ đầu
                    continue
                print(f"    [API] lỗi ({attempt + 1}/5): {type(e).__name__} "
                      f"{str(e)[:120]}", flush=True)
                if attempt < 3:
                    print(f"    [API] chờ {20 * (attempt + 1)}s rồi thử lại",
                          flush=True)
                    time.sleep(20 * (attempt + 1))
        if got is None:
            out.extend([np.nan] * len(chunk))
        else:
            cache.put(ck, got)
            out.extend(got)
            time.sleep(sleep)

    return np.asarray(out, dtype=np.float64)


def basket_looks_wrong(scores: np.ndarray, threshold: float = 10.0) -> bool:
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]
    return bool(len(s) and s.max() < threshold)


def basket_uncertain(base_scores, *, shallow: int = 20,
                     threshold: float = 0.12) -> bool:
    s = np.asarray(base_scores, dtype=np.float64)[:shallow]
    if len(s) < 2 or s[0] <= 0:
        return False
    return bool((s[0] - s[-1]) / s[0] < threshold)


def vlm_scores_adaptive(query_vi: str, image_paths, base_scores, *,
                        shallow: int = 20, deep: int = 200,
                        threshold: float = 0.12, **kw) -> tuple[np.ndarray, bool]:
    paths = list(image_paths)
    shallow = min(shallow, len(paths))
    head = vlm_scores(query_vi, paths[:shallow], **kw)

    if len(paths) <= shallow or not basket_uncertain(
            base_scores, shallow=shallow, threshold=threshold):
        return head, False

    tail = vlm_scores(query_vi, paths[shallow:deep], **kw)
    return np.concatenate([head, tail]), True


def all_failed(scores) -> bool:
    """VLM không chấm được ô nào — lỗi mạng/hạn mức, không phải lỗi dữ liệu."""
    s = np.asarray(scores, dtype=np.float64)
    return bool(s.size == 0 or np.isnan(s).all())


# ──────────────────────────── kênh encoder ────────────────────────────

def encoder_scores(query_en: str, image_paths, *, model_name: str = "ViT-B-16-SigLIP2-384",
                   pretrained: str = "webli", batch: int = 32,
                   _cache: dict | None = None) -> np.ndarray:
    import open_clip
    import torch
    from PIL import Image

    store = _cache if _cache is not None else _MODELS
    tag = f"{model_name}|{pretrained}"
    if tag not in store:
        m, _, prep = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        m.eval()
        store[tag] = (m, prep, open_clip.get_tokenizer(model_name))
    model, prep, tokz = store[tag]

    with torch.no_grad():
        tv = model.encode_text(tokz([query_en])).float()
        tv /= tv.norm(dim=-1, keepdim=True)
        tv = tv.numpy()[0]

        paths = list(image_paths)
        out = np.zeros(len(paths), dtype=np.float64)
        for i in range(0, len(paths), batch):
            ims = torch.stack([prep(Image.open(p).convert("RGB"))
                               for p in paths[i:i + batch]])
            v = model.encode_image(ims).float()
            v /= v.norm(dim=-1, keepdim=True)
            out[i:i + len(ims)] = v.numpy() @ tv
    return out


_MODELS: dict = {}


def encoder_scores_precomputed(qvec: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Phiên bản rẻ: vector ảnh đã mã hoá sẵn, chỉ còn một phép nhân ma trận."""
    q = np.asarray(qvec, dtype=np.float32).ravel()
    q = q / (np.linalg.norm(q) + 1e-9)
    return np.asarray(feats, dtype=np.float32) @ q


# ──────────────────────────────── ghép ────────────────────────────────

def combine(base_scores, *, vlm=None, enc=None,
            w_enc: float = 2.0, w_img: float | None = None) -> np.ndarray:
    if w_img is None:
        w_img = 0.5 if enc is not None else 0.2

    img = zscore(base_scores)
    if enc is not None:
        img = zscore(img + w_enc * zscore(enc))
    if vlm is None:
        return img

    v = np.asarray(vlm, dtype=np.float64)
    if np.isnan(v).all():
        return img
    v = np.where(np.isnan(v), np.nanmean(v), v)
    return zscore(v) + w_img * img


def rerank_order(base_scores, *, vlm=None, enc=None, **kw) -> np.ndarray:
    return np.argsort(-combine(base_scores, vlm=vlm, enc=enc, **kw), kind="stable")
