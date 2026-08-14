"""Tầng chấm lại: xáo thứ tự rổ ứng viên do CLIP/FAISS trả về.

Chỉ đổi thứ tự BÊN TRONG rổ top-K, nên không bao giờ đổi được R@K — muốn đụng
R@K thì phải chấm toàn corpus (`src/ensemble.py`).

Danh sách những cách đã thử và bị loại, cùng cơ sở đo: xem
docs/bao_cao_he_thong.tex, mục "Tầng xếp hạng" và "Các hướng đã đóng bằng số đo".
"""

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
VLM_PX = 224          # CLIP nhìn ảnh ở 224px, gửi to hơn chỉ tốn token
VLM_TIMEOUT_MS = 90_000   # chỉ để chặn treo hẳn; một mẻ bình thường mất 5-8 giây

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
    """Chuẩn hoá theo từng hàng. Bắt buộc trước khi cộng hai nguồn điểm:
    cosine CLIP quanh 0,2-0,35 còn điểm VLM là 0-10.
    """
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
    """Mọi khoá Gemini trong .env, theo thứ tự ưu tiên.

    Thêm khoá bằng GEMINI_API_KEY_2, _3, … — free tier bóp theo ngày và một buổi
    đo ngốn hơn 1.000 lần gọi.
    """
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
            # Phải GIỮ tham chiếu client, và phải có timeout — xem sổ bẫy trong
            # báo cáo. Cả hai lỗi này đều biểu hiện thành thứ khác hẳn nguyên nhân.
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
    """Chấm 0-10 cho từng ảnh, cùng thứ tự `image_paths`.

    Nhận thẳng câu tiếng Việt. Chia rổ thành nhiều mẻ được vì thang 0-10 là tuyệt
    đối chứ không phải xếp hạng tương đối, nên điểm giữa các mẻ so được với nhau.
    """
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
                # Thỉnh thoảng model trả thiếu 1 điểm (2/405 lần khi chạy thật).
                # Bắt lỗi rồi gọi lại là qua, đừng cố vá bằng cách đệm 0.
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
                    # 429 có HAI loại và phải xử lý khác nhau:
                    #   - chặn theo PHÚT  -> chỉ cần chờ, khoá vẫn tốt
                    #   - cạn hạn mức NGÀY -> chờ bao lâu cũng vô ích, phải đổi khoá
                    # Thông báo lỗi không phân biệt rõ hai loại, nên: lần 429 đầu
                    # thì CHỜ rồi thử lại; vẫn 429 nữa mới coi là cạn ngày.
                    # (Loại bỏ khoá ngay từ lần đầu là vứt oan một khoá còn tốt —
                    # đã dính, khiến cả lượt eval hỏng.)
                    quota_hits += 1
                    if quota_hits == 1:
                        # In BẤT KỂ verbose. Đây là lúc job đứng im 30 giây mà
                        # không tiêu CPU — nếu không in thì nó trông y hệt một job
                        # treo cứng, và người đang xem sẽ đi chẩn đoán nhầm chỗ
                        # (đã mất thời gian vì đúng chuyện này).
                        print("    [API] 429 — chờ 30s xem có phải chặn theo phút",
                              flush=True)
                        time.sleep(30)
                        continue
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
            # Hỏng cả 4 lần: trả NaN cho mẻ này. KHÔNG cache thất bại — cache
            # giá trị hỏng là cái bẫy đã dính một lần ở module dịch.
            out.extend([np.nan] * len(chunk))
        else:
            cache.put(ck, got)
            out.extend(got)
            time.sleep(sleep)

    return np.asarray(out, dtype=np.float64)


def basket_looks_wrong(scores: np.ndarray, threshold: float = 10.0) -> bool:
    """Luật CŨ: đỉnh điểm VLM < 10 ⇒ đáp án không nằm trong rổ.

    ⚠️ Đừng dùng cho nền SigLIP2 — chỉ còn 31% chính xác (nền B-32 là 83%).
    Dùng `basket_uncertain`. Giữ lại để đối chiếu và cho encoder yếu.

    API hỏng hết thì trả False: lúc đó ta KHÔNG BIẾT rổ tốt hay xấu, báo "đáp án
    không có trong rổ" là đổ oan cho dữ liệu. Bên gọi tự kiểm `all_failed()`.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]
    return bool(len(s) and s.max() < threshold)


def basket_uncertain(base_scores, *, shallow: int = 20,
                     threshold: float = 0.12) -> bool:
    """Rổ nông có đáng ngờ không, dựa trên biên độ điểm giữa hạng 1 và hạng cuối.

    Rổ "phẳng" = encoder không phân biệt nổi 20 ứng viên = đáp án hay vắng mặt.
    AUC 0.907; ngưỡng 0.12 gắn cờ ~26% query và bắt được 85% query thật sự hỏng.

    Dùng biên độ TƯƠNG ĐỐI dù AUC nhỉnh thua tuyệt đối: tuyệt đối gắn với thang
    cosine của một encoder cụ thể nên đổi encoder là phải dò lại ngưỡng.
    """
    s = np.asarray(base_scores, dtype=np.float64)[:shallow]
    if len(s) < 2 or s[0] <= 0:
        return False
    return bool((s[0] - s[-1]) / s[0] < threshold)


def vlm_scores_adaptive(query_vi: str, image_paths, base_scores, *,
                        shallow: int = 20, deep: int = 200,
                        threshold: float = 0.12, **kw) -> tuple[np.ndarray, bool]:
    """Chấm rổ nông; chỉ đào sâu tới `deep` khi rổ nông đáng ngờ.

    Trả (điểm của những ảnh đã chấm, có đào sâu hay không); bên gọi tự đệm NaN
    cho đuôi. Đào sâu cho MỌI query là lỗ, nên phải có cò — và cò tính từ điểm
    truy xuất nên chạy trước khi tiêu lần gọi API nào.

    Mặc định TẮT ở giao diện: đào sâu chỉ lợi cho người tự tìm, hại cho danh sách
    nộp xếp tự động.
    """
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
    """Chấm lại rổ bằng một encoder khác encoder tìm kiếm.

    ⚠️ Chậm trên CPU (~2 ảnh/s, rổ 100 mất ~50 giây). Vector cả corpus đã có sẵn
    trong data/kernel_out/ — dùng `encoder_scores_precomputed()` thì tức thì.
    """
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
    """Trộn các nguồn điểm thành MỘT điểm cuối. Điểm cao = xếp trước.

    Công thức và trọng số đều lấy từ đo thật trên 81 query, rổ top-100:

        thang ảnh  = z(B-32) + w_enc·z(encoder mới)        w_enc=2.0 -> 0.5185
        điểm cuối  = z(VLM)  + w_img·z(thang ảnh)          w_img=0.5 -> 0.5531

    `w_img=None` = tự chọn theo có encoder hay không, và ĐÂY LÀ ĐIỀU CẦN THIẾT
    chứ không phải tiện tay: khi chỉ có B-32 thì thang ảnh chính là thứ VLM vừa
    sửa sai, cộng lại nhiều là kéo ngược. Đo được: không encoder thì w_img 0.5
    làm R@1 tụt 0.333 → 0.321, còn 0.0-0.2 giữ nguyên 0.333. Có encoder thì
    ngược lại, 0.5 mới đạt 0.5531.

    Vì sao ghép được mà không giẫm chân nhau: VLM chỉ phát ~3 mức điểm rời rạc
    nên trung bình 9,8 ảnh HOÀ NHAU ở đỉnh mỗi rổ 100. Thang ảnh mịn hơn nhiều
    nên nó phá hoà. Nói gọn: VLM quyết TẦNG, encoder quyết THỨ TỰ TRONG TẦNG.
    Bằng chứng: phá hoà bằng B-32 cho 0.5383, bằng B-16 cho 0.5506.

    ⚠️ ĐỪNG dùng RRF để ghép VLM với CLIP — thử rồi, R@1 tụt về đúng 0.235 tức
    bằng CLIP thuần, xoá sạch cái lời. RRF chỉ nhìn thứ hạng nên vứt mất thông
    tin "ảnh này 10 điểm còn ảnh kia 3 điểm", mà đó chính là thứ VLM đóng góp.
    """
    if w_img is None:
        w_img = 0.5 if enc is not None else 0.2

    img = zscore(base_scores)
    if enc is not None:
        # Chuẩn hoá LẠI sau khi cộng: z(B32) + 2·z(B16) có độ lệch chuẩn ~2,5
        # chứ không phải 1, nên nếu không chuẩn hoá lại thì w_img bên dưới không
        # còn đọc đúng nghĩa (0.5 hoá ra chỉ bằng ~0.17). Đã dính lỗi này một lần.
        img = zscore(img + w_enc * zscore(enc))
    if vlm is None:
        return img

    v = np.asarray(vlm, dtype=np.float64)
    if np.isnan(v).all():
        return img
    # Mẻ nào VLM hỏng thì cho điểm trung bình -> không được ưu ái cũng không bị
    # phạt, thứ hạng của chúng do thang ảnh quyết.
    v = np.where(np.isnan(v), np.nanmean(v), v)
    return zscore(v) + w_img * img


def rerank_order(base_scores, *, vlm=None, enc=None, **kw) -> np.ndarray:
    """Trả chỉ số sắp xếp lại (mảng vị trí trong rổ, tốt nhất trước).

    `kind="stable"` là BẮT BUỘC chứ không phải cho đẹp: rất nhiều ảnh hoà điểm,
    và khi hoà thì phải giữ nguyên thứ tự CLIP ban đầu. Dùng sort không ổn định
    là mất phần lớn cái lời đo được.
    """
    return np.argsort(-combine(base_scores, vlm=vlm, enc=enc, **kw), kind="stable")
