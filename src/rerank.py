"""
rerank.py — tầng re-rank: xáo lại thứ tự rổ ứng viên do CLIP/FAISS trả về.

VÌ SAO CẦN TẦNG NÀY (số đo trên 81 query bộ eval 2026, xem eval/queries_hcmc2026.csv)

    đáp án nằm ở hạng nào trong rổ CLIP:
        R@1 0.235 · R@5 0.395 · R@20 0.506 · R@50 0.593 · R@100 0.654
        R@200 0.741 · R@500 0.802 · R@1000 0.877 · R@2000 0.975

    71/81 query đã có đáp án trong rổ top-1000. Bài toán KHÔNG còn là "tìm không
    ra" mà là "xếp sai chỗ". Re-rank hoàn hảo trên rổ 100 cho FINAL 0.654, trên
    rổ 1000 cho 0.877 — so với 0.4765 đang có.

    Lưu ý bản chất: re-rank trên rổ top-K KHÔNG BAO GIỜ đổi R@K, nó chỉ chia lại
    thứ hạng bên trong rổ. Rổ càng sâu trần càng cao nhưng càng đắt.

⚠️ MỐC ĐÃ ĐỔI (2026-08-09): hệ thống chuyển từ ViT-B-32 của BTC sang
   ViT-L-16-SigLIP2-384/webli, nền đi từ 0.4765 lên **0.7802**. Mọi con số bên
   dưới đo trên nền CŨ — giữ lại vì bài học vẫn đúng (cái gì ăn, cái gì không),
   nhưng ĐỪNG trích như mốc hiện hành. Xem src/install_features.py.

ĐÃ THỬ NHỮNG GÌ (FINAL, nền CŨ = 0.4765)

    đa dạng hoá ≤3 khung/video      0.4346  ❌ lỗ nặng
    làm mượt thời gian              0.4148  ❌ lỗ nặng
    prior video từ chính CLIP       0.4914  ~nhiễu
    objects BM25 (score BTC cấp)    0.4840  ~nhiễu
    ASR mức video                   0.4790  ~nhiễu
    ── những cách trên đều KHÔNG thêm thông tin mới, hoặc chỉ thêm tín hiệu yếu ──
    encoder B-16 chấm lại rổ        0.5185  ✅
    VLM Gemini chấm lại rổ-100      0.5383  ✅
    ★ GHÉP CẢ HAI                   0.5531  ✅ +16% so với nền, R@1 0.235→0.383

    Bài học đọng lại: xáo lại thứ tự mà không mang THÔNG TIN MỚI vào thì lỗ, vì
    thứ tự của CLIP vốn đã là cách dùng tốt nhất thông tin của chính CLIP. Chỉ
    thứ vừa NHÌN được ảnh vừa ĐỌC được câu (VLM) hoặc nhìn ảnh bằng con mắt tinh
    hơn (encoder mạnh hơn) mới sửa được lỗi.
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

# Gửi bao nhiêu ảnh trong MỘT lần gọi Gemini. 20 là điểm cân bằng đã chạy thật:
# một lần gọi ~5k token (rẻ hơn nhiều so với 20 lần gọi lẻ) mà model vẫn chấm
# phân biệt được — trung bình chỉ 2,2/20 ảnh được 10 điểm, 3,2 mức điểm mỗi rổ.
VLM_BATCH = 20
# CLIP nhìn ảnh ở 224px nên gửi ảnh to hơn chỉ tốn token. 224 vừa đủ cho VLM.
VLM_PX = 224

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
    """Chuẩn hoá theo TỪNG HÀNG (mỗi hàng = một query).

    Bắt buộc phải có trước khi cộng hai nguồn điểm: cosine CLIP nằm quanh
    0,2-0,35 còn điểm VLM là 0-10. Cộng thẳng thì VLM nuốt trọn CLIP.
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
    """Mọi khoá Gemini có trong .env, theo thứ tự ưu tiên.

    Free tier bóp theo NGÀY, và một buổi đo có thể ngốn hơn 1.000 lần gọi (đã
    dính: hết hạn mức giữa lúc đang chạy eval, job kẹt trong vòng retry vô ích).
    Có khoá dự phòng thì đổi khoá rồi chạy tiếp, không mất công đã làm.

    Đặt thêm khoá bằng cách khai báo GEMINI_API_KEY_2, _3, … trong .env.
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
    """401/403 = khoá SAI hoặc chưa bật API ⇒ bỏ ngay, thử lại là phí.

    Khác hẳn 429: 429 có thể do chặn theo phút (chờ là qua), còn khoá sai thì
    chờ đến sáng cũng vẫn sai.
    """
    s = f"{type(e).__name__} {e}"
    return "401" in s or "403" in s or "UNAUTHENTICATED" in s or "PERMISSION_DENIED" in s


def _is_quota_error(e: Exception) -> bool:
    """429/RESOURCE_EXHAUSTED = hết hạn mức ⇒ đổi khoá. Lỗi khác thì thử lại.

    Phân biệt hai loại là cần thiết: 503 hay lỗi mạng thì đổi khoá vô ích (khoá
    nào cũng gặp), còn 429 thì thử lại bao nhiêu lần cũng vô ích.
    """
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
            # GIỮ client trong dict: nếu tạo rồi vứt (vd genai.Client(...).models...)
            # thì đối tượng bị thu hồi rác và httpx đóng kết nối, gây lỗi khó hiểu
            # "Cannot send a request, as the client has been closed". Đã dính.
            self._made[self.i] = genai.Client(api_key=self._keys[self.i])
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
    """Chấm 0-10 cho từng ảnh. Trả mảng cùng độ dài và cùng thứ tự image_paths.

    Dùng thẳng câu TIẾNG VIỆT — Gemini đọc tốt, đây là chỗ duy nhất trong cả
    pipeline mà tiếng Việt không phải điểm yếu (CLIP thì tiếng Việt = 0 tuyệt đối).

    Chia rổ thành nhiều mẻ `batch` ảnh: thang 0-10 là TUYỆT ĐỐI (không phải xếp
    hạng tương đối) nên điểm giữa các mẻ so được với nhau.
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
                        if verbose:
                            print("    [API] 429 — chờ 30s xem có phải chặn theo phút",
                                  flush=True)
                        time.sleep(30)
                        continue
                    if clients.retire() is None:
                        break
                    quota_hits = 0          # khoá mới, đếm lại từ đầu
                    continue
                if verbose:
                    print(f"    VLM lỗi ({attempt + 1}/4): {type(e).__name__} {str(e)[:120]}")
                if attempt < 3:
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
    """VLM tự báo 'đáp án không nằm trong rổ này'.

    ⚠️ **ĐỪNG DÙNG CHO NỀN SigLIP2 — dùng `basket_uncertain` thay thế.**
    Trên nền ViT-B-32 luật này đúng 83% (tỉ lệ nền 49%). Trên nền SigLIP2-L đo
    lại chỉ còn **31% chính xác, 31% độ phủ**: top-20 giờ sát nghĩa đến mức luôn
    có ảnh xứng đáng 10 điểm dù đúng khung vắng mặt. Giữ hàm này để đối chiếu
    lịch sử và cho trường hợp quay lại encoder yếu.

    Không có điểm nào hợp lệ (API hỏng) thì trả False: lúc đó ta KHÔNG BIẾT rổ
    tốt hay xấu, mà báo "đáp án không nằm trong rổ" là đổ oan cho dữ liệu trong
    khi lỗi nằm ở đường mạng. Bên gọi tự kiểm `all_failed()` để báo đúng nguyên nhân.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]
    return bool(len(s) and s.max() < threshold)


def basket_uncertain(base_scores, *, shallow: int = 20,
                     threshold: float = 0.12) -> bool:
    """Rổ nông có đáng ngờ không? Dựa trên BIÊN ĐỘ điểm giữa hạng 1 và hạng cuối.

    Trực giác: rổ "phẳng" nghĩa là encoder không phân biệt nổi 20 ứng viên —
    đúng lúc đáp án hay vắng mặt. Rổ "dốc" nghĩa là nó tự tin, thường là đúng.

    ĐO TRÊN 77 QUERY (SigLIP2-L), dự báo "đáp án KHÔNG nằm trong top-20":
        biên độ tương đối (s[0]-s[19])/s[0]   AUC **0.907**
        biên độ tuyệt đối s[0]-s[19]          AUC 0.918
        điểm top-1                            AUC 0.601
        đỉnh điểm VLM (luật cũ)               AUC 0.585
        số ảnh được VLM cho 10 điểm           AUC 0.538
    Trung bình biên độ: 0.02 khi hỏng vs 0.06 khi ổn — chênh 3 lần.

    Dùng biên độ TƯƠNG ĐỐI dù AUC nhỉnh thua: tuyệt đối gắn chặt với thang điểm
    của một encoder cụ thể (cosine SigLIP2 quanh 0.19, model khác sẽ khác) nên
    đổi encoder là phải dò lại ngưỡng. Tương đối thì bền hơn.

    Ngưỡng 0.12 ⇒ gắn cờ ~26% số query, bắt được **85%** query thật sự hỏng.

    ⚠️ ĐỪNG dùng lại luật cũ `basket_looks_wrong` cho việc này: nó đo trên nền
    ViT-B-32 (83% chính xác) nhưng sang nền SigLIP2 tụt còn **31%**, vì top-20
    giờ sát nghĩa đến mức luôn có ảnh xứng đáng 10 điểm dù đáp án thật vắng mặt.
    """
    s = np.asarray(base_scores, dtype=np.float64)[:shallow]
    if len(s) < 2 or s[0] <= 0:
        return False
    return bool((s[0] - s[-1]) / s[0] < threshold)


def vlm_scores_adaptive(query_vi: str, image_paths, base_scores, *,
                        shallow: int = 20, deep: int = 200,
                        threshold: float = 0.12, **kw) -> tuple[np.ndarray, bool]:
    """Chấm rổ NÔNG; chỉ đào sâu tới `deep` khi rổ nông đáng ngờ.

    Trả (điểm của những ảnh đã chấm, có đào sâu hay không). Phần chưa chấm không
    có trong mảng — bên gọi tự đệm NaN cho đuôi.

    VÌ SAO PHẢI CHỌN LỌC — đo trên mẫu 30 query (13 khó + 17 dễ) với rổ 200:
        nhóm KHÓ (đáp án ngoài top-20)    0.2000 -> 0.2545  (+0.055, kéo lên được)
        nhóm DỄ  (đáp án đã trong top-20) 0.8800 -> 0.8600  (−0.020, bị phá)
    Ngoại suy ra 81 query (13 khó, 68 dễ): lời +0.009, lỗ −0.017 ⇒ **tổng ≈ −0.008**.
    Đào sâu cho MỌI query là LỖ — rổ càng sâu càng nhiều cơ hội để VLM đẩy nhầm
    ảnh sai lên đỉnh (thấy thật: a22 tụt hạng 30 → 173).

    Cò quyết định (`basket_uncertain`) tính từ ĐIỂM TRUY XUẤT nên chạy TRƯỚC khi
    tiêu bất kỳ lần gọi API nào — query bình thường vẫn chỉ tốn 1 lần gọi.
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
    """Chấm lại rổ bằng một encoder KHÁC với encoder đang dùng để tìm kiếm.

    ⚠️ LỊCH SỬ — đọc kỹ trước khi dùng: hàm này sinh ra hồi hệ thống còn chạy
    vector ViT-B-32 của BTC (FINAL 0.4765), lúc đó chấm lại rổ bằng B-16 ăn
    +0.042. Từ 2026-08-09 hệ thống đã chuyển hẳn sang ViT-L-16-SigLIP2-384
    (FINAL 0.7802), nên "chấm lại bằng B-16/openai" KHÔNG còn ý nghĩa — nó yếu
    hơn hẳn thứ đang dùng. Mặc định vì thế đổi sang B-16-SigLIP2.

    ⚠️ CHẬM trên CPU (~2 ảnh/s → rổ 100 mất ~50 giây) và SigLIP2-384 còn chậm
    hơn nữa vì ảnh vào 384px. Đã có sẵn vector của cả corpus cho nhiều model
    trong data/kernel_out/ — dùng encoder_scores_precomputed() thì tức thì.
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
