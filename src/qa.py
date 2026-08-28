
from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

QA_PX = int(os.environ.get("QA_PX", 512))

PROMPT = """Bạn đang giúp thí sinh trả lời một câu hỏi về video.

Mô tả sự kiện: "{desc}"
CÂU HỎI: "{question}"

Dưới đây là {k} ảnh trích từ video, đánh số 1..{k} theo thứ tự gửi.

Nhiệm vụ:
1. Chọn ảnh trả lời được câu hỏi RÕ NHẤT. Ảnh đúng chủ đề nhưng không nhìn thấy
   thứ được hỏi thì KHÔNG chọn — hãy chọn ảnh thấy rõ chi tiết cần trả lời.
2. Trả lời câu hỏi, NGẮN GỌN, bằng tiếng Việt. Chỉ nêu thông tin nhìn thấy được
   trong ảnh. Nếu là màu sắc, nói rõ "xanh lá" hay "xanh dương", đừng nói trống
   "xanh". Nếu không ảnh nào trả lời được, để answer là "" và confidence 0.
3. Cho biết mức chắc chắn 0-10.

CHỈ trả JSON:
{{"best": <số thứ tự ảnh 1..{k}>, "answer": "<câu trả lời tiếng Việt>",
  "confidence": <0-10>, "reason": "<một câu ngắn vì sao chọn ảnh đó>"}}"""

# Lời nói gắn theo TỪNG ảnh, không gộp thành một khối: rổ 20 ảnh thường
# chứa nhiều video, mỗi video một tên riêng khác nhau.
PROMPT_RO_ASR = """Bạn đang giúp thí sinh trả lời một câu hỏi về video.

Mô tả sự kiện: "{desc}"
CÂU HỎI: "{question}"

Dưới đây là {k} ảnh trích từ video, đánh số 1..{k} theo thứ tự gửi.
Kèm theo là lời thuyết minh đọc ĐÚNG LÚC từng ảnh xuất hiện:

{loinoi}

Nhiệm vụ:
1. Chọn ảnh khớp mô tả sự kiện RÕ NHẤT.
2. Trả lời câu hỏi, NGẮN GỌN, bằng tiếng Việt. Được lấy đáp án từ lời thuyết
   minh, NHƯNG CHỈ lời thuyết minh của CHÍNH ảnh đã chọn ở bước 1 — mỗi ảnh
   thường thuộc một bản tin khác nhau, lấy nhầm dòng là trả lời sai sự kiện.
   Nếu không ảnh nào trả lời được, để answer là "" và confidence 0.
3. Cho biết mức chắc chắn 0-10.

CHỈ trả JSON:
{{"best": <số thứ tự ảnh 1..{k}>, "answer": "<câu trả lời tiếng Việt>",
  "confidence": <0-10>, "reason": "<một câu ngắn vì sao chọn ảnh đó>"}}"""


# Bản có thêm chữ OCR. Tách hẳn khỏi PROMPT_RO_ASR chứ không sửa tại chỗ, để
# các phép đo đã chạy trên prompt kia còn tái lập được nguyên văn.
PROMPT_RO_PHU = """Bạn đang giúp thí sinh trả lời một câu hỏi về video.

Mô tả sự kiện: "{desc}"
CÂU HỎI: "{question}"

Dưới đây là {k} ảnh trích từ video, đánh số 1..{k} theo thứ tự gửi.
Kèm theo, cho từng ảnh, là lời thuyết minh đọc đúng lúc ảnh đó xuất hiện và
chữ mà máy OCR đọc được trên chính ảnh đó:

{loinoi}

Nhiệm vụ:
1. Chọn ảnh khớp mô tả sự kiện RÕ NHẤT.
2. Trả lời câu hỏi, NGẮN GỌN, bằng tiếng Việt. Được lấy đáp án từ lời thuyết
   minh hoặc chữ OCR, NHƯNG CHỈ của CHÍNH ảnh đã chọn ở bước 1 — mỗi ảnh
   thường thuộc một bản tin khác nhau, lấy nhầm dòng là trả lời sai sự kiện.
   Máy OCR không bỏ dấu tiếng Việt và hay dính chữ; đọc nó như chữ gần đúng,
   và tin ĐIỂM ẢNH hơn khi hai bên đá nhau. Nếu không ảnh nào trả lời được,
   để answer là "" và confidence 0.
3. Cho biết mức chắc chắn 0-10.

CHỈ trả JSON:
{{"best": <số thứ tự ảnh 1..{k}>, "answer": "<câu trả lời tiếng Việt>",
  "confidence": <0-10>, "reason": "<một câu ngắn vì sao chọn ảnh đó>"}}"""


def ask(question_vi: str, image_paths, *, desc: str = "", model: str | None = None,
        px: int | None = None, retries: int = 5, verbose: bool = False,
        loi_noi: list[str] | None = None,
        chu: list[str] | None = None,
        sua_ten: bool = True) -> dict | None:
    from google.genai import types
    import rerank

    clients = rerank._CLIENTS            # dùng chung bể khoá + cơ chế đổi khoá
    if not clients:
        raise RuntimeError("Không thấy GEMINI_API_KEY trong .env")

    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    px = px or QA_PX
    paths = list(image_paths)

    co_chu = bool(chu) and any((t or "").strip() for t in chu)
    if loi_noi and any(t.strip() for t in loi_noi):
        dong = []
        for i, t in enumerate(loi_noi[:len(paths)], 1):
            d = f"Ảnh {i}: {(t or '(không có lời)').strip()[:400]}"
            if co_chu:
                c = (chu[i - 1] if i - 1 < len(chu) else "") or ""
                c = " · ".join(c.split(chr(10)))[:200]
                d += f"{chr(10)}   [chữ trên khung] {c or '(không quét được)'}"
            dong.append(d)
        keo = chr(10).join(dong)
        mau = PROMPT_RO_PHU if co_chu else PROMPT_RO_ASR
        dau = mau.format(desc=desc or "(không có)", question=question_vi,
                         k=len(paths), loinoi=keo)
        if sua_ten:
            dau += NHAC_SUA_TEN
    else:
        dau = PROMPT.format(desc=desc or "(không có)", question=question_vi,
                            k=len(paths))
    parts = [types.Part.from_text(text=dau)]
    parts += [types.Part.from_bytes(data=rerank._thumb(p, px), mime_type="image/jpeg")
              for p in paths]

    quota_hits = 0
    for attempt in range(retries):
        client = clients.current()
        if client is None:
            break
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json"))
            d = json.loads(resp.text)
            b = int(d.get("best", 0)) - 1                 # 1-based -> 0-based
            if not (0 <= b < len(paths)):
                raise ValueError(f"best={d.get('best')} ngoài khoảng 1..{len(paths)}")
            ans = str(d.get("answer", "")).strip()
            conf = float(d.get("confidence", 0))
            # Prompt BẢO mô hình để answer rỗng khi không ảnh nào trả lời được.
            # Coi đó là lỗi thì mỗi lần bó tay tốn 5 lời gọi và 200 giây chờ —
            # mà thử lại cũng ra đúng câu rỗng đó. Rỗng + confidence 0 là câu
            # trả lời hợp lệ ("không đủ bằng chứng"), chỉ rỗng mà tự tin mới sai.
            if not ans and conf > 0:
                raise ValueError("model trả câu trả lời rỗng mà confidence > 0")
            return {"best_index": b,
                    "answer": ans,
                    "confidence": conf,
                    "reason": str(d.get("reason", "")).strip()}
        except Exception as e:
            if rerank._is_auth_error(e):      # khoá sai -> bỏ ngay
                if clients.retire(verbose=False) is None:
                    break
                continue
            if rerank._is_quota_error(e):
                quota_hits += 1
                if quota_hits == 1:
                    print(f"    [API] 429 ở khoá #{clients.i + 1} — chờ 30s rồi "
                          f"thử LẠI chính khoá này", flush=True)
                    time.sleep(30)
                    continue                  # thử lại CÙNG khoá
                if clients.retire() is None:
                    break
                quota_hits = 0                # khoá mới, đếm lại từ đầu
                continue
            print(f"  [API] Q&A lỗi ({attempt + 1}/{retries}): "
                  f"{type(e).__name__} {str(e)[:120]}", flush=True)
            if attempt < retries - 1:
                time.sleep(20 * (attempt + 1))
    return None


# ───────────────── chọn kênh theo Ý ĐỊNH của câu hỏi ─────────────────
# Đo được: gắn lời nói cho MỌI câu là lỗ. Câu tên riêng +0,400, nhưng câu
# nhìn -0,125 và câu đọc chữ -0,200 — chữ phụ trợ kéo mô hình rời mắt khỏi
# ảnh. Nên kênh phải bật theo loại câu, không bật đại trà.

# "ghi " một mình bắt nhầm "ghi NHẬN" — câu đếm động đất bị đẩy sang kênh chữ.
# Dấu hiệu chữ-trên-màn-hình phải là cụm CỤ THỂ, không phải một âm tiết.
_TU_CHU = ("chữ", "ghi gì", "ghi tên", "ghi số", "ghi dòng", "ghi là",
           "có ghi", "biển", "bảng tên", "số hiệu", "hiển thị",
           "màn hình", "khắc", "logo", "phông nền", "nhãn", "vỏ hộp",
           "đăng ký", "dòng nào", "thanh thông tin")
_TU_TEN = ("tên", "gọi là", "ở đâu", "nhắc tới", "nhắc đến", "địa danh",
           "tỉnh nào", "huyện nào", "xã nào", "thuộc đâu")


# Máy nghe viết sai tên riêng hiếm: đo được 15% (3/20) nhãn trong bộ tên
# riêng chính là lỗi Whisper — "Tằng Quái" thành "Tăng Quái", "Bông Lau" thành
# "Bông Lâu", "Ông Chưởng" thành "Ông Trưởng". Mô hình ngôn ngữ biết địa danh
# thật, nên bảo nó viết lại cho đúng rẻ hơn nhiều so với đổi bộ nhận giọng.
NHAC_SUA_TEN = (
    chr(10) * 2 +
    "LƯU Ý VỀ CHÍNH TẢ: lời thuyết minh do máy nghe tự động ghi lại, "
    "nên tên riêng hiếm hay sai dấu hoặc sai phụ âm. Nếu bạn nhận ra đó là "
    "một địa danh có thật, hãy viết lại theo CHÍNH TẢ ĐÚNG của địa danh đó "
    "(ví dụ nghe ra \"đèo Tăng Quái\" thì viết \"Tằng Quái\"). Chỉ sửa khi "
    "bạn thật sự nhận ra tên; không nhận ra thì chép nguyên văn."
)


def chon_kenh(question_vi: str) -> dict:
    """Câu hỏi này cần kênh phụ trợ nào. Trả {"asr": bool, "ocr": bool}.

    Chữ-trên-màn-hình thắng trước: "tấm biển ghi TÊN chương trình là gì" có
    cả hai dấu hiệu, nhưng đáp án nằm trên biển chứ không trong lời đọc.
    """
    c = str(question_vi).lower()
    if any(t in c for t in _TU_CHU):
        return {"asr": False, "ocr": True}
    if any(t in c for t in _TU_TEN):
        return {"asr": True, "ocr": False}
    return {"asr": False, "ocr": False}


def ro_goi_y(question_vi: str) -> int:
    """Rổ nên lấy bao nhiêu ảnh cho câu hỏi này.

    Đo 28/08, cùng bộ câu, chỉ đổi cỡ rổ:
      kênh ảnh thuần   20 -> 30 : 0,620 -> 0,680  (+0,060)
      kênh lời nói     20 -> 30 : 0,450 -> 0,350  (-0,100)
    Rổ to mua thêm độ phủ (0,767 -> 0,837 trên 86 câu), nhưng khi kênh phụ trợ
    bật thì mỗi ảnh thêm cũng kéo theo MỘT DÒNG CHỮ thêm, và lấy nhầm dòng đắt
    hơn khoản độ phủ mua được.
    """
    t = chon_kenh(question_vi)
    return 20 if (t["asr"] or t["ocr"]) else 30


def answer_over_hits(question_vi: str, hits, *, desc: str = "",
                     top: int | None = None, frame_from: str = "clip",
                     kem_asr: bool | None = None,
                     kem_ocr: bool | None = None, **kw) -> dict | None:
    # None = tự chọn theo ý định câu hỏi; True/False/số = người gọi ép tay.
    tu = chon_kenh(question_vi)
    kem_asr = tu["asr"] if kem_asr is None else kem_asr
    kem_ocr = tu["ocr"] if kem_ocr is None else kem_ocr
    if top is None:
        top = 20 if (kem_asr or kem_ocr) else 30
    sub = hits.iloc[:top]
    if kem_asr and "loi_noi" not in kw:
        kw["loi_noi"] = [asr_khung(v, int(f)) for v, f
                         in zip(sub["video_id"], sub["frame_idx"])]
    if kem_ocr and "chu" not in kw:
        kw["chu"] = [ocr_chi_muc(v, int(n)) for v, n
                     in zip(sub["video_id"], sub["n"])]
    kw.setdefault("sua_ten", kem_asr)
    got = ask(question_vi, sub["image_path"].tolist(), desc=desc, **kw)
    if got is None:
        return None

    vrow = sub.iloc[got["best_index"]]
    got["vlm_frame_idx"] = int(vrow["frame_idx"])
    got["vlm_video_id"] = str(vrow["video_id"])

    row = sub.iloc[0] if frame_from == "clip" else vrow
    got.update(video_id=str(row["video_id"]), frame_idx=int(row["frame_idx"]),
               n=int(row["n"]), image_path=str(row["image_path"]),
               frame_from=frame_from,
               kenh=("lời nói" if kem_asr else "chữ trên khung" if kem_ocr
                     else "chỉ ảnh"))
    return got


def submission_rows(hits, chosen: dict | None, *, limit: int = 100) -> list[tuple]:
    ans = (chosen or {}).get("answer", "")
    rows, seen = [], set()
    if chosen:
        k = (chosen["video_id"], chosen["frame_idx"])
        rows.append((*k, ans))
        seen.add(k)
    for r in hits.itertuples():
        k = (str(r.video_id), int(r.frame_idx))
        if k in seen:
            continue
        rows.append((*k, ans))
        seen.add(k)
        if len(rows) >= limit:
            break
    return rows


# ─────────────────────────── kênh lời nói (ASR) ───────────────────────────

_ASR = None


def _bang_asr():
    """Bảng ASR đã căn sẵn theo từng keyframe, nạp một lần rồi giữ luôn."""
    global _ASR
    if _ASR is None:
        import pandas as pd
        f = ROOT / "data" / "processed_hcmc2026" / "asr_by_keyframe.parquet"
        d = pd.read_parquet(f, columns=["video_id", "frame_idx", "asr_text"])
        _ASR = {(v, int(i)): t or "" for v, i, t in d.itertuples(index=False)}
    return _ASR


_OCR_IDX = None


def _bang_ocr():
    """Chỉ mục chữ đã quét sẵn, tra theo (video, n). Nạp một lần rồi giữ."""
    global _OCR_IDX
    if _OCR_IDX is None:
        import pandas as pd
        f = ROOT / "data" / "processed_hcmc2026" / "ocr_chu.parquet"
        if not f.exists():
            _OCR_IDX = {}
        else:
            d = pd.read_parquet(f, columns=["video_id", "n", "text"])
            _OCR_IDX = {(v, int(n)): t or "" for v, n, t
                        in d.itertuples(index=False)}
    return _OCR_IDX


def ocr_chi_muc(video_id: str, n: int) -> str:
    """Chữ đã quét sẵn của một khung, "" nếu khung chưa nằm trong chỉ mục."""
    return _bang_ocr().get((str(video_id), int(n)), "")


def asr_khung(video_id: str, frame_idx: int) -> str:
    """Lời nói phủ đúng khung này. 88,5% khung có chữ, trung bình 412 ký tự."""
    return _bang_asr().get((str(video_id), int(frame_idx)), "")



# ──────────────────────────── kênh OCR một khung ────────────────────────────

_OCR = None
OCR_SIDE = int(os.environ.get("QA_OCR_SIDE", 736))
OCR_TIN = float(os.environ.get("QA_OCR_TIN", 0.55))

PROMPT_OCR = """Một máy OCR vừa quét khung hình này và đọc ra các mẩu chữ sau,
xếp theo thứ tự từ trên xuống:

{ocr}

CÂU HỎI: "{question}"

Lưu ý về máy OCR này:
- Nó KHÔNG có dấu tiếng Việt: "THAM PHAN" thật ra là "THẨM PHÁN".
- Nó hay lẫn ký tự giống nhau: 8 thành B, 0 thành O, 1 thành l.
- Nó đọc cả chữ của nhà đài (logo, đồng hồ, dải tin chạy) --- những thứ đó
  KHÔNG phải nội dung cảnh quay, đừng lấy làm đáp án.

Chỉ trả lời nếu các mẩu chữ trên CHỨA đáp án. Hãy khôi phục lại dấu và sửa ký
tự bị đọc nhầm. Không chứa thì để answer rỗng.

CHỈ trả JSON: {{"answer": "<đáp án hoặc rỗng>", "confidence": <0-10>}}"""


def _may_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR(det_limit_side_len=OCR_SIDE)
    return _OCR


def ocr_khung(image_path, tin: float = OCR_TIN) -> str:
    """Chữ đọc được trong một khung, mỗi mẩu một dòng. Khoảng 2,8 giây trên CPU.

    Không lập chỉ mục cả corpus được: 177.321 khung ở tốc độ này là 140 giờ một
    luồng. Nên OCR chỉ chạy đúng khung người thi đã chốt.
    """
    try:
        kq, _ = _may_ocr()(str(image_path))
    except Exception:
        return ""
    mau = [c for c in (kq or []) if float(c[2]) >= tin]
    mau.sort(key=lambda c: (min(p[1] for p in c[0]), min(p[0] for p in c[0])))
    return "\n".join(f"- {c[1]}" for c in mau)

# ────────────────────── đọc kỹ một khung: ô + bỏ phiếu ──────────────────────

O_CANH = int(os.environ.get("QA_O_CANH", 512))
O_BUOC = int(os.environ.get("QA_O_BUOC", 384))
O_PHONG = float(os.environ.get("QA_O_PHONG", 1.0))
QA_TOAN_PX = int(os.environ.get("QA_TOAN_PX", 768))

PROMPT_O = """Bạn đang trả lời một câu hỏi về MỘT khung hình video.
{ngu_canh}CÂU HỎI: "{question}"

Ảnh kèm đây là {nhan}.
Quy tắc bắt buộc:
- Chỉ trả lời nếu NHÌN THẤY RÕ trong chính ảnh này thứ được hỏi.
- Không thấy, hoặc phải đoán, thì để answer là "" (rỗng) — đó là câu trả lời hợp lệ.
- Không dùng kiến thức bên ngoài, không suy diễn từ phần ảnh bị cắt mất.
- Trả lời NGẮN GỌN bằng tiếng Việt. Màu thì nói rõ "xanh lá" hay "xanh dương".

CHỈ trả JSON: {{"answer": "<đáp án hoặc rỗng>", "confidence": <0-10>}}"""

PROMPT_ASR = """Dưới đây là lời thuyết minh đọc đúng lúc khung hình này xuất hiện.

LỜI THUYẾT MINH: "{asr}"

CÂU HỎI: "{question}"

Chỉ trả lời nếu lời thuyết minh NÓI THẲNG ra đáp án. Suy đoán thì để answer rỗng.
Lời thuyết minh do máy nhận dạng nên tên riêng có thể sai chính tả — hãy sửa về
dạng đúng nếu bạn nhận ra tên đó.

CHỈ trả JSON: {{"answer": "<đáp án hoặc rỗng>", "confidence": <0-10>}}"""


def _cat_o(path, canh: int = O_CANH, buoc: int = O_BUOC):
    """Cắt ảnh thành các ô vuông chồng lấn, kiểu cửa sổ trượt của DAM-QA."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W, H = im.size
    canh = min(canh, W, H)

    def moc(n):
        r = list(range(0, max(n - canh, 0) + 1, buoc))
        if r[-1] != n - canh:
            r.append(n - canh)
        return r

    ra = []
    for y in moc(H):
        for x in moc(W):
            ra.append((im.crop((x, y, x + canh, y + canh)), (x, y)))
    return ra, (W, H)


def _nen(im, px: int | None = None) -> bytes:
    from PIL import Image
    if px:
        im = im.copy()
        im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _chuan(s: str) -> str:
    """Gom các cách viết cùng một đáp án về một khoá để cộng phiếu."""
    s = re.sub(r"[\s.,;:!?\"'()\[\]-]+", " ", str(s).lower()).strip()
    return re.sub(r"^(là|đó là|có|khoảng)\s+", "", s)


def _hoi_mot(prompt: str, anh: bytes | None, *, model=None, retries=3) -> dict | None:
    """Một lời gọi VLM trả JSON {answer, confidence}; dùng chung bể khoá."""
    from google.genai import types
    import rerank

    clients = rerank._CLIENTS
    if not clients:
        raise RuntimeError("Không thấy GEMINI_API_KEY trong .env")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    parts = [types.Part.from_text(text=prompt)]
    if anh is not None:
        parts.append(types.Part.from_bytes(data=anh, mime_type="image/jpeg"))

    for attempt in range(retries):
        client = clients.current()
        if client is None:
            return None
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json"))
            d = json.loads(resp.text)
            return {**d, "answer": str(d.get("answer", "")).strip(),
                    "confidence": float(d.get("confidence", 0) or 0)}
        except Exception as e:
            if rerank._is_auth_error(e):
                if clients.retire(verbose=False) is None:
                    return None
                continue
            if rerank._is_quota_error(e):
                if attempt == 0:
                    time.sleep(30)
                    continue
                if clients.retire() is None:
                    return None
                continue
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
    return None


def doc_ky(question_vi: str, image_path, *, desc: str = "", asr: str | None = None,
           canh: int = O_CANH, buoc: int = O_BUOC, w_asr: float = 1.0,
           ocr: bool = True, w_ocr: float = 0.5, phong: float = O_PHONG,
           verbose: bool = False, **kw) -> dict | None:
    """Đọc MỘT khung bằng nhiều góc nhìn rồi bỏ phiếu có trọng số.

    Khung 1280x720 gửi ở 512px chỉ còn 512x288 — số trên biển báo mất hẳn. Ở đây
    ảnh toàn cảnh đi kèm các ô cắt ở độ phân giải GỐC, mỗi ô một lời gọi riêng.
    Trọng số theo diện tích, phiếu "không thấy" tính 0 — theo DAM-QA (ICCVW 2025).
    """
    from PIL import Image
    ngu_canh = ('Bối cảnh: "%s"' % desc + "\n") if desc else ""
    goc = []

    toan = Image.open(image_path).convert("RGB")
    W, H = toan.size
    # Trọng số = phần diện tích nhìn thấy x (độ phóng so với ảnh gốc)^2. Toàn
    # cảnh thấy hết nhưng bị thu nhỏ nên chi tiết nhỏ mờ đi; ô cắt thấy ít mà
    # giữ nguyên điểm ảnh. Chỉ tính diện tích thì toàn cảnh đọc sai vẫn thắng.
    ti_toan = min(1.0, QA_TOAN_PX / float(max(W, H)))
    goc.append(("toàn cảnh", ti_toan ** 2,
                PROMPT_O.format(ngu_canh=ngu_canh, question=question_vi,
                                nhan="TOÀN BỘ khung hình"),
                _nen(toan, QA_TOAN_PX)))

    o, _ = _cat_o(image_path, canh, buoc)
    w_o = (canh * canh) / float(W * H)
    for im, (x, y) in o:
        goc.append((f"ô ({x},{y})", w_o,
                    PROMPT_O.format(
                        ngu_canh=ngu_canh, question=question_vi,
                        nhan=f"một PHẦN CẮT của khung hình, vùng ({x},{y}) "
                             f"kích thước {canh}x{canh} trong ảnh gốc {W}x{H}"),
                    _nen(im if phong == 1 else im.resize(
                        (int(im.width * phong), int(im.height * phong)),
                        Image.LANCZOS))))

    if asr:
        goc.append(("lời nói", w_asr,
                    PROMPT_ASR.format(asr=asr[:2000], question=question_vi), None))

    if ocr:
        chu = ocr_khung(image_path)
        if chu:
            goc.append(("chữ OCR", w_ocr,
                        PROMPT_OCR.format(ocr=chu[:2000], question=question_vi),
                        None))

    phieu, chi_tiet = {}, []
    for nhan, w, prompt, anh in goc:
        d = _hoi_mot(prompt, anh, **kw)
        ans = (d or {}).get("answer", "")
        chi_tiet.append({"view": nhan, "answer": ans,
                         "confidence": (d or {}).get("confidence", 0), "weight": w})
        if verbose:
            print(f"    {nhan:>16} · {ans or '(không thấy)'}", flush=True)
        if not ans:                       # phiếu "không thấy" nặng 0, đúng DAM-QA
            continue
        k = _chuan(ans)
        v = phieu.setdefault(k, {"ans": ans, "w_max": 0.0, "tong": 0.0})
        if w > v["w_max"]:                # giữ cách viết của góc nhìn nặng nhất
            v["ans"], v["w_max"] = ans, w
        v["tong"] += w

    if not phieu:
        return None
    tot = max(phieu.values(), key=lambda v: v["tong"])
    ans, diem = tot["ans"], tot["tong"]
    tong = sum(v["tong"] for v in phieu.values())
    return {"answer": ans, "confidence": round(10 * diem / tong, 1),
            "score": diem, "n_view": len(goc), "views": chi_tiet,
            "reason": f"{diem:.2f}/{tong:.2f} phiếu trên {len(goc)} góc nhìn"}


# ─────────────────────── câu ĐẾM: chia ô rồi cộng dồn ───────────────────────

PROMPT_DEM = """Bạn đang đếm vật trên MỘT PHẦN CẮT của khung hình video.
{ngu_canh}CẦN ĐẾM: {vat}

Phần cắt này là vùng ({x},{y}) kích thước {w}x{h} trong ảnh gốc {W}x{H}.
Chỉ đếm những thứ nằm TRONG phần cắt này. Không đoán phần bị cắt mất.
Không có cái nào thì trả 0.

CHỈ trả JSON: {{"count": <số nguyên>, "confidence": <0-10>}}"""


def dem_o(vat: str, image_path, *, desc: str = "", hang: int = 2, cot: int = 3,
          bo_qua=None, verbose: bool = False, **kw) -> dict | None:
    """Đếm bằng cách chia ô KHÔNG chồng lấn rồi CỘNG, không bỏ phiếu.

    Bỏ phiếu chỉ đúng cho câu nhận dạng; câu đếm mà bỏ phiếu thì các ô đều thấy
    ít nên đáp án chung cuộc bé hơn cả ô lớn nhất. Đo 23/08: bỏ phiếu ra 0.
    Ô không chồng lấn nên không có vật nào bị đếm hai lần.
    """
    from PIL import Image
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    ngu_canh = ('Bối cảnh: "%s"' % desc + "\n") if desc else ""

    # Hỏi toàn khung trước: vật to và ít thì đây mới là câu trả lời đúng.
    d0 = _hoi_mot(PROMPT_DEM.format(
        ngu_canh=ngu_canh, vat=vat, x=0, y=0, w=W, h=H, W=W, H=H),
        _nen(im, QA_TOAN_PX), **kw)
    try:
        toan = max(0, int(float((d0 or {}).get("count", 0) or 0)))
    except (TypeError, ValueError):
        toan = 0
    if verbose:
        print(f"    toàn khung · {toan}", flush=True)

    tong, chi_tiet, thieu = 0, [], 0
    for r in range(hang):
        for c in range(cot):
            x, y = c * W // cot, r * H // hang
            x2, y2 = (c + 1) * W // cot, (r + 1) * H // hang
            if bo_qua and _chong((x, y, x2, y2), bo_qua) > 0.5:
                continue                  # ô nằm gọn trong vùng phải loại, vd chú giải
            o = im.crop((x, y, x2, y2))
            d = _hoi_mot(PROMPT_DEM.format(
                ngu_canh=ngu_canh, vat=vat, x=x, y=y,
                w=x2 - x, h=y2 - y, W=W, H=H), _nen(o), **kw)
            if d is None:
                thieu += 1
                continue
            try:
                k = max(0, int(float(d.get("count", 0) or 0)))
            except (TypeError, ValueError):
                k = 0
            tong += k
            chi_tiet.append({"o": (x, y), "count": k})
            if verbose:
                print(f"    ô ({x},{y}) · {k}", flush=True)

    # Ngưỡng 10 là chỗ hai phép đo đổi chiều: dưới nó toàn khung thắng 4/4 so
    # với 1/4, trên nó toàn khung trả lời khác nhau mỗi lần chạy.
    chon = toan if toan and toan <= 10 else tong
    ly_do = ("toàn khung thấy hết" if chon == toan
             else f"cộng {len(chi_tiet)} ô không chồng lấn")
    if toan and tong and abs(toan - tong) > max(2, 0.3 * max(toan, tong)):
        ly_do += f" · ⚠ hai cách lệch nhau: toàn khung {toan}, cộng ô {tong}"
    return {"answer": str(chon), "count": chon, "count_toan": toan,
            "count_cong": tong, "n_o": len(chi_tiet), "thieu": thieu,
            "views": [{"o": "toàn khung", "count": toan}] + chi_tiet,
            "reason": ly_do + (f", {thieu} ô hỏng" if thieu else "")}


def dem_lap(vat: str, image_path, *, lan: int = 3, **kw) -> dict | None:
    """Chạy dem_o nhiều lần rồi trả CẢ PHỔ, không trả một con số.

    Đo 27/08 trên bản đồ chấn tâm: hỏi cùng một khung 5 lần ra 2, 3, 1, 1, 11.
    Một con số đơn lẻ ở đây là rút thăm. Trả trung vị kèm khoảng để người thi
    nhìn thấy mình đang tin vào cái gì.
    """
    lan = max(1, int(lan))
    chay = [dem_o(vat, image_path, **kw) for _ in range(lan)]
    chay = [c for c in chay if c]
    if not chay:
        return None
    so = sorted(c["count"] for c in chay)
    tv = so[len(so) // 2]
    on_dinh = len(set(so)) == 1
    d = dict(chay[0])
    d.update(count=tv, answer=str(tv), cac_lan=so, on_dinh=on_dinh,
             pho=f"{min(so)}–{max(so)}" if not on_dinh else str(tv))
    d["reason"] = (f"{lan} lần chạy: {so} → trung vị {tv}"
                   + ("" if on_dinh else " · ⚠ MỖI LẦN MỘT KHÁC, đừng tin một số"))
    return d


def _chong(a, b) -> float:
    """Phần diện tích của ô a nằm trong hộp b."""
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y / float((a[2] - a[0]) * (a[3] - a[1]) or 1)
