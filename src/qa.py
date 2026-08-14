"""Dạng truy vấn Q&A: mô tả sự kiện + một câu hỏi.

Nộp `<video_id>, <frame_id>, <answer>`; **sai câu trả lời là 0 điểm dù đúng
khung**, nên đây là dạng duy nhất mà tìm kiếm giỏi vẫn có thể ăn 0.

Mặc định CLIP chọn khung còn VLM chỉ trả lời — để VLM tự chọn khung thì tệ hơn
hẳn (14% so với 36%), vì nó chọn ảnh nào nó TRẢ LỜI ĐƯỢC chứ không phải ảnh khớp
mô tả. Đặt `frame_from="vlm"` để quay lại hành vi cũ. Xem báo cáo, mục "Dạng Q&A".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """Bạn đang giúp thí sinh trả lời một câu hỏi về video.

Mô tả sự kiện: "{desc}"
CÂU HỎI: "{question}"

Dưới đây là {k} ảnh trích từ video, đánh số 1..{k} theo thứ tự gửi.

Nhiệm vụ:
1. Chọn ảnh trả lời được câu hỏi RÕ NHẤT. Ảnh đúng chủ đề nhưng không nhìn thấy
   thứ được hỏi thì KHÔNG chọn — hãy chọn ảnh thấy rõ chi tiết cần trả lời.
2. Trả lời câu hỏi, NGẮN GỌN, bằng tiếng Việt. Chỉ nêu thông tin nhìn thấy được
   trong ảnh. Nếu không ảnh nào trả lời được, để answer là "" và confidence 0.
3. Cho biết mức chắc chắn 0-10.

CHỈ trả JSON:
{{"best": <số thứ tự ảnh 1..{k}>, "answer": "<câu trả lời tiếng Việt>",
  "confidence": <0-10>, "reason": "<một câu ngắn vì sao chọn ảnh đó>"}}"""


def ask(question_vi: str, image_paths, *, desc: str = "", model: str | None = None,
        px: int | None = None, retries: int = 3, verbose: bool = False) -> dict | None:
    """Hỏi VLM một câu về rổ ảnh. Trả dict {best_index, answer, confidence, reason}.

    `best_index` là chỉ số 0-based vào `image_paths` (prompt đánh số từ 1 cho
    người/model dễ đọc, ở đây trừ lại 1 — đừng quên chỗ này).
    """
    from google.genai import types
    import rerank

    clients = rerank._CLIENTS            # dùng chung bể khoá + cơ chế đổi khoá
    if not clients:
        raise RuntimeError("Không thấy GEMINI_API_KEY trong .env")

    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    px = px or rerank.VLM_PX
    paths = list(image_paths)

    parts = [types.Part.from_text(text=PROMPT.format(
        desc=desc or "(không có)", question=question_vi, k=len(paths)))]
    parts += [types.Part.from_bytes(data=rerank._thumb(p, px), mime_type="image/jpeg")
              for p in paths]

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
            return {"best_index": b,
                    "answer": str(d.get("answer", "")).strip(),
                    "confidence": float(d.get("confidence", 0)),
                    "reason": str(d.get("reason", "")).strip()}
        except Exception as e:
            if rerank._is_auth_error(e):      # khoá sai -> bỏ ngay
                if clients.retire(verbose=False) is None:
                    break
                continue
            if rerank._is_quota_error(e):     # có thể chỉ là chặn theo phút
                time.sleep(30)
                if clients.retire() is None:
                    break
                continue
            print(f"  [API] Q&A lỗi ({attempt + 1}/{retries}): "
                  f"{type(e).__name__} {str(e)[:120]}", flush=True)
            if attempt < retries - 1:
                time.sleep(20 * (attempt + 1))
    return None


def answer_over_hits(question_vi: str, hits, *, desc: str = "", top: int = 20,
                     frame_from: str = "clip", **kw) -> dict | None:
    """Chạy `ask` trên `top` kết quả đầu của một DataFrame hits.

    Trả thêm `vlm_frame_idx` bên cạnh khung sẽ nộp: hai cái lệch nhau là dấu
    hiệu đáng xem lại bằng mắt.
    """
    sub = hits.iloc[:top]
    got = ask(question_vi, sub["image_path"].tolist(), desc=desc, **kw)
    if got is None:
        return None

    vrow = sub.iloc[got["best_index"]]
    got["vlm_frame_idx"] = int(vrow["frame_idx"])
    got["vlm_video_id"] = str(vrow["video_id"])

    row = sub.iloc[0] if frame_from == "clip" else vrow
    got.update(video_id=str(row["video_id"]), frame_idx=int(row["frame_idx"]),
               n=int(row["n"]), image_path=str(row["image_path"]),
               frame_from=frame_from)
    return got


def submission_rows(hits, chosen: dict | None, *, limit: int = 100) -> list[tuple]:
    """Dựng danh sách nộp `(video_id, frame_idx, answer)`, tối đa `limit` dòng.

    Luôn nộp đủ 100 dòng: R@k lấy MAX trên k kết quả đầu nên thêm dòng không bao
    giờ làm mất điểm đã có. Khung VLM chọn được đẩy lên đầu, phần còn lại giữ
    thứ tự tìm kiếm và dùng chung câu trả lời.
    """
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
