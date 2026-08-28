# -*- coding: utf-8 -*-
"""Khoá Gemini nào còn dùng được, ngay lúc này. Mỗi khoá đúng MỘT lời gọi bé.

Hạn mức free-tier tính THEO NGÀY và THEO MÔ HÌNH, nên phải thử đúng mô hình hệ
thống dùng. Chạy trước giờ thi để biết còn bao nhiêu đạn.

  python eval/kiem_khoa.py
  python eval/kiem_khoa.py --model gemini-3.5-flash
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rerank  # noqa: E402


def thu(key: str, model: str) -> tuple[str, str]:
    """Trả (trạng thái, ghi chú) cho một khoá."""
    from google import genai
    from google.genai import types
    try:
        c = genai.Client(api_key=key)
        r = c.models.generate_content(
            model=model, contents="1+1=?",
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=4))
        return "DÙNG ĐƯỢC", (r.text or "").strip()[:20]
    except Exception as e:
        s = f"{type(e).__name__} {e}"
        if rerank._is_quota_error(e):
            return "HẾT HẠN MỨC", "429 — chờ sang ngày mới (00:00 giờ Thái Bình Dương)"
        if rerank._is_auth_error(e):
            return "KHOÁ HỎNG", "401/403 — sai khoá hoặc chưa bật API"
        if "404" in s or "NOT_FOUND" in s:
            return "SAI MÔ HÌNH", f"tài khoản này không có {model}"
        return "LỖI KHÁC", s[:70]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    args = ap.parse_args()
    keys = rerank.api_keys()
    if not keys:
        print("Không thấy khoá nào trong .env"); return 1

    print(f"Thử {len(keys)} khoá trên mô hình {args.model}\n")
    dem = {}
    for i, k in enumerate(keys, 1):
        tt, note = thu(k, args.model)
        dem[tt] = dem.get(tt, 0) + 1
        dau = {"DÙNG ĐƯỢC": "✅", "HẾT HẠN MỨC": "⏳"}.get(tt, "🔴")
        print(f"  {dau} #{i} ...{k[-6:]}  {tt:12s} {note}")
    ok = dem.get("DÙNG ĐƯỢC", 0)
    print(f"\n  CÒN DÙNG ĐƯỢC: {ok}/{len(keys)}")
    for t, n in sorted(dem.items()):
        if t != "DÙNG ĐƯỢC":
            print(f"  {t}: {n}")
    if ok == 0:
        print("\n  ⚠ KHÔNG còn khoá nào. Tab KIS vẫn chạy bình thường "
              "(không cần API); chỉ khâu TRẢ LỜI Q&A và re-rank VLM là tắt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
