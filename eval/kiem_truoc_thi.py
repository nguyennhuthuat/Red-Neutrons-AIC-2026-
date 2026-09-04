# -*- coding: utf-8 -*-
"""Chạy đúng một lệnh trước giờ thi: mọi thứ sống hay chết.

    python eval/kiem_truoc_thi.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
LOI: list[str] = []


def bao(ten: str, ok: bool, chi_tiet: str = "") -> None:
    print(f"  {'✅' if ok else '🔴'} {ten:<34} {chi_tiet}")
    if not ok:
        LOI.append(ten)


print("── kho vector ──────────────────────────────────────────────")
proc = ROOT / "data" / "processed_hcmc2026"
try:
    man = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    import numpy as np
    F = np.load(proc / "features.npy", mmap_mode="r")
    bao("features.npy", F.shape[0] > 100000, f"{F.shape[0]} khung · {man['clip_model']}")
    import pandas as pd
    meta = pd.read_parquet(proc / "metadata.parquet")
    bao("metadata.parquet", len(meta) == F.shape[0], f"{meta['video_id'].nunique()} video")
    # Đường dẫn tuyệt đối trong metadata là bẫy khi bê máy khác sang.
    p0 = str(meta["image_path"].iloc[0]) if "image_path" in meta.columns else ""
    bao("ảnh khung mở được", (not p0) or pathlib.Path(p0).exists(), p0[:52])
except Exception as e:
    bao("kho vector", False, f"{type(e).__name__}: {e}")

print("── đường dịch ──────────────────────────────────────────────")
try:
    import translate as tr
    t = time.time()
    # Câu MỚI mỗi lần chạy: ăn kho thì có kiểm được mạng đâu.
    r = tr.to_english(f"một người đàn ông mặc áo đỏ đứng bên đường lúc "
                      f"{time.strftime('%H giờ %M phút %S giây')}")
    bao("Google dịch", tr.NGUON_CUOI == "mạng",
        f"{tr.NGUON_CUOI} · {time.time() - t:.1f}s · {str(r)[:34]}")
    bao("kho dịch trên đĩa", tr.TEP_KHO.exists(), f"{len(tr._CACHE)} câu")
    bao("cầu dao còn đóng", tr.mach_mang_con_song())
except Exception as e:
    bao("đường dịch", False, f"{type(e).__name__}: {e}")

print("── khoá Gemini (khâu Q&A) ──────────────────────────────────")
try:
    import rerank
    from google import genai
    mo = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    song = []
    for i, k in enumerate(rerank._CLIENTS._keys, 1):
        try:
            # Phải GIỮ biến: client tạm không ai tham chiếu thì bị thu gom
            # ngay giữa lúc gọi, báo "client has been closed" — 8/8 hỏng giả.
            c = genai.Client(api_key=k)
            c.models.generate_content(model=mo, contents="OK")
            song.append(i)
        except Exception:
            pass
    bao("khoá sống", len(song) >= 3, f"{len(song)}/{len(rerank._CLIENTS._keys)} trên {mo}")
except Exception as e:
    bao("khoá Gemini", False, f"{type(e).__name__}: {e}")

print("── kho bài nộp ─────────────────────────────────────────────")
sub = ROOT / "submission"
bao("thư mục submission/", sub.is_dir(), f"{len(list(sub.glob('*.csv'))) if sub.is_dir() else 0} tệp csv")

print()
print("🔴 CÓ VẤN ĐỀ: " + ", ".join(LOI) if LOI else "✅ sẵn sàng thi")
sys.exit(1 if LOI else 0)
