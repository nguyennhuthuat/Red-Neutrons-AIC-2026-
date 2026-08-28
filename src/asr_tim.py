# -*- coding: utf-8 -*-
"""Kênh tìm bằng LỜI NÓI: BM25 trên bản chép âm đã căn theo từng keyframe.

Khác kênh chữ trên khung ở một chỗ quyết định: lời nói là văn xuôi, cùng loại từ
với câu truy vấn, nên ở đây phép gộp với CLIP có lý do để thắng --- chỗ mà kênh
chữ đồ hoạ đã đo và thua.

Hai điều bắt buộc, không phải tuỳ chọn:

1. **Ghép âm tiết đôi.** Tiếng Việt đơn âm, bỏ dấu xong "gà" thành "ga", "cá"
   thành "ca" --- có mặt ở hàng chục nghìn khung nên IDF gần bằng không. Đơn vị
   mang nghĩa là cặp âm tiết ("ga nuong", "sat lo"), nên chỉ mục chứa cả âm tiết
   lẻ lẫn cặp kề nhau.
2. **Gộp trùng trước khi lập chỉ mục.** Bản chép căn theo cửa sổ nên các keyframe
   liền nhau mang y hệt một đoạn: 177.321 khung chỉ có 25.174 đoạn khác nhau.
   Chấm trên đoạn rồi mới trải về khung là nhanh gấp bảy và cho đúng cùng kết quả.
"""
from __future__ import annotations

import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NGUON = ROOT / "data" / "processed_hcmc2026" / "asr_by_keyframe.parquet"
K1, B = 1.5, 0.75


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s.replace("đ", "d")).strip()


def manh(s: str) -> list[str]:
    """Âm tiết lẻ cộng cặp âm tiết kề nhau."""
    t = bo_dau(s).split()
    return t + [t[i] + "_" + t[i + 1] for i in range(len(t) - 1)]


_KM = None


class _Kho:
    """Chỉ mục ngược trên các đoạn lời nói KHÁC NHAU, cùng bảng trải về khung."""

    def __init__(self, d: pd.DataFrame):
        self.khung = d[["video_id", "n", "frame_idx"]].reset_index(drop=True)
        chu = d.asr_text.fillna("")
        ma, self.ve_doan = np.unique(chu.to_numpy(), return_inverse=True)
        self.n_doan = len(ma)

        self.df: dict[str, int] = {}
        self.oi: dict[str, list[int]] = {}
        self.tf: dict[str, list[int]] = {}
        dai = np.zeros(self.n_doan, dtype=np.float32)
        for j, t in enumerate(ma):
            m = manh(t)
            dai[j] = len(m)
            dem: dict[str, int] = {}
            for w in m:
                dem[w] = dem.get(w, 0) + 1
            for w, c in dem.items():
                self.oi.setdefault(w, []).append(j)
                self.tf.setdefault(w, []).append(c)
                self.df[w] = self.df.get(w, 0) + 1
        self.dai = dai
        self.dai_tb = float(dai.mean()) or 1.0

    def diem_doan(self, cau: str) -> np.ndarray:
        v = np.zeros(self.n_doan, dtype=np.float32)
        for w in dict.fromkeys(manh(cau)):
            df = self.df.get(w)
            if not df or df > 0.5 * self.n_doan:
                continue
            idf = math.log(1 + (self.n_doan - df + 0.5) / (df + 0.5))
            j = np.asarray(self.oi[w])
            f = np.asarray(self.tf[w], dtype=np.float32)
            chuan = K1 * (1 - B + B * self.dai[j] / self.dai_tb)
            v[j] += idf * f * (K1 + 1) / (f + chuan)
        return v


def kho() -> _Kho:
    """Dựng chỉ mục một lần rồi giữ luôn. Mất chừng một phút."""
    global _KM
    if _KM is None:
        d = pd.read_parquet(NGUON,
                            columns=["video_id", "n", "frame_idx", "asr_text"])
        _KM = _Kho(d)
    return _KM


def diem_khung(cau: str) -> np.ndarray:
    """Điểm BM25 của từng keyframe, đúng thứ tự hàng của bảng nguồn."""
    k = kho()
    return k.diem_doan(cau)[k.ve_doan]


def tim(cau: str, limit: int = 50) -> pd.DataFrame:
    k = kho()
    v = diem_khung(cau)
    top = np.argsort(-v, kind="stable")[:limit]
    ra = k.khung.iloc[top].copy()
    ra["diem"] = v[top].round(3)
    return ra[ra.diem > 0]


if __name__ == "__main__":
    import time

    t0 = time.time()
    k = kho()
    print(f"{len(k.khung):,} khung · {k.n_doan:,} đoạn khác nhau · "
          f"{len(k.df):,} mảnh · dựng {time.time() - t0:.0f}s")
    for c in sys.argv[1:]:
        print(f"\n[{c}]")
        print(tim(c, 5).to_string(index=False))
