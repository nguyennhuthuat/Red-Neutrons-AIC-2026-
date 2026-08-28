# -*- coding: utf-8 -*-
"""Câu KIS tả NHIỀU CẢNH nối tiếp: tách ra rồi chấm theo đúng thứ tự thời gian.

70% câu KIS đợt 1 có mốc thời gian ("bắt đầu bằng… sau đó… kết thúc bằng"),
trong khi bộ đo cũ có 0/81. Nhồi cả câu vào một vector là bắt CLIP tìm một
khung vừa giống cảnh đầu vừa giống cảnh cuối — thường chẳng khung nào như thế.
Thứ tự thời gian là tín hiệu duy nhất kho có sẵn (pts_time) mà CLIP không dùng.
"""
from __future__ import annotations

import re

import numpy as np

# Mốc tách cảnh, xếp theo độ dài để khớp cụm dài trước.
MOC = ["ngay sau cảnh này", "ngay sau đó", "cảnh quay tiếp theo", "cảnh quay cuối",
       "đoạn clip kết thúc", "cảnh quay kết thúc", "kết thúc bằng", "kết thúc với",
       "sau vài giây", "tiếp theo là", "tiếp theo có", "tiếp đến là", "tiếp đến",
       "sau đó là", "sau đó có", "sau đó", "tiếp theo", "trước đó là", "trước đó",
       "đầu tiên là", "cuối cùng", "rồi sau đó"]
# CỐ Ý không có "lần lượt" và "đầu tiên" trơ trọi: "lần lượt xuất hiện bốn lần"
# và "bún được cho đầu tiên" là trạng ngữ trong MỘT cảnh, không phải mốc đổi cảnh.
MO_DAU = ["đoạn clip bắt đầu bằng", "đoạn phim bắt đầu bằng",
          "cảnh quay bắt đầu bằng", "mẩu tin bắt đầu với", "bắt đầu bằng",
          "bắt đầu với"]
_NOI = re.compile(r"^(bằng|với|là|có|cảnh|hình ảnh)\s+", re.I)
_TACH = re.compile("|".join(re.escape(m) for m in
                            sorted(MOC, key=len, reverse=True)), re.I)


def tach_canh(text: str, it_nhat: int = 12) -> list[str]:
    """Cắt câu tả thành các cảnh theo mốc thời gian; 1 phần tử = câu một cảnh."""
    t = " ".join(str(text).split())
    for m in sorted(MO_DAU, key=len, reverse=True):
        t = re.sub(re.escape(m), " ", t, flags=re.I)
    # Đo độ dài TRƯỚC khi bỏ chữ nối: bỏ xong "cảnh kéo lưới cá" còn 11 ký tự
    # và bị lọc mất, làm câu ba cảnh chỉ còn hai.
    tho = [p.strip(" ,.;:") for p in _TACH.split(t)]
    phan = [_NOI.sub("", x).strip(" ,.;:") for x in tho if len(x) >= it_nhat]
    phan = [p for p in phan if p]
    return phan or [" ".join(str(text).split())]


def diem_chuoi(S: np.ndarray, video_ids: np.ndarray, pts: np.ndarray,
               cua_so: float = 90.0) -> np.ndarray:
    """Điểm cho từng khung khi câu tả có K cảnh nối tiếp.

    S[k, i] = độ giống của khung i với cảnh thứ k. Khung i được chấm như thể nó
    là CẢNH ĐẦU: cộng thêm, cho từng cảnh sau, khung khớp nhất nằm SAU nó trong
    CÙNG video và trong `cua_so` giây. Không có khung sau nào thì cảnh đó tính 0
    — câu tả hai cảnh mà video chỉ có một thì phải bị phạt.
    """
    K, N = S.shape
    if K == 1:
        return S[0].copy()
    thu = np.lexsort((pts, video_ids))
    tong = S[0].copy()
    for k in range(1, K):
        tot = np.zeros(N, dtype=np.float32)
        dau = 0
        while dau < N:
            cuoi = dau
            while cuoi + 1 < N and video_ids[thu[cuoi + 1]] == video_ids[thu[dau]]:
                cuoi += 1
            idx = thu[dau:cuoi + 1]
            t, s = pts[idx], S[k][idx]
            # hậu tố cực đại: max điểm cảnh k trong [i+1, ...]
            hau = np.maximum.accumulate(s[::-1])[::-1]
            j = np.searchsorted(t, t + cua_so, side="right")
            for a in range(len(idx)):
                b = min(j[a], len(idx) - 1)
                tot[idx[a]] = hau[a + 1] if a + 1 <= b else 0.0
            dau = cuoi + 1
        tong = tong + tot
    return tong / K
