# -*- coding: utf-8 -*-
"""Câu KIS tả NHIỀU CẢNH nối tiếp: tách ra rồi chấm theo đúng thứ tự thời gian.

70% câu KIS đợt 1 có mốc thời gian ("bắt đầu bằng… sau đó… kết thúc bằng"),
trong khi bộ đo cũ có 0/81. Nhồi cả câu vào một vector là bắt CLIP tìm một
khung vừa giống cảnh đầu vừa giống cảnh cuối — thường chẳng khung nào như thế.
Thứ tự thời gian là tín hiệu duy nhất kho có sẵn (pts_time) mà CLIP không dùng.

Đợt 2 đổi cách viết: BTC XUỐNG DÒNG mỗi cảnh thay vì viết "sau đó". Bám vào
chữ mốc thì 7 câu bốn cảnh bị gộp làm một rồi cắt cụt ở token 64. Dấu xuống
dòng mới là ranh giới cảnh chắc nhất — và nó phân biệt được với nhiều câu
CÙNG một dòng, vốn hay tả cùng MỘT khung.
"""
from __future__ import annotations

import re
from collections import deque

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

_CAU = re.compile(r"(?<=[.!?;])\s+|\n+")

# Cửa sổ giây giữa HAI CẢNH LIỀN NHAU. Con số cũ (30) đo từ cảnh ĐẦU nên cả
# câu phải gói trong 30 s; sau khi ép đủ thứ tự thì ý nghĩa đổi, quét lại trên
# CẢ HAI bộ và 60 s thắng ở mọi thước.
CUA_SO = 60.0

# Trọng số của vector NỀN (cả câu nhồi làm một) khi gộp với điểm chuỗi. Chính
# vector đó là thứ tệp này sinh ra để tránh, nên nó phải nhẹ: quét 0 -> 2 thì
# 0,25 là chỗ vừa được hạng-1 (4/14 -> 8/14) vừa không mất câu nào ở đuôi.
# Bỏ hẳn nền (0) thì p2-15 rơi khỏi cả 40.000 khung đầu.
ALPHA_NEN = 0.25

# Nhãn BTC đánh trước mỗi cảnh: "Cảnh 1:", "E1:", "- ", "1)".
_NHAN = re.compile(
    r"^\s*(?:[-•*]\s*|(?:cảnh|scene|sự\s+kiện|event)\s*\d+\s*[:.)\-]\s*"
    r"|[ec]\s*\d+\s*[:.)\-]\s*|\d+\s*[).]\s+)", re.I)
# Câu chỉ nói về CẤU TRÚC của đề, không tả gì nhìn thấy được.
_META = re.compile(
    r"^(?:\d+\s+cảnh\s+này\s+(?:xảy|diễn)\s+ra[^.]*"
    r"|đây\s+là\s+(?:một\s+)?(?:đoạn|phần)\s+(?:trong\s+)?bài\s+giảng[^.]*"
    r"|đây\s+là\s+chuỗi\s+(?:liên\s+tiếp\s+)?(?:các\s+)?cảnh[^.]*"
    r"|(?:các\s+)?cảnh\s+(?:này|sau)\s+(?:xảy|diễn)\s+ra[^.]*)\.?$", re.I)


def _bo_meta(dong: str) -> str:
    """Bỏ các câu chỉ mô tả cấu trúc đề, giữ lại phần tả cảnh trong cùng dòng."""
    cau = [c.strip() for c in re.split(r"(?<=[.!?])\s+", dong) if c.strip()]
    return " ".join(c for c in cau if not _META.match(c)).strip()


def cat_dong(text: str) -> list[str]:
    """Cắt theo DÒNG mà BTC tự xuống — ranh giới cảnh rõ hơn dấu chấm nhiều.

    Dòng kết bằng ':' là tiêu đề ("Trên slide bao gồm:"), ghép vào dòng sau
    chứ không đứng riêng làm một cảnh — tự nó chẳng tả cái gì.
    """
    ra, cho = [], ""
    for d in str(text or "").split("\n"):
        d = _bo_meta(" ".join(_NHAN.sub("", d).split()))
        if not d:
            continue
        if cho:
            d, cho = f"{cho} {d}", ""
        if d.endswith(":"):
            cho = d
            continue
        ra.append(d)
    if cho:
        ra.append(cho.rstrip(" :"))
    return ra


def co_moc(text: str) -> bool:
    """Câu có mốc thời gian không — tức có phải câu tả nhiều cảnh NỐI TIẾP."""
    t = " ".join(str(text or "").split())
    return bool(_TACH.search(t)
                or any(re.search(re.escape(m), t, re.I) for m in MO_DAU))


def moc_trake(text: str) -> tuple[list[str], list[str]]:
    """Đề TRAKE -> (danh sách MỐC, các dòng bối cảnh bị bỏ).

    Số mốc phải khớp đúng số cột phải nộp, nên một dòng thừa là hỏng cả câu.
    Khi đề có đánh nhãn ("E1:", "Cảnh 1:") thì CHỈ dòng có nhãn mới là mốc —
    dòng không nhãn là bối cảnh ("Video về một khu vườn cây ăn trái ở miền Tây
    Nam Bộ."), tả cả video chứ không tả một khoảnh khắc.
    """
    dong = [" ".join(d.split())
            for d in str(text or "").split("\n") if d.strip()]
    co_nhan = [bool(_NHAN.match(d)) for d in dong]
    if any(co_nhan):
        moc = [_NHAN.sub("", d).strip() for d, c in zip(dong, co_nhan) if c]
        bo = [d for d, c in zip(dong, co_nhan) if not c]
    else:
        moc = [d for d in (_bo_meta(_NHAN.sub("", d)) for d in dong) if d]
        bo = [d for d in dong if not _bo_meta(_NHAN.sub("", d))]
    return moc, bo


def tach_canh(text: str, it_nhat: int = 12) -> list[str]:
    """Cắt câu tả thành các cảnh; 1 phần tử = câu tả một cảnh.

    Thứ tự ưu tiên: BTC đã xuống dòng thì tin dấu xuống dòng; chưa xuống dòng
    thì mới dò mốc thời gian. Không có cả hai thì KHÔNG cắt — nhiều câu trong
    CÙNG một dòng thường tả CÙNG một khung ("…2 thanh niên phóng xe. Trong
    khung hình còn có ô tô xanh."), cắt ra rồi chấm theo thứ tự là bắt hai chi
    tiết cùng khung phải cách nhau về thời gian.
    """
    dong = cat_dong(text)
    if len(dong) >= 2:
        return dong
    goc = dong[0] if dong else " ".join(str(text or "").split())
    if not co_moc(goc):
        return [goc] if goc else [" ".join(str(text or "").split())]
    # Cắt theo CÂU trước rồi mới cắt theo mốc trong từng câu. Cắt thẳng theo mốc
    # thì "Nguyên liệu sau đó được chuyển…" đứt ngay sau chủ ngữ: cảnh sau mất
    # chủ ngữ, cảnh trước lại dính thêm đuôi của câu kế.
    phan = []
    for cau in (_CAU.split(goc) or [goc]):
        t = " ".join(cau.split())
        for m in sorted(MO_DAU, key=len, reverse=True):
            t = re.sub(re.escape(m), " ", t, flags=re.I)
        # Đo độ dài TRƯỚC khi bỏ chữ nối: bỏ xong "cảnh kéo lưới cá" còn 11 ký
        # tự và bị lọc mất, làm câu ba cảnh chỉ còn hai.
        tho = [p.strip(" ,.;:") for p in _TACH.split(t)]
        # Mốc nằm ngay sau chủ ngữ ("Nguyên liệu | sau đó | được chuyển…") thì
        # mảnh trái quá ngắn và bị vứt, cảnh sau mất luôn chủ ngữ. Gặp thế thì
        # để nguyên cả câu làm MỘT cảnh — thà thô còn hơn cụt.
        tho = [x for x in tho if x]      # mốc đứng đầu câu để lại mảnh rỗng
        if any(len(x) < it_nhat for x in tho):
            tho = [t.strip(" ,.;:")]
        phan += [_NOI.sub("", x).strip(" ,.;:") for x in tho if len(x) >= it_nhat]
    phan = [p for p in phan if p]
    return phan or [goc]


def _dp(S: np.ndarray, video_ids: np.ndarray, pts: np.ndarray,
        cua_so: float, luu_vet: bool):
    """Lõi quy hoạch động. Trả (điểm, vết) — vết là None khi không cần.

    Với mỗi khung i: tổng lớn nhất của một DÃY khung tăng dần theo thời gian
    trong cùng video, mỗi cảnh một khung, hai cảnh liền nhau cách nhau tối đa
    `cua_so` giây. Dựng không nổi dãy đủ K cảnh thì khung đó ăn 0.
    """
    K, N = S.shape
    if cua_so <= 0:
        cua_so = float("inf")
    thu = np.lexsort((pts, video_ids))
    ra = np.zeros(N, dtype=np.float32)
    vet = np.full((N, K), -1, dtype=np.int64) if luu_vet else None

    dau = 0
    while dau < N:
        cuoi = dau
        while cuoi + 1 < N and video_ids[thu[cuoi + 1]] == video_ids[thu[dau]]:
            cuoi += 1
        idx = thu[dau:cuoi + 1]
        t = pts[idx].astype(np.float64)
        n = len(idx)
        j = np.searchsorted(t, t + cua_so, side="right")
        # tot[i] = điểm tốt nhất của đoạn cảnh k..K-1 khi cảnh k ở khung i.
        tot = S[K - 1][idx].astype(np.float32)
        ke = np.full((K, n), -1, dtype=np.int64) if luu_vet else None
        for k in range(K - 2, -1, -1):
            moi_tot = np.empty(n, dtype=np.float32)
            # Cảnh k ở khung i thì cảnh k+1 nằm trong (i, b]. Hai mép cùng tăng
            # theo i nên một hàng đợi đơn điệu là đủ --- O(n), không cần cây.
            hang, phai = deque(), 0
            for a in range(n):
                b = int(j[a]) - 1
                while phai <= b:
                    while hang and tot[hang[-1]] <= tot[phai]:
                        hang.pop()
                    hang.append(phai)
                    phai += 1
                while hang and hang[0] <= a:
                    hang.popleft()
                # -inf chứ không phải 0: lấy 0 làm dấu vô hiệu thì dãy được phép
                # đứt giữa chừng mà vẫn ăn điểm của những cảnh đã khớp.
                if hang:
                    moi_tot[a] = S[k][idx[a]] + tot[hang[0]]
                    if luu_vet:
                        ke[k][a] = hang[0]
                else:
                    moi_tot[a] = -np.inf
            tot = moi_tot
        ra[idx] = np.where(np.isfinite(tot), tot, 0.0)
        if luu_vet:
            for a in range(n):
                if not np.isfinite(tot[a]):
                    continue
                cur = a
                for k in range(K):
                    vet[idx[a], k] = idx[cur]
                    if k + 1 < K:
                        cur = int(ke[k][cur])
        dau = cuoi + 1
    return ra / K, vet


def diem_chuoi(S: np.ndarray, video_ids: np.ndarray, pts: np.ndarray,
               cua_so: float = CUA_SO) -> np.ndarray:
    """Điểm cho từng khung khi câu tả có K cảnh nối tiếp --- xem _dp.

    Ép thứ tự cho ĐỦ chứ không chỉ giữa cảnh đầu và phần còn lại: bản trước lấy
    max độc lập cho từng cảnh trên cùng một khoảng, nên "A rồi B rồi C" chấm y
    hệt "A rồi C rồi B".
    """
    if S.shape[0] == 1:
        return S[0].copy()
    return _dp(S, video_ids, pts, cua_so, False)[0]


def mat_xich(S: np.ndarray, video_ids: np.ndarray, pts: np.ndarray,
             cua_so: float = CUA_SO):
    """Như diem_chuoi, nhưng trả thêm KHUNG NÀO đã đóng vai cảnh nào.

    vet[i, k] = chỉ số khung được chọn cho cảnh k khi khung i làm cảnh đầu;
    -1 nếu không dựng nổi dãy. Cần vì đáp án của ban tổ chức không phải lúc nào
    cũng là cảnh đầu: trên 14 câu nhiều cảnh của đề thật chỉ 8/14 gần cảnh đầu,
    6/14 rơi vào cảnh sau, mà mắt xích rơi trúng với lệch trung vị 0,8 s.
    """
    K, N = S.shape
    if K == 1:
        return S[0].copy(), np.arange(N, dtype=np.int64).reshape(N, 1)
    return _dp(S, video_ids, pts, cua_so, True)


def _chuan(a: np.ndarray) -> np.ndarray:
    """Đưa về trung bình 0, lệch chuẩn 1."""
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def gop(nen: np.ndarray, diem_c: np.ndarray,
        alpha: float = ALPHA_NEN) -> np.ndarray:
    """Gộp điểm nền với điểm chuỗi, sau khi kéo hai vế về CÙNG thang.

    Bắt buộc phải chuẩn hoá: giao diện xưa nay cộng zscore(nền) --- lệch chuẩn
    1 --- với điểm chuỗi thô --- lệch chuẩn ~0,03 --- nên điểm chuỗi chỉ còn
    khoảng 1/30 trọng lượng, và cấu hình đo được chưa bao giờ là cấu hình chạy
    thật. Mọi chỗ gộp hai vế này phải gọi qua đây.
    """
    return alpha * _chuan(nen) + _chuan(diem_c)
