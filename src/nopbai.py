# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import re
import zipfile

MAX_DONG = 100
MAX_KY_TU_DAP_AN = 100
DANG = ("kis", "qa", "trake")
# Tên gói thật của BTC có tiền tố đợt: query-p1-16-trake, không chỉ query-16-trake.
TEN_HOP_LE = re.compile(r"^query-[0-9A-Za-z]+(?:-[0-9A-Za-z]+)*-(kis|qa|trake)$")


def _boc(s) -> str:
    """Bọc ngoặc kép, nhân đôi ngoặc kép bên trong (RFC 4180)."""
    return '"' + str(s).replace('"', '""') + '"'


# Không đặt khoảng trắng sau dấu phẩy: thể lệ giữ nguyên khoảng trắng đầu/cuối
# trường, mà ngoặc kép chỉ có hiệu lực khi đứng ở KÝ TỰ ĐẦU của trường.
def dong_kis(video_id: str, frame_idx) -> str:
    return f"{video_id},{int(frame_idx)}"


def dong_qa(video_id: str, frame_idx, answer: str) -> str:
    a = " ".join(str(answer).split())      # gộp xuống dòng: một dòng một bản ghi
    if not a:
        raise ValueError("câu trả lời Q&A rỗng — rỗng là chắc chắn 0 điểm")
    if len(a) > MAX_KY_TU_DAP_AN:
        raise ValueError(
            f"câu trả lời {len(a)} ký tự, vượt trần {MAX_KY_TU_DAP_AN}: {a[:60]}…")
    # Luôn bọc ngoặc kép, kể cả khi không có ký tự đặc biệt.
    return f"{video_id},{int(frame_idx)},{_boc(a)}"


def dong_trake(video_id: str, frames) -> str:
    f = [int(x) for x in frames]
    if any(b <= a for a, b in zip(f, f[1:])):
        raise ValueError(f"mốc TRAKE phải tăng NGẶT (hai sự kiện khác nhau "
                         f"không thể cùng một khung): {f}")
    return ",".join([str(video_id)] + [str(x) for x in f])


def dang_cua(ten: str) -> str | None:
    """Suy dạng truy vấn từ tên file query-<số>-<dạng>."""
    goc = ten[:-4] if ten.endswith(".csv") else ten
    m = TEN_HOP_LE.match(goc)
    return m.group(1) if m else None


def kiem_tep(ten: str, dong: list[str]) -> list[str]:
    """Soi một tệp trước khi đóng gói. Trả danh sách lỗi (rỗng là đạt)."""
    loi = []
    dang = dang_cua(ten)
    if dang is None:
        loi.append(f"{ten}: tên sai quy ước — phải là query-<số>-<kis|qa|trake>.csv, "
                   f"lấy đúng tên file truy vấn BTC phát")
    if not dong:
        loi.append(f"{ten}: rỗng — nộp trống là chắc chắn 0 điểm")
    if len(dong) > MAX_DONG:
        loi.append(f"{ten}: {len(dong)} dòng, vượt trần {MAX_DONG}")
    if len(set(dong)) != len(dong):
        loi.append(f"{ten}: có dòng trùng — trùng là vứt một chỗ trong 100")

    so_truong = set()
    for i, d in enumerate(dong, 1):
        # Soi bằng chính bộ đọc CSV, không soi bằng mắt: đó là thứ BTC sẽ chạy.
        truong = next(csv.reader(io.StringIO(d)), [])
        so_truong.add(len(truong))
        if not d.strip():
            loi.append(f"{ten} dòng {i}: rỗng")
            continue
        if len(truong) < 2:
            loi.append(f"{ten} dòng {i}: thiếu dấu phẩy ngăn trường — {d[:40]}")
            continue
        if any(t != t.strip() for t in truong):
            loi.append(f"{ten} dòng {i}: có khoảng trắng thừa đầu/cuối trường "
                       f"(thể lệ KHÔNG tự trim) — {d[:40]}")

        if dang is None:                  # chưa biết dạng thì thôi soi kiểu
            continue

        # Số trường trước, kiểu dữ liệu sau: sai số trường mới là nguyên nhân,
        # "frame không phải số nguyên" chỉ là hệ quả và đọc dễ lạc hướng.
        if dang == "kis" and len(truong) != 2:
            loi.append(f"{ten} dòng {i}: KIS phải đúng 2 trường, thấy {len(truong)}")
            continue
        if dang == "qa" and len(truong) != 3:
            loi.append(f"{ten} dòng {i}: Q&A phải đúng 3 trường, thấy "
                       f"{len(truong)} — đáp án có dấu phẩy mà thiếu ngoặc kép?")
            continue

        try:
            [int(t) for t in (truong[1:2] if dang == "qa" else truong[1:])]
        except ValueError:
            loi.append(f"{ten} dòng {i}: frame phải là số nguyên — {d[:40]}")
            continue

        if dang == "qa":
            if not truong[2].strip():
                loi.append(f"{ten} dòng {i}: đáp án rỗng")
            elif len(truong[2]) > MAX_KY_TU_DAP_AN:
                loi.append(f"{ten} dòng {i}: đáp án {len(truong[2])} ký tự, "
                           f"vượt trần {MAX_KY_TU_DAP_AN}")
        elif dang == "trake":
            moc = [int(t) for t in truong[1:]]
            if any(b <= a for a, b in zip(moc, moc[1:])):
                loi.append(f"{ten} dòng {i}: mốc TRAKE phải tăng NGẶT — {d[:40]}")

    # Mọi dòng TRAKE là ứng viên của CÙNG một chuỗi nên phải cùng số mốc.
    if dang == "trake" and len(so_truong) > 1:
        loi.append(f"{ten}: các dòng có số mốc khác nhau {sorted(so_truong)} — "
                   f"số Frame ID phải khớp số events của đề")
    return loi


def dong_goi(bai: dict[str, list[str]], *, chat_che: bool = True) -> bytes:
    """Nén thành .zip có thư mục submission/ như thể lệ yêu cầu."""
    loi = [e for ten, d in bai.items() for e in kiem_tep(ten, d)]
    if loi and chat_che:
        raise ValueError("bài nộp có lỗi:\n  " + "\n  ".join(loi))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for ten, d in bai.items():
            ten = ten if ten.endswith(".csv") else f"{ten}.csv"
            # UTF-8 không BOM: BOM sẽ dính vào trường đầu của dòng đầu.
            z.writestr(f"submission/{ten}", "\n".join(d) + "\n")
    return buf.getvalue()


def mot_tep(dong: list[str]) -> bytes:
    """Nội dung một file .csv để tải lẻ."""
    return ("\n".join(dong) + "\n").encode("utf-8")
