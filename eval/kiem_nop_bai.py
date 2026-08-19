# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import corpus  # noqa: E402
import qa as qamod  # noqa: E402
import trake as tkmod  # noqa: E402

loi = []


def kiem(dieu_kien: bool, mo_ta: str):
    print(f"  {'✅' if dieu_kien else '🔴'} {mo_ta}")
    if not dieu_kien:
        loi.append(mo_ta)


meta = corpus.load_metadata()
hop_le = set(zip(meta.video_id.astype(str), meta.frame_idx.astype(int)))
print(f"corpus: {len(meta):,} khung · {meta.video_id.nunique()} video\n")

# ── KIS ──────────────────────────────────────────────────────────────────────
print("KIS — danh sách 100 dòng (video_id, frame_idx)")
gia_lap = meta.sample(120, random_state=0).reset_index(drop=True)
nop = gia_lap[["video_id", "frame_idx"]].iloc[:100]
kiem(len(nop) == 100, "đúng 100 dòng — R@k lấy MAX trên k dòng đầu, bỏ trống là vứt điểm")
kiem(nop.frame_idx.dtype.kind in "iu", "frame_idx là số nguyên")
kiem(not nop.duplicated().any(), "không có dòng trùng")
kiem(all((str(v), int(f)) in hop_le for v, f in
         zip(nop.video_id, nop.frame_idx)), "mọi (video_id, frame_idx) có thật trong corpus")
# Bẫy đã dính: nộp nhầm cột `n`.
kiem(not gia_lap.frame_idx.equals(gia_lap.n),
     "frame_idx KHÁC n — nếu hai cột bằng nhau thì đang nộp nhầm cột")

# ── Q&A ──────────────────────────────────────────────────────────────────────
print("\nQ&A — danh sách (video_id, frame_idx, answer)")
hits = gia_lap.copy()
hits["score"] = 0.0
chon = dict(video_id=str(hits.video_id.iloc[0]),
            frame_idx=int(hits.frame_idx.iloc[0]), answer="màu đỏ")
rows = qamod.submission_rows(hits, chon, limit=100)
kiem(len(rows) == 100, "đúng 100 dòng")
kiem(all(len(r) == 3 for r in rows), "mỗi dòng đủ 3 trường")
kiem(len({(r[0], r[1]) for r in rows}) == len(rows), "không trùng khung")
kiem(all(r[2] == "màu đỏ" for r in rows),
     "mọi dòng dùng CHUNG một câu trả lời (quy chế chấm câu trả lời của dòng trúng)")
kiem(rows[0][:2] == (chon["video_id"], chon["frame_idx"]),
     "khung đã chọn nằm ở dòng đầu")
kiem(all(str(r[2]).strip() for r in rows),
     "không dòng nào bỏ trống câu trả lời — rỗng là chắc chắn 0 điểm")

# ── TRAKE ────────────────────────────────────────────────────────────────────
print("\nTRAKE — một dòng (video_id, mốc_1, …, mốc_N)")
vid = str(meta.video_id.iloc[0])
khung = sorted(meta[meta.video_id == vid].frame_idx.astype(int))[:5]
moments = [dict(video_id=vid, frame_idx=int(f)) for f in khung]
dong = tkmod.submission_row(vid, moments)
kiem(dong[0] == vid, "trường đầu là video_id")
kiem(len(dong) == len(moments) + 1, "đủ số mốc theo đề")
kiem(list(dong[1:]) == sorted(dong[1:]),
     "các mốc TĂNG DẦN theo thời gian — chấm theo từng giai đoạn, đảo là mất điểm")
kiem(all(isinstance(x, (int, np.integer)) for x in dong[1:]), "mốc là số nguyên")


# ── đóng gói .zip ────────────────────────────────────────────────────────────
print("\nĐÓNG GÓI — submission.zip chứa submission/<câu>.csv")
import io  # noqa: E402
import zipfile  # noqa: E402

import nopbai  # noqa: E402

# Tên file = tên file truy vấn BTC phát: query-<số>-<dạng>. Số đánh tuần tự
# trong GÓI nên một gói có thể có nhiều câu cùng dạng.
bai = {
    "query-1-kis": [nopbai.dong_kis(r.video_id, r.frame_idx) for r in nop.itertuples()],
    "query-2-qa": [nopbai.dong_qa(*r) for r in rows],
    "query-3-trake": [nopbai.dong_trake(dong[0], dong[1:])],
}
try:
    ten = zipfile.ZipFile(io.BytesIO(nopbai.dong_goi(bai))).namelist()
    kiem(all(x.startswith("submission/") for x in ten),
         "mọi tệp nằm trong thư mục submission/")
    kiem(all(x.endswith(".csv") for x in ten), "mọi tệp có đuôi .csv")
    kiem(len(ten) == len(bai), f"đủ {len(bai)} tệp — mỗi câu truy vấn một tệp")
except ValueError as e:
    kiem(False, f"đóng gói hỏng: {e}")

# Hai ràng buộc của ban tổ chức dễ quên nhất, mỗi cái đủ để mất trắng một câu.
try:
    nopbai.dong_qa("L01_V001", 100, "x" * 101)
    kiem(False, "phải chặn câu trả lời Q&A quá 100 ký tự")
except ValueError:
    kiem(True, "chặn được câu trả lời Q&A quá 100 ký tự")

# Lỗi thể lệ liệt kê là một trong 5 lỗi hay gặp nhất: đáp án có dấu phẩy.
import csv  # noqa: E402
for ans in ["Màu đỏ, rất đẹp", 'Anh ấy nói "Xin chào"', "Dòng 1\nDòng 2"]:
    truong = next(csv.reader(io.StringIO(nopbai.dong_qa("L01_V001", 100, ans))))
    kiem(len(truong) == 3 and truong[2] == " ".join(ans.split()),
         f"đáp án {ans[:22]!r} vẫn ra đúng 3 trường (thấy {len(truong)})")
kiem(nopbai.dong_qa("L01_V001", 100, "5").endswith('"5"'),
     "answer LUÔN được bọc ngoặc kép, kể cả khi đơn giản")
kiem(nopbai.kiem_tep("query-2", ["L01_V001,1"]),
     "chặn được tên file sai quy ước (query-2 thiếu hậu tố dạng)")
try:
    nopbai.dong_trake("L01_V001", [300, 200])
    kiem(False, "phải chặn mốc TRAKE đảo thứ tự")
except ValueError:
    kiem(True, "chặn được mốc TRAKE đảo thứ tự")

print("\n" + ("🔴 CÓ LỖI: " + " · ".join(loi) if loi
              else "✅ ba dạng đúng định dạng · .zip đúng cấu trúc"))
sys.exit(1 if loi else 0)
