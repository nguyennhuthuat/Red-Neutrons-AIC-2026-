# -*- coding: utf-8 -*-
"""Đo KHÂU TRẢ LỜI TRÊN RỔ: tự tìm rồi tự trả lời, không cho sẵn khung đúng.

danh_gia_qa.py cho sẵn khung đáp án nên không thấy được lỗi thật của phòng thi:
rổ 20 ảnh gom nhiều video khác nhau, VLM chỉ nhìn điểm ảnh nên câu hỏi tên riêng
("tên con đèo là gì") luôn ra "không có thông tin". Tệp này đo đúng chuỗi đó,
bật/tắt kênh lời nói gắn theo từng ảnh.

  python eval/danh_gia_ro.py --bo queries_qa_tenrieng.csv --asr 0
  python eval/danh_gia_ro.py --bo queries_qa_tenrieng.csv --asr 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import corpus  # noqa: E402
import qa  # noqa: E402
from danh_gia_qa import trung  # noqa: E402
from evaluate import load_clip, encode_batch, translate  # noqa: E402

TEN, PRE = "ViT-L-16-SigLIP2-512", "webli"


def trong_cua(r, video_id: str, frame_idx: int) -> bool:
    """Khung có nằm trong cửa sổ đáp án của câu này không."""
    return (str(video_id) == str(r.video_id)
            and int(r.frame_idx_min) <= int(frame_idx) <= int(r.frame_idx_max))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="queries_qa_tenrieng.csv")
    ap.add_argument("--asr", type=int, default=1, help="1 = kèm lời nói theo ảnh")
    ap.add_argument("--sua-ten", type=int, default=1,
                    help="0 = tắt nhắc sửa chính tả địa danh")
    ap.add_argument("--ocr", type=int, default=0,
                    help="1 = kèm chữ đã quét sẵn theo ảnh (chỉ mục, 0 giây)")
    ap.add_argument("--top", type=int, default=20, help="số ảnh trong rổ")
    ap.add_argument("--translate", default="gemini")
    ap.add_argument("--kind", default="", help="chỉ chạy một loại câu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ra", default="")
    args = ap.parse_args()

    d = pd.read_csv(ROOT / "eval" / args.bo)
    if args.kind:
        d = d[d["kind"] == args.kind].reset_index(drop=True)
    if args.limit:
        d = d.iloc[:args.limit]
    meta = corpus.load_metadata()
    feats = np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy",
                    mmap_mode="r")

    truy, _ = translate(d["text_vi"].tolist(), args.translate)
    model, tok = load_clip(TEN, PRE)
    Q = encode_batch(truy, model, tok)

    print(f"[rổ] {args.bo} · {len(d)} câu · top={args.top} · "
          f"lời nói={'CÓ' if args.asr else 'KHÔNG'} · "
          f"chữ={'CÓ' if args.ocr else 'KHÔNG'}")

    # Nhân một lần cho cả bộ: đọc 726 MB vector từ đĩa 20 lần thì chậm vô ích.
    S = np.asarray(feats) @ Q.T

    dung = trong_ro = 0
    ghi = []
    for k, r in enumerate(d.itertuples()):
        thu = np.argsort(-S[:, k])[:args.top]
        hits = meta.iloc[thu].reset_index(drop=True)
        co = any(trong_cua(r, v, f) for v, f
                 in zip(hits.video_id, hits.frame_idx))
        trong_ro += co

        got = qa.answer_over_hits(r.question_vi, hits, desc=r.text_vi,
                                  top=args.top, kem_asr=bool(args.asr),
                                  kem_ocr=bool(args.ocr),
                                  sua_ten=bool(args.sua_ten))
        ans = (got or {}).get("answer", "")
        ok = bool(ans) and trung(ans, r.answer)
        dung += ok
        print(f"  {r.query_id} {'✅' if ok else '❌'} "
              f"{'[rổ có khung đúng]' if co else '[rổ TRƯỢT]':18s} "
              f"đáp án='{ans[:48]}' · chuẩn='{r.answer}'")
        ghi.append({"query_id": r.query_id, "answer": ans, "chuan": r.answer,
                    "dung": ok, "ro_co_khung_dung": bool(co)})

    n = len(d)
    print(f"\n  rổ chứa khung đáp án : {trong_ro}/{n} = {trong_ro / n:.3f}")
    print(f"  TRẢ LỜI ĐÚNG        : {dung}/{n} = {dung / n:.3f}")
    if trong_ro:
        tren = sum(g["dung"] for g in ghi if g["ro_co_khung_dung"])
        print(f"  đúng KHI rổ có khung: {tren}/{trong_ro} = {tren / trong_ro:.3f}"
              f"  ← trần thật của khâu trả lời")
    if args.ra:
        Path(args.ra).write_text(json.dumps(ghi, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
