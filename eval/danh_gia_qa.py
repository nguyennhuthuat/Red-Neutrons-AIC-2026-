# -*- coding: utf-8 -*-
"""Đo KHÂU TRẢ LỜI của Q&A: khung đáp án cho sẵn, chỉ chấm câu trả lời.

Tách khỏi evaluate.py vì hai thứ đo hai việc khác nhau — evaluate.py đo khâu
TÌM, tệp này đo khâu TRẢ LỜI. Trộn hai khâu thì một câu sai không biết sai ở đâu.

  python eval/danh_gia_qa.py --cach doc_ky
  python eval/danh_gia_qa.py --cach cu --kind "đọc chữ"
  python eval/danh_gia_qa.py --bo queries_qa_tenrieng.csv --cach ocr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import corpus  # noqa: E402
import qa  # noqa: E402

CACH = ("cu", "doc_ky", "asr", "ocr", "dem")

# Ngưỡng độ dài cho luật "chứa nhau". 0,3 là chỗ tách được hai ca thật:
# "Cầu Phước Long, Cầu Rạch Đĩa" so với "Phước Long" (0,36 — đúng) và
# đáp án đọc cả bảng nguyên liệu so với "1 trái" (0,03 — sai).
TI_DAI = 0.3

# Đáp án đếm đúng nghĩa là "con số + nhiều nhất một từ chỉ loại"
# ("bốn miếng", "hai người"), nên phần không phải số phải ngắn.
DU_TOI_DA = 15


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


SO_CHU = {"khong": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5,
          "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10, "chuc": 10}


def so_trong(s: str) -> set[int]:
    """Số ĐỨNG RIÊNG trong câu, kể cả viết bằng chữ.

    Chỉ nhận token toàn chữ số: "4 miếng" cho {4}, còn "N182WT" cho tập rỗng —
    số dính trong mã hiệu không phải con số đếm được.
    """
    tu = re.split(r"[^0-9A-Za-zÀ-ỹ]+", str(s))
    ra = {int(w) for w in tu if w.isdigit()}
    # Số VIẾT BẰNG CHỮ chỉ tính khi nó đứng ĐẦU. Không có ràng buộc này thì
    # "Hải" trong tên riêng bị đọc thành số 2: "Hòn Hải Tặc" khớp "Tiên Hải",
    # và cả bộ 20 câu tên riêng bị chấm nới ra thành đúng giả.
    chu = bo_dau(s).split()
    if chu and chu[0] in SO_CHU:
        ra.add(SO_CHU[chu[0]])
    return ra


def du_chu(s: str) -> str:
    """Phần còn lại sau khi bỏ hết chữ số và chữ chỉ số lượng."""
    return " ".join(w for w in bo_dau(s).split()
                    if not w.isdigit() and w not in SO_CHU)


# Chỉ ba chữ nối. KHÔNG mở rộng danh sách này: bỏ dấu xong thì "lá" trùng
# "là", "cỏ" trùng "có" — thêm chữ đệm là tự tay xoá mất chữ mang nghĩa. Bản
# vá đầu mắc đúng lỗi đó và cho "xanh dương" khớp "xanh lá".
_NOI = {"va", "hoac", "roi"}


def _bien_the(that: str) -> list[str]:
    """Nhãn có ngoặc hoặc gạch chéo là LIỆT KÊ cách nói chấp nhận được.

    "chân (vùng bắp chân / khoeo chân)" nghĩa là ba cách nói đều đúng.
    """
    t = str(that)
    ra = [t, re.sub(r"\([^)]*\)", " ", t)]
    ra += re.findall(r"\(([^)]*)\)", t)
    ra += [x for p in list(ra) for x in re.split(r"[/;]", p)]
    return [x.strip() for x in dict.fromkeys(ra) if x and x.strip()]


def _loi(s: str, hoi: str) -> set[str]:
    """Tập chữ MANG NGHĨA: bỏ chữ mà câu hỏi đã nói sẵn, và ba chữ nối.

    "mặc áo màu gì?" thì "áo" và "màu" do câu hỏi cấp, không phân biệt được gì.
    Không bỏ thì "màu đỏ" bị chấm khác "áo đỏ" dù cùng một ý — đo được 6/50 câu
    bộ chính trượt oan chỉ vì chuyện này.
    """
    cho = set(bo_dau(hoi).split()) | _NOI
    return {w for w in bo_dau(s).split() if w not in cho}


def _khop_tap(dap: str, that: str, hoi: str) -> bool:
    """Mọi chữ mang nghĩa của nhãn phải CÓ MẶT trong đáp án.

    Dùng phép bao hàm TẬP chứ không phải chuỗi con: chuỗi con cho "màu vàng"
    khớp "đen và vàng kim" (thiếu hẳn "đen"), tập thì không. Chặn thêm đáp án
    bắn vãi: liệt kê gấp đôi số chữ của nhãn thì không tính.
    """
    b = _loi(that, hoi)
    if not b:
        return False
    a = _loi(dap, hoi)
    return b <= a and len(a) <= 2 * len(b) + 1


def trung(dap: str, that: str, hoi: str = "") -> bool:
    """Chấm nới tay: bỏ dấu, bỏ dấu câu, một bên chứa bên kia là tính đúng.

    Nới vì bộ chấm chặt tạo kết quả âm giả — "TP.HCM" với "TPHCM" từng bị chấm
    sai trong khi hai đáp án như nhau. Ban tổ chức chấm thế nào thì chưa biết.
    """
    if not str(dap).strip():
        return False
    # Câu đếm: đáp án nhãn viết bằng chữ ("bốn miếng") mà dem_o trả số ("4"),
    # so chuỗi thẳng thì trượt hết. So bằng CON SỐ khi cả hai bên đều có số.
    sd, st = so_trong(str(dap)), so_trong(str(that))
    if (sd and st and sd == st
            and len(du_chu(dap)) <= DU_TOI_DA and len(du_chu(that)) <= DU_TOI_DA):
        return True
    if any(_khop_tap(dap, bt, hoi) for bt in _bien_the(that)):
        return True
    a, b = bo_dau(dap), bo_dau(that)
    # Chứa nhau thì tính đúng, NHƯNG hai bên phải dài xấp xỉ nhau. Không có
    # chặn này thì đáp án đọc cả bảng nguyên liệu ăn điểm chỉ vì trong đó có
    # chuỗi đúng — đo được trên câu c15 của bộ chữ nhỏ.
    if a and b and min(len(a), len(b)) / max(len(a), len(b)) >= TI_DAI:
        if a in b or b in a:
            return True
    # Bỏ khoảng trắng chỉ khi KHÔNG bên nào có số: nó chữa được "TP.HCM" với
    # "TPHCM", nhưng nếu có số thì nó nuốt cả dấu thập phân và "13 Km" thành
    # khớp "1.3 Km".
    if not sd and not st:
        a2, b2 = a.replace(" ", ""), b.replace(" ", "")
        if (a2 and b2
                and min(len(a2), len(b2)) / max(len(a2), len(b2)) >= TI_DAI):
            return a2 in b2 or b2 in a2
    return False


def tra_loi(cach: str, r, p: str, verbose: bool = False,
            retries: int = 2, phong: float = 1.0) -> str:
    asr = qa.asr_khung(r.video_id, int(r.frame_idx))
    if cach == "cu":
        # Cách cũ coi đáp án rỗng là LỖI nên thử lại cả năm lượt, mỗi lượt chờ
        # 20-80 giây. Trên câu tên riêng nó luôn rỗng, nên đo với retries mặc
        # định thì mất 215 giây một câu mà kết quả không đổi.
        return (qa.ask(r.question_vi, [p], desc=r.text_vi, px=512,
                       retries=retries) or {}).get("answer", "")
    if cach == "doc_ky":
        return (qa.doc_ky(r.question_vi, p, desc=r.text_vi, asr=asr,
                          phong=phong, verbose=verbose) or {}).get("answer", "")
    if cach == "dem":
        return (qa.dem_o(r.question_vi, p, desc=r.text_vi,
                         verbose=verbose) or {}).get("answer", "")
    if cach == "asr":                       # riêng kênh lời nói, không nhìn ảnh
        if not asr:
            return ""
        return (qa._hoi_mot(qa.PROMPT_ASR.format(asr=asr[:2000],
                                                 question=r.question_vi), None)
                or {}).get("answer", "")
    if cach == "ocr":                       # riêng kênh chữ, không nhìn ảnh
        chu = qa.ocr_khung(p)
        if not chu:
            return ""
        return (qa._hoi_mot(qa.PROMPT_OCR.format(ocr=chu[:2000],
                                                 question=r.question_vi), None)
                or {}).get("answer", "")
    raise ValueError(cach)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", default="queries_qa_hcmc2026.csv",
                    help="tệp bộ đo trong eval/")
    ap.add_argument("--cach", default="doc_ky", choices=CACH)
    ap.add_argument("--kind", default="", help="chỉ chạy một loại câu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--retries", type=int, default=2,
                    help="số lượt thử lại của cách cũ (mặc định của nó là 5)")
    ap.add_argument("--phong", type=float, default=1.0,
                    help="phóng to mỗi ô trước khi gửi (chỉ dùng với doc_ky)")
    ap.add_argument("--ra", default="", help="ghi kết quả từng câu ra JSON")
    args = ap.parse_args()

    q = pd.read_csv(ROOT / "eval" / args.bo)
    if args.kind:
        q = q[q.kind == args.kind]
    if args.limit:
        q = q.head(args.limit)
    if q.empty:
        print("bộ đo rỗng sau khi lọc")
        return 1

    meta = pd.read_parquet(ROOT / "data" / "processed_hcmc2026"
                           / "metadata.parquet")
    meta["image_path"] = corpus.resolve_paths(meta["image_path"])
    duong = {(r.video_id, int(r.frame_idx)): r.image_path
             for r in meta.itertuples()}

    print(f"[qa] {args.bo} · {len(q)} câu · cách={args.cach}"
          f"{' · kind=' + args.kind if args.kind else ''}"
          f"{' · phóng ×' + str(args.phong) if args.phong != 1 else ''}\n")

    ket, dung, t0 = [], 0, time.time()
    for i, r in enumerate(q.itertuples(), 1):
        p = duong.get((r.video_id, int(r.frame_idx)))
        if p is None:
            print(f"{r.query_id}: KHÔNG thấy khung {r.video_id}/{r.frame_idx} "
                  f"— chạy lại src/prepare_data.py")
            continue
        t = time.time()
        a = tra_loi(args.cach, r, p, args.verbose, args.retries, args.phong)
        ok = trung(a, r.answer)
        dung += ok
        ket.append({"query_id": r.query_id, "kind": r.kind, "that": r.answer,
                    "dap": a, "ok": bool(ok), "giay": round(time.time() - t, 1)})
        print(f"{i:3}/{len(q)} {r.query_id} [{'✓' if ok else '✗'}] "
              f"{str(r.answer)[:30]:<32} -> {a or '(rỗng)'}", flush=True)

    n = len(ket)
    if not n:
        return 1
    print(f"\n{'=' * 62}\n{n} câu · {time.time() - t0:.0f}s · "
          f"đúng {dung}/{n} = {dung / n:.4f}")
    for k in sorted({x["kind"] for x in ket}):
        s = [x for x in ket if x["kind"] == k]
        print(f"  {k:>10} ({len(s):2}): {sum(x['ok'] for x in s) / len(s):.4f}")
    im = sum(1 for x in ket if not x["dap"])
    print(f"  bỏ trống {im}/{n} — với kênh lẻ (asr/ocr) im lặng là hành vi ĐÚNG")

    if args.ra:
        Path(args.ra).write_text(json.dumps(ket, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"  ghi {args.ra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
