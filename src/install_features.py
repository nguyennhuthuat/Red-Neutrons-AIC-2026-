"""Thay bộ vector CLIP đang dùng bằng bộ mới mã hoá từ Kaggle.

    python src/install_features.py --list
    python src/install_features.py --model ViT-L-16-SigLIP2-512 --apply
    python src/install_features.py --restore

Cần script riêng vì đổi bộ vector là đổi BA thứ phải khớp nhau — `features.npy`,
`faiss.index` (dựng lại, không tái dùng), `manifest.json` — và sai một cái thì hệ
thống vẫn chạy, chỉ trả kết quả rác.

Điểm đã đo của từng model và cơ sở chọn: docs/bao_cao_he_thong.tex, mục
"Chọn encoder".
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "kernel_out"
DST = ROOT / "data" / "processed_hcmc2026"

# Tra bảng thay vì suy từ tên file: encode câu hỏi sai cặp là sai không gian vector.
PRETRAINED = {
    "ViT-B-32-quickgelu": "openai",
    "ViT-L-14": "openai",
    "ViT-B-16-SigLIP2-384": "webli",
    "ViT-L-16-SigLIP2-384": "webli",
    "ViT-L-16-SigLIP2-512": "webli",
    "ViT-SO400M-16-SigLIP2-384": "webli",
    "ViT-gopt-16-SigLIP2-384": "webli",
}

# ⚠️ L-16-384 và L-16-512 ĐỀU 1024 chiều nên guard số chiều không phân biệt được.
# Vì vậy install() lưu thêm vân tay `features_head_sha1`; xem kiem_khop().


def available():
    out = {}
    for p in sorted(SRC.glob("features_*.npy")):
        name = p.stem[len("features_"):].split("__")[0]
        out[name] = p
    return out


def van_tay(p: Path) -> str:
    """SHA-1 của 1 MB đầu file vector — đủ để phân biệt hai bộ CÙNG SỐ CHIỀU."""
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha1(f.read(1024 * 1024)).hexdigest()[:16]


def kiem_khop() -> None:
    """Nổ nếu features.npy đang dùng KHÔNG phải bộ mà manifest khai.

    Rẻ (đọc 1 MB) nên gọi được ở mọi điểm vào. Manifest cũ chưa có vân tay thì
    bỏ qua — không ép chạy lại install chỉ vì thiếu trường mới.
    """
    man = doc_manifest()
    mong = man.get("features_head_sha1")
    if not mong:
        return
    thuc = van_tay(DST / "features.npy")
    if thuc != mong:
        raise SystemExit(
            f"features.npy KHÔNG khớp manifest: vân tay {thuc} nhưng manifest "
            f"khai {mong} cho {man.get('clip_model')}.\n"
            f"Hai bộ SigLIP2-L (384 và 512) cùng 1024 chiều nên không guard nào "
            f"khác bắt được. Chạy lại: python src/install_features.py "
            f"--model {man.get('clip_model')} --apply")


def doc_manifest() -> dict:
    """manifest hiện có, hoặc dict rỗng nếu máy này chưa cài bộ nào.

    Chưa có là bình thường chứ không phải lỗi — máy mới clone về thì chưa có.
    """
    p = DST / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def backup_current():
    """Cất bộ đang dùng vào thư mục riêng, KHÔNG ghi đè bản cất trước đó."""
    man = doc_manifest()
    if not man:
        print("[backup] chưa có bộ nào đang dùng — bỏ qua bước cất")
        return None
    tag = man.get("clip_model", "unknown")
    bak = DST / "backup" / tag
    if bak.exists():
        print(f"[backup] {bak} đã có, giữ nguyên bản cũ")
        return bak
    bak.mkdir(parents=True)
    for f in ("features.npy", "faiss.index", "manifest.json"):
        if (DST / f).exists():
            shutil.copy2(DST / f, bak / f)
    print(f"[backup] bộ hiện tại ({tag}) -> {bak}")
    return bak


def install(model: str):
    feats_path = available()[model]
    man_path = DST / "manifest.json"
    man = doc_manifest()          # rỗng nếu máy này chưa cài bộ nào

    meta_path = DST / "metadata.parquet"
    if not meta_path.exists():
        raise SystemExit(
            f"thiếu {meta_path}.\n"
            f"metadata.parquet đi kèm bộ vector và PHẢI cùng thứ tự hàng với nó — "
            f"tải cùng chỗ với file features_*.npy, đừng tự dựng lại.")

    print(f"[nạp] {feats_path.name} ({feats_path.stat().st_size/1024**2:.0f} MB)")
    a = np.load(feats_path)
    # float16 đủ cho vector đã chuẩn hoá, nhưng FAISS cần float32.
    x = a.astype(np.float32)

    n = np.linalg.norm(x, axis=1)
    assert not np.isnan(x).any(), "có NaN trong vector"
    assert abs(n.mean() - 1.0) < 1e-3, f"vector chưa chuẩn hoá (norm TB {n.mean():.4f})"
    assert (n > 1e-6).all(), f"{int((n <= 1e-6).sum())} hàng toàn 0"

    meta_rows = len(__import__("pandas").read_parquet(DST / "metadata.parquet"))
    assert len(x) == meta_rows, (
        f"vector có {len(x):,} hàng nhưng metadata có {meta_rows:,} — "
        f"KHÔNG được cài, sẽ lệch hàng")
    print(f"[kiểm] {len(x):,} hàng · {x.shape[1]} chiều · norm TB {n.mean():.5f} · ok")

    backup_current()

    np.save(DST / "features.npy", x)
    print(f"[ghi] features.npy ({(DST/'features.npy').stat().st_size/1024**2:.0f} MB)")

    import faiss
    # IndexFlatIP + vector đã chuẩn hoá = tìm theo cosine. Dựng LẠI từ đầu.
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    faiss.write_index(index, str(DST / "faiss.index"))
    print(f"[ghi] faiss.index · {index.ntotal:,} vector · {index.d} chiều")

    man.update({
        "clip_model": model,
        "clip_pretrained": PRETRAINED[model],
        "feature_dim": int(x.shape[1]),
        "features_source": feats_path.name,
        "features_head_sha1": van_tay(DST / "features.npy"),
    })
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ghi] manifest.json -> {model}/{PRETRAINED[model]}")
    print("\nXong. Chạy để xác nhận:\n"
          "  python eval/evaluate.py --dataset hcmc2026 --field text_en")


def restore():
    man = json.loads((DST / "manifest.json").read_text(encoding="utf-8"))
    baks = sorted((DST / "backup").glob("*")) if (DST / "backup").exists() else []
    if not baks:
        raise SystemExit("không có bản backup nào")
    print("bản đã cất:", [b.name for b in baks])
    src = baks[0] if len(baks) == 1 else None
    if src is None:
        raise SystemExit("nhiều bản backup — copy tay bản bạn muốn")
    for f in ("features.npy", "faiss.index", "manifest.json"):
        shutil.copy2(src / f, DST / f)
    print(f"đã khôi phục {src.name} (đang dùng {man.get('clip_model')} trước đó)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-L-16-SigLIP2-512")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()

    if a.restore:
        restore()
        raise SystemExit

    av = available()
    cur = json.loads((DST / "manifest.json").read_text(encoding="utf-8"))
    print(f"đang dùng: {cur.get('clip_model')}/{cur.get('clip_pretrained')} "
          f"· {cur.get('feature_dim')} chiều\n")
    print("có sẵn trong data/kernel_out/:")
    for k, p in av.items():
        print(f"  {k:26s} {p.stat().st_size/1024**2:6.0f} MB"
              f"{'   <- sẽ cài' if k == a.model and a.apply else ''}")

    if not a.apply:
        print("\n(xem trước — thêm --apply để cài)")
        raise SystemExit
    if a.model not in av:
        raise SystemExit(f"\nkhông thấy {a.model} trong {SRC}")
    print()
    install(a.model)
