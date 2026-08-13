"""
install_features.py — thay bộ vector CLIP đang dùng bằng bộ mới mã hoá từ Kaggle.

VÌ SAO CẦN SCRIPT RIÊNG thay vì copy tay: đổi bộ vector là đổi ba thứ PHẢI khớp
nhau, sai một cái là hệ thống vẫn chạy nhưng trả kết quả rác —
    features.npy   (vector ảnh)
    faiss.index    (dựng lại TỪ features.npy, không tái dùng bản cũ)
    manifest.json  (tên model để UI/eval encode CÂU HỎI bằng đúng model đó)

Bẫy đã ghi ở memory data-hcmc2026: bộ 2023 chạy B-16, bộ 2026 chạy B-32, **cả hai
đều 512 chiều** nên cái guard `qvec.shape[1] != index.d` trong UI KHÔNG bắt được
nhầm lẫn. Lần này SigLIP2-L là 1024 chiều nên guard sẽ bắt — nhưng đừng trông chờ
vào may mắn đó, manifest mới là nguồn sự thật.

Kết quả đo trên 81 query (notebooks/kaggle_encode_corpus.ipynb, T4x2):
    ViT-B-32-quickgelu/openai  (BTC cấp)      0.4765
    ViT-L-14/openai                           0.5580
    ViT-B-16-SigLIP2-384/webli                0.7556
    ViT-L-16-SigLIP2-384/webli                0.7802   <- mặc định

Chạy:
    python src/install_features.py --list
    python src/install_features.py --model ViT-L-16-SigLIP2-384 --apply
    python src/install_features.py --restore          # quay lại bộ của BTC
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "kernel_out"
DST = ROOT / "data" / "processed_hcmc2026"

# open_clip cần đúng cặp (tên model, trọng số) để encode CÂU HỎI cùng không gian
# vector với ảnh. Suy từ tên file thì mong manh, nên tra bảng cho chắc.
PRETRAINED = {
    "ViT-B-32-quickgelu": "openai",
    "ViT-L-14": "openai",
    "ViT-B-16-SigLIP2-384": "webli",
    "ViT-L-16-SigLIP2-384": "webli",
}


def available():
    out = {}
    for p in sorted(SRC.glob("features_*.npy")):
        name = p.stem[len("features_"):].split("__")[0]
        out[name] = p
    return out


def doc_manifest() -> dict:
    """manifest hiện có, hoặc dict rỗng nếu MÁY NÀY CHƯA CÀI BỘ NÀO.

    Trường hợp chưa có là bình thường chứ không phải lỗi: người mới clone repo về
    rồi tải bộ vector từ Kaggle sẽ chưa có manifest nào cả. Trước đây hàm gọi
    thẳng read_text() nên máy mới cài luôn chết ngay ở dòng đầu.
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
    # IndexFlatIP + vector đã L2-chuẩn hoá = tìm theo cosine. Dựng LẠI từ đầu chứ
    # không sửa index cũ: số chiều đổi thì index cũ vô dụng.
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    faiss.write_index(index, str(DST / "faiss.index"))
    print(f"[ghi] faiss.index · {index.ntotal:,} vector · {index.d} chiều")

    man.update({
        "clip_model": model,
        "clip_pretrained": PRETRAINED[model],
        "feature_dim": int(x.shape[1]),
        "features_source": feats_path.name,
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
    ap.add_argument("--model", default="ViT-L-16-SigLIP2-384")
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
