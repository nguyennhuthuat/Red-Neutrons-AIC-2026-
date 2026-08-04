"""
evaluate.py — Đo chất lượng retrieval trên bộ query tự annotate.

Chạy:
    python eval/evaluate.py --dataset hcmc2026 --field text_en
    python eval/evaluate.py --dataset hcmc2026 --field text_vi

Mốc chấm {1,5,20,50,100} và Final Score lấy theo quy chế AIC 2026: với KIS
thì R-Score là nhị phân, nên R@k = max{R-Score(r_1..r_k)} đúng bằng Recall@k.
Nghĩa là cách đo cũ đã đúng bản chất, chỉ cần đổi mốc và lấy trung bình.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch
import open_clip

ROOT = Path(__file__).resolve().parent.parent
KS = [1, 5, 20, 50, 100]


def load_clip(model_name: str, pretrained: str):
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer


def encode_batch(texts, model, tokenizer) -> np.ndarray:
    """Encode list câu -> ma trận [n, dim] đã L2-normalize."""
    with torch.no_grad():
        vecs = model.encode_text(tokenizer(texts)).float().numpy()
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32)


def hit_rank(row, cand: pd.DataFrame):
    """Hạng 1-based của kết quả đúng đầu tiên, None nếu trượt."""
    ok = ((cand["video_id"] == row.video_id)
          & (cand["frame_idx"] >= row.frame_idx_min)
          & (cand["frame_idx"] <= row.frame_idx_max))
    if not ok.any():
        return None
    return int(np.argmax(ok.to_numpy())) + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hcmc2026")
    ap.add_argument("--field", default="text_en", choices=["text_en", "text_vi"])
    ap.add_argument("--queries", default=None,
                    help="mặc định eval/queries_<dataset>.csv")
    args = ap.parse_args()

    processed = ROOT / "data" / f"processed_{args.dataset}"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))

    # Cố ý dùng [] chứ không .get(..., default): artifacts cũ thiếu khoá phải nổ
    # KeyError ngay, thay vì âm thầm encode bằng sai model rồi trả kết quả rác.
    clip_model = manifest["clip_model"]
    clip_pretrained = manifest["clip_pretrained"]

    queries_path = (Path(args.queries) if args.queries
                    else ROOT / "eval" / f"queries_{args.dataset}.csv")

    meta = pd.read_parquet(processed / "metadata.parquet")
    index = faiss.read_index(str(processed / "faiss.index"))

    # Giao ước sống còn của cả hệ thống: dòng i của parquet <-> vector i của FAISS.
    # Lệch một dòng thì kết quả vẫn trông bình thường, chỉ là ảnh sai.
    assert index.ntotal == len(meta), (
        f"index có {index.ntotal} vector nhưng metadata có {len(meta)} dòng "
        f"— chạy lại src/prepare_data.py")

    queries = pd.read_csv(queries_path)
    print(f"[eval] {args.dataset} · {clip_model}/{clip_pretrained} · "
          f"{index.ntotal} vector · {len(queries)} query · field={args.field}")

    model, tokenizer = load_clip(clip_model, clip_pretrained)
    qvecs = encode_batch(queries[args.field].tolist(), model, tokenizer)
    scores, ids = index.search(qvecs, max(KS))

    ranks = []
    for i, row in queries.iterrows():
        cand = meta.iloc[ids[i]].reset_index(drop=True)
        ranks.append(hit_rank(row, cand))

    print(f"\n=== Recall trên {len(queries)} query ({args.field}) ===")
    recalls = {}
    for K in KS:
        hits = sum(1 for r in ranks if r is not None and r <= K)
        recalls[K] = hits / len(queries)
        print(f"R@{K:<3}: {recalls[K]:.3f}  ({hits}/{len(queries)})")

    print(f"\n>>> FINAL SCORE = {sum(recalls.values()) / len(KS):.4f}")

    # Độ khó chênh rất xa giữa các chương trình trong corpus 2026 (bản tin dễ,
    # đua xe đạp / nấu ăn toàn cảnh gần trùng), nên tách ra xem cho rõ.
    queries = queries.assign(
        rank=[np.nan if r is None else float(r) for r in ranks],
        prefix=queries["video_id"].str[:3])
    print("\n=== Theo chương trình ===")
    for pfx, g in queries.groupby("prefix"):
        found = int(g["rank"].notna().sum())
        best = "" if found == 0 else f", hạng tốt nhất {int(g['rank'].min())}"
        print(f"{pfx}: {found}/{len(g)} lọt top-{max(KS)}{best}")

    missed = queries[queries["rank"].isna()]
    if len(missed):
        print(f"\nTrượt top-{max(KS)} ({len(missed)}/{len(queries)}):")
        for _, row in missed.iterrows():
            print(f"  {row.query_id} [{row.video_id}] {row[args.field][:65]}")
