
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import corpus  # noqa: E402  (ghép lại đường dẫn ảnh theo máy hiện tại)
import rerank  # noqa: E402

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


HF_MODELS = {
    "vinai": "vinai/vinai-translate-vi2en-v2",     # mBART ~2.4GB, tốt nhất
    "envit5": "VietAI/envit5-translation",         # T5 ~1.2GB, cần tiền tố "vi: "
    "opus": "Helsinki-NLP/opus-mt-vi-en",          # ~75MB, nhanh nhưng yếu
}


def tr_google(texts: list[str]) -> list[str | None]:
    import translate as tr
    return tr.to_english_batch(texts)


def tr_hf(texts: list[str], backend: str) -> list[str]:
    """NMT chạy offline — lưới an toàn cho lúc thi nếu mất mạng."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        raise SystemExit("Thiếu gói: pip install transformers sentencepiece")

    name = HF_MODELS[backend]
    kwargs = {"src_lang": "vi_VN"} if backend == "vinai" else {}
    tok = AutoTokenizer.from_pretrained(name, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()

    # envit5 phân biệt hướng dịch bằng tiền tố, quên là nó trả về rác.
    src = [f"vi: {t}" for t in texts] if backend == "envit5" else texts

    gen = {"num_beams": 5, "max_length": 128, "early_stopping": True}
    if backend == "vinai":
        gen["decoder_start_token_id"] = (
            tok.lang_code_to_id["en_XX"] if hasattr(tok, "lang_code_to_id")
            else tok.convert_tokens_to_ids("en_XX"))

    out = []
    with torch.no_grad():
        for i in range(0, len(src), 8):
            batch = tok(src[i:i + 8], padding=True, truncation=True,
                        return_tensors="pt")
            ids = model.generate(**batch, **gen)
            out += tok.batch_decode(ids, skip_special_tokens=True)

    if backend == "envit5":
        out = [t[3:].strip() if t.startswith("en:") else t for t in out]
    return out


GEMINI_PROMPT = (
    "Dịch mô tả cảnh quay sau sang tiếng Anh, viết thành MỘT câu miêu tả "
    "hình ảnh cụ thể: ưu tiên danh từ vật thể nhìn thấy được, màu sắc, hành "
    "động, bối cảnh. Bỏ các cụm như 'cảnh quay', 'hình ảnh cho thấy'. "
    "Chỉ trả về câu tiếng Anh, không giải thích.\n\n")


def tr_gemini(texts: list[str]) -> list[str | None]:
    """LLM không chỉ dịch mà viết lại theo khẩu vị CLIP — thứ NMT không làm được."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Không thấy GEMINI_API_KEY trong .env hay biến môi trường")
    from google import genai
    client = genai.Client(api_key=key)
    # Alias -latest trỏ vào bản flash còn sống, khỏi sửa mỗi lần Google đổi tên.
    model_id = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    out = []
    for t in texts:
        # Free tier ở đây bóp theo PHÚT nên gặp 429 thì chờ rồi thử lại, đừng bỏ câu.
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=GEMINI_PROMPT + t)
                out.append(resp.text.strip())
                break
            except Exception as err:
                if attempt == 3:
                    print(f"  [dịch] thất bại, giữ nguyên tiếng Việt: {err}")
                    out.append(None)
                else:
                    wait = 20 * (attempt + 1)
                    print(f"  [dịch] {type(err).__name__}, chờ {wait}s rồi thử lại")
                    time.sleep(wait)
        time.sleep(4.0)                   # ~15 request/phút, đúng hạn free tier
    return out


def translate(texts: list[str], backend: str) -> tuple[list[str], float]:
    if backend == "none":
        return list(texts), 0.0

    cache_file = ROOT / "eval" / ".cache" / f"translate_{backend}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = (json.loads(cache_file.read_text(encoding="utf-8"))
             if cache_file.exists() else {})

    todo = sorted({t for t in texts if t not in cache})
    secs = 0.0
    if todo:
        print(f"[dịch] {backend}: {len(todo)} câu mới "
              f"({len(texts) - len(todo)} câu lấy từ cache)")
        t0 = time.perf_counter()
        if backend == "google":
            done = tr_google(todo)
        elif backend == "gemini":
            done = tr_gemini(todo)
        else:
            done = tr_hf(todo, backend)
        secs = (time.perf_counter() - t0) / len(todo)
        ok = {s: d for s, d in zip(todo, done) if d}
        if len(ok) < len(todo):
            print(f"[dịch] ⚠ {len(todo) - len(ok)}/{len(todo)} câu hỏng, "
                  f"không cache — chạy lại lệnh này để dịch nốt")
        cache.update(ok)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    else:
        print(f"[dịch] {backend}: toàn bộ lấy từ cache, không đo được thời gian")

    return [cache.get(t, t) for t in texts], secs


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
    ap.add_argument("--translate", default="none",
                    choices=["none", "google", "vinai", "envit5", "opus",
                             "gemini"],
                    help="dịch --field sang tiếng Anh trước khi encode")
    ap.add_argument("--rerank", default="none",
                    choices=["none", "vlm", "vlm+enc", "vlm-auto"],
                    help="tầng re-rank: vlm = Gemini chấm lại rổ cố định · "
                         "vlm-auto = chấm rổ nông, chỉ đào sâu khi VLM báo rổ hỏng · "
                         "vlm+enc = thêm encoder thứ hai (chậm trên CPU)")
    ap.add_argument("--rerank-depth", type=int, default=20,
                    help="chấm lại bao nhiêu kết quả đầu (với vlm-auto: rổ NÔNG)")
    ap.add_argument("--rerank-deep", type=int, default=200,
                    help="chỉ dùng với vlm-auto: rổ SÂU khi bị báo rổ hỏng")
    ap.add_argument("--rerank-model", default="ViT-B-16-SigLIP2-384",
                    help="encoder dùng cho vlm+enc")
    ap.add_argument("--ensemble", type=float, default=0.0, metavar="W",
                    help="cộng W·z(encoder phụ) trên TOÀN corpus (0 = tắt); "
                         "cả dải 0.2-1.5 đều dương, 0.5 cho 0.8148")
    ap.add_argument("--limit", type=int, default=0,
                    help="chỉ chạy N query đầu — để thử nhanh trước khi tốn cả "
                         "hạn mức API cho toàn bộ bộ eval")
    args = ap.parse_args()

    processed = ROOT / "data" / f"processed_{args.dataset}"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))

    clip_model = manifest["clip_model"]
    clip_pretrained = manifest["clip_pretrained"]

    queries_path = (Path(args.queries) if args.queries
                    else ROOT / "eval" / f"queries_{args.dataset}.csv")

    meta = pd.read_parquet(processed / "metadata.parquet")
    if "image_path" in meta.columns:
        meta["image_path"] = corpus.resolve_paths(meta["image_path"])
    feats = np.load(processed / "features.npy", mmap_mode="r")

    assert feats.shape[0] == len(meta), (
        f"features.npy có {feats.shape[0]} vector nhưng metadata có {len(meta)} "
        f"dòng — chạy lại src/prepare_data.py")

    queries = pd.read_csv(queries_path)
    if args.limit:
        queries = queries.head(args.limit).reset_index(drop=True)
    print(f"[eval] {args.dataset} · {clip_model}/{clip_pretrained} · "
          f"{feats.shape[0]} vector · {len(queries)} query · field={args.field} "
          f"· translate={args.translate}")
    if args.limit:
        print(f"[eval] ⚠ CHỈ {args.limit} query đầu — con số KHÔNG so được với "
              f"mốc đo trên đủ 81 query")

    texts = queries[args.field].tolist()
    texts, secs = translate(texts, args.translate)
    if args.translate != "none":
        if secs:
            print(f"[dịch] {secs:.2f} s/câu")
        print("[dịch] ba câu đầu:")
        for src, dst in list(zip(queries[args.field], texts))[:3]:
            print(f"  {src}\n   -> {dst}")

    model, tokenizer = load_clip(clip_model, clip_pretrained)
    qvecs = encode_batch(texts, model, tokenizer)
    # Trần của tầng chấm lại chính là R@D, nên rổ phải sâu ít nhất bằng độ sâu chấm.
    n_cand = max(max(KS),
                 args.rerank_depth if args.rerank != "none" else 0,
                 args.rerank_deep if args.rerank == "vlm-auto" else 0)
    if args.ensemble > 0:
        import ensemble as ens
        print(f"[ensemble] + {args.ensemble}·z({ens.PHU[0]}) trên toàn corpus")
    S = np.asarray(feats, dtype=np.float32) @ qvecs.T              # (n_kf, n_query)
    ids = np.empty((len(qvecs), n_cand), dtype=np.int64)
    scores = np.empty((len(qvecs), n_cand), dtype=np.float32)
    for i, t in enumerate(texts):
        tong = (ens.ghep(S[:, i], t, w=args.ensemble) if args.ensemble > 0
                else S[:, i])
        top = np.argsort(-tong, kind="stable")[:n_cand]
        ids[i], scores[i] = top, tong[top]
    del S

    n_bad_basket = n_deep = 0
    if args.rerank != "none":
        D = min(args.rerank_depth, ids.shape[1])
        DEEP = min(args.rerank_deep, ids.shape[1])
        label = f"rổ {D}" + (f" → {DEEP} khi cần" if args.rerank == "vlm-auto" else "")
        print(f"\n[re-rank] {args.rerank} · {label} · "
              f"{len(queries)} query (lần đầu chậm, sau đó lấy từ cache)")
        cache = rerank._Cache()
        t0 = time.perf_counter()
        for i, row in queries.iterrows():
            # VLM nhận thẳng tiếng Việt, không cần bản dịch.
            if args.rerank == "vlm-auto":
                paths_all = meta["image_path"].to_numpy()[ids[i][:DEEP]]
                vlm, went_deep = rerank.vlm_scores_adaptive(
                    row.text_vi, paths_all, scores[i][:DEEP],
                    shallow=D, deep=DEEP, cache=cache)
                n_deep += went_deep
                k = len(vlm)                      # số ảnh thật sự đã chấm
            else:
                k = D
                paths_all = meta["image_path"].to_numpy()[ids[i][:k]]
                vlm = rerank.vlm_scores(row.text_vi, paths_all, cache=cache)

            enc = (rerank.encoder_scores(texts[i], paths_all[:k],
                                         model_name=args.rerank_model)
                   if args.rerank == "vlm+enc" else None)
            top = ids[i][:k].copy()
            ids[i][:k] = top[rerank.rerank_order(scores[i][:k], vlm=vlm, enc=enc)]
            if rerank.basket_looks_wrong(vlm[:D]):
                n_bad_basket += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(queries)} · {time.perf_counter() - t0:.0f}s"
                      + (f" · đào sâu {n_deep}" if args.rerank == "vlm-auto" else ""),
                      flush=True)
        print(f"[re-rank] xong trong {time.perf_counter() - t0:.0f}s"
              + (f" · đào sâu {n_deep}/{len(queries)} query"
                 if args.rerank == "vlm-auto" else ""))

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
    if args.rerank != "none":
        # Mốc tra theo model đang dùng, không hằng số cứng.
        base = {"ViT-B-32-quickgelu": 0.4765, "ViT-L-14": 0.5580,
                "ViT-gopt-16-SigLIP2-384": 0.7457,
                "ViT-B-16-SigLIP2-384": 0.7556,
                "ViT-SO400M-16-SigLIP2-384": 0.7704,
                "ViT-L-16-SigLIP2-384": 0.7802,
                "ViT-L-16-SigLIP2-512": 0.7901}.get(clip_model)
        print(f"    (nền không re-rank của {clip_model} = "
              f"{base:.4f})" if base else f"    (chưa có mốc nền cho {clip_model})")
        print(f"    {n_bad_basket}/{len(queries)} query bị VLM báo "
              f"'đáp án có vẻ không nằm trong rổ'")

    # Độ khó chênh rất xa giữa các chương trình nên tách ra xem cho rõ.
    queries = queries.assign(
        rank=[np.nan if r is None else float(r) for r in ranks],
        prefix=queries["video_id"].str[:3],
        sent=texts)          # câu THỰC SỰ đưa vào CLIP, để soi khi trượt
    print("\n=== Theo chương trình ===")
    for pfx, g in queries.groupby("prefix"):
        found = int(g["rank"].notna().sum())
        best = "" if found == 0 else f", hạng tốt nhất {int(g['rank'].min())}"
        print(f"{pfx}: {found}/{len(g)} lọt top-{max(KS)}{best}")

    missed = queries[queries["rank"].isna()]
    if len(missed):
        print(f"\nTrượt top-{max(KS)} ({len(missed)}/{len(queries)}):")
        for _, row in missed.iterrows():
            print(f"  {row.query_id} [{row.video_id}] {row.sent[:65]}")
