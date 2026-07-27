import pandas as pd 
import numpy as np 
from pathlib import Path 
import faiss 
import torch 
import open_clip 
import argparse 

# Config 
PROCESSED = Path("data/processed")
CLIP_MODEL = "ViT-B-16-quickgelu"
MODEL_PRETRAINED = "openai"
QUERIES = Path("eval/queries.csv")


def load_metadata() -> pd.DataFrame: 
    return pd.read_parquet(PROCESSED / "metadata.parquet")

def load_index(): 
    return faiss.read_index(str(PROCESSED / "faiss.index"))
    
def load_clip():
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained= MODEL_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    return model, tokenizer 

def load_queries() -> pd.DataFrame: 
    return pd.read_csv(QUERIES)

# Encode list into matrix[n, 512] what had L2-normalized
def encode_batch(texts, model, tokenizer): 
    with torch.no_grad(): 
        tokens = tokenizer(texts)
        vecs = model.encode_text(tokens).float().numpy()

    vecs /= np.linalg.norm(vecs, axis =1, keepdims=True)
    return vecs.astype(np.float32)

#Return 1-based rank of first hit, else None if missed 
def hit_rank(row, cand): 
    ok = ((cand["video_id"] == row.video_id)
          & (cand["frame_idx"] >= row.frame_idx_min)
          & (cand["frame_idx"] <= row.frame_idx_max))
    if not ok.any(): 
        return None 
    return int(np.argmax(ok.to_numpy())) + 1 

if __name__ == "__main__": 
    meta = load_metadata()
    index = load_index()
    model, tokenizer = load_clip()
    queries = load_queries()

    qvecs = encode_batch(queries["text_en"].tolist(), model, tokenizer)
    scores, ids = index.search(qvecs, 100)

    ranks = []
    for i, row in queries.iterrows(): 
        cand = meta.iloc[ids[i]].reset_index(drop = True)
        ranks.append(hit_rank(row, cand))

    print(f"\n=== Recall trên {len(queries)} query (text_en) ===")
    for K in [1, 5, 10, 50, 100]: 
        hits = sum(1 for r in ranks if r is not None and r <= K)
        print(f"Recall@{K:<3}: {hits/len(queries):.3f} ({hits}/{len(queries)})")

    print("\nQuery trượt top-100:")
    for r, (_, row) in zip(ranks, queries.iterrows()): 
        if r is None: 
            print(f" {row.query_id}: {row.text_en[:70]}")


