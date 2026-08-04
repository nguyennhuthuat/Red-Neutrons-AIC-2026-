"""
search_ui.py — Minimal Streamlit search UI over the prepared artifacts.

Requires artifacts from src/prepare_data.py in data/processed_hcmc2026/:
    metadata.parquet, faiss.index (or features.npy as fallback)

Run:  streamlit run ui/search_ui.py

"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATASET = "hcmc2026"
PROCESSED = ROOT / "data" / f"processed_{DATASET}"
with open(PROCESSED / "manifest.json", encoding = "utf-8") as fp: 
    MANIFEST = json.load(fp)
CLIP_MODEL = MANIFEST["clip_model"]
CLIP_PRETRAINED = MANIFEST["clip_pretrained"]

st.set_page_config(page_title="RED-NEUTRONS · AIC 2026", layout="wide")


@st.cache_data(show_spinner="Loading metadata ...")
def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "metadata.parquet")


@st.cache_resource(show_spinner="Loading FAISS index ...")
def load_index():
    import faiss
    return faiss.read_index(str(PROCESSED / "faiss.index"))


@st.cache_resource(show_spinner="Loading CLIP text encoder ...")
def load_clip():
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    return model, tokenizer, torch


def preprocess_query(query: str) -> str:
    """Hook for Vietnamese handling (translation / rewriting).

    Plug Vietnamese->English translation or multilingual
    handling here. For now the query is passed through unchanged, so
    English queries will work best.
    """
    return query.strip()


def encode_text(query: str) -> np.ndarray:
    model, tokenizer, torch = load_clip()
    with torch.no_grad():
        tokens = tokenizer([query])
        vec = model.encode_text(tokens).float().numpy()
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec.astype(np.float32)


def search(query: str, top_k: int) -> pd.DataFrame:
    meta = load_metadata()
    index = load_index()
    qvec = encode_text(preprocess_query(query))

    if qvec.shape[1] != index.d:
        st.error(f"Encoder dim {qvec.shape[1]} != index dim {index.d}. "
                 f"Wrong CLIP variant — check which model BTC used.")
        st.stop()

    scores, ids = index.search(qvec, top_k)
    hits = meta.iloc[ids[0]].copy()
    hits["score"] = scores[0]
    return hits.reset_index(drop=True)


def diversify_by_video(hits: pd.DataFrame, per_video: int) -> pd.DataFrame:
    """Keep at most `per_video` keyframes per video (AVS-style spread)."""
    return (hits.groupby("video_id", sort=False)
                .head(per_video)
                .reset_index(drop=True))



st.title("RED-NEUTRONS — KIS")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K results", 10, 200, 50, step=10)
    per_video = st.slider("Max keyframes per video", 1, 20, 5,
                          help="Lower = more diverse videos (useful for AVS)")
    cols_per_row = st.slider("Grid columns", 3, 8, 5)

query = st.text_input(
    "Query (English works best until Vietnamese module is plugged in)",
    placeholder="e.g. women in ao dai posing in a lotus field")

if query:
    hits = search(query, top_k)
    hits = diversify_by_video(hits, per_video)
    st.caption(f"{len(hits)} keyframes from {hits['video_id'].nunique()} videos")

    for start in range(0, len(hits), cols_per_row):
        row = hits.iloc[start:start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, hit) in zip(cols, row.iterrows()):
            with col:
                img = hit.get("image_path", "")
                if img and Path(img).exists():
                    st.image(img, width="stretch")
                else:
                    st.markdown(":grey_background[no image]")
                st.caption(
                    f"**{hit['video_id']}** · n={int(hit['n'])}\n\n"
                    f"t={hit['pts_time']:.1f}s · frame_idx={int(hit['frame_idx'])} "
                    f"· score={hit['score']:.3f}")
else:
    st.info("Enter a query to search.")