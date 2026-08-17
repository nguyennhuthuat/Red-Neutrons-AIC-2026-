"""Metadata-aware CLIP/BLIP search and timestamp localization UI.

Run:
    streamlit run ui/enriched_search_ui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trake import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    TrakePlan,
    create_trake_planner,
)
from src.retrieval import (  # noqa: E402
    available_object_labels,
    load_enrichment,
    localize_intervals,
    merge_enrichment,
    rank_trake_candidates,
    score_metadata_subset,
)


PROCESSED = ROOT / "data" / "processed"
CLIP_MODEL = "ViT-B-16-quickgelu"
CLIP_PRETRAINED = "openai"
BLIP_MODEL = "Salesforce/blip-itm-base-coco"
BACKENDS = {
    "CLIP": {
        "name": "clip",
        "metadata": "metadata.parquet",
        "features": "features.npy",
        "index": "faiss.index",
    },
    "BLIP": {
        "name": "blip",
        "metadata": "metadata.blip.parquet",
        "features": "features.blip.npy",
        "index": "faiss.blip.index",
    },
}

st.set_page_config(
    page_title="RED-NEUTRONS - Enriched Retrieval",
    layout="wide",
)


@st.cache_data(show_spinner="Loading metadata ...")
def load_search_metadata(filename: str) -> pd.DataFrame:
    metadata = pd.read_parquet(PROCESSED / filename)
    enrichment = load_enrichment(PROCESSED / "enrichment.parquet")
    return merge_enrichment(metadata, enrichment)


@st.cache_resource(show_spinner="Loading feature vectors ...")
def load_features(filename: str) -> np.ndarray:
    return np.load(PROCESSED / filename, mmap_mode="r")


@st.cache_resource(show_spinner="Loading FAISS index ...")
def load_index(filename: str):
    import faiss

    return faiss.read_index(str(PROCESSED / filename))


@st.cache_data(show_spinner="Reading available object labels ...")
def load_object_options(metadata_filename: str) -> list[str]:
    return available_object_labels(load_search_metadata(metadata_filename))


@st.cache_resource(show_spinner="Loading CLIP encoder ...")
def load_clip():
    import open_clip
    import torch

    model, _, image_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    return model, tokenizer, image_preprocess, torch


@st.cache_resource(show_spinner="Loading BLIP encoder ...")
def load_blip():
    import torch
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForImageTextRetrieval.from_pretrained(
        BLIP_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()
    return processor, model, torch, device, dtype


@st.cache_resource(show_spinner=False)
def load_trake_planner(model: str):
    return create_trake_planner(
        model=model,
        cache_path=PROCESSED / "trake_plans.sqlite",
    )


def encode_clip_text(query: str) -> np.ndarray:
    model, tokenizer, _, torch = load_clip()
    device = next(model.parameters()).device
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        vector = model.encode_text(tokens)
        vector = torch.nn.functional.normalize(vector.float(), dim=-1)
    return vector.cpu().numpy().astype(np.float32)


def encode_clip_image(image: Image.Image) -> np.ndarray:
    model, _, preprocess, torch = load_clip()
    device = next(model.parameters()).device
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model.encode_image(tensor)
        vector = torch.nn.functional.normalize(vector.float(), dim=-1)
    return vector.cpu().numpy().astype(np.float32)


def encode_blip_text(query: str) -> np.ndarray:
    processor, model, torch, device, _ = load_blip()
    inputs = processor(text=[query], padding=True, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        vector = model.text_proj(output.last_hidden_state[:, 0, :])
        vector = torch.nn.functional.normalize(vector, dim=-1)
    return vector.float().cpu().numpy().astype(np.float32)


def encode_blip_image(image: Image.Image) -> np.ndarray:
    processor, model, torch, device, dtype = load_blip()
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.inference_mode():
        output = model.vision_model(pixel_values=pixel_values)
        vector = model.vision_proj(output.last_hidden_state[:, 0, :])
        vector = torch.nn.functional.normalize(vector, dim=-1)
    return vector.float().cpu().numpy().astype(np.float32)


def encode_query(
    backend: dict[str, str],
    *,
    text: str = "",
    image: Image.Image | None = None,
) -> np.ndarray:
    if bool(text) == bool(image):
        raise ValueError("Provide exactly one text or image query")
    if text:
        return (
            encode_blip_text(text.strip())
            if backend["name"] == "blip"
            else encode_clip_text(text.strip())
        )
    return (
        encode_blip_image(image)
        if backend["name"] == "blip"
        else encode_clip_image(image)
    )


def encode_text_queries(
    queries: tuple[str, ...] | list[str],
    backend: dict[str, str],
) -> np.ndarray:
    """Encode ordered TRAKE actions in the configured embedding space."""
    if not queries:
        raise ValueError("At least one text query is required")
    return np.concatenate(
        [encode_query(backend, text=query) for query in queries], axis=0
    ).astype(np.float32)


def minmax(values: pd.Series) -> pd.Series:
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower < 1e-8:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - lower) / (upper - lower)


def global_faiss_search(
    metadata: pd.DataFrame,
    query_vector: np.ndarray,
    backend: dict[str, str],
    top_k: int,
) -> pd.DataFrame:
    index = load_index(backend["index"])
    vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if vector.shape[1] != index.d:
        raise ValueError(
            f"Encoder dim {vector.shape[1]} != {backend['index']} dim {index.d}"
        )
    count = min(top_k, index.ntotal)
    scores, ids = index.search(vector, count)
    valid = ids[0] >= 0
    hits = metadata.iloc[ids[0][valid]].copy()
    hits["semantic_score"] = scores[0][valid]
    hits["semantic_score_norm"] = minmax(hits["semantic_score"])
    hits["ocr_score"] = 0.0
    hits["final_score"] = hits["semantic_score_norm"]
    return hits.reset_index(drop=True)


def diversify_by_video(hits: pd.DataFrame, maximum: int) -> pd.DataFrame:
    return hits.groupby("video_id", sort=False).head(maximum).reset_index(drop=True)


def timestamp_label(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"
    return f"{minutes:02d}:{remaining:05.2f}"


def render_grid(
    hits: pd.DataFrame,
    *,
    columns_per_row: int,
) -> None:
    for start in range(0, len(hits), columns_per_row):
        row = hits.iloc[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, (_, hit) in zip(columns, row.iterrows()):
            with column:
                image_path = str(hit.get("image_path", ""))
                if image_path and Path(image_path).is_file():
                    st.image(image_path, use_container_width=True)
                else:
                    st.markdown(":grey_background[no image]")

                semantic = float(hit.get("semantic_score", 0.0))
                final = float(hit.get("final_score", semantic))
                ocr_score = float(hit.get("ocr_score", 0.0))
                score_text = f"semantic={semantic:.3f} - final={final:.3f}"
                if ocr_score:
                    score_text += f" - OCR={ocr_score:.3f}"
                st.caption(
                    f"**{hit['video_id']}** - n={int(hit['n'])}\n\n"
                    f"t={float(hit['pts_time']):.1f}s - "
                    f"frame_idx={int(hit['frame_idx'])} - {score_text}"
                )


def render_trake_matches(
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    maximum_candidates: int,
) -> None:
    """Show ordered action hypotheses for the highest-ranked TRAKE videos."""
    for _, candidate in candidates.head(maximum_candidates).iterrows():
        rank = int(candidate["rank"])
        video_id = str(candidate["video_id"])
        score = float(candidate["mean_action_score"])
        with st.expander(f"#{rank} {video_id} - mean action score {score:.4f}"):
            video_matches = matches[matches["video_id"] == video_id]
            for hypothesis_rank, hypothesis in video_matches.groupby(
                "hypothesis_rank", sort=True
            ):
                hypothesis_score = float(
                    hypothesis.iloc[0]["hypothesis_mean_action_score"]
                )
                submission_column = f"hypothesis_{int(hypothesis_rank)}_submission"
                submission = str(candidate.get(submission_column, ""))
                st.markdown(
                    f"**Hypothesis {int(hypothesis_rank)}** - "
                    f"mean score `{hypothesis_score:.4f}` - `{submission}`"
                )
                for start in range(0, len(hypothesis), 4):
                    row = hypothesis.iloc[start : start + 4]
                    columns = st.columns(len(row))
                    for column, (_, match) in zip(columns, row.iterrows()):
                        with column:
                            image_path = str(match.get("image_path", ""))
                            if image_path and Path(image_path).is_file():
                                st.image(image_path, use_container_width=True)
                            else:
                                st.markdown(":grey_background[no image]")
                            frame = match.get("frame_idx")
                            frame_text = (
                                str(int(frame)) if pd.notna(frame) else "unavailable"
                            )
                            st.caption(
                                f"**{int(match['action_index'])}. "
                                f"{match['action']}**\n\n"
                                f"frame={frame_text} - "
                                f"keyframe={match['keyframe_uid']} - "
                                f"similarity={float(match['similarity_score']):.4f} - "
                                f"action top-{int(match['action_candidate_rank'])}\n\n"
                                f"Prompt: {match['retrieval_prompt']}"
                            )


st.title("RED-NEUTRONS - Metadata-aware retrieval")

with st.sidebar:
    mode = st.selectbox(
        "Mode",
        ["Global keyframe search", "TRAKE", "Find timestamp in video"],
    )
    backend_label = st.selectbox("Embedding backend", list(BACKENDS))
    backend = BACKENDS[backend_label]

    required_paths = [
        PROCESSED / backend["metadata"],
        PROCESSED / backend["features"],
        PROCESSED / backend["index"],
    ]
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        st.error(f"Missing artifacts: {', '.join(missing_paths)}")
        st.stop()

    metadata = load_search_metadata(backend["metadata"])
    features = load_features(backend["features"])
    if len(metadata) != len(features):
        st.error(
            f"{backend['metadata']} has {len(metadata)} rows but "
            f"{backend['features']} has {len(features)} vectors."
        )
        st.stop()

    enrichment_available = (PROCESSED / "enrichment.parquet").is_file()
    if enrichment_available:
        labels = load_object_options(backend["metadata"])
        selected_objects = st.multiselect("Required objects", labels)
        required_objects = {
            label: int(
                st.number_input(
                    f"Minimum {label} count",
                    min_value=1,
                    max_value=20,
                    value=1,
                    step=1,
                    key=f"minimum_{label}",
                )
            )
            for label in selected_objects
        }
        ocr_query = st.text_input(
            "OCR text",
            help="Optional Vietnamese text expected to be visible.",
        )
    else:
        required_objects = {}
        ocr_query = ""
        st.info(
            "Run `python src/enrich_metadata.py` to enable object and OCR filters."
        )

    if mode == "Global keyframe search":
        top_k = st.slider("Top-K results", 10, 200, 50, step=10)
        per_video = st.slider("Max keyframes per video", 1, 20, 5)
        columns_per_row = st.slider("Grid columns", 3, 8, 5)
    elif mode == "TRAKE":
        default_trake_model = (
            os.getenv("GEMINI_TRAKE_MODEL", "") or DEFAULT_GEMINI_MODEL
        )
        entered_trake_model = st.text_input(
            "TRAKE Gemini model",
            value=default_trake_model,
        ).strip()
        trake_model = entered_trake_model or default_trake_model
        retrieval_depth = st.slider(
            "Main-task retrieval depth",
            100,
            5000,
            1000,
            step=100,
            help="Keyframes retrieved with the Gemini-isolated main task.",
        )
        trake_candidate_pool = st.slider(
            "Candidate video pool",
            20,
            500,
            100,
            step=20,
            help="Unique videos whose full keyframe timelines are action-scored.",
        )
        trake_gallery_candidates = st.slider(
            "Candidate galleries",
            1,
            20,
            5,
            help="The result table and CSV always contain the top 20.",
        )
    else:
        video_ids = sorted(metadata["video_id"].astype(str).unique())
        selected_video = st.selectbox("Video", video_ids)
        threshold = st.slider(
            "Interval score threshold",
            0.0,
            1.0,
            0.70,
            0.05,
        )
        merge_gap = st.slider(
            "Merge gap (seconds)",
            0.0,
            30.0,
            5.0,
            1.0,
        )
        columns_per_row = st.slider("Grid columns", 2, 6, 4)

query_types = ["Text"] if mode == "TRAKE" else ["Text", "Image"]
query_type = st.radio("Query type", query_types, horizontal=True)
text_query = ""
image_query = None
uploaded_name = ""

if query_type == "Text":
    text_query = st.text_input(
        "English query",
        placeholder="e.g. a bus stopping beside a building",
    )
else:
    upload = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if upload is not None:
        with Image.open(upload) as source:
            image_query = source.convert("RGB")
        uploaded_name = upload.name
        st.image(image_query, caption="Query image", width=320)

if not text_query.strip() and image_query is None:
    st.info("Enter text or upload an image to search.")
    st.stop()

trake_plan: TrakePlan | None = None
action_vectors: np.ndarray | None = None
if mode == "TRAKE":
    planner = load_trake_planner(trake_model)
    with st.spinner(f"Planning TRAKE query with Gemini/{trake_model} ..."):
        trake_plan = planner.plan(text_query)
    if trake_plan.used_fallback:
        st.warning(
            "Gemini planning was unavailable. TRAKE is continuing as a "
            f"single-action query. Reason: {trake_plan.fallback_reason}"
        )
    with st.expander("TRAKE plan", expanded=True):
        st.json(trake_plan.to_dict())
    with st.spinner(f"Encoding the main task and actions with {backend_label} ..."):
        query_vector = encode_query(backend, text=trake_plan.main_task)
        action_vectors = encode_text_queries(
            list(trake_plan.action_prompts), backend
        )
else:
    with st.spinner(f"Encoding query with {backend_label} ..."):
        query_vector = encode_query(
            backend,
            text=text_query.strip(),
            image=image_query,
        )

if query_vector.shape[1] != features.shape[1]:
    st.error(
        f"Query dimension {query_vector.shape[1]} does not match "
        f"{backend['features']} dimension {features.shape[1]}."
    )
    st.stop()
if action_vectors is not None and action_vectors.shape[1] != features.shape[1]:
    st.error(
        f"TRAKE action dimension {action_vectors.shape[1]} does not match "
        f"{backend['features']} dimension {features.shape[1]}."
    )
    st.stop()

if mode in {"Global keyframe search", "TRAKE"}:
    search_depth = top_k * 5 if mode == "Global keyframe search" else retrieval_depth
    if required_objects or ocr_query.strip():
        hits = score_metadata_subset(
            metadata,
            features,
            query_vector,
            required_objects=required_objects,
            ocr_query=ocr_query,
        ).head(search_depth)
    else:
        hits = global_faiss_search(metadata, query_vector, backend, search_depth)

    if hits.empty:
        st.warning("No keyframes satisfy the query and metadata filters.")
        st.stop()
    if mode == "Global keyframe search":
        hits = diversify_by_video(hits.head(top_k), per_video)
        st.caption(f"{len(hits)} keyframes from {hits['video_id'].nunique()} videos")
        render_grid(hits, columns_per_row=columns_per_row)
    else:
        candidate_video_ids = (
            hits["video_id"]
            .astype(str)
            .drop_duplicates()
            .head(trake_candidate_pool)
            .tolist()
        )
        candidates, action_matches = rank_trake_candidates(
            metadata,
            features,
            action_vectors,
            trake_plan.actions,
            candidate_video_ids,
            action_prompts=trake_plan.action_prompts,
            maximum_candidates=20,
            top_frames_per_action=5,
            minimum_gap_keyframes=1,
            maximum_hypotheses=3,
        )
        if candidates.empty:
            st.warning("No TRAKE candidates could be scored.")
            st.stop()

        candidates["query"] = text_query.strip()
        candidates["main_task"] = trake_plan.main_task
        candidates["gemini_model"] = trake_plan.model
        candidates["gemini_fallback"] = trake_plan.used_fallback
        st.subheader("TRAKE top 20 video candidates")
        st.caption(
            f"Gemini isolated `{trake_plan.main_task}` and produced "
            f"{len(trake_plan.actions)} ordered actions. The first stage found "
            f"{len(candidate_video_ids)} candidate videos from "
            f"{min(len(hits), retrieval_depth)} keyframes. Each action may "
            "select only one of its top five frames; dynamic programming returns "
            "up to three ordered hypotheses."
        )
        result_columns = [
            "rank",
            "video_id",
            "mean_action_score",
            "hypothesis_count",
            "main_task_rank",
        ]
        for hypothesis_index in range(1, 4):
            score_column = f"hypothesis_{hypothesis_index}_mean_action_score"
            submission_column = f"hypothesis_{hypothesis_index}_submission"
            if score_column in candidates.columns:
                result_columns.extend([score_column, submission_column])
        for action_index in range(1, len(trake_plan.actions) + 1):
            result_columns.extend(
                [f"frame_{action_index}", f"score_{action_index}"]
            )
        result_columns.append("submission")
        st.dataframe(
            candidates[result_columns],
            use_container_width=True,
            hide_index=True,
        )
        if action_matches["frame_idx"].isna().any():
            st.warning(
                "Some corpus rows do not contain frame_idx. Their submission "
                "frame is intentionally blank; keyframe n is not a frame number."
            )
        render_trake_matches(
            candidates,
            action_matches,
            maximum_candidates=trake_gallery_candidates,
        )
        st.download_button(
            "Download TRAKE top 20 CSV",
            data=candidates.to_csv(index=False).encode("utf-8"),
            file_name="trake_top20.csv",
            mime="text/csv",
        )
else:
    scored = score_metadata_subset(
        metadata,
        features,
        query_vector,
        video_id=selected_video,
        required_objects=required_objects,
        ocr_query=ocr_query,
    )
    if scored.empty:
        st.warning("No keyframes in this video satisfy the current filters.")
        st.stop()

    intervals = localize_intervals(
        scored,
        threshold=threshold,
        merge_gap_seconds=merge_gap,
    )
    if intervals.empty:
        st.warning("No interval reached the selected score threshold.")
        st.stop()

    intervals["query_type"] = query_type.lower()
    intervals["query"] = text_query.strip() if text_query else uploaded_name
    intervals["backend"] = backend["name"]
    intervals["required_objects"] = ", ".join(
        f"{label}>={minimum}" for label, minimum in required_objects.items()
    )
    intervals["ocr_query"] = ocr_query.strip()
    intervals["estimated"] = True

    display = intervals.copy()
    display["start"] = display["event_start"].apply(timestamp_label)
    display["end"] = display["event_end"].apply(timestamp_label)
    display["best time"] = display["best_timestamp"].apply(timestamp_label)

    st.subheader(f"Estimated event intervals in {selected_video}")
    st.caption(
        "Intervals are estimates from sparse keyframes, not exact frame-level "
        "event boundaries."
    )
    st.dataframe(
        display[
            [
                "interval_id",
                "start",
                "end",
                "best time",
                "best_keyframe_uid",
                "best_score",
                "semantic_score",
                "ocr_score",
                "matched_keyframes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    scored_by_uid = scored.set_index("keyframe_uid", drop=False)
    best_rows = [
        scored_by_uid.loc[keyframe_uid]
        for keyframe_uid in intervals["best_keyframe_uid"]
        if keyframe_uid in scored_by_uid.index
    ]
    if best_rows:
        render_grid(
            pd.DataFrame(best_rows),
            columns_per_row=columns_per_row,
        )

    st.download_button(
        "Download timestamp CSV",
        data=intervals.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_video}_timestamps.csv",
        mime="text/csv",
    )
