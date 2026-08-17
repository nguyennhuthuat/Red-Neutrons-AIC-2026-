"""Reusable retrieval, metadata filtering, and timestamp localization helpers.

This module deliberately has no Streamlit or model-loading dependencies. Query
encoders live in the UI; the functions here operate on NumPy vectors and
Pandas metadata so they can be tested independently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add_keyframe_uid(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return metadata with the stable ``video_id:n`` keyframe identifier."""
    if "keyframe_uid" in metadata.columns:
        return metadata.copy()

    required = {"video_id", "n"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Cannot construct keyframe_uid; missing columns: {sorted(missing)}"
        )

    result = metadata.copy()
    result["keyframe_uid"] = (
        result["video_id"].astype(str)
        + ":"
        + result["n"].astype("int64").astype(str)
    )
    return result


def merge_enrichment(
    metadata: pd.DataFrame,
    enrichment: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join optional enrichment without disturbing vector row alignment.

    If manual inspection reveals duplicate enrichment IDs, the most recently
    written row is used by retrieval while the source file remains unchanged.
    """
    base = add_keyframe_uid(metadata)
    base["_feature_row"] = np.arange(len(base), dtype=np.int64)

    if enrichment is None or enrichment.empty:
        return base

    extra = add_keyframe_uid(enrichment)
    extra = extra.drop_duplicates("keyframe_uid", keep="last")
    overlapping = [
        column
        for column in extra.columns
        if column in base.columns and column != "keyframe_uid"
    ]
    extra = extra.drop(columns=overlapping)
    return base.merge(extra, on="keyframe_uid", how="left", sort=False)


def parse_object_counts(value: Any) -> dict[str, int]:
    """Decode object-count metadata stored as JSON, dict, or parallel arrays."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, Mapping):
        return {str(key): int(count) for key, count in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): int(count) for key, count in parsed.items()}
    return {}


def parse_object_labels(value: Any) -> list[str]:
    """Decode labels stored as a list/array or JSON string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [part.strip() for part in text.split(",") if part.strip()]
        if isinstance(parsed, Sequence) and not isinstance(parsed, str):
            return [str(item) for item in parsed]
        return []
    if isinstance(value, Sequence) or isinstance(value, np.ndarray):
        return [str(item) for item in value]
    return []


def available_object_labels(metadata: pd.DataFrame) -> list[str]:
    """Return the sorted set of labels available in enriched metadata."""
    labels: set[str] = set()
    if "object_labels" in metadata.columns:
        for value in metadata["object_labels"]:
            labels.update(parse_object_labels(value))
    if "object_counts" in metadata.columns:
        for value in metadata["object_counts"]:
            labels.update(parse_object_counts(value))
    return sorted(labels)


def _ocr_similarity(query: str, text: str) -> float:
    query = query.strip().lower()
    text = text.strip().lower()
    if not query or not text:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.WRatio(query, text)) / 100.0
    except ImportError:
        return float(SequenceMatcher(None, query, text).ratio())


def normalize_scores(values: pd.Series) -> pd.Series:
    """Min-max normalize scores while keeping a stable all-equal result."""
    if values.empty:
        return values.astype(float)
    numeric = values.astype(float)
    lower = float(numeric.min())
    upper = float(numeric.max())
    if upper - lower < 1e-8:
        return pd.Series(np.ones(len(numeric)), index=numeric.index)
    return (numeric - lower) / (upper - lower)


def score_metadata_subset(
    metadata: pd.DataFrame,
    features: np.ndarray,
    query_vector: np.ndarray,
    *,
    video_id: str | None = None,
    required_objects: Mapping[str, int] | None = None,
    ocr_query: str = "",
) -> pd.DataFrame:
    """Score a filtered metadata subset by exact cosine/inner-product search.

    ``features`` must remain row-aligned with the original backend metadata.
    Metadata filters are applied before scoring so requested objects cannot
    disappear merely because they were outside a global FAISS candidate pool.
    """
    if len(metadata) != len(features):
        raise ValueError(
            f"Metadata/features mismatch: {len(metadata)} rows vs "
            f"{len(features)} vectors"
        )

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 2:
        if query.shape[0] != 1:
            raise ValueError(f"Expected one query vector, received {query.shape}")
        query = query[0]
    if query.ndim != 1 or query.shape[0] != features.shape[1]:
        raise ValueError(
            f"Query dim {query.shape} does not match feature dim {features.shape[1]}"
        )

    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise ValueError("Cannot search with a zero query vector")
    query = query / norm

    candidates = metadata.copy()
    if "_feature_row" not in candidates.columns:
        candidates["_feature_row"] = np.arange(len(candidates), dtype=np.int64)

    if video_id is not None:
        candidates = candidates[candidates["video_id"].astype(str) == str(video_id)]

    object_requirements = {
        str(label): max(1, int(minimum))
        for label, minimum in (required_objects or {}).items()
    }
    if object_requirements:
        if "object_counts" not in candidates.columns:
            return candidates.iloc[0:0].assign(
                semantic_score=pd.Series(dtype=float),
                semantic_score_norm=pd.Series(dtype=float),
                ocr_score=pd.Series(dtype=float),
                final_score=pd.Series(dtype=float),
            )

        keep = candidates["object_counts"].apply(
            lambda value: all(
                parse_object_counts(value).get(label, 0) >= minimum
                for label, minimum in object_requirements.items()
            )
        )
        candidates = candidates[keep]

    if candidates.empty:
        return candidates.assign(
            semantic_score=pd.Series(dtype=float),
            semantic_score_norm=pd.Series(dtype=float),
            ocr_score=pd.Series(dtype=float),
            final_score=pd.Series(dtype=float),
        )

    rows = candidates["_feature_row"].to_numpy(dtype=np.int64)
    candidate_vectors = np.asarray(features[rows], dtype=np.float32)
    semantic_scores = candidate_vectors @ query

    result = candidates.copy()
    result["semantic_score"] = semantic_scores
    result["semantic_score_norm"] = normalize_scores(result["semantic_score"])

    clean_ocr_query = ocr_query.strip()
    if clean_ocr_query:
        texts = (
            result["ocr_text"].fillna("").astype(str)
            if "ocr_text" in result.columns
            else pd.Series("", index=result.index)
        )
        result["ocr_score"] = texts.apply(
            lambda text: _ocr_similarity(clean_ocr_query, text)
        )
        result["final_score"] = (
            0.8 * result["semantic_score_norm"] + 0.2 * result["ocr_score"]
        )
    else:
        result["ocr_score"] = 0.0
        result["final_score"] = result["semantic_score_norm"]

    return result.sort_values("final_score", ascending=False).reset_index(drop=True)


def localize_intervals(
    scored_keyframes: pd.DataFrame,
    *,
    threshold: float = 0.70,
    merge_gap_seconds: float = 5.0,
) -> pd.DataFrame:
    """Convert scored keyframes from one video into estimated event intervals."""
    output_columns = [
        "interval_id",
        "video_id",
        "event_start",
        "event_end",
        "best_timestamp",
        "best_keyframe_uid",
        "best_n",
        "best_frame_idx",
        "best_score",
        "semantic_score",
        "ocr_score",
        "image_path",
        "matched_keyframes",
    ]
    if scored_keyframes.empty:
        return pd.DataFrame(columns=output_columns)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds cannot be negative")

    required = {"video_id", "pts_time", "final_score"}
    missing = required - set(scored_keyframes.columns)
    if missing:
        raise ValueError(f"Timestamp localization missing columns: {sorted(missing)}")

    timeline = scored_keyframes.copy()
    timeline["pts_time"] = pd.to_numeric(timeline["pts_time"], errors="coerce")
    timeline = timeline.dropna(subset=["pts_time"]).sort_values("pts_time")
    if timeline.empty:
        return pd.DataFrame(columns=output_columns)
    if timeline["video_id"].astype(str).nunique() != 1:
        raise ValueError("Timestamp localization requires exactly one video")

    timeline = timeline.reset_index(drop=True)
    timeline["smoothed_score"] = (
        timeline["final_score"]
        .astype(float)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    selected_positions = timeline.index[
        timeline["smoothed_score"] >= float(threshold)
    ].tolist()
    if not selected_positions:
        return pd.DataFrame(columns=output_columns)

    groups: list[list[int]] = [[selected_positions[0]]]
    for position in selected_positions[1:]:
        previous = groups[-1][-1]
        gap = float(timeline.loc[position, "pts_time"]) - float(
            timeline.loc[previous, "pts_time"]
        )
        if gap <= merge_gap_seconds:
            groups[-1].append(position)
        else:
            groups.append([position])

    intervals: list[dict[str, Any]] = []
    for interval_id, positions in enumerate(groups, start=1):
        first_position = positions[0]
        last_position = positions[-1]
        first_time = float(timeline.loc[first_position, "pts_time"])
        last_time = float(timeline.loc[last_position, "pts_time"])

        if first_position > 0:
            previous_time = float(timeline.loc[first_position - 1, "pts_time"])
            event_start = (previous_time + first_time) / 2.0
        else:
            event_start = first_time

        if last_position < len(timeline) - 1:
            next_time = float(timeline.loc[last_position + 1, "pts_time"])
            event_end = (last_time + next_time) / 2.0
        else:
            event_end = last_time

        segment = timeline.loc[positions]
        best = segment.loc[segment["final_score"].astype(float).idxmax()]
        intervals.append(
            {
                "interval_id": interval_id,
                "video_id": str(best["video_id"]),
                "event_start": event_start,
                "event_end": event_end,
                "best_timestamp": float(best["pts_time"]),
                "best_keyframe_uid": str(best.get("keyframe_uid", "")),
                "best_n": int(best.get("n", 0)),
                "best_frame_idx": int(best.get("frame_idx", 0)),
                "best_score": float(best["final_score"]),
                "semantic_score": float(best.get("semantic_score", 0.0)),
                "ocr_score": float(best.get("ocr_score", 0.0)),
                "image_path": str(best.get("image_path", "")),
                "matched_keyframes": len(positions),
            }
        )

    return (
        pd.DataFrame(intervals, columns=output_columns)
        .sort_values("best_score", ascending=False)
        .reset_index(drop=True)
    )

TRAKE_CANDIDATE_COLUMNS = [
    "rank",
    "video_id",
    "mean_action_score",
    "hypothesis_count",
    "action_count",
    "main_task_rank",
    "submission",
]

TRAKE_MATCH_COLUMNS = [
    "candidate_rank",
    "video_id",
    "hypothesis_rank",
    "hypothesis_mean_action_score",
    "action_index",
    "action",
    "retrieval_prompt",
    "similarity_score",
    "action_candidate_rank",
    "keyframe_uid",
    "n",
    "keyframe_idx",
    "frame_idx",
    "image_path",
]


def _empty_trake_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=TRAKE_CANDIDATE_COLUMNS),
        pd.DataFrame(columns=TRAKE_MATCH_COLUMNS),
    )


def _top_monotonic_paths(
    similarities: np.ndarray,
    candidate_positions: Sequence[Sequence[int]],
    *,
    minimum_gap: int,
    maximum_paths: int,
) -> list[tuple[float, tuple[int, ...]]]:
    """Return exact top-scoring monotonic paths through action candidates.

    Dynamic-programming states are keyed by the last selected frame. Keeping
    the top ``maximum_paths`` prefixes per state is sufficient because the
    additive score and future feasibility depend only on that last frame.
    """
    states: dict[int, list[tuple[float, tuple[int, ...]]]] = {
        int(position): [
            (float(similarities[0, int(position)]), (int(position),))
        ]
        for position in candidate_positions[0]
    }
    for action_index in range(1, similarities.shape[0]):
        next_states: dict[int, list[tuple[float, tuple[int, ...]]]] = {}
        for position_value in candidate_positions[action_index]:
            position = int(position_value)
            prefixes: list[tuple[float, tuple[int, ...]]] = []
            for previous_position, previous_paths in states.items():
                if position - previous_position < minimum_gap:
                    continue
                for previous_score, previous_path in previous_paths:
                    prefixes.append(
                        (
                            previous_score
                            + float(similarities[action_index, position]),
                            previous_path + (position,),
                        )
                    )
            if prefixes:
                next_states[position] = sorted(
                    prefixes,
                    key=lambda item: (-item[0], item[1]),
                )[:maximum_paths]
        states = next_states
        if not states:
            return []

    complete_paths = [path for paths in states.values() for path in paths]
    return sorted(
        complete_paths,
        key=lambda item: (-item[0], item[1]),
    )[:maximum_paths]


def rank_trake_candidates(
    corpus_metadata: pd.DataFrame,
    corpus_features: np.ndarray,
    action_vectors: np.ndarray,
    actions: Sequence[str],
    candidate_video_ids: Sequence[str],
    *,
    action_prompts: Sequence[str] | None = None,
    maximum_candidates: int = 20,
    top_frames_per_action: int = 5,
    minimum_gap_keyframes: int = 1,
    maximum_hypotheses: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank TRAKE videos by their best globally optimized action sequence.

    ``candidate_video_ids`` must be ordered by the main-task first-stage
    retrieval. Every candidate video's complete keyframe timeline is scored
    against every action. Each action is restricted to its five highest-scoring
    frames by default. Exact dynamic programming returns up to three best paths
    whose consecutive selections advance by at least one keyframe. The video
    score is the arithmetic mean of the best path's action similarities.

    The first returned frame is one row per ranked video. The second is the
    long-form action-to-keyframe evidence for every returned hypothesis.
    ``frame_idx`` is deliberately left missing when the corpus does not expose
    a true frame number; keyframe ordinal ``n`` is never mislabeled as a frame.
    """
    action_texts = tuple(str(action).strip() for action in actions)
    prompt_texts = (
        action_texts
        if action_prompts is None
        else tuple(str(prompt).strip() for prompt in action_prompts)
    )
    ordered_candidates = list(
        dict.fromkeys(str(video_id) for video_id in candidate_video_ids)
    )
    if not action_texts:
        raise ValueError("TRAKE requires at least one action")
    if any(not action for action in action_texts):
        raise ValueError("TRAKE actions cannot be empty")
    if len(prompt_texts) != len(action_texts) or any(
        not prompt for prompt in prompt_texts
    ):
        raise ValueError("TRAKE retrieval prompts must match non-empty actions")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    if top_frames_per_action < 1:
        raise ValueError("top_frames_per_action must be positive")
    if minimum_gap_keyframes < 0:
        raise ValueError("minimum_gap_keyframes cannot be negative")
    if maximum_hypotheses < 1:
        raise ValueError("maximum_hypotheses must be positive")
    if not ordered_candidates:
        return _empty_trake_results()

    required = {"video_id", "n"}
    missing = required - set(corpus_metadata.columns)
    if missing:
        raise ValueError(f"TRAKE corpus metadata missing columns: {sorted(missing)}")

    features = np.asarray(corpus_features)
    if features.ndim != 2:
        raise ValueError(f"TRAKE corpus features must be 2D, received {features.shape}")
    if len(corpus_metadata) != len(features):
        raise ValueError(
            f"Metadata/features mismatch: {len(corpus_metadata)} rows vs "
            f"{len(features)} vectors"
        )

    queries = np.asarray(action_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2 or queries.shape[0] != len(action_texts):
        raise ValueError(
            "TRAKE action vector count must match the linked-list action count"
        )
    if queries.shape[1] != features.shape[1]:
        raise ValueError(
            f"TRAKE action dim {queries.shape[1]} does not match "
            f"feature dim {features.shape[1]}"
        )
    if not np.isfinite(queries).all():
        raise ValueError("TRAKE action vectors contain non-finite values")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if bool((query_norms == 0).any()):
        raise ValueError("TRAKE action vectors cannot contain a zero vector")
    queries = queries / query_norms

    metadata = add_keyframe_uid(corpus_metadata).copy()
    if "_feature_row" not in metadata.columns:
        metadata["_feature_row"] = np.arange(len(metadata), dtype=np.int64)
    feature_rows = pd.to_numeric(metadata["_feature_row"], errors="coerce")
    if feature_rows.isna().any():
        raise ValueError("TRAKE metadata contains an invalid _feature_row")
    metadata["_feature_row"] = feature_rows.astype(np.int64)
    if (
        (metadata["_feature_row"] < 0).any()
        or (metadata["_feature_row"] >= len(features)).any()
    ):
        raise ValueError("TRAKE metadata _feature_row is outside the feature matrix")
    metadata["_trake_video_id"] = metadata["video_id"].astype(str)
    candidate_set = set(ordered_candidates)
    candidate_timelines = {
        str(video_id): timeline
        for video_id, timeline in metadata[
            metadata["_trake_video_id"].isin(candidate_set)
        ].groupby("_trake_video_id", sort=False)
    }

    candidate_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    for main_task_rank, video_id in enumerate(ordered_candidates, start=1):
        source_timeline = candidate_timelines.get(str(video_id))
        if source_timeline is None or source_timeline.empty:
            continue
        timeline = source_timeline.copy()
        frame_order = (
            pd.to_numeric(timeline["frame_idx"], errors="coerce")
            if "frame_idx" in timeline.columns
            else pd.Series(np.nan, index=timeline.index)
        )
        if frame_order.notna().all():
            timeline["_trake_temporal_order"] = frame_order
        else:
            # ``n`` is the stable per-video keyframe ordinal when a true frame
            # number is unavailable. It is used for ordering only and is never
            # emitted as a frame number in the submission.
            timeline["_trake_temporal_order"] = pd.to_numeric(
                timeline["n"], errors="raise"
            )
        timeline = timeline.sort_values(
            ["_trake_temporal_order", "_feature_row"], kind="stable"
        ).reset_index(drop=True)
        timeline["keyframe_idx"] = np.arange(len(timeline), dtype=np.int64)
        rows = timeline["_feature_row"].to_numpy(dtype=np.int64)
        vectors = np.asarray(features[rows], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError(f"TRAKE features for video {video_id!r} are non-finite")
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vector_norms[vector_norms == 0] = 1.0
        similarities = queries @ (vectors / vector_norms).T
        candidate_positions: list[tuple[int, ...]] = []
        candidate_ranks: list[dict[int, int]] = []
        for action_index in range(len(action_texts)):
            ranked_positions = sorted(
                range(len(timeline)),
                key=lambda position: (
                    -float(similarities[action_index, position]),
                    position,
                ),
            )[:top_frames_per_action]
            candidate_ranks.append(
                {
                    int(position): rank
                    for rank, position in enumerate(ranked_positions, start=1)
                }
            )
            candidate_positions.append(tuple(sorted(ranked_positions)))

        hypotheses = _top_monotonic_paths(
            similarities,
            candidate_positions,
            minimum_gap=minimum_gap_keyframes,
            maximum_paths=maximum_hypotheses,
        )
        if not hypotheses:
            continue

        candidate: dict[str, Any] = {
            "video_id": str(video_id),
            "mean_action_score": float(hypotheses[0][0] / len(action_texts)),
            "hypothesis_count": len(hypotheses),
            "action_count": len(action_texts),
            "main_task_rank": main_task_rank,
        }
        for hypothesis_rank, (total_score, positions) in enumerate(
            hypotheses, start=1
        ):
            mean_score = float(total_score / len(action_texts))
            submission_frames: list[str] = []
            for action_index, (action, prompt, position) in enumerate(
                zip(action_texts, prompt_texts, positions), start=1
            ):
                best = timeline.iloc[int(position)]
                score = float(similarities[action_index - 1, position])
                frame_idx = (
                    int(best["frame_idx"])
                    if "frame_idx" in best.index and pd.notna(best.get("frame_idx"))
                    else None
                )
                submission_frames.append("" if frame_idx is None else str(frame_idx))
                if hypothesis_rank == 1:
                    candidate[f"action_{action_index}"] = action
                    candidate[f"frame_{action_index}"] = frame_idx
                    candidate[f"keyframe_uid_{action_index}"] = str(
                        best["keyframe_uid"]
                    )
                    candidate[f"score_{action_index}"] = score
                match_records.append(
                    {
                        "video_id": str(video_id),
                        "hypothesis_rank": hypothesis_rank,
                        "hypothesis_mean_action_score": mean_score,
                        "action_index": action_index,
                        "action": action,
                        "retrieval_prompt": prompt,
                        "similarity_score": score,
                        "action_candidate_rank": candidate_ranks[
                            action_index - 1
                        ][int(position)],
                        "keyframe_uid": str(best["keyframe_uid"]),
                        "n": int(best["n"]),
                        "keyframe_idx": int(best["keyframe_idx"]),
                        "frame_idx": frame_idx,
                        "image_path": str(best.get("image_path", "")),
                    }
                )
            submission = ", ".join([str(video_id), *submission_frames])
            candidate[f"hypothesis_{hypothesis_rank}_mean_action_score"] = mean_score
            candidate[f"hypothesis_{hypothesis_rank}_submission"] = submission
            if hypothesis_rank == 1:
                candidate["submission"] = submission
        candidate_records.append(candidate)

    if not candidate_records:
        return _empty_trake_results()

    candidates = pd.DataFrame(candidate_records).sort_values(
        ["mean_action_score", "main_task_rank", "video_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(maximum_candidates).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1, dtype=np.int64))

    rank_by_video = dict(zip(candidates["video_id"], candidates["rank"]))
    matches = pd.DataFrame(match_records)
    matches = matches[matches["video_id"].isin(rank_by_video)].copy()
    matches.insert(
        0,
        "candidate_rank",
        matches["video_id"].map(rank_by_video).astype(np.int64),
    )
    matches = matches.sort_values(
        ["candidate_rank", "hypothesis_rank", "action_index"], kind="stable"
    ).reset_index(drop=True)

    base = TRAKE_CANDIDATE_COLUMNS[:-1]
    dynamic = [
        column
        for column in candidates.columns
        if column not in base + ["submission"]
    ]
    candidates = candidates[base + dynamic + ["submission"]]
    return candidates, matches[TRAKE_MATCH_COLUMNS]


def load_enrichment(path: Path) -> pd.DataFrame | None:
    """Load enrichment when present; absence is a supported state."""
    if not path.is_file():
        return None
    return pd.read_parquet(path)
