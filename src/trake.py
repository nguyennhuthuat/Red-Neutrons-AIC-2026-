"""Gemini planning primitives for Temporal Action Keyframe Retrieval (TRAKE).

The hosted model is used only to turn a natural-language temporal question
into one broad retrieval task and an ordered linked list of visible actions.
Embedding and ranking remain local to the configured retrieval backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


TRAKE_PROMPT_VERSION = "trake-temporal-plan-v2-contextual-actions"
MAX_TRAKE_ACTIONS = 12

TRAKE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "main_task": {
            "type": "string",
            "description": (
                "A concise English description of the overall activity used "
                "to retrieve candidate videos."
            ),
        },
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_TRAKE_ACTIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "A concise label for the requested visible moment."
                        ),
                    },
                    "retrieval_prompt": {
                        "type": "string",
                        "description": (
                            "A self-contained English visual retrieval prompt that "
                            "places this moment in the overall activity and "
                            "distinguishes it from adjacent requested moments."
                        ),
                    },
                },
                "required": ["action", "retrieval_prompt"],
            },
            "description": (
                "The requested visually observable actions in chronological "
                "order, each with a contextual retrieval prompt."
            ),
        },
    },
    "required": ["main_task", "actions"],
}

TRAKE_SYSTEM_PROMPT = """You plan Temporal Action Keyframe Retrieval (TRAKE) queries.

Return only the requested structured object. Follow these rules:
1. Read the complete English or Vietnamese query and isolate its overall video-level activity as main_task.
2. Write main_task as one concise English visual-retrieval phrase. It should be broad enough to retrieve the correct video, but preserve distinctive subjects, objects, setting, counts, numbers, and visible text.
3. Extract every explicitly requested moment into actions. For each item, write a concise action label and a self-contained retrieval_prompt.
4. Keep actions in the exact semantic and chronological order requested by the user.
5. Each retrieval_prompt must preserve relevant subject, object, setting, and overall-activity context from main_task while emphasizing only that action. Use visible context to distinguish it from adjacent moments; do not rely on phrases such as "the previous action" or "step 2".
6. Remove list numbers such as '(1)' from action text. Do not merge distinct requested moments.
7. Do not invent intermediate actions, identities, objects, colors, locations, or outcomes.
8. Preserve negations, proper names, route numbers, other numbers, and visible text.
9. If the query requests no explicit sequence, return one action containing its most concrete visible event.
10. main_task is at most 40 words, each action label is at most 24 words, and each retrieval_prompt is at most 48 words.
11. Do not include explanations, Markdown, confidence, or alternatives.
"""


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


@dataclass(frozen=True)
class ActionNode:
    """One action in the immutable, singly linked TRAKE action list."""

    position: int
    action: str
    retrieval_prompt: str
    next: "ActionNode | None" = None


@dataclass(frozen=True)
class ActionLinkedList:
    """Ordered actions stored as linked nodes rather than a mutable array."""

    head: ActionNode | None
    length: int

    @classmethod
    def from_actions(
        cls,
        actions: list[str] | tuple[str, ...],
        retrieval_prompts: list[str] | tuple[str, ...] | None = None,
    ) -> "ActionLinkedList":
        prompts = tuple(actions) if retrieval_prompts is None else tuple(retrieval_prompts)
        if len(prompts) != len(actions):
            raise ValueError("TRAKE action and retrieval-prompt counts must match")
        head: ActionNode | None = None
        for position in range(len(actions), 0, -1):
            head = ActionNode(
                position,
                str(actions[position - 1]),
                str(prompts[position - 1]),
                head,
            )
        return cls(head=head, length=len(actions))

    def __iter__(self) -> Iterator[ActionNode]:
        current = self.head
        traversed = 0
        while current is not None:
            yield current
            current = current.next
            traversed += 1
            if traversed > self.length:
                raise RuntimeError("TRAKE action linked list contains a cycle")
        if traversed != self.length:
            raise RuntimeError("TRAKE action linked list length is inconsistent")

    def to_tuple(self) -> tuple[str, ...]:
        return tuple(node.action for node in self)

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "position": node.position,
                "action": node.action,
                "retrieval_prompt": node.retrieval_prompt,
                "next_position": node.next.position if node.next is not None else None,
            }
            for node in self
        ]


@dataclass(frozen=True)
class TrakePlan:
    """A validated Gemini decomposition and its ordered action linked list."""

    original_query: str
    main_task: str
    action_list: ActionLinkedList
    model: str
    prompt_version: str = TRAKE_PROMPT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    fallback_reason: str = ""

    @property
    def actions(self) -> tuple[str, ...]:
        return self.action_list.to_tuple()

    @property
    def action_prompts(self) -> tuple[str, ...]:
        return tuple(node.retrieval_prompt for node in self.action_list)

    @property
    def used_fallback(self) -> bool:
        return bool(self.fallback_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "main_task": self.main_task,
            "actions": list(self.actions),
            "action_prompts": list(self.action_prompts),
            "action_nodes": self.action_list.to_records(),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "fallback_reason": self.fallback_reason,
            "used_fallback": self.used_fallback,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrakePlan":
        actions = tuple(str(action) for action in value.get("actions", ()))
        prompts = tuple(
            str(prompt) for prompt in value.get("action_prompts", actions)
        )
        return cls(
            original_query=str(value["original_query"]),
            main_task=str(value["main_task"]),
            action_list=ActionLinkedList.from_actions(actions, prompts),
            model=str(value["model"]),
            prompt_version=str(value.get("prompt_version", TRAKE_PROMPT_VERSION)),
            input_tokens=int(value.get("input_tokens", 0) or 0),
            output_tokens=int(value.get("output_tokens", 0) or 0),
            latency_ms=float(value.get("latency_ms", 0.0) or 0.0),
            cache_hit=bool(value.get("cache_hit", False)),
            fallback_reason=str(value.get("fallback_reason", "")),
        )


@dataclass(frozen=True)
class TrakeProviderResponse:
    payload: Mapping[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class TrakeProvider(Protocol):
    model: str

    def generate(self, query: str) -> TrakeProviderResponse:
        """Return the provider's parsed TRAKE plan."""


def validate_trake_payload(
    payload: Mapping[str, Any],
    original_query: str,
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    prompt_version: str = TRAKE_PROMPT_VERSION,
) -> TrakePlan:
    """Validate Gemini output and materialize actions as a linked list."""
    if not isinstance(payload, Mapping):
        raise ValueError("TRAKE provider output must be a JSON object")
    if set(payload) != {"main_task", "actions"}:
        raise ValueError("TRAKE provider output must contain main_task and actions only")

    original = _normalize_text(original_query)
    if not original:
        raise ValueError("TRAKE query cannot be empty")
    main_value = payload.get("main_task")
    actions_value = payload.get("actions")
    if not isinstance(main_value, str):
        raise ValueError("main_task must be a string")
    if not isinstance(actions_value, list):
        raise ValueError("actions must be a JSON array")

    main_task = _normalize_text(main_value)
    if not main_task:
        raise ValueError("main_task cannot be empty")
    if len(main_task.split()) > 40:
        raise ValueError("main_task exceeds the 40-word limit")
    if not 1 <= len(actions_value) <= MAX_TRAKE_ACTIONS:
        raise ValueError(f"actions must contain between 1 and {MAX_TRAKE_ACTIONS} items")

    actions: list[str] = []
    retrieval_prompts: list[str] = []
    for position, action_value in enumerate(actions_value, start=1):
        if isinstance(action_value, str):
            # Accept legacy/custom providers while the Gemini schema emits the
            # richer object form. Versioned caches prevent old hosted responses
            # from bypassing the new prompt.
            action = _normalize_text(action_value)
            retrieval_prompt = _normalize_text(f"{action} during {main_task}")
        elif isinstance(action_value, Mapping):
            if set(action_value) != {"action", "retrieval_prompt"}:
                raise ValueError(
                    f"action {position} must contain action and retrieval_prompt only"
                )
            action_raw = action_value.get("action")
            prompt_raw = action_value.get("retrieval_prompt")
            if not isinstance(action_raw, str):
                raise ValueError(f"action {position} label must be a string")
            if not isinstance(prompt_raw, str):
                raise ValueError(
                    f"action {position} retrieval_prompt must be a string"
                )
            action = _normalize_text(action_raw)
            retrieval_prompt = _normalize_text(prompt_raw)
        else:
            raise ValueError(f"action {position} must be a string or object")
        if not action:
            raise ValueError(f"action {position} cannot be empty")
        if len(action.split()) > 24:
            raise ValueError(f"action {position} exceeds the 24-word limit")
        if not retrieval_prompt:
            raise ValueError(f"action {position} retrieval_prompt cannot be empty")
        if len(retrieval_prompt.split()) > 48:
            raise ValueError(
                f"action {position} retrieval_prompt exceeds the 48-word limit"
            )
        actions.append(action)
        retrieval_prompts.append(retrieval_prompt)

    return TrakePlan(
        original_query=original,
        main_task=main_task,
        action_list=ActionLinkedList.from_actions(actions, retrieval_prompts),
        model=model,
        prompt_version=prompt_version,
        input_tokens=max(0, int(input_tokens or 0)),
        output_tokens=max(0, int(output_tokens or 0)),
        latency_ms=max(0.0, float(latency_ms)),
    )


def fallback_trake_plan(
    query: str,
    *,
    model: str,
    reason: str,
    latency_ms: float = 0.0,
) -> TrakePlan:
    """Keep retrieval usable if Gemini or its structured output is unavailable."""
    original = _normalize_text(query)
    if not original:
        raise ValueError("TRAKE query cannot be empty")
    return TrakePlan(
        original_query=original,
        main_task=original,
        action_list=ActionLinkedList.from_actions((original,)),
        model=model,
        latency_ms=max(0.0, float(latency_ms)),
        fallback_reason=reason[:500],
    )


class SQLiteTrakeCache:
    """Persistent cache isolated by query, model, and TRAKE prompt version."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def make_key(query: str, model: str, prompt_version: str) -> str:
        material = "\x1f".join((_normalize_text(query), model, prompt_version))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trake_plans (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return connection

    def get(self, query: str, model: str, prompt_version: str) -> TrakePlan | None:
        key = self.make_key(query, model, prompt_version)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result_json FROM trake_plans WHERE cache_key = ?", (key,)
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else TrakePlan.from_dict(json.loads(row[0]))

    def put(self, plan: TrakePlan) -> None:
        if plan.used_fallback:
            return
        key = self.make_key(plan.original_query, plan.model, plan.prompt_version)
        payload = json.dumps(plan.to_dict(), ensure_ascii=False)
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO trake_plans(cache_key, result_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (key, payload),
            )
            connection.commit()
        finally:
            connection.close()


class GeminiTrakeProvider:
    """Google Gen AI adapter using a strict JSON schema for TRAKE planning."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        client: Any | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        # Gemini rejects manually configured deadlines below 10 seconds.
        self.timeout_seconds = max(10.0, float(timeout_seconds))
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout_seconds * 1000)
                ),
            )
        return self._client

    def generate(self, query: str) -> TrakeProviderResponse:
        query_input = json.dumps(query, ensure_ascii=False)
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=f"Plan this temporal video query JSON string:\n{query_input}",
            config={
                "system_instruction": TRAKE_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_json_schema": TRAKE_OUTPUT_SCHEMA,
                "temperature": 0,
                "max_output_tokens": 700,
            },
        )
        payload = json.loads(response.text)
        if not isinstance(payload, Mapping):
            raise ValueError("Gemini returned TRAKE JSON that is not an object")
        usage = getattr(response, "usage_metadata", None)
        return TrakeProviderResponse(
            payload=payload,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )


class GeminiTrakePlanner:
    """Generate, validate, cache, and safely fall back from a TRAKE plan."""

    def __init__(
        self,
        provider: TrakeProvider,
        *,
        cache: SQLiteTrakeCache | None = None,
        prompt_version: str = TRAKE_PROMPT_VERSION,
    ):
        self.provider = provider
        self.cache = cache
        self.prompt_version = prompt_version

    def plan(self, query: str) -> TrakePlan:
        original = _normalize_text(query)
        if not original:
            raise ValueError("TRAKE query cannot be empty")
        started = time.perf_counter()

        if self.cache is not None:
            try:
                cached = self.cache.get(
                    original, self.provider.model, self.prompt_version
                )
            except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
                cached = None
            if cached is not None:
                return replace(
                    cached,
                    cache_hit=True,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

        try:
            response = self.provider.generate(original)
            plan = validate_trake_payload(
                response.payload,
                original,
                model=self.provider.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                prompt_version=self.prompt_version,
            )
        except Exception as error:  # hosted failures must not break local retrieval
            return fallback_trake_plan(
                original,
                model=self.provider.model,
                reason=f"{type(error).__name__}: {error}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if self.cache is not None:
            try:
                self.cache.put(plan)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass
        return plan


def create_trake_planner(
    *,
    model: str | None = None,
    cache_path: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> GeminiTrakePlanner:
    """Build the Gemini TRAKE planner from explicit values and environment."""
    configured_model = (
        model
        or os.getenv("GEMINI_TRAKE_MODEL", "")
        or os.getenv("GEMINI_REWRITE_MODEL", "")
        or DEFAULT_GEMINI_MODEL
    )
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("TRAKE_TIMEOUT_SECONDS", "15")
    )
    provider = GeminiTrakeProvider(
        model=configured_model,
        timeout_seconds=timeout,
    )
    cache = SQLiteTrakeCache(cache_path) if cache_path is not None else None
    return GeminiTrakePlanner(provider, cache=cache)
