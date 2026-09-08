#!/usr/bin/env python3
"""OpenAI-compatible text-judge reward used by the GRPO recipes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_CLIENT = None
_CACHE_CONNECTION: sqlite3.Connection | None = None
_LOCK = threading.Lock()


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def extract_question(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        return ""
    value = extra_info.get("question") or extra_info.get("user_question") or ""
    return clean_text(str(value).replace("<image>", " "))


def api_config() -> tuple[str, str, str]:
    model = os.environ.get("VISIONFOUNDRY_REWARD_MODEL", "Qwen2.5-3B")
    api_key = os.environ.get("VISIONFOUNDRY_REWARD_API_KEY")
    base_url = os.environ.get("VISIONFOUNDRY_REWARD_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("Set VISIONFOUNDRY_REWARD_API_KEY")
    return model, api_key, base_url


def client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _, api_key, base_url = api_config()
        timeout = float(os.environ.get("VISIONFOUNDRY_REWARD_TIMEOUT", "90"))
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return _CLIENT


def cache_connection() -> sqlite3.Connection:
    global _CACHE_CONNECTION
    if _CACHE_CONNECTION is None:
        default_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        path = Path(
            os.environ.get(
                "VISIONFOUNDRY_REWARD_CACHE_DB",
                default_root / "visionfoundry" / "text_judge_reward.sqlite",
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_CONNECTION = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
        _CACHE_CONNECTION.execute("pragma journal_mode=wal")
        _CACHE_CONNECTION.execute(
            "create table if not exists reward_cache ("
            "cache_key text primary key, model text not null, question text not null, "
            "ground_truth text not null, answer text not null, score real not null, "
            "raw_output text not null, created_at integer not null)"
        )
        _CACHE_CONNECTION.commit()
    return _CACHE_CONNECTION


def parse_yes_no(text: str) -> bool | None:
    cleaned = clean_text(text)
    matches = re.findall(r"answer\s*[:：]\s*(yes|no)\b", cleaned, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "yes"
    words = re.sub(r"[^a-zA-Z]+", " ", cleaned).strip().lower().split()
    if len(words) <= 4 and "yes" in words and "no" not in words:
        return True
    if len(words) <= 4 and "no" in words and "yes" not in words:
        return False
    return None


def judge_messages(question: str, ground_truth: str, answer: str) -> list[dict[str, str]]:
    system = (
        "You are a strict answer verifier for visual question answering. You are not "
        "given the image. Judge only whether the model answer is semantically consistent "
        "with the ground-truth answer for the same question. Accept synonyms, concise "
        "paraphrases, and differences in case, punctuation, and articles. Require the same "
        "polarity or target value. Reject ambiguous, conflicting, or contradictory answers. "
        "Output exactly one line: Answer: YES or Answer: NO."
    )
    user = (
        f"Question:\n{question or '[not provided]'}\n\n"
        f"Ground-truth answer:\n{ground_truth}\n\n"
        f"Model answer:\n{answer}\n\n"
        "Does the model answer match the ground-truth answer?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def cache_key(model: str, question: str, ground_truth: str, answer: str) -> str:
    payload = json.dumps(
        [model, question, ground_truth, answer], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def read_cache(key: str) -> float | None:
    with _LOCK:
        row = cache_connection().execute(
            "select score from reward_cache where cache_key = ?", (key,)
        ).fetchone()
    return None if row is None else float(row[0])


def write_cache(
    key: str,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
    score: float,
    raw_output: str,
) -> None:
    with _LOCK:
        connection = cache_connection()
        connection.execute(
            "insert or replace into reward_cache values (?, ?, ?, ?, ?, ?, ?, ?)",
            (key, model, question, ground_truth, answer, score, raw_output, int(time.time())),
        )
        connection.commit()


def call_judge(question: str, ground_truth: str, answer: str) -> tuple[float, str]:
    model, _, _ = api_config()
    retries = int(os.environ.get("VISIONFOUNDRY_REWARD_MAX_RETRIES", "10"))
    retry_sleep = float(os.environ.get("VISIONFOUNDRY_REWARD_RETRY_SLEEP", "3"))
    max_tokens = int(os.environ.get("VISIONFOUNDRY_REWARD_MAX_TOKENS", "64"))
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client().chat.completions.create(
                model=model,
                messages=judge_messages(question, ground_truth, answer),
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw = clean_text(response.choices[0].message.content)
            parsed = parse_yes_no(raw)
            if parsed is None:
                raise ValueError(f"Unparseable judge response: {raw!r}")
            return (1.0 if parsed else 0.0), raw
        except Exception as exc:  # The API client exposes provider-specific exceptions.
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"Reward judge failed after {retries} attempts") from last_error


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> float:
    del data_source, kwargs
    question = extract_question(extra_info)
    reference = clean_text(ground_truth)
    answer = clean_text(solution_str)
    if not reference or not answer:
        return 0.0

    model, _, _ = api_config()
    key = cache_key(model, question, reference, answer)
    cached = read_cache(key)
    if cached is not None:
        return cached
    score, raw = call_judge(question, reference, answer)
    write_cache(key, model, question, reference, answer, score, raw)
    return score


if __name__ == "__main__":
    for sample in ("YES", "yes.", "Answer: YES", "Answer: no", "No, they differ."):
        print(sample, "=>", parse_yes_no(sample))

