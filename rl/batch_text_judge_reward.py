#!/usr/bin/env python3
"""Concurrent batch adapter for :mod:`text_judge_reward`."""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_REWARD_PATH = Path(__file__).with_name("text_judge_reward.py")
_SPEC = importlib.util.spec_from_file_location("visionfoundry_text_judge_reward", _REWARD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load reward module from {_REWARD_PATH}")
_REWARD_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWARD_MODULE)
compute_one = _REWARD_MODULE.compute_score


def as_list(value: Any, size: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default] * size
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value] * size


def compute_score(
    data_source: Any = None,
    solution_str: Any = None,
    ground_truth: Any = None,
    extra_info: Any = None,
    data_sources: Any = None,
    solution_strs: Any = None,
    ground_truths: Any = None,
    extra_infos: Any = None,
    concurrency: int | None = None,
    **kwargs: Any,
) -> float | list[float]:
    if solution_strs is None:
        return compute_one(data_source, solution_str, ground_truth, extra_info, **kwargs)

    solutions = list(solution_strs)
    size = len(solutions)
    sources = as_list(data_sources, size, data_source)
    references = as_list(ground_truths, size, ground_truth)
    extras = as_list(extra_infos, size, extra_info)
    workers = min(
        size,
        max(
            1,
            int(concurrency or os.environ.get("VISIONFOUNDRY_REWARD_CONCURRENCY", "32")),
        ),
    )

    def score(index: int) -> float:
        return compute_one(
            sources[index], solutions[index], references[index], extras[index], **kwargs
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(score, range(size)))

