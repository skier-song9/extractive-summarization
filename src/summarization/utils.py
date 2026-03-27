from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

import numpy as np

T = TypeVar("T")

LANGUAGE_CODE_TO_SUMY = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "tr": "turkish",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ar": "arabic",
    "el": "greek",
    "pl": "polish",
    "uk": "ukrainian",
    "cs": "czech",
    "sk": "slovak",
    "he": "hebrew",
    "th": "thai",
}


def get_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def normalize_language_name(language: str) -> str:
    return LANGUAGE_CODE_TO_SUMY.get(language.lower(), language.lower())


def count_tokens(text: str) -> int:
    return len(text.split())


def minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi, abs_tol=1e-9):
        return {key: 1.0 for key in scores}
    normalized = {}
    for key, value in scores.items():
        scaled = (float(value) - lo) / (hi - lo)
        normalized[key] = float(np.clip(scaled, 0.0, 1.0))
    return normalized


def rank_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    ranked_indices = sorted(scores, key=scores.get)
    total = len(ranked_indices)
    return {index: (rank + 1) / total for rank, index in enumerate(ranked_indices)}


def chunked(items: Iterable[T], chunk_size: int) -> Iterator[list[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_worker_count(n_workers: int) -> int:
    if n_workers == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, n_workers)
