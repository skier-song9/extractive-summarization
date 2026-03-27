from __future__ import annotations

import numpy as np

from .config import ExtractionConfig, FusionConfig
from .utils import count_tokens


def fuse_scores(
    lsa_scores: dict[int, float],
    pagerank_scores: dict[int, float],
    cfg: FusionConfig,
) -> dict[int, float]:
    return {
        index: cfg.alpha * lsa_scores[index] + (1.0 - cfg.alpha) * pagerank_scores[index]
        for index in lsa_scores
    }


def is_redundant(
    candidate_index: int,
    kept_indices: list[int],
    sim_matrix: np.ndarray,
    threshold: float,
) -> bool:
    return any(sim_matrix[candidate_index, kept_index] >= threshold for kept_index in kept_indices)


def resolve_token_budget(
    source_token_count: int | None,
    sentence_token_counts: dict[int, int],
    token_budget_ratio: float,
) -> int:
    if not sentence_token_counts:
        return 0

    resolved_source_tokens = source_token_count or sum(sentence_token_counts.values())
    ratio_budget = max(1, int(resolved_source_tokens * token_budget_ratio))

    # Sentences are indivisible extraction units, so keep enough budget
    # to include at least the shortest candidate sentence.
    return max(ratio_budget, min(sentence_token_counts.values()))


def resolve_top_k_limit(sentence_count: int, top_k: int) -> int:
    return min(sentence_count, top_k)


def passes_final_score_threshold(score: float, threshold: float) -> bool:
    return score >= threshold


def extract_top_sentences(
    sentences: list[str],
    fused_scores: dict[int, float],
    sim_matrix: np.ndarray,
    cfg: ExtractionConfig,
    source_token_count: int | None = None,
) -> list[str]:
    if not sentences:
        return []

    sentence_token_counts = {
        index: count_tokens(sentence)
        for index, sentence in enumerate(sentences)
    }
    ranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
    kept_indices: list[int] = []
    selected_token_count = 0
    token_budget: int | None = None
    top_k_limit: int | None = None
    final_score_threshold: float | None = None

    if cfg.token_budget_ratio is not None:
        token_budget = resolve_token_budget(
            source_token_count,
            sentence_token_counts,
            cfg.token_budget_ratio,
        )
    elif cfg.top_k is not None:
        top_k_limit = resolve_top_k_limit(len(sentences), cfg.top_k or 0)
    else:
        final_score_threshold = cfg.final_score_threshold

    for index in ranked_indices:
        if final_score_threshold is not None:
            if not passes_final_score_threshold(fused_scores[index], final_score_threshold):
                break
        if is_redundant(index, kept_indices, sim_matrix, cfg.redundancy_threshold):
            continue

        if token_budget is not None:
            sentence_token_count = sentence_token_counts[index]
            if selected_token_count + sentence_token_count > token_budget:
                continue

        kept_indices.append(index)
        if token_budget is not None:
            selected_token_count += sentence_token_count

        if token_budget is not None and selected_token_count >= token_budget:
            break
        if top_k_limit is not None and len(kept_indices) >= top_k_limit:
            break

    if cfg.preserve_order:
        kept_indices = sorted(kept_indices)

    return [sentences[index] for index in kept_indices]
