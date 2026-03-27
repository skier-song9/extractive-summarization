from __future__ import annotations

import numpy as np
import pytest

from summarization.config import ExtractionConfig, FusionConfig
from summarization.fusion import extract_top_sentences, fuse_scores


def test_fuse_scores_blends_lsa_and_pagerank() -> None:
    fused = fuse_scores(
        {0: 0.2, 1: 0.8},
        {0: 0.6, 1: 0.4},
        FusionConfig(alpha=0.25),
    )

    assert fused[0] == pytest.approx(0.5)
    assert fused[1] == pytest.approx(0.5)


def test_extract_top_sentences_preserves_order_and_removes_redundancy() -> None:
    sentences = ["Sentence A", "Sentence B", "Sentence C"]
    fused_scores = {0: 0.95, 1: 0.90, 2: 0.30}
    sim_matrix = np.array(
        [
            [1.0, 0.98, 0.10],
            [0.98, 1.0, 0.15],
            [0.10, 0.15, 1.0],
        ],
        dtype=np.float32,
    )
    cfg = ExtractionConfig(token_budget_ratio=0.67, preserve_order=True, redundancy_threshold=0.95)

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg)

    assert selected == ["Sentence A", "Sentence C"]


def test_extract_top_sentences_respects_token_budget_ratio() -> None:
    sentences = [
        "a",
        "b",
        "c one two three four",
        "d one",
        "e",
    ]
    fused_scores = {0: 0.50, 1: 0.60, 2: 0.70, 3: 0.80, 4: 0.90}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=0.30,
        preserve_order=False,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg, source_token_count=10)

    assert selected == ["e", "d one"]


def test_extract_top_sentences_respects_top_k() -> None:
    sentences = ["Sentence A", "Sentence B", "Sentence C", "Sentence D"]
    fused_scores = {0: 0.20, 1: 0.95, 2: 0.70, 3: 0.85}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=None,
        top_k=2,
        preserve_order=False,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg)

    assert selected == ["Sentence B", "Sentence D"]


def test_extract_top_sentences_caps_top_k_to_sentence_count() -> None:
    sentences = ["Sentence A", "Sentence B", "Sentence C"]
    fused_scores = {0: 0.30, 1: 0.90, 2: 0.60}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=None,
        top_k=10,
        preserve_order=True,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg)

    assert selected == ["Sentence A", "Sentence B", "Sentence C"]


def test_extract_top_sentences_prefers_token_budget_ratio_over_top_k() -> None:
    sentences = ["a", "b", "c one two three four", "d one", "e"]
    fused_scores = {0: 0.50, 1: 0.60, 2: 0.70, 3: 0.80, 4: 0.90}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=0.30,
        top_k=5,
        final_score_threshold=0.95,
        preserve_order=False,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg, source_token_count=10)

    assert selected == ["e", "d one"]


def test_extract_top_sentences_respects_final_score_threshold() -> None:
    sentences = ["Sentence A", "Sentence B", "Sentence C", "Sentence D"]
    fused_scores = {0: 0.20, 1: 0.95, 2: 0.70, 3: 0.85}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=None,
        top_k=None,
        final_score_threshold=0.80,
        preserve_order=True,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg)

    assert selected == ["Sentence B", "Sentence D"]


def test_extract_top_sentences_prefers_top_k_over_final_score_threshold() -> None:
    sentences = ["Sentence A", "Sentence B", "Sentence C", "Sentence D"]
    fused_scores = {0: 0.20, 1: 0.95, 2: 0.70, 3: 0.85}
    sim_matrix = np.eye(len(sentences), dtype=np.float32)
    cfg = ExtractionConfig(
        token_budget_ratio=None,
        top_k=3,
        final_score_threshold=0.90,
        preserve_order=False,
        redundancy_threshold=0.95,
    )

    selected = extract_top_sentences(sentences, fused_scores, sim_matrix, cfg)

    assert selected == ["Sentence B", "Sentence D", "Sentence C"]
