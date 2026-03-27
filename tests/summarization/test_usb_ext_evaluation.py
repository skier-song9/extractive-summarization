from __future__ import annotations

import pytest

from summarization.benchmark.usb_ext import (
    USBExtSentenceScore,
    _build_usb_ext_report,
    _score_usb_ext_row,
)
from summarization.config import SummarizationConfig


def test_score_usb_ext_row_emits_sentence_level_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig()
    row = {
        "id": "companies/example-1.json",
        "input_lines": ["Title", "Sentence one.", "Sentence two."],
        "labels": [1, 0, 1],
    }

    monkeypatch.setattr(
        "summarization.benchmark.usb_ext._compute_usb_sentence_scores",
        lambda sentences, cfg: (
            {0: 0.1, 1: 0.4, 2: 0.9},
            {0: 0.2, 1: 0.5, 2: 0.8},
            {0: 0.15, 1: 0.45, 2: 0.85},
        ),
    )

    records = _score_usb_ext_row(row, cfg, split="test")

    assert len(records) == 3
    assert records[0].domain == "companies"
    assert records[0].label == 1
    assert records[2].final_score == pytest.approx(0.85)


def test_build_usb_ext_report_returns_auc_and_best_threshold() -> None:
    sentence_scores = [
        USBExtSentenceScore(
            example_id="doc-1",
            domain="companies",
            split="test",
            sentence_index=0,
            sentence="A",
            label=1,
            lsa_score=0.9,
            pagerank_score=0.8,
            final_score=0.95,
        ),
        USBExtSentenceScore(
            example_id="doc-1",
            domain="companies",
            split="test",
            sentence_index=1,
            sentence="B",
            label=0,
            lsa_score=0.2,
            pagerank_score=0.3,
            final_score=0.15,
        ),
        USBExtSentenceScore(
            example_id="doc-2",
            domain="schools",
            split="test",
            sentence_index=0,
            sentence="C",
            label=1,
            lsa_score=0.8,
            pagerank_score=0.7,
            final_score=0.85,
        ),
        USBExtSentenceScore(
            example_id="doc-2",
            domain="schools",
            split="test",
            sentence_index=1,
            sentence="D",
            label=0,
            lsa_score=0.3,
            pagerank_score=0.4,
            final_score=0.25,
        ),
    ]
    spec = type(
        "Spec",
        (),
        {
            "name": "usb_ext",
            "source_id": "kundank/usb",
            "config": "extractive_summarization",
        },
    )()

    report = _build_usb_ext_report(
        sentence_scores=sentence_scores,
        spec=spec,
        config_path="config/benchmark.yaml",
        split="test",
        threshold_step=0.5,
        max_samples=None,
        elapsed_seconds=1.0,
        preview_documents=[],
    )

    assert report["paper_comparison_metrics"]["auc"] == pytest.approx(1.0)
    assert report["best_threshold_by_f1"]["f1"] == pytest.approx(1.0)
    assert report["best_threshold_by_f1"]["threshold"] <= 0.85
    assert set(report["domain_metrics"]) == {"companies", "schools"}
