from __future__ import annotations

from pathlib import Path

import pytest

from summarization.benchmark.usb_ext import (
    USBExtSentenceScore,
    _build_usb_ext_gidf_artifact,
    _build_usb_ext_report,
    _load_usb_ext_gidf_artifact,
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
        lambda sentences, cfg, gidf: (
            {0: 0.1, 1: 0.4, 2: 0.9},
            {0: 0.15, 1: 0.45, 2: 0.95},
            {0: 0.2, 1: 0.5, 2: 0.8},
            {0: 0.18, 1: 0.48, 2: 0.88},
        ),
    )

    records = _score_usb_ext_row(row, cfg, split="test", gidf={"sentence": 2.0})

    assert len(records) == 3
    assert records[0].domain == "companies"
    assert records[0].label == 1
    assert records[2].gidf_lsa_score == pytest.approx(0.95)
    assert records[2].final_score == pytest.approx(0.88)


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
            gidf_lsa_score=0.95,
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
            gidf_lsa_score=0.15,
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
            gidf_lsa_score=0.85,
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
            gidf_lsa_score=0.25,
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
        gidf_metadata={
            "artifact_path": "tmp/usb_ext_gidf.json",
            "source_splits": ["train", "validation", "test"],
            "document_count": 10,
            "term_count": 20,
            "language_code": "en",
            "min_df": 2,
            "max_df": 0.85,
            "sublinear_tf": True,
        },
    )

    assert report["paper_comparison_metrics"]["auc"] == pytest.approx(1.0)
    assert report["best_threshold_by_f1"]["f1"] == pytest.approx(1.0)
    assert report["best_threshold_by_f1"]["threshold"] <= 0.85
    assert report["best_threshold_by_binary_auc"]["binary_auc"] == pytest.approx(1.0)
    assert report["best_threshold_by_binary_auc"]["threshold"] == pytest.approx(0.5)
    assert report["gidf"]["artifact_path"] == "tmp/usb_ext_gidf.json"
    assert set(report["domain_metrics"]) == {"companies", "schools"}


def test_build_usb_ext_gidf_artifact_saves_and_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = SummarizationConfig()
    cfg.gidf.max_df = 1.0
    rows_by_split = {
        "train": [
            {"input_lines": ["alpha beta", "gamma"]},
        ],
        "validation": [
            {"input_lines": ["alpha delta"]},
        ],
    }

    def fake_iter_raw_rows(dataset_name: str, *, data_dir: Path, split: str):
        assert dataset_name == "usb_ext"
        assert data_dir == tmp_path
        return iter(rows_by_split[split])

    monkeypatch.setattr("summarization.benchmark.usb_ext.iter_raw_rows", fake_iter_raw_rows)

    artifact_path = tmp_path / "tmp" / "usb_ext_gidf.json"
    gidf_scores, metadata = _build_usb_ext_gidf_artifact(
        cfg=cfg,
        data_dir=tmp_path,
        output_path=artifact_path,
        splits=("train", "validation"),
    )
    loaded_scores, loaded_metadata = _load_usb_ext_gidf_artifact(artifact_path)

    assert artifact_path.exists() is True
    assert gidf_scores == loaded_scores
    assert set(gidf_scores) == {"alpha"}
    assert metadata["artifact_path"] == str(artifact_path)
    assert metadata["document_count"] == 2
    assert metadata["term_count"] == 1
    assert metadata["source_splits"] == ["train", "validation"]
    assert loaded_metadata["source_splits"] == ["train", "validation"]
