from __future__ import annotations

from summarization.benchmark.datasets import BenchmarkExample
from summarization.benchmark.metrics import (
    evaluate_predictions,
    extractive_fragment_stats,
    novel_ngram_ratio,
)


def test_extractive_fragment_stats_reports_expected_values() -> None:
    stats = extractive_fragment_stats(
        "alpha beta gamma delta",
        "beta gamma",
    )

    assert stats["coverage"] == 1.0
    assert stats["density"] == 2.0
    assert stats["compression"] == 2.0


def test_novel_ngram_ratio_detects_unseen_ngrams() -> None:
    ratio = novel_ngram_ratio(
        "alpha beta gamma delta",
        "alpha beta epsilon",
        2,
    )

    assert ratio == 0.5


def test_evaluate_predictions_returns_rouge_and_diagnostics() -> None:
    examples = [
        BenchmarkExample(
            dataset_name="cnn_dailymail",
            split="test",
            example_id="1",
            source="alpha beta gamma delta",
            reference="alpha beta",
        )
    ]
    predictions = ["alpha beta"]

    metrics = evaluate_predictions(examples, predictions, ("rouge", "extractive_fragments"))

    assert metrics["example_count"] == 1
    assert metrics["rouge"]["rouge1"]["f1"] == 1.0
    assert metrics["prediction_extractiveness"]["coverage"] == 1.0
    assert metrics["reference_extractiveness"]["coverage"] == 1.0
