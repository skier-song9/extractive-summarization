from __future__ import annotations

from pathlib import Path

import pytest

from summarization.benchmark.datasets import BenchmarkExample
from summarization.benchmark.pipeline import run_benchmark
from summarization.config import SummarizationConfig


def test_run_benchmark_generates_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    examples = [
        BenchmarkExample(
            dataset_name="cnn_dailymail",
            split="test",
            example_id="a",
            source="Source text one.",
            reference="Summary one.",
        ),
        BenchmarkExample(
            dataset_name="cnn_dailymail",
            split="test",
            example_id="b",
            source="Source text two.",
            reference="Summary two.",
        ),
    ]

    class FakeSummarizer:
        def __init__(self, cfg: SummarizationConfig) -> None:
            self.cfg = cfg

        def summarize_batch(self, texts: list[str]) -> list[str]:
            return [f"pred::{text}" for text in texts]

    monkeypatch.setattr("summarization.benchmark.pipeline.prepare_dataset", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("summarization.benchmark.pipeline.iter_benchmark_examples", lambda *args, **kwargs: examples)
    monkeypatch.setattr(
        "summarization.benchmark.pipeline.SummarizationConfig.from_yaml",
        lambda path: SummarizationConfig(),
    )
    monkeypatch.setattr("summarization.benchmark.pipeline.HybridExtractiveSummarizer", FakeSummarizer)

    report = run_benchmark(
        "cnn_dailymail",
        config_path="config/summarization.yaml",
        data_dir=tmp_path,
        max_samples=2,
        batch_size=2,
    )

    assert report["dataset"]["name"] == "cnn_dailymail"
    assert report["run"]["max_samples"] == 2
    assert len(report["predictions"]) == 2
    assert report["predictions"][0]["prediction"] == "pred::Source text one."
