from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import SummarizationConfig
from ..summarizer import HybridExtractiveSummarizer
from ..utils import chunked
from .datasets import (
    BenchmarkExample,
    dataset_manifest_path,
    get_dataset_spec,
    iter_benchmark_examples,
    prepare_dataset,
)
from .metrics import evaluate_predictions


@dataclass(slots=True)
class BenchmarkPrediction:
    example_id: str
    prediction: str
    reference: str
    source_preview: str
    prediction_preview: str
    reference_preview: str
    metadata: dict[str, Any]


def run_benchmark(
    dataset_name: str,
    *,
    config_path: str | Path = "config/benchmark.yaml",
    data_dir: str | Path = "data/benchmarks",
    split: str | None = None,
    max_samples: int = 100,
    batch_size: int = 8,
    force_prepare: bool = False,
    pubmed_max_files: int | None = None,
    preview_examples: int = 3,
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    spec = get_dataset_spec(dataset_name)
    prepare_dataset(
        dataset_name,
        data_dir=data_dir,
        force=force_prepare,
        pubmed_max_files=pubmed_max_files,
    )
    examples = iter_benchmark_examples(
        dataset_name,
        data_dir=data_dir,
        split=split,
        max_samples=max_samples,
    )
    if not examples:
        raise ValueError(f"No usable examples found for dataset '{dataset_name}'")

    cfg = SummarizationConfig.from_yaml(config_path)
    summarizer = HybridExtractiveSummarizer(cfg)

    started_at = time.perf_counter()
    predictions: list[str] = []
    for batch in chunked(examples, batch_size):
        batch_predictions = summarizer.summarize_batch([example.source for example in batch])
        predictions.extend(batch_predictions)
    elapsed_seconds = time.perf_counter() - started_at

    metrics = evaluate_predictions(examples, predictions, spec.metric_profile)
    report = {
        "dataset": {
            "name": spec.name,
            "source_id": spec.source_id,
            "config": spec.config,
            "split": split or spec.default_split,
            "metric_profile": list(spec.metric_profile),
            "notes": list(spec.notes),
            "manifest_path": str(dataset_manifest_path(data_dir, dataset_name)),
        },
        "run": {
            "config_path": str(config_path),
            "max_samples": len(examples),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed_seconds,
            "examples_per_second": len(examples) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        },
        "metrics": metrics,
        "predictions": serialize_predictions(examples, predictions, preview_examples=preview_examples),
    }
    return report


def serialize_predictions(
    examples: list[BenchmarkExample],
    predictions: list[str],
    *,
    preview_examples: int,
) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for example, prediction in list(zip(examples, predictions, strict=True))[:preview_examples]:
        item = BenchmarkPrediction(
            example_id=example.example_id,
            prediction=prediction,
            reference=example.reference,
            source_preview=example.source[:500],
            prediction_preview=prediction[:500],
            reference_preview=example.reference[:500],
            metadata=example.metadata,
        )
        preview.append(asdict(item))
    return preview


def save_benchmark_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
