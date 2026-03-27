from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics

from ..config import SummarizationConfig
from ..embedder import compute_similarity_matrix, embed_sentences
from ..fusion import fuse_scores
from ..graph_ranker import compute_pagerank_scores
from ..lsa_scorer import compute_lsa_scores
from ..utils import get_logger
from .datasets import get_dataset_spec, iter_raw_rows, prepare_dataset


@dataclass(slots=True)
class USBExtSentenceScore:
    example_id: str
    domain: str
    split: str
    sentence_index: int
    sentence: str
    label: int
    lsa_score: float
    pagerank_score: float
    final_score: float


def run_usb_ext_evaluation(
    *,
    config_path: str | Path = "config/benchmark.yaml",
    data_dir: str | Path = "data/benchmarks",
    split: str | None = None,
    max_samples: int | None = None,
    force_prepare: bool = False,
    threshold_step: float = 0.01,
    preview_examples: int = 3,
) -> tuple[dict[str, Any], list[USBExtSentenceScore]]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if not 0.0 < threshold_step <= 1.0:
        raise ValueError("threshold_step must be within the range (0.0, 1.0].")

    dataset_name = "usb_ext"
    spec = get_dataset_spec(dataset_name)
    resolved_split = split or spec.default_split
    logger = get_logger("benchmark.usb_ext", "INFO")

    prepare_dataset(
        dataset_name,
        data_dir=data_dir,
        force=force_prepare,
    )
    cfg = SummarizationConfig.from_yaml(config_path)

    started_at = time.perf_counter()
    sentence_scores: list[USBExtSentenceScore] = []
    preview_documents: list[dict[str, Any]] = []
    document_count = 0

    for row in iter_raw_rows(dataset_name, data_dir=data_dir, split=resolved_split):
        document_count += 1
        document_scores = _score_usb_ext_row(row, cfg, split=resolved_split)
        if not document_scores:
            continue
        sentence_scores.extend(document_scores)
        if len(preview_documents) < preview_examples:
            preview_documents.append(_build_preview_document(document_scores))
        if document_count % 50 == 0:
            logger.info("Scored %d USB EXT documents", document_count)
        if max_samples is not None and document_count >= max_samples:
            break

    elapsed_seconds = time.perf_counter() - started_at
    if not sentence_scores:
        raise ValueError("USB EXT evaluation produced no sentence scores.")

    report = _build_usb_ext_report(
        sentence_scores=sentence_scores,
        spec=spec,
        config_path=config_path,
        split=resolved_split,
        threshold_step=threshold_step,
        max_samples=max_samples,
        elapsed_seconds=elapsed_seconds,
        preview_documents=preview_documents,
    )
    return report, sentence_scores


def save_usb_ext_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_usb_ext_sentence_scores(
    sentence_scores: list[USBExtSentenceScore],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sentence_scores:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def _score_usb_ext_row(
    row: dict[str, Any],
    cfg: SummarizationConfig,
    *,
    split: str,
) -> list[USBExtSentenceScore]:
    raw_sentences = row.get("input_lines")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        return []

    sentences = [_normalize_usb_sentence(sentence) for sentence in raw_sentences]
    labels = _normalize_usb_labels(row.get("labels"), len(sentences))
    lsa_scores, pagerank_scores, final_scores = _compute_usb_sentence_scores(sentences, cfg)

    example_id_raw = row.get("id")
    example_id = str(example_id_raw) if example_id_raw not in {None, ""} else f"{split}:{hash(tuple(sentences))}"
    domain = row.get("domain") or _infer_domain(example_id)

    return [
        USBExtSentenceScore(
            example_id=example_id,
            domain=domain,
            split=split,
            sentence_index=index,
            sentence=sentence,
            label=labels[index],
            lsa_score=float(lsa_scores.get(index, 0.0)),
            pagerank_score=float(pagerank_scores.get(index, 0.0)),
            final_score=float(final_scores.get(index, 0.0)),
        )
        for index, sentence in enumerate(sentences)
    ]


def _compute_usb_sentence_scores(
    sentences: list[str],
    cfg: SummarizationConfig,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    if not sentences:
        return {}, {}, {}

    embeddings = embed_sentences(sentences, cfg.embedding)
    sim_matrix = compute_similarity_matrix(embeddings)
    lsa_scores = compute_lsa_scores(
        sentences,
        cfg.lsa,
        cfg.embedding,
        cfg.preprocessing.language,
    )
    pagerank_scores = compute_pagerank_scores(sim_matrix, cfg.graph)
    final_scores = fuse_scores(lsa_scores, pagerank_scores, cfg.fusion)
    return lsa_scores, pagerank_scores, final_scores


def _normalize_usb_sentence(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_usb_labels(labels: Any, sentence_count: int) -> list[int]:
    normalized: list[int] = []
    raw_labels = labels if isinstance(labels, list) else []
    for index in range(sentence_count):
        raw_value = raw_labels[index] if index < len(raw_labels) else 0
        try:
            normalized.append(1 if int(raw_value) == 1 else 0)
        except (TypeError, ValueError):
            normalized.append(0)
    return normalized


def _infer_domain(example_id: str) -> str:
    prefix, _, _ = example_id.partition("/")
    return prefix


def _build_usb_ext_report(
    *,
    sentence_scores: list[USBExtSentenceScore],
    spec,
    config_path: str | Path,
    split: str,
    threshold_step: float,
    max_samples: int | None,
    elapsed_seconds: float,
    preview_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([record.label for record in sentence_scores], dtype=np.int32)
    scores = np.asarray([record.final_score for record in sentence_scores], dtype=np.float64)
    domains = sorted({record.domain for record in sentence_scores})

    threshold_grid = _build_threshold_grid(threshold_step)
    threshold_sweep = [_threshold_metrics(labels, scores, threshold) for threshold in threshold_grid]
    best_threshold = _best_threshold_by_f1(labels, scores)
    rounded_metrics = _rounded_metrics(labels, scores)
    continuous_auc = _safe_roc_auc(labels, scores)

    domain_metrics: dict[str, Any] = {}
    for domain in domains:
        domain_records = [record for record in sentence_scores if record.domain == domain]
        domain_labels = np.asarray([record.label for record in domain_records], dtype=np.int32)
        domain_scores = np.asarray([record.final_score for record in domain_records], dtype=np.float64)
        domain_metrics[domain] = {
            "sentence_count": int(len(domain_records)),
            "positive_labels": int(domain_labels.sum()),
            "positive_rate": float(domain_labels.mean()) if len(domain_labels) else 0.0,
            "auc": _safe_roc_auc(domain_labels, domain_scores),
            "default_round_metrics": _rounded_metrics(domain_labels, domain_scores),
            "best_threshold_by_f1": _best_threshold_by_f1(domain_labels, domain_scores),
        }

    return {
        "dataset": {
            "name": spec.name,
            "source_id": spec.source_id,
            "config": spec.config,
            "split": split,
        },
        "run": {
            "config_path": str(config_path),
            "max_samples": max_samples,
            "documents_evaluated": len({record.example_id for record in sentence_scores}),
            "sentences_evaluated": len(sentence_scores),
            "elapsed_seconds": elapsed_seconds,
            "documents_per_second": (
                len({record.example_id for record in sentence_scores}) / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
            "sentences_per_second": len(sentence_scores) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        },
        "paper_comparison_metrics": {
            "auc": continuous_auc,
            "default_round_metrics": rounded_metrics,
            "note": (
                "USB's reference evaluation computes AUC from continuous sentence scores. "
                "Threshold sweeps are useful for selecting an operating point, but do not change this AUC."
            ),
        },
        "best_threshold_by_f1": best_threshold,
        "threshold_sweep_step": threshold_step,
        "threshold_sweep": threshold_sweep,
        "score_distribution": {
            "min_final_score": float(scores.min()),
            "max_final_score": float(scores.max()),
            "mean_final_score": float(scores.mean()),
            "median_final_score": float(np.median(scores)),
            "positive_rate": float(labels.mean()),
        },
        "domain_metrics": domain_metrics,
        "preview_documents": preview_documents,
    }


def _build_threshold_grid(step: float) -> list[float]:
    thresholds = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float64)
    clipped = np.clip(thresholds, 0.0, 1.0)
    return [float(value) for value in np.unique(np.round(clipped, 10))]


def _rounded_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = np.rint(scores).astype(np.int32)
    metrics_dict = _classification_metrics(labels, predictions)
    metrics_dict["threshold"] = 0.5
    return metrics_dict


def _best_threshold_by_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        best = _threshold_metrics(labels, scores, 0.5)
        best["search_candidate_count"] = 1
        return best

    precisions = precision[:-1]
    recalls = recall[:-1]
    f1_scores = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )

    best_index = max(
        range(len(thresholds)),
        key=lambda index: (
            float(f1_scores[index]),
            float(precisions[index]),
            -float(thresholds[index]),
        ),
    )
    best = _threshold_metrics(labels, scores, float(thresholds[best_index]))
    best["search_candidate_count"] = int(len(thresholds))
    return best


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int32)
    metrics_dict = _classification_metrics(labels, predictions)
    metrics_dict["threshold"] = float(threshold)
    metrics_dict["selected_sentences"] = int(predictions.sum())
    metrics_dict["selected_rate"] = float(predictions.mean()) if len(predictions) else 0.0
    metrics_dict["binary_auc"] = _safe_roc_auc(labels, predictions.astype(np.float64))
    return metrics_dict


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(metrics.precision_score(labels, predictions, zero_division=0)),
        "recall": float(metrics.recall_score(labels, predictions, zero_division=0)),
        "f1": float(metrics.f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(metrics.accuracy_score(labels, predictions)),
    }


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(metrics.roc_auc_score(labels, scores))


def _build_preview_document(records: list[USBExtSentenceScore]) -> dict[str, Any]:
    ranked = sorted(records, key=lambda item: item.final_score, reverse=True)
    top_sentences = [
        {
            "sentence_index": item.sentence_index,
            "label": item.label,
            "final_score": item.final_score,
            "sentence": item.sentence[:200],
        }
        for item in ranked[:3]
    ]
    return {
        "example_id": records[0].example_id,
        "domain": records[0].domain,
        "sentence_count": len(records),
        "positive_labels": int(sum(record.label for record in records)),
        "top_sentences": top_sentences,
    }
