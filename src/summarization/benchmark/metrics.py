from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rouge_score import rouge_scorer

from .datasets import BenchmarkExample

ROUGE_METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def evaluate_predictions(
    examples: list[BenchmarkExample],
    predictions: list[str],
    metric_profile: tuple[str, ...],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")
    if not examples:
        raise ValueError("At least one benchmark example is required for evaluation")

    results: dict[str, Any] = {
        "example_count": len(examples),
        "metric_profile": list(metric_profile),
        "length": _compute_length_stats(examples, predictions),
    }

    if "rouge" in metric_profile:
        results["rouge"] = _compute_rouge(examples, predictions)
    if "extractive_fragments" in metric_profile:
        results["prediction_extractiveness"] = _compute_extractive_summary_stats(
            [example.source for example in examples],
            predictions,
        )
        results["reference_extractiveness"] = _compute_extractive_summary_stats(
            [example.source for example in examples],
            [example.reference for example in examples],
        )
    if "novel_ngrams" in metric_profile:
        results["prediction_novel_ngrams"] = _compute_novel_ngram_stats(
            [example.source for example in examples],
            predictions,
        )
        results["reference_novel_ngrams"] = _compute_novel_ngram_stats(
            [example.source for example in examples],
            [example.reference for example in examples],
        )

    return results


def _compute_rouge(examples: list[BenchmarkExample], predictions: list[str]) -> dict[str, dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)
    totals = {
        metric: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for metric in ROUGE_METRICS
    }

    for example, prediction in zip(examples, predictions, strict=True):
        scores = scorer.score(example.reference, prediction)
        for metric in ROUGE_METRICS:
            score = scores[metric]
            totals[metric]["precision"] += float(score.precision)
            totals[metric]["recall"] += float(score.recall)
            totals[metric]["f1"] += float(score.fmeasure)

    sample_count = float(len(examples))
    return {
        metric: {
            key: value / sample_count
            for key, value in values.items()
        }
        for metric, values in totals.items()
    }


def _compute_length_stats(examples: list[BenchmarkExample], predictions: list[str]) -> dict[str, float]:
    total_source_tokens = 0.0
    total_reference_tokens = 0.0
    total_prediction_tokens = 0.0
    total_prediction_to_reference = 0.0

    for example, prediction in zip(examples, predictions, strict=True):
        source_len = len(tokenize(example.source))
        reference_len = len(tokenize(example.reference))
        prediction_len = len(tokenize(prediction))

        total_source_tokens += source_len
        total_reference_tokens += reference_len
        total_prediction_tokens += prediction_len
        if reference_len:
            total_prediction_to_reference += prediction_len / reference_len

    count = float(len(examples))
    return {
        "avg_source_tokens": total_source_tokens / count,
        "avg_reference_tokens": total_reference_tokens / count,
        "avg_prediction_tokens": total_prediction_tokens / count,
        "avg_prediction_to_reference_ratio": total_prediction_to_reference / count,
    }


def _compute_extractive_summary_stats(sources: list[str], summaries: list[str]) -> dict[str, float]:
    totals = defaultdict(float)
    for source, summary in zip(sources, summaries, strict=True):
        stats = extractive_fragment_stats(source, summary)
        for key, value in stats.items():
            totals[key] += value

    count = float(len(sources))
    return {key: value / count for key, value in totals.items()}


def _compute_novel_ngram_stats(sources: list[str], summaries: list[str]) -> dict[str, float]:
    totals = defaultdict(float)
    for source, summary in zip(sources, summaries, strict=True):
        totals["novel_1gram_ratio"] += novel_ngram_ratio(source, summary, 1)
        totals["novel_2gram_ratio"] += novel_ngram_ratio(source, summary, 2)
        totals["novel_3gram_ratio"] += novel_ngram_ratio(source, summary, 3)

    count = float(len(sources))
    return {key: value / count for key, value in totals.items()}


def extractive_fragment_stats(source: str, summary: str) -> dict[str, float]:
    source_tokens = tokenize(source)
    summary_tokens = tokenize(summary)
    if not source_tokens or not summary_tokens:
        return {
            "coverage": 0.0,
            "density": 0.0,
            "compression": 0.0,
        }

    fragment_lengths = extractive_fragment_lengths(source_tokens, summary_tokens)
    summary_length = float(len(summary_tokens))
    return {
        "coverage": sum(fragment_lengths) / summary_length,
        "density": sum(length * length for length in fragment_lengths) / summary_length,
        "compression": len(source_tokens) / summary_length,
    }


def extractive_fragment_lengths(source_tokens: list[str], summary_tokens: list[str]) -> list[int]:
    source_index: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(source_tokens):
        source_index[token].append(index)

    fragment_lengths: list[int] = []
    summary_idx = 0
    while summary_idx < len(summary_tokens):
        best_length = 0
        for source_idx in source_index.get(summary_tokens[summary_idx], []):
            match_length = 0
            while (
                summary_idx + match_length < len(summary_tokens)
                and source_idx + match_length < len(source_tokens)
                and summary_tokens[summary_idx + match_length] == source_tokens[source_idx + match_length]
            ):
                match_length += 1
            if match_length > best_length:
                best_length = match_length

        if best_length > 0:
            fragment_lengths.append(best_length)
            summary_idx += best_length
        else:
            summary_idx += 1

    return fragment_lengths


def novel_ngram_ratio(source: str, summary: str, n: int) -> float:
    source_tokens = tokenize(source)
    summary_tokens = tokenize(summary)
    if len(summary_tokens) < n or n <= 0:
        return 0.0

    source_ngrams = set(generate_ngrams(source_tokens, n))
    summary_ngrams = list(generate_ngrams(summary_tokens, n))
    if not summary_ngrams:
        return 0.0

    novel_count = sum(1 for ngram in summary_ngrams if ngram not in source_ngrams)
    return novel_count / len(summary_ngrams)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
