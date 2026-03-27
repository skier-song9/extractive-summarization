from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarization.config import EmbeddingConfig, ExtractionConfig, ParallelConfig, PreprocessingConfig, SummarizationConfig
from summarization.fusion import extract_top_sentences, fuse_scores
from summarization.graph_ranker import compute_pagerank_scores
from summarization.lsa_scorer import compute_lsa_scores
from summarization.sentence_splitter import split_sentences
from summarization.text_unit_dataset import create_postgres_storage, load_text_unit_contents
from summarization.embedder import compute_similarity_matrix
from summarization.utils import count_tokens

REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "live_text_unit_test_report.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "live_text_unit_test_report.md"
INTEGRATION_TEST_TARGETS = [
    "tests/integration/test_text_unit_data_preparation.py",
    "tests/integration/test_extractive_summarization_pipeline.py",
]


def deterministic_embeddings(sentences: list[str], _: EmbeddingConfig) -> np.ndarray:
    features: list[list[float]] = []
    for index, sentence in enumerate(sentences):
        token_count = len(sentence.split())
        char_count = len(sentence)
        features.append([float(index + 1), float(token_count), float(char_count)])
    return np.asarray(features, dtype=np.float32)


def summarize_scores(scores: dict[int, float], top_k: int = 5) -> list[dict[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [{"index": int(index), "score": round(float(score), 6)} for index, score in ranked]


def preview_matrix(matrix: np.ndarray, size: int = 3) -> list[list[float]]:
    if matrix.size == 0:
        return []
    clipped = matrix[:size, :size]
    return [[round(float(value), 6) for value in row] for row in clipped.tolist()]


def extract_selected_indices(sentences: list[str], selected_sentences: list[str]) -> list[int]:
    selected_indices: list[int] = []
    search_start = 0
    for selected in selected_sentences:
        for index in range(search_start, len(sentences)):
            if sentences[index] == selected:
                selected_indices.append(index)
                search_start = index + 1
                break
    return selected_indices


def summarize_text(text: str, cfg: SummarizationConfig) -> dict[str, Any]:
    sentences = split_sentences(text, cfg.preprocessing)
    if len(sentences) < 2:
        return {
            "sentence_count": len(sentences),
            "sentences_preview": sentences[:3],
            "embeddings_shape": [0, 0],
            "similarity_matrix_shape": [0, 0],
            "similarity_matrix_preview": [],
            "lsa_top_scores": [],
            "pagerank_top_scores": [],
            "fused_top_scores": [],
            "selected_sentence_indices": [],
            "selected_sentences": sentences,
            "summary": text.strip(),
            "early_return": True,
        }

    embeddings = deterministic_embeddings(sentences, cfg.embedding)
    sim_matrix = compute_similarity_matrix(embeddings)
    lsa_scores = compute_lsa_scores(sentences, cfg.lsa, cfg.preprocessing.language)
    pagerank_scores = compute_pagerank_scores(sim_matrix, cfg.graph)
    fused_scores = fuse_scores(lsa_scores, pagerank_scores, cfg.fusion)

    selected_sentences = extract_top_sentences(
        sentences,
        fused_scores,
        sim_matrix,
        cfg.extraction,
        source_token_count=count_tokens(text),
    )
    selected_sentence_indices = extract_selected_indices(sentences, selected_sentences)

    return {
        "sentence_count": len(sentences),
        "sentences_preview": sentences[:3],
        "embeddings_shape": list(embeddings.shape),
        "similarity_matrix_shape": list(sim_matrix.shape),
        "similarity_matrix_preview": preview_matrix(sim_matrix),
        "lsa_top_scores": summarize_scores(lsa_scores),
        "pagerank_top_scores": summarize_scores(pagerank_scores),
        "fused_top_scores": summarize_scores(fused_scores),
        "selected_sentence_indices": [int(index) for index in selected_sentence_indices],
        "selected_sentences": selected_sentences,
        "summary": " ".join(selected_sentences),
        "early_return": False,
    }


def run_pytest_targets() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *INTEGRATION_TEST_TARGETS, "-rA"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "per_test": {
            "tests/integration/test_text_unit_data_preparation.py::test_loads_300_text_unit_contents":
                "PASSED tests/integration/test_text_unit_data_preparation.py::test_loads_300_text_unit_contents"
                in completed.stdout,
            "tests/integration/test_extractive_summarization_pipeline.py::test_summarizer_runs_on_postgres_loaded_texts":
                "PASSED tests/integration/test_extractive_summarization_pipeline.py::test_summarizer_runs_on_postgres_loaded_texts"
                in completed.stdout,
        },
    }


async def fetch_available_text_count() -> int:
    pg = create_postgres_storage()
    try:
        rows = await pg.fetch_all(
            """
            SELECT COUNT(*)
            FROM public.text_unit
            WHERE content IS NOT NULL
              AND btrim(content) <> ''
            """
        )
        return int(rows[0][0])
    finally:
        await pg.close()


def build_dataset_entries(texts: list[str], cfg: SummarizationConfig) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        sentences = split_sentences(text, cfg.preprocessing)
        entries.append(
            {
                "index": index,
                "char_length": len(text),
                "token_count": len(text.split()),
                "sentence_count": len(sentences),
                "text": text,
                "preview": text[:200],
            }
        )
    return entries


def summarize_dataset(texts: list[str], cfg: SummarizationConfig) -> dict[str, Any]:
    char_lengths = [len(text) for text in texts]
    token_lengths = [len(text.split()) for text in texts]
    sentence_counts = [len(split_sentences(text, cfg.preprocessing)) for text in texts]
    return {
        "count": len(texts),
        "non_empty_count": sum(1 for text in texts if text.strip()),
        "char_length": {
            "min": min(char_lengths),
            "median": statistics.median(char_lengths),
            "max": max(char_lengths),
        },
        "token_count": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "max": max(token_lengths),
        },
        "sentence_count": {
            "min": min(sentence_counts),
            "median": statistics.median(sentence_counts),
            "max": max(sentence_counts),
        },
    }


def build_report() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")

    cfg = SummarizationConfig(
        preprocessing=PreprocessingConfig(min_sentence_tokens=1),
        embedding=EmbeddingConfig(device="cpu", output_dim=3, batch_size=8),
        extraction=ExtractionConfig(token_budget_ratio=0.40),
        parallel=ParallelConfig(n_workers=1),
    )

    texts = load_text_unit_contents(limit=300)
    available_count = asyncio.run(fetch_available_text_count())
    pytest_result = run_pytest_targets()
    dataset_entries = build_dataset_entries(texts, cfg)

    summarization_pairs: list[dict[str, Any]] = []
    for index, text in enumerate(texts[:3]):
        details = summarize_text(text, cfg)
        summarization_pairs.append(
            {
                "input_index": index,
                "input_preview": text[:240],
                "output_summary": details["summary"],
                "intermediate": {
                    key: value
                    for key, value in details.items()
                    if key != "summary"
                },
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "postgres": {
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "db": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "table": "public.text_unit",
            "query_filter": "content IS NOT NULL AND btrim(content) <> ''",
            "requested_limit": 300,
            "available_non_empty_rows": available_count,
        },
        "tests": {
            "integration_pytest_run": pytest_result,
            "data_preparation_test": {
                "target": "tests/integration/test_text_unit_data_preparation.py::test_loads_300_text_unit_contents",
                "input": {
                    "source": "public.text_unit.content",
                    "limit": 300,
                },
                "intermediate_summary": summarize_dataset(texts, cfg),
                "output_assertions": [
                    "len(live_text_unit_contents) == 300",
                    "all entries are non-empty strings",
                ],
                "passed": pytest_result["per_test"][
                    "tests/integration/test_text_unit_data_preparation.py::test_loads_300_text_unit_contents"
                ],
            },
            "summarization_test": {
                "target": "tests/integration/test_extractive_summarization_pipeline.py::test_summarizer_runs_on_postgres_loaded_texts",
                "input_selection": "live_text_unit_contents[:3]",
                "config": asdict(cfg),
                "input_output_pairs": summarization_pairs,
                "output_assertions": [
                    "len(summaries) == 3",
                    "all generated summaries are non-empty strings",
                ],
                "passed": pytest_result["per_test"][
                    "tests/integration/test_extractive_summarization_pipeline.py::test_summarizer_runs_on_postgres_loaded_texts"
                ],
            },
        },
        "dataset_inputs": dataset_entries,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    data_prep = report["tests"]["data_preparation_test"]
    summarization = report["tests"]["summarization_test"]
    lines = [
        "# Live Text Unit Test Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Postgres: `{report['postgres']['host']}:{report['postgres']['port']}/{report['postgres']['db']}`",
        f"- Source table: `{report['postgres']['table']}`",
        f"- Non-empty rows available: `{report['postgres']['available_non_empty_rows']}`",
        f"- Requested limit: `{report['postgres']['requested_limit']}`",
        "",
        "## Pytest Result",
        "",
        f"- Command: `{report['tests']['integration_pytest_run']['command']}`",
        f"- Return code: `{report['tests']['integration_pytest_run']['returncode']}`",
        f"- Passed: `{report['tests']['integration_pytest_run']['passed']}`",
        "",
        "```text",
        report["tests"]["integration_pytest_run"]["stdout"].rstrip(),
        "```",
        "",
        "## Data Preparation Test",
        "",
        f"- Target: `{data_prep['target']}`",
        f"- Input source: `{data_prep['input']['source']}`",
        f"- Input limit: `{data_prep['input']['limit']}`",
        f"- Output assertions passed: `{data_prep['passed']}`",
        f"- Count: `{data_prep['intermediate_summary']['count']}`",
        f"- Non-empty count: `{data_prep['intermediate_summary']['non_empty_count']}`",
        f"- Char length summary: `{data_prep['intermediate_summary']['char_length']}`",
        f"- Token count summary: `{data_prep['intermediate_summary']['token_count']}`",
        f"- Sentence count summary: `{data_prep['intermediate_summary']['sentence_count']}`",
        "",
        "## Summarization Test",
        "",
        f"- Target: `{summarization['target']}`",
        f"- Input selection: `{summarization['input_selection']}`",
        f"- Output assertions passed: `{summarization['passed']}`",
        "",
    ]

    for pair in summarization["input_output_pairs"]:
        intermediate = pair["intermediate"]
        lines.extend(
            [
                f"### Input {pair['input_index']}",
                "",
                f"- Input preview: `{pair['input_preview']}`",
                f"- Output summary: `{pair['output_summary']}`",
                f"- Sentence count: `{intermediate['sentence_count']}`",
                f"- Embeddings shape: `{intermediate['embeddings_shape']}`",
                f"- Similarity matrix shape: `{intermediate['similarity_matrix_shape']}`",
                f"- Selected sentence indices: `{intermediate['selected_sentence_indices']}`",
                f"- Selected sentences: `{intermediate['selected_sentences']}`",
                f"- LSA top scores: `{intermediate['lsa_top_scores']}`",
                f"- PageRank top scores: `{intermediate['pagerank_top_scores']}`",
                f"- Fused top scores: `{intermediate['fused_top_scores']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Dataset Inputs",
            "",
            "- Full 300 input elements are stored in the sibling JSON report under `dataset_inputs`.",
            f"- JSON path: `{REPORT_PATH}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MARKDOWN_REPORT_PATH.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )
    print(f"saved {REPORT_PATH}")
    print(f"saved {MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
