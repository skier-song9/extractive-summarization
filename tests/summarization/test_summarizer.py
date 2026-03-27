from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from summarization.config import ExtractionConfig, ParallelConfig, PreprocessingConfig, SummarizationConfig
from summarization.summarizer import HybridExtractiveSummarizer


def test_summarize_one_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig(
        preprocessing=PreprocessingConfig(min_sentence_tokens=1),
        extraction=ExtractionConfig(token_budget_ratio=0.67),
    )
    summarizer = HybridExtractiveSummarizer(cfg)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "summarization.summarizer.split_sentences",
        lambda text, preprocessing: ["Sentence one.", "Sentence two.", "Sentence three."],
    )
    monkeypatch.setattr(
        "summarization.summarizer.embed_sentences",
        lambda sentences, embedding_cfg: np.eye(len(sentences), dtype=np.float32),
    )
    monkeypatch.setattr(
        "summarization.summarizer.compute_similarity_matrix",
        lambda embeddings: embeddings,
    )
    monkeypatch.setattr(
        "summarization.summarizer.compute_lsa_scores",
        lambda sentences, lsa_cfg, language, spacy_model, embedding_cfg: (
            captured.update({"embedding_cfg": embedding_cfg, "language": language, "spacy_model": spacy_model})
            or {0: 0.2, 1: 0.8, 2: 0.4}
        ),
    )
    monkeypatch.setattr(
        "summarization.summarizer.apply_gidf_boost",
        lambda lsa_scores, sentences, gidf, language, spacy_model, embedding_cfg: (
            captured.update(
                {
                    "gidf": gidf,
                    "gidf_language": language,
                    "gidf_spacy_model": spacy_model,
                    "gidf_embedding_cfg": embedding_cfg,
                }
            )
            or {0: 0.1, 1: 0.9, 2: 0.3}
        ),
    )
    monkeypatch.setattr(
        "summarization.summarizer.compute_pagerank_scores",
        lambda sim_matrix, graph_cfg: {0: 0.1, 1: 0.5, 2: 0.9},
    )

    summary = summarizer.summarize_one("Sentence one. Sentence two. Sentence three.")

    assert summary == "Sentence two. Sentence three."
    assert captured["embedding_cfg"] is cfg.embedding
    assert captured["language"] == cfg.preprocessing.language
    assert captured["spacy_model"] == cfg.preprocessing.spacy_model
    assert captured["gidf"] == {}
    assert captured["gidf_language"] == cfg.preprocessing.language
    assert captured["gidf_spacy_model"] == cfg.preprocessing.spacy_model
    assert captured["gidf_embedding_cfg"] is cfg.embedding


def test_summarizer_loads_gidf_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig(
        preprocessing=PreprocessingConfig(min_sentence_tokens=1),
        extraction=ExtractionConfig(token_budget_ratio=0.67),
    )
    cfg.gidf.enabled = True

    monkeypatch.setattr(
        "summarization.summarizer.load_gidf",
        lambda gidf_cfg, storage=None: {"covenant": 2.0},
    )

    summarizer = HybridExtractiveSummarizer(cfg)

    assert summarizer._gidf == {"covenant": 2.0}


def test_summarize_batch_accepts_list_and_runs_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig(parallel=ParallelConfig(n_workers=2))
    summarizer = HybridExtractiveSummarizer(cfg)
    thread_ids: list[int] = []
    lock = threading.Lock()

    def fake_summarize_one(text: str) -> str:
        time.sleep(0.05)
        with lock:
            thread_ids.append(threading.get_ident())
        return f"summary:{text}"

    monkeypatch.setattr("summarization.summarizer.resolve_embedding_device", lambda device: "cpu")
    monkeypatch.setattr(summarizer, "summarize_one", fake_summarize_one)

    results = summarizer.summarize_batch(["doc-1", "doc-2", "doc-3", "doc-4"])

    assert results == [
        "summary:doc-1",
        "summary:doc-2",
        "summary:doc-3",
        "summary:doc-4",
    ]
    assert len(set(thread_ids)) >= 2


def test_summarize_batch_uses_single_worker_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig(parallel=ParallelConfig(n_workers=4))
    summarizer = HybridExtractiveSummarizer(cfg)
    thread_ids: list[int] = []

    def fake_summarize_one(text: str) -> str:
        thread_ids.append(threading.get_ident())
        return f"summary:{text}"

    monkeypatch.setattr("summarization.summarizer.resolve_embedding_device", lambda device: "mps")
    monkeypatch.setattr(summarizer, "summarize_one", fake_summarize_one)

    results = summarizer.summarize_batch(["doc-1", "doc-2", "doc-3"])

    assert results == [
        "summary:doc-1",
        "summary:doc-2",
        "summary:doc-3",
    ]
    assert len(set(thread_ids)) == 1


def test_summarize_batch_keeps_parallelism_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummarizationConfig(parallel=ParallelConfig(n_workers=2))
    summarizer = HybridExtractiveSummarizer(cfg)
    thread_ids: list[int] = []
    lock = threading.Lock()

    def fake_summarize_one(text: str) -> str:
        time.sleep(0.05)
        with lock:
            thread_ids.append(threading.get_ident())
        return f"summary:{text}"

    monkeypatch.setattr("summarization.summarizer.resolve_embedding_device", lambda device: "cuda")
    monkeypatch.setattr(summarizer, "summarize_one", fake_summarize_one)

    results = summarizer.summarize_batch(["doc-1", "doc-2", "doc-3", "doc-4"])

    assert results == [
        "summary:doc-1",
        "summary:doc-2",
        "summary:doc-3",
        "summary:doc-4",
    ]
    assert len(set(thread_ids)) >= 2
