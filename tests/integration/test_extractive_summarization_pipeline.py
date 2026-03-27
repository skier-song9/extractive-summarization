from __future__ import annotations

import numpy as np
import pytest

from summarization.config import EmbeddingConfig, ExtractionConfig, ParallelConfig, PreprocessingConfig, SummarizationConfig
from summarization.summarizer import HybridExtractiveSummarizer


def _deterministic_embeddings(sentences: list[str], _: EmbeddingConfig) -> np.ndarray:
    features: list[list[float]] = []
    for index, sentence in enumerate(sentences):
        token_count = len(sentence.split())
        char_count = len(sentence)
        features.append([float(index + 1), float(token_count), float(char_count)])
    return np.asarray(features, dtype=np.float32)


@pytest.mark.integration
def test_summarizer_runs_on_postgres_loaded_texts(
    live_text_unit_contents: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SummarizationConfig(
        preprocessing=PreprocessingConfig(min_sentence_tokens=1),
        embedding=EmbeddingConfig(device="cpu", output_dim=3, batch_size=8),
        extraction=ExtractionConfig(token_budget_ratio=0.40),
        parallel=ParallelConfig(n_workers=1),
    )
    summarizer = HybridExtractiveSummarizer(cfg)

    monkeypatch.setattr("summarization.summarizer.embed_sentences", _deterministic_embeddings)

    summaries = summarizer.summarize_batch(live_text_unit_contents[:3])

    assert len(summaries) == 3
    assert all(isinstance(summary, str) and summary.strip() for summary in summaries)
