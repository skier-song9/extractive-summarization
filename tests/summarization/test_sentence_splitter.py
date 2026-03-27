from __future__ import annotations

import pytest

from summarization.config import PreprocessingConfig
from summarization.sentence_splitter import split_sentences


def test_split_sentences_pysbd_handles_abbreviations() -> None:
    cfg = PreprocessingConfig(sentence_splitter="pysbd", language="en", min_sentence_tokens=1)
    text = "Dr. Smith reviewed Fig. 2 carefully. The result was statistically significant."

    sentences = split_sentences(text, cfg)

    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")


def test_split_sentences_spacy_branch_and_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = PreprocessingConfig(
        sentence_splitter="spacy",
        min_sentence_tokens=2,
        max_sentence_tokens=3,
    )

    monkeypatch.setattr(
        "summarization.sentence_splitter._split_spacy",
        lambda text, model_name: ["short", "this sentence is definitely too long"],
    )

    sentences = split_sentences("ignored", cfg)

    assert sentences == ["this sentence is"]


def test_split_sentences_invalid_engine_raises() -> None:
    cfg = PreprocessingConfig(sentence_splitter="pysbd")
    object.__setattr__(cfg, "sentence_splitter", "unknown")

    with pytest.raises(ValueError):
        split_sentences("Example sentence.", cfg)
