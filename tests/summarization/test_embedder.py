from __future__ import annotations

import numpy as np
import pytest

from summarization.config import EmbeddingConfig
from summarization.embedder import compute_similarity_matrix, embed_sentences, get_embedding_tokenizer, resolve_embedding_device


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, **kwargs):
        self.calls.append(kwargs)
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_embed_sentences_passes_expected_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeModel()
    captured: dict[str, str] = {}

    def _fake_load_model(model_name: str, device: str) -> _FakeModel:
        captured["model_name"] = model_name
        captured["device"] = device
        return fake_model

    monkeypatch.setattr("summarization.embedder._load_model", _fake_load_model)
    monkeypatch.setattr("summarization.embedder.resolve_embedding_device", lambda device: "cpu")
    cfg = EmbeddingConfig(model_name="demo-model", device="cpu", batch_size=8, output_dim=256)

    embeddings = embed_sentences(["alpha", "beta"], cfg)

    assert embeddings.shape == (2, 2)
    assert captured["model_name"] == "demo-model"
    assert captured["device"] == "cpu"
    assert fake_model.calls[0]["batch_size"] == 8
    assert fake_model.calls[0]["truncate_dim"] == 256
    assert "task" not in fake_model.calls[0]
    assert "prompt_name" not in fake_model.calls[0]


def test_embed_sentences_uses_document_prompt_for_jina_v5_text_models(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr("summarization.embedder._load_model", lambda model_name, device: fake_model)
    monkeypatch.setattr("summarization.embedder.resolve_embedding_device", lambda device: "mps")
    cfg = EmbeddingConfig(
        model_name="jinaai/jina-embeddings-v5-text-nano-text-matching",
        device="auto",
        batch_size=4,
        output_dim=768,
    )

    embeddings = embed_sentences(["alpha", "beta"], cfg)

    assert embeddings.shape == (2, 2)
    assert fake_model.calls[0]["prompt_name"] == "document"
    assert fake_model.calls[0]["batch_size"] == 4
    assert fake_model.calls[0]["truncate_dim"] is None


def test_get_embedding_tokenizer_returns_model_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tokenizer = object()

    class _ModelWithTokenizer:
        tokenizer = fake_tokenizer

    monkeypatch.setattr("summarization.embedder._load_model_for_config", lambda cfg: _ModelWithTokenizer())

    assert get_embedding_tokenizer(EmbeddingConfig()) is fake_tokenizer


def test_resolve_embedding_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_embedding_device.cache_clear()
    monkeypatch.setattr("summarization.embedder._is_cuda_available", lambda: True)
    monkeypatch.setattr("summarization.embedder._is_mps_available", lambda: True)

    assert resolve_embedding_device("auto") == "cuda"


def test_resolve_embedding_device_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_embedding_device.cache_clear()
    monkeypatch.setattr("summarization.embedder._is_cuda_available", lambda: False)
    monkeypatch.setattr("summarization.embedder._is_mps_available", lambda: True)

    assert resolve_embedding_device("auto") == "mps"


def test_resolve_embedding_device_falls_back_from_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_embedding_device.cache_clear()
    monkeypatch.setattr("summarization.embedder._is_cuda_available", lambda: False)
    monkeypatch.setattr("summarization.embedder._is_mps_available", lambda: True)

    assert resolve_embedding_device("cuda") == "mps"


def test_resolve_embedding_device_falls_back_from_unavailable_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_embedding_device.cache_clear()
    monkeypatch.setattr("summarization.embedder._is_cuda_available", lambda: False)
    monkeypatch.setattr("summarization.embedder._is_mps_available", lambda: False)

    assert resolve_embedding_device("mps") == "cpu"


def test_compute_similarity_matrix_returns_cosine_like_dot_products() -> None:
    embeddings = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)

    matrix = compute_similarity_matrix(embeddings)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.6)
    assert np.allclose(matrix, matrix.T)
