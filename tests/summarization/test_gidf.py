from __future__ import annotations

import numpy as np
import pytest

from summarization.config import GIDFConfig
from summarization.lsa_scorer import apply_gidf_boost
from summarization.vocab_builder import _compute_gidf, build_and_store_gidf, load_gidf


class _FakeGIDFStorage:
    def __init__(
        self,
        *,
        documents: list[str] | None = None,
        latest_version_id: int | None = None,
        gidf_scores: dict[int, list[tuple[str, float]]] | None = None,
        version_id_to_return: int = 7,
    ) -> None:
        self.documents = documents or []
        self.latest_version_id = latest_version_id
        self.gidf_scores = gidf_scores or {}
        self.version_id_to_return = version_id_to_return
        self.closed = False
        self.index_done = False
        self.requested_limit: int | None = -1
        self.requested_version_id: int | None = None
        self.stored_payload: dict[str, object] | None = None

    async def fetch_text_unit_contents(self, limit: int | None = 300) -> list[str]:
        self.requested_limit = limit
        return self.documents

    async def store_gidf_version(self, **kwargs) -> int:
        self.stored_payload = kwargs
        return self.version_id_to_return

    async def index_done_callback(self) -> None:
        self.index_done = True

    async def fetch_latest_gidf_version_id(self) -> int | None:
        return self.latest_version_id

    async def fetch_gidf_scores(self, version_id: int) -> list[tuple[str, float]]:
        self.requested_version_id = version_id
        return self.gidf_scores.get(version_id, [])

    async def close(self) -> None:
        self.closed = True


def test_compute_gidf_tracks_document_frequency() -> None:
    cfg = GIDFConfig(min_df=1, max_df=1.0, sublinear_tf=False, language_code="unknown")

    terms, gidf_scores, doc_frequency = _compute_gidf(
        ["alpha beta", "alpha gamma"],
        cfg,
    )

    assert isinstance(terms, np.ndarray)
    assert isinstance(gidf_scores, np.ndarray)
    assert isinstance(doc_frequency, np.ndarray)
    assert dict(zip(terms.tolist(), doc_frequency.tolist(), strict=True)) == {
        "alpha": 2,
        "beta": 1,
        "gamma": 1,
    }


def test_gidf_disabled_returns_identity_mapping() -> None:
    lsa_scores = {0: 0.0, 1: 1.0}
    sentences = ["covenant hope", "weather report"]

    boosted = apply_gidf_boost(lsa_scores, sentences, gidf={}, language="unknown")

    assert boosted == lsa_scores


def test_gidf_boosts_domain_terms() -> None:
    lsa_scores = {0: 0.5, 1: 0.5}
    sentences = ["covenant grace doctrine", "weather traffic update"]
    gidf = {
        "covenant": 3.0,
        "grace": 2.5,
        "doctrine": 2.8,
        "weather": 1.0,
        "traffic": 1.0,
        "update": 1.0,
    }

    boosted = apply_gidf_boost(lsa_scores, sentences, gidf, language="unknown")

    assert boosted[0] > boosted[1]


def test_build_and_store_gidf_uses_storage_and_commits() -> None:
    storage = _FakeGIDFStorage(
        documents=["alpha beta", "alpha gamma"],
        version_id_to_return=11,
    )

    version_id = build_and_store_gidf(
        GIDFConfig(min_df=1, max_df=1.0, sublinear_tf=False, language_code="unknown"),
        description="test build",
        storage=storage,
    )

    assert version_id == 11
    assert storage.requested_limit is None
    assert storage.index_done is True
    assert storage.closed is False
    assert storage.stored_payload is not None
    assert storage.stored_payload["total_docs"] == 2
    assert storage.stored_payload["total_terms"] == 3


def test_load_gidf_auto_selects_latest_version() -> None:
    storage = _FakeGIDFStorage(
        latest_version_id=5,
        gidf_scores={5: [("alpha", 1.7), ("beta", 2.1)]},
    )

    gidf = load_gidf(GIDFConfig(enabled=True), storage=storage)

    assert gidf == {"alpha": 1.7, "beta": 2.1}
    assert storage.requested_version_id == 5
    assert storage.closed is False


def test_load_gidf_requires_existing_version() -> None:
    storage = _FakeGIDFStorage(latest_version_id=None)

    with pytest.raises(RuntimeError, match="빌드된 GIDF 버전이 없습니다"):
        load_gidf(GIDFConfig(enabled=True), storage=storage)
