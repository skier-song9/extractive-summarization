from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from storage.postgres_storage import PostgresStorage

from .config import GIDFConfig
from .term_tokenizer import get_term_tokenizer
from .text_unit_dataset import close_async_loop, create_postgres_storage, resolve_awaitable

_GIDF_DISABLED: dict[str, float] = {}


def _build_vectorizer(cfg: GIDFConfig) -> TfidfVectorizer:
    tokenizer = get_term_tokenizer(cfg.language_code)
    return TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        tokenizer=tokenizer.to_words,
        token_pattern=None,
        lowercase=False,
    )


def _compute_gidf(
    documents: list[str],
    cfg: GIDFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not documents:
        raise ValueError("text_unit에 유효한 content가 없습니다.")

    vectorizer = _build_vectorizer(cfg)
    vectorizer.fit(documents)

    terms = vectorizer.get_feature_names_out()
    gidf_scores = vectorizer.idf_
    tfidf_matrix = vectorizer.transform(documents)
    doc_frequency = np.asarray((tfidf_matrix > 0).sum(axis=0)).ravel().astype(int)
    return terms, gidf_scores, doc_frequency


async def build_and_store_gidf_async(
    cfg: GIDFConfig,
    *,
    description: str | None = None,
    storage: PostgresStorage | None = None,
) -> int:
    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        documents = await current_storage.fetch_text_unit_contents(limit=None)
        terms, gidf_scores, doc_frequency = _compute_gidf(documents, cfg)
        gidf_rows: Sequence[tuple[str, float, int]] = [
            (str(term), float(score), int(doc_freq))
            for term, score, doc_freq in zip(terms, gidf_scores, doc_frequency, strict=True)
        ]
        version_id = await current_storage.store_gidf_version(
            total_docs=len(documents),
            total_terms=len(terms),
            min_df=cfg.min_df,
            max_df_ratio=cfg.max_df,
            language_code=cfg.language_code,
            description=description,
            gidf_rows=gidf_rows,
        )
        await current_storage.index_done_callback()
        return version_id
    finally:
        if owns_storage:
            await current_storage.close()


def build_and_store_gidf(
    cfg: GIDFConfig,
    *,
    description: str | None = None,
    storage: PostgresStorage | None = None,
) -> int:
    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        return resolve_awaitable(
            build_and_store_gidf_async(
                cfg,
                description=description,
                storage=current_storage,
            )
        )
    finally:
        if owns_storage:
            resolve_awaitable(current_storage.close())
            close_async_loop()


async def load_gidf_async(
    cfg: GIDFConfig,
    *,
    storage: PostgresStorage | None = None,
) -> dict[str, float]:
    if not cfg.enabled:
        return _GIDF_DISABLED

    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        version_id = cfg.version_id
        if version_id is None:
            version_id = await current_storage.fetch_latest_gidf_version_id()
        if version_id is None:
            raise RuntimeError(
                "빌드된 GIDF 버전이 없습니다. build_and_store_gidf()를 먼저 실행하세요."
            )

        rows = await current_storage.fetch_gidf_scores(version_id)
        if not rows:
            raise RuntimeError(f"GIDF 항목을 찾을 수 없습니다. version_id={version_id}")
        return {term: score for term, score in rows}
    finally:
        if owns_storage:
            await current_storage.close()


def load_gidf(
    cfg: GIDFConfig,
    *,
    storage: PostgresStorage | None = None,
) -> dict[str, float]:
    if not cfg.enabled:
        return _GIDF_DISABLED

    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        return resolve_awaitable(load_gidf_async(cfg, storage=current_storage))
    finally:
        if owns_storage:
            resolve_awaitable(current_storage.close())
            close_async_loop()
