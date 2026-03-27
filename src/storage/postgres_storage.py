from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import psycopg

from .base import StorageNameSpace

DEFAULT_POSTGRES_CONNECT_TIMEOUT = 3


def _build_postgres_config() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "postgres"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


@dataclass(slots=True)
class PostgresStorage(StorageNameSpace):
    postgres_config: Mapping[str, Any] | None = None
    connect_timeout: int = DEFAULT_POSTGRES_CONNECT_TIMEOUT
    _conn: psycopg.AsyncConnection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.postgres_config is None:
            self.postgres_config = _build_postgres_config()

    async def _ensure_connection(self) -> psycopg.AsyncConnection:
        if self._conn is not None and not self._conn.closed:
            return self._conn

        self._conn = await psycopg.AsyncConnection.connect(
            **dict(self.postgres_config or {}),
            connect_timeout=self.connect_timeout,
        )
        return self._conn

    async def fetch_all(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(query, params or {})
            return await cursor.fetchall()

    async def fetch_text_unit_contents(self, limit: int | None = 300) -> list[str]:
        query = """
            SELECT content
            FROM public.text_unit
            WHERE content IS NOT NULL
              AND btrim(content) <> ''
        """
        params: dict[str, Any] = {}
        if limit is not None:
            query = f"{query}\nLIMIT %(limit)s"
            params["limit"] = max(1, int(limit))

        rows = await self.fetch_all(query, params)
        return [str(row[0]) for row in rows]

    async def fetch_latest_gidf_version_id(self) -> int | None:
        rows = await self.fetch_all(
            """
            SELECT version_id
            FROM public.vocab_corpus_version
            ORDER BY created_at DESC, version_id DESC
            LIMIT 1
            """
        )
        if not rows:
            return None
        return int(rows[0][0])

    async def fetch_gidf_scores(self, version_id: int) -> list[tuple[str, float]]:
        rows = await self.fetch_all(
            """
            SELECT term, gidf_score
            FROM public.vocab_gidf
            WHERE version_id = %(version_id)s
            ORDER BY term
            """,
            {"version_id": int(version_id)},
        )
        return [(str(term), float(score)) for term, score in rows]

    async def store_gidf_version(
        self,
        *,
        total_docs: int,
        total_terms: int,
        min_df: int,
        max_df_ratio: float,
        language_code: str,
        description: str | None,
        gidf_rows: Sequence[tuple[str, float, int]],
    ) -> int:
        conn = await self._ensure_connection()
        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO public.vocab_corpus_version
                        (total_docs, total_terms, min_df, max_df_ratio, language_code, description)
                    VALUES
                        (%(total_docs)s, %(total_terms)s, %(min_df)s, %(max_df_ratio)s, %(language_code)s, %(description)s)
                    RETURNING version_id
                    """,
                    {
                        "total_docs": int(total_docs),
                        "total_terms": int(total_terms),
                        "min_df": int(min_df),
                        "max_df_ratio": float(max_df_ratio),
                        "language_code": language_code,
                        "description": description,
                    },
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("Failed to create a vocab_corpus_version row.")

                version_id = int(row[0])
                await cursor.executemany(
                    """
                    INSERT INTO public.vocab_gidf
                        (version_id, term, gidf_score, doc_frequency)
                    VALUES
                        (%(version_id)s, %(term)s, %(gidf_score)s, %(doc_frequency)s)
                    """,
                    [
                        {
                            "version_id": version_id,
                            "term": term,
                            "gidf_score": float(score),
                            "doc_frequency": int(doc_frequency),
                        }
                        for term, score, doc_frequency in gidf_rows
                    ],
                )
        return version_id

    async def index_done_callback(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
