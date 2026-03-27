from __future__ import annotations

import asyncio

from summarization.text_unit_dataset import (
    close_async_loop,
    create_postgres_storage,
    fetch_text_unit_contents,
    load_project_env,
    load_text_unit_contents,
)


class _FakeStorage:
    def __init__(self, rows: list[str]) -> None:
        self.rows = rows
        self.closed = False
        self.limit: int | None = None

    async def fetch_text_unit_contents(self, limit: int = 300) -> list[str]:
        self.limit = limit
        return self.rows[:limit]

    async def close(self) -> None:
        self.closed = True


def test_load_project_env_reads_repo_dotenv(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    load_project_env()

    assert "POSTGRES_HOST" in __import__("os").environ


def test_create_postgres_storage_uses_canonical_env(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "example")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    storage = create_postgres_storage()

    assert dict(storage.postgres_config) == {
        "host": "db.local",
        "port": "5433",
        "dbname": "example",
        "user": "tester",
        "password": "secret",
    }


def test_load_text_unit_contents_returns_list_of_strings_with_external_storage() -> None:
    storage = _FakeStorage(["alpha", "beta", "gamma"])

    texts = load_text_unit_contents(limit=2, storage=storage)

    assert texts == ["alpha", "beta"]
    assert storage.limit == 2
    assert storage.closed is False
    close_async_loop()


def test_fetch_text_unit_contents_returns_list_of_strings_with_external_storage() -> None:
    storage = _FakeStorage(["alpha", "beta", "gamma"])

    texts = asyncio.run(fetch_text_unit_contents(limit=3, storage=storage))

    assert texts == ["alpha", "beta", "gamma"]
    assert storage.closed is False


def test_load_text_unit_contents_closes_owned_storage(monkeypatch) -> None:
    storage = _FakeStorage(["alpha", "beta", "gamma"])
    monkeypatch.setattr("summarization.text_unit_dataset.create_postgres_storage", lambda: storage)

    texts = load_text_unit_contents(limit=2)

    assert texts == ["alpha", "beta"]
    assert storage.closed is True


def test_fetch_text_unit_contents_closes_owned_storage(monkeypatch) -> None:
    storage = _FakeStorage(["alpha", "beta", "gamma"])
    monkeypatch.setattr("summarization.text_unit_dataset.create_postgres_storage", lambda: storage)

    texts = asyncio.run(fetch_text_unit_contents(limit=3))

    assert texts == ["alpha", "beta", "gamma"]
    assert storage.closed is True
