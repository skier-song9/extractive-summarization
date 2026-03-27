from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

from storage.postgres_storage import PostgresStorage

T = TypeVar("T")

_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")


def create_postgres_storage() -> PostgresStorage:
    load_project_env()
    return PostgresStorage(namespace="default", global_config={})


def resolve_awaitable(value: T) -> T:
    global _ASYNC_LOOP
    if inspect.isawaitable(value):
        if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
            _ASYNC_LOOP = asyncio.new_event_loop()
            asyncio.set_event_loop(_ASYNC_LOOP)
        return _ASYNC_LOOP.run_until_complete(value)
    return value


def close_async_loop() -> None:
    global _ASYNC_LOOP
    if _ASYNC_LOOP is not None and not _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP.close()
    _ASYNC_LOOP = None


async def fetch_text_unit_contents(limit: int = 300, storage: PostgresStorage | None = None) -> list[str]:
    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        return await current_storage.fetch_text_unit_contents(limit=limit)
    finally:
        if owns_storage:
            await current_storage.close()


def load_text_unit_contents(limit: int = 300, storage: PostgresStorage | None = None) -> list[str]:
    owns_storage = storage is None
    current_storage = storage or create_postgres_storage()
    try:
        return resolve_awaitable(current_storage.fetch_text_unit_contents(limit=limit))
    finally:
        if owns_storage:
            resolve_awaitable(current_storage.close())
            close_async_loop()
