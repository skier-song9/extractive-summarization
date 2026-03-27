from __future__ import annotations

import pytest
import psycopg

from summarization.text_unit_dataset import load_text_unit_contents


@pytest.fixture(scope="session")
def live_text_unit_contents() -> list[str]:
    try:
        return load_text_unit_contents(limit=300)
    except psycopg.Error as exc:
        pytest.skip(f"PostgreSQL integration is unavailable: {exc}")
