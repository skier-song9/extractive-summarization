from __future__ import annotations

import pytest


@pytest.mark.integration
def test_loads_300_text_unit_contents(live_text_unit_contents: list[str]) -> None:
    assert len(live_text_unit_contents) == 300
    assert all(isinstance(text, str) and text.strip() for text in live_text_unit_contents)
