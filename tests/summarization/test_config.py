from __future__ import annotations

from pathlib import Path

import pytest

from summarization.config import SummarizationConfig


def test_from_yaml_ignores_deprecated_embedding_task(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
embedding:
  model_name: "demo-model"
  task: "text-matching"
  output_dim: 768
  device: "cpu"
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.embedding.model_name == "demo-model"
    assert cfg.embedding.output_dim == 768
    assert cfg.embedding.device == "cpu"


def test_from_yaml_defaults_embedding_device_to_auto(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
embedding:
  model_name: "demo-model"
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.embedding.device == "auto"


def test_from_yaml_loads_gidf_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
gidf:
  enabled: true
  version_id: 9
  min_df: 3
  max_df: 0.7
  sublinear_tf: false
  language_code: "en"
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.gidf.enabled is True
    assert cfg.gidf.version_id == 9
    assert cfg.gidf.min_df == 3
    assert cfg.gidf.max_df == 0.7
    assert cfg.gidf.sublinear_tf is False
    assert cfg.gidf.language_code == "en"


def test_from_yaml_supports_top_k_when_token_budget_ratio_is_null(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  token_budget_ratio: null
  top_k: 3
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.extraction.token_budget_ratio is None
    assert cfg.extraction.top_k == 3


def test_from_yaml_supports_top_k_when_token_budget_ratio_is_omitted(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  top_k: 3
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.extraction.token_budget_ratio is None
    assert cfg.extraction.top_k == 3


def test_from_yaml_supports_final_score_threshold_when_other_options_are_null(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  token_budget_ratio: null
  top_k: null
  final_score_threshold: 0.75
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.extraction.token_budget_ratio is None
    assert cfg.extraction.top_k is None
    assert cfg.extraction.final_score_threshold == 0.75


def test_from_yaml_supports_final_score_threshold_when_other_options_are_omitted(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  final_score_threshold: 0.75
""".strip(),
        encoding="utf-8",
    )

    cfg = SummarizationConfig.from_yaml(path)

    assert cfg.extraction.token_budget_ratio is None
    assert cfg.extraction.top_k is None
    assert cfg.extraction.final_score_threshold == 0.75


def test_from_yaml_rejects_extraction_when_all_selection_options_are_null(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  token_budget_ratio: null
  top_k: null
  final_score_threshold: null
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="At least one of token_budget_ratio, top_k, or final_score_threshold must be provided.",
    ):
        SummarizationConfig.from_yaml(path)


def test_from_yaml_rejects_non_positive_top_k(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  token_budget_ratio: null
  top_k: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer."):
        SummarizationConfig.from_yaml(path)


def test_from_yaml_rejects_invalid_final_score_threshold(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
extraction:
  final_score_threshold: 1.1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="final_score_threshold must be within the range \\[0.0, 1.0\\].",
    ):
        SummarizationConfig.from_yaml(path)
