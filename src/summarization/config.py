from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(slots=True)
class PreprocessingConfig:
    sentence_splitter: Literal["pysbd", "spacy", "wtp"] = "pysbd"
    language: str = "en"
    spacy_model: str = "en_core_web_sm"
    wtp_model: str = "wtp-canine-s-12l"
    min_sentence_tokens: int = 5
    max_sentence_tokens: int = 512


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "jinaai/jina-embeddings-v5-text-nano-text-matching"
    output_dim: int = 768
    normalize: bool = True
    batch_size: int = 32
    device: str = "auto"


@dataclass(slots=True)
class LSAConfig:
    n_components: int | None = None
    normalize_method: Literal["minmax", "rank"] = "minmax"


@dataclass(slots=True)
class GraphConfig:
    similarity_threshold: float = 0.20
    damping: float = 0.85
    max_iter: int = 100
    tol: float = 1e-6
    normalize_method: Literal["minmax", "rank"] = "minmax"


@dataclass(slots=True)
class FusionConfig:
    alpha: float = 0.5


@dataclass(slots=True)
class ExtractionConfig:
    token_budget_ratio: float | None = None
    top_k: int | None = None
    final_score_threshold: float | None = None
    preserve_order: bool = True
    redundancy_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.token_budget_ratio is not None and not 0.0 < self.token_budget_ratio <= 1.0:
            raise ValueError("token_budget_ratio must be within the range (0.0, 1.0].")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.final_score_threshold is not None and not 0.0 <= self.final_score_threshold <= 1.0:
            raise ValueError("final_score_threshold must be within the range [0.0, 1.0].")
        if (
            self.token_budget_ratio is None
            and self.top_k is None
            and self.final_score_threshold is None
        ):
            raise ValueError(
                "At least one of token_budget_ratio, top_k, or final_score_threshold must be provided."
            )


@dataclass(slots=True)
class ParallelConfig:
    n_workers: int = -1
    embedding_chunk_size: int = 10


@dataclass(slots=True)
class LoggingConfig:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_sentence_scores: bool = False


@dataclass(slots=True)
class GIDFConfig:
    enabled: bool = False
    version_id: int | None = None
    min_df: int = 2
    max_df: float = 0.85
    sublinear_tf: bool = True
    language_code: str = "ko"


@dataclass(slots=True)
class SummarizationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    lsa: LSAConfig = field(default_factory=LSAConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    extraction: ExtractionConfig = field(
        default_factory=lambda: ExtractionConfig(token_budget_ratio=0.30)
    )
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gidf: GIDFConfig = field(default_factory=GIDFConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SummarizationConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        embedding_raw = dict(raw.get("embedding", {}))
        embedding_raw.pop("task", None)
        extraction_raw = dict(raw.get("extraction", {}))
        if "reduction_ratio" in extraction_raw and "token_budget_ratio" not in extraction_raw:
            extraction_raw["token_budget_ratio"] = extraction_raw.pop("reduction_ratio")
        if "ratio" in extraction_raw:
            raise ValueError(
                "extraction.ratio is no longer supported. "
                "Use extraction.token_budget_ratio instead."
            )

        return cls(
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            embedding=EmbeddingConfig(**embedding_raw),
            lsa=LSAConfig(**raw.get("lsa", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            fusion=FusionConfig(**raw.get("fusion", {})),
            extraction=(
                ExtractionConfig(**extraction_raw)
                if extraction_raw
                else ExtractionConfig(token_budget_ratio=0.30)
            ),
            parallel=ParallelConfig(**raw.get("parallel", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            gidf=GIDFConfig(**raw.get("gidf", {})),
        )
