from __future__ import annotations

from functools import lru_cache
import logging
from threading import Lock

import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)
_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _uses_jina_v5_text_family(model_name: str) -> bool:
    return model_name.startswith("jinaai/jina-embeddings-v5-text")


def _resolve_prompt_name(model_name: str) -> str | None:
    # Jina v5 text models use task-specific prompts. For our summarization pipeline
    # we encode comparable text units, not retrieval queries, so "document" is the
    # appropriate explicit prompt.
    if _uses_jina_v5_text_family(model_name):
        return "document"
    return None


@lru_cache(maxsize=1)
def _load_torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_cuda_available() -> bool:
    torch = _load_torch_module()
    if torch is None:
        return False
    return bool(torch.cuda.is_available())


def _is_mps_available() -> bool:
    torch = _load_torch_module()
    if torch is None:
        return False

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is None:
        return False
    return bool(mps_backend.is_built() and mps_backend.is_available())


def _detect_best_available_device() -> str:
    if _is_cuda_available():
        return "cuda"
    if _is_mps_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=None)
def resolve_embedding_device(requested_device: str | None) -> str:
    normalized_device = (requested_device or "auto").strip().lower()

    if normalized_device in {"", "auto", "gpu"}:
        resolved_device = _detect_best_available_device()
        logger.info(
            "Resolved embedding device '%s' from requested device '%s'.",
            resolved_device,
            requested_device or "auto",
        )
        return resolved_device

    if normalized_device.startswith("cuda"):
        if _is_cuda_available():
            return normalized_device
        fallback_device = "mps" if _is_mps_available() else "cpu"
        logger.warning(
            "Requested embedding device '%s' is not available; falling back to '%s'.",
            requested_device,
            fallback_device,
        )
        return fallback_device

    if normalized_device == "mps":
        if _is_mps_available():
            return normalized_device
        fallback_device = "cuda" if _is_cuda_available() else "cpu"
        logger.warning(
            "Requested embedding device '%s' is not available; falling back to '%s'.",
            requested_device,
            fallback_device,
        )
        return fallback_device

    if normalized_device == "cpu":
        return "cpu"

    return normalized_device


def _load_model(model_name: str, device: str):
    cache_key = (model_name, device)
    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

    from sentence_transformers import SentenceTransformer

    suppress_default_prompt_warning = _uses_jina_v5_text_family(model_name)
    sentence_transformers_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
    previous_level = sentence_transformers_logger.level

    try:
        if suppress_default_prompt_warning:
            sentence_transformers_logger.setLevel(logging.ERROR)

        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    finally:
        if suppress_default_prompt_warning:
            sentence_transformers_logger.setLevel(previous_level)

    if suppress_default_prompt_warning:
        model.default_prompt_name = None

    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model
        _MODEL_CACHE[cache_key] = model

    return model


def _load_model_for_config(cfg: EmbeddingConfig):
    resolved_device = resolve_embedding_device(cfg.device)
    return _load_model(cfg.model_name, resolved_device)


def get_embedding_tokenizer(cfg: EmbeddingConfig):
    model = _load_model_for_config(cfg)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    first_module_getter = getattr(model, "_first_module", None)
    if callable(first_module_getter):
        first_module = first_module_getter()
        tokenizer = getattr(first_module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

    raise RuntimeError(
        f"Embedding model '{cfg.model_name}' does not expose a tokenizer compatible with LSA tokenization."
    )


def embed_sentences(sentences: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    if not sentences:
        output_dim = max(1, cfg.output_dim)
        return np.zeros((0, output_dim), dtype=np.float32)

    model = _load_model_for_config(cfg)
    prompt_name = _resolve_prompt_name(cfg.model_name)
    encode_kwargs = {
        "sentences": sentences,
        "normalize_embeddings": cfg.normalize,
        "batch_size": cfg.batch_size,
        "truncate_dim": cfg.output_dim if cfg.output_dim < 768 else None,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    if prompt_name is not None:
        encode_kwargs["prompt_name"] = prompt_name

    try:
        embeddings = model.encode(**encode_kwargs)
    except TypeError:
        encode_kwargs.pop("prompt_name", None)
        truncate_dim = encode_kwargs.pop("truncate_dim", None)
        embeddings = model.encode(**encode_kwargs)
        if truncate_dim is not None and embeddings.shape[1] > truncate_dim:
            embeddings = embeddings[:, :truncate_dim]

    return np.asarray(embeddings, dtype=np.float32)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(embeddings @ embeddings.T, dtype=np.float32)
