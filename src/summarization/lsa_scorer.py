from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.linalg import svd
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

from .config import EmbeddingConfig, LSAConfig
from .term_tokenizer import get_term_tokenizer
from .utils import minmax_normalize, normalize_language_name, rank_normalize


def _build_document(
    sentences: list[str],
    language: str,
    spacy_model: str,
    embedding_cfg: EmbeddingConfig | None,
) -> ObjectDocumentModel:
    sentence_tokenizer = get_term_tokenizer(language, spacy_model=spacy_model, embedding_cfg=embedding_cfg)
    sentence_objects = [Sentence(text, sentence_tokenizer) for text in sentences]
    return ObjectDocumentModel([Paragraph(sentence_objects)])


def _compute_ranks(document: ObjectDocumentModel, summarizer: LsaSummarizer, n_components: int | None) -> list[float]:
    dictionary = summarizer._create_dictionary(document)
    if not dictionary:
        return [0.0 for _ in document.sentences]

    matrix = summarizer._create_matrix(document, dictionary)
    matrix = summarizer._compute_term_frequency(matrix)
    _, sigma, v_matrix = svd(matrix, full_matrices=False)

    if sigma.size == 0:
        return [0.0 for _ in document.sentences]

    dimensions = n_components if n_components is not None else max(
        LsaSummarizer.MIN_DIMENSIONS,
        int(len(sigma) * LsaSummarizer.REDUCTION_RATIO),
    )
    dimensions = max(1, min(int(dimensions), len(sigma)))
    powered_sigma = [value**2 if i < dimensions else 0.0 for i, value in enumerate(sigma)]

    ranks: list[float] = []
    for column_vector in v_matrix.T:
        rank = sum(weight * vector_value**2 for weight, vector_value in zip(powered_sigma, column_vector))
        ranks.append(float(np.sqrt(rank)))
    return ranks


def compute_lsa_scores(
    sentences: list[str],
    cfg: LSAConfig,
    language: str = "english",
    spacy_model: str = "en_core_web_sm",
    embedding_cfg: EmbeddingConfig | None = None,
) -> dict[int, float]:
    if not sentences:
        return {}

    normalized_language = normalize_language_name(language)
    summarizer = LsaSummarizer(Stemmer(normalized_language))
    try:
        summarizer.stop_words = get_stop_words(normalized_language)
    except LookupError:
        summarizer.stop_words = ()

    document = _build_document(sentences, language, spacy_model, embedding_cfg)
    raw_scores = {index: score for index, score in enumerate(_compute_ranks(document, summarizer, cfg.n_components))}

    if cfg.normalize_method == "rank":
        return rank_normalize(raw_scores)
    return minmax_normalize(raw_scores)


def apply_gidf_boost(
    lsa_scores: dict[int, float],
    sentences: list[str],
    gidf: dict[str, float],
    *,
    language: str = "en",
    spacy_model: str = "en_core_web_sm",
    embedding_cfg: EmbeddingConfig | None = None,
) -> dict[int, float]:
    if not lsa_scores:
        return {}

    tokenizer = get_term_tokenizer(language, spacy_model=spacy_model, embedding_cfg=embedding_cfg)
    boosted: dict[int, float] = {}

    for index, sentence in enumerate(sentences):
        tokens = tokenizer.to_words(sentence)
        original_score = float(lsa_scores.get(index, 0.0))
        if not tokens:
            boosted[index] = original_score
            continue

        term_frequencies = Counter(tokens)
        numerator = sum(count * gidf.get(token, 1.0) for token, count in term_frequencies.items())
        boosted[index] = original_score * (numerator / len(tokens))

    return minmax_normalize(boosted)
