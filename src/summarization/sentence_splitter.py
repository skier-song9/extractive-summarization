from __future__ import annotations

from functools import lru_cache

from .config import PreprocessingConfig


def _split_pysbd(text: str, language: str) -> list[str]:
    import pysbd

    segmenter = pysbd.Segmenter(language=language, clean=True)
    return segmenter.segment(text)


@lru_cache(maxsize=4)
def _load_spacy(model_name: str):
    import spacy

    nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _split_spacy(text: str, model_name: str) -> list[str]:
    nlp = _load_spacy(model_name)
    return [sentence.text.strip() for sentence in nlp(text).sents if sentence.text.strip()]


@lru_cache(maxsize=2)
def _load_wtp(model_name: str):
    try:
        from wtpsplit import WtP
    except ImportError as exc:
        raise ImportError(
            "wtpsplit is required when sentence_splitter='wtp'. Install wtpsplit first."
        ) from exc

    return WtP(model_name)


def _split_wtp(text: str, model_name: str, language: str) -> list[str]:
    splitter = _load_wtp(model_name)
    return [sentence.strip() for sentence in splitter.split(text, lang_code=language) if sentence.strip()]


def _filter(sentences: list[str], min_tok: int, max_tok: int) -> list[str]:
    filtered: list[str] = []
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) < min_tok:
            continue
        if len(tokens) > max_tok:
            sentence = " ".join(tokens[:max_tok])
        filtered.append(sentence)
    return filtered


def split_sentences(text: str, cfg: PreprocessingConfig) -> list[str]:
    match cfg.sentence_splitter:
        case "pysbd":
            raw_sentences = _split_pysbd(text, cfg.language)
        case "spacy":
            raw_sentences = _split_spacy(text, cfg.spacy_model)
        case "wtp":
            raw_sentences = _split_wtp(text, cfg.wtp_model, cfg.language)
        case _:
            raise ValueError(
                f"Unknown sentence_splitter: {cfg.sentence_splitter!r}. "
                "Choose from 'pysbd', 'spacy', or 'wtp'."
            )

    return _filter(raw_sentences, cfg.min_sentence_tokens, cfg.max_sentence_tokens)
