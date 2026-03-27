from __future__ import annotations

from functools import lru_cache
import re

from .config import EmbeddingConfig
from .embedder import get_embedding_tokenizer

_WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['-])*", re.UNICODE)
_TOKEN_PREFIXES = ("##", "▁", "Ġ")
_DEFAULT_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "el": "el_core_news_sm",
}
_LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "ko": "ko",
    "korean": "ko",
    "ja": "ja",
    "japanese": "ja",
    "zh": "zh",
    "chinese": "zh",
    "el": "el",
    "greek": "el",
}


def _normalize_language_code(language: str) -> str:
    normalized = (language or "").strip().lower()
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def _has_term_characters(text: str) -> bool:
    return any(char.isalnum() for char in text)


class _RegexTermTokenizer:
    def to_words(self, sentence: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in _WORD_RE.finditer(sentence.lower()))


class _EmbeddingModelTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self._special_tokens = frozenset(str(token) for token in getattr(tokenizer, "all_special_tokens", ()))

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.strip()
        while normalized.startswith(_TOKEN_PREFIXES):
            for prefix in _TOKEN_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized.removeprefix(prefix)
                    break
        return normalized.lower()

    def to_words(self, sentence: str) -> tuple[str, ...]:
        tokenize = getattr(self._tokenizer, "tokenize", None)
        if not callable(tokenize):
            raise RuntimeError("Embedding tokenizer must provide a callable tokenize() method.")

        tokens: list[str] = []
        for raw_token in tokenize(sentence):
            text = str(raw_token)
            if text in self._special_tokens:
                continue

            normalized = self._normalize_token(text)
            if not normalized or not _has_term_characters(normalized):
                continue
            tokens.append(normalized)

        return tuple(tokens)


@lru_cache(maxsize=4)
def _load_spacy_tokenizer(language_code: str, model_name: str):
    import spacy

    try:
        return spacy.load(model_name, disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
    except OSError:
        return spacy.blank(language_code)


class _SpacyTermTokenizer:
    def __init__(self, language_code: str, model_name: str) -> None:
        self._language_code = language_code
        self._model_name = model_name

    def to_words(self, sentence: str) -> tuple[str, ...]:
        nlp = _load_spacy_tokenizer(self._language_code, self._model_name)
        return tuple(
            token.text.lower()
            for token in nlp.make_doc(sentence)
            if not token.is_space and not token.is_punct and _has_term_characters(token.text)
        )


@lru_cache(maxsize=1)
def _load_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ImportError as exc:
        raise ImportError(
            "kiwipiepy is required for Korean term tokenization. Install kiwipiepy first."
        ) from exc

    return Kiwi()


class _KiwiTermTokenizer:
    def to_words(self, sentence: str) -> tuple[str, ...]:
        kiwi = _load_kiwi()
        return tuple(
            token.form
            for token in kiwi.tokenize(sentence)
            if token.form and _has_term_characters(token.form)
        )


@lru_cache(maxsize=1)
def _load_fugashi_tagger():
    try:
        from fugashi import Tagger
    except ImportError as exc:
        raise ImportError(
            "fugashi and unidic-lite are required for Japanese term tokenization. "
            "Install fugashi and unidic-lite first."
        ) from exc

    return Tagger()


class _FugashiTermTokenizer:
    def to_words(self, sentence: str) -> tuple[str, ...]:
        tagger = _load_fugashi_tagger()
        return tuple(
            token.surface
            for token in tagger(sentence)
            if token.surface and _has_term_characters(token.surface)
        )


@lru_cache(maxsize=1)
def _load_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise ImportError(
            "jieba is required for Chinese term tokenization. Install jieba first."
        ) from exc

    return jieba


class _JiebaTermTokenizer:
    def to_words(self, sentence: str) -> tuple[str, ...]:
        jieba = _load_jieba()
        return tuple(
            token.lower()
            for token in jieba.cut(sentence, cut_all=False)
            if token and _has_term_characters(token)
        )


def get_term_tokenizer(
    language: str,
    *,
    spacy_model: str = "en_core_web_sm",
    embedding_cfg: EmbeddingConfig | None = None,
):
    language_code = _normalize_language_code(language)

    if language_code == "en":
        return _SpacyTermTokenizer("en", spacy_model)
    if language_code == "ko":
        return _KiwiTermTokenizer()
    if language_code == "ja":
        return _FugashiTermTokenizer()
    if language_code == "zh":
        return _JiebaTermTokenizer()
    if language_code == "el":
        return _SpacyTermTokenizer("el", _DEFAULT_SPACY_MODELS["el"])
    if embedding_cfg is not None:
        return _EmbeddingModelTokenizer(get_embedding_tokenizer(embedding_cfg))
    return _RegexTermTokenizer()
