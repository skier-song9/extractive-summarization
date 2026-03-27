from __future__ import annotations

from summarization.config import EmbeddingConfig
from summarization.term_tokenizer import get_term_tokenizer


def test_get_term_tokenizer_uses_spacy_for_english(monkeypatch) -> None:
    class _Token:
        def __init__(self, text: str, *, is_space: bool = False, is_punct: bool = False) -> None:
            self.text = text
            self.is_space = is_space
            self.is_punct = is_punct

    class _FakeNlp:
        def make_doc(self, sentence: str):
            return [
                _Token("Alpha"),
                _Token(",", is_punct=True),
                _Token("Beta"),
            ]

    monkeypatch.setattr("summarization.term_tokenizer._load_spacy_tokenizer", lambda language_code, model_name: _FakeNlp())

    tokenizer = get_term_tokenizer("en", spacy_model="en_core_web_sm")

    assert tokenizer.to_words("ignored") == ("alpha", "beta")


def test_get_term_tokenizer_uses_kiwi_for_korean(monkeypatch) -> None:
    class _Token:
        def __init__(self, form: str) -> None:
            self.form = form

    class _FakeKiwi:
        def tokenize(self, sentence: str):
            return [_Token("구원"), _Token("으로")]

    monkeypatch.setattr("summarization.term_tokenizer._load_kiwi", lambda: _FakeKiwi())

    tokenizer = get_term_tokenizer("ko")

    assert tokenizer.to_words("ignored") == ("구원", "으로")


def test_get_term_tokenizer_uses_fugashi_for_japanese(monkeypatch) -> None:
    class _Token:
        def __init__(self, surface: str) -> None:
            self.surface = surface

    class _FakeTagger:
        def __call__(self, sentence: str):
            return [_Token("救"), _Token("済")]

    monkeypatch.setattr("summarization.term_tokenizer._load_fugashi_tagger", lambda: _FakeTagger())

    tokenizer = get_term_tokenizer("ja")

    assert tokenizer.to_words("ignored") == ("救", "済")


def test_get_term_tokenizer_uses_jieba_for_chinese(monkeypatch) -> None:
    class _FakeJieba:
        @staticmethod
        def cut(sentence: str, cut_all: bool = False):
            return ["救援", "行动"]

    monkeypatch.setattr("summarization.term_tokenizer._load_jieba", lambda: _FakeJieba())

    tokenizer = get_term_tokenizer("zh")

    assert tokenizer.to_words("ignored") == ("救援", "行动")


def test_get_term_tokenizer_uses_spacy_for_greek(monkeypatch) -> None:
    class _Token:
        def __init__(self, text: str, *, is_space: bool = False, is_punct: bool = False) -> None:
            self.text = text
            self.is_space = is_space
            self.is_punct = is_punct

    class _FakeNlp:
        def make_doc(self, sentence: str):
            return [
                _Token("Σωτηρία"),
                _Token(".", is_punct=True),
                _Token("λόγος"),
            ]

    captured: dict[str, str] = {}

    def _fake_load_spacy_tokenizer(language_code: str, model_name: str):
        captured["language_code"] = language_code
        captured["model_name"] = model_name
        return _FakeNlp()

    monkeypatch.setattr("summarization.term_tokenizer._load_spacy_tokenizer", _fake_load_spacy_tokenizer)

    tokenizer = get_term_tokenizer("el")

    assert tokenizer.to_words("ignored") == ("σωτηρία", "λόγος")
    assert captured["language_code"] == "el"
    assert captured["model_name"] == "el_core_news_sm"


def test_get_term_tokenizer_falls_back_to_embedding_tokenizer(monkeypatch) -> None:
    class _FakeTokenizer:
        all_special_tokens = ("[CLS]", "[SEP]")

        @staticmethod
        def tokenize(sentence: str):
            return ["[CLS]", "▁Alpha", "##Beta", "[SEP]"]

    monkeypatch.setattr("summarization.term_tokenizer.get_embedding_tokenizer", lambda cfg: _FakeTokenizer())

    tokenizer = get_term_tokenizer("th", embedding_cfg=EmbeddingConfig())

    assert tokenizer.to_words("ignored") == ("alpha", "beta")
