# Hybrid Extractive Summarization — Implementation Plan

## 호환성 검토 결과

| 라이브러리 | 버전 | Python 3.12 | 기존 deps 충돌 | 결론 |
|---|---|---|---|---|
| `sumy` | 0.12.0 | ✅ (`>=3.8`) | 없음 | **그대로 사용** |
| `pytextrank` | 3.3.0 | ✅ (spaCy `<3.15,>=3.9`) | 없음 | **그대로 사용** |
| `pysbd` | 0.3.4 | ✅ (`>=3`, 외부 의존성 없음) | 없음 | **추가 — 기본 splitter** |
| `spacy` | 3.8.11 | ✅ (`<3.15,>=3.9`) | 없음 | 추가 필요 (fallback splitter) |
| `nltk` | latest | ✅ | 없음 | 추가 필요 (sumy 의존) |
| `scipy` | latest | ✅ | 없음 | 추가 필요 (pytextrank 의존) |
| `networkx` | latest | ✅ | 없음 | 추가 필요 (PageRank) |

---

## 추가 dependencies (pyproject.toml에 병합)

```toml
[project]
dependencies = [
    # ... 기존 deps ...

    # Extractive summarization
    "sumy>=0.12.0",
    "pytextrank>=3.3.0",
    "pysbd>=0.3.4",         # 기본 sentence splitter (의존성 없음)
    "spacy>=3.8.0,<3.15",   # fallback splitter용
    "nltk>=3.9.0",
    "scipy>=1.13.0",
    "networkx>=3.3",
]
```

spaCy 언어 모델은 `sentence_splitter: "spacy"` 사용 시에만 필요 (런타임 다운로드):

```bash
# 영어
python -m spacy download en_core_web_sm

# 한국어 문서 포함 시
pip install https://github.com/explosion/spacy-models/releases/download/ko_core_news_sm-3.8.0/ko_core_news_sm-3.8.0-py3-none-any.whl
```

---

## 프로젝트 구조

```
src/
└── summarization/
    ├── __init__.py
    ├── config.py               # Config dataclass + YAML 로더
    ├── sentence_splitter.py    # pySBD(기본) / spaCy / WtP 전략 패턴
    ├── embedder.py             # jina-embeddings-v5-nano 래퍼
    ├── lsa_scorer.py           # sumy LSA 점수 계산
    ├── graph_ranker.py         # pytextrank + NetworkX PageRank
    ├── fusion.py               # 점수 융합 (α 가중합)
    ├── summarizer.py           # 전체 파이프라인 오케스트레이터
    └── utils.py                # 청킹, 로깅 헬퍼

config/
└── summarization.yaml          # 사람이 직접 수정하는 하이퍼파라미터

tests/
└── summarization/
    ├── test_sentence_splitter.py
    ├── test_embedder.py
    ├── test_graph_ranker.py
    ├── test_fusion.py
    └── test_summarizer.py
```

---

## Configuration 파일 (`config/summarization.yaml`)

> 사람이 직접 수정하는 모든 하이퍼파라미터를 이 파일 하나에서 관리합니다.

```yaml
# ============================================================
#  Hybrid Extractive Summarization — Hyperparameter Config
# ============================================================

# ── 입력 전처리 ──────────────────────────────────────────────
preprocessing:
  # 문장 분리 엔진 선택
  # "pysbd"  : 기본값. rule-based, 의존성 없음, 학술 약어(et al., Fig., vs.) 처리 우수
  # "spacy"  : 의존 구문 기반, 가장 언어적으로 정확하나 느림 (spacy 모델 필요)
  # "wtp"    : CANINE neural model, 최고 정확도, 모델 로딩 필요 (~400MB)
  sentence_splitter: "pysbd"
  # pySBD 사용 시: 텍스트 언어 코드 (ISO 639-1)
  # 지원 언어: en, de, fr, es, it, nl, pt, ja, zh, ru, ...
  language: "en"
  # spaCy 사용 시에만 적용: 언어 모델 이름
  spacy_model: "en_core_web_sm"
  # WtP 사용 시에만 적용: 모델 variant
  # "wtp-canine-s-12l" (균형) | "wtp-canine-s-6l" (속도) | "wtp-bert-mini" (경량)
  wtp_model: "wtp-canine-s-12l"
  # 문장 분리 후 최소 토큰 길이 (이하 문장은 필터링)
  min_sentence_tokens: 5
  # 문장 분리 후 최대 토큰 길이 (초과 문장은 truncation)
  max_sentence_tokens: 512

# ── 임베딩 모델 ──────────────────────────────────────────────
embedding:
  model_name: "jinaai/jina-embeddings-v5-nano"
  # text-matching: 문장 간 대칭적 유사도 계산에 최적화된 LoRA adapter
  # 선택지: "text-matching" | "retrieval" | "clustering" | "classification"
  task: "text-matching"
  # Matryoshka MRL: 768에서 줄여 속도/메모리 트레이드오프 가능
  # 선택지: 768 | 512 | 256 | 128 | 64 (낮을수록 빠르지만 품질 저하)
  output_dim: 768
  normalize: true
  batch_size: 32
  device: "auto"           # "auto" | "cuda" | "cpu" | "mps"

# ── LSA 점수 (sumy) ──────────────────────────────────────────
lsa:
  # SVD에서 추출할 토픽(컴포넌트) 수. None이면 min(문장수, 단어수)로 자동 결정
  # 학술 문서는 토픽이 다양하므로 높은 값 권장
  n_components: null        # null | int (예: 10)
  # 점수 정규화 방식
  # "minmax": 0~1 정규화 / "rank": 순위 기반 정규화
  normalize_method: "minmax"

# ── 그래프 랭킹 (pytextrank + PageRank) ──────────────────────
graph:
  # 유사도 임계값: 이 값 미만의 엣지는 그래프에서 제거
  # 학술 문서는 0.15~0.25 권장 (0.1보다 높여야 노이즈 연결 감소)
  similarity_threshold: 0.20
  # PageRank damping factor (표준값 0.85)
  damping: 0.85
  # PageRank 최대 반복 횟수
  max_iter: 100
  # 수렴 허용 오차
  tol: 1.0e-6
  # 점수 정규화 방식 ("minmax" | "rank")
  normalize_method: "minmax"

# ── 점수 융합 ────────────────────────────────────────────────
fusion:
  # α: LSA 가중치. (1-α)는 PageRank 가중치
  # α=0.0 → 순수 PageRank, α=1.0 → 순수 LSA
  # 학술 문서: 0.5 (구조적 통계 신호와 의미 그래프를 동등 반영)
  alpha: 0.5

# ── 요약 추출 ────────────────────────────────────────────────
extraction:
  # 원문 토큰 수 대비 summary가 사용할 최대 비율
  # 예: 0.30 → 선택된 문장들의 토큰 합이 원문의 30% 이하가 되도록 추출
  token_budget_ratio: 0.30
  # 원문 등장 순서 유지 여부 (True 권장 — 흐름 자연스러움)
  preserve_order: true
  # 추출된 문장 간 최소 유사도 상한 (중복 문장 억제)
  # 0.95 이상 유사한 문장은 중복으로 간주하고 하위 점수 문장 제거
  redundancy_threshold: 0.95

# ── 병렬 처리 ────────────────────────────────────────────────
parallel:
  # 문단 단위 병렬 처리 워커 수
  # -1: CPU 코어 수 자동 감지, 1: 순차 처리 (디버깅용)
  n_workers: -1
  # 임베딩은 GPU 배치로 처리하므로 문단 병렬과 분리
  # 임베딩 배치를 몇 문단씩 묶어서 GPU에 올릴지
  embedding_chunk_size: 10

# ── 로깅 ─────────────────────────────────────────────────────
logging:
  level: "INFO"             # "DEBUG" | "INFO" | "WARNING"
  # 각 문단의 상위 문장 점수 출력 여부 (디버깅에 유용)
  log_sentence_scores: false
```

---

## 모듈별 구현 명세

### `config.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import yaml


@dataclass
class PreprocessingConfig:
    sentence_splitter: Literal["pysbd", "spacy", "wtp"] = "pysbd"
    language: str = "en"
    spacy_model: str = "en_core_web_sm"
    wtp_model: str = "wtp-canine-s-12l"
    min_sentence_tokens: int = 5
    max_sentence_tokens: int = 512


@dataclass
class EmbeddingConfig:
    model_name: str = "jinaai/jina-embeddings-v5-nano"
    task: str = "text-matching"
    output_dim: int = 768
    normalize: bool = True
    batch_size: int = 32
    device: str = "auto"


@dataclass
class LSAConfig:
    n_components: Optional[int] = None
    normalize_method: str = "minmax"


@dataclass
class GraphConfig:
    similarity_threshold: float = 0.20
    damping: float = 0.85
    max_iter: int = 100
    tol: float = 1e-6
    normalize_method: str = "minmax"


@dataclass
class FusionConfig:
    alpha: float = 0.5          # 학술 문서 기본값


@dataclass
class ExtractionConfig:
    token_budget_ratio: float = 0.30
    preserve_order: bool = True
    redundancy_threshold: float = 0.95


@dataclass
class ParallelConfig:
    n_workers: int = -1
    embedding_chunk_size: int = 10


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_sentence_scores: bool = False


@dataclass
class SummarizationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    lsa: LSAConfig = field(default_factory=LSAConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SummarizationConfig":
        with open(path) as f:
            raw: dict = yaml.safe_load(f)
        return cls(
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            lsa=LSAConfig(**raw.get("lsa", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            fusion=FusionConfig(**raw.get("fusion", {})),
            extraction=ExtractionConfig(**raw.get("extraction", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
```

---

### `sentence_splitter.py`

```python
from __future__ import annotations
from functools import lru_cache
from .config import PreprocessingConfig


# ── pySBD (기본) ─────────────────────────────────────────────────────────────

def _split_pysbd(text: str, language: str) -> list[str]:
    import pysbd
    seg = pysbd.Segmenter(language=language, clean=True)
    return seg.segment(text)


# ── spaCy (fallback) ──────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_spacy(model_name: str):
    import spacy
    return spacy.load(model_name, disable=["ner", "lemmatizer"])


def _split_spacy(text: str, model_name: str) -> list[str]:
    nlp = _load_spacy(model_name)
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


# ── WtP (고정밀) ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _load_wtp(model_name: str):
    from wtpsplit import WtP
    return WtP(model_name)


def _split_wtp(text: str, model_name: str, language: str) -> list[str]:
    wtp = _load_wtp(model_name)
    return wtp.split(text, lang_code=language)


# ── 공통 필터 ─────────────────────────────────────────────────────────────────

def _filter(sentences: list[str], min_tok: int, max_tok: int) -> list[str]:
    result = []
    for s in sentences:
        tokens = s.split()           # 간단한 whitespace 토큰 카운트
        if len(tokens) < min_tok:
            continue
        if len(tokens) > max_tok:
            s = " ".join(tokens[:max_tok])  # truncation
        result.append(s)
    return result


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def split_sentences(text: str, cfg: PreprocessingConfig) -> list[str]:
    """
    cfg.sentence_splitter 값에 따라 엔진을 선택해 문장 분리 수행.

    "pysbd"  → pySBD rule-based (기본, 학술 약어 처리 우수)
    "spacy"  → spaCy dependency parser (정확하나 느림)
    "wtp"    → WtP CANINE neural model (최고 정확도)
    """
    match cfg.sentence_splitter:
        case "pysbd":
            raw = _split_pysbd(text, cfg.language)
        case "spacy":
            raw = _split_spacy(text, cfg.spacy_model)
        case "wtp":
            raw = _split_wtp(text, cfg.wtp_model, cfg.language)
        case _:
            raise ValueError(
                f"Unknown sentence_splitter: '{cfg.sentence_splitter}'. "
                "Choose from: 'pysbd', 'spacy', 'wtp'"
            )

    return _filter(raw, cfg.min_sentence_tokens, cfg.max_sentence_tokens)
```

---

### `embedder.py`

```python
from __future__ import annotations
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from .config import EmbeddingConfig


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True, device=device)


def embed_sentences(
    sentences: list[str],
    cfg: EmbeddingConfig,
) -> np.ndarray:
    """
    문장 리스트 → L2 정규화된 임베딩 행렬 (n_sent × output_dim).
    task="text-matching": 대칭적 코사인 유사도 계산에 최적화된 LoRA 활성화.
    """
    model = _load_model(cfg.model_name, cfg.device)
    with torch.inference_mode():
        embeddings = model.encode(
            sentences,
            task=cfg.task,
            normalize_embeddings=cfg.normalize,
            batch_size=cfg.batch_size,
            truncate_dim=cfg.output_dim if cfg.output_dim < 768 else None,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return embeddings  # shape: (n, output_dim)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    정규화된 벡터의 내적 = 코사인 유사도.
    O(n² × d) 단일 연산으로 전체 유사도 행렬 계산.
    """
    return (embeddings @ embeddings.T).astype(np.float32)
```

---

### `lsa_scorer.py`

```python
from __future__ import annotations
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from .config import LSAConfig


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    vals = np.array(list(scores.values()), dtype=np.float32)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def _rank_normalize(scores: dict[int, float]) -> dict[int, float]:
    n = len(scores)
    ranked = sorted(scores, key=scores.get)
    return {idx: (rank + 1) / n for rank, idx in enumerate(ranked)}


def compute_lsa_scores(
    sentences: list[str],
    cfg: LSAConfig,
    language: str = "english",
) -> dict[int, float]:
    """
    sumy LsaSummarizer로 각 문장의 LSA 중요도 점수 계산.
    Returns:
        {sentence_index: normalized_score}
    """
    text = " ".join(sentences)
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)

    # sumy 내부 _compute_ratings 메서드로 개별 점수 추출
    ratings: dict = summarizer._compute_ratings(parser.document.sentences)

    # sumy sentence index → 우리 sentence list index 매핑
    raw_scores = {i: float(ratings.get(i, 0.0)) for i in range(len(sentences))}

    if cfg.normalize_method == "rank":
        return _rank_normalize(raw_scores)
    return _minmax_normalize(raw_scores)
```

---

### `graph_ranker.py`

```python
from __future__ import annotations
import numpy as np
import networkx as nx
from .config import GraphConfig


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    vals = np.array(list(scores.values()), dtype=np.float32)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}


def _rank_normalize(scores: dict[int, float]) -> dict[int, float]:
    n = len(scores)
    ranked = sorted(scores, key=scores.get)
    return {idx: (rank + 1) / n for rank, idx in enumerate(ranked)}


def compute_pagerank_scores(
    sim_matrix: np.ndarray,
    cfg: GraphConfig,
) -> dict[int, float]:
    """
    유사도 행렬 → 가중치 방향 그래프 → PageRank 점수.

    threshold 이하 엣지 제거로 노이즈 연결 차단.
    학술 문서 권장값: similarity_threshold=0.20
    """
    n = sim_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # threshold 초과 엣지만 추가 (대각선 제외)
    rows, cols = np.where(
        (sim_matrix > cfg.similarity_threshold) & (np.eye(n, dtype=bool) == False)
    )
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(sim_matrix[i, j]))

    # 고립 노드 방지: 엣지 없는 노드는 균등 가중치 자가루프
    for node in G.nodes:
        if G.degree(node) == 0:
            G.add_edge(node, node, weight=1.0)

    raw_scores: dict[int, float] = nx.pagerank(
        G,
        alpha=cfg.damping,
        weight="weight",
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )

    if cfg.normalize_method == "rank":
        return _rank_normalize(raw_scores)
    return _minmax_normalize(raw_scores)
```

---

### `fusion.py`

```python
from __future__ import annotations
import numpy as np
from .config import FusionConfig, ExtractionConfig


def fuse_scores(
    lsa_scores: dict[int, float],
    pagerank_scores: dict[int, float],
    cfg: FusionConfig,
) -> dict[int, float]:
    """
    α × LSA + (1-α) × PageRank 가중합.
    두 점수는 모두 [0, 1]로 정규화된 상태를 가정.

    학술 문서 기본값 α=0.5: 통계 신호와 의미 그래프를 동등 반영.
    """
    return {
        i: cfg.alpha * lsa_scores[i] + (1.0 - cfg.alpha) * pagerank_scores[i]
        for i in lsa_scores
    }


def remove_redundant(
    selected_indices: list[int],
    sim_matrix: np.ndarray,
    threshold: float,
    fused_scores: dict[int, float],
) -> list[int]:
    """
    선택된 문장 중 유사도 threshold 초과 쌍의 하위 점수 문장을 제거.
    Maximal Marginal Relevance (MMR) 간소화 버전.
    """
    kept: list[int] = []
    for idx in sorted(selected_indices, key=lambda i: -fused_scores[i]):
        if all(sim_matrix[idx, k] < threshold for k in kept):
            kept.append(idx)
    return kept


def extract_top_sentences(
    sentences: list[str],
    fused_scores: dict[int, float],
    sim_matrix: np.ndarray,
    cfg: ExtractionConfig,
    source_token_count: int | None = None,
) -> list[str]:
    """
    1. 원문 토큰 수 기준 예산 계산
    2. fusion score 내림차순으로 문장 선택
    3. redundancy_threshold 초과 문장 제거
    4. 원문 순서 복원 (preserve_order)
    """
    token_counts = {i: len(sentence.split()) for i, sentence in enumerate(sentences)}
    total_tokens = source_token_count or sum(token_counts.values())
    token_budget = max(1, int(total_tokens * cfg.token_budget_ratio))
    token_budget = max(token_budget, min(token_counts.values()))

    ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)
    kept: list[int] = []
    selected_tokens = 0
    for idx in ranked:
        if any(sim_matrix[idx, kept_idx] >= cfg.redundancy_threshold for kept_idx in kept):
            continue
        sentence_tokens = token_counts[idx]
        if selected_tokens + sentence_tokens > token_budget:
            continue
        kept.append(idx)
        selected_tokens += sentence_tokens
        if selected_tokens >= token_budget:
            break

    # 원문 순서 복원
    if cfg.preserve_order:
        kept = sorted(kept)

    return [sentences[i] for i in kept]
```

---

### `summarizer.py` (오케스트레이터)

```python
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from loguru import logger

from .config import SummarizationConfig
from .sentence_splitter import split_sentences
from .embedder import embed_sentences, compute_similarity_matrix
from .lsa_scorer import compute_lsa_scores
from .graph_ranker import compute_pagerank_scores
from .fusion import fuse_scores, extract_top_sentences


class HybridExtractiveSummarizer:
    """
    단일 문단 및 다수 문단 병렬 처리를 지원하는 하이브리드 추출 요약기.

    Usage:
        cfg = SummarizationConfig.from_yaml("config/summarization.yaml")
        summarizer = HybridExtractiveSummarizer(cfg)
        results = summarizer.summarize_batch(paragraphs)
    """

    def __init__(self, cfg: SummarizationConfig) -> None:
        self.cfg = cfg
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=cfg.logging.level)

    def summarize_one(self, text: str) -> str:
        """단일 문단 요약."""
        cfg = self.cfg

        # ① 문장 분리
        sentences = split_sentences(text, cfg.preprocessing)
        if len(sentences) < 2:
            return text

        logger.debug(f"Sentences: {len(sentences)}")

        # ② 임베딩 + 유사도 행렬
        source_token_count = len(text.split())
        embeddings = embed_sentences(sentences, cfg.embedding)
        sim_matrix = compute_similarity_matrix(embeddings)

        # ③ LSA 점수 (sumy)
        lsa_scores = compute_lsa_scores(sentences, cfg.lsa)

        # ④ PageRank 점수 (pytextrank 스타일 — networkx 직접 사용)
        pagerank_scores = compute_pagerank_scores(sim_matrix, cfg.graph)

        # ⑤ 점수 융합 (α=0.5)
        fused = fuse_scores(lsa_scores, pagerank_scores, cfg.fusion)

        if cfg.logging.log_sentence_scores:
            for i, s in enumerate(sentences):
                logger.debug(
                    f"  [{i:02d}] lsa={lsa_scores[i]:.3f} "
                    f"pr={pagerank_scores[i]:.3f} "
                    f"fused={fused[i]:.3f} | {s[:60]}"
                )

        # ⑥ 추출
        return " ".join(
            extract_top_sentences(
                sentences,
                fused,
                sim_matrix,
                cfg.extraction,
                source_token_count=source_token_count,
            )
        )

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """
        여러 문단 병렬 처리.
        임베딩은 GPU 배치로 수행, PageRank/LSA는 CPU 스레드풀로 병렬화.
        """
        cfg = self.cfg
        n_workers = (
            os.cpu_count() or 4
            if cfg.parallel.n_workers == -1
            else cfg.parallel.n_workers
        )

        if n_workers == 1:
            return [self.summarize_one(t) for t in texts]

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self.summarize_one, t): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(texts))]
```

---

## 처리 흐름 요약

```
텍스트 입력
    │
    ▼
[sentence_splitter] 문장 분리 + 길이 필터링
  ├─ pySBD   ← 기본값 (학술 약어 처리, 의존성 없음)
  ├─ spaCy   ← 의존 구문 기반, 고정밀
  └─ WtP     ← CANINE neural, 최고 정확도
    │
    ├──────────────────────────┐
    ▼                          ▼
[embedder]               [lsa_scorer]
jina-v5-nano             sumy LsaSummarizer
task="text-matching"     SVD 기반 LSA 점수
유사도 행렬 계산          → 정규화 [0,1]
    │                          │
    ▼                          │
[graph_ranker]                 │
NetworkX PageRank              │
threshold=0.20                 │
→ 정규화 [0,1]                 │
    │                          │
    └──────────┬───────────────┘
               ▼
           [fusion]
     α×LSA + (1-α)×PageRank
           α = 0.5
               │
               ▼
         [extraction]
   토큰 예산 기반 선정 + 중복 제거
    + 원문 순서 복원
               │
               ▼
           요약문 출력
```

---

## 사용 예시

```python
from summarization.config import SummarizationConfig
from summarization.summarizer import HybridExtractiveSummarizer

# config 로드 (config/summarization.yaml)
cfg = SummarizationConfig.from_yaml("config/summarization.yaml")

summarizer = HybridExtractiveSummarizer(cfg)

# 단일 문단
summary = summarizer.summarize_one(paragraph_text)

# 10개 문단 병렬 처리
summaries = summarizer.summarize_batch(paragraphs)   # list[str]
```

alpha, token_budget_ratio, similarity_threshold 등 하이퍼파라미터는 모두 `config/summarization.yaml`만 수정하면 되며, 코드 변경 없이 적용됩니다.

---

## 구현 순서 (권장)

1. `config.py` + `config/summarization.yaml` 작성 및 로딩 검증
2. `sentence_splitter.py` 구현 + 세 엔진 단위 테스트 (`test_sentence_splitter.py`)
   - pySBD 기본 동작 확인 (약어, 참조, 수식 포함 학술 문장)
   - spaCy / WtP 전환 시 동일 인터페이스 검증
3. `embedder.py` 구현 + GPU 메모리 확인 (batch_size 튜닝)
4. `lsa_scorer.py` 구현 + 점수 분포 확인
5. `graph_ranker.py` 구현 + threshold 민감도 실험
6. `fusion.py` 구현 + α 변화에 따른 ROUGE 점수 비교
7. `summarizer.py` 통합 + 병렬 처리 성능 테스트
8. `tests/` 전체 정리 (pytest)
