# GIDF 통합 구현 계획

## 목표

기존 LSA + TextRank hybrid summarizer에 GIDF 사후 보정을 추가한다.
`config.yaml`의 `gidf.enabled` 플래그로 GIDF 반영 여부를 런타임에 제어한다.

---

## 핵심 수학적 근거

GIDF 비활성화 시 모든 어휘의 $GIDF_i = 1$ 로 설정하면:

$$\bar{G}_j = \frac{\sum_{i \in W_j} \text{count}(i,j) \times 1}{|W_j|} = 1$$

$$\tilde{s}_j^{LSA} = s_j^{LSA} \times 1 = s_j^{LSA}$$

기존 LSA 점수와 동일하므로 **분기 없이 단일 코드 경로**로 처리 가능하다.

---

## 변경 범위 요약

| 파일 | 변경 유형 | 내용 |
|---|---|---|
| `config/summarization.yaml` | 수정 | `gidf` 섹션 추가 |
| `config.py` | 수정 | `GIDFConfig` dataclass 추가, `SummarizationConfig`에 포함 |
| `vocab_builder.py` | 신규 | GIDF 빌드/로드 함수 |
| `lsa_scorer.py` | 수정 | `apply_gidf_boost()` 추가 |
| `summarizer.py` | 수정 | GIDF 로드 및 boost 호출 연결 |
| `tests/test_gidf.py` | 신규 | GIDF 단위 테스트 |

기존 `lsa_scorer.py`, `graph_ranker.py`, `fusion.py`의 내부 로직은 **변경하지 않는다.**
GIDF 보정은 LSA 점수 출력 직후, Score Fusion 진입 직전에만 개입한다.

---

## 처리 흐름 변경

### 기존 흐름

```
문장 분리
    │
    ├── [LSA scorer]     → lsa_scores
    ├── [embedding]
    └── [PageRank]       → pagerank_scores
                │
            [fusion]  α×LSA + (1-α)×PageRank
```

### 변경 후 흐름

```
문장 분리
    │
    ├── [LSA scorer]     → lsa_scores
    │       │
    │   [GIDF boost]     → boosted_lsa_scores   ← 신규 (GIDF=1이면 lsa_scores와 동일)
    │
    ├── [embedding]
    └── [PageRank]       → pagerank_scores
                │
            [fusion]  α×boosted_LSA + (1-α)×PageRank
```

---

## 상세 구현

### 1. `config/summarization.yaml` 수정

```yaml
# ── Global GIDF 사전 ───────────────────────────────────────
gidf:
  # GIDF 사후 보정 활성화 여부
  # false: 모든 GIDF=1 → 기존 LSA 동작과 동일
  # true : DB에서 GIDF 로드 후 LSA 점수에 보정 적용
  enabled: true

  # enabled: true 일 때만 적용
  # null이면 가장 최근 version 자동 선택
  version_id: null

  # 빌드 파라미터 (build_and_store_gidf() 호출 시 사용)
  min_df: 2
  max_df: 0.85
  sublinear_tf: true
  language_code: "ko"
```

### 2. `config.py` 수정

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GIDFConfig:
    enabled: bool          = False        # 기본값: 비활성화 (기존 동작 보장)
    version_id: Optional[int] = None      # None → 최신 버전 자동 선택
    min_df: int            = 2
    max_df: float          = 0.85
    sublinear_tf: bool     = True
    language_code: str     = "ko"


@dataclass
class SummarizationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig         = field(default_factory=EmbeddingConfig)
    lsa: LSAConfig                     = field(default_factory=LSAConfig)
    graph: GraphConfig                 = field(default_factory=GraphConfig)
    fusion: FusionConfig               = field(default_factory=FusionConfig)
    extraction: ExtractionConfig       = field(default_factory=ExtractionConfig)
    parallel: ParallelConfig           = field(default_factory=ParallelConfig)
    logging: LoggingConfig             = field(default_factory=LoggingConfig)
    gidf: GIDFConfig                   = field(default_factory=GIDFConfig)  # 신규

    @classmethod
    def from_yaml(cls, path) -> "SummarizationConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            # ... 기존 필드 ...
            gidf=GIDFConfig(**raw.get("gidf", {})),   # 신규
        )
```

### 3. `vocab_builder.py` 신규

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import psycopg
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from .config import GIDFConfig


# ── 상수 ──────────────────────────────────────────────────────

# GIDF 비활성화 시 반환할 기본값
_GIDF_DISABLED: dict[str, float] = {}   # 빈 dict → get() 시 default=1.0 사용


# ── 순수 계산 (DB 없음) ───────────────────────────────────────

def _compute_gidf(
    documents: list[str],
    cfg: GIDFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TfidfVectorizer.fit으로 GIDF 계산.
    DB를 건드리지 않으므로 실패해도 재시도 가능.

    Returns:
        terms         : shape (V,)
        gidf_scores   : shape (V,)
        doc_frequency : shape (V,)
    """
    vectorizer = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
    )
    vectorizer.fit(documents)

    terms         = vectorizer.get_feature_names_out()
    gidf_scores   = vectorizer.idf_
    tfidf_matrix  = vectorizer.transform(documents)
    doc_frequency = np.diff(tfidf_matrix.tocsc().indptr).astype(int)

    return terms, gidf_scores, doc_frequency


# ── 오프라인 빌드 ─────────────────────────────────────────────

def build_and_store_gidf(
    conn: psycopg.Connection,
    cfg: GIDFConfig,
    description: str | None = None,
) -> int:
    """
    text_unit.content 전체로 GIDF 계산 후 DB 저장.

    멱등성 보장:
    - _compute_gidf()는 DB 무관 순수 계산 → 실패 시 재시도 가능
    - vocab_corpus_version + vocab_gidf를 단일 트랜잭션으로 묶음
    - 실패 시 양쪽 모두 롤백 → 고아 레코드 발생 없음

    Returns:
        version_id
    """
    rows = conn.execute("""
        SELECT candidate_id, content
        FROM text_unit
        WHERE content IS NOT NULL AND content != ''
        ORDER BY candidate_id
    """).fetchall()

    if not rows:
        raise ValueError("text_unit에 유효한 content가 없습니다.")

    documents = [r[1] for r in rows]
    logger.info(f"GIDF 빌드 시작 — text_unit {len(documents)}개")

    # 순수 계산 (DB 상태 변화 없음)
    terms, gidf_scores, doc_frequency = _compute_gidf(documents, cfg)
    logger.info(f"계산 완료 — 고유 어휘 {len(terms)}개")

    # 단일 트랜잭션으로 원자적 저장
    try:
        with conn.transaction():
            version_id: int = conn.execute("""
                INSERT INTO vocab_corpus_version
                    (total_docs, total_terms, min_df, max_df_ratio,
                     language_code, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING version_id
            """, (
                len(documents), len(terms),
                cfg.min_df, cfg.max_df,
                cfg.language_code, description,
            )).fetchone()[0]

            conn.executemany("""
                INSERT INTO vocab_gidf (version_id, term, gidf_score, doc_frequency)
                VALUES (%s, %s, %s, %s)
            """, [
                (version_id, term, float(score), int(df))
                for term, score, df in zip(terms, gidf_scores, doc_frequency)
            ])

        logger.info(f"GIDF 저장 완료 — version_id={version_id}")
        return version_id

    except Exception as e:
        logger.error(f"GIDF 저장 실패 — 전체 롤백: {e}")
        raise


# ── 온라인 로드 ───────────────────────────────────────────────

def load_gidf(
    conn: psycopg.Connection,
    cfg: GIDFConfig,
) -> dict[str, float]:
    """
    cfg.enabled=False → 빈 dict 반환 (boost에서 GIDF=1.0으로 처리)
    cfg.enabled=True  → DB에서 해당 버전 GIDF 로드

    Returns:
        {term: gidf_score} 또는 {} (비활성화 시)
    """
    if not cfg.enabled:
        logger.debug("GIDF 비활성화 — 기본값(1.0) 사용")
        return _GIDF_DISABLED

    version_id = cfg.version_id
    if version_id is None:
        row = conn.execute("""
            SELECT version_id FROM vocab_corpus_version
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()
        if row is None:
            raise RuntimeError(
                "빌드된 GIDF 버전이 없습니다. build_and_store_gidf()를 먼저 실행하세요."
            )
        version_id = row[0]

    rows = conn.execute("""
        SELECT term, gidf_score FROM vocab_gidf WHERE version_id = %s
    """, (version_id,)).fetchall()

    logger.debug(f"GIDF 로드 — version_id={version_id}, {len(rows)}개 어휘")
    return {term: score for term, score in rows}
```

### 4. `lsa_scorer.py` 수정 — `apply_gidf_boost()` 추가

```python
from collections import Counter
import numpy as np


def apply_gidf_boost(
    lsa_scores: dict[int, float],
    sentences: list[str],
    gidf: dict[str, float],
) -> dict[int, float]:
    """
    LSA 점수에 문장별 GIDF 보정 계수를 곱한다.

    수식:
        G_bar_j = Σ count(i,j) × GIDF_i / |W_j|
        s~_j    = s_j^LSA × G_bar_j

    gidf가 빈 dict일 때 (GIDF 비활성화):
        gidf.get(token, 1.0) → 모든 GIDF=1.0
        G_bar_j = |W_j| / |W_j| = 1.0
        s~_j = s_j^LSA   (기존과 동일)
    """
    boosted: dict[int, float] = {}

    for j, sentence in enumerate(sentences):
        tokens = sentence.lower().split()

        if not tokens:
            boosted[j] = lsa_scores[j]
            continue

        tf = Counter(tokens)

        # gidf가 빈 dict이면 get() default=1.0 → G_bar_j=1.0
        numerator   = sum(count * gidf.get(token, 1.0) for token, count in tf.items())
        denominator = len(tokens)
        g_bar_j     = numerator / denominator

        boosted[j] = lsa_scores[j] * g_bar_j

    return _minmax_normalize(boosted)
```

### 5. `summarizer.py` 수정

```python
from .vocab_builder import load_gidf
from .lsa_scorer import compute_lsa_scores, apply_gidf_boost


class HybridExtractiveSummarizer:

    def __init__(self, cfg: SummarizationConfig, conn=None) -> None:
        self.cfg  = cfg
        self.conn = conn   # GIDF 활성화 시 필요

        # GIDF 사전을 인스턴스 초기화 시 한 번만 로드 (매 요약마다 DB 조회 방지)
        self._gidf: dict[str, float] = {}
        if cfg.gidf.enabled:
            if conn is None:
                raise ValueError("GIDF 활성화 시 DB 연결(conn)이 필요합니다.")
            self._gidf = load_gidf(conn, cfg.gidf)

    def summarize_one(self, text: str) -> str:
        cfg = self.cfg

        # ① 문장 분리
        sentences = split_sentences(text, cfg.preprocessing)
        if len(sentences) < 2:
            return text

        # ② 임베딩 + 유사도 행렬
        embeddings = embed_sentences(sentences, cfg.embedding)
        sim_matrix = compute_similarity_matrix(embeddings)

        # ③ LSA 점수
        lsa_scores = compute_lsa_scores(sentences, cfg.lsa)

        # ④ GIDF 사후 보정
        #    enabled=False → self._gidf={} → G_bar_j=1.0 → 기존과 동일
        boosted_lsa = apply_gidf_boost(lsa_scores, sentences, self._gidf)

        # ⑤ PageRank 점수
        pagerank_scores = compute_pagerank_scores(sim_matrix, cfg.graph)

        # ⑥ Score Fusion
        fused = fuse_scores(boosted_lsa, pagerank_scores, cfg.fusion)

        # ⑦ 추출
        return " ".join(
            extract_top_sentences(sentences, fused, sim_matrix, cfg.extraction)
        )
```

---

## 테스트 계획 (`tests/test_gidf.py`)

```python
import pytest
from summarization.lsa_scorer import apply_gidf_boost

# ── 1. GIDF 비활성화 시 LSA 점수 불변 ────────────────────────
def test_gidf_disabled_identity():
    """gidf={} 이면 boosted == original (G_bar_j = 1.0)"""
    lsa_scores = {0: 0.8, 1: 0.5, 2: 0.3}
    sentences  = ["성령이 임하셨다", "오늘 날씨가 맑다", "구원의 은혜"]
    boosted    = apply_gidf_boost(lsa_scores, sentences, gidf={})

    # 정규화 후에도 순위는 동일해야 함
    assert sorted(boosted, key=boosted.get, reverse=True) == [0, 1, 2]


# ── 2. 도메인 용어 포함 문장 증폭 확인 ───────────────────────
def test_gidf_boosts_domain_terms():
    """GIDF가 높은 신학 용어를 포함한 문장의 점수가 증폭되어야 함"""
    lsa_scores = {0: 0.5, 1: 0.5}   # LSA 점수 동일
    sentences  = [
        "성령 구원 칭의 성화",     # 신학 용어 집중
        "오늘 날씨가 맑고 따뜻하다",  # 일반 문장
    ]
    gidf = {"성령": 0.95, "구원": 0.90, "칭의": 0.92, "성화": 0.88,
            "오늘": 0.10, "날씨": 0.08, "맑고": 0.05, "따뜻하다": 0.06}

    boosted = apply_gidf_boost(lsa_scores, sentences, gidf)
    assert boosted[0] > boosted[1]


# ── 3. 멱등성: build 실패 시 롤백 확인 ───────────────────────
def test_build_gidf_rollback_on_failure(conn):
    """vocab_gidf INSERT 실패 시 vocab_corpus_version도 롤백되어야 함"""
    version_count_before = conn.execute(
        "SELECT COUNT(*) FROM vocab_corpus_version"
    ).fetchone()[0]

    with pytest.raises(Exception):
        # 강제로 INSERT 실패 유발 (잘못된 language_code)
        build_and_store_gidf(conn, GIDFConfig(language_code="INVALID"))

    version_count_after = conn.execute(
        "SELECT COUNT(*) FROM vocab_corpus_version"
    ).fetchone()[0]

    assert version_count_before == version_count_after  # 고아 레코드 없음


# ── 4. load_gidf — 최신 버전 자동 선택 ───────────────────────
def test_load_gidf_auto_latest(conn):
    cfg = GIDFConfig(enabled=True, version_id=None)
    gidf = load_gidf(conn, cfg)
    assert isinstance(gidf, dict)
    assert len(gidf) > 0
```

---

## 구현 순서

1. `config.py` — `GIDFConfig` 추가, `SummarizationConfig`에 포함, `from_yaml()` 수정
2. `config/summarization.yaml` — `gidf` 섹션 추가 (`enabled: false` 기본값)
3. `vocab_builder.py` 신규 작성
4. `lsa_scorer.py` — `apply_gidf_boost()` 추가
5. `summarizer.py` — `__init__`에서 GIDF 로드, `summarize_one()`에서 boost 호출
6. `tests/test_gidf.py` 작성 및 전체 테스트 통과 확인
7. `gidf.enabled: true`로 전환 후 기존 요약 결과와 ROUGE 점수 비교
