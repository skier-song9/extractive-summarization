# Global Vocabulary GIDF 구축 계획

## 개요

`text_unit.content` 전체를 corpus로 삼아 어휘별 Global IDF(GIDF) 점수를 계산하고,
결과를 PostgreSQL(`vocab_corpus_version`, `vocab_gidf`)에 저장하는 오프라인 파이프라인.

GIDF는 온라인 요약 파이프라인에서 LSA 점수의 사후 보정 계수로 사용된다.

$$\tilde{s}_j^{LSA} = s_j^{LSA} \times \bar{G}_j, \qquad
\bar{G}_j = \frac{\sum_{i \in W_j} \text{count}(i,j) \times GIDF_i}{|W_j|}$$

---

## DB 스키마

```sql
-- 빌드 이력 및 corpus 조건 추적
CREATE TABLE vocab_corpus_version (
  version_id    bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
  total_docs    integer NOT NULL,
  total_terms   integer NOT NULL,
  min_df        integer NOT NULL,
  max_df_ratio  float   NOT NULL,
  language_code text    NOT NULL REFERENCES language(language_code),
  description   text,
  PRIMARY KEY (version_id)
);

COMMENT ON TABLE vocab_corpus_version IS 'TF-IDF 어휘 사전 빌드 이력 및 corpus 조건을 추적하는 테이블 (PK: version_id, auto-increment, FK: language_code -> language.language_code).';
COMMENT ON COLUMN vocab_corpus_version.version_id IS '빌드 이력을 고유하게 식별하는 ID (PK, auto-increment).';
COMMENT ON COLUMN vocab_corpus_version.created_at IS '어휘 사전이 빌드된 시각.';
COMMENT ON COLUMN vocab_corpus_version.total_docs IS '이 버전 빌드에 사용된 text_unit 수.';
COMMENT ON COLUMN vocab_corpus_version.total_terms IS '이 버전에서 추출된 고유 어휘 수.';
COMMENT ON COLUMN vocab_corpus_version.min_df IS '어휘 포함 기준 최소 등장 text_unit 수 threshold.';
COMMENT ON COLUMN vocab_corpus_version.max_df_ratio IS '어휘 제외 기준 최대 등장 비율 threshold (0.0 ~ 1.0).';
COMMENT ON COLUMN vocab_corpus_version.language_code IS '이 버전 어휘 사전의 대상 언어 코드 (FK -> language.language_code).';
COMMENT ON COLUMN vocab_corpus_version.description IS '이 버전 빌드에 대한 부가 설명 또는 메모.';


-- 어휘별 GIDF 가중치 (vocab_corpus_version과 1:N)
CREATE TABLE vocab_gidf (
  vocab_id      bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  version_id    bigint  NOT NULL
                REFERENCES vocab_corpus_version(version_id) ON DELETE CASCADE,
  term          text    NOT NULL,
  gidf_score    float   NOT NULL,
  doc_frequency integer NOT NULL,
  PRIMARY KEY (vocab_id),
  UNIQUE (version_id, term)
);

COMMENT ON TABLE vocab_gidf IS 'text_unit corpus 기준 어휘별 GIDF 가중치 사전. version_id로 빌드 이력과 연결 (PK: vocab_id, auto-increment, FK: version_id -> vocab_corpus_version.version_id).';
COMMENT ON COLUMN vocab_gidf.vocab_id IS '어휘 항목을 고유하게 식별하는 ID (PK, auto-increment).';
COMMENT ON COLUMN vocab_gidf.version_id IS '이 어휘가 속한 빌드 버전 ID (FK -> vocab_corpus_version.version_id).';
COMMENT ON COLUMN vocab_gidf.term IS '어휘 사전에 등록된 단어 또는 토큰.';
COMMENT ON COLUMN vocab_gidf.gidf_score IS 'text_unit 전체 corpus 기준으로 계산된 global IDF 가중치 점수.';
COMMENT ON COLUMN vocab_gidf.doc_frequency IS '해당 term이 등장한 text_unit 수 (GIDF 해석 및 검증 용도).';

-- 인덱스
CREATE INDEX idx_vocab_gidf_version_term ON vocab_gidf (version_id, term);
CREATE INDEX idx_vocab_gidf_score        ON vocab_gidf (version_id, gidf_score DESC);
```

---

## Versioning의 역할

문서가 추가될 때마다 GIDF 전체를 재계산한다. Versioning은 "현재 최신값 관리"가
목적이 아니라 아래 세 가지를 위한 것이다.

| 목적 | 설명 |
|---|---|
| 재현성 | 과거 시점의 요약이 어떤 GIDF 기준으로 생성됐는지 역추적 가능 |
| A/B 비교 | `version_id`만 바꿔 구버전·신버전 GIDF의 요약 품질(ROUGE) 비교 |
| 롤백 | 신규 빌드 품질이 낮으면 코드 변경 없이 `version_id` 지정으로 즉시 롤백 |

---

## 멱등성 설계

### 문제: 고아(orphan) 레코드 발생 시나리오

```
① vocab_corpus_version INSERT  → commit
② vocab_gidf 배치 INSERT       → 실패 (OOM / 네트워크 / DB timeout)

결과: version 레코드는 있고 gidf 레코드는 없는 불일치 상태
```

### 해결 원칙 2가지

**계산과 저장 분리**
`_compute_gidf()`는 DB를 전혀 건드리지 않는 순수 계산 함수로 분리한다.
실패해도 DB 상태가 변하지 않으므로 횟수 제한 없이 재시도 가능하다.

**단일 트랜잭션 보장**
`vocab_corpus_version INSERT`와 `vocab_gidf INSERT`를 `with conn.transaction()`으로
하나의 원자적 단위로 묶는다. 어느 한쪽이라도 실패하면 양쪽 모두 롤백된다.

```
_compute_gidf()          ← DB 없음, 실패해도 무해
        │
        ▼
with conn.transaction():
    ① vocab_corpus_version INSERT
    ② vocab_gidf 배치 INSERT
        │ 실패 시 ①②  모두 자동 롤백
        ▼
    commit                ← 여기까지 도달해야만 저장
```

---

## 구현

### `config/summarization.yaml` 추가 항목

```yaml
# ── Global GIDF 사전 ───────────────────────────────────────
gidf:
  # 어휘 포함 기준: 최소 등장 text_unit 수
  min_df: 2
  # 어휘 제외 기준: 전체 text_unit의 85% 이상 등장 시 제거
  max_df: 0.85
  # TF에 log 스케일 적용 여부 (1 + log(tf))
  sublinear_tf: true
  language_code: "ko"
  # 최신 버전을 자동 선택할지 여부
  # false로 설정 시 version_id를 명시적으로 지정해야 함
  auto_latest: true
  # auto_latest: false 일 때 사용할 version_id
  version_id: null
```

### `config.py` 추가

```python
@dataclass
class GIDFConfig:
    min_df: int        = 2
    max_df: float      = 0.85
    sublinear_tf: bool = True
    language_code: str = "ko"
    auto_latest: bool  = True
    version_id: int | None = None
```

### `vocab_builder.py`

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import psycopg
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from .config import GIDFConfig


# ── 순수 계산 함수 (DB 없음) ──────────────────────────────────

def _compute_gidf(
    documents: list[str],
    cfg: GIDFConfig,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    TfidfVectorizer.fit으로 GIDF 계산.
    DB를 건드리지 않으므로 실패해도 무해 — 재시도 자유.

    Returns:
        terms         : shape (V,)  — 어휘 배열
        gidf_scores   : shape (V,)  — IDF 점수 배열
        doc_frequency : shape (V,)  — 각 term의 등장 text_unit 수
    """
    vectorizer = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
    )
    vectorizer.fit(documents)

    terms         = vectorizer.get_feature_names_out()       # (V,)
    gidf_scores   = vectorizer.idf_                          # (V,)
    tfidf_matrix  = vectorizer.transform(documents)          # sparse (N, V)
    doc_frequency = np.diff(tfidf_matrix.tocsc().indptr).astype(int)  # (V,)

    return terms, gidf_scores, doc_frequency


# ── 오프라인 빌드 진입점 ──────────────────────────────────────

def build_and_store_gidf(
    conn: psycopg.Connection,
    cfg: GIDFConfig,
    description: str | None = None,
) -> int:
    """
    text_unit.content 전체로 GIDF 계산 후 DB 저장.
    멱등성 보장: 단일 트랜잭션으로 version + gidf를 원자적으로 저장.

    Returns:
        version_id : 생성된 vocab_corpus_version.version_id
    """
    # ① corpus 로드
    rows = conn.execute("""
        SELECT candidate_id, content
        FROM text_unit
        WHERE content IS NOT NULL AND content != ''
        ORDER BY candidate_id
    """).fetchall()

    if not rows:
        raise ValueError("text_unit에 유효한 content가 없습니다.")

    documents  = [r[1] for r in rows]
    total_docs = len(documents)
    logger.info(f"GIDF 빌드 시작 — text_unit {total_docs}개")

    # ② 순수 계산 (DB 상태 변화 없음 — 실패해도 재시도 가능)
    terms, gidf_scores, doc_frequency = _compute_gidf(documents, cfg)
    total_terms = len(terms)
    logger.info(f"계산 완료 — 고유 어휘 {total_terms}개")

    # ③ 단일 트랜잭션으로 원자적 저장
    #    version INSERT 와 gidf INSERT 중 하나라도 실패하면 모두 롤백
    try:
        with conn.transaction():
            version_id: int = conn.execute("""
                INSERT INTO vocab_corpus_version
                    (total_docs, total_terms, min_df, max_df_ratio,
                     language_code, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING version_id
            """, (
                total_docs,
                total_terms,
                cfg.min_df,
                cfg.max_df,
                cfg.language_code,
                description,
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
        # with conn.transaction() 블록이 자동 롤백
        # → vocab_corpus_version 고아 레코드 발생하지 않음
        logger.error(f"GIDF 저장 실패 — 전체 롤백: {e}")
        raise


# ── 온라인 로드 ───────────────────────────────────────────────

def load_gidf(
    conn: psycopg.Connection,
    version_id: int | None = None,
) -> dict[str, float]:
    """
    온라인 요약 파이프라인에서 호출.
    version_id=None 이면 가장 최근 버전 자동 선택.

    Returns:
        {term: gidf_score}
    """
    if version_id is None:
        row = conn.execute("""
            SELECT version_id FROM vocab_corpus_version
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()
        if row is None:
            raise RuntimeError(
                "빌드된 GIDF 버전이 없습니다. build_and_store_gidf()를 먼저 실행하세요."
            )
        version_id = row[0]

    rows = conn.execute("""
        SELECT term, gidf_score
        FROM vocab_gidf
        WHERE version_id = %s
    """, (version_id,)).fetchall()

    logger.debug(f"GIDF 로드 완료 — version_id={version_id}, {len(rows)}개 어휘")
    return {term: score for term, score in rows}
```

### `lsa_scorer.py` 수정 — GIDF 사후 보정 적용

```python
import numpy as np
from .config import LSAConfig


def apply_gidf_boost(
    lsa_scores: dict[int, float],
    sentences: list[str],
    gidf: dict[str, float],
) -> dict[int, float]:
    """
    LSA 점수에 문장별 GIDF 보정 계수를 곱해 도메인 용어 포함 문장을 증폭.

    수식:
        G_bar_j = Σ count(i,j)×GIDF_i / |W_j|   (TF 가중 평균 GIDF)
        s~_j    = s_j^LSA × G_bar_j
    """
    boosted: dict[int, float] = {}
    for j, sentence in enumerate(sentences):
        tokens = sentence.lower().split()
        if not tokens:
            boosted[j] = lsa_scores[j]
            continue

        # count(i, j) : 문장 j에서 단어 i의 등장 횟수
        from collections import Counter
        tf = Counter(tokens)

        # G_bar_j = Σ count(i,j)×GIDF_i / |W_j|
        numerator   = sum(count * gidf.get(token, 0.0) for token, count in tf.items())
        denominator = len(tokens)   # |W_j| = 전체 토큰 수
        g_bar_j     = numerator / denominator if denominator > 0 else 1.0

        boosted[j] = lsa_scores[j] * g_bar_j

    return _minmax_normalize(boosted)
```

---

## 운영 시나리오

```python
from summarization.vocab_builder import build_and_store_gidf, load_gidf
from summarization.config import GIDFConfig

cfg = GIDFConfig(min_df=2, max_df=0.85, language_code="ko")

# ── 최초 구축 ────────────────────────────────────────────────
v1 = build_and_store_gidf(conn, cfg, description="초기 빌드 — text_unit 320개")
# → version_id = 1

# ── 문서 추가 후 재구축 ──────────────────────────────────────
v2 = build_and_store_gidf(conn, cfg, description="2차 빌드 — text_unit 450개")
# → version_id = 2  (v1은 이력으로 보존)

# ── 온라인 요약 — 최신 버전 자동 사용 ───────────────────────
gidf = load_gidf(conn)                  # version_id=2 자동 선택

# ── 특정 버전 지정 (A/B 비교, 롤백) ─────────────────────────
gidf_v1 = load_gidf(conn, version_id=1)
gidf_v2 = load_gidf(conn, version_id=2)

# ── 빌드 중 실패 → 재시도 ────────────────────────────────────
try:
    v3 = build_and_store_gidf(conn, cfg, description="3차 빌드")
except Exception:
    # 트랜잭션 자동 롤백 — DB에 고아 레코드 없음
    # 동일 cfg로 재호출하면 깨끗한 상태에서 재시도
    v3 = build_and_store_gidf(conn, cfg, description="3차 빌드 (재시도)")
```

---

## 구현 순서

1. DDL 적용 (`vocab_corpus_version`, `vocab_gidf` 테이블 생성)
2. `config.py`에 `GIDFConfig` 추가 및 `summarization.yaml`에 `gidf` 섹션 추가
3. `vocab_builder.py` 구현 — `_compute_gidf()`, `build_and_store_gidf()`, `load_gidf()`
4. 단위 테스트
   - 정상 빌드 후 `vocab_corpus_version` 레코드 1개, `vocab_gidf` N개 확인
   - `vocab_gidf` INSERT 도중 강제 예외 발생 시 `vocab_corpus_version`도 롤백 확인 (멱등성)
   - `load_gidf(version_id=None)`이 최신 버전을 반환하는지 확인
5. `lsa_scorer.py`에 `apply_gidf_boost()` 통합
6. `summarizer.py`에서 `gidf = load_gidf(conn)` 후 `apply_gidf_boost()` 호출 연결
