# Benchmark Evaluation Guide

## 1. 목적

이 문서는 현재 프로젝트의 `HybridExtractiveSummarizer`를 기준으로

- 벤치마크 데이터셋을 어떻게 준비하는지
- 어떤 명령어로 평가를 실행하는지
- 결과가 어디에 저장되는지
- 결과를 어떻게 읽고 해석해야 하는지

를 한 번에 정리한 실행 가이드입니다.

벤치마크 파이프라인의 핵심 코드는 아래 파일에 있습니다.

- 데이터셋 준비/정규화: `src/summarization/benchmark/datasets.py`
- 평가 지표 계산: `src/summarization/benchmark/metrics.py`
- 요약기 연결 및 리포트 생성: `src/summarization/benchmark/pipeline.py`
- 데이터셋 준비 CLI: `scripts/prepare_benchmark_datasets.py`
- 평가 실행 CLI: `scripts/run_benchmark.py`
- 벤치마크 실행 기본 설정: `config/benchmark.yaml`

---

## 2. 지원 데이터셋

### 2.1 CNN/DailyMail

- 소스: `abisee/cnn_dailymail`
- 로컬 저장 형식: Hugging Face parquet shard
- 기본 평가 split: `test`
- 입력 필드: `article`
- 정답 요약 필드: `highlights`
- 사용 지표: `ROUGE`, `extractive fragment diagnostics`

구성 특징:

- 뉴스 기사 본문과 하이라이트 요약 쌍으로 이루어진 대표적인 뉴스 요약 벤치마크입니다.
- 추출적 요약에서도 비교적 직관적으로 해석 가능한 데이터셋입니다.
- 기준 요약(`highlights`)이 비교적 짧아서, 예측 요약이 너무 길면 `precision`이 낮아지기 쉽습니다.

### 2.2 PubMed

- 원 요청 소스: `ncbi/pubmed`
- 로컬 저장 형식: NCBI baseline XML gzip shard
- 기본 평가 split: `train`
- 입력 필드: `MedlineCitation.Article.Abstract.AbstractText`
- 정답 필드: `MedlineCitation.Article.ArticleTitle`
- 사용 지표: `ROUGE`, `extractive fragment diagnostics`

구성 특징:

- 이 데이터셋은 전형적인 문서-요약 벤치마크가 아니라, PubMed citation baseline입니다.
- 현재 파이프라인에서는 `abstract -> title` 프록시(proxy) 평가로 사용합니다.
- 따라서 이 결과는 일반적인 long-document summarization 점수와 같은 의미로 해석하면 안 됩니다.
- 2026-03-25 기준, Hugging Face `datasets`는 이 dataset script를 직접 실행하지 못하므로 NCBI baseline XML을 직접 내려받는 방식으로 처리합니다.
- 같은 날짜 기준 NCBI baseline 인덱스는 `pubmed26...` shard를 제공하고 있으므로, 요청한 `2024` 구성이 없으면 최신 가용 연도로 자동 폴백합니다.

### 2.3 BookSum

- 소스: `kmfoda/booksum`
- 로컬 저장 형식: CSV
- 기본 평가 split: `test`
- 입력 필드: `chapter`
- 정답 요약 필드: `summary_text`
- 사용 지표: `ROUGE`, `extractive fragment diagnostics`, `novel n-gram ratios`

구성 특징:

- 장(chapter) 단위의 긴 입력과 서술형 요약으로 이루어진 long-form narrative summarization 데이터셋입니다.
- 참조 요약이 뉴스보다 훨씬 abstractive한 편이라, 추출적 요약기는 ROUGE가 상대적으로 낮게 나오는 것이 자연스럽습니다.
- 그래서 BookSum은 `ROUGE`만 보지 말고 `reference_novel_ngrams`와 `reference_extractiveness`를 반드시 같이 봐야 합니다.

### 2.4 USB EXT

- 소스: `kundank/usb`
- 로컬 저장 형식: Hugging Face `processed_data.tar.gz`에서 추출한 JSONL
- 기본 평가 split: `test`
- 입력 필드: `input_lines`
- 정답 필드: `labels == 1`인 source sentence 연결본
- 사용 지표: `ROUGE`, `extractive fragment diagnostics`

구성 특징:

- USB는 여러 summarization task를 함께 묶은 benchmark이고, 이 프로젝트는 그중 `EXT`(`extractive_summarization`)만 사용합니다.
- 원본 row는 자유형 summary 텍스트가 아니라 sentence-level importance label(`labels`)을 제공합니다.
- 따라서 현재 파이프라인은 `input_lines` 중 `labels == 1`인 문장들을 이어 붙여 참조 extractive summary로 변환한 뒤 평가합니다.
- `domain`은 `id`의 prefix(`companies/...`, `disasters/...`)에서 복원합니다.
- 현재 `datasets` 버전은 Hugging Face dataset script(`usb.py`)를 직접 실행하지 못하므로, `processed_data.tar.gz`를 직접 내려받아 `extractive_summarization/*.jsonl`만 추출하는 방식으로 처리합니다.

---

## 3. 데이터 저장 위치

데이터는 기본적으로 아래 구조로 저장됩니다.

```text
data/
└── benchmarks/
    ├── cnn_dailymail/
    │   ├── raw/
    │   ├── inspection.json
    │   └── manifest.json
    ├── pubmed/
    │   ├── raw/
    │   ├── inspection.json
    │   └── manifest.json
    ├── booksum/
    │   ├── raw/
    │   ├── inspection.json
    │   └── manifest.json
    ├── usb_ext/
    │   ├── raw/
    │   ├── inspection.json
    │   └── manifest.json
    └── results/
        ├── cnn_dailymail.json
        ├── pubmed.json
        ├── booksum.json
        └── usb_ext.json
```

파일 의미:

- `raw/`: 원본 데이터셋 파일
- `inspection.json`: 샘플 1개를 기준으로 정리한 구조 확인 결과
- `manifest.json`: 데이터셋별 설정, metric profile, raw 경로, PubMed 다운로드 범위 등의 메타데이터
- `results/<dataset>.json`: 실제 평가 실행 결과 리포트

---

## 4. 데이터셋 준비 방법

### 4.1 전체 준비

```bash
.venv/bin/python scripts/prepare_benchmark_datasets.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --pubmed-max-files 1
```

설명:

- `cnn_dailymail`, `booksum`은 원본 전체를 로컬로 내려받습니다.
- `pubmed`는 매우 큰 데이터셋이므로 처음에는 `--pubmed-max-files 1` 같은 제한으로 부트스트랩하는 것이 안전합니다.
- `usb_ext`는 `kundank/usb` 전체 task를 모두 풀지 않고 `extractive_summarization` split 파일만 추출합니다.

### 4.2 PubMed 전체 다운로드

```bash
.venv/bin/python scripts/prepare_benchmark_datasets.py \
  --datasets pubmed
```

주의:

- 이 경우 최신 가용 NCBI baseline shard 전체를 받게 됩니다.
- 다운로드 시간과 저장 공간 사용량이 큽니다.
- 실제 운영 전에는 먼저 `manifest.json`과 디스크 여유 공간을 확인하는 것이 좋습니다.

### 4.3 기존 데이터를 다시 받을 때

```bash
.venv/bin/python scripts/prepare_benchmark_datasets.py \
  --datasets cnn_dailymail booksum \
  --force
```

`--force`를 사용하면 기존 로컬 디렉터리를 지우고 다시 준비합니다.

---

## 5. 데이터 구조 확인 방법

데이터셋 준비가 끝나면 각 데이터셋마다 `inspection.json`이 생성됩니다.

예시:

- `data/benchmarks/cnn_dailymail/inspection.json`
- `data/benchmarks/pubmed/inspection.json`
- `data/benchmarks/booksum/inspection.json`
- `data/benchmarks/usb_ext/inspection.json`

`inspection.json`에는 아래 정보가 들어 있습니다.

- `raw_columns`: 원본 row 기준 컬럼 목록
- `raw_structure`: 샘플 row의 타입/preview
- `normalized_example`: 실제 벤치마크 파이프라인이 사용하는 `source`, `reference`, `metadata`

즉, 새로운 데이터셋을 붙이거나 필드 매핑을 검증할 때는 먼저 `inspection.json`을 보는 것이 가장 빠릅니다.

---

## 6. 벤치마크 실행 방법

### 6.1 기본 실행

```bash
.venv/bin/python scripts/run_benchmark.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --max-samples 100 \
  --batch-size 8 \
  --pubmed-max-files 1
```

설명:

- `--datasets`: 실행할 벤치마크 목록
- `--max-samples`: 각 데이터셋에서 평가할 최대 예제 수
- `--batch-size`: 기존 `summarize_batch`에 넘길 배치 크기
- `--pubmed-max-files`: PubMed 준비 단계에서 사용할 XML shard 제한

### 6.2 단일 데이터셋만 실행

```bash
.venv/bin/python scripts/run_benchmark.py \
  --datasets cnn_dailymail \
  --max-samples 500 \
  --batch-size 16
```

### 6.3 준비부터 다시 하고 실행

```bash
.venv/bin/python scripts/run_benchmark.py \
  --datasets booksum \
  --max-samples 50 \
  --batch-size 4 \
  --force-prepare
```

### 6.4 다른 설정 파일로 실행

```bash
.venv/bin/python scripts/run_benchmark.py \
  --datasets cnn_dailymail \
  --config config/benchmark.yaml
```

기본값은 `config/benchmark.yaml`입니다.

이 파일은 다음 목적을 가집니다.

- 로컬 환경에서 바로 실행되도록 `device: cpu` 사용
- 공개 접근 가능한 임베딩 모델 사용
- 기존 요약 알고리즘은 그대로 유지

---

## 7. 평가 결과 저장 위치

평가 결과는 기본적으로 아래에 저장됩니다.

```text
data/benchmarks/results/<dataset_name>.json
```

예:

- `data/benchmarks/results/cnn_dailymail.json`
- `data/benchmarks/results/pubmed.json`
- `data/benchmarks/results/booksum.json`
- `data/benchmarks/results/usb_ext.json`

결과 파일 구조는 크게 네 부분입니다.

### 7.1 `dataset`

데이터셋 이름, source id, split, metric profile, manifest 경로 등을 기록합니다.

### 7.2 `run`

실행 조건과 처리 속도 정보입니다.

- `config_path`
- `max_samples`
- `batch_size`
- `elapsed_seconds`
- `examples_per_second`

### 7.3 `metrics`

실제 핵심 평가 수치입니다.

- `length`
- `rouge`
- `prediction_extractiveness`
- `reference_extractiveness`
- `prediction_novel_ngrams`
- `reference_novel_ngrams`

데이터셋에 따라 일부 항목은 생략될 수 있습니다.

### 7.4 `predictions`

미리보기용 예시 출력입니다.

- `example_id`
- `prediction`
- `reference`
- `source_preview`
- `prediction_preview`
- `reference_preview`
- `metadata`

이 섹션은 점수만 봐서는 알기 어려운 오류를 빠르게 확인할 때 유용합니다.

---

## 8. 지표 해석 방법

### 8.1 `length`

주요 필드:

- `avg_source_tokens`
- `avg_reference_tokens`
- `avg_prediction_tokens`
- `avg_prediction_to_reference_ratio`

해석:

- `avg_prediction_tokens`가 지나치게 크면 추출 예산이 과도하다는 뜻일 수 있습니다.
- `avg_prediction_to_reference_ratio`가 1보다 훨씬 크면 참조 요약보다 예측이 길다는 의미입니다.
- CNN/DailyMail처럼 참조가 짧은 데이터셋에서는 이 값이 너무 크면 ROUGE precision이 떨어지기 쉽습니다.
- PubMed에서는 title proxy 평가이므로 이 값이 크면 거의 반드시 verbose한 출력입니다.

### 8.2 `rouge`

주요 필드:

- `rouge1`
- `rouge2`
- `rougeL`
- `rougeLsum`

각 지표에 대해 아래 3개가 있습니다.

- `precision`
- `recall`
- `f1`

해석:

- `precision`이 높다: 예측에 불필요한 내용이 적다
- `recall`이 높다: 참조 요약의 내용을 더 많이 덮는다
- `f1`이 높다: precision/recall 균형이 좋다

보통 비교용 대표 수치로는 `rougeL.f1` 또는 `rougeLsum.f1`을 많이 봅니다.

### 8.3 `prediction_extractiveness`

예측 요약이 원문에서 얼마나 직접적으로 복사된 문장 조각으로 구성되는지를 보여줍니다.

주요 필드:

- `coverage`
- `density`
- `compression`

해석:

- `coverage`가 높다: 예측 요약 토큰이 원문 연속 구간에서 많이 왔다
- `density`가 높다: 더 긴 연속 span을 그대로 가져왔다
- `compression`이 높다: 원문 대비 요약이 더 짧다

현재 시스템은 추출적 요약기이므로 보통 `prediction_extractiveness.coverage`는 높게 나오는 것이 자연스럽습니다.

### 8.4 `reference_extractiveness`

참조 요약이 원문에 대해 얼마나 extractive한지를 보여줍니다.

이 값은 데이터셋 자체의 난도를 읽는 데 중요합니다.

- `reference_extractiveness.coverage`가 높다:
  참조 요약도 원문 문장을 많이 재사용함
- `reference_extractiveness.coverage`가 낮다:
  참조 요약이 더 abstractive함

즉, 추출적 모델의 점수를 해석할 때는 `prediction`만 보지 말고 `reference`도 같이 봐야 합니다.

### 8.5 `novel_ngrams`

BookSum처럼 abstractive reference가 강한 데이터셋에서만 사용합니다.

주요 필드:

- `novel_1gram_ratio`
- `novel_2gram_ratio`
- `novel_3gram_ratio`

해석:

- 값이 높다: 원문에 없는 표현을 더 많이 사용했다
- 추출적 요약기의 `prediction_novel_ngrams`는 보통 낮아야 자연스럽습니다
- 대신 `reference_novel_ngrams`가 높게 나오면, 그 데이터셋에서 추출적 요약기의 상한선이 구조적으로 낮을 수 있음을 의미합니다

---

## 9. 데이터셋별 결과 해석 포인트

### 9.1 CNN/DailyMail

좋게 봐야 할 것:

- `rougeL.f1`
- `avg_prediction_to_reference_ratio`
- `prediction_extractiveness.coverage`

해석 포인트:

- 참조 요약이 짧기 때문에 예측이 길어지면 `precision`이 빨리 떨어집니다.
- `prediction_extractiveness.coverage`는 높고 `rouge`가 낮다면, 중요 문장 선택이 어긋났을 가능성이 큽니다.
- `avg_prediction_to_reference_ratio`가 너무 크면 `token_budget_ratio`를 줄여보는 것이 좋습니다.

### 9.2 PubMed

좋게 봐야 할 것:

- `rouge1`, `rougeL`
- `avg_prediction_to_reference_ratio`

해석 포인트:

- 현재 평가는 `abstract -> title` 프록시입니다.
- title은 매우 짧기 때문에, 긴 extractive summary는 구조적으로 불리합니다.
- 따라서 절대 점수보다 “너무 장황한지”를 먼저 봐야 합니다.
- `avg_prediction_to_reference_ratio`가 1에 가까울수록 title 길이에 더 근접한 출력입니다.

### 9.3 BookSum

좋게 봐야 할 것:

- `rougeL.f1`
- `reference_novel_ngrams`
- `reference_extractiveness.coverage`

해석 포인트:

- BookSum 참조 요약은 서술적으로 재구성된 경우가 많아, 추출적 모델이 본질적으로 불리합니다.
- 따라서 ROUGE가 뉴스보다 낮아도 이상하지 않습니다.
- `reference_novel_ngrams`가 높고 `reference_extractiveness.coverage`가 낮다면, 이 데이터셋은 본질적으로 abstractive 성향이 강하다는 뜻입니다.
- 이 경우 ROUGE만 보고 모델이 “망했다”고 판단하면 안 됩니다.

### 9.4 USB EXT

좋게 봐야 할 것:

- `rougeL.f1`
- `avg_prediction_to_reference_ratio`
- `prediction_extractiveness.coverage`

해석 포인트:

- 참조 요약은 human-written abstractive summary가 아니라, gold sentence subset을 이어 붙인 extractive reference입니다.
- 따라서 추출 모델이라면 `prediction_extractiveness.coverage`가 높게 나오는 것이 자연스럽습니다.
- `avg_prediction_to_reference_ratio`가 지나치게 크면 gold sentence subset보다 훨씬 긴 출력을 만들고 있다는 뜻입니다.
- `predictions[].metadata.domain`을 보면 어떤 Wikipedia 도메인에서 실패하는지 빠르게 확인할 수 있습니다.

---

## 10. 추천 분석 순서

평가 결과를 읽을 때는 아래 순서를 권장합니다.

1. `run.max_samples`와 `dataset.split`이 의도한 값인지 확인
2. `length.avg_prediction_to_reference_ratio`로 출력 길이부터 확인
3. `rougeL.f1`와 `rouge2.f1`를 보고 전반적인 품질 확인
4. `prediction_extractiveness`와 `reference_extractiveness`를 비교
5. BookSum이면 `reference_novel_ngrams`까지 확인
6. 마지막으로 `predictions` 샘플을 직접 읽어 qualitative sanity check 수행

---

## 11. 자주 보는 문제와 대응

### 11.1 PubMed가 Hugging Face에서 바로 안 열리는 경우

정상입니다.

- 현재 `ncbi/pubmed`는 dataset script 기반이라 최신 `datasets`에서 바로 실행되지 않습니다.
- 이 프로젝트는 NCBI XML 직접 다운로드 방식으로 우회합니다.

### 11.2 임베딩 모델 접근 오류가 나는 경우

벤치마크는 기본적으로 `config/benchmark.yaml`을 사용하세요.

이 설정은:

- `device: cpu`
- 공개 접근 가능한 `sentence-transformers/all-MiniLM-L6-v2`

를 사용하도록 잡혀 있습니다.

### 11.3 USB가 Hugging Face에서 바로 안 열리는 경우

정상입니다.

- `kundank/usb`는 dataset script(`usb.py`) 기반 배포본입니다.
- 현재 프로젝트의 `datasets` 버전은 이 스크립트를 직접 실행하지 못합니다.
- 이 프로젝트는 대신 `processed_data.tar.gz`를 직접 내려받고 `extractive_summarization` JSONL만 추출합니다.

### 11.4 속도가 너무 느린 경우

다음 순서로 조절하는 것이 안전합니다.

- `--max-samples`를 줄인다
- `--batch-size`를 줄이거나 늘려 본다
- `config/benchmark.yaml`의 `parallel.n_workers`를 조정한다
- `extraction.token_budget_ratio`를 줄여 출력 길이를 줄인다

---

## 12. 빠른 시작 예시

### 최소 실행

```bash
.venv/bin/python scripts/prepare_benchmark_datasets.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --pubmed-max-files 1

.venv/bin/python scripts/run_benchmark.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --max-samples 10 \
  --batch-size 4 \
  --pubmed-max-files 1
```

### 결과 확인

```bash
ls data/benchmarks/results
cat data/benchmarks/results/cnn_dailymail.json
```

이 순서대로 보면, 데이터 준비부터 결과 해석까지 한 흐름으로 확인할 수 있습니다.
