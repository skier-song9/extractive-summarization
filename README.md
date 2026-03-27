# Extractive Summarization

## Benchmark Datasets

벤치마크용 데이터셋 준비:

```bash
uv run python scripts/prepare_benchmark_datasets.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --pubmed-max-files 1
```

- 저장 위치: `data/benchmarks/<dataset_name>/raw`
- 구조 확인 결과: `data/benchmarks/<dataset_name>/inspection.json`
- 데이터셋 메타데이터: `data/benchmarks/<dataset_name>/manifest.json`

기본적으로 다음 매핑을 사용합니다.

- `cnn_dailymail`: `article -> highlights`
- `booksum`: `chapter -> summary_text`
- `pubmed`: `AbstractText -> ArticleTitle`
- `usb_ext`: `input_lines + labels -> selected source sentences`

`pubmed`는 현재 Hugging Face `datasets`에서 원격 스크립트를 바로 실행할 수 없어서 NCBI baseline XML을 직접 내려받습니다. 2026-03-25 기준 NCBI baseline 인덱스에는 `pubmed26...` 샤드가 노출되어 있어, 요청한 `2024` 구성이 없으면 최신 가용 연도로 자동 폴백합니다.
`usb_ext`도 현재 프로젝트의 `datasets` 버전에서는 Hugging Face dataset script(`usb.py`)를 직접 실행할 수 없어서, `kundank/usb`의 `processed_data.tar.gz`를 내려받은 뒤 `extractive_summarization/*.jsonl`만 추출해 사용합니다.

## Benchmark Run

CPU 기준 벤치마크 실행:

```bash
uv run python scripts/run_benchmark.py \
  --datasets cnn_dailymail booksum pubmed usb_ext \
  --max-samples 100 \
  --batch-size 8 \
  --pubmed-max-files 1
```

- 기본 실행 설정: `config/benchmark.yaml`
- 결과 리포트: `data/benchmarks/results/<dataset_name>.json`

전체 `test` split 평가 예시:

```bash
uv run python scripts/run_benchmark.py \
  --datasets cnn_dailymail \
  --max-samples 11490
```

```bash
uv run python scripts/run_benchmark.py \
  --datasets booksum \
  --max-samples 1428
```

```bash
uv run python scripts/run_benchmark.py \
  --datasets usb_ext \
  --max-samples 500
```

평가 지표는 데이터셋별로 다음 프로파일을 사용합니다.

- `cnn_dailymail`: `ROUGE` + extractive fragment diagnostics
- `booksum`: `ROUGE` + extractive fragment diagnostics + novel n-gram ratios
- `pubmed`: `ROUGE` + extractive fragment diagnostics
- `usb_ext`: `ROUGE` + extractive fragment diagnostics

## USB EXT Sentence-Level Eval

USB EXT는 summary string보다 문장별 score가 더 중요하므로, 별도 전용 평가 스크립트를 사용합니다.

```bash
uv run python scripts/run_usb_ext_evaluation.py \
  --data-dir data/benchmarks \
  --output-dir data/benchmarks/results/usb_ext
```

- sentence-level 점수 저장: `data/benchmarks/results/usb_ext/sentence_scores.jsonl`
- 전용 리포트 저장: `data/benchmarks/results/usb_ext/report.json`
- 논문 비교용 핵심 수치: `paper_comparison_metrics.auc`
- threshold 운영점 선택: `best_threshold_by_f1`

## Existing Integration Test

실제 PostgreSQL DB에서 `text_unit` 300개를 읽어오는 integration test 실행:

```bash
/usr/bin/time -p uv run pytest -q tests/integration/test_text_unit_data_preparation.py -rA
```

이 테스트는 `tests/conftest.py`의 `live_text_unit_contents` fixture를 통해 `load_text_unit_contents(limit=300)`를 호출합니다. DB 연결 정보는 프로젝트의 `.env` 기준으로 로드됩니다.
