# USB Benchmark Results

## Run Setup

- Task: `usb_ext` (`kundank/usb`, `extractive_summarization`)
- Split: `test`
- Config: `config/benchmark.yaml`
- Command:

```bash
uv run python scripts/run_usb_ext_evaluation.py \
  --data-dir data/benchmarks \
  --output-dir data/benchmarks/results/usb_ext
```

- Evaluated documents: `843`
- Evaluated sentences: `68,369`
- Sentence-level score dump: `data/benchmarks/results/usb_ext/sentence_scores.jsonl`
- Structured report: `data/benchmarks/results/usb_ext/report.json`

## Metric Definition

- USB EXT 논문 비교용 핵심 점수는 `continuous final_score`로 계산한 sentence-level `ROC AUC`이다.
- 따라서 `AUC` 자체는 threshold와 무관하다.
- Threshold sweep은 binary yes/no 운영점을 정하기 위한 용도로만 사용했다.
- 최종 threshold 선택 기준은 `F1` 최대값이다.

참고:

- USB repo evaluation code: <https://github.com/kukrishna/usb/blob/master/experiments/evaluate_all.py>
- USB paper: <https://aclanthology.org/2023.findings-emnlp.592.pdf>

## Overall Results

| Metric | Value |
|---|---:|
| Paper-compatible AUC | `0.5467` |
| Default rounded threshold | `0.5000` |
| Default precision | `0.1838` |
| Default recall | `0.6977` |
| Default F1 | `0.2909` |
| Default accuracy | `0.4409` |
| Best threshold by F1 | `0.4286672764` |
| Best-threshold precision | `0.1796` |
| Best-threshold recall | `0.8341` |
| Best-threshold F1 | `0.2955` |
| Best-threshold accuracy | `0.3462` |
| Selected sentences @ best threshold | `52,213` |
| Selected rate @ best threshold | `0.7637` |

### Threshold Notes

- Exact best threshold by F1: `0.4286672764`
- Stored `0.01` grid 기준 최고 F1 구간은 `0.42` ~ `0.43`였다.
- Grid top entries:

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| `0.42` | `0.1788` | `0.8461` | `0.2952` | `0.3358` |
| `0.43` | `0.1794` | `0.8309` | `0.2951` | `0.3475` |
| `0.41` | `0.1778` | `0.8592` | `0.2946` | `0.3237` |
| `0.45` | `0.1807` | `0.7967` | `0.2946` | `0.3726` |
| `0.44` | `0.1798` | `0.8135` | `0.2945` | `0.3592` |

## Domain Breakdown

| Domain | AUC | Best Threshold | Best F1 | Sentences |
|---|---:|---:|---:|---:|
| `schools` | `0.5841` | `0.5170` | `0.2624` | `4,274` |
| `landmarks` | `0.5810` | `0.4199` | `0.2772` | `3,228` |
| `disasters` | `0.5780` | `0.4328` | `0.2350` | `5,693` |
| `companies` | `0.5477` | `0.3788` | `0.2719` | `2,483` |
| `biographies` | `0.5393` | `0.4230` | `0.3094` | `50,379` |
| `newspapers` | `0.4733` | `0.1606` | `0.2507` | `2,312` |

## Runtime

이 값들은 실제 전체 `test` split 실행에서 생성된 `report.json`의 `run` 필드에 기록된 로그를 사용했다. 따라서 별도 재실행 없이 실제 측정값을 기록한다.

- Total elapsed time: `402.86 s`
- Wall-clock summary: 약 `6분 42.86초`
- Documents per second: `2.09`
- Sentences per second: `169.71`
- Average time per document: `0.4779 s`
- Average time per sentence: `0.00589 s` (`5.89 ms`)

## Comparison Note

- 현재 파이프라인은 USB supervision으로 fine-tuning된 classifier가 아니라, 하이브리드 extractive sentence scoring을 USB EXT 문장 선택 문제에 그대로 적용한 것이다.
- 그럼에도 이제 metric 정의는 USB 원본과 맞췄기 때문에 논문 수치와 직접 비교 가능한 형태가 되었다.
- 예를 들어 paper Table 4의 EXT AUC는 `Companies 66.36`, `Disasters 77.89`, `Schools 73.92`이고, 현재 결과는 각각 `54.77`, `57.80`, `58.41`이다.
- 따라서 지금 성능은 paper의 task-trained baselines보다 낮지만, 비교 기준은 동일하다.
