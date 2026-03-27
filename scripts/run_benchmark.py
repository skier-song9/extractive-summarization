from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from summarization.benchmark import list_dataset_names, run_benchmark, save_benchmark_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractive summarization benchmarks")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_dataset_names(),
        choices=list_dataset_names(),
        help="Benchmark datasets to run.",
    )
    parser.add_argument(
        "--config",
        default="config/benchmark.yaml",
        help="Summarizer config file.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/benchmarks",
        help="Prepared benchmark dataset directory.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split override. Defaults to the dataset's recommended evaluation split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of benchmark examples to evaluate per dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size passed to summarize_batch.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/results",
        help="Directory where benchmark reports will be written.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-download the requested dataset(s) before running the benchmark.",
    )
    parser.add_argument(
        "--pubmed-max-files",
        type=int,
        default=None,
        help="Optional limit for the number of PubMed XML shards to use.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    for dataset_name in args.datasets:
        report = run_benchmark(
            dataset_name,
            config_path=args.config,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            force_prepare=args.force_prepare,
            pubmed_max_files=args.pubmed_max_files,
        )
        output_path = Path(args.output_dir) / f"{dataset_name}.json"
        save_benchmark_report(report, output_path)
        rouge = report["metrics"].get("rouge", {})
        rouge_l = rouge.get("rougeL", {})
        print(
            f"[benchmark] {dataset_name}: "
            f"examples={report['run']['max_samples']} "
            f"rougeL_f1={rouge_l.get('f1', 0.0):.4f} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
