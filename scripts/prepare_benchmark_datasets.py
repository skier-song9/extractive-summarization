from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from summarization.benchmark import list_dataset_names, prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_dataset_names(),
        choices=list_dataset_names(),
        help="Benchmark datasets to prepare.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/benchmarks",
        help="Directory where raw benchmark artifacts will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any existing local dataset directory before downloading again.",
    )
    parser.add_argument(
        "--pubmed-max-files",
        type=int,
        default=None,
        help="Optional limit for the number of PubMed XML shards to download.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    for dataset_name in args.datasets:
        target_dir = prepare_dataset(
            dataset_name,
            data_dir=args.data_dir,
            force=args.force,
            pubmed_max_files=args.pubmed_max_files,
        )
        print(f"[prepared] {dataset_name}: {target_dir}")


if __name__ == "__main__":
    main()
