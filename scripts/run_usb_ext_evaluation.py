from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from summarization.benchmark.usb_ext import (
    run_usb_ext_evaluation,
    save_usb_ext_report,
    save_usb_ext_sentence_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run USB EXT sentence-level evaluation")
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
        default="test",
        help="USB EXT split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for the number of USB EXT documents to evaluate.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size used when saving threshold sweep metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/results/usb_ext",
        help="Directory where the USB EXT report and sentence scores will be written.",
    )
    parser.add_argument(
        "--gidf-path",
        default=None,
        help="Optional path for the temporary USB EXT GIDF artifact JSON.",
    )
    parser.add_argument(
        "--rebuild-gidf",
        action="store_true",
        help="Rebuild the USB EXT GIDF artifact even if the JSON already exists.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-download the USB EXT dataset before running the evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    output_dir = Path(args.output_dir)
    gidf_path = Path(args.gidf_path) if args.gidf_path else output_dir / "tmp" / "usb_ext_gidf.json"

    report, sentence_scores = run_usb_ext_evaluation(
        config_path=args.config,
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        force_prepare=args.force_prepare,
        threshold_step=args.threshold_step,
        gidf_path=gidf_path,
        rebuild_gidf=args.rebuild_gidf,
    )

    report_path = save_usb_ext_report(report, output_dir / "report.json")
    sentence_scores_path = save_usb_ext_sentence_scores(sentence_scores, output_dir / "sentence_scores.jsonl")

    paper_metrics = report["paper_comparison_metrics"]
    best_threshold = report["best_threshold_by_f1"]
    best_auc_threshold = report["best_threshold_by_binary_auc"]
    print(
        "[usb_ext] "
        f"documents={report['run']['documents_evaluated']} "
        f"sentences={report['run']['sentences_evaluated']} "
        f"auc={paper_metrics['auc']:.4f} "
        f"best_threshold={best_threshold['threshold']:.4f} "
        f"best_f1={best_threshold['f1']:.4f} "
        f"best_auc_threshold={best_auc_threshold['threshold']:.4f} "
        f"best_binary_auc={best_auc_threshold['binary_auc']:.4f} "
        f"gidf={report['gidf']['artifact_path']} "
        f"report={report_path} "
        f"sentence_scores={sentence_scores_path}"
    )


if __name__ == "__main__":
    main()
