from __future__ import annotations

import argparse
import sys
from pathlib import Path

from summarization.config import SummarizationConfig
from summarization.summarizer import HybridExtractiveSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid extractive summarization")
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to summarize. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--config",
        default="config/summarization.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else sys.stdin.read().strip()
    if not text:
        raise SystemExit("No input text provided.")

    cfg = SummarizationConfig.from_yaml(Path(args.config))
    summarizer = HybridExtractiveSummarizer(cfg)
    print(summarizer.summarize_one(text))


if __name__ == "__main__":
    main()
