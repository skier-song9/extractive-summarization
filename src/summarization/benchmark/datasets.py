from __future__ import annotations

import gzip
import json
import re
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

from ..utils import get_logger

PUBMED_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
USB_HF_ARCHIVE_NAME = "processed_data.tar.gz"
USB_EXT_TASK_NAME = "extractive_summarization"
USB_EXT_SPLITS = ("train", "validation", "test")


@dataclass(slots=True, frozen=True)
class BenchmarkDatasetSpec:
    name: str
    source_id: str
    config: str | None
    backend: Literal["hf_parquet", "hf_csv", "pubmed_xml", "usb_jsonl"]
    default_split: str
    source_path: tuple[str, ...]
    reference_path: tuple[str, ...]
    id_path: tuple[str, ...] | None
    metric_profile: tuple[str, ...]
    description: str
    allow_patterns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class BenchmarkExample:
    dataset_name: str
    split: str
    example_id: str
    source: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


DATASET_SPECS: dict[str, BenchmarkDatasetSpec] = {
    "cnn_dailymail": BenchmarkDatasetSpec(
        name="cnn_dailymail",
        source_id="abisee/cnn_dailymail",
        config="3.0.0",
        backend="hf_parquet",
        default_split="test",
        source_path=("article",),
        reference_path=("highlights",),
        id_path=("id",),
        metric_profile=("rouge", "extractive_fragments"),
        description="English news summarization benchmark with article/highlights pairs.",
        allow_patterns=("3.0.0/*.parquet", "README.md"),
        notes=(
            "Uses Hugging Face parquet shards for config 3.0.0.",
            "Recommended evaluation uses ROUGE plus extractive-fragment diagnostics.",
        ),
    ),
    "pubmed": BenchmarkDatasetSpec(
        name="pubmed",
        source_id="ncbi/pubmed",
        config="2024",
        backend="pubmed_xml",
        default_split="train",
        source_path=("MedlineCitation", "Article", "Abstract", "AbstractText"),
        reference_path=("MedlineCitation", "Article", "ArticleTitle"),
        id_path=("MedlineCitation", "PMID"),
        metric_profile=("rouge", "extractive_fragments"),
        description="PubMed citation baseline data from NCBI 2024 XML files.",
        notes=(
            "The Hugging Face dataset uses a dataset script that current datasets releases do not execute.",
            "This pipeline downloads the raw NCBI baseline XML shards directly and uses abstract-to-title as a proxy evaluation pair.",
        ),
    ),
    "booksum": BenchmarkDatasetSpec(
        name="booksum",
        source_id="kmfoda/booksum",
        config="default",
        backend="hf_csv",
        default_split="test",
        source_path=("chapter",),
        reference_path=("summary_text",),
        id_path=("summary_id",),
        metric_profile=("rouge", "extractive_fragments", "novel_ngrams"),
        description="Long-form narrative summarization benchmark with chapter/summary pairs.",
        allow_patterns=("train.csv", "dev.csv", "test.csv", "README.md", "LICENSE.txt"),
        notes=(
            "Uses chapter as the source document and summary_text as the reference summary.",
            "Adds novel n-gram diagnostics because BookSum references are more abstractive than news summaries.",
        ),
    ),
    "usb_ext": BenchmarkDatasetSpec(
        name="usb_ext",
        source_id="kundank/usb",
        config=USB_EXT_TASK_NAME,
        backend="usb_jsonl",
        default_split="test",
        source_path=("input_lines",),
        reference_path=("labels",),
        id_path=("id",),
        metric_profile=("rouge", "extractive_fragments"),
        description="USB benchmark EXT task with sentence-level extractive labels across multiple Wikipedia domains.",
        notes=(
            "Downloads the processed USB archive from Hugging Face and extracts only the extractive_summarization task files.",
            "Builds the reference summary by concatenating source sentences whose EXT labels are 1.",
        ),
    ),
}


def list_dataset_names() -> list[str]:
    return sorted(DATASET_SPECS)


def get_dataset_spec(name: str) -> BenchmarkDatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark dataset: {name}") from exc


def dataset_dir(data_dir: str | Path, dataset_name: str) -> Path:
    return Path(data_dir) / dataset_name


def dataset_raw_dir(data_dir: str | Path, dataset_name: str) -> Path:
    return dataset_dir(data_dir, dataset_name) / "raw"


def dataset_inspection_path(data_dir: str | Path, dataset_name: str) -> Path:
    return dataset_dir(data_dir, dataset_name) / "inspection.json"


def dataset_manifest_path(data_dir: str | Path, dataset_name: str) -> Path:
    return dataset_dir(data_dir, dataset_name) / "manifest.json"


def prepare_dataset(
    dataset_name: str,
    data_dir: str | Path = "data/benchmarks",
    *,
    force: bool = False,
    pubmed_max_files: int | None = None,
    download_workers: int = 8,
) -> Path:
    spec = get_dataset_spec(dataset_name)
    target_dir = dataset_dir(data_dir, dataset_name)
    raw_dir = dataset_raw_dir(data_dir, dataset_name)
    logger = get_logger(f"benchmark.prepare.{dataset_name}", "INFO")

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    if spec.backend in {"hf_parquet", "hf_csv"}:
        logger.info("Downloading %s into %s", spec.source_id, raw_dir)
        snapshot_download(
            repo_id=spec.source_id,
            repo_type="dataset",
            local_dir=raw_dir,
            allow_patterns=list(spec.allow_patterns),
            force_download=force,
            max_workers=max(1, download_workers),
        )
    elif spec.backend == "pubmed_xml":
        downloaded, actual_year_suffix = _download_pubmed_baseline(
            raw_dir,
            spec.config or "2024",
            force=force,
            max_files=pubmed_max_files,
        )
        logger.info("Prepared %d PubMed baseline shard(s) in %s", downloaded, raw_dir)
    else:
        extracted = _download_usb_ext_dataset(raw_dir, force=force)
        logger.info("Prepared %d USB EXT split file(s) in %s", extracted, raw_dir)

    inspection = inspect_dataset(dataset_name, data_dir=data_dir)
    manifest = {
        "dataset_name": spec.name,
        "source_id": spec.source_id,
        "config": spec.config,
        "backend": spec.backend,
        "default_split": spec.default_split,
        "raw_dir": str(raw_dir),
        "inspection_path": str(dataset_inspection_path(data_dir, dataset_name)),
        "metric_profile": list(spec.metric_profile),
        "notes": list(spec.notes),
    }
    if spec.backend == "pubmed_xml" and pubmed_max_files is not None:
        manifest["download_scope"] = {
            "type": "partial",
            "pubmed_max_files": pubmed_max_files,
        }
        manifest["resolved_pubmed_year_suffix"] = actual_year_suffix
    elif spec.backend == "pubmed_xml":
        manifest["download_scope"] = {"type": "full"}
        manifest["resolved_pubmed_year_suffix"] = actual_year_suffix
    elif spec.backend == "usb_jsonl":
        manifest["download_scope"] = {
            "type": "task_only",
            "task": spec.config,
            "splits": list(USB_EXT_SPLITS),
        }
        manifest["archive_filename"] = USB_HF_ARCHIVE_NAME

    manifest_path = dataset_manifest_path(data_dir, dataset_name)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_dir


def inspect_dataset(dataset_name: str, data_dir: str | Path = "data/benchmarks") -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    split = spec.default_split
    raw_example = next(iter_raw_rows(dataset_name, data_dir=data_dir, split=split))
    normalized = normalize_example(spec, raw_example, split=split)

    inspection = {
        "dataset_name": spec.name,
        "source_id": spec.source_id,
        "config": spec.config,
        "backend": spec.backend,
        "inspected_split": split,
        "description": spec.description,
        "metric_profile": list(spec.metric_profile),
        "notes": list(spec.notes),
        "raw_columns": list(raw_example) if isinstance(raw_example, dict) else [],
        "raw_structure": summarize_structure(raw_example),
        "normalized_example": {
            "example_id": normalized.example_id,
            "source_chars": len(normalized.source),
            "reference_chars": len(normalized.reference),
            "source_preview": normalized.source[:400],
            "reference_preview": normalized.reference[:400],
            "metadata": normalized.metadata,
        },
    }
    inspection_path = dataset_inspection_path(data_dir, dataset_name)
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_path.write_text(json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8")
    return inspection


def iter_benchmark_examples(
    dataset_name: str,
    data_dir: str | Path = "data/benchmarks",
    *,
    split: str | None = None,
    max_samples: int | None = None,
) -> list[BenchmarkExample]:
    spec = get_dataset_spec(dataset_name)
    resolved_split = split or spec.default_split
    rows = iter_raw_rows(dataset_name, data_dir=data_dir, split=resolved_split)
    examples: list[BenchmarkExample] = []
    for row in rows:
        example = normalize_example(spec, row, split=resolved_split)
        if not example.source or not example.reference:
            continue
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def iter_raw_rows(
    dataset_name: str,
    *,
    data_dir: str | Path = "data/benchmarks",
    split: str | None = None,
):
    spec = get_dataset_spec(dataset_name)
    raw_dir = dataset_raw_dir(data_dir, dataset_name)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' has not been prepared yet. "
            f"Run the prepare script first or call prepare_dataset()."
        )

    resolved_split = split or spec.default_split
    if spec.backend == "hf_parquet":
        yield from _iter_cnn_dailymail_rows(raw_dir, spec.config or "3.0.0", resolved_split)
        return
    if spec.backend == "hf_csv":
        yield from _iter_booksum_rows(raw_dir, resolved_split)
        return
    if spec.backend == "usb_jsonl":
        yield from _iter_usb_ext_rows(raw_dir, resolved_split)
        return
    if resolved_split != "train":
        raise ValueError("PubMed only provides a train split in this pipeline.")
    yield from _iter_pubmed_rows(raw_dir)


def normalize_example(spec: BenchmarkDatasetSpec, row: dict[str, Any], *, split: str) -> BenchmarkExample:
    if spec.name == "usb_ext":
        source_lines = _normalize_usb_lines(row.get("input_lines"))
        reference_lines = _select_usb_reference_lines(source_lines, row.get("labels"))
        example_id_raw = row.get("id")
        domain = row.get("domain") or _infer_usb_domain(example_id_raw)
        example_id = str(example_id_raw) if example_id_raw not in {None, ""} else f"{spec.name}:{hash(tuple(source_lines))}"
        return BenchmarkExample(
            dataset_name=spec.name,
            split=split,
            example_id=example_id,
            source=normalize_text(source_lines),
            reference=normalize_text(reference_lines),
            metadata={
                "domain": domain,
                "source_sentence_count": len(source_lines),
                "selected_sentence_count": len(reference_lines),
            },
        )

    source = normalize_text(extract_nested_value(row, spec.source_path))
    reference = normalize_text(extract_nested_value(row, spec.reference_path))
    example_id_raw = extract_nested_value(row, spec.id_path) if spec.id_path else None

    metadata: dict[str, Any]
    if spec.name == "cnn_dailymail":
        metadata = {"id": str(row.get("id", ""))}
    elif spec.name == "booksum":
        metadata = {
            "book_id": row.get("book_id"),
            "source": row.get("source"),
            "is_aggregate": row.get("is_aggregate"),
            "summary_name": row.get("summary_name"),
        }
    else:
        article = row.get("MedlineCitation", {}).get("Article", {})
        metadata = {
            "pmid": str(row.get("MedlineCitation", {}).get("PMID", "")),
            "language": article.get("Language", ""),
        }

    if example_id_raw in {None, ""}:
        example_id = f"{spec.name}:{hash((source, reference))}"
    else:
        example_id = str(example_id_raw)

    return BenchmarkExample(
        dataset_name=spec.name,
        split=split,
        example_id=example_id,
        source=source,
        reference=reference,
        metadata=metadata,
    )


def summarize_structure(value: Any, *, max_depth: int = 3) -> Any:
    if max_depth <= 0:
        return {"type": type(value).__name__}

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value),
            "sample": {key: summarize_structure(child, max_depth=max_depth - 1) for key, child in islice(value.items(), 10)},
        }
    if isinstance(value, list):
        sample = value[0] if value else None
        return {
            "type": "list",
            "length": len(value),
            "sample": summarize_structure(sample, max_depth=max_depth - 1) if sample is not None else None,
        }
    if isinstance(value, str):
        return {
            "type": "str",
            "chars": len(value),
            "preview": value[:200],
        }
    if value is None:
        return {"type": "null"}
    return {
        "type": type(value).__name__,
        "value": value,
    }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(normalize_text(item) for item in value if item is not None)
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def extract_nested_value(row: dict[str, Any], path: tuple[str, ...] | None) -> Any:
    if not path:
        return None

    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _normalize_usb_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        normalized = normalize_text(value)
        return [normalized] if normalized else []

    lines: list[str] = []
    for item in value:
        normalized = normalize_text(item)
        if normalized:
            lines.append(normalized)
    return lines


def _select_usb_reference_lines(source_lines: list[str], labels: Any) -> list[str]:
    if not isinstance(labels, list):
        return []

    selected: list[str] = []
    for line, label in zip(source_lines, labels, strict=False):
        if _coerce_usb_label(label):
            selected.append(line)
    return selected


def _coerce_usb_label(label: Any) -> bool:
    try:
        return int(label) == 1
    except (TypeError, ValueError):
        return False


def _infer_usb_domain(example_id: Any) -> str:
    if not isinstance(example_id, str):
        return ""
    prefix, _, _ = example_id.partition("/")
    return prefix


def _iter_cnn_dailymail_rows(raw_dir: Path, config: str, split: str):
    split_files = sorted((raw_dir / config).glob(f"{split}-*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"No parquet shards found for split '{split}' in {raw_dir / config}")

    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in split_files]},
        split=split,
        streaming=True,
    )
    yield from dataset


def _iter_booksum_rows(raw_dir: Path, split: str):
    file_mapping = {
        "train": raw_dir / "train.csv",
        "validation": raw_dir / "dev.csv",
        "test": raw_dir / "test.csv",
    }
    if split not in file_mapping:
        raise ValueError(f"Unsupported BookSum split: {split}")
    path = file_mapping[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing BookSum split file: {path}")

    dataset = load_dataset(
        "csv",
        data_files={split: str(path)},
        split=split,
        streaming=True,
    )
    yield from dataset


def _iter_pubmed_rows(raw_dir: Path):
    xml_paths = sorted(raw_dir.glob("pubmed*.xml.gz"))
    if not xml_paths:
        raise FileNotFoundError(f"No PubMed XML shards found in {raw_dir}")

    for path in xml_paths:
        yield from _iter_pubmed_rows_from_file(path)


def _iter_usb_ext_rows(raw_dir: Path, split: str):
    path = raw_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing USB EXT split file: {path}")

    dataset = load_dataset(
        "json",
        data_files={split: str(path)},
        split=split,
        streaming=True,
    )
    yield from dataset


def _iter_pubmed_rows_from_file(path: Path):
    with gzip.open(path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            pmid = _extract_xml_text(elem.find("./MedlineCitation/PMID"))
            article_title = _extract_xml_text(elem.find("./MedlineCitation/Article/ArticleTitle"))
            abstract_nodes = elem.findall("./MedlineCitation/Article/Abstract/AbstractText")
            abstract_parts: list[str] = []
            for abstract_node in abstract_nodes:
                abstract_text = _extract_xml_text(abstract_node)
                if not abstract_text:
                    continue
                label = normalize_text(abstract_node.attrib.get("Label", ""))
                abstract_parts.append(f"{label}: {abstract_text}" if label else abstract_text)

            if pmid and article_title and abstract_parts:
                yield {
                    "MedlineCitation": {
                        "PMID": pmid,
                        "Article": {
                            "ArticleTitle": article_title,
                            "Abstract": {"AbstractText": " ".join(abstract_parts)},
                            "Language": _extract_xml_text(elem.find("./MedlineCitation/Article/Language")),
                        },
                    }
                }

            elem.clear()


def _extract_xml_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return normalize_text("".join(elem.itertext()))


def _download_pubmed_baseline(raw_dir: Path, config: str, *, force: bool, max_files: int | None) -> tuple[int, str]:
    year_suffix = str(config)[-2:]
    available_files, resolved_year_suffix = _list_pubmed_baseline_files(year_suffix)
    selected_files = available_files[:max_files] if max_files is not None else available_files

    downloaded = 0
    for filename in selected_files:
        destination = raw_dir / filename
        if destination.exists() and not force:
            downloaded += 1
            continue

        with urllib.request.urlopen(f"{PUBMED_BASE_URL}{filename}") as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        downloaded += 1
    return downloaded, resolved_year_suffix


def _download_usb_ext_dataset(raw_dir: Path, *, force: bool) -> int:
    archive_path = Path(
        hf_hub_download(
            repo_id="kundank/usb",
            repo_type="dataset",
            filename=USB_HF_ARCHIVE_NAME,
        )
    )

    extracted = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for split in USB_EXT_SPLITS:
            member_name = f"{USB_EXT_TASK_NAME}/{split}.jsonl"
            destination = raw_dir / f"{split}.jsonl"
            if destination.exists() and not force:
                extracted += 1
                continue

            member = tar.getmember(member_name)
            source_handle = tar.extractfile(member)
            if source_handle is None:
                raise FileNotFoundError(f"Could not extract {member_name} from {archive_path}")

            with source_handle:
                with destination.open("wb") as handle:
                    shutil.copyfileobj(source_handle, handle)
            extracted += 1
    return extracted


def _list_pubmed_baseline_files(year_suffix: str) -> tuple[list[str], str]:
    with urllib.request.urlopen(PUBMED_BASE_URL) as response:
        html = response.read().decode("utf-8", errors="ignore")

    requested_pattern = re.compile(rf"(pubmed{re.escape(year_suffix)}n\d{{4}}\.xml\.gz)")
    requested_matches = sorted(set(requested_pattern.findall(html)))
    if requested_matches:
        return requested_matches, year_suffix

    available_years = sorted(set(re.findall(r"pubmed(\d{2})n\d{4}\.xml\.gz", html)))
    if not available_years:
        raise FileNotFoundError("Could not discover any PubMed baseline XML shards from the NCBI index page.")

    fallback_year_suffix = available_years[-1]
    fallback_pattern = re.compile(rf"(pubmed{re.escape(fallback_year_suffix)}n\d{{4}}\.xml\.gz)")
    fallback_matches = sorted(set(fallback_pattern.findall(html)))
    return fallback_matches, fallback_year_suffix
