from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

from summarization.benchmark.datasets import (
    BenchmarkExample,
    _iter_pubmed_rows_from_file,
    get_dataset_spec,
    iter_benchmark_examples,
    normalize_example,
    prepare_dataset,
)


def test_normalize_example_maps_cnn_dailymail_fields() -> None:
    spec = get_dataset_spec("cnn_dailymail")
    row = {
        "article": "Sentence one. Sentence two.",
        "highlights": "Sentence one.",
        "id": "cnn-1",
    }

    example = normalize_example(spec, row, split="test")

    assert isinstance(example, BenchmarkExample)
    assert example.example_id == "cnn-1"
    assert example.source == "Sentence one. Sentence two."
    assert example.reference == "Sentence one."


def test_normalize_example_maps_booksum_fields() -> None:
    spec = get_dataset_spec("booksum")
    row = {
        "chapter": "Long chapter text.",
        "summary_text": "Short summary.",
        "summary_id": "chapter-2",
        "book_id": "book-1",
        "source": "cliffnotes",
        "is_aggregate": True,
        "summary_name": "Chapters 1-2",
    }

    example = normalize_example(spec, row, split="test")

    assert example.example_id == "chapter-2"
    assert example.source == "Long chapter text."
    assert example.reference == "Short summary."
    assert example.metadata["book_id"] == "book-1"


def test_normalize_example_maps_usb_ext_labels_to_reference() -> None:
    spec = get_dataset_spec("usb_ext")
    row = {
        "id": "disasters/example-1.json",
        "input_lines": [
            "Document title",
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
        ],
        "labels": [1, 0, 1, 0],
    }

    example = normalize_example(spec, row, split="test")

    assert example.example_id == "disasters/example-1.json"
    assert example.source == "Document title Sentence one. Sentence two. Sentence three."
    assert example.reference == "Document title Sentence two."
    assert example.metadata["domain"] == "disasters"
    assert example.metadata["selected_sentence_count"] == 2


def test_prepare_dataset_extracts_usb_ext_files(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "processed_data.tar.gz"
    rows = {
        "train": {
            "id": "companies/train-1.json",
            "input_lines": ["Train title", "Train body."],
            "labels": [1, 0],
        },
        "validation": {
            "id": "schools/validation-1.json",
            "input_lines": ["Validation title", "Validation body."],
            "labels": [1, 1],
        },
        "test": {
            "id": "disasters/test-1.json",
            "input_lines": ["Test title", "Sentence one.", "Sentence two."],
            "labels": [1, 0, 1],
        },
    }

    with tarfile.open(archive_path, "w:gz") as tar:
        for split, row in rows.items():
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            info = tarfile.TarInfo(name=f"extractive_summarization/{split}.jsonl")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        ignored_payload = b'{"id":"ignored"}\n'
        ignored_info = tarfile.TarInfo(name="topicbased_summarization/test.jsonl")
        ignored_info.size = len(ignored_payload)
        tar.addfile(ignored_info, io.BytesIO(ignored_payload))

    monkeypatch.setattr("summarization.benchmark.datasets.hf_hub_download", lambda **kwargs: str(archive_path))

    target_dir = prepare_dataset("usb_ext", data_dir=tmp_path)

    assert target_dir == tmp_path / "usb_ext"
    raw_dir = target_dir / "raw"
    assert (raw_dir / "train.jsonl").exists()
    assert (raw_dir / "validation.jsonl").exists()
    assert (raw_dir / "test.jsonl").exists()
    assert not (raw_dir / "topicbased_summarization").exists()

    examples = iter_benchmark_examples("usb_ext", data_dir=tmp_path, split="test", max_samples=5)

    assert len(examples) == 1
    assert examples[0].example_id == "disasters/test-1.json"
    assert examples[0].reference == "Test title Sentence two."


def test_iter_pubmed_rows_from_file_extracts_title_and_abstract(tmp_path: Path) -> None:
    xml = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Sample title</ArticleTitle>
        <Abstract>
          <AbstractText Label="Background">First abstract sentence.</AbstractText>
          <AbstractText>Second abstract sentence.</AbstractText>
        </Abstract>
        <Language>eng</Language>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
    path = tmp_path / "pubmed24n0001.xml.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(xml.encode("utf-8"))

    rows = list(_iter_pubmed_rows_from_file(path))

    assert len(rows) == 1
    row = rows[0]
    assert row["MedlineCitation"]["PMID"] == "12345"
    assert row["MedlineCitation"]["Article"]["ArticleTitle"] == "Sample title"
    assert row["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] == (
        "Background: First abstract sentence. Second abstract sentence."
    )
