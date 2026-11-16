"""Gold ID metrics and baseline integrity; tests require no models or API keys."""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.id_baseline import (
    CORPUS_PATH,
    DATASET_PATH,
    evaluate_id_cases,
    faq_text,
    load_labels,
    model_fingerprint,
)
from src.evaluation.metrics import calculate_id_metrics


def test_id_metrics_use_all_gold_documents_and_fixed_k():
    metrics = calculate_id_metrics(["wrong", "a"], ["a", "b"], 5)
    assert metrics["hit@5"] == 1
    assert metrics["recall@5"] == 0.5
    assert metrics["precision@5"] == 0.2
    assert metrics["mrr@5"] == 0.5
    assert metrics["ndcg@5"] == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_duplicates_do_not_inflate_recall_or_change_ranks():
    metrics = calculate_id_metrics(["a", "a", "b", "c"], ["a", "b", "c", "d"], 3)
    assert metrics["recall@3"] == 0.5
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["ndcg@3"] == pytest.approx(1.5 / (1 + 1 / math.log2(3) + 0.5))


def test_empty_results_and_cutoff():
    assert all(value == 0 for value in calculate_id_metrics([], ["a"]).values())
    assert calculate_id_metrics(["x"] * 5 + ["a"], ["a"])["hit@5"] == 0


@pytest.mark.parametrize("gold,k", [([], 5), ([""], 5), (["a"], 0)])
def test_invalid_metric_inputs_fail(gold, k):
    with pytest.raises(ValueError):
        calculate_id_metrics([], gold, k)


def test_frozen_labels_and_original_text_are_preserved():
    cases, corpus = load_labels(DATASET_PATH, CORPUS_PATH)
    assert len(cases) == 49 and len(corpus) == 60
    assert sum(case["retrieval_status"] == "answerable" for case in cases) == 34
    assert corpus[0]["faq_id"] not in faq_text(corpus[0], 1)
    assert corpus[0]["question"] in faq_text(corpus[0], 1)
    assert corpus[0]["answer"] in faq_text(corpus[0], 1)


def test_evaluator_roundtrip_preserves_labels(tmp_path):
    evaluator = RetrievalEvaluator()
    cases = evaluator.load_dataset(DATASET_PATH)
    path = tmp_path / "cases.json"
    evaluator.save_dataset(cases, path)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "mutation", ["unknown", "duplicate", "missing", "status", "empty"]
)
def test_bad_labels_are_rejected(tmp_path, mutation):
    cases, _ = load_labels(DATASET_PATH, CORPUS_PATH)
    if mutation == "unknown":
        cases[0]["relevant_doc_ids"] = ["not-in-corpus"]
    elif mutation == "duplicate":
        cases[1]["case_id"] = cases[0]["case_id"]
    elif mutation == "missing":
        del cases[0]["relevant_doc_ids"]
    elif mutation == "status":
        cases[0]["retrieval_status"] = "out_of_scope"
    else:
        cases = []
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        load_labels(path, CORPUS_PATH)


def test_skips_are_explicit_and_failures_are_inspectable():
    cases, _ = load_labels(DATASET_PATH, CORPUS_PATH)
    selected = [cases[0], cases[6], cases[18]]  # answerable, partial, out of scope
    calls = []

    def search(query):
        calls.append(query)
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    chunk_id="chunk",
                    text="evidence",
                    score=0.5,
                    metadata={"faq_id": "faq-001"},
                )
            ]
        )

    report = evaluate_id_cases(selected, search)
    assert len(calls) == 1
    assert report["total_cases"] == 3 and report["scored_cases"] == 1
    assert report["hit@5"] == 1 and report["recall@5"] == 0.5
    assert report["miss_case_ids"] == ["eval-001"]
    assert report["zero_hit_case_ids"] == []
    assert report["details"][0]["missed_doc_ids"] == ["faq-002"]
    assert report["details"][1]["scored"] is False


def test_missing_result_id_and_backend_errors_fail():
    cases, _ = load_labels(DATASET_PATH, CORPUS_PATH)
    with pytest.raises(ValueError, match="missing faq_id"):
        evaluate_id_cases(
            cases[:1], lambda q: SimpleNamespace(results=[SimpleNamespace(metadata={})])
        )

    def broken_search(query):
        raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        evaluate_id_cases(cases[:1], broken_search)


def test_model_manifest_detects_same_size_weight_change(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    before = model_fingerprint(str(tmp_path))
    weights.write_bytes(b"other")
    after = model_fingerprint(str(tmp_path))
    assert before["sha256"] != after["sha256"]


def test_faq_id_survives_real_chroma_roundtrip(tmp_path):
    from src.embedding.chroma_store import ChromaStore
    from src.embedding.models import EmbeddingResult

    store = ChromaStore(
        persist_dir=str(tmp_path / "index"), collection="test_id_baseline"
    )
    store.insert(
        [
            EmbeddingResult(
                "chunk-a", [1.0, 0.0], "first", {"faq_id": "faq-a", "access_level": 0}
            ),
            EmbeddingResult(
                "chunk-b", [0.0, 1.0], "second", {"faq_id": "faq-b", "access_level": 0}
            ),
        ]
    )
    response = store.dense_search([1.0, 0.0], top_k=2)
    assert [row.metadata["faq_id"] for row in response.results] == ["faq-a", "faq-b"]
    assert (
        calculate_id_metrics(
            [row.metadata["faq_id"] for row in response.results], ["faq-b"]
        )["mrr@5"]
        == 0.5
    )


def test_cli_rejects_ambiguous_generation_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_benchmark.py", "--with-generation"],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert b"--legacy-similarity" in result.stderr
