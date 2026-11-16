"""Frozen FAQ retrieval baseline, isolated from application data and caches."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .metrics import calculate_id_metrics

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data/faq/ecommerce_kb.json"
DATASET_PATH = ROOT / "data/eval_dataset.json"
STATUSES = {"answerable", "partial_evidence", "out_of_scope"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(dataset_path: Path, corpus_path: Path) -> tuple[list[dict], list[dict]]:
    """Fail on missing/unknown labels instead of silently changing denominators."""
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        not isinstance(cases, list)
        or not cases
        or not isinstance(corpus, list)
        or not corpus
    ):
        raise ValueError("Dataset and corpus must be non-empty JSON arrays")
    for rows, key in [(corpus, "faq_id"), (cases, "case_id")]:
        ids = [row.get(key) for row in rows]
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError(f"Missing or invalid {key}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {key}")
    known = {row["faq_id"] for row in corpus}
    for row in corpus:
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"Empty FAQ: {row['faq_id']}")
    for row in cases:
        status = row.get("retrieval_status")
        ids = row.get("relevant_doc_ids")
        if status not in STATUSES or not isinstance(ids, list):
            raise ValueError(f"Invalid retrieval labels: {row['case_id']}")
        if any(not isinstance(value, str) or value not in known for value in ids):
            raise ValueError(f"Unknown relevant document: {row['case_id']}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate relevant document: {row['case_id']}")
        if (status == "out_of_scope") != (not ids):
            raise ValueError(f"Evidence/status mismatch: {row['case_id']}")
        if not row.get("question") or not row.get("annotation_note"):
            raise ValueError(f"Missing question/annotation note: {row['case_id']}")
    return cases, corpus


def faq_text(item: dict, index: int) -> str:
    """Match FAQ parser's Q/A text; IDs remain metadata, outside model input."""
    text = f"## Q{index}: {item['question']}\n\n{item['answer']}"
    if item.get("keywords"):
        text += f"\n\n*关键词: {', '.join(item['keywords'])}*"
    return text


def evaluate_id_cases(cases: list[dict], search, k: int = 5) -> dict:
    """Evaluate a search callable returning SearchResponse (also usable in tests)."""
    details = []
    for case in cases:
        detail = {
            key: case[key]
            for key in (
                "case_id",
                "question",
                "retrieval_status",
                "relevant_doc_ids",
                "annotation_note",
            )
        }
        detail.update(
            intent=case.get("intent", ""), difficulty=case.get("difficulty", "")
        )
        if case["retrieval_status"] != "answerable":
            detail["scored"] = False
            details.append(detail)
            continue
        start = time.perf_counter()
        response = search(case["question"])
        elapsed = (time.perf_counter() - start) * 1000
        results = response.results[:k]
        ids = [result.metadata.get("faq_id", "") for result in results]
        if any(not doc_id for doc_id in ids):
            raise ValueError("Search result missing faq_id; baseline index is invalid")
        scores = calculate_id_metrics(ids, case["relevant_doc_ids"], k)
        detail.update(
            scored=True,
            metrics=scores,
            latency_ms=round(elapsed, 3),
            retrieved_doc_ids=ids,
            matched_doc_ids=sorted(set(ids) & set(case["relevant_doc_ids"])),
            missed_doc_ids=sorted(set(case["relevant_doc_ids"]) - set(ids)),
            retrieved=[
                {
                    "faq_id": result.metadata["faq_id"],
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "rerank_score": result.metadata.get("rerank_score"),
                    "text": result.text,
                }
                for result in results
            ],
        )
        details.append(detail)
    scored = [detail for detail in details if detail["scored"]]
    if not scored:
        raise ValueError("No answerable retrieval cases selected")
    keys = scored[0]["metrics"]
    latencies = sorted(detail["latency_ms"] for detail in scored)
    from .evaluator import _percentile

    return {
        "schema_version": "id-retrieval-v1",
        "metric_basis": "binary_faq_id_labels",
        "total_cases": len(cases),
        "scored_cases": len(scored),
        "status_counts": dict(Counter(case["retrieval_status"] for case in cases)),
        **{
            key: round(sum(row["metrics"][key] for row in scored) / len(scored), 6)
            for key in keys
        },
        "miss_case_ids": [row["case_id"] for row in scored if row["missed_doc_ids"]],
        "zero_hit_case_ids": [
            row["case_id"] for row in scored if not row["matched_doc_ids"]
        ],
        "latency": {
            "scope": "query embedding + dense search + optional reranking; excludes index/model warmup",
            "avg_ms": round(sum(latencies) / len(latencies), 3),
            **{f"p{p}_ms": round(_percentile(latencies, p), 3) for p in (50, 95, 99)},
        },
        "details": details,
    }


def model_fingerprint(path: str) -> dict:
    folder = Path(path)
    if not path or not folder.is_dir():
        raise ValueError(
            "Local model directory missing; configure model path in .env or environment"
        )
    files = sorted(
        p for p in folder.rglob("*") if p.is_file() and ".git" not in p.parts
    )
    if not files:
        raise ValueError("Local model directory is empty")
    manifest = [
        {"file": p.relative_to(folder).as_posix(), "sha256": file_sha256(p)}
        for p in files
    ]
    return {
        "sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files": manifest,
    }


def git_metadata() -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_baseline(
    dataset_path: Path,
    corpus_path: Path,
    run_dir: Path,
    *,
    use_rerank: bool = True,
    limit: int = 0,
) -> dict:
    """Build an isolated corpus snapshot; fail on model/index/rerank errors."""
    import torch

    from src.config import settings
    from src.embedding.chroma_store import ChromaStore
    from src.embedding.embedder import QUERY_INSTRUCTION, Embedder
    from src.retrieval.reranker import Reranker

    torch.manual_seed(0)
    torch.set_num_threads(4)

    cases, corpus = load_labels(dataset_path, corpus_path)
    if limit:
        cases = cases[:limit]
    # Fresh directory prevents stale vectors or another corpus from entering the run.
    run_dir.mkdir(parents=True, exist_ok=False)
    embedder = Embedder()
    reranker = Reranker() if use_rerank else None
    models = {"embedding": model_fingerprint(settings.embedding_model_path)}
    if use_rerank:
        models["reranker"] = model_fingerprint(settings.reranker_model_path)
    embedder._get_model().to("cpu")
    if reranker:
        reranker._get_model().model.to("cpu")
    store = ChromaStore(
        persist_dir=str(run_dir / "index"), collection="faq_id_baseline"
    )
    batch = embedder.embed_chunks(
        texts=[faq_text(item, i) for i, item in enumerate(corpus, 1)],
        chunk_ids=[item["faq_id"] for item in corpus],
        metadata_list=[
            {
                "faq_id": item["faq_id"],
                "access_level": 0,
                "source_file": corpus_path.name,
                "doc_type": "faq_json",
            }
            for item in corpus
        ],
        show_progress=False,
    )
    if batch.errors or len(batch.embeddings) != len(corpus):
        raise RuntimeError("Incomplete corpus embedding; refusing to score")
    if store.insert(batch.embeddings) != len(corpus):
        raise RuntimeError("Incomplete corpus index; refusing to score")
    # Explicit warmup makes query timings separate from first model load.
    embedder.embed_query("预热")
    if reranker:
        reranker.rerank(query="预热", documents=["预热文档"], top_n=1)
    top_k = 5
    candidates = max(top_k, settings.retrieval_dense_top_k) if use_rerank else top_k

    def search(query):
        response = store.dense_search(
            query_vector=embedder.embed_query(query).tolist(),
            top_k=candidates,
            threshold=settings.retrieval_similarity_threshold,
        )
        if reranker and response.results:
            # rerank_search_response silently falls back; benchmark must fail instead.
            ranked = reranker.rerank(
                query, [r.text for r in response.results], top_n=top_k
            )
            results = []
            for item in ranked:
                result = response.results[item["index"]]
                result.metadata["rerank_score"] = item["relevance_score"]
                results.append(result)
            response.results = results
        return response

    report = evaluate_id_cases(cases, search, top_k)
    packages = (
        "numpy",
        "chromadb",
        "sentence-transformers",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "scipy",
        "scikit-learn",
    )
    source_paths = [
        "src/evaluation/metrics.py",
        "src/evaluation/id_baseline.py",
        "src/evaluation/evaluator.py",
        "scripts/run_benchmark.py",
        "src/embedding/embedder.py",
        "src/embedding/chroma_store.py",
        "src/retrieval/reranker.py",
        "src/config.py",
    ]
    report["reproducibility"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "dataset": {"sha256": file_sha256(dataset_path), "filename": dataset_path.name},
        "corpus": {
            "sha256": file_sha256(corpus_path),
            "filename": corpus_path.name,
            "faq_count": len(corpus),
        },
        "source_sha256": {path: file_sha256(ROOT / path) for path in source_paths},
        "models": models,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "parameters": {
            "top_k": top_k,
            "candidate_k": candidates,
            "use_rerank": use_rerank,
            "threshold": settings.retrieval_similarity_threshold,
            "limit": limit,
            "query_instruction": QUERY_INSTRUCTION,
            "cache": "disabled",
            "chunking": "one_faq_per_chunk",
            "access_level": "public",
            "embedding_model": settings.embedding_model,
            "embedding_dimension": embedder.dimension,
            "reranker_model": settings.reranker_model if use_rerank else None,
            "device": "cpu",
            "torch_seed": 0,
            "torch_threads": 4,
        },
        "scope": "isolated FAQ retrieval only; no intent routing, LLM generation or handoff",
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
