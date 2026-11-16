"""Start an isolated real-model HTTP service, evaluate frozen FAQ dialogues, stop it."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def isolated_environment(run_dir: Path, secret: str) -> dict:
    """Set every storage path before importing application singletons in child."""
    env = os.environ.copy()
    paths = {
        "DATA_DIR": run_dir / "data",
        "LOG_DIR": run_dir / "logs",
        "CHROMA_PERSIST_DIR": run_dir / "index",
        "WORKFLOW_CHECKPOINT_PATH": run_dir / "data/checkpoints.db",
        "WORKFLOW_REVIEW_PATH": run_dir / "data/reviews.db",
        "WORKFLOW_LOCK_DIR": run_dir / "locks",
    }
    env.update({key: str(value.resolve()) for key, value in paths.items()})
    env.update(
        {
            "JWT_SECRET": secret,
            "JWT_ALGORITHM": "HS256",
            "ENABLE_DEMO_USERS": "false",
            "WEB_SEARCH_ENABLED": "false",
            "ZHIPU_WEB_SEARCH_ENABLED": "false",
            "CHROMA_COLLECTION": "dialogue_faq",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ANONYMIZED_TELEMETRY": "false",
            "PYTHONIOENCODING": "utf-8",
            # This suite needs FAQ evidence, not live customer order data.
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "9",
            "MYSQL_PASSWORD": "",
            "RAG_EVAL_RUN_DIR": str(run_dir.resolve()),
        }
    )
    return env


def serve(run_dir: Path, port: int, corpus_path: Path):
    import importlib.metadata
    import platform

    import torch
    import uvicorn

    from src.config import settings
    from src.embedding.chroma_store import ChromaStore
    from src.embedding.embedder import get_embedder
    from src.evaluation.id_baseline import (
        faq_text,
        file_sha256,
        git_metadata,
        model_fingerprint,
    )
    from src.retrieval.reranker import get_reranker

    expected = isolated_environment(run_dir, settings.jwt_secret)
    for key in (
        "DATA_DIR",
        "CHROMA_PERSIST_DIR",
        "WORKFLOW_CHECKPOINT_PATH",
        "WORKFLOW_REVIEW_PATH",
        "WORKFLOW_LOCK_DIR",
        "LOG_DIR",
    ):
        if os.environ.get(key) != expected[key]:
            raise RuntimeError("Isolation environment missing")
    if settings.web_search_enabled or settings.enable_demo_users:
        raise RuntimeError("Isolated mode must disable web search and demo login")
    if not settings.deepseek_api_key:
        raise RuntimeError("Configured real dialogue model key required")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    embedder = get_embedder()
    embedder._get_model().to("cpu")
    reranker = get_reranker()
    reranker._get_model().model.to("cpu")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    ids = [row["faq_id"] for row in corpus]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Nonempty corpus with unique FAQ IDs required")
    store = ChromaStore()
    batch = embedder.embed_chunks(
        [faq_text(row, i) for i, row in enumerate(corpus, 1)],
        ids,
        [
            {
                "faq_id": row["faq_id"],
                "access_level": 0,
                "source_file": corpus_path.name,
                "doc_type": "faq_json",
            }
            for row in corpus
        ],
        show_progress=False,
    )
    if (
        batch.errors
        or len(batch.embeddings) != len(corpus)
        or store.insert(batch.embeddings) != len(corpus)
    ):
        raise RuntimeError("Incomplete FAQ index")
    embedder.embed_query("预热")
    reranker.rerank("预热", ["预热文档"], top_n=1)
    source_files = [
        "scripts/run_live_rag_benchmark.py",
        "src/evaluation/dialogue.py",
        "src/graph/workflow.py",
        "src/conversation/coreference.py",
        "src/conversation/human_in_loop.py",
        "src/conversation/memory.py",
        "src/routing/router.py",
        "src/routing/classifier.py",
        "src/embedding/degradation.py",
        "src/embedding/retriever.py",
        "src/embedding/embedder.py",
        "src/retrieval/reranker.py",
        "src/generation/self_correction.py",
        "src/generation/hallucination_detector.py",
        "src/api/chat_service.py",
        "src/config.py",
        "data/policies/returns.json",
    ]
    manifest = {
        "execution": "http_service",
        "git": git_metadata(),
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
        "corpus_sha256": file_sha256(corpus_path),
        "faq_count": len(corpus),
        "models": {
            "embedding": model_fingerprint(settings.embedding_model_path),
            "reranker": model_fingerprint(settings.reranker_model_path),
            "dialogue": settings.default_model,
        },
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "sentence-transformers",
                "transformers",
                "huggingface-hub",
                "tokenizers",
                "chromadb",
                "langgraph",
                "fastapi",
                "pydantic",
            )
        },
        "parameters": {
            "device": "cpu",
            "torch_threads": 4,
            "torch_seed": 0,
            "web_search": False,
            "cache": "fresh process-local memory, reused within run",
            "order_backend": "disabled; FAQ suite only",
            "access_level": "public",
            "top_k": 5,
            "use_reranker": True,
            "dense_threshold": settings.retrieval_similarity_threshold,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from src.api.app import create_app

    app = create_app()

    @app.get("/__eval_ready")
    def ready():
        return {"run_id": run_dir.name}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/rag_dialogue_eval.json"
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/faq/ecommerce_kb.json"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args.run_dir.resolve(), args.port, args.corpus.resolve())
        return 0
    from src.evaluation.dialogue import evaluate_dialogues, http_chat, load_dialogues
    from src.evaluation.id_baseline import file_sha256

    cases = load_dialogues(args.dataset, corpus_path=args.corpus)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "dialogues": len(cases),
                    "turns": sum(len(c.turns) for c in cases),
                    "dataset_sha256": file_sha256(args.dataset),
                }
            )
        )
        return 0
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("Positive timeouts required")
    run_dir = (args.run_dir or ROOT / "reports/rag-live" / uuid.uuid4().hex).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    # Freeze inputs before any model calls; report hashes refer to these snapshots.
    dataset = run_dir / "dataset.json"
    corpus = run_dir / "corpus.json"
    dataset.write_bytes(args.dataset.read_bytes())
    corpus.write_bytes(args.corpus.read_bytes())
    secret = secrets.token_urlsafe(48)
    env = isolated_environment(run_dir, secret)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    import jwt

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": "rag-evaluation",
            "role": "normal",
            "iat": now,
            "exp": now + timedelta(hours=4),
        },
        secret,
        algorithm="HS256",
    )
    with (run_dir / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve",
                "--run-dir",
                str(run_dir),
                "--port",
                str(port),
                "--corpus",
                str(corpus),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        report = None
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Service startup failed; inspect private server.log"
                    )
                try:
                    with urlopen(
                        f"http://127.0.0.1:{port}/__eval_ready", timeout=2
                    ) as response:
                        if json.load(response).get("run_id") == run_dir.name:
                            break
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("Service startup timeout")
            send = http_chat(f"http://127.0.0.1:{port}", token, args.timeout)
            completed = 0

            def progress(query, session_id):
                nonlocal completed
                result = send(query, session_id)
                completed += 1
                print(
                    json.dumps(
                        {
                            "completed_turns": completed,
                            "needs_human": result.get("needs_human"),
                        }
                    ),
                    flush=True,
                )
                return result

            report = evaluate_dialogues(cases, progress, execution="http_service")
            report.update(
                created_at=datetime.now(timezone.utc).isoformat(),
                dataset_sha256=file_sha256(dataset),
                deployment=json.loads(
                    (run_dir / "manifest.json").read_text(encoding="utf-8")
                ),
                limitations="Authored development suite, not user traffic or a blinded holdout. Real BGE and dialogue API; FAQ-only corpus, web search and order backend disabled. Phrase/evidence checks are not a semantic judge.",
            )
        except Exception as exc:
            report = {
                "execution": "http_service",
                "complete": False,
                "startup_error": type(exc).__name__,
                "dataset_sha256": file_sha256(dataset),
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if report is not None:
                (run_dir / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    print(
        json.dumps(
            {
                "report": str(run_dir / "report.json"),
                "complete": report["complete"],
                "metrics": report.get("metrics"),
            }
        ),
        flush=True,
    )
    return 0 if report["complete"] and report["metrics"]["turn_pass_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
