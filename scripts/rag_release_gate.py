"""Run and gate real HTTP RAG dialogues in separate local benchmark services."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.release_gate import SUITES, check_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240, help="Seconds per turn")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recheck reports without calling models",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    run_dir = args.run_dir.resolve()
    if args.check_only:
        if not run_dir.is_dir():
            parser.error("check-only requires an existing run directory")
    else:
        # A new directory prevents silently reusing old results after startup failure.
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error("use a new run directory or pass --check-only")
        for suite in SUITES:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_live_rag_benchmark.py"),
                    "--dataset",
                    str(suite.dataset),
                    "--run-dir",
                    str(run_dir / suite.name),
                    "--timeout",
                    str(args.timeout),
                ],
                cwd=ROOT,
                check=False,
            )
    # The runner requires every turn to pass. Release thresholds intentionally
    # differ between suites, so the validated reports drive the return code here.
    result = check_release(run_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "gate.json").write_text(payload, encoding="utf-8")
    print(payload, end="", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
