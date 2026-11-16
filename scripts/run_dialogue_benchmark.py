"""Evaluate an explicitly selected running chat service using a dedicated account."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.dialogue import evaluate_dialogues, http_chat, load_dialogues
from src.evaluation.id_baseline import ROOT, file_sha256, git_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/dialogue_eval.json"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--base-url", help="Explicit test service URL; no default production target"
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--output", type=Path, help="Default: reports/dialogue/<run_id>.json"
    )
    parser.add_argument(
        "--deployment-id", help="Server commit/image identifier supplied by operator"
    )
    args = parser.parse_args()
    cases = load_dialogues(args.dataset)
    if args.validate_only:
        print(
            json.dumps(
                {"dialogues": len(cases), "turns": sum(len(c.turns) for c in cases)}
            )
        )
        return 0
    if (
        not args.base_url
        or not args.deployment_id
        or not os.getenv("DIALOGUE_EVAL_TOKEN")
    ):
        parser.error(
            "Require --base-url, --deployment-id and DIALOGUE_EVAL_TOKEN (dedicated test account)"
        )
    report = evaluate_dialogues(
        cases,
        http_chat(args.base_url, os.environ["DIALOGUE_EVAL_TOKEN"], args.timeout),
        execution="http_service",
    )
    report.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=file_sha256(args.dataset),
        client_git=git_metadata(),
        deployment_id=args.deployment_id,
        limitations="Authored regression suite, not an independent user sample. Deployment ID is operator supplied. Reply checks do not prove semantic faithfulness.",
    )
    output = args.output or ROOT / "reports/dialogue" / (report["run_id"] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "metrics": report["metrics"],
                "report": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["complete"] and report["metrics"]["turn_pass_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
