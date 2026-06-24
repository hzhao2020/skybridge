from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .broker import UserSideBroker
from .factory import build_workflow_providers
from .models import WorkflowRequest
from .optimizer import load_runtime_profile, plan_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the user-side cross-cloud video QA prototype.")
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.mock.json"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--planner",
        choices=["logical_optimal", "single_cloud", "greedy", "dpgm", "mtgp", "decomposition", "skyflow"],
        help="Choose providers using the runtime optimizer before executing the workflow.",
    )
    parser.add_argument("--profile", type=Path, default=Path("configs/planner.mock.json"))
    parser.add_argument("--shot-provider")
    parser.add_argument("--split-provider")
    parser.add_argument("--caption-provider")
    parser.add_argument("--qa-provider")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    overrides = {
        "shot_detection": args.shot_provider,
        "split_sample": args.split_provider,
        "caption": args.caption_provider,
        "qa": args.qa_provider,
    }
    planning_result = None
    if args.planner:
        profile, planner_config = load_runtime_profile(args.profile)
        with contextlib.redirect_stdout(sys.stderr):
            planning_result = plan_workflow(args.planner, profile, planner_config)
        overrides.update(planning_result.selected_providers)

    shot_detector, split_sampler, captioner, qa, selected = build_workflow_providers(config, overrides)

    broker = UserSideBroker(
        shot_detector=shot_detector,
        split_sampler=split_sampler,
        captioner=captioner,
        qa=qa,
        runs_dir=args.runs_dir,
    )
    result = broker.run(
        WorkflowRequest(
            video_path=args.video,
            question=args.question,
            selected_providers=selected,
        )
    )

    payload = result.to_dict()
    if planning_result is not None:
        payload["planning"] = planning_result.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
