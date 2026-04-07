#!/usr/bin/env python3
"""Run the OpenAI baseline across all SRE-Bench tasks and print a summary."""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

from openai import OpenAI

from sre_bench import ALL_TASK_IDS
from baseline import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SRE-Bench benchmark across all tasks")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--steps", type=int, default=30, help="Max steps per task")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step logs")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    results = {}

    print(f"SRE-Bench benchmark | model={args.model}")
    print("=" * 72)

    for task_id in ALL_TASK_IDS:
        result = run_agent(
            client=client,
            model=args.model,
            task_id=task_id,
            max_steps=args.steps,
            verbose=not args.quiet,
        )
        results[task_id] = result

    print()
    print(f"{'Task':<30} {'Score':>8} {'Success':>10} {'Steps':>8} {'Time':>8}")
    print("-" * 72)
    for task_id in ALL_TASK_IDS:
        result = results[task_id]
        print(
            f"{task_id:<30} {result.episode_score:>8.4f} "
            f"{'yes' if result.success else 'no':>10} "
            f"{result.steps_taken:>8} {result.simulated_minutes:>8}"
        )

    scores = [result.episode_score for result in results.values()]
    print("-" * 72)
    print(f"Average score: {mean(scores):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
