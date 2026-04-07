#!/usr/bin/env python3
"""
SRE-Bench Baseline
==================
Runs an OpenAI model as an SRE agent against all three tasks.
Reads OPENAI_API_KEY from environment.

Usage
-----
    pip install openai pydantic
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Run a single task:
    python baseline.py --task task1_oom

    # Use a different model:
    python baseline.py --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List

from openai import OpenAI

from sre_bench import (
    SREBenchEnv, Action, ActionType, ALL_TASK_IDS,
    Observation, EpisodeResult,
)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) on-call.
You are operating inside SRE-Bench, a production incident simulation.
Your job: diagnose the incident and restore service as quickly as possible.

Available actions (respond with EXACTLY one JSON object):
  {"action_type": "get_topology",    "params": {}}
  {"action_type": "get_metrics",     "params": {"service": "<name>"}}
  {"action_type": "read_logs",       "params": {"service": "<name>", "window_minutes": 10}}
  {"action_type": "restart_service", "params": {"service": "<name>"}}
  {"action_type": "scale_up",        "params": {"service": "<name>", "replicas": <int>}}
  {"action_type": "rollback_deploy", "params": {"service": "<name>"}}
  {"action_type": "page_team",       "params": {"message": "<explanation>"}}
  {"action_type": "mark_resolved",   "params": {"root_cause": "<explanation>"}}
  {"action_type": "write_postmortem","params": {"root_cause": "<str>", "timeline": "<str>", "prevention": "<str>"}}

Strategy:
1. Start with get_topology to understand the service graph.
2. Check metrics on unhealthy or alerting services.
3. Read logs on the most suspicious service.
4. Identify root cause BEFORE remediating — wrong actions cause blast radius.
5. Apply the correct fix (restart / rollback / scale_up).
6. Write a postmortem explaining the root cause, timeline, and prevention.

Respond with ONLY a JSON object. No markdown, no explanation text.
""").strip()


def obs_to_prompt(obs: Observation) -> str:
    lines = [
        f"=== Step {obs.step} | Simulated time: {obs.simulated_time_min} min ===",
        f"Incident active: {obs.incident_active}",
        "",
        "ACTIVE ALERTS:",
    ]
    for a in obs.alerts:
        lines.append(
            f"  [{a.severity.upper()}] {a.service} — {a.message} "
            f"(metric={a.metric}, value={a.value}, threshold={a.threshold})"
        )

    if obs.topology:
        lines.append("\nTOPOLOGY:")
        for node in obs.topology:
            status = "healthy" if node.healthy else "UNHEALTHY"
            lines.append(f"  {node.name} ({node.tier}, {status}) → depends_on: {node.depends_on}")

    if obs.visible_metrics:
        lines.append("\nMETRICS:")
        for svc, m in obs.visible_metrics.items():
            lines.append(
                f"  {svc}: cpu={m.cpu_percent:.1f}% mem={m.memory_percent:.1f}% "
                f"err={m.error_rate:.2f}/s p99={m.p99_latency_ms:.0f}ms "
                f"healthy={m.healthy} sha={m.last_deploy_sha}"
            )

    if obs.visible_logs:
        lines.append("\nLOGS:")
        for svc, entries in obs.visible_logs.items():
            lines.append(f"  {svc}:")
            for e in entries[-8:]:   # last 8 lines per service
                lines.append(f"    [{e.level}] t={e.timestamp}m  {e.message}")

    if obs.last_action_result:
        lines.append(f"\nLAST ACTION RESULT: {obs.last_action_result}")

    lines.append("\nChoose your next action:")
    return "\n".join(lines)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    client: OpenAI,
    model: str,
    task_id: str,
    max_steps: int = 30,
    verbose: bool = True,
) -> EpisodeResult:

    env = SREBenchEnv(task_id=task_id)
    obs = env.reset()
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_result: EpisodeResult | None = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_id}")
        print(f"  Model: {model}")
        print(f"{'='*60}")

    for step in range(max_steps):
        user_content = obs_to_prompt(obs)
        messages.append({"role": "user", "content": user_content})

        # Call the model
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 400,
            )
        except Exception as e:
            print(f"[ERROR] OpenAI API error on step {step+1}: {e}")
            break

        raw = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": raw})

        if verbose:
            print(f"\n[Step {step+1}] Model response: {raw[:200]}")

        # Parse action
        try:
            # Strip markdown fences if model wrapped it
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            action_dict = json.loads(clean)
            action = Action(**action_dict)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not parse action: {e}. Defaulting to get_topology.")
            action = Action(action_type=ActionType.GET_TOPOLOGY)

        obs, reward, done, info = env.step(action)

        if verbose:
            print(
                f"  Reward: step={reward.step_reward:+.3f} "
                f"time={reward.time_penalty:+.3f} "
                f"blast={reward.blast_radius_penalty:+.3f} "
                f"cumulative={reward.cumulative_reward:+.3f}"
            )

        if done:
            episode_result = info.get("episode_result")
            if verbose and episode_result:
                print(f"\n  ✓ Episode complete!")
                print(f"  Score:        {episode_result.episode_score:.4f}")
                print(f"  Success:      {episode_result.success}")
                print(f"  Steps:        {episode_result.steps_taken}")
                print(f"  Sim time:     {episode_result.simulated_minutes} min")
                print(f"  Root cause ✓: {episode_result.root_cause_correct}")
                print(f"  Blast radius: {episode_result.blast_radius_score:.2f}")
                print(f"  Postmortem:   {episode_result.postmortem_score:.2f}")
                if episode_result.wrong_services_hit:
                    print(f"  Wrong hits:   {episode_result.wrong_services_hit}")
            break

        time.sleep(0.3)  # gentle rate-limiting

    if episode_result is None:
        # Timeout — build a partial result
        state  = env.state()
        env._incident_resolved = True   # force termination grading
        _, _, _, info = env.step(
            Action(action_type=ActionType.MARK_RESOLVED,
                   params={"root_cause": "timeout — no resolution reached"})
        )
        episode_result = info.get("episode_result") or EpisodeResult(
            task_id="unknown", success=False, episode_score=0.0,
            steps_taken=max_steps, simulated_minutes=0,
            root_cause_correct=False, blast_radius_score=1.0,
            time_efficiency=0.0, postmortem_score=0.0,
            notes="timeout",
        )
        if verbose:
            print(f"\n  ✗ Max steps reached. Score: {episode_result.episode_score:.4f}")

    return episode_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SRE-Bench baseline inference script")
    parser.add_argument("--model",  default="gpt-4o-mini",
                        help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--task",   default=None,
                        help="Single task id to run (default: all tasks)")
    parser.add_argument("--steps",  type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress step-by-step output")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client   = OpenAI(api_key=api_key)
    task_ids = [args.task] if args.task else ALL_TASK_IDS
    results  = {}

    print(f"\nSRE-Bench Baseline | model={args.model}")
    print("=" * 60)

    for task_id in task_ids:
        result = run_agent(
            client    = client,
            model     = args.model,
            task_id   = task_id,
            max_steps = args.steps,
            verbose   = not args.quiet,
        )
        results[task_id] = result

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    print(f"{'Task':<30} {'Score':>6} {'Success':>8} {'Steps':>6} {'RCA✓':>6}")
    print("-" * 60)

    total = 0.0
    for tid, r in results.items():
        print(
            f"{tid:<30} {r.episode_score:>6.4f} "
            f"{'Yes' if r.success else 'No':>8} "
            f"{r.steps_taken:>6} "
            f"{'Yes' if r.root_cause_correct else 'No':>6}"
        )
        total += r.episode_score

    avg = total / len(results) if results else 0.0
    print("-" * 60)
    print(f"{'AVERAGE':<30} {avg:>6.4f}")

    # Save JSON
    out = {
        "model":   args.model,
        "results": {tid: r.dict() for tid, r in results.items()},
        "average_score": avg,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
