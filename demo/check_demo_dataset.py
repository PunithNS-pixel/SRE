from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_json(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=20) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data) if data else {}
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} for {method} {url}: {detail}") from err
    except URLError as err:
        raise RuntimeError(f"Network error for {method} {url}: {err}") from err


def run_scenario(base_url: str, scenario: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    reset_payload = {
        "task_id": scenario["task_id"],
        "seed": scenario.get("seed", 42),
    }
    reset_resp = http_json("POST", f"{base_url}/reset", reset_payload)
    episode_id = reset_resp.get("episode_id")
    if not episode_id:
        return False, {"error": "Missing episode_id in /reset response", "scenario": scenario["id"]}

    done = False
    last_step_resp: Dict[str, Any] = {}
    for idx, step in enumerate(scenario.get("steps", []), start=1):
        step_payload = {
            "episode_id": episode_id,
            "action_type": step["action_type"],
            "params": step.get("params", {}),
        }
        last_step_resp = http_json("POST", f"{base_url}/step", step_payload)
        done = bool(last_step_resp.get("done", False))
        if done:
            break

    state_resp = http_json("POST", f"{base_url}/state", {"episode_id": episode_id})
    state = state_resp.get("state", {})
    info = last_step_resp.get("info", {})
    episode_result = info.get("episode_result") if isinstance(info, dict) else None

    score = None
    success = None
    if isinstance(episode_result, dict):
        score = episode_result.get("episode_score")
        success = episode_result.get("success")

    summary = {
        "scenario": scenario["id"],
        "task_id": scenario["task_id"],
        "done": done,
        "incident_resolved": state.get("incident_resolved"),
        "resolved_correctly": state.get("resolved_correctly"),
        "score": score,
        "success": success,
        "steps_defined": len(scenario.get("steps", [])),
    }

    return True, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SRE-Bench demo dataset against local API")
    parser.add_argument("--base-url", default="http://127.0.0.1:7861", help="Base URL of running app")
    parser.add_argument(
        "--dataset",
        default="demo/demo_dataset.json",
        help="Path to demo dataset JSON file",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenarios: List[Dict[str, Any]] = dataset.get("scenarios", [])
    if not scenarios:
        print("No scenarios found in dataset")
        return 1

    # Quick health check to fail fast if server is not up.
    try:
        health = http_json("GET", f"{args.base_url}/health")
    except Exception as err:
        print(f"Health check failed: {err}")
        return 1
    if health.get("status") != "ok":
        print(f"Unexpected health response: {health}")
        return 1

    print(f"Running {len(scenarios)} scenarios from {dataset_path} on {args.base_url}\n")

    passed = 0
    for scenario in scenarios:
        try:
            ok, result = run_scenario(args.base_url, scenario)
        except Exception as err:
            ok = False
            result = {"scenario": scenario.get("id", "unknown"), "error": str(err)}

        if ok:
            passed += 1
            print(
                f"[PASS] {result['scenario']} | task={result['task_id']} | "
                f"done={result['done']} | resolved={result['incident_resolved']} | "
                f"correct={result['resolved_correctly']} | score={result['score']}"
            )
        else:
            print(f"[FAIL] {result.get('scenario', 'unknown')} | {result.get('error', 'unknown error')}")

    print(f"\nSummary: {passed}/{len(scenarios)} scenarios executed successfully")
    return 0 if passed == len(scenarios) else 2


if __name__ == "__main__":
    sys.exit(main())
