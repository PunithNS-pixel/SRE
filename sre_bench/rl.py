from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .env import SREBenchEnv
from .models import Action, ActionType


def _action_payload(action_type: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "action_type": action_type,
        "params": params or {},
    }


def _action_to_key(action: Dict[str, Any]) -> str:
    params = action.get("params", {})
    return json.dumps(
        {
            "action_type": action.get("action_type", ""),
            "params": params,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _key_to_action(key: str) -> Dict[str, Any]:
    data = json.loads(key)
    return {
        "action_type": data.get("action_type", "get_topology"),
        "params": data.get("params", {}),
    }


def _default_policy_action(task_id: str, step: int) -> Dict[str, Any]:
    defaults = {
        "task1_oom": [
            _action_payload("get_topology"),
            _action_payload("read_logs", {"service": "payments-service", "window_minutes": 10}),
            _action_payload("get_metrics", {"service": "payments-service"}),
            _action_payload("restart_service", {"service": "payments-service"}),
            _action_payload(
                "write_postmortem",
                {
                    "root_cause": "payments-service OOM due to heap exhaustion",
                    "timeline": "Investigated alerts, logs, and metrics then restarted payments-service.",
                    "prevention": "Increase heap and improve pre-OOM alerting.",
                },
            ),
        ],
        "task2_bad_deploy": [
            _action_payload("get_topology"),
            _action_payload("read_logs", {"service": "auth-service", "window_minutes": 10}),
            _action_payload("get_metrics", {"service": "auth-service"}),
            _action_payload("rollback_deploy", {"service": "auth-service"}),
            _action_payload("mark_resolved", {"root_cause": "bad deploy in auth-service caused cascading 503s"}),
        ],
        "task3_phantom_slowdown": [
            _action_payload("get_topology"),
            _action_payload("get_metrics", {"service": "checkout-service"}),
            _action_payload("read_logs", {"service": "data-pipeline-service", "window_minutes": 10}),
            _action_payload("scale_up", {"service": "data-pipeline-service", "replicas": 6}),
            _action_payload(
                "write_postmortem",
                {
                    "root_cause": "periodic cron in data-pipeline-service caused DB lock and checkout latency spikes",
                    "timeline": "Correlated periodic spikes with data-pipeline cron lock windows.",
                    "prevention": "Stagger cron schedule and add lock duration alerts.",
                },
            ),
        ],
    }
    task_steps = defaults.get(task_id) or [_action_payload("get_topology")]
    idx = min(step, len(task_steps) - 1)
    return task_steps[idx]


class RLEpisodePolicy:
    """A small tabular policy keyed by (task_id, step_index)."""

    def __init__(self, q_table: Dict[str, Dict[str, float]] | None = None):
        self.q_table: Dict[str, Dict[str, float]] = q_table or {}

    def _state_key(self, task_id: str, step: int) -> str:
        return f"{task_id}|{step}"

    def suggest(self, task_id: str, step: int) -> Dict[str, Any]:
        state_key = self._state_key(task_id, step)
        actions = self.q_table.get(state_key, {})
        if not actions:
            return _default_policy_action(task_id, step)
        best_key = max(actions, key=lambda k: actions[k])
        return _key_to_action(best_key)

    def confidence(self, task_id: str, step: int) -> float:
        state_key = self._state_key(task_id, step)
        actions = self.q_table.get(state_key, {})
        if not actions:
            return 0.0
        vals = sorted(actions.values(), reverse=True)
        if len(vals) == 1:
            return 1.0
        best = vals[0]
        second = vals[1]
        return round(max(0.0, min(1.0, best - second)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"q_table": self.q_table}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RLEpisodePolicy":
        return cls(q_table=payload.get("q_table", {}))


def load_scenarios_from_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    scenarios = dataset.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("dataset scenarios must be a list")
    return scenarios


def train_policy_from_scenarios(
    scenarios: List[Dict[str, Any]],
    gamma: float = 0.95,
    alpha: float = 0.35,
) -> Tuple[RLEpisodePolicy, Dict[str, Any]]:
    q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    training_report: List[Dict[str, Any]] = []
    for scenario in scenarios:
        task_id = scenario.get("task_id")
        steps = scenario.get("steps", [])
        seed = int(scenario.get("seed", 42))
        if not task_id or not isinstance(steps, list):
            continue

        env = SREBenchEnv(task_id=task_id, seed=seed)
        env.reset()

        score = 0.0
        done = False
        for idx, step in enumerate(steps):
            atype = step.get("action_type")
            params = step.get("params", {})
            if not atype:
                continue
            action = Action(action_type=ActionType(atype), params=params)
            _, reward, done, info = env.step(action)
            score = reward.episode_score if done else score
            if done:
                ep = info.get("episode_result")
                if ep and hasattr(ep, "episode_score"):
                    score = float(ep.episode_score)
                break

        returns = []
        for idx in range(len(steps)):
            returns.append((gamma ** idx) * score)

        for idx, step in enumerate(steps):
            atype = step.get("action_type")
            if not atype:
                continue
            state_key = f"{task_id}|{idx}"
            action_key = _action_to_key({"action_type": atype, "params": step.get("params", {})})

            old_q = q_table[state_key].get(action_key, 0.0)
            target = returns[idx] if idx < len(returns) else score
            new_q = old_q + alpha * (target - old_q)
            q_table[state_key][action_key] = round(new_q, 6)
            counts[state_key][action_key] += 1

        training_report.append(
            {
                "scenario": scenario.get("id", "unknown"),
                "task_id": task_id,
                "score": round(score, 4),
                "steps_used": len(steps),
                "terminated": done,
            }
        )

    policy = RLEpisodePolicy(q_table={k: dict(v) for k, v in q_table.items()})
    metrics = {
        "scenarios_trained": len(training_report),
        "states": len(policy.q_table),
        "action_entries": sum(len(v) for v in policy.q_table.values()),
        "reports": training_report,
    }
    return policy, metrics


def autoplay_episode(
    policy: RLEpisodePolicy,
    task_id: str,
    seed: int = 42,
    max_actions: int = 8,
) -> Dict[str, Any]:
    env = SREBenchEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    trace: List[Dict[str, Any]] = []
    done = False
    info: Dict[str, Any] = {}
    reward_payload: Dict[str, Any] = {}

    for step in range(max_actions):
        suggestion = policy.suggest(task_id, step)
        action = Action(
            action_type=ActionType(suggestion["action_type"]),
            params=suggestion.get("params", {}),
        )
        obs, reward, done, info = env.step(action)
        reward_payload = reward.model_dump() if hasattr(reward, "model_dump") else reward.dict()
        trace.append(
            {
                "step": step,
                "action": suggestion,
                "reward": reward_payload,
                "done": done,
                "last_action_result": obs.last_action_result,
            }
        )
        if done:
            break

    episode_result = info.get("episode_result")
    if episode_result is not None:
        episode_result = (
            episode_result.model_dump()
            if hasattr(episode_result, "model_dump")
            else episode_result.dict()
        )

    return {
        "task_id": task_id,
        "seed": seed,
        "done": done,
        "trace": trace,
        "episode_result": episode_result,
        "final_reward": reward_payload,
    }