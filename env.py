"""
SRE-Bench Environment
=====================
Implements the full OpenEnv interface:
  reset()       → Observation
  step(action)  → (Observation, Reward, done, info)
  state()       → dict

Usage
-----
    from sre_bench import SREBenchEnv, Action, ActionType

    env = SREBenchEnv(task_id="task1_oom")
    obs = env.reset()

    while True:
        action = Action(action_type=ActionType.READ_LOGS,
                        params={"service": "payments-service", "window_minutes": 10})
        obs, reward, done, info = env.step(action)
        if done:
            print(info["episode_result"])
            break
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, ActionType, Alert, EpisodeResult,
    LogEntry, Observation, Reward,
)
from .simulator import ServiceSimulator, ACTION_TIME_COST
from .tasks import get_task, Task3_PhantomSlowdown


class SREBenchEnv:
    """
    OpenEnv-compatible environment for AI SRE agent evaluation.

    Parameters
    ----------
    task_id : str
        One of "task1_oom", "task2_bad_deploy", "task3_phantom_slowdown".
    seed : int
        Random seed for reproducible jitter in metrics.
    max_steps : int | None
        Override the task's default max_steps.
    """

    VERSION = "1.0.0"

    def __init__(self, task_id: str = "task1_oom", seed: int = 42,
                 max_steps: Optional[int] = None):
        self.task_id   = task_id
        self.seed      = seed
        self._task     = get_task(task_id)
        self._sim      = ServiceSimulator(seed=seed)
        self._max_steps = max_steps or self._task.max_steps

        # Episode state (populated by reset)
        self._step_count:       int          = 0
        self._cumulative_reward: float       = 0.0
        self._incident_resolved: bool        = False
        self._resolved_correctly: bool       = False
        self._root_cause_stated:  str        = ""
        self._postmortem_stated:  dict       = {}
        self._topology_consulted: bool       = False
        self._wrong_services:     List[str]  = []
        self._visible_metrics:    dict       = {}
        self._visible_logs:       dict       = {}
        self._topology_cache:     Optional[list] = None
        self._last_action_result: str        = ""
        self._done:               bool       = False

        # Task-3 cron phase tracking
        self._cron_phase: int = 0

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to a clean incident state. Returns initial observation."""
        self._sim.reset()
        self._task.setup(self._sim)

        self._step_count         = 0
        self._cumulative_reward  = 0.0
        self._incident_resolved  = False
        self._resolved_correctly = False
        self._root_cause_stated  = ""
        self._postmortem_stated  = {}
        self._topology_consulted = False
        self._wrong_services     = []
        self._visible_metrics    = {}
        self._visible_logs       = {}
        self._topology_cache     = None
        self._last_action_result = (
            "Incident detected. Active alerts are listed below. "
            "Use get_topology, get_metrics, and read_logs to investigate. "
            "When ready, remediate with restart_service, rollback_deploy, or scale_up. "
            "Close the incident with mark_resolved and write_postmortem."
        )
        self._done               = False
        self._cron_phase         = 0

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """
        Execute one action and return (observation, reward, done, info).
        """
        if self._done and action.action_type != ActionType.WRITE_POSTMORTEM:
            raise RuntimeError("Episode finished. Call reset() to start a new episode.")

        self._step_count += 1
        step_reward      = 0.0
        blast_penalty    = 0.0
        resolution_bonus = 0.0
        postmortem_bonus = 0.0

        # ── Small time-step penalty (encourages efficiency) ──────────────────
        time_cost = ACTION_TIME_COST.get(action.action_type.value, 1)
        self._sim.advance_time(time_cost)
        time_penalty = -0.02 * time_cost

        # ── Cron phase advance (Task 3) ───────────────────────────────────────
        if isinstance(self._task, Task3_PhantomSlowdown):
            self._cron_phase = (self._sim.simulated_time % self._task.CRON_PERIOD)

        # ── Dispatch action ───────────────────────────────────────────────────
        atype = action.action_type
        p     = action.params

        if atype == ActionType.GET_TOPOLOGY:
            self._topology_cache     = self._sim.get_topology()
            self._topology_consulted = True
            step_reward              = 0.05
            self._last_action_result = (
                "Topology retrieved. "
                + ", ".join(
                    f"{n.name}({'healthy' if n.healthy else 'UNHEALTHY'}, "
                    f"depends_on={n.depends_on})"
                    for n in self._topology_cache
                )
            )

        elif atype == ActionType.GET_METRICS:
            svc     = p.get("service", "")
            metrics = self._sim.get_metrics(svc)
            if metrics:
                self._visible_metrics[svc] = metrics
                step_reward = 0.03
                self._last_action_result = (
                    f"Metrics for {svc}: "
                    f"cpu={metrics.cpu_percent:.1f}% "
                    f"mem={metrics.memory_percent:.1f}% "
                    f"err_rate={metrics.error_rate:.2f}/s "
                    f"p50={metrics.p50_latency_ms:.0f}ms "
                    f"p99={metrics.p99_latency_ms:.0f}ms "
                    f"rps={metrics.requests_per_second:.0f} "
                    f"healthy={metrics.healthy} "
                    f"sha={metrics.last_deploy_sha}"
                )
            else:
                self._last_action_result = f"Unknown service: '{svc}'"

        elif atype == ActionType.READ_LOGS:
            svc  = p.get("service", "")
            is_spike = isinstance(self._task, Task3_PhantomSlowdown) and self._cron_phase == 0
            if hasattr(self._task, 'get_logs'):
                import inspect
                nparams = len(inspect.signature(self._task.get_logs).parameters)
                raw_logs = self._task.get_logs(svc, self._sim.simulated_time, is_spike) if nparams >= 4 else self._task.get_logs(svc, self._sim.simulated_time)
            else:
                raw_logs = []

            # Normalise duck-typed log objects to LogEntry
            entries: List[LogEntry] = []
            for lg in raw_logs:
                if isinstance(lg, LogEntry):
                    entries.append(lg)
                else:
                    entries.append(LogEntry(
                        timestamp=getattr(lg, "timestamp", 0),
                        service=getattr(lg, "service", svc),
                        level=getattr(lg, "level", "INFO"),
                        message=getattr(lg, "message", ""),
                    ))

            self._visible_logs.setdefault(svc, [])
            self._visible_logs[svc].extend(entries)
            step_reward = 0.04 if entries else 0.01
            self._last_action_result = (
                f"Logs for {svc} (last {len(entries)} entries):\n" +
                "\n".join(f"  [{e.level}] t={e.timestamp}m  {e.message}" for e in entries)
                if entries else f"No logs available for '{svc}'"
            )

        elif atype == ActionType.RESTART_SERVICE:
            svc = p.get("service", "")
            fixed, msg = self._sim.restart_service(svc, self._task.correct_service)
            self._last_action_result = msg
            if fixed and self._task.correct_action == "restart_service":
                self._incident_resolved  = True
                self._resolved_correctly = True
                resolution_bonus         = 0.40
                step_reward              = 0.10
            elif not fixed and svc in self._sim.blast_radius_events:
                blast_penalty            = -0.15
                self._wrong_services.append(svc)
                step_reward              = -0.10

        elif atype == ActionType.ROLLBACK_DEPLOY:
            svc = p.get("service", "")
            fixed, msg = self._sim.rollback_deploy(svc, self._task.correct_service)
            self._last_action_result = msg
            if fixed and self._task.correct_action == "rollback_deploy":
                self._incident_resolved  = True
                self._resolved_correctly = True
                resolution_bonus         = 0.40
                step_reward              = 0.10
            elif not fixed:
                step_reward = -0.05

        elif atype == ActionType.SCALE_UP:
            svc      = p.get("service", "")
            replicas = int(p.get("replicas", 4))
            fixed, msg = self._sim.scale_up(svc, replicas, self._task.correct_service)
            self._last_action_result = msg
            if fixed and self._task.correct_action == "scale_up":
                self._incident_resolved  = True
                self._resolved_correctly = True
                resolution_bonus         = 0.40
                step_reward              = 0.10
            elif not fixed:
                step_reward = -0.02

        elif atype == ActionType.PAGE_TEAM:
            msg  = p.get("message", "")
            # Task 3 special: paging team WITH the correct root cause counts as resolution
            if isinstance(self._task, Task3_PhantomSlowdown):
                keywords = self._task.correct_root_cause_keywords
                if any(kw in msg.lower() for kw in keywords):
                    self._incident_resolved  = True
                    self._resolved_correctly = True
                    resolution_bonus         = 0.30   # slightly lower than direct fix
                    step_reward              = 0.05
            self._last_action_result = f"Team paged: '{msg}'"

        elif atype == ActionType.MARK_RESOLVED:
            root_cause = p.get("root_cause", "")
            self._root_cause_stated  = root_cause.lower()
            self._incident_resolved  = True
            step_reward              = 0.02
            self._last_action_result = f"Incident marked resolved. Root cause: '{root_cause}'"

        elif atype == ActionType.WRITE_POSTMORTEM:
            root_cause  = p.get("root_cause",  "")
            timeline    = p.get("timeline",    "")
            prevention  = p.get("prevention",  "")
            self._postmortem_stated  = {
                "root_cause": root_cause,
                "timeline":   timeline,
                "prevention": prevention,
            }
            self._root_cause_stated  = root_cause.lower()
            postmortem_bonus         = self._score_postmortem(root_cause, timeline, prevention)
            step_reward              = postmortem_bonus * 0.2
            self._last_action_result = (
                f"Postmortem written. Score: {postmortem_bonus:.2f}. "
                f"Root cause: '{root_cause}'"
            )

        # ── Compute reward ────────────────────────────────────────────────────
        total_step = step_reward + time_penalty + blast_penalty + resolution_bonus
        self._cumulative_reward = round(self._cumulative_reward + total_step, 4)

        # ── Termination check ─────────────────────────────────────────────────
        done = (
            self._incident_resolved or
            self._step_count >= self._max_steps
        )
        self._done = done

        reward = Reward(
            step_reward          = round(step_reward, 4),
            time_penalty         = round(time_penalty, 4),
            blast_radius_penalty = round(blast_penalty, 4),
            resolution_bonus     = round(resolution_bonus, 4),
            postmortem_bonus     = round(postmortem_bonus, 4),
            cumulative_reward    = self._cumulative_reward,
            episode_score        = self._compute_episode_score() if done else 0.0,
            reason               = self._last_action_result[:120],
        )

        obs  = self._build_observation()
        info: Dict[str, Any] = {}

        if done:
            ep = self._build_episode_result(reward.episode_score)
            info["episode_result"] = ep
            info["episode_score"]  = ep.episode_score

        return obs, reward, done, info

    def state(self) -> dict:
        """Return raw internal state (for debugging / logging)."""
        return {
            "task_id":              self.task_id,
            "step":                 self._step_count,
            "simulated_time_min":   self._sim.simulated_time,
            "incident_resolved":    self._incident_resolved,
            "resolved_correctly":   self._resolved_correctly,
            "cumulative_reward":    self._cumulative_reward,
            "blast_events":         self._sim.blast_radius_events,
            "wrong_services":       self._wrong_services,
            "topology_consulted":   self._topology_consulted,
            "services": {
                name: {
                    "healthy":    svc["healthy"],
                    "error_rate": svc["error_rate"],
                }
                for name, svc in self._sim.services.items()
            },
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        alerts = self._task.initial_alerts()
        # Hide is_noise from the agent (they see the alerts, not the label)
        clean_alerts = [
            Alert(**{**a.dict(), "is_noise": False}) for a in alerts
        ]
        return Observation(
            alerts             = clean_alerts,
            visible_metrics    = copy.deepcopy(self._visible_metrics),
            visible_logs       = copy.deepcopy(self._visible_logs),
            topology           = self._topology_cache,
            simulated_time_min = self._sim.simulated_time,
            step               = self._step_count,
            incident_active    = not self._incident_resolved,
            last_action_result = self._last_action_result,
        )

    def _score_postmortem(self, root_cause: str, timeline: str, prevention: str) -> float:
        """Grade the postmortem 0.0–1.0 based on keyword presence."""
        rc_kw    = getattr(self._task, "correct_root_cause_keywords",
                           [self._task.correct_root_cause])
        score = 0.0
        rc_lower = root_cause.lower()
        if any(kw in rc_lower for kw in rc_kw):
            score += 0.50
        if len(timeline.split()) >= 10:
            score += 0.25
        if len(prevention) > 20 and any(
            w in prevention.lower()
            for w in ["monitor", "alert", "limit", "schedule", "reduce", "fix", "add", "increase"]
        ):
            score += 0.25
        return round(score, 4)

    def _root_cause_correct(self) -> bool:
        stated = self._root_cause_stated
        keyword = self._task.correct_root_cause
        keywords = getattr(self._task, "correct_root_cause_keywords", [keyword])
        return any(kw in stated for kw in keywords)

    def _compute_episode_score(self) -> float:
        result = self._build_episode_result(0.0)
        score  = self._task.grade(result)
        return score

    def _build_episode_result(self, episode_score: float) -> EpisodeResult:
        n_wrong     = len(self._sim.blast_radius_events)
        blast_score = max(0.0, 1.0 - 0.25 * n_wrong)
        time_ratio  = self._step_count / self._max_steps
        time_eff    = max(0.0, 1.0 - time_ratio)
        pm_score    = self._score_postmortem(
            self._postmortem_stated.get("root_cause", ""),
            self._postmortem_stated.get("timeline", ""),
            self._postmortem_stated.get("prevention", ""),
        )
        rc_correct = self._root_cause_correct()
        notes_parts = []
        if self._topology_consulted:
            notes_parts.append("topology")
        if self._postmortem_stated:
            notes_parts.append("postmortem")

        result = EpisodeResult(
            task_id            = self.task_id,
            success            = self._resolved_correctly,
            episode_score      = episode_score,
            steps_taken        = self._step_count,
            simulated_minutes  = self._sim.simulated_time,
            root_cause_correct = rc_correct,
            blast_radius_score = blast_score,
            time_efficiency    = time_eff,
            postmortem_score   = pm_score,
            wrong_services_hit = list(self._sim.blast_radius_events),
            notes              = ",".join(notes_parts),
        )
        result.episode_score = self._task.grade(result)
        return result
