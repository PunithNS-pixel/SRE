"""
SRE-Bench Tasks
===============
Three incident scenarios with deterministic graders.

Task 1 — Easy:   OOM crash on payments-service           (3 distractors)
Task 2 — Medium: Bad deploy cascading from auth-service  (6 distractors)
Task 3 — Hard:   Intermittent cron-caused DB lock        (8 distractors)

Each task returns:
  - setup(sim)            → mutates the simulator
  - initial_alerts()      → List[Alert] (what the agent sees first)
  - get_logs(sim, svc, t) → logs revealed by read_logs action
  - grade(episode)        → float 0.0–1.0
"""

from __future__ import annotations

from typing import List

from .models import Alert, AlertSeverity, EpisodeResult
from .simulator import (
    ServiceSimulator,
    oom_logs, noise_logs_cpu, bad_deploy_logs,
    cascade_logs, cron_lock_logs,
)


class Task1_OOM:
    id          = "task1_oom"
    name        = "The Classic OOM"
    difficulty  = "easy"
    description = (
        "payments-service is crash-looping with an OutOfMemoryError. "
        "Identify the root cause from logs and metrics, then restart the correct service. "
        "Three distractor alerts are firing. Do not restart healthy services."
    )
    max_steps            = 15
    time_limit_minutes   = 30
    correct_service      = "payments-service"
    correct_action       = "restart_service"
    correct_root_cause   = "oom"

    def setup(self, sim: ServiceSimulator) -> None:
        sim.set_oom("payments-service")
        sim.add_noise_to_service("analytics-service")
        sim.add_noise_to_service("notification-service")

    def initial_alerts(self) -> List[Alert]:
        return [
            Alert(id="a001", service="payments-service", severity=AlertSeverity.CRITICAL,
                  message="payments-service is in CrashLoopBackOff — health check failing",
                  metric="health_check", value=0.0, threshold=1.0, fired_at=0, is_noise=False),
            Alert(id="a002", service="payments-service", severity=AlertSeverity.CRITICAL,
                  message="payments-service memory at 99.8% — OOM killer active",
                  metric="memory_percent", value=99.8, threshold=90.0, fired_at=0, is_noise=False),
            Alert(id="a003", service="checkout-service", severity=AlertSeverity.WARNING,
                  message="checkout-service error rate elevated (payments dependency unhealthy)",
                  metric="error_rate", value=3.4, threshold=1.0, fired_at=1, is_noise=False),
            Alert(id="n001", service="analytics-service", severity=AlertSeverity.WARNING,
                  message="analytics-service CPU at 78% — batch aggregation running",
                  metric="cpu_percent", value=78.0, threshold=70.0, fired_at=0, is_noise=True),
            Alert(id="n002", service="notification-service", severity=AlertSeverity.INFO,
                  message="notification-service p99 elevated (142ms vs 80ms baseline)",
                  metric="p99_latency_ms", value=142.0, threshold=80.0, fired_at=0, is_noise=True),
            Alert(id="n003", service="inventory-service", severity=AlertSeverity.INFO,
                  message="inventory-service scheduled maintenance window — elevated latency expected",
                  metric="p99_latency_ms", value=38.0, threshold=30.0, fired_at=0, is_noise=True),
        ]

    def get_logs(self, service: str, t: int) -> List:
        if service == "payments-service":
            return oom_logs("payments-service", t)
        if service == "analytics-service":
            return noise_logs_cpu("analytics-service", t)
        if service == "notification-service":
            return [
                type('L', (), {"timestamp": t, "service": service, "level": "INFO",
                               "message": "Processing 1240 queued notifications — nominal"})()
            ]
        return []

    def grade(self, result: EpisodeResult) -> float:
        rc_score = 0.40 if result.root_cause_correct else 0.0
        remediation_score = 0.30 if (
            result.success and len(result.wrong_services_hit) == 0
        ) else (0.15 if result.success else 0.0)
        time_score = max(0.0, 0.20 * (1 - max(0, result.steps_taken - 8) / 7))
        blast_score = 0.10 * result.blast_radius_score
        total = rc_score + remediation_score + time_score + blast_score
        return round(min(total, 1.0), 4)


class Task2_BadDeploy:
    id          = "task2_bad_deploy"
    name        = "The Bad Deploy"
    difficulty  = "medium"
    description = (
        "A bad deploy of auth-service is causing cascading 503 errors across checkout-service, "
        "payments-service, and api-gateway. Six alerts are firing. Identify the root cause by "
        "tracing the dependency graph and roll back the correct service. Restarting downstream "
        "services is a trap — they will only recover after auth-service is fixed."
    )
    max_steps            = 25
    time_limit_minutes   = 60
    correct_service      = "auth-service"
    correct_action       = "rollback_deploy"
    correct_root_cause   = "deploy"

    def setup(self, sim: ServiceSimulator) -> None:
        sim.set_bad_deploy("auth-service", bad_sha="bad0000")
        sim.cascade_from("auth-service")
        sim.add_noise_to_service("analytics-service")
        sim.add_noise_to_service("data-pipeline-service")

    def initial_alerts(self) -> List[Alert]:
        return [
            Alert(id="b001", service="api-gateway", severity=AlertSeverity.CRITICAL,
                  message="api-gateway error rate 9.4/s — returning 503 to clients",
                  metric="error_rate", value=9.4, threshold=1.0, fired_at=0, is_noise=False),
            Alert(id="b002", service="checkout-service", severity=AlertSeverity.CRITICAL,
                  message="checkout-service error rate 11.2/s — auth dependency failing",
                  metric="error_rate", value=11.2, threshold=1.0, fired_at=0, is_noise=False),
            Alert(id="b003", service="payments-service", severity=AlertSeverity.CRITICAL,
                  message="payments-service error rate 8.7/s — auth dependency failing",
                  metric="error_rate", value=8.7, threshold=1.0, fired_at=0, is_noise=False),
            Alert(id="b004", service="auth-service", severity=AlertSeverity.CRITICAL,
                  message="auth-service returning HTTP 500 on /auth/validate — 2340 rpm failing",
                  metric="error_rate", value=18.7, threshold=1.0, fired_at=0, is_noise=False),
            Alert(id="n101", service="analytics-service", severity=AlertSeverity.WARNING,
                  message="analytics-service CPU high — long-running aggregation query",
                  metric="cpu_percent", value=81.0, threshold=70.0, fired_at=1, is_noise=True),
            Alert(id="n102", service="data-pipeline-service", severity=AlertSeverity.WARNING,
                  message="data-pipeline-service memory elevated — large dataset in memory",
                  metric="memory_percent", value=73.0, threshold=70.0, fired_at=2, is_noise=True),
            Alert(id="n103", service="notification-service", severity=AlertSeverity.INFO,
                  message="notification-service queue depth 1840 — delivery delayed",
                  metric="queue_depth", value=1840.0, threshold=1000.0, fired_at=0, is_noise=True),
            Alert(id="n104", service="inventory-service", severity=AlertSeverity.INFO,
                  message="inventory-service p99 elevated — unrelated index rebuild",
                  metric="p99_latency_ms", value=52.0, threshold=30.0, fired_at=0, is_noise=True),
        ]

    def get_logs(self, service: str, t: int):
        if service == "auth-service":
            return bad_deploy_logs("auth-service", t)
        if service in ("checkout-service", "payments-service", "api-gateway"):
            return cascade_logs(service, t)
        if service == "analytics-service":
            return noise_logs_cpu("analytics-service", t)
        return []

    def grade(self, result: EpisodeResult) -> float:
        rc_score = 0.35 if result.root_cause_correct else 0.0
        rem_score = 0.25 if result.success else 0.0
        blast_score = 0.20 * result.blast_radius_score
        time_score = max(0.0, 0.15 * (1 - max(0, result.steps_taken - 14) / 11))
        topology_bonus = 0.05 if "topology" in result.notes else 0.0
        total = rc_score + rem_score + blast_score + time_score + topology_bonus
        return round(min(total, 1.0), 4)


class Task3_PhantomSlowdown:
    id          = "task3_phantom_slowdown"
    name        = "The Phantom Slowdown"
    difficulty  = "hard"
    description = (
        "checkout-service p99 latency spikes every ~15 simulated minutes. There are no crashes, "
        "no obvious errors — only intermittent slowness. Root cause: data-pipeline-service runs a "
        "heavy cron job that table-locks a shared PostgreSQL table, starving checkout-service queries. "
        "Eight distractor alerts are firing. You must correlate time-series patterns across services "
        "and identify the cron schedule. Correct fix: scale up data-pipeline-service to reduce lock "
        "duration, OR page the team with the correct root cause."
    )
    max_steps            = 40
    time_limit_minutes   = 120
    correct_service      = "data-pipeline-service"
    correct_action       = "scale_up"
    correct_root_cause   = "cron"
    CRON_PERIOD          = 15
    correct_root_cause_keywords = ["cron", "data-pipeline", "lock", "query", "periodic"]

    def setup(self, sim: ServiceSimulator) -> None:
        sim.add_noise_to_service("analytics-service")
        sim.add_noise_to_service("notification-service")
        sim.services["data-pipeline-service"]["cpu"] = 48.0

    def initial_alerts(self) -> List[Alert]:
        return [
            Alert(id="c001", service="checkout-service", severity=AlertSeverity.WARNING,
                  message="checkout-service p99 spike: 4200ms (resolved 12min ago, 2 occurrences in 30min)",
                  metric="p99_latency_ms", value=4200.0, threshold=500.0, fired_at=0, is_noise=False),
            Alert(id="c002", service="checkout-service", severity=AlertSeverity.INFO,
                  message="checkout-service DB query timeout (intermittent) — table 'orders'",
                  metric="db_timeout_count", value=14.0, threshold=5.0, fired_at=0, is_noise=False),
            Alert(id="n201", service="analytics-service", severity=AlertSeverity.WARNING,
                  message="analytics-service p99 high: 610ms — ongoing",
                  metric="p99_latency_ms", value=610.0, threshold=400.0, fired_at=0, is_noise=True),
            Alert(id="n202", service="analytics-service", severity=AlertSeverity.WARNING,
                  message="analytics-service CPU 72% — model retraining pipeline running",
                  metric="cpu_percent", value=72.0, threshold=70.0, fired_at=3, is_noise=True),
            Alert(id="n203", service="notification-service", severity=AlertSeverity.WARNING,
                  message="notification-service p99 elevated: 290ms (SLA threshold 200ms)",
                  metric="p99_latency_ms", value=290.0, threshold=200.0, fired_at=0, is_noise=True),
            Alert(id="n204", service="notification-service", severity=AlertSeverity.INFO,
                  message="notification-service delivery retry rate 4.2% — upstream delay",
                  metric="retry_rate", value=4.2, threshold=3.0, fired_at=1, is_noise=True),
            Alert(id="n205", service="inventory-service", severity=AlertSeverity.INFO,
                  message="inventory-service cache hit rate dropped: 71% (baseline 85%)",
                  metric="cache_hit_rate", value=71.0, threshold=80.0, fired_at=0, is_noise=True),
            Alert(id="n206", service="inventory-service", severity=AlertSeverity.INFO,
                  message="inventory-service scheduled index rebuild in progress",
                  metric="index_build", value=1.0, threshold=0.0, fired_at=0, is_noise=True),
            Alert(id="n207", service="auth-service", severity=AlertSeverity.INFO,
                  message="auth-service token refresh rate elevated — expected after morning login wave",
                  metric="token_refresh_rate", value=840.0, threshold=600.0, fired_at=0, is_noise=True),
            Alert(id="n208", service="data-pipeline-service", severity=AlertSeverity.INFO,
                  message="data-pipeline-service memory 71% — within normal range for batch workloads",
                  metric="memory_percent", value=71.0, threshold=80.0, fired_at=0, is_noise=True),
        ]

    def get_logs(self, service: str, t: int, is_spike_phase: bool = False):
        if service == "data-pipeline-service":
            return cron_lock_logs("data-pipeline-service", "checkout-service", t, is_spike_phase)
        if service == "checkout-service":
            if is_spike_phase:
                return cron_lock_logs("data-pipeline-service", "checkout-service", t, True)[2:]
            return [
                type('L', (), {
                    "timestamp": t, "service": service,
                    "level": "INFO",
                    "message": "DB queries nominal (between spike windows)"
                })()
            ]
        if service == "analytics-service":
            return noise_logs_cpu("analytics-service", t)
        return []

    def grade(self, result: EpisodeResult) -> float:
        rc_score = 0.30 if result.root_cause_correct else 0.0
        rem_score = 0.25 if result.success else 0.0
        pm_score = 0.20 * result.postmortem_score
        time_score = max(0.0, 0.15 * (1 - max(0, result.steps_taken - 25) / 15))
        blast_score = 0.10 * result.blast_radius_score
        total = rc_score + rem_score + pm_score + time_score + blast_score
        return round(min(total, 1.0), 4)


TASKS = {
    "task1_oom":            Task1_OOM,
    "task2_bad_deploy":     Task2_BadDeploy,
    "task3_phantom_slowdown": Task3_PhantomSlowdown,
}

ALL_TASK_IDS = list(TASKS.keys())


def get_task(task_id: str):
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}. Valid: {', '.join(ALL_TASK_IDS)}")
    return TASKS[task_id]()
