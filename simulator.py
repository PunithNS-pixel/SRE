"""
SRE-Bench Simulator
===================
Simulates a realistic 8-service microservices platform.
Each incident scenario mutates this base state to create
the conditions an agent must diagnose and fix.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from .models import (
    Alert, AlertSeverity, LogEntry, ServiceMetrics, ServiceNode
)


# ── Base service catalog ──────────────────────────────────────────────────────

BASE_SERVICES: Dict[str, dict] = {
    "api-gateway": {
        "tier": "frontend",
        "depends_on": ["auth-service", "checkout-service"],
        "cpu": 28.0, "memory": 41.0, "error_rate": 0.01,
        "p50": 12.0, "p99": 45.0, "rps": 820.0, "replicas": 3,
        "sha": "abc1234",
    },
    "auth-service": {
        "tier": "backend",
        "depends_on": [],
        "cpu": 22.0, "memory": 38.0, "error_rate": 0.00,
        "p50": 8.0,  "p99": 28.0, "rps": 410.0, "replicas": 2,
        "sha": "def5678",
    },
    "checkout-service": {
        "tier": "backend",
        "depends_on": ["auth-service", "payments-service", "inventory-service"],
        "cpu": 35.0, "memory": 52.0, "error_rate": 0.02,
        "p50": 18.0, "p99": 62.0, "rps": 280.0, "replicas": 2,
        "sha": "ghi9012",
    },
    "payments-service": {
        "tier": "backend",
        "depends_on": ["auth-service"],
        "cpu": 31.0, "memory": 58.0, "error_rate": 0.01,
        "p50": 22.0, "p99": 78.0, "rps": 190.0, "replicas": 2,
        "sha": "jkl3456",
    },
    "inventory-service": {
        "tier": "backend",
        "depends_on": [],
        "cpu": 19.0, "memory": 33.0, "error_rate": 0.00,
        "p50": 6.0,  "p99": 21.0, "rps": 340.0, "replicas": 2,
        "sha": "mno7890",
    },
    "analytics-service": {
        "tier": "data",
        "depends_on": ["data-pipeline-service"],
        "cpu": 55.0, "memory": 61.0, "error_rate": 0.00,
        "p50": 120.0, "p99": 380.0, "rps": 40.0, "replicas": 1,
        "sha": "pqr1234",
    },
    "data-pipeline-service": {
        "tier": "data",
        "depends_on": [],
        "cpu": 42.0, "memory": 49.0, "error_rate": 0.00,
        "p50": 200.0, "p99": 620.0, "rps": 8.0, "replicas": 1,
        "sha": "stu5678",
    },
    "notification-service": {
        "tier": "infra",
        "depends_on": [],
        "cpu": 12.0, "memory": 24.0, "error_rate": 0.00,
        "p50": 4.0,  "p99": 18.0, "rps": 60.0, "replicas": 1,
        "sha": "vwx9012",
    },
}

# How many simulated minutes each action costs
ACTION_TIME_COST: Dict[str, int] = {
    "read_logs":        2,
    "get_metrics":      1,
    "get_topology":     1,
    "restart_service":  4,
    "scale_up":         3,
    "rollback_deploy":  6,
    "page_team":        1,
    "mark_resolved":    0,
    "write_postmortem": 2,
}


class ServiceSimulator:
    """
    Maintains the live state of the simulated microservices system.
    Incident scenarios call mutate_*() methods to inject failures.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.services: Dict[str, dict] = copy.deepcopy(BASE_SERVICES)
        for svc in self.services.values():
            svc["healthy"] = True
            svc["has_pending_deploy"] = False
        self.simulated_time: int = 0
        self.blast_radius_events: List[str] = []   # services broken by wrong actions
        self.action_log: List[dict] = []            # every action taken

    # ── State readers ─────────────────────────────────────────────────────────

    def get_metrics(self, service: str) -> Optional[ServiceMetrics]:
        svc = self.services.get(service)
        if not svc:
            return None
        jitter = lambda v, pct=0.03: round(v * (1 + self._rng.uniform(-pct, pct)), 2)
        return ServiceMetrics(
            cpu_percent         = jitter(svc["cpu"]),
            memory_percent      = jitter(svc["memory"]),
            error_rate          = jitter(svc["error_rate"], 0.1),
            p50_latency_ms      = jitter(svc["p50"]),
            p99_latency_ms      = jitter(svc["p99"]),
            requests_per_second = jitter(svc["rps"]),
            replica_count       = svc["replicas"],
            last_deploy_sha     = svc["sha"],
            healthy             = svc["healthy"],
        )

    def get_topology(self) -> List[ServiceNode]:
        return [
            ServiceNode(
                name       = name,
                depends_on = svc["depends_on"],
                healthy    = svc["healthy"],
                tier       = svc["tier"],
            )
            for name, svc in self.services.items()
        ]

    def service_exists(self, name: str) -> bool:
        return name in self.services

    def is_healthy(self, name: str) -> bool:
        return self.services.get(name, {}).get("healthy", False)

    # ── Mutation helpers (used by incident scenarios) ─────────────────────────

    def set_oom(self, service: str) -> None:
        """Simulate OOM: memory maxed, high errors, unhealthy."""
        svc = self.services[service]
        svc["memory"]     = 99.8
        svc["cpu"]        = 95.0
        svc["error_rate"] = 12.4
        svc["p99"]        = 8000.0
        svc["healthy"]    = False

    def set_bad_deploy(self, service: str, bad_sha: str = "bad0000") -> None:
        """Simulate a broken deploy: high error rate, low latency (fast fails)."""
        svc = self.services[service]
        svc["sha"]        = bad_sha
        svc["error_rate"] = 18.7
        svc["p50"]        = 3.0
        svc["p99"]        = 11.0
        svc["healthy"]    = False
        svc["has_pending_deploy"] = True

    def cascade_from(self, source: str) -> List[str]:
        """
        Mark all services that depend on `source` as degraded.
        Returns list of affected service names.
        """
        affected = []
        for name, svc in self.services.items():
            if source in svc["depends_on"] and name != source:
                svc["error_rate"] = round(svc["error_rate"] + 8.0 + self._rng.uniform(0, 3), 2)
                svc["p99"]        = round(svc["p99"] * 4.5, 1)
                svc["healthy"]    = False
                affected.append(name)
        return affected

    def set_cron_lock(self, cron_service: str, victim_service: str, phase: int) -> None:
        """
        Simulate a DB-lock caused by a cron job every 15 minutes (phase = 0..14).
        Victim service spikes at phase == 0.
        """
        svc_cron   = self.services[cron_service]
        svc_victim = self.services[victim_service]
        if phase == 0:
            svc_cron["cpu"]        = 88.0
            svc_cron["memory"]     = 71.0
            svc_victim["p99"]      = 4200.0
            svc_victim["error_rate"] = 3.1
            svc_victim["healthy"]  = False
        else:
            # Between spikes: both look mostly normal (that's the trick)
            svc_cron["cpu"]        = 44.0
            svc_victim["p99"]      = BASE_SERVICES[victim_service]["p99"]
            svc_victim["error_rate"] = BASE_SERVICES[victim_service]["error_rate"]
            svc_victim["healthy"]  = True

    def add_noise_to_service(self, service: str, reason: str = "unrelated") -> None:
        """Spike a metric on a service to create a distractor alert."""
        svc = self.services[service]
        svc["cpu"]    = min(svc["cpu"] + self._rng.uniform(25, 40), 99.0)
        svc["p99"]    = svc["p99"] * self._rng.uniform(1.8, 2.5)

    # ── Remediation actions ───────────────────────────────────────────────────

    def restart_service(self, service: str, correct_target: str) -> Tuple[bool, str]:
        """
        Returns (fixed_incident, message).
        If wrong service: adds blast radius event.
        """
        self.action_log.append({"action": "restart", "service": service})
        self.simulated_time += ACTION_TIME_COST["restart_service"]

        if service not in self.services:
            return False, f"Service '{service}' not found."

        if service == correct_target:
            svc = self.services[service]
            base = BASE_SERVICES[service]
            svc["memory"]     = base["memory"]
            svc["cpu"]        = base["cpu"]
            svc["error_rate"] = base["error_rate"]
            svc["p99"]        = base["p99"]
            svc["healthy"]    = True
            return True, f"✓ {service} restarted successfully. Metrics normalizing."
        else:
            # Restarting a healthy or wrong service creates a brief outage
            if self.services[service]["healthy"]:
                self.blast_radius_events.append(service)
                self.services[service]["healthy"] = False
                self.services[service]["error_rate"] += 4.0
                return False, (
                    f"⚠ {service} restarted but incident continues. "
                    f"{service} is now temporarily unavailable (blast radius)."
                )
            return False, f"{service} restarted. Incident still active."

    def rollback_deploy(self, service: str, correct_target: str) -> Tuple[bool, str]:
        self.action_log.append({"action": "rollback", "service": service})
        self.simulated_time += ACTION_TIME_COST["rollback_deploy"]

        if service not in self.services:
            return False, f"Service '{service}' not found."

        svc = self.services[service]
        if service == correct_target and svc.get("has_pending_deploy"):
            # Fix the root service and heal its dependents
            base = BASE_SERVICES[service]
            svc["sha"]        = base["sha"]
            svc["error_rate"] = base["error_rate"]
            svc["p50"]        = base["p50"]
            svc["p99"]        = base["p99"]
            svc["healthy"]    = True
            svc["has_pending_deploy"] = False
            # Cascade heal
            for name, s in self.services.items():
                if service in s["depends_on"]:
                    b = BASE_SERVICES[name]
                    s["error_rate"] = b["error_rate"]
                    s["p99"]        = b["p99"]
                    s["healthy"]    = True
            return True, f"✓ {service} rolled back to {base['sha']}. Downstream services recovering."
        elif not svc.get("has_pending_deploy"):
            return False, f"{service} has no recent deploy to roll back."
        else:
            return False, f"Rolled back {service} but incident continues."

    def scale_up(self, service: str, replicas: int, correct_target: str) -> Tuple[bool, str]:
        self.action_log.append({"action": "scale_up", "service": service, "replicas": replicas})
        self.simulated_time += ACTION_TIME_COST["scale_up"]

        if service not in self.services:
            return False, f"Service '{service}' not found."

        svc = self.services[service]
        svc["replicas"] = replicas
        if service == correct_target:
            base = BASE_SERVICES[service]
            svc["cpu"]        = base["cpu"] * 0.6
            svc["error_rate"] = base["error_rate"]
            svc["p99"]        = base["p99"] * 0.8
            svc["healthy"]    = True
            return True, f"✓ {service} scaled to {replicas} replicas. Load distributed, latency recovering."
        return False, f"{service} scaled to {replicas} replicas. Incident still active."

    def advance_time(self, minutes: int) -> None:
        self.simulated_time += minutes


# ── Log generators (per incident type) ───────────────────────────────────────

def oom_logs(service: str, t: int) -> List[LogEntry]:
    return [
        LogEntry(timestamp=t-8, service=service, level="WARN",
                 message="GC overhead limit exceeded, heap at 89%"),
        LogEntry(timestamp=t-5, service=service, level="ERROR",
                 message="java.lang.OutOfMemoryError: Java heap space"),
        LogEntry(timestamp=t-5, service=service, level="ERROR",
                 message="Killing container due to OOM. Exit code 137."),
        LogEntry(timestamp=t-4, service=service, level="FATAL",
                 message="Process terminated. Restart #1 of 3 attempted."),
        LogEntry(timestamp=t-2, service=service, level="FATAL",
                 message="Restart loop detected. CrashLoopBackOff."),
        LogEntry(timestamp=t,   service=service, level="ERROR",
                 message="Health check /health returning 503 — service not ready."),
    ]


def noise_logs_cpu(service: str, t: int) -> List[LogEntry]:
    return [
        LogEntry(timestamp=t-3, service=service, level="WARN",
                 message="CPU throttling detected, batch job running"),
        LogEntry(timestamp=t-1, service=service, level="INFO",
                 message="Batch aggregation job completed in 142s"),
    ]


def bad_deploy_logs(service: str, t: int) -> List[LogEntry]:
    return [
        LogEntry(timestamp=t-6, service=service, level="INFO",
                 message=f"Deploy bad0000 started by ci-bot"),
        LogEntry(timestamp=t-5, service=service, level="INFO",
                 message="Rolling update: 1/2 pods replaced"),
        LogEntry(timestamp=t-4, service=service, level="ERROR",
                 message="NullPointerException in JWTValidator.validate() line 88"),
        LogEntry(timestamp=t-4, service=service, level="ERROR",
                 message="Token validation failed: unexpected nil claim"),
        LogEntry(timestamp=t-3, service=service, level="ERROR",
                 message="HTTP 500 returned for POST /auth/validate — 2340 req/min failing"),
        LogEntry(timestamp=t-2, service=service, level="WARN",
                 message="Downstream services receiving 503 due to auth failures"),
    ]


def cascade_logs(service: str, t: int) -> List[LogEntry]:
    return [
        LogEntry(timestamp=t-2, service=service, level="ERROR",
                 message="upstream auth-service returning HTTP 500, retrying (1/3)"),
        LogEntry(timestamp=t-1, service=service, level="ERROR",
                 message="upstream auth-service returning HTTP 500, retrying (3/3)"),
        LogEntry(timestamp=t,   service=service, level="ERROR",
                 message="All retries exhausted. Returning 503 to caller."),
    ]


def cron_lock_logs(cron_service: str, victim: str, t: int, is_spike: bool) -> List[LogEntry]:
    if is_spike:
        return [
            LogEntry(timestamp=t, service=cron_service, level="INFO",
                     message="[CRON] nightly_aggregate_report job started"),
            LogEntry(timestamp=t, service=cron_service, level="INFO",
                     message="Executing: SELECT * FROM orders JOIN inventory ... (full scan)"),
            LogEntry(timestamp=t, service=victim,       level="WARN",
                     message="DB query waiting for lock on table 'orders' — 8400ms"),
            LogEntry(timestamp=t, service=victim,       level="ERROR",
                     message="DB query timeout after 10000ms. Returning 504 to client."),
        ]
    return [
        LogEntry(timestamp=t, service=victim, level="INFO",
                 message="DB queries nominal, avg 18ms"),
    ]
