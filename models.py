"""
SRE-Bench: Typed models for the OpenEnv interface.
Uses pydantic v2 when available; falls back to stdlib dataclasses.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field as PField
    def _field(default=dataclasses.MISSING, default_factory=None, **kw):
        if default_factory is not None:
            return PField(default_factory=default_factory, **kw)
        if default is not dataclasses.MISSING:
            return PField(default=default, **kw)
        return PField(**kw)
    Field = _field
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False
    def Field(default=dataclasses.MISSING, default_factory=None, **kw):
        if default_factory is not None:
            return dataclasses.field(default_factory=default_factory)
        if default is not dataclasses.MISSING:
            return dataclasses.field(default=default)
        return dataclasses.field()

    @dataclasses.dataclass
    class _Base:
        def dict(self):
            return dataclasses.asdict(self)
        def model_dump(self):
            return self.dict()
    BaseModel = _Base


# ── Enums ────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"


class ActionType(str, Enum):
    READ_LOGS        = "read_logs"
    GET_METRICS      = "get_metrics"
    GET_TOPOLOGY     = "get_topology"
    RESTART_SERVICE  = "restart_service"
    SCALE_UP         = "scale_up"
    ROLLBACK_DEPLOY  = "rollback_deploy"
    PAGE_TEAM        = "page_team"
    MARK_RESOLVED    = "mark_resolved"
    WRITE_POSTMORTEM = "write_postmortem"


# ── Models ────────────────────────────────────────────────────────────────────

if _PYDANTIC:
    class Alert(BaseModel):
        id:        str
        service:   str
        severity:  AlertSeverity
        message:   str
        metric:    str
        value:     float
        threshold: float
        fired_at:  int
        is_noise:  bool = False

    class ServiceMetrics(BaseModel):
        cpu_percent:         float
        memory_percent:      float
        error_rate:          float
        p50_latency_ms:      float
        p99_latency_ms:      float
        requests_per_second: float
        replica_count:       int
        last_deploy_sha:     str
        healthy:             bool

    class LogEntry(BaseModel):
        timestamp: int
        service:   str
        level:     str
        message:   str

    class ServiceNode(BaseModel):
        name:       str
        depends_on: List[str]
        healthy:    bool
        tier:       str

    class Observation(BaseModel):
        alerts:             List[Alert]
        visible_metrics:    Dict[str, ServiceMetrics] = Field(default_factory=dict)
        visible_logs:       Dict[str, List[LogEntry]] = Field(default_factory=dict)
        topology:           Optional[List[ServiceNode]] = None
        simulated_time_min: int  = 0
        step:               int  = 0
        incident_active:    bool = True
        last_action_result: str  = ""

    class Action(BaseModel):
        action_type: ActionType
        params:      Dict[str, Any] = Field(default_factory=dict)

    class Reward(BaseModel):
        step_reward:          float = 0.0
        time_penalty:         float = 0.0
        blast_radius_penalty: float = 0.0
        resolution_bonus:     float = 0.0
        postmortem_bonus:     float = 0.0
        cumulative_reward:    float = 0.0
        episode_score:        float = 0.0
        reason:               str   = ""

    class EpisodeResult(BaseModel):
        task_id:            str
        success:            bool
        episode_score:      float
        steps_taken:        int
        simulated_minutes:  int
        root_cause_correct: bool
        blast_radius_score: float
        time_efficiency:    float
        postmortem_score:   float
        wrong_services_hit: List[str] = Field(default_factory=list)
        notes:              str = ""

else:
    # ── Dataclass fallback (no pydantic) ─────────────────────────────────────
    @dataclasses.dataclass
    class Alert:
        id:        str
        service:   str
        severity:  AlertSeverity
        message:   str
        metric:    str
        value:     float
        threshold: float
        fired_at:  int
        is_noise:  bool = False
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class ServiceMetrics:
        cpu_percent: float; memory_percent: float; error_rate: float
        p50_latency_ms: float; p99_latency_ms: float
        requests_per_second: float; replica_count: int
        last_deploy_sha: str; healthy: bool
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class LogEntry:
        timestamp: int; service: str; level: str; message: str
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class ServiceNode:
        name: str; depends_on: dataclasses.field(default_factory=list)
        healthy: bool = True; tier: str = ""
        def __post_init__(self):
            if not isinstance(self.depends_on, list):
                self.depends_on = list(self.depends_on)
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class Observation:
        alerts:             List[Any]
        visible_metrics:    dict = dataclasses.field(default_factory=dict)
        visible_logs:       dict = dataclasses.field(default_factory=dict)
        topology:           Any  = None
        simulated_time_min: int  = 0
        step:               int  = 0
        incident_active:    bool = True
        last_action_result: str  = ""
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class Action:
        action_type: ActionType
        params:      dict = dataclasses.field(default_factory=dict)
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class Reward:
        step_reward: float = 0.0; time_penalty: float = 0.0
        blast_radius_penalty: float = 0.0; resolution_bonus: float = 0.0
        postmortem_bonus: float = 0.0; cumulative_reward: float = 0.0
        episode_score: float = 0.0; reason: str = ""
        def dict(self): return dataclasses.asdict(self)

    @dataclasses.dataclass
    class EpisodeResult:
        task_id: str; success: bool; episode_score: float
        steps_taken: int; simulated_minutes: int
        root_cause_correct: bool; blast_radius_score: float
        time_efficiency: float; postmortem_score: float
        wrong_services_hit: List[str] = dataclasses.field(default_factory=list)
        notes: str = ""
        def dict(self): return dataclasses.asdict(self)
