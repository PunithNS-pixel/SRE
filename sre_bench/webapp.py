from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import Action, ActionType, ALL_TASK_IDS, SREBenchEnv
from .rl import (
    RLEpisodePolicy,
    autoplay_episode,
    load_scenarios_from_dataset,
    train_policy_from_scenarios,
)
from .ui import build_ui


def _to_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    return value


class ResetRequest(BaseModel):
    task_id: str = Field(default=ALL_TASK_IDS[0])
    seed: int = 42


class StepRequest(BaseModel):
    episode_id: str
    action_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StateRequest(BaseModel):
    episode_id: str


app = FastAPI(title="SRE-Bench", version="1.0.0")
_sessions: Dict[str, SREBenchEnv] = {}
_rl_policies: Dict[str, RLEpisodePolicy] = {"default": RLEpisodePolicy()}


_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "get_topology",
                "get_metrics",
                "read_logs",
                "restart_service",
                "scale_up",
                "rollback_deploy",
                "page_team",
                "mark_resolved",
                "write_postmortem",
            ],
        },
        "params": {"type": "object"},
    },
    "required": ["action_type"],
    "additionalProperties": False,
}

_OBSERVATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "alerts": {"type": "array"},
        "visible_metrics": {"type": "object"},
        "visible_logs": {"type": "object"},
        "topology": {"type": ["array", "null"]},
        "simulated_time_min": {"type": "integer"},
        "step": {"type": "integer"},
        "incident_active": {"type": "boolean"},
        "last_action_result": {"type": "string"},
    },
    "required": [
        "alerts",
        "visible_metrics",
        "visible_logs",
        "topology",
        "simulated_time_min",
        "step",
        "incident_active",
        "last_action_result",
    ],
    "additionalProperties": True,
}

_STATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "step": {"type": "integer"},
        "simulated_time_min": {"type": "integer"},
        "incident_active": {"type": "boolean"},
        "resolved": {"type": "boolean"},
    },
    "required": ["task_id", "step", "simulated_time_min", "incident_active", "resolved"],
    "additionalProperties": True,
}


class RLTrainRequest(BaseModel):
    model_name: str = "default"
    dataset_path: str = "demo/demo_dataset.json"
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    gamma: float = 0.95
    alpha: float = 0.35


class RLSuggestRequest(BaseModel):
    model_name: str = "default"
    task_id: str
    step: int = 0


class RLAutoplayRequest(BaseModel):
    model_name: str = "default"
    task_id: str = Field(default=ALL_TASK_IDS[0])
    seed: int = 42
    max_actions: int = 8


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> Dict[str, str]:
    return {
        "name": "SRE-Bench",
        "description": "An OpenEnv-compatible incident response environment for evaluating AI SRE agents.",
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": _ACTION_SCHEMA,
        "observation": _OBSERVATION_SCHEMA,
        "state": _STATE_SCHEMA,
    }


@app.post("/mcp")
def mcp_root() -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "status": "ok",
        },
    }


@app.post("/reset")
def reset_episode(req: ResetRequest) -> Dict[str, Any]:
    if req.task_id not in ALL_TASK_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")

    env = SREBenchEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    episode_id = str(uuid4())
    _sessions[episode_id] = env
    return {
        "episode_id": episode_id,
        "task_id": req.task_id,
        "observation": _to_json(obs),
    }


@app.post("/step")
def step_episode(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="episode_id not found")

    try:
        action = Action(action_type=ActionType(req.action_type), params=req.params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {exc}") from exc

    obs, reward, done, info = env.step(action)
    return {
        "observation": _to_json(obs),
        "reward": _to_json(reward),
        "done": done,
        "info": _to_json(info),
    }


@app.post("/state")
def get_state(req: StateRequest) -> Dict[str, Any]:
    env = _sessions.get(req.episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="episode_id not found")
    return {"state": _to_json(env.state())}


@app.post("/rl/train")
def rl_train(req: RLTrainRequest) -> Dict[str, Any]:
    if req.scenarios:
        scenarios = req.scenarios
    else:
        try:
            scenarios = load_scenarios_from_dataset(req.dataset_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {exc}") from exc

    if not scenarios:
        raise HTTPException(status_code=400, detail="No scenarios provided for training")

    try:
        policy, metrics = train_policy_from_scenarios(scenarios, gamma=req.gamma, alpha=req.alpha)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"RL training failed: {exc}") from exc

    _rl_policies[req.model_name] = policy
    return {
        "model_name": req.model_name,
        "metrics": metrics,
    }


@app.post("/rl/suggest")
def rl_suggest(req: RLSuggestRequest) -> Dict[str, Any]:
    policy = _rl_policies.get(req.model_name)
    if policy is None:
        raise HTTPException(status_code=404, detail=f"RL model '{req.model_name}' not found")

    if req.task_id not in ALL_TASK_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")

    action = policy.suggest(req.task_id, req.step)
    confidence = policy.confidence(req.task_id, req.step)
    return {
        "model_name": req.model_name,
        "task_id": req.task_id,
        "step": req.step,
        "action": action,
        "confidence": confidence,
    }


@app.post("/rl/autoplay")
def rl_autoplay(req: RLAutoplayRequest) -> Dict[str, Any]:
    policy = _rl_policies.get(req.model_name)
    if policy is None:
        raise HTTPException(status_code=404, detail=f"RL model '{req.model_name}' not found")

    if req.task_id not in ALL_TASK_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")

    try:
        result = autoplay_episode(
            policy=policy,
            task_id=req.task_id,
            seed=req.seed,
            max_actions=req.max_actions,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Autoplay failed: {exc}") from exc

    return result


ui = build_ui()
app = gr.mount_gradio_app(app, ui, path="/")
