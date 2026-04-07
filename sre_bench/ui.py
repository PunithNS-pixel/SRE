from __future__ import annotations

from typing import Any, Dict, Tuple

from . import ALL_TASK_IDS, Action, ActionType, Observation, SREBenchEnv
from .rl import RLEpisodePolicy, autoplay_episode, load_scenarios_from_dataset, train_policy_from_scenarios


ACTION_LABELS = [
    ("Investigate: get_topology", ActionType.GET_TOPOLOGY.value),
    ("Investigate: get_metrics", ActionType.GET_METRICS.value),
    ("Investigate: read_logs", ActionType.READ_LOGS.value),
    ("Remediate: restart_service", ActionType.RESTART_SERVICE.value),
    ("Remediate: rollback_deploy", ActionType.ROLLBACK_DEPLOY.value),
    ("Remediate: scale_up", ActionType.SCALE_UP.value),
    ("Coordinate: page_team", ActionType.PAGE_TEAM.value),
    ("Close: mark_resolved", ActionType.MARK_RESOLVED.value),
    ("Close: write_postmortem", ActionType.WRITE_POSTMORTEM.value),
]


def _format_last_action_result(text: str) -> str:
    if not text:
        return ""
    if "\n" in text:
        return f"### Last Action Result\n```text\n{text}\n```"
    return f"### Last Action Result\n- {text}"


def _step_hint_from_state(state: Dict[str, Any] | None) -> int:
    if not state:
        return 0
    obs = state.get("obs")
    if obs is None:
        return 0
    return int(getattr(obs, "step", 0))


def _action_to_form_fields(action: Dict[str, Any]) -> Tuple[str, str, int, str, str, str, str]:
    action_type = action.get("action_type", ActionType.GET_TOPOLOGY.value)
    params = action.get("params", {})
    service = params.get("service", "")
    replicas = int(params.get("replicas", 4) or 4)
    message = params.get("message", "")
    root_cause = params.get("root_cause", "")
    timeline = params.get("timeline", "")
    prevention = params.get("prevention", "")
    return action_type, service, replicas, message, root_cause, timeline, prevention


def rl_train_ui(
    model_name: str,
    dataset_path: str,
    gamma: float,
    alpha: float,
    rl_state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    model_name = (model_name or "default").strip()
    dataset_path = (dataset_path or "demo/demo_dataset.json").strip()
    try:
        scenarios = load_scenarios_from_dataset(dataset_path)
        policy, metrics = train_policy_from_scenarios(scenarios, gamma=gamma, alpha=alpha)
    except Exception as exc:
        return f"### RL Training Failed\n- {exc}", {}, rl_state

    new_state = {
        "policies": dict(rl_state.get("policies", {})),
        "last_model": model_name,
    }
    new_state["policies"][model_name] = policy
    status = (
        "### RL Model Trained\n"
        f"- Model: **{model_name}**\n"
        f"- Scenarios: **{metrics.get('scenarios_trained', 0)}**\n"
        f"- Learned states: **{metrics.get('states', 0)}**\n"
        f"- Q entries: **{metrics.get('action_entries', 0)}**"
    )
    return status, metrics, new_state


def rl_suggest_ui(
    model_name: str,
    task_id: str,
    step_override: int,
    state: Dict[str, Any],
    rl_state: Dict[str, Any],
) -> Tuple[str, str, str, int, str, str, str, str]:
    model_name = (model_name or rl_state.get("last_model") or "default").strip()
    policies = rl_state.get("policies", {})
    policy = policies.get(model_name)
    if policy is None:
        return (
            f"### RL Suggestion Unavailable\n- Model '{model_name}' is not trained yet.",
            ActionType.GET_TOPOLOGY.value,
            "",
            4,
            "",
            "",
            "",
            "",
        )

    step = int(step_override) if int(step_override) >= 0 else 0
    if step == 0:
        step = _step_hint_from_state(state)

    action = policy.suggest(task_id, step)
    confidence = policy.confidence(task_id, step)
    action_type, service, replicas, message, root_cause, timeline, prevention = _action_to_form_fields(action)
    md = (
        "### Suggested Next Action\n"
        f"- Model: **{model_name}**\n"
        f"- Task: **{task_id}**\n"
        f"- Step used: **{step}**\n"
        f"- Action: **{action_type}**\n"
        f"- Confidence: **{confidence:.2f}**"
    )
    return md, action_type, service, replicas, message, root_cause, timeline, prevention


def rl_autoplay_ui(
    model_name: str,
    task_id: str,
    seed: int,
    max_actions: int,
    rl_state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    model_name = (model_name or rl_state.get("last_model") or "default").strip()
    policies = rl_state.get("policies", {})
    policy = policies.get(model_name)
    if policy is None:
        return f"### RL Auto-Run Unavailable\n- Model '{model_name}' is not trained yet.", {}

    try:
        result = autoplay_episode(policy=policy, task_id=task_id, seed=int(seed), max_actions=int(max_actions))
    except Exception as exc:
        return f"### RL Auto-Run Failed\n- {exc}", {}

    ep = result.get("episode_result") or {}
    summary = (
        "### RL Auto-Run Result\n"
        f"- Model: **{model_name}**\n"
        f"- Task: **{task_id}**\n"
        f"- Completed: **{result.get('done', False)}**\n"
        f"- Success: **{ep.get('success', False)}**\n"
        f"- Episode score: **{ep.get('episode_score', 0)}**\n"
        f"- Steps taken: **{ep.get('steps_taken', 0)}**"
    )
    return summary, result


def judge_demo_mode_ui(
    model_name: str,
    dataset_path: str,
    gamma: float,
    alpha: float,
    seed: int,
    max_actions: int,
    rl_state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    model_name = (model_name or "hackathon_v1").strip()
    dataset_path = (dataset_path or "demo/meta_hackathon_dataset.template.json").strip()

    try:
        scenarios = load_scenarios_from_dataset(dataset_path)
        policy, train_metrics = train_policy_from_scenarios(scenarios, gamma=gamma, alpha=alpha)
    except Exception as exc:
        return f"### Judge Demo Failed\n- Training error: {exc}", {}, rl_state

    new_state = {
        "policies": dict(rl_state.get("policies", {})),
        "last_model": model_name,
    }
    new_state["policies"][model_name] = policy

    pass_thresholds = {
        "task1_oom": 0.60,
        "task2_bad_deploy": 0.55,
        "task3_phantom_slowdown": 0.45,
    }

    runs = []
    passed = 0
    for task_id in ALL_TASK_IDS:
        result = autoplay_episode(policy=policy, task_id=task_id, seed=int(seed), max_actions=int(max_actions))
        ep = result.get("episode_result") or {}
        score = float(ep.get("episode_score", 0.0) or 0.0)
        threshold = pass_thresholds.get(task_id, 0.5)
        ok = score >= threshold and bool(ep.get("success", False))
        if ok:
            passed += 1
        runs.append(
            {
                "task_id": task_id,
                "success": bool(ep.get("success", False)),
                "score": round(score, 4),
                "threshold": threshold,
                "passed": ok,
                "steps_taken": ep.get("steps_taken", 0),
            }
        )

    total = len(runs)
    report = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "training": train_metrics,
        "benchmark": runs,
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
            "seed": int(seed),
            "max_actions": int(max_actions),
        },
    }

    md = (
        "### Judge Demo Report Card\n"
        f"- Model: **{model_name}**\n"
        f"- Dataset: **{dataset_path}**\n"
        f"- Trained on scenarios: **{train_metrics.get('scenarios_trained', 0)}**\n"
        f"- Benchmark pass: **{passed}/{total}**\n"
        f"- Pass rate: **{report['summary']['pass_rate'] * 100:.1f}%**"
    )
    return md, report, new_state


def fmt_obs(obs: Observation) -> str:
    lines = [
        "## Incident Snapshot",
        f"- Step: **{obs.step}**",
        f"- Simulated time: **{obs.simulated_time_min} min**",
        f"- Status: **{'Incident Active (🔴)' if obs.incident_active else 'Resolved (✅)'}**",
        "",
        "## Active Alerts",
    ]
    for a in obs.alerts:
        icon = "🔴" if a.severity == "critical" else ("🟡" if a.severity == "warning" else "🔵")
        lines.append(f"- {icon} **{a.service}**: {a.message}")

    if obs.topology:
        lines.append("\n## Service Topology")
        for node in obs.topology:
            status = "✅" if node.healthy else "❌"
            deps = ", ".join(node.depends_on) if node.depends_on else "none"
            lines.append(f"- {status} **{node.name}** depends on: {deps}")

    if obs.visible_metrics:
        lines.append("\n## Metrics")
        lines.append("| Service | Health | CPU | Memory | Error Rate | P99 | Deploy SHA |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for svc, m in obs.visible_metrics.items():
            health = "✅" if m.healthy else "❌"
            lines.append(
                f"| {svc} | {health} | {m.cpu_percent:.1f}% | {m.memory_percent:.1f}% | "
                f"{m.error_rate:.2f}/s | {m.p99_latency_ms:.0f}ms | {m.last_deploy_sha} |"
            )

    if obs.visible_logs:
        lines.append("\n## Logs")
        for svc, entries in obs.visible_logs.items():
            lines.append(f"### {svc}")
            lines.append("```text")
            for e in entries[-6:]:
                icon = "🔴" if e.level in ("FATAL", "ERROR") else ("🟡" if e.level == "WARN" else "⚪")
                lines.append(f"{icon} [{e.level}] t={e.timestamp}m - {e.message}")
            lines.append("```")

    if obs.last_action_result:
        lines.append(f"\n{_format_last_action_result(obs.last_action_result)}")

    return "\n".join(lines)


def parse_action_inputs(
    action_type: str,
    service: str,
    replicas: int,
    message: str,
    root_cause: str,
    timeline: str,
    prevention: str,
) -> Action:
    atype = ActionType(action_type)
    params: dict = {}
    if atype in (ActionType.GET_METRICS, ActionType.READ_LOGS,
                 ActionType.RESTART_SERVICE, ActionType.ROLLBACK_DEPLOY):
        params["service"] = service.strip()
    if atype == ActionType.READ_LOGS:
        params["window_minutes"] = 10
    if atype == ActionType.SCALE_UP:
        params["service"]  = service.strip()
        params["replicas"] = replicas
    if atype == ActionType.PAGE_TEAM:
        params["message"] = message.strip()
    if atype == ActionType.MARK_RESOLVED:
        params["root_cause"] = root_cause.strip()
    if atype == ActionType.WRITE_POSTMORTEM:
        params["root_cause"] = root_cause.strip()
        params["timeline"]   = timeline.strip()
        params["prevention"] = prevention.strip()
    return Action(action_type=atype, params=params)


def start_episode(task_id: str) -> Tuple[str, str, dict]:
    env   = SREBenchEnv(task_id=task_id)
    obs   = env.reset()
    state = {"env": env, "obs": obs, "log": [], "done": False, "score": 0.0}
    msg = (
        "### Episode Started\n"
        "Follow this flow: **Investigate -> Confirm Root Cause -> Remediate -> Close**.\n"
        "Use `get_topology`, `read_logs`, and `get_metrics` first, then apply one remediation action."
    )
    return fmt_obs(obs), msg, state


def take_action(
    action_type, service, replicas, message, root_cause, timeline, prevention, state
) -> Tuple[str, str, dict, dict]:
    if not state or state.get("done"):
        return "Episode not started or already finished.", "", {}, state or {}

    env = state["env"]
    try:
        action = parse_action_inputs(action_type, service, replicas,
                                     message, root_cause, timeline, prevention)
    except Exception as e:
        safe_obs = fmt_obs(state.get("obs")) if state.get("obs") else ""
        return safe_obs, f"### Invalid Action\n- {e}", {"score": state.get("score", 0)}, state

    obs, reward, done, info = env.step(action)
    state["obs"] = obs
    state["done"] = done

    reward_text = (
        "### Action Feedback\n"
        f"- Step reward: **{reward.step_reward:+.3f}**\n"
        f"- Time penalty: **{reward.time_penalty:+.3f}**\n"
        f"- Blast penalty: **{reward.blast_radius_penalty:+.3f}**\n"
        f"- Cumulative reward: **{reward.cumulative_reward:+.3f}**"
    )

    score_text = ""
    score_payload = {
        "score": state.get("score", 0),
        "done": done,
        "step": obs.step,
        "simulated_time_min": obs.simulated_time_min,
    }
    if done:
        ep = info.get("episode_result")
        if ep:
            state["score"] = ep.episode_score
            score_payload = {
                "score": ep.episode_score,
                "success": ep.success,
                "steps_taken": ep.steps_taken,
                "root_cause_correct": ep.root_cause_correct,
                "blast_radius_score": ep.blast_radius_score,
                "time_efficiency": ep.time_efficiency,
                "postmortem_score": ep.postmortem_score,
            }
            score_text = (
                f"\n\n## ✅ Episode complete!\n"
                f"**Score: {ep.episode_score:.4f}**\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| Success | {'Yes ✅' if ep.success else 'No ❌'} |\n"
                f"| Steps | {ep.steps_taken} |\n"
                f"| Root cause correct | {'Yes ✅' if ep.root_cause_correct else 'No ❌'} |\n"
                f"| Blast radius | {ep.blast_radius_score:.2f} |\n"
                f"| Time efficiency | {ep.time_efficiency:.2f} |\n"
                f"| Postmortem | {ep.postmortem_score:.2f} |\n"
            )

    obs_md  = fmt_obs(obs) + score_text
    info_md = reward_text
    return obs_md, info_md, score_payload, state


def build_ui():
    import gradio as gr

    with gr.Blocks(
        title="SRE-Bench",
        css="""
        .gradio-container {max-width: 1300px !important;}
        .block-title {font-weight: 700;}
        """,
    ) as demo:
        gr.Markdown(
            """
# 🛠 SRE-Bench — Production Incident Response
**An OpenEnv environment for evaluating AI SRE agents.**
Choose a task, investigate the incident, and restore service.

The right panel is formatted like an incident report so judges can quickly understand what happened.
            """
        )

        state = gr.State({})
        rl_state = gr.State({"policies": {"default": RLEpisodePolicy()}, "last_model": "default"})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙ Controls")
                task_dd = gr.Dropdown(
                    choices=ALL_TASK_IDS, value=ALL_TASK_IDS[0], label="Task"
                )
                start_btn = gr.Button("🚀 Start Episode", variant="primary")

                gr.Markdown("---")
                gr.Markdown("#### Action")

                action_dd = gr.Dropdown(
                    choices=ACTION_LABELS,
                    value="get_topology",
                    label="Action type",
                )
                service_tb = gr.Textbox(label="Service name", placeholder="e.g. payments-service")
                replicas_n = gr.Number(label="Replicas (scale_up only)", value=4, minimum=1, maximum=20)
                message_tb = gr.Textbox(label="Message (page_team)")
                rc_tb      = gr.Textbox(label="Root cause (mark_resolved / postmortem)")
                timeline_tb= gr.Textbox(label="Timeline (postmortem)", lines=2)
                prev_tb    = gr.Textbox(label="Prevention (postmortem)", lines=2)

                step_btn   = gr.Button("▶ Take Action", variant="secondary")

                with gr.Accordion("🧠 RL Controls (Hackathon Demo)", open=False):
                    rl_model_tb = gr.Textbox(label="Model name", value="hackathon_v1")
                    rl_dataset_tb = gr.Textbox(
                        label="Training dataset path",
                        value="demo/meta_hackathon_dataset.template.json",
                        placeholder="e.g. demo/meta_hackathon_dataset.template.json",
                    )
                    with gr.Row():
                        rl_gamma = gr.Slider(label="Gamma", minimum=0.5, maximum=0.999, step=0.001, value=0.95)
                        rl_alpha = gr.Slider(label="Alpha", minimum=0.05, maximum=1.0, step=0.05, value=0.35)
                    rl_train_btn = gr.Button("Train RL Model", variant="primary")
                    rl_train_md = gr.Markdown()
                    rl_train_json = gr.JSON(label="Training Metrics")

                    gr.Markdown("#### Suggest Next Action")
                    with gr.Row():
                        rl_suggest_task = gr.Dropdown(choices=ALL_TASK_IDS, value=ALL_TASK_IDS[0], label="Task")
                        rl_suggest_step = gr.Number(label="Step override (0 = auto from episode)", value=0)
                    rl_suggest_btn = gr.Button("Suggest Next Action")
                    rl_suggest_md = gr.Markdown()

                    gr.Markdown("#### Auto-Run Episode")
                    with gr.Row():
                        rl_auto_task = gr.Dropdown(choices=ALL_TASK_IDS, value=ALL_TASK_IDS[0], label="Task")
                        rl_auto_seed = gr.Number(label="Seed", value=42)
                        rl_auto_max_actions = gr.Slider(label="Max actions", minimum=3, maximum=20, step=1, value=8)
                    rl_autoplay_btn = gr.Button("Auto-Run Episode")
                    rl_autoplay_md = gr.Markdown()
                    rl_autoplay_json = gr.JSON(label="Auto-Run Trace")

                    gr.Markdown("#### Judge Demo Mode")
                    rl_judge_btn = gr.Button("Run Judge Demo Mode", variant="primary")
                    rl_judge_md = gr.Markdown()
                    rl_judge_json = gr.JSON(label="Judge Report Card")

            with gr.Column(scale=2):
                obs_md   = gr.Markdown("Press **Start Episode** to begin.")
                info_out = gr.Markdown()
                score_out= gr.JSON(label="Score")

        start_btn.click(
            fn=start_episode,
            inputs=[task_dd],
            outputs=[obs_md, info_out, state],
        )

        step_btn.click(
            fn=take_action,
            inputs=[
                action_dd, service_tb, replicas_n,
                message_tb, rc_tb, timeline_tb, prev_tb, state,
            ],
            outputs=[obs_md, info_out, score_out, state],
        )

        rl_train_btn.click(
            fn=rl_train_ui,
            inputs=[rl_model_tb, rl_dataset_tb, rl_gamma, rl_alpha, rl_state],
            outputs=[rl_train_md, rl_train_json, rl_state],
        )

        rl_suggest_btn.click(
            fn=rl_suggest_ui,
            inputs=[rl_model_tb, rl_suggest_task, rl_suggest_step, state, rl_state],
            outputs=[
                rl_suggest_md,
                action_dd,
                service_tb,
                replicas_n,
                message_tb,
                rc_tb,
                timeline_tb,
                prev_tb,
            ],
        )

        rl_autoplay_btn.click(
            fn=rl_autoplay_ui,
            inputs=[rl_model_tb, rl_auto_task, rl_auto_seed, rl_auto_max_actions, rl_state],
            outputs=[rl_autoplay_md, rl_autoplay_json],
        )

        rl_judge_btn.click(
            fn=judge_demo_mode_ui,
            inputs=[rl_model_tb, rl_dataset_tb, rl_gamma, rl_alpha, rl_auto_seed, rl_auto_max_actions, rl_state],
            outputs=[rl_judge_md, rl_judge_json, rl_state],
        )

        gr.Markdown(
            """
---
**Scoring:** Time efficiency × Blast radius × Root cause accuracy × Postmortem quality
| Task | Difficulty | Correct fix |
|---|---|---|
| task1_oom | Easy | `restart_service payments-service` |
| task2_bad_deploy | Medium | `rollback_deploy auth-service` |
| task3_phantom_slowdown | Hard | `scale_up data-pipeline-service` or `page_team` with correct RCA |
            """
        )

    return demo
