# SRE-Bench Manual

SRE-Bench is an OpenEnv-compatible incident response environment for evaluating SRE agents. The app simulates production-style outages with logs, metrics, service topology, and postmortems.

## What You Can Do

- Open the Gradio UI and step through incidents manually.
- Use the OpenEnv API endpoints for scripted checks.
- Run the bundled demo dataset to verify the app end-to-end.
- Use the Python environment directly from code or notebooks.

## Core Concepts

The simulated system has 8 services across frontend, backend, data, and infra tiers. Each task starts with noisy alerts and partial observability, so you must investigate before remediating.

Tasks:

| ID                       | Scenario                                            | Correct Fix                                            |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------ |
| `task1_oom`              | payments-service crash loops from memory exhaustion | `restart_service payments-service`                     |
| `task2_bad_deploy`       | auth-service bad deploy causes cascading 503s       | `rollback_deploy auth-service`                         |
| `task3_phantom_slowdown` | periodic latency spikes from a cron-driven DB lock  | `scale_up data-pipeline-service` or correct paging/RCA |

## Setup

Install dependencies:

```bash
cd "/Users/punithns/Downloads/files (1)"
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run The App

Start the app locally:

```bash
cd "/Users/punithns/Downloads/files (1)"
.venv/bin/python app.py
```

Open:

- UI: `http://127.0.0.1:7860/`
- Health: `http://127.0.0.1:7860/health`

If port 7860 is busy, start it on another port:

```bash
PORT=7861 .venv/bin/python app.py
```

## How To Use The UI

1. Select a task from the dropdown.
2. Click Start Episode.
3. Use `get_topology`, `get_metrics`, and `read_logs` to investigate.
4. Apply the correct remediation action.
5. Finish with `mark_resolved` or `write_postmortem`.

Recommended actions by task:

- Task 1: inspect `payments-service`, then restart it.
- Task 2: inspect `auth-service`, then roll back the deploy.
- Task 3: correlate topology, metrics, and logs, then mitigate the pipeline bottleneck.

## API Endpoints

The app also exposes a small JSON API for automation:

- `POST /reset` starts a new episode.
- `POST /step` executes one action.
- `POST /state` returns the current environment state.
- `GET /health` checks whether the app is alive.

RL endpoints:

- `POST /rl/train` trains a tabular policy from user/demo scenarios.
- `POST /rl/suggest` returns the best next action for `(task_id, step)`.
- `POST /rl/autoplay` runs one episode using the trained policy.

Example reset request:

```bash
curl -s -X POST http://127.0.0.1:7861/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_oom","seed":42}'
```

Example RL training using bundled dataset:

```bash
curl -s -X POST http://127.0.0.1:7861/rl/train \
  -H "Content-Type: application/json" \
  -d '{"model_name":"hackathon_v1","dataset_path":"demo/demo_dataset.json"}'
```

Example RL next-action suggestion:

```bash
curl -s -X POST http://127.0.0.1:7861/rl/suggest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"hackathon_v1","task_id":"task1_oom","step":0}'
```

Example RL autoplay run:

```bash
curl -s -X POST http://127.0.0.1:7861/rl/autoplay \
  -H "Content-Type: application/json" \
  -d '{"model_name":"hackathon_v1","task_id":"task2_bad_deploy","seed":42,"max_actions":8}'
```

## Demo Dataset

Use the bundled demo dataset to verify the full loop:

```bash
.venv/bin/python demo/check_demo_dataset.py \
  --base-url http://127.0.0.1:7861 \
  --dataset demo/demo_dataset.json
```

The checker resets episodes, runs the sample actions, and prints pass/fail summaries.

Use your own user data by following the same schema as `demo/demo_dataset.json`:

- top-level key: `scenarios`
- each scenario: `id`, `task_id`, `seed`, `steps`
- each step: `action_type`, `params`

Quick start template: `demo/meta_hackathon_dataset.template.json`

Then point RL training to your file:

```bash
curl -s -X POST http://127.0.0.1:7861/rl/train \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta_hackathon","dataset_path":"demo/your_dataset.json"}'
```

## Direct Python Usage

```python
from sre_bench import SREBenchEnv, Action, ActionType

env = SREBenchEnv(task_id="task1_oom")
obs = env.reset()

obs, reward, done, info = env.step(Action(ActionType.GET_TOPOLOGY))
obs, reward, done, info = env.step(Action(
    action_type=ActionType.READ_LOGS,
    params={"service": "payments-service", "window_minutes": 10},
))
```

## Validate The Package

Run the OpenEnv validator:

```bash
.venv-openenv/bin/openenv validate .
```

Expected result: validation passes for multi-mode deployment.

## Docker

```bash
docker build -t sre-bench .
docker run -p 7860:7860 sre-bench
```

## Hugging Face Spaces Deployment

This repository already includes an entrypoint for Spaces in `hf_space/app.py`.

1. Create a new Hugging Face Space:
  - SDK: `gradio`
  - Hardware: `CPU Basic` (enough for this app)
2. Upload these folders/files:
  - `sre_bench/`
  - `hf_space/app.py`
  - `hf_space/requirements.txt`
  - `demo/demo_dataset.json` (optional but recommended)
3. In Space Settings, set app file to `app.py` (the file from `hf_space/`).
4. Deploy and test:
  - `GET /health`
  - `POST /rl/train`
  - open the UI root page.

For hackathon demos, pre-train a policy at startup by calling `/rl/train` once with your curated user dataset.

Automated Space deployment script:

```bash
export HF_TOKEN=<your_huggingface_write_token>
./scripts/deploy_hf_space.sh <hf_username> <space_name>
```

## Repository Layout

```text
app.py                 Root launcher
sre_bench/             Core package and API
server/app.py          OpenEnv entrypoint
hf_space/              Hugging Face Space entrypoint
demo/                  Demo dataset and checker
openenv.yaml           Environment spec
pyproject.toml         Packaging metadata
requirements.txt       Runtime dependencies
```

## License

MIT.
