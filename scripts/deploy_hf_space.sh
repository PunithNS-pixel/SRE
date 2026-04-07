#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf_username> <space_name>"
  exit 1
fi

HF_USERNAME="$1"
SPACE_NAME="$2"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export a Hugging Face write token first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="/tmp/hf_space_${HF_USERNAME}_${SPACE_NAME}"
SPACE_URL="https://oauth2:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

rm -rf "$WORKDIR"
git clone "$SPACE_URL" "$WORKDIR"

rsync -a --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/sre_bench" "$WORKDIR/"

rsync -a "$PROJECT_ROOT/demo" "$WORKDIR/"
cp "$PROJECT_ROOT/hf_space/app.py" "$WORKDIR/app.py"
cp "$PROJECT_ROOT/hf_space/requirements.txt" "$WORKDIR/requirements.txt"
cp "$PROJECT_ROOT/hf_space/openenv.yaml" "$WORKDIR/openenv.yaml"
cp "$PROJECT_ROOT/README.md" "$WORKDIR/README.md"

cd "$WORKDIR"

git add .
if git diff --cached --quiet; then
  echo "No changes to deploy."
  exit 0
fi

git commit -m "Deploy SRE-Bench RL app to Hugging Face Space"
git push origin main

echo "Deploy complete: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
