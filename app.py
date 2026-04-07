"""SRE-Bench — Hugging Face Space entrypoint."""

from __future__ import annotations

import os

import uvicorn

from sre_bench.webapp import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
    )
