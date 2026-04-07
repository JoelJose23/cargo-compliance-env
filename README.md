---
title: Cargo-DeAI-Dispatcher
sdk: docker
license: apache-2.0
---

# Cargo Compliance Environment

This repository contains a benchmark-style environment and agent runner for cargo compliance checks across bilateral import/export rules.

## What Judges Should Know

The project is organized around a 3-phase loop:

1. Extraction: Parse shipment text into `qty`, `category`, `Destination`, `Origin`.
2. Selection: Choose laws, regulators, and documents from environment-provided options.
3. Verdict: Submit short legal reasoning scored by an LLM judge.

The environment applies dense rewards for partial correctness and penalties for unnecessary questions or hallucinated selections.

## Runtime Components

- Agent entrypoint: `i.py`
- Environment API: `server/environment.py` + FastAPI routes
- Dataset source: `data/final_dataset.json`
- Benchmark wrapper config: `openenv.yaml`

## Quick Run

Install dependencies and run local server:

```bash
pip install -r requirements.txt
python -m server.main
```

Run the agent:

```bash
python i.py
```

## Configuration

Primary environment variables used by `i.py`:

- `HF_TOKEN` or `API_KEY` for model access
- `API_BASE_URL` (default Hugging Face Inference API)
- `MODEL_NAME` (default `Qwen/Qwen2.5-72B-Instruct`)
- `WORLD_ENV_URL` (environment service URL)
- `MY_ENV_V4_TASK`
- `MY_ENV_V4_BENCHMARK`

## Scoring Notes

- `i.py` normalizes score by `MAX_TOTAL_REWARD` (default `5.5`).
- Success threshold defaults to `0.44`.
- Environment phase rewards are implemented in `server/environment.py`.

## File-Level Documentation

For a full file-by-file guide used by judges:

- `FILE_INDEX.md` (entire repository map)
- `server/README.md` (server internals and API contract)
