# Development

Local development uses the host-native launcher. Docker is for production packaging and smoke tests.

## Toolchain

Install and activate [mise](https://mise.jdx.dev/), then let the repo install the pinned tools:

- Python 3.11
- Node.js 20
- pnpm 9.15.9
- conda via mise's experimental conda backend

`.mise.toml` must keep the conda tool as `"conda:conda" = "latest"`. Do not change it to bare `conda = "latest"`; some mise versions resolve that through the normal registry and fail.

The backend runs in `backend/.conda`, created from `backend/environment.yml`; Python packages are installed from `backend/pyproject.toml`.

## Setup

```bash
mise install
mise run setup
```

`mise run setup` creates:

- `backend/.conda/`
- `backend/data/audio/`
- `backend/data/uploads/`
- `.cache/models/checkpoints/`
- `.cache/models/chat-llm/`
- `.cache/models/huggingface/hub/`

Downloaded models live in `.cache/models/`, so dependency reinstalls and `mise run clean` do not delete them.

Reset runtime state:

```bash
mise run clean
```

This deletes `BANGERS_DATA_DIR` (`backend/data/` by default), then recreates `audio/` and `uploads/`.

## Run

```bash
mise run dev
```

This is app-only: it starts the backend and frontend, but no external TRT-LLM server.

The launcher starts:

- backend: `https://localhost:8000`
- frontend: `https://localhost:3000`
- LAN frontend, when a LAN IP is detected: `https://<ip>:3000`

It also writes combined runtime output to `runtime.log`.

For external TensorRT-LLM FP4 chat, use two terminals:

```bash
mise run trtllm:qwen3-8b-nvfp4
mise run dev:with-trt
```

The TRT-LLM tasks run `sudo docker run` with `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1`, mount the matching `.cache/models/chat-llm/<model>` directory read-only, and expose an OpenAI-compatible server on `http://127.0.0.1:8010`. Available tasks are `trtllm:qwen3-8b-nvfp4`, `trtllm:qwen3-14b-nvfp4`, and `trtllm:qwen3-30b-fp4`. Stop the current TRT server before starting another one because all tasks use port `8010`.

You can run the launcher directly after setup:

```bash
python start.py
python start.py --no-open
python start.py --install
```

## Dev TLS

The launcher generates one self-signed cert/key pair for both uvicorn and Next.js:

- `.cache/tls/dev.pem`
- `.cache/tls/dev-key.pem`

The cert covers `localhost`, `127.0.0.1`, and the detected LAN IP. Browsers will warn on first visit because the cert is self-signed. Click through for the local origin.

HTTPS is used so browser APIs required by radio playback, including `AudioContext.audioWorklet`, work on localhost and LAN URLs.

## Models

The backend starts with no active models. Open **Models** in the running app and download/select:

- one DiT model
- optionally one ACE language model
- optionally one chat LLM for AI DJ, generated titles, and lyric helpers

Selections are persisted in `backend/data/conda-install-bangers.db` and restored on restart. There are no environment variables for preselecting models.

Switching to a TensorRT-LLM chat model requires the matching external server to already be running. The backend verifies local model files, checks `/health`, sends a warmup request to `/v1/chat/completions`, and only then saves `dj_model`. A failed switch returns the exact `mise run trtllm:<model>` command to start. Keep only one chat model resident by default: either an embedded FP8 Transformers model or an external FP4 TRT-LLM server.

## Cache Paths

Default local paths:

```bash
BANGERS_DATA_DIR=./backend/data
BANGERS_MODEL_CACHE_DIR=./.cache/models
ACESTEP_PROJECT_ROOT=./.cache/models
HF_HOME=./.cache/models/huggingface
HF_HUB_CACHE=./.cache/models/huggingface/hub
BANGERS_TRTLLM_MANAGED=false
BANGERS_TRTLLM_SERVER_URL=http://127.0.0.1:8010
BANGERS_TRTLLM_TIMEOUT_SECONDS=120
```

Use another disk by exporting paths before setup/dev:

```bash
export BANGERS_MODEL_CACHE_DIR=/Volumes/AI/models/conda-install-bangers
export ACESTEP_PROJECT_ROOT="$BANGERS_MODEL_CACHE_DIR"
export HF_HOME="$BANGERS_MODEL_CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mise run setup
mise run dev
```

If models redownload unexpectedly, make sure all four model/cache variables point at the same storage layout.

## Tests

```bash
mise run test
```

This runs:

- backend: `cd backend && conda run --prefix .conda pytest -v`
- frontend: `pnpm --dir frontend exec vitest --run`

Run one side manually:

```bash
(cd backend && conda run --prefix .conda pytest -v)
pnpm --dir frontend exec vitest --run
```

## Frontend Only

For frontend-only work, you can run `pnpm dev` inside `frontend/`, but that does not start the backend or configure the shared dev HTTPS cert. Prefer `mise run dev` unless you are deliberately isolating the frontend.
