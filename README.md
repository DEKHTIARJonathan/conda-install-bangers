<div align="center">

<h1>conda install bangers</h1>

**Local-first AI music generation studio**

![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![ACE-Step](https://img.shields.io/badge/ACE--Step-1.5-purple?style=flat-square)

Local AI Music Studio to Generate and remix music entirely on your own machine.
Built on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).

![screenshot](assets/screenshot.png)

</div>

Use it to generate, remix, save, and play AI music on your own machine. The app includes text-to-music, custom generation, remix mode, AI DJ chat, radio stations, a library, and a full audio player.

## Quick Start

### Prerequisites

- Git
- [mise](https://mise.jdx.dev/) installed and activated in your shell

The repo pins Python 3.11, Node.js 20, pnpm 9.15.9, and conda in `.mise.toml`.

### Run Locally

```bash
git clone https://github.com/DEKHTIARJonathan/conda-install-bangers.git
cd conda-install-bangers
mise install
mise run setup
mise run dev
```

The launcher starts:

- backend: `https://localhost:8000`
- frontend: `https://localhost:3000`
- runtime data: `backend/data/`
- model cache: `.cache/models/`

The dev server uses a self-signed HTTPS certificate. Your browser will warn the first time; proceed to the local site. Press `Ctrl+C` to stop both servers.

On first launch, no model is loaded. Open the **Models** page, download/select a DiT model, and optionally select an ACE language model and a chat LLM. Selections are stored in `backend/data/conda-install-bangers.db` and restored on restart.

## Daily Commands

```bash
mise run dev        # Start backend and frontend only
mise run test       # Run backend and frontend tests once
mise run clean      # Reset local DB/audio/uploads, keep downloaded models
```

TensorRT-LLM companion server flow on NVIDIA Linux hosts:

```bash
mise run trtllm:qwen3-8b-nvfp4   # Terminal 1: serve the FP4 chat model on :8010
mise run dev:with-trt            # Terminal 2: start the app pointed at that server
```

Only one TRT-LLM chat model is served on port `8010` at a time. To switch TRT models, stop the current `trtllm:*` task and start the matching task before switching in the Models page.

Launcher flags:

```bash
python start.py --install   # Force dependency reinstall
python start.py --no-open   # Do not auto-open the browser
```

After pulling updates:

```bash
git pull
mise run setup
mise run dev
```

## Models

All downloads and active selections happen in the **Models** page. Model selection is intentionally not controlled by environment variables.

Current ACE model registry:

| Type | Models |
|------|--------|
| DiT | `acestep-v15-turbo`, `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo-continuous`, `acestep-v15-xl-turbo` |
| ACE language model | `acestep-5Hz-lm-1.7B`, `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-4B`, or no LM |
| Chat LLM, MLX | `Qwen3-0.6B-4bit`, `Qwen3-1.7B-4bit`, `Qwen3-4B-4bit`, `Qwen3-8B-4bit` |
| Chat LLM, Transformers | `Qwen3-1.7B`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B-FP8`, `Qwen3-14B-FP8`, `Qwen3-30B-A3B-Instruct-2507-FP8` |
| Chat LLM, external TensorRT-LLM | `Qwen3-8B-NVFP4`, `Qwen3-14B-NVFP4`, `Qwen3-30B-FP4` |

Disk usage depends on what you download. The default ACE bundle is about 10 GB, the XL DiT is about 20 GB, and larger chat LLMs add more.

For TensorRT-LLM FP4 chat, the backend is an external client. Start the matching `mise run trtllm:*` task first, then use `mise run dev:with-trt` and switch to that same model in the Models page. The backend checks `/health` and sends a tiny warmup chat completion before saving `dj_model`; it does not launch or stop Docker in the normal app path. By default, keep either one embedded Transformers FP8 chat model or one external FP4 TRT-LLM model active, not both.

Rough ACE LM guidance:

| VRAM | Suggested LM |
|------|--------------|
| <=6 GB | none |
| 6-8 GB | `acestep-5Hz-lm-0.6B` |
| 8-16 GB | `acestep-5Hz-lm-1.7B` |
| 16-24 GB | `acestep-5Hz-lm-1.7B` or `acestep-5Hz-lm-4B` |
| >=24 GB | `acestep-5Hz-lm-4B` |

## Configuration

Common environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `BANGERS_HOST` | `0.0.0.0` | Backend bind address |
| `BANGERS_PORT` | `8000` | Backend port |
| `BANGERS_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `BANGERS_LM_BACKEND` | `mlx` on macOS, `nano-vllm` elsewhere | ACE LM backend |
| `BANGERS_AUDIO_FORMAT` | `flac` | Default output format |
| `BANGERS_BATCH_SIZE` | `2` | Default samples per generation |
| `BANGERS_INFERENCE_STEPS` | `8` | Default DiT steps |
| `BANGERS_GUIDANCE_SCALE` | `7.0` | Default guidance scale |
| `BANGERS_THINKING` | `true` | Default 5 Hz LM thinking mode |
| `BANGERS_DATA_DIR` | `backend/data` | SQLite DB, audio, uploads |
| `BANGERS_MODEL_CACHE_DIR` | `.cache/models` | Model/cache root |
| `ACESTEP_PROJECT_ROOT` | `.cache/models` | ACE checkpoints and chat LLM root |
| `BANGERS_TRTLLM_MANAGED` | `false` | Keep TRT-LLM Docker lifecycle external to the app |
| `BANGERS_TRTLLM_SERVER_URL` | `http://127.0.0.1:8010` | External TRT-LLM OpenAI-compatible server URL |
| `BANGERS_TRTLLM_TIMEOUT_SECONDS` | `120` | Timeout for TRT-LLM health and chat requests |

Most generation defaults can also be changed in the app under **Settings**.

## Production

The current Docker Compose stack targets a single NVIDIA Linux host and serves HTTP:

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Open `http://localhost:3000`.

Compose uses two named volumes:

- `bangers-data`: SQLite DB, generated audio, uploads
- `bangers-models`: model weights and Hugging Face cache

See [DEPLOY.md](DEPLOY.md) for GPU prerequisites, upgrades, backups, and failure checks.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local setup, TLS behavior, cache paths, and test commands.

Run tests:

```bash
mise run test
```

Or run each side directly:

```bash
(cd backend && conda run --prefix .conda pytest -v)
pnpm --dir frontend exec vitest --run
```

## Credits and License

Built on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). See [ATTRIBUTION.md](ATTRIBUTION.md) for community inspirations.

Licensed under the [MIT License](LICENSE).
