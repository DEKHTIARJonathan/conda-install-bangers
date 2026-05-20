import asyncio
import argparse
import os
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from bangers.config import (
    DISTRIBUTED_CAPABILITY_ACE_LM,
    DISTRIBUTED_CAPABILITY_CHAT_LLM,
    DISTRIBUTED_CAPABILITY_MUSIC,
    normalize_lm_backend,
    settings,
)
from bangers.db.connection import init_db, close_db
from bangers.services.generation import generation_service
from bangers.routers.health import router as health_router, health_ws_router, health_broadcast_loop
from bangers.routers.generation import router as generation_router, format_router, ws_router
from bangers.routers.songs import router as songs_router
from bangers.routers.models import router as models_router
from bangers.routers.uploads import router as uploads_router
from bangers.routers.history import router as history_router
from bangers.routers.radio import router as radio_router
from bangers.routers.dj import router as dj_router
from bangers.routers.internal_workers import router as internal_worker_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("conda install bangers starting up...")
    settings.ensure_dirs()
    await init_db()
    logger.info("Database initialized")

    # Mark any stale running/pending history entries as failed (e.g. from a crash or restart)
    from bangers.db.connection import get_db
    db = await get_db()
    await db.execute(
        "UPDATE generation_history SET status = 'failed', error_message = 'Server restarted' "
        "WHERE status IN ('running', 'pending')"
    )
    await db.commit()

    # Load saved settings for model init
    cursor = await db.execute("SELECT key, value FROM settings")
    rows = await cursor.fetchall()
    saved = {row["key"]: row["value"] for row in rows}
    startup_overrides = settings.startup_setting_overrides()
    if startup_overrides:
        saved.update(startup_overrides)
        for key, value in startup_overrides.items():
            await db.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) "
                "VALUES (?, ?, datetime('now'))",
                (key, value),
            )
        await db.commit()

    # Seed radio presets
    from bangers.db.schema import RADIO_PRESETS
    from bangers.services.duration_settings import coerce_duration
    radio_default_duration = coerce_duration(saved.get("default_duration"))
    for preset in RADIO_PRESETS:
        cursor = await db.execute(
            "SELECT id FROM radio_stations WHERE name = ? AND is_preset = 1",
            (preset["name"],),
        )
        if await cursor.fetchone() is None:
            preset_id = str(_uuid.uuid4())
            now_str = datetime.now(timezone.utc).isoformat()
            await db.execute(
                """INSERT INTO radio_stations (
                    id, name, description, is_preset, caption_template,
                    genre, mood, instrumental, bpm_min, bpm_max,
                    duration_min, duration_max, created_at, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    preset_id,
                    preset["name"],
                    preset["description"],
                    preset.get("caption_template", ""),
                    preset.get("genre", ""),
                    preset.get("mood", ""),
                    1 if preset.get("instrumental", True) else 0,
                    preset.get("bpm_min"),
                    preset.get("bpm_max"),
                    preset.get("duration_min", radio_default_duration),
                    preset.get("duration_max", radio_default_duration),
                    now_str,
                    now_str,
                ),
            )
    await db.commit()
    logger.info("Radio presets seeded")

    # Restore GPU throttle settings from DB
    from bangers.ace_handler import set_vae_throttle, set_dit_throttle, set_throttle_radio_only
    set_vae_throttle(
        int(saved.get("vae_chunk_size", "256")),
        int(saved.get("vae_sleep_ms", "100")),
    )
    set_dit_throttle(int(saved.get("dit_sleep_ms", "100")))
    set_throttle_radio_only(saved.get("throttle_radio_only", "false") == "true")
    logger.info("GPU throttle settings restored from database")

    logger.info("Music generation engine: ACE-Step")

    def _local_capability_enabled(capability: str) -> bool:
        return (
            settings.DISTRIBUTED_ROLE == "standalone"
            or settings.has_distributed_capability(capability)
        )

    # Initialize ACE-Step models if the user has selected them and this node
    # is configured to host the corresponding inference capability.
    async def _init_active():
        load_music = _local_capability_enabled(DISTRIBUTED_CAPABILITY_MUSIC)
        load_ace_lm = _local_capability_enabled(DISTRIBUTED_CAPABILITY_ACE_LM)
        if not load_music and not load_ace_lm:
            logger.info("ACE-Step: no local music or ACE LM capability configured; using remote workers.")
            return

        dit_model = saved.get("dit_model", "")
        lm_model = saved.get("lm_model", "")
        requested_lm_backend = saved.get("lm_backend", settings.DEFAULT_LM_BACKEND)
        lm_backend = normalize_lm_backend(requested_lm_backend)
        if lm_backend != requested_lm_backend:
            logger.info(
                f"Using ACE LM backend '{lm_backend}' for configured runtime '{requested_lm_backend}'"
            )

        # Clamp LM model to GPU tier's supported list, downloading if needed
        if load_ace_lm and lm_model and not settings.is_lm_disabled(lm_model) and sys.platform != "darwin":
            from acestep.gpu_config import get_gpu_config, is_lm_model_size_allowed
            gpu_config = get_gpu_config()
            if gpu_config.available_lm_models and not is_lm_model_size_allowed(lm_model, gpu_config.available_lm_models):
                recommended = gpu_config.recommended_lm_model
                logger.warning(
                    f"LM model '{lm_model}' too large for {gpu_config.tier} "
                    f"({gpu_config.gpu_memory_gb:.1f} GB), switching to '{recommended}'"
                )
                lm_model = recommended
                # Download the recommended model if it's not on disk.
                # Registry-driven; works even for curated entries that
                # are not in upstream's SUBMODEL_REGISTRY.
                from bangers.services.ace_downloads import ensure_ace_model
                from pathlib import Path as _Path
                dl_ok, dl_msg = await asyncio.to_thread(
                    ensure_ace_model,
                    model_name=lm_model,
                    checkpoints_dir=_Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints",
                )
                if dl_ok:
                    logger.info(f"LM model ready: {dl_msg}")
                else:
                    logger.warning(f"LM model download failed: {dl_msg}")

        device = saved.get("device", settings.DEFAULT_DEVICE)

        dit_ok = False
        if load_music:
            if not dit_model:
                logger.info("ACE-Step: no DiT model selected, skipping DiT initialization. Pick one on the Models page.")
            else:
                logger.info(f"Initializing DiT model: {dit_model}")
                generation_service._set_loading("dit", dit_model)
                try:
                    status, ok = await generation_service.initialize_dit(
                        config_path=dit_model,
                        device=device,
                    )
                finally:
                    generation_service._clear_loading()
                if ok:
                    dit_ok = True
                    logger.info("DiT model loaded successfully")
                else:
                    logger.warning(f"DiT model failed to load: {status}")
        else:
            logger.info("ACE-Step: local music capability disabled; skipping DiT initialization")

        if load_ace_lm:
            if not lm_model:
                logger.info("ACE-Step: no LM model selected, skipping LM initialization")
            elif settings.DISTRIBUTED_ROLE == "standalone" and load_music and not dit_ok:
                logger.info("ACE-Step: skipping LM initialization because DiT did not load")
            else:
                logger.info(f"Initializing LM model: {lm_model}")
                generation_service._set_loading("lm", lm_model)
                try:
                    status, ok = await generation_service.initialize_lm(
                        lm_model_path=lm_model,
                        backend=lm_backend,
                        device=device,
                    )
                finally:
                    generation_service._clear_loading()
                if ok:
                    logger.info("LM model loaded successfully")
                else:
                    logger.warning(f"LM model failed to load: {status}")
        else:
            logger.info("ACE-Step: local ACE LM capability disabled; skipping LM initialization")

    async def _init_active_chat():
        if not _local_capability_enabled(DISTRIBUTED_CAPABILITY_CHAT_LLM):
            logger.info("Chat LLM: local chat capability disabled; skipping initialization")
            return
        chat_model = saved.get("dj_model", "")
        if not chat_model:
            logger.info("Chat LLM: no model selected, skipping initialization")
            return
        try:
            from bangers.services.chat_llm import warm_chat_model

            logger.info(f"Initializing Chat LLM: {chat_model}")
            generation_service._set_loading("chat_llm", chat_model)
            try:
                await warm_chat_model(chat_model)
            finally:
                generation_service._clear_loading()
            logger.info("Chat LLM loaded successfully")
        except Exception as exc:
            logger.warning(f"Chat LLM failed to load: {exc}")

    async def _init_models():
        await _init_active()
        await _init_active_chat()

    asyncio.create_task(_init_models())

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)
            generation_service.cleanup_old_jobs()

    cleanup_task = asyncio.create_task(_cleanup_loop())
    health_broadcast_task = asyncio.create_task(health_broadcast_loop())

    yield

    cleanup_task.cancel()
    health_broadcast_task.cancel()
    logger.info("conda install bangers shutting down...")
    await close_db()


app = FastAPI(
    title="conda install bangers",
    version="0.1.0",
    description="Premium Local Music Generation API",
    lifespan=lifespan,
)

# CORS: this is a single-user, no-auth local/LAN app served on whatever host
# happens to be running it (localhost, DGX LAN IP, SSH tunnel, reverse proxy).
# Origin-pinning provides no security here and only causes deployment friction,
# so we accept every origin. `allow_origin_regex` is used instead of `["*"]`
# because the CORS spec forbids combining `*` with credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Mount routers under /api
app.include_router(health_router, prefix="/api")
app.include_router(health_ws_router, prefix="/api")
app.include_router(generation_router, prefix="/api")
app.include_router(format_router, prefix="/api")
app.include_router(ws_router, prefix="/api")
app.include_router(songs_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(uploads_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(radio_router, prefix="/api")
app.include_router(dj_router, prefix="/api")
app.include_router(internal_worker_router, prefix="/api")

# Serve audio files
audio_dir = settings.AUDIO_DIR
if audio_dir.exists():
    app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")

# Serve uploads directory
uploads_dir = settings.UPLOADS_DIR
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


def run(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run conda install bangers backend")
    parser.parse_args(argv)

    import os
    import uvicorn

    # Optional TLS — `start.py` generates a self-signed cert covering localhost
    # and the detected LAN IP, then points us at it via env so the frontend
    # (which Next.js is serving over HTTPS) doesn't get mixed-content blocked
    # when fetch()ing the API. Empty/unset values mean plain HTTP.
    ssl_certfile = os.environ.get("BANGERS_SSL_CERTFILE") or None
    ssl_keyfile = os.environ.get("BANGERS_SSL_KEYFILE") or None

    uvicorn.run(
        "bangers.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    run()
