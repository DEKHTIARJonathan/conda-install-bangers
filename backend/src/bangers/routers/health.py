import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from bangers.config import settings
from bangers.models.common import HealthResponse, SettingsResponse, SettingsUpdate
from bangers.services.generation import generation_service
from bangers.db.connection import get_db, get_instance_id
from bangers.ws.manager import health_ws_manager

router = APIRouter(tags=["health"])


def build_health_snapshot() -> HealthResponse:
    """Single source of truth for /api/health and /api/ws/health.

    Reads live state from generation_service + the cached instance_id; does
    no I/O so it's cheap to call from a polling loop.
    """
    svc = generation_service
    instance_id = get_instance_id()
    if settings.delegates_to_workers and not svc.dit_initialized:
        return HealthResponse(
            status="ok",
            dit_model_loaded=True,
            lm_model_loaded=True,
            dit_model=svc.active_dit_model or "distributed-workers",
            lm_model=svc.active_lm_model or "distributed-workers",
            device="distributed",
            init_stage=svc.init_stage,
            init_error=svc.init_error,
            download_progress=svc.download_progress,
            instance_id=instance_id,
        )
    lm_ready = svc.lm_initialized or svc.lm_disabled
    status = "ok" if svc.dit_initialized else "degraded"
    loading_state = svc.loading_state or {}
    init_stage = (
        "loading_chat_llm"
        if loading_state.get("kind") == "chat_llm"
        else svc.init_stage
    )
    return HealthResponse(
        status=status,
        dit_model_loaded=svc.dit_initialized,
        lm_model_loaded=lm_ready,
        dit_model=svc.active_dit_model,
        lm_model=svc.active_lm_model,
        device=svc.device,
        init_stage=init_stage,
        init_error=svc.init_error,
        download_progress=svc.download_progress,
        instance_id=instance_id,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return build_health_snapshot()


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    db = await get_db()
    cursor = await db.execute("SELECT key, value FROM settings")
    rows = await cursor.fetchall()
    return SettingsResponse(settings={row["key"]: row["value"] for row in rows})


@router.patch("/settings", response_model=SettingsResponse)
async def update_settings(body: SettingsUpdate) -> SettingsResponse:
    db = await get_db()
    for key, value in body.settings.items():
        await db.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value),
        )
    await db.commit()
    return await get_settings()


# ----- /api/ws/health -------------------------------------------------------
#
# Replaces the prior `useQuery({ refetchInterval })` polling that had five
# separate observers each scheduling its own /api/health roundtrip. The
# server holds a single snapshot, broadcasts on change to all subscribers,
# and each new subscriber gets the snapshot on connect.

health_ws_router = APIRouter(tags=["websocket"])


@health_ws_router.websocket("/ws/health")
async def websocket_health(websocket: WebSocket) -> None:
    await health_ws_manager.connect(websocket)
    try:
        # Send the current snapshot immediately so the new subscriber doesn't
        # have to wait for the next state change.
        snapshot = build_health_snapshot()
        await health_ws_manager.send_to(websocket, snapshot.model_dump())
        # Keep the socket open; the broadcaster pushes from the background.
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        await health_ws_manager.disconnect(websocket)


async def health_broadcast_loop(poll_interval_seconds: float = 1.0) -> None:
    """Background task that broadcasts /health snapshots on state change.

    Polls the in-process service state at `poll_interval_seconds` (cheap;
    no DB or HTTP) and only emits when the snapshot actually differs from
    the last broadcast. Cancellable from the FastAPI lifespan shutdown.
    """
    last_payload: dict | None = None
    while True:
        try:
            snapshot = build_health_snapshot()
            payload = snapshot.model_dump()
            if payload != last_payload:
                await health_ws_manager.broadcast(payload)
                last_payload = payload
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"health broadcast loop error: {exc}")
        await asyncio.sleep(poll_interval_seconds)
