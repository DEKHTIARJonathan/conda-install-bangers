import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from bangers.config import (
    DISTRIBUTED_CAPABILITY_ACE_LM,
    DISTRIBUTED_CAPABILITY_CHAT_LLM,
    DISTRIBUTED_CAPABILITY_MUSIC,
    normalize_lm_backend,
    settings,
)
from bangers.db.connection import get_db
from bangers.services.generation import generation_service
from bangers.services.chat_llm import (
    ChatLlmUnavailable,
    get_loaded_chat_model_name,
    switch_chat_model,
)
from bangers.services.llm_provider import ChatRuntimeBusy
from bangers.services.gpu_stats import read_local_gpu_stats
from bangers.services.gpu_lock import gpu_lock
from bangers.models.common import GpuStatsResponse, SwitchModelRequest


router = APIRouter(prefix="/internal/worker", tags=["internal-worker"])


class WorkerGenerateRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    lyrics_prepared: bool = False


class WorkerAceLmRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class WorkerChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0


class WorkerGenerationCancelled(BaseException):
    """Cooperative cancellation for internal worker jobs."""


def _check_worker_token(token: str | None) -> None:
    expected = settings.DISTRIBUTED_TOKEN
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Invalid worker token")


def _sanitized_audios(result: dict[str, Any]) -> list[dict[str, Any]]:
    audios: list[dict[str, Any]] = []
    for audio in result.get("audios", []) or []:
        audios.append({
            "path": audio.get("path", ""),
            "key": audio.get("key", ""),
            "sample_rate": audio.get("sample_rate", 48000),
            "params": {
                k: v
                for k, v in (audio.get("params") or {}).items()
                if k != "tensor" and not str(k).startswith("_")
            },
        })
    return audios


def _content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".opus": "audio/opus",
        ".aac": "audio/aac",
    }.get(ext, "application/octet-stream")


def _task_skips_lm(params: dict[str, Any]) -> bool:
    return params.get("task_type", "text2music") in {"cover", "repaint"}


def _user_metadata(params: Any) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if params.bpm is not None:
        metadata["bpm"] = int(params.bpm)
    if params.keyscale:
        metadata["keyscale"] = params.keyscale
    if params.timesignature:
        metadata["timesignature"] = params.timesignature
    if params.duration is not None and float(params.duration) > 0:
        metadata["duration"] = int(float(params.duration))
    return metadata or None


def _update_params_from_metadata(
    params_dict: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(params_dict)
    if updated.get("bpm") is None and metadata.get("bpm") not in (None, "", "N/A"):
        try:
            updated["bpm"] = int(float(metadata["bpm"]))
        except (TypeError, ValueError):
            pass
    if not updated.get("keyscale") and metadata.get("keyscale") not in (None, "", "N/A"):
        updated["keyscale"] = metadata["keyscale"]
    if not updated.get("timesignature") and metadata.get("timesignature") not in (None, "", "N/A"):
        updated["timesignature"] = metadata["timesignature"]
    if (updated.get("duration") is None or float(updated.get("duration") or 0) <= 0) and metadata.get("duration") not in (None, "", "N/A"):
        try:
            updated["duration"] = float(metadata["duration"])
        except (TypeError, ValueError):
            pass
    if updated.get("use_cot_caption") and metadata.get("caption"):
        updated["caption"] = metadata["caption"]
    if updated.get("use_cot_language") and metadata.get("vocal_language"):
        updated["vocal_language"] = metadata["vocal_language"]
    elif updated.get("use_cot_language") and metadata.get("language"):
        updated["vocal_language"] = metadata["language"]
    if not updated.get("lyrics") and metadata.get("lyrics"):
        updated["lyrics"] = metadata["lyrics"]
    return updated


def _seed_list(seed: int, batch_size: int) -> list[int] | None:
    if seed == -1:
        return None
    return [int(seed)] + [-1] * max(0, batch_size - 1)


@router.get("/status")
async def worker_status(
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_worker_token(x_bangers_worker_token)
    svc = generation_service
    capabilities = settings.DISTRIBUTED_CAPABILITIES
    music_ready = (
        DISTRIBUTED_CAPABILITY_MUSIC in capabilities
        and svc.dit_initialized
    )
    ace_lm_ready = (
        DISTRIBUTED_CAPABILITY_ACE_LM in capabilities
        and svc.lm_initialized
    )
    chat_llm_model = get_loaded_chat_model_name()
    chat_ready = (
        DISTRIBUTED_CAPABILITY_CHAT_LLM in capabilities
    )
    ready = bool(capabilities)
    if DISTRIBUTED_CAPABILITY_MUSIC in capabilities:
        ready = ready and music_ready
    if DISTRIBUTED_CAPABILITY_ACE_LM in capabilities:
        ready = ready and ace_lm_ready
    if DISTRIBUTED_CAPABILITY_CHAT_LLM in capabilities:
        ready = ready and chat_ready
    return {
        "node_id": settings.DISTRIBUTED_NODE_ID,
        "role": settings.DISTRIBUTED_ROLE,
        "capabilities": sorted(capabilities),
        "busy": gpu_lock.is_locked,
        "holder": gpu_lock.holder,
        "ready": ready,
        "music_ready": music_ready,
        "ace_lm_ready": ace_lm_ready,
        "chat_ready": chat_ready,
        "dit_model": svc.active_dit_model,
        "lm_model": svc.active_lm_model,
        "chat_llm_model": chat_llm_model,
        "device": svc.device,
        "init_stage": svc.init_stage,
        "init_error": svc.init_error,
    }


@router.get("/gpu-stats", response_model=GpuStatsResponse)
async def worker_gpu_stats(
    x_bangers_worker_token: str | None = Header(default=None),
) -> GpuStatsResponse:
    _check_worker_token(x_bangers_worker_token)
    return await read_local_gpu_stats(
        node_id=settings.DISTRIBUTED_NODE_ID,
        node_role=settings.DISTRIBUTED_ROLE,
        device=generation_service.device or "unknown",
        busy=gpu_lock.is_locked,
        holder=gpu_lock.holder,
    )


async def _persist_setting(key: str, value: str) -> None:
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) "
        "VALUES (?, ?, datetime('now'))",
        (key, value),
    )
    await db.commit()


@router.post("/models/switch-dit")
async def switch_worker_dit_model(
    request: SwitchModelRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, str]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_MUSIC not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise music capability")
    await _persist_setting("dit_model", request.model_name)
    generation_service._set_loading("dit", request.model_name)
    try:
        status, ok = await generation_service.initialize_dit(
            config_path=request.model_name,
            device=generation_service.device or settings.DEFAULT_DEVICE,
        )
    finally:
        generation_service._clear_loading()
    if not ok:
        raise HTTPException(status_code=500, detail=status)
    return {"message": f"DiT model loaded on {settings.DISTRIBUTED_NODE_ID}"}


@router.post("/models/switch-lm")
async def switch_worker_lm_model(
    request: SwitchModelRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, str]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_ACE_LM not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise ACE LM capability")
    runtime = normalize_lm_backend(request.runtime or settings.DEFAULT_LM_BACKEND)
    await _persist_setting("lm_model", request.model_name)
    await _persist_setting("lm_backend", runtime)
    generation_service._set_loading("lm", request.model_name)
    try:
        status, ok = await generation_service.initialize_lm(
            lm_model_path=request.model_name,
            backend=runtime,
            device=generation_service.device or settings.DEFAULT_DEVICE,
        )
    finally:
        generation_service._clear_loading()
    if not ok:
        raise HTTPException(status_code=500, detail=status)
    return {"message": f"LM model loaded on {settings.DISTRIBUTED_NODE_ID}"}


@router.post("/models/switch-chat-llm")
async def switch_worker_chat_llm_model(
    request: SwitchModelRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, str]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_CHAT_LLM not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise chat LLM capability")
    try:
        await switch_chat_model(request.model_name)
    except ChatRuntimeBusy as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "chat_llm_busy", "message": str(exc)},
        ) from exc
    except ChatLlmUnavailable as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "chat_llm_unavailable", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception(f"Worker Chat LLM switch failed: {request.model_name}")
        raise HTTPException(
            status_code=500,
            detail={"error": "chat_llm_load_failed", "message": str(exc)},
        ) from exc
    return {"message": f"Chat LLM loaded on {settings.DISTRIBUTED_NODE_ID}"}


@router.post("/jobs")
async def create_worker_job(
    request: WorkerGenerateRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, str]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_MUSIC not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise music capability")
    if not generation_service.backend_ready:
        raise HTTPException(status_code=503, detail="Music backend is not loaded")
    job_id = generation_service.create_job()
    asyncio.create_task(_run_worker_generation(job_id, request))
    return {"job_id": job_id, "status": "queued"}


async def _run_worker_generation(job_id: str, request: WorkerGenerateRequest) -> None:
    timings: dict[str, float] = {}
    generation_service.update_job(job_id, status="queued", progress=0.0, stage="queued")
    if generation_service.is_cancelled(job_id):
        generation_service.update_job(
            job_id,
            status="cancelled",
            error="Cancelled by coordinator",
            timings=timings,
        )
        return

    await gpu_lock.await_acquire("worker")
    generation_service.update_job(job_id, status="running", progress=0.01, stage="starting")

    def progress_callback(value: float, desc: str = "") -> None:
        if generation_service.is_cancelled(job_id):
            raise WorkerGenerationCancelled()
        generation_service.update_job(job_id, progress=value, stage=desc)

    try:
        if generation_service.is_cancelled(job_id):
            raise WorkerGenerationCancelled()

        params = dict(request.params)
        if generation_service.active_dit_model and not params.get("dit_model"):
            params["dit_model"] = generation_service.active_dit_model
        if generation_service.active_lm_model and not params.get("lm_model"):
            params["lm_model"] = generation_service.active_lm_model
        result = await generation_service.generate(
            params,
            progress_callback=progress_callback,
            lyrics_prepared=request.lyrics_prepared,
        )
        if result.get("success"):
            generation_service.update_job(
                job_id,
                status="completed",
                progress=1.0,
                stage="completed",
                results=_sanitized_audios(result),
                timings=timings,
            )
        else:
            generation_service.update_job(
                job_id,
                status="failed",
                error=result.get("error", "Remote worker generation failed"),
                timings=timings,
            )
    except WorkerGenerationCancelled:
        generation_service.update_job(
            job_id,
            status="cancelled",
            error="Cancelled by coordinator",
            timings=timings,
        )
    except BaseException as exc:
        logger.exception(f"Internal worker generation {job_id} failed")
        generation_service.update_job(
            job_id,
            status="failed",
            error=str(exc),
            timings=timings,
        )
    finally:
        await gpu_lock.release("worker")


@router.get("/jobs/{job_id}")
async def get_worker_job(
    job_id: str,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_worker_token(x_bangers_worker_token)
    job = generation_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Worker job not found")
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0.0),
        "stage": job.get("stage", ""),
        "results": job.get("results", []),
        "error": job.get("error"),
        "timings": job.get("timings", {}),
    }


@router.delete("/jobs/{job_id}")
async def cancel_worker_job(
    job_id: str,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, bool]:
    _check_worker_token(x_bangers_worker_token)
    return {"cancelled": generation_service.cancel_job(job_id)}


@router.get("/jobs/{job_id}/artifacts/{index}")
async def get_worker_artifact(
    job_id: str,
    index: int,
    x_bangers_worker_token: str | None = Header(default=None),
) -> FileResponse:
    _check_worker_token(x_bangers_worker_token)
    job = generation_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Worker job not found")
    results = job.get("results", [])
    if index < 0 or index >= len(results):
        raise HTTPException(status_code=404, detail="Artifact not found")
    audio_path = Path(str(results[index].get("path") or ""))
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(str(audio_path), media_type=_content_type(audio_path), filename=audio_path.name)


@router.post("/ace-lm/prepare")
async def prepare_with_ace_lm(
    request: WorkerAceLmRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_ACE_LM not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise ACE LM capability")
    if _task_skips_lm(request.params):
        return {"success": True, "params": request.params, "error": None}
    acquired = await gpu_lock.await_acquire("ace-lm")
    try:
        result = await asyncio.to_thread(_prepare_with_ace_lm_sync, dict(request.params))
        return result
    finally:
        if acquired:
            await gpu_lock.release("ace-lm")


def _prepare_with_ace_lm_sync(params_dict: dict[str, Any]) -> dict[str, Any]:
    from acestep.inference import GenerationConfig, GenerationParams

    backend = generation_service.backend
    llm_handler = getattr(backend, "llm_handler", None)
    if llm_handler is None or not getattr(llm_handler, "llm_initialized", False):
        return {"success": False, "params": params_dict, "error": "ACE 5Hz LM is not initialized"}

    params = GenerationParams(**{
        k: v for k, v in params_dict.items()
        if k in GenerationParams.__dataclass_fields__
    })
    batch_size = int(params_dict.get("batch_size", settings.DEFAULT_BATCH_SIZE))
    config = GenerationConfig(
        batch_size=batch_size,
        audio_format=params_dict.get("audio_format", settings.DEFAULT_AUDIO_FORMAT),
        use_random_seed=params.seed == -1,
        seeds=_seed_list(params.seed, batch_size),
    )
    need_audio_codes = not bool(str(params.audio_codes or "").strip())
    need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas
    if not ((params.thinking or need_lm_for_cot) and need_audio_codes):
        return {"success": True, "params": params_dict, "error": None}

    result = llm_handler.generate_with_stop_condition(
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        infer_type="llm_dit" if params.thinking else "dit",
        temperature=params.lm_temperature,
        cfg_scale=params.lm_cfg_scale,
        negative_prompt=params.lm_negative_prompt,
        top_k=None if not params.lm_top_k else int(params.lm_top_k),
        top_p=None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p,
        target_duration=params.duration,
        user_metadata=_user_metadata(params),
        use_cot_caption=params.use_cot_caption,
        use_cot_language=params.use_cot_language,
        use_cot_metas=params.use_cot_metas,
        use_constrained_decoding=params.use_constrained_decoding,
        batch_size=batch_size,
        seeds=config.seeds,
    )
    if not result.get("success", False):
        return {"success": False, "params": params_dict, "error": result.get("error", "ACE LM failed")}

    updated = dict(params_dict)
    metadata = result.get("metadata") or {}
    if isinstance(metadata, list):
        metadata_for_params = metadata[0] if metadata else {}
    else:
        metadata_for_params = metadata
    if isinstance(metadata_for_params, dict):
        updated = _update_params_from_metadata(updated, metadata_for_params)

    audio_codes = result.get("audio_codes")
    if audio_codes:
        updated["audio_codes"] = audio_codes

    updated["_distributed_ace_lm_prepared"] = True
    updated["thinking"] = False
    updated["use_cot_metas"] = False
    updated["use_cot_caption"] = False
    updated["use_cot_language"] = False
    return {"success": True, "params": updated, "error": None}


@router.post("/chat")
async def run_worker_chat(
    request: WorkerChatRequest,
    x_bangers_worker_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_worker_token(x_bangers_worker_token)
    if DISTRIBUTED_CAPABILITY_CHAT_LLM not in settings.DISTRIBUTED_CAPABILITIES:
        raise HTTPException(status_code=409, detail="This worker does not advertise chat LLM capability")
    try:
        from bangers.services.llm_provider import get_chat_runtime

        runtime = get_chat_runtime(request.model)
        if runtime is None or not runtime.is_model_loadable(request.model):
            return {
                "success": False,
                "text": "",
                "error": f"Chat model '{request.model}' is not loadable on this worker",
            }
        text = await runtime.chat(
            request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"success": True, "text": text, "error": None}
    except Exception as exc:
        logger.exception("Internal worker chat failed")
        return {"success": False, "text": "", "error": str(exc)}
