import asyncio
import os
import re
import sys
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from bangers.config import normalize_lm_backend, settings
from bangers.model_registry import (
    ACE_DIT_MODELS,
    ACE_LM_MODELS,
    ACE_MAIN_BUNDLE,
    ACE_MODEL_BY_NAME,
    CHAT_LLM_COMPATIBILITY,
    CHAT_LLM_DESCRIPTIONS,
    CHAT_LLM_FORMATS,
    CHAT_LLM_QUANTIZATIONS,
    CHAT_LLM_REGISTRY,
    CHAT_LLM_SIZES,
    main_bundle_visible_components,
)
from bangers.models.common import (
    AvailableModel,
    AvailableModelsResponse,
    DownloadModelRequest,
    GpuStatsResponse,
    SwitchModelRequest,
)
from bangers.services.generation import generation_service
from bangers.services.chat_llm import (
    ChatLlmUnavailable,
    get_loaded_chat_model_name,
    switch_chat_model,
)
from bangers.services.distributed import distributed_cluster
from bangers.services.gpu_lock import gpu_lock
from bangers.services.gpu_stats import merge_gpu_stats, read_local_gpu_stats
from bangers.services.llm_provider import ChatRuntimeBusy

router = APIRouter(tags=["models"])

_SAFE_MODEL_NAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class ModelInfo(BaseModel):
    name: str
    model_type: str
    is_active: bool = False
    is_loading: bool = False
    loaded_on: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    format: str = ""
    quantization: str = ""


class ModelsResponse(BaseModel):
    dit_models: list[ModelInfo]
    lm_models: list[ModelInfo]
    chat_llm_models: list[ModelInfo] = []


ActiveModelMap = dict[str, dict[str, set[str]]]


def _active_nodes(active_models: ActiveModelMap, model_type: str, name: str) -> list[str]:
    return sorted(active_models.get(model_type, {}).get(name, set()))


def _append_remote_active_models(
    rows: list[ModelInfo],
    *,
    active_models: ActiveModelMap,
    model_type: str,
) -> None:
    existing = {row.name for row in rows}
    for name, nodes in sorted(active_models.get(model_type, {}).items()):
        if not name or name in existing:
            continue
        rows.append(
            ModelInfo(
                name=name,
                model_type=model_type,
                is_active=True,
                loaded_on=sorted(nodes),
                compatibility=list(CHAT_LLM_COMPATIBILITY.get(name, ()))
                if model_type == "chat_llm"
                else [],
                format=CHAT_LLM_FORMATS.get(name, "") if model_type == "chat_llm" else "",
                quantization=CHAT_LLM_QUANTIZATIONS.get(name, "")
                if model_type == "chat_llm"
                else "",
            )
        )


async def _distributed_active_models() -> ActiveModelMap:
    active: ActiveModelMap = {"dit": {}, "lm": {}, "chat_llm": {}}
    if not settings.delegates_to_workers:
        return active
    workers = await distributed_cluster.refresh(force=True)
    for worker in workers:
        node = worker.node_id or worker.url
        if worker.ready_for("music") and worker.active_dit_model:
            active["dit"].setdefault(worker.active_dit_model, set()).add(node)
        if worker.ready_for("ace_lm") and worker.active_lm_model:
            active["lm"].setdefault(worker.active_lm_model, set()).add(node)
        if worker.ready_for("chat_llm") and worker.active_chat_llm_model:
            active["chat_llm"].setdefault(worker.active_chat_llm_model, set()).add(node)
    return active


def _scan_checkpoints(
    loaded_chat_llm: str = "",
    *,
    active_models: ActiveModelMap | None = None,
) -> ModelsResponse:
    active_models = active_models or {"dit": {}, "lm": {}, "chat_llm": {}}
    checkpoints_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints"
    chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"
    dit_models: list[ModelInfo] = []
    lm_models: list[ModelInfo] = []
    chat_llm_models: list[ModelInfo] = []
    loading_state = generation_service.loading_state or {}
    loading_kind = loading_state.get("kind")
    loading_model_name = loading_state.get("model_name")

    if checkpoints_dir.exists():
        active_dit = "" if settings.delegates_to_workers else generation_service.active_dit_model
        active_lm = "" if settings.delegates_to_workers else generation_service.active_lm_model

        for entry in sorted(checkpoints_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith(("acestep-v15-", "acestep-v1-")):
                loaded_on = _active_nodes(active_models, "dit", name)
                dit_models.append(ModelInfo(
                    name=name,
                    model_type="dit",
                    is_active=bool(loaded_on) or (name == active_dit),
                    is_loading=(loading_kind == "dit" and name == loading_model_name),
                    loaded_on=loaded_on,
                ))
            elif name.startswith("acestep-5Hz-lm-"):
                loaded_on = _active_nodes(active_models, "lm", name)
                lm_models.append(ModelInfo(
                    name=name,
                    model_type="lm",
                    is_active=bool(loaded_on) or (name == active_lm),
                    is_loading=(loading_kind == "lm" and name == loading_model_name),
                    loaded_on=loaded_on,
                ))

    if chat_llm_dir.exists():
        for entry in sorted(chat_llm_dir.iterdir()):
            if entry.is_dir() and (entry / "config.json").exists():
                name = entry.name
                loaded_on = _active_nodes(active_models, "chat_llm", name)
                chat_llm_models.append(ModelInfo(
                    name=name,
                    model_type="chat_llm",
                    is_active=bool(loaded_on) or (name == loaded_chat_llm),
                    is_loading=(loading_kind == "chat_llm" and name == loading_model_name),
                    loaded_on=loaded_on,
                    compatibility=list(CHAT_LLM_COMPATIBILITY.get(name, ())),
                    format=CHAT_LLM_FORMATS.get(name, ""),
                    quantization=CHAT_LLM_QUANTIZATIONS.get(name, ""),
                ))

    _append_remote_active_models(
        dit_models,
        active_models=active_models,
        model_type="dit",
    )
    _append_remote_active_models(
        lm_models,
        active_models=active_models,
        model_type="lm",
    )
    _append_remote_active_models(
        chat_llm_models,
        active_models=active_models,
        model_type="chat_llm",
    )

    return ModelsResponse(
        dit_models=dit_models,
        lm_models=lm_models,
        chat_llm_models=chat_llm_models,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    return _scan_checkpoints(
        loaded_chat_llm=get_loaded_chat_model_name(),
        active_models=await _distributed_active_models(),
    )


def _scan_dir_bytes(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _hf_cache_blob_dir(repo_id: str) -> Path:
    cache_root = settings.HF_HUB_CACHE_DIR
    folder_name = "models--" + repo_id.replace("/", "--")
    return Path(cache_root) / folder_name / "blobs"


def _download_progress_fs(repo_id: str, model_dir: Path, expected_mb: int) -> float:
    if expected_mb <= 0:
        return 0.0
    expected_bytes = expected_mb * 1_000_000
    total = _scan_dir_bytes(_hf_cache_blob_dir(repo_id))
    if model_dir.exists():
        total += _scan_dir_bytes(model_dir)
    return min(total / expected_bytes, 0.99)


def _chat_llm_runtime_supported(model_name: str) -> bool:
    runtimes = CHAT_LLM_COMPATIBILITY.get(model_name, ())
    if "mlx" in runtimes:
        return sys.platform == "darwin"
    return True


@router.get("/models/available", response_model=AvailableModelsResponse)
async def list_available_models() -> AvailableModelsResponse:
    """List curated ACE-Step DiT/LM and chat LLM models with install status."""
    checkpoints_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints"
    chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"
    result: list[AvailableModel] = []
    main_downloading = any(
        generation_service.model_download_status.get(name) == "downloading"
        for name in main_bundle_visible_components()
    )

    for model in ACE_DIT_MODELS + ACE_LM_MODELS:
        name = model.name
        installed = (checkpoints_dir / name).exists()
        if model.bundled:
            downloading = main_downloading or (
                generation_service.model_download_status.get(name) == "downloading"
            )
            progress = 0.0
            if downloading:
                progress = _download_progress_fs(
                    ACE_MAIN_BUNDLE.repo_id,
                    checkpoints_dir,
                    ACE_MAIN_BUNDLE.bundle_size_mb,
                )
            repo_id = ACE_MAIN_BUNDLE.repo_id
        else:
            downloading = (
                generation_service.model_download_status.get(name) == "downloading"
            )
            progress = 0.0
            if downloading:
                progress = _download_progress_fs(
                    model.repo_id, checkpoints_dir / name, model.size_mb,
                )
            repo_id = model.repo_id
        result.append(AvailableModel(
            name=name,
            model_type=model.model_type,
            repo_id=repo_id,
            installed=installed,
            description=model.description,
            downloading=downloading,
            download_progress=progress,
            size_mb=model.size_mb,
        ))

    for name, repo_id in CHAT_LLM_REGISTRY.items():
        installed = (chat_llm_dir / name).exists() and (chat_llm_dir / name / "config.json").exists()
        downloading = generation_service.model_download_status.get(name) == "downloading"
        progress = 0.0
        if downloading:
            progress = _download_progress_fs(
                repo_id, chat_llm_dir / name, CHAT_LLM_SIZES.get(name, 0),
            )
        result.append(AvailableModel(
            name=name,
            model_type="chat_llm",
            repo_id=repo_id,
            installed=installed,
            description=CHAT_LLM_DESCRIPTIONS.get(name, ""),
            downloading=downloading,
            download_progress=progress,
            size_mb=CHAT_LLM_SIZES.get(name, 0),
            compatibility=list(CHAT_LLM_COMPATIBILITY.get(name, ())),
            format=CHAT_LLM_FORMATS.get(name, ""),
            quantization=CHAT_LLM_QUANTIZATIONS.get(name, ""),
        ))

    return AvailableModelsResponse(models=result)


@router.post("/models/download")
async def download_model(request: DownloadModelRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger download of a curated ACE-Step or chat LLM model."""
    model_name = request.model_name
    if not _SAFE_MODEL_NAME.match(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    ace_model = ACE_MODEL_BY_NAME.get(model_name)
    is_ace_main = ace_model is not None and ace_model.bundled
    is_ace_solo = ace_model is not None and not ace_model.bundled
    is_chat_llm = model_name in CHAT_LLM_REGISTRY
    if is_chat_llm and not _chat_llm_runtime_supported(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"{model_name} requires MLX on macOS.",
        )
    if not (is_chat_llm or is_ace_main or is_ace_solo):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    status_names = main_bundle_visible_components() if is_ace_main else (model_name,)
    if any(generation_service.model_download_status.get(name) == "downloading" for name in status_names):
        return {"status": "already_downloading"}

    for status_name in status_names:
        generation_service.model_download_status[status_name] = "downloading"

    if is_ace_main:
        bundle_status_names = tuple(status_names)

        def _do_download() -> None:
            try:
                from acestep.model_downloader import download_main_model
                checkpoint_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints"
                success, msg = download_main_model(checkpoint_dir)
                status = "done" if success else f"error:{msg}"
                for status_name in bundle_status_names:
                    generation_service.model_download_status[status_name] = status
                logger.info(f"Main ACE-Step model download: {msg}")
            except Exception as e:
                for status_name in bundle_status_names:
                    generation_service.model_download_status[status_name] = f"error:{e}"
                logger.exception("Main ACE-Step model download failed")
    elif is_ace_solo:
        assert ace_model is not None
        ace_repo_id = ace_model.repo_id

        def _do_download() -> None:
            try:
                from huggingface_hub import snapshot_download
                local_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints" / model_name
                snapshot_download(repo_id=ace_repo_id, local_dir=str(local_dir))
                generation_service.model_download_status[model_name] = "done"
                logger.info(f"ACE submodel download complete: {model_name} ({ace_repo_id})")
            except Exception as e:
                generation_service.model_download_status[model_name] = f"error:{e}"
                logger.exception(f"ACE submodel download failed: {model_name}")
    else:
        def _do_download() -> None:
            try:
                from huggingface_hub import snapshot_download
                repo_id = CHAT_LLM_REGISTRY[model_name]
                local_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm" / model_name
                snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
                generation_service.model_download_status[model_name] = "done"
                logger.info(f"Chat LLM download complete: {model_name}")
            except Exception as e:
                generation_service.model_download_status[model_name] = f"error:{e}"
                logger.exception(f"Chat LLM download failed: {model_name}")

    async def _run_download() -> None:
        await asyncio.to_thread(_do_download)

    background_tasks.add_task(_run_download)
    return {"status": "started"}


@router.get("/models/download-status")
async def get_download_status() -> dict[str, str]:
    return generation_service.model_download_status


def _reject_if_loading(model_name: str = "") -> None:
    state = generation_service.loading_state
    if state is None:
        return
    if not model_name or state.get("model_name") == model_name:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "load_in_flight",
                "message": "This model is already loading.",
                "loading": state,
            },
        )
    raise HTTPException(
        status_code=409,
        detail={
            "error": "load_in_flight",
            "message": (
                f"Another model is loading ({state.get('kind')} / "
                f"{state.get('model_name')}). Wait for it to finish."
            ),
            "loading": state,
        },
    )


@router.post("/models/switch-dit")
async def switch_dit_model(request: SwitchModelRequest) -> dict[str, str]:
    from bangers.db.connection import get_db

    _reject_if_loading(request.model_name)
    generation_service._set_loading("dit", request.model_name)

    try:
        db = await get_db()
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("dit_model", request.model_name),
        )
        await db.commit()

        if settings.delegates_to_workers:
            try:
                nodes = await distributed_cluster.switch_dit_model(request.model_name)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"message": f"DiT model switched to {request.model_name} on {', '.join(nodes)}"}

        status, ok = await generation_service.initialize_dit(
            config_path=request.model_name,
            device=generation_service.device or settings.DEFAULT_DEVICE,
        )
    finally:
        generation_service._clear_loading()
    if not ok:
        raise HTTPException(status_code=500, detail=status)

    return {"message": f"DiT model switched to {request.model_name}"}


@router.post("/models/switch-lm")
async def switch_lm_model(request: SwitchModelRequest) -> dict[str, str]:
    from bangers.db.connection import get_db

    _reject_if_loading(request.model_name)
    generation_service._set_loading("lm", request.model_name)

    try:
        db = await get_db()
        runtime = normalize_lm_backend(request.runtime or settings.DEFAULT_LM_BACKEND)
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("lm_model", request.model_name),
        )
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("lm_backend", runtime),
        )
        await db.commit()

        if settings.delegates_to_workers:
            try:
                nodes = await distributed_cluster.switch_lm_model(request.model_name, runtime)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"message": f"LM model switched to {request.model_name} on {', '.join(nodes)}"}

        status, ok = await generation_service.initialize_lm(
            lm_model_path=request.model_name,
            backend=runtime,
            device=generation_service.device or settings.DEFAULT_DEVICE,
        )
    finally:
        generation_service._clear_loading()
    if not ok:
        raise HTTPException(status_code=500, detail=status)

    return {"message": f"LM model switched to {request.model_name}"}


@router.post("/models/switch-chat-llm")
async def switch_chat_llm_model(request: SwitchModelRequest) -> dict[str, str]:
    model_name = request.model_name
    if not _SAFE_MODEL_NAME.match(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")
    if model_name not in CHAT_LLM_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown Chat LLM: {model_name}")
    if not _chat_llm_runtime_supported(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"{model_name} requires MLX on macOS.",
        )

    _reject_if_loading(model_name)
    generation_service._set_loading("chat_llm", model_name)
    try:
        if settings.delegates_to_workers:
            from bangers.db.connection import get_db

            db = await get_db()
            await db.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("dj_model", model_name),
            )
            await db.commit()
            try:
                nodes = await distributed_cluster.switch_chat_llm_model(model_name)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"message": f"Chat LLM loaded on {', '.join(nodes)}"}

        await switch_chat_model(model_name)
    except HTTPException:
        raise
    except ChatRuntimeBusy as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chat_llm_busy",
                "message": str(exc),
            },
        ) from exc
    except ChatLlmUnavailable as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "chat_llm_unavailable",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception(f"Chat LLM switch failed: {model_name}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "chat_llm_load_failed",
                "message": str(exc),
            },
        ) from exc
    finally:
        generation_service._clear_loading()

    return {"message": f"Chat LLM loaded: {model_name}"}


@router.get("/models/gpu-stats", response_model=GpuStatsResponse)
async def get_gpu_stats() -> GpuStatsResponse:
    if settings.delegates_to_workers:
        worker_stats = await distributed_cluster.collect_gpu_stats()
        if worker_stats:
            return merge_gpu_stats(worker_stats, device="distributed")

    device = generation_service.device or "unknown"
    try:
        return await read_local_gpu_stats(
            node_id=settings.DISTRIBUTED_NODE_ID,
            node_role=settings.DISTRIBUTED_ROLE,
            device=device,
            busy=gpu_lock.is_locked,
            holder=gpu_lock.holder,
        )
    except Exception as exc:
        logger.warning(f"Failed to read GPU stats: {exc}")
        return GpuStatsResponse(
            device=device,
            updated_at=time.time(),
            error=str(exc),
        )
