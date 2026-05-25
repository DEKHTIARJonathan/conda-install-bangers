import asyncio
import os
import re
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from bangers.config import settings
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
    TrtLlmServerUnavailable,
    get_loaded_chat_model_name,
    switch_chat_model,
)
from bangers.services.llm_provider import ChatRuntimeBusy

router = APIRouter(tags=["models"])

_SAFE_MODEL_NAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class ModelInfo(BaseModel):
    name: str
    model_type: str
    is_active: bool = False
    is_loading: bool = False
    compatibility: list[str] = Field(default_factory=list)
    format: str = ""
    quantization: str = ""


class ModelsResponse(BaseModel):
    dit_models: list[ModelInfo]
    lm_models: list[ModelInfo]
    chat_llm_models: list[ModelInfo] = []


def _scan_checkpoints(loaded_chat_llm: str = "") -> ModelsResponse:
    checkpoints_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints"
    chat_llm_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm"
    dit_models: list[ModelInfo] = []
    lm_models: list[ModelInfo] = []
    chat_llm_models: list[ModelInfo] = []
    loading_state = generation_service.loading_state or {}
    loading_kind = loading_state.get("kind")
    loading_model_name = loading_state.get("model_name")

    if checkpoints_dir.exists():
        active_dit = generation_service.active_dit_model
        active_lm = generation_service.active_lm_model

        for entry in sorted(checkpoints_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith(("acestep-v15-", "acestep-v1-")):
                dit_models.append(ModelInfo(
                    name=name,
                    model_type="dit",
                    is_active=(name == active_dit),
                    is_loading=(loading_kind == "dit" and name == loading_model_name),
                ))
            elif name.startswith("acestep-5Hz-lm-"):
                lm_models.append(ModelInfo(
                    name=name,
                    model_type="lm",
                    is_active=(name == active_lm),
                    is_loading=(loading_kind == "lm" and name == loading_model_name),
                ))

    if chat_llm_dir.exists():
        for entry in sorted(chat_llm_dir.iterdir()):
            if entry.is_dir() and (entry / "config.json").exists():
                name = entry.name
                if name not in CHAT_LLM_REGISTRY:
                    continue
                chat_llm_models.append(ModelInfo(
                    name=name,
                    model_type="chat_llm",
                    is_active=(name == loaded_chat_llm),
                    is_loading=(loading_kind == "chat_llm" and name == loading_model_name),
                    compatibility=list(CHAT_LLM_COMPATIBILITY.get(name, ())),
                    format=CHAT_LLM_FORMATS.get(name, ""),
                    quantization=CHAT_LLM_QUANTIZATIONS.get(name, ""),
                ))

    return ModelsResponse(
        dit_models=dit_models,
        lm_models=lm_models,
        chat_llm_models=chat_llm_models,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    return _scan_checkpoints(loaded_chat_llm=get_loaded_chat_model_name())


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
    if "trtllm" in runtimes:
        return sys.platform.startswith("linux")
    if "mlx" in runtimes:
        return sys.platform == "darwin"
    return True


def _chat_llm_runtime_requirement(model_name: str) -> str:
    runtimes = CHAT_LLM_COMPATIBILITY.get(model_name, ())
    if "trtllm" in runtimes:
        return f"{model_name} requires TensorRT-LLM on NVIDIA Linux."
    if "mlx" in runtimes:
        return f"{model_name} requires MLX on macOS."
    return f"{model_name} requires a supported chat runtime."


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
            detail=_chat_llm_runtime_requirement(model_name),
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
        runtime = request.runtime or settings.DEFAULT_LM_BACKEND
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("lm_model", request.model_name),
        )
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("lm_backend", runtime),
        )
        await db.commit()

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
            detail=_chat_llm_runtime_requirement(model_name),
        )

    _reject_if_loading(model_name)

    generation_service._set_loading("chat_llm", model_name)
    try:
        await switch_chat_model(model_name)
    except ChatRuntimeBusy as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chat_llm_busy",
                "message": str(exc),
            },
        ) from exc
    except TrtLlmServerUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "trtllm_server_unavailable",
                "message": str(exc),
                "command": exc.command,
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
    device = generation_service.device or "unknown"

    try:
        import torch

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return GpuStatsResponse(
                device=device,
                vram_used_mb=round(vram_used, 1),
                vram_total_mb=round(vram_total, 1),
                vram_percent=round(vram_used / vram_total * 100, 1) if vram_total > 0 else None,
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            allocated = 0.0
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
            elif hasattr(torch.mps, "driver_allocated_size"):
                allocated = torch.mps.driver_allocated_size() / (1024 * 1024)
            return GpuStatsResponse(
                device=device,
                vram_used_mb=round(allocated, 1),
            )
    except Exception as e:
        logger.warning(f"Failed to read GPU stats: {e}")

    return GpuStatsResponse(device=device)
