import asyncio
import os
import shutil
import threading
from typing import Any, Callable, Optional

from loguru import logger

from bangers.config import normalize_lm_backend, settings


class _DownloadProgressTracker:
    """Intercepts huggingface_hub's tqdm to track actual bytes downloaded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_bytes: int = 0
        self.downloaded_bytes: int = 0

    @property
    def progress(self) -> float:
        with self._lock:
            if self.total_bytes <= 0:
                return 0.0
            return min(self.downloaded_bytes / self.total_bytes, 0.99)

    def make_tqdm_class(self):
        tracker = self
        from tqdm import tqdm as _base

        class _TrackedTqdm(_base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.total and self.total > 1_000_000:
                    with tracker._lock:
                        tracker.total_bytes += self.total

            def update(self, n=1):
                if self.total and self.total > 1_000_000:
                    with tracker._lock:
                        tracker.downloaded_bytes += n
                return super().update(n)

        return _TrackedTqdm


ACE_STEP_SUPPORTED_TASK_TYPES = ("text2music", "music2music", "cover")
ACE_STEP_SUPPORTED_AUDIO_FORMATS = ("flac", "mp3", "wav", "wav32", "opus", "aac")

_FS_EXPECTED_BYTES: int = 15_000_000_000
_VLLM_ENFORCE_EAGER_ENV = "BANGERS_VLLM_ENFORCE_EAGER"


def _bool_env(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _should_enforce_vllm_eager(device: str) -> bool:
    configured = os.getenv(_VLLM_ENFORCE_EAGER_ENV, "auto")
    parsed = _bool_env(configured)
    if parsed is not None:
        return parsed
    if configured.strip().lower() not in {"", "auto"}:
        logger.warning(f"Ignoring invalid {_VLLM_ENFORCE_EAGER_ENV}={configured!r}; using auto")

    if device not in {"auto", "cuda"}:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        current_device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current_device).upper()
        capability = torch.cuda.get_device_capability(current_device)
        return "GB10" in name or capability >= (12, 1)
    except Exception as exc:
        logger.debug(f"Could not determine whether vLLM eager mode is needed: {exc}")
        return False


def _install_vllm_eager_wrapper(handler: Any, device: str) -> None:
    initializer = getattr(handler, "_initialize_5hz_lm_vllm", None)
    if initializer is None:
        return
    setattr(handler, "_bangers_vllm_device", device)
    if getattr(handler, "_bangers_vllm_eager_wrapped", False):
        return

    def _wrapped_initialize_vllm(model_path: str, enforce_eager: bool = False) -> str:
        wrapper_device = getattr(handler, "_bangers_vllm_device", "auto")
        if _should_enforce_vllm_eager(wrapper_device) and not enforce_eager:
            logger.info(
                "Forcing vLLM eager mode to avoid CUDA graph capture failures "
                f"(set {_VLLM_ENFORCE_EAGER_ENV}=false to opt out)"
            )
            enforce_eager = True
        return initializer(model_path, enforce_eager=enforce_eager)

    setattr(handler, "_initialize_5hz_lm_vllm", _wrapped_initialize_vllm)
    setattr(handler, "_bangers_vllm_eager_wrapped", True)


class AceStepBackend:
    """ACE-Step music generation backend."""

    def __init__(self) -> None:
        self.dit_handler = None
        self.llm_handler = None
        self.dit_initialized = False
        self.lm_initialized = False
        self.lm_disabled = False
        self.active_dit_model = ""
        self.active_lm_model = ""
        self._device = ""
        self._init_stage: str = "idle"
        self._init_error: str = ""
        self._download_progress: float = 0.0
        self._download_tracker: Optional[_DownloadProgressTracker] = None

    @property
    def is_ready(self) -> bool:
        return self.dit_initialized

    @property
    def device(self) -> str:
        return self._device

    @property
    def init_stage(self) -> str:
        return self._init_stage

    @property
    def init_error(self) -> str:
        return self._init_error

    @property
    def download_progress(self) -> float:
        return self._download_progress

    @staticmethod
    def _scan_dir_bytes(path: str) -> int:
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

    async def _monitor_download_progress(self) -> None:
        checkpoint_dir = os.path.join(settings.ACESTEP_PROJECT_ROOT, "checkpoints")
        baseline = self._scan_dir_bytes(checkpoint_dir)

        while self._init_stage == "downloading":
            tracker = self._download_tracker
            if tracker and tracker.total_bytes > 0:
                self._download_progress = tracker.progress
            else:
                delta = self._scan_dir_bytes(checkpoint_dir) - baseline
                self._download_progress = min(delta / _FS_EXPECTED_BYTES, 0.99)
            await asyncio.sleep(2)
        self._download_progress = 0.0

    async def unload(self) -> None:
        def _sync_unload():
            import gc
            import torch
            self.dit_handler = None
            self.llm_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        await asyncio.to_thread(_sync_unload)
        self.dit_initialized = False
        self.lm_initialized = False
        self.lm_disabled = False
        self.active_dit_model = ""
        self.active_lm_model = ""
        self._device = ""
        self._init_stage = "idle"
        self._init_error = ""
        logger.info("ACE-Step backend unloaded")

    async def initialize(self, **kwargs: Any) -> tuple[str, bool]:
        """Initialize both DiT and LM models.

        Caller MUST supply ``config_path``; ``lm_model_path`` is optional and
        defaults to '' (i.e. no LM loaded). There are no environment-variable
        fallbacks - model selection is purely user-driven via the Models page.
        """
        config_path = kwargs.get("config_path", "")
        if not config_path:
            return "No DiT model selected", False
        device = kwargs.get("device", settings.DEFAULT_DEVICE)

        status, ok = await self.initialize_dit(config_path=config_path, device=device)
        if not ok:
            return status, False

        lm_model = kwargs.get("lm_model_path", "")
        if not lm_model:
            # No LM selected - leave the LM uninitialized but report success
            # so the caller's overall init flow proceeds.
            return "DiT loaded; no LM selected", True
        lm_backend = kwargs.get("lm_backend", settings.DEFAULT_LM_BACKEND)
        return await self.initialize_lm(
            lm_model_path=lm_model, backend=lm_backend, device=device,
        )

    async def initialize_dit(
        self,
        config_path: str,
        device: str = "auto",
    ) -> tuple[str, bool]:
        from bangers.services.runtime_settings import keep_active_models_resident
        keep_resident = await keep_active_models_resident()
        try:
            project_root = settings.ACESTEP_PROJECT_ROOT
            checkpoint_dir = os.path.join(project_root, "checkpoints")

            try:
                from pathlib import Path
                from acestep.model_downloader import check_main_model_exists
                if not check_main_model_exists(Path(checkpoint_dir)):
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    disk = shutil.disk_usage(checkpoint_dir)
                    required_bytes = 10_000_000_000
                    if disk.free < required_bytes:
                        free_gb = disk.free / (1024 ** 3)
                        self._init_stage = "error"
                        self._init_error = f"Not enough disk space ({free_gb:.1f} GB free, need ~10 GB)"
                        return self._init_error, False
                    self._init_stage = "downloading"
                else:
                    self._init_stage = "loading_dit"
            except Exception:
                self._init_stage = "loading_dit"

            if self._init_stage == "downloading":
                self._download_tracker = _DownloadProgressTracker()
                asyncio.create_task(self._monitor_download_progress())

            tracker = self._download_tracker

            # Detect GPU tier for quantization / offload defaults
            from acestep.gpu_config import get_gpu_config, is_mps_platform

            gpu_config = get_gpu_config()

            quantization = None
            offload_dit_to_cpu = False
            offload_to_cpu = False
            compile_model = False

            if not is_mps_platform():
                # On CUDA, treat 20+ GB as "enough VRAM to keep models
                # resident" *unless* the user explicitly turned residency off.
                # When residency is off we honor the original heuristic so
                # smaller GPUs and conservative users keep their offload
                # safety nets.
                enough_vram_for_resident_cuda = (
                    gpu_config.gpu_memory_gb >= 20.0 and keep_resident
                )
                if gpu_config.quantization_default and not enough_vram_for_resident_cuda:
                    quantization = "int8_weight_only"
                    compile_model = True
                if not enough_vram_for_resident_cuda:
                    offload_dit_to_cpu = gpu_config.offload_dit_to_cpu_default
                    offload_to_cpu = gpu_config.offload_to_cpu_default

            logger.info(
                f"GPU config: {gpu_config.tier} ({gpu_config.gpu_memory_gb:.1f} GB), "
                f"keep_resident={'yes' if keep_resident else 'no'}, "
                f"quantization={'int8' if quantization else 'off'}, "
                f"offload_dit={'yes' if offload_dit_to_cpu else 'no'}, "
                f"offload_to_cpu={'yes' if offload_to_cpu else 'no'}"
            )

            def _sync_init():
                from bangers.ace_handler import BangersHandler

                _orig_tqdm = None
                if tracker is not None:
                    try:
                        import huggingface_hub.file_download as _hf_dl
                        _orig_tqdm = _hf_dl.tqdm
                        _hf_dl.tqdm = tracker.make_tqdm_class()
                    except (ImportError, AttributeError):
                        pass

                try:
                    if self.dit_handler is None:
                        self.dit_handler = BangersHandler(project_root=project_root)
                    return self.dit_handler.initialize_service(
                        project_root=project_root,
                        config_path=config_path,
                        device=device,
                        quantization=quantization,
                        offload_dit_to_cpu=offload_dit_to_cpu,
                        offload_to_cpu=offload_to_cpu,
                        compile_model=compile_model,
                    )
                finally:
                    if _orig_tqdm is not None:
                        _hf_dl.tqdm = _orig_tqdm

            status, success = await asyncio.to_thread(_sync_init)

            if success:
                self.dit_initialized = True
                self.active_dit_model = config_path
                self._device = self.dit_handler.device
                self._init_stage = "loading_lm"
                logger.info(f"DiT initialized: {config_path} on {self._device}")
            else:
                self._init_stage = "error"
                self._init_error = status
                logger.error(f"DiT init failed: {status}")

            return status, success

        except Exception as e:
            self._init_stage = "error"
            self._init_error = str(e)
            logger.exception("Failed to initialize DiT")
            return f"Error: {e}", False

    async def initialize_lm(
        self,
        lm_model_path: str,
        backend: str = "mlx",
        device: str = "auto",
    ) -> tuple[str, bool]:
        try:
            if settings.is_lm_disabled(lm_model_path):
                self.llm_handler = None
                self.lm_initialized = False
                self.lm_disabled = True
                self.active_lm_model = "none"
                self._init_stage = "ready"
                logger.info("LM initialization skipped: DiT-only profile")
                return "LM disabled", True

            checkpoint_dir = os.path.join(settings.ACESTEP_PROJECT_ROOT, "checkpoints")
            ace_backend = normalize_lm_backend(backend)
            if ace_backend != backend:
                logger.info(f"Mapping LM backend '{backend}' to ACE-Step backend '{ace_backend}'")

            def _sync_init():
                from acestep.llm_inference import LLMHandler
                if self.llm_handler is None:
                    self.llm_handler = LLMHandler()
                if ace_backend == "vllm":
                    _install_vllm_eager_wrapper(self.llm_handler, device)
                return self.llm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=lm_model_path,
                    backend=ace_backend,
                    device=device,
                )

            status, success = await asyncio.to_thread(_sync_init)

            if success:
                self.lm_initialized = True
                self.lm_disabled = False
                self.active_lm_model = lm_model_path
                self._init_stage = "ready"
                logger.info(f"LM initialized: {lm_model_path}")
            else:
                self._init_stage = "error"
                self._init_error = status
                logger.error(f"LM init failed: {status}")

            return status, success

        except Exception as e:
            self._init_stage = "error"
            self._init_error = str(e)
            logger.exception("Failed to initialize LM")
            return f"Error: {e}", False

    async def generate(
        self,
        params_dict: dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        TASK_TYPE_MAP = {"music2music": "cover"}
        params_dict["task_type"] = TASK_TYPE_MAP.get(
            params_dict.get("task_type", "text2music"),
            params_dict["task_type"],
        )

        if params_dict.get("src_audio_path"):
            params_dict["src_audio"] = params_dict.pop("src_audio_path")

        params = GenerationParams(**{
            k: v for k, v in params_dict.items()
            if k in GenerationParams.__dataclass_fields__
        })

        batch_size = params_dict.get("batch_size", settings.DEFAULT_BATCH_SIZE)
        audio_format = params_dict.get("audio_format", settings.DEFAULT_AUDIO_FORMAT)

        config = GenerationConfig(
            batch_size=batch_size,
            audio_format=audio_format,
            use_random_seed=params.seed == -1,
            seeds=[params.seed] if params.seed != -1 else None,
        )

        settings.ensure_dirs()
        save_dir = str(settings.AUDIO_DIR)

        thinking = params_dict.get("thinking", False)
        self._log_lm_info(
            "generate_music",
            task_type=params.task_type,
            thinking=thinking,
            temp=getattr(params, "lm_temperature", None),
        )
        logger.info(
            "DiT params [generate_music]: steps={} cfg={} shift={} method={} seed={} batch={} fmt={}",
            getattr(params, "inference_steps", "?"),
            getattr(params, "guidance_scale", "?"),
            getattr(params, "shift", "?"),
            getattr(params, "infer_method", "ode"),
            getattr(params, "seed", "?"),
            batch_size,
            audio_format,
        )

        def _run_generation():
            return generate_music(
                self.dit_handler,
                self.llm_handler,
                params,
                config,
                save_dir=save_dir,
                progress=progress_callback,
            )

        result = await asyncio.to_thread(_run_generation)
        return result.to_dict()

    def _log_lm_info(self, task: str, **extra: Any) -> None:
        """Log which 5Hz LM backend/model is being used for a task."""
        h = self.llm_handler
        backend = getattr(h, "llm_backend", "unknown") if h else "unknown"
        model = self.active_lm_model or "unknown"
        parts = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        logger.info(f"5Hz LM [{task}]: backend={backend} model={model} {parts}".rstrip())

    async def create_sample(
        self, query: str, instrumental: bool = False, vocal_language: Optional[str] = None,
        temperature: float = 0.85,
    ) -> dict[str, Any]:
        from acestep.inference import create_sample

        self._log_lm_info("create_sample", temp=temperature, instrumental=instrumental, language=vocal_language)
        result = await asyncio.to_thread(
            create_sample, self.llm_handler,
            query=query, instrumental=instrumental, vocal_language=vocal_language,
            temperature=temperature,
        )
        return result.to_dict()

    async def format_sample(
        self, caption: str, lyrics: str, user_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        from acestep.inference import format_sample

        self._log_lm_info("format_sample")
        result = await asyncio.to_thread(
            format_sample, self.llm_handler,
            caption=caption, lyrics=lyrics, user_metadata=user_metadata,
        )
        return result.to_dict()
