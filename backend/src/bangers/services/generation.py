import time
import uuid
from typing import Any, Callable, Optional

from bangers.config import settings
from bangers.backends.ace_step_backend import AceStepBackend
from bangers.services.generation_request_builder import normalize_generation_params
from bangers.services.lyrics_pipeline import (
    format_song_spec,
    is_lyrics_pipeline_prepared,
    prepare_generation_params,
    strip_lyrics_pipeline_internal_keys,
)
from bangers.services.music_specs import build_music_spec


class GenerationService:
    """Owns the single ACE-Step music generation backend."""

    def __init__(self) -> None:
        self.backend = AceStepBackend()
        self.model_download_status: dict[str, str] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancelled: set[str] = set()
        self._loading_state: dict[str, Any] | None = None

    @property
    def backend_ready(self) -> bool:
        return self.backend.is_ready

    @property
    def loading_state(self) -> dict[str, Any] | None:
        """Snapshot of the currently-running ACE-Step model load."""
        return dict(self._loading_state) if self._loading_state else None

    def _set_loading(self, kind: str, model_name: str = "") -> None:
        self._loading_state = {
            "kind": kind,
            "model_name": model_name,
            "started_at": time.time(),
        }

    def _clear_loading(self) -> None:
        self._loading_state = None

    @property
    def dit_initialized(self) -> bool:
        return self.backend.dit_initialized

    @property
    def lm_initialized(self) -> bool:
        return self.backend.lm_initialized

    @property
    def lm_disabled(self) -> bool:
        return bool(getattr(self.backend, "lm_disabled", False))

    @property
    def active_dit_model(self) -> str:
        return self.backend.active_dit_model

    @property
    def active_lm_model(self) -> str:
        return self.backend.active_lm_model

    @property
    def device(self) -> str:
        return self.backend.device

    @property
    def init_stage(self) -> str:
        return self.backend.init_stage

    @property
    def init_error(self) -> str:
        return self.backend.init_error

    @property
    def download_progress(self) -> float:
        return self.backend.download_progress

    @property
    def dit_handler(self):
        return self.backend.dit_handler

    async def initialize_dit(self, **kwargs: Any) -> tuple[str, bool]:
        return await self.backend.initialize_dit(**kwargs)

    async def initialize_lm(self, **kwargs: Any) -> tuple[str, bool]:
        return await self.backend.initialize_lm(**kwargs)

    async def create_sample(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.pop("query", "") or "")
        kwargs.pop("vocal_language", None)
        temperature = float(kwargs.pop("temperature", 0.85) or 0.85)
        return await build_music_spec(
            prompt=query,
            instrumental=bool(kwargs.pop("instrumental", False)),
            source="create",
            temperature=temperature,
            **kwargs,
        )

    async def format_sample(self, **kwargs: Any) -> dict[str, Any]:
        return await format_song_spec(**kwargs)

    async def prepare_params(
        self,
        params_dict: dict[str, Any],
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        normalized_params = normalize_generation_params(params_dict)
        return await prepare_generation_params(
            normalized_params,
            allow_holders=allow_holders,
        )

    def public_params(self, params_dict: dict[str, Any]) -> dict[str, Any]:
        params = strip_lyrics_pipeline_internal_keys(params_dict)
        return {
            key: value
            for key, value in params.items()
            if not key.startswith("_distributed_")
        }

    async def generate(
        self,
        params_dict: dict[str, Any],
        progress_callback: Optional[Callable] = None,
        *,
        lyrics_prepared: bool = False,
    ) -> dict[str, Any]:
        if not settings.delegates_to_workers and not self.backend.is_ready:
            return {"success": False, "error": "ACE-Step backend not initialized"}

        try:
            trusted_lyrics_prepared = lyrics_prepared and (
                is_lyrics_pipeline_prepared(params_dict)
                or params_dict.get("_distributed_lyrics_prepared") is True
            )
            if trusted_lyrics_prepared:
                normalized_params = normalize_generation_params(params_dict)
            else:
                from bangers.services.gpu_lock import gpu_lock

                holder = gpu_lock.holder if gpu_lock.is_locked else None
                allow_holders = frozenset({holder}) if holder else None
                normalized_params = await self.prepare_params(
                    params_dict,
                    allow_holders=allow_holders,
                )
            normalized_params = self.public_params(normalized_params)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        if settings.delegates_to_workers:
            try:
                from bangers.services.distributed import distributed_cluster

                remote_result = await distributed_cluster.generate_on_remote_music_worker(
                    normalized_params,
                    progress_callback=progress_callback,
                    lyrics_prepared=lyrics_prepared,
                )
                if remote_result is not None:
                    return remote_result
            except Exception as exc:
                return {"success": False, "error": str(exc)}
            return {"success": False, "error": "No remote music worker is ready"}

        return await self.backend.generate(normalized_params, progress_callback=progress_callback)

    def create_job(self, client_id: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "stage": "",
            "results": [],
            "error": None,
            "timings": {},
            "created_at": time.time(),
            "client_id": client_id or "",
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def cancel_job(self, job_id: str) -> bool:
        self._cancelled.add(job_id)
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "cancelled"
            return True
        return False

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> None:
        now = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if now - job.get("created_at", 0) > max_age_seconds
        ]
        for jid in expired:
            del self._jobs[jid]


generation_service = GenerationService()
