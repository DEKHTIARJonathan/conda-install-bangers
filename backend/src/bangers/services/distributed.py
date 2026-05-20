import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from loguru import logger

from bangers.config import (
    DISTRIBUTED_CAPABILITY_ACE_LM,
    DISTRIBUTED_CAPABILITY_CHAT_LLM,
    DISTRIBUTED_CAPABILITY_MUSIC,
    settings,
)
from bangers.models.common import GpuDeviceStats, GpuStatsResponse


ProgressCallback = Callable[[float, str], None]


@dataclass
class WorkerState:
    url: str
    node_id: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    busy: bool = False
    ready: bool = False
    capability_ready: dict[str, bool] = field(default_factory=dict)
    active_dit_model: str = ""
    active_lm_model: str = ""
    active_chat_llm_model: str = ""
    last_seen: float = 0.0
    error: str = ""

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def ready_for(self, capability: str) -> bool:
        return self.supports(capability) and self.capability_ready.get(capability, self.ready)


class DistributedGenerationError(RuntimeError):
    """Raised when a remote worker cannot complete a delegated operation."""


def _headers() -> dict[str, str]:
    if not settings.DISTRIBUTED_TOKEN:
        return {}
    return {"X-Bangers-Worker-Token": settings.DISTRIBUTED_TOKEN}


def _normalize_worker_url(url: str) -> str:
    return url.rstrip("/")


def _task_skips_ace_lm(params: dict[str, Any]) -> bool:
    return params.get("task_type", "text2music") in {"cover", "repaint"}


def _needs_ace_lm(params: dict[str, Any]) -> bool:
    if _task_skips_ace_lm(params):
        return False
    if str(params.get("audio_codes", "") or "").strip():
        return False
    thinking = bool(params.get("thinking", True))
    cot = (
        bool(params.get("use_cot_caption", False))
        or bool(params.get("use_cot_language", True))
        or bool(params.get("use_cot_metas", True))
    )
    return thinking or cot


class DistributedCluster:
    """Client for coordinator-to-worker inference delegation."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerState] = {}
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()
        self.refresh_from_settings()

    def refresh_from_settings(self) -> None:
        for url in settings.DISTRIBUTED_WORKERS:
            normalized = _normalize_worker_url(url)
            self._workers.setdefault(normalized, WorkerState(url=normalized))
        stale = set(self._workers) - {
            _normalize_worker_url(url) for url in settings.DISTRIBUTED_WORKERS
        }
        for url in stale:
            del self._workers[url]

    @property
    def enabled(self) -> bool:
        return settings.delegates_to_workers

    async def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=_headers(),
            timeout=timeout or settings.DISTRIBUTED_REQUEST_TIMEOUT_SECONDS,
        )

    async def refresh(self, *, force: bool = False) -> list[WorkerState]:
        self.refresh_from_settings()
        if not self.enabled:
            return []
        now = time.time()
        if not force and now - self._last_refresh < 2.0:
            return list(self._workers.values())

        async with self._refresh_lock:
            now = time.time()
            if not force and now - self._last_refresh < 2.0:
                return list(self._workers.values())
            async with await self._client(timeout=10.0) as client:
                await asyncio.gather(
                    *(self._refresh_worker(client, worker) for worker in self._workers.values()),
                    return_exceptions=True,
                )
            self._last_refresh = time.time()
            return list(self._workers.values())

    async def _refresh_worker(self, client: httpx.AsyncClient, worker: WorkerState) -> None:
        try:
            response = await client.get(f"{worker.url}/api/internal/worker/status")
            response.raise_for_status()
            data = response.json()
            worker.node_id = str(data.get("node_id") or worker.url)
            worker.capabilities = frozenset(str(c) for c in data.get("capabilities", []))
            worker.busy = bool(data.get("busy", False))
            worker.ready = bool(data.get("ready", False))
            worker.capability_ready = {
                DISTRIBUTED_CAPABILITY_MUSIC: bool(data.get("music_ready", False)),
                DISTRIBUTED_CAPABILITY_ACE_LM: bool(data.get("ace_lm_ready", False)),
                DISTRIBUTED_CAPABILITY_CHAT_LLM: bool(data.get("chat_ready", False)),
            }
            worker.active_dit_model = str(data.get("dit_model") or "")
            worker.active_lm_model = str(data.get("lm_model") or "")
            worker.active_chat_llm_model = str(data.get("chat_llm_model") or "")
            worker.last_seen = time.time()
            worker.error = ""
        except Exception as exc:
            worker.ready = False
            worker.capability_ready = {}
            worker.busy = False
            worker.active_dit_model = ""
            worker.active_lm_model = ""
            worker.active_chat_llm_model = ""
            worker.error = str(exc)

    async def collect_gpu_stats(self) -> list[GpuStatsResponse]:
        if not self.enabled:
            return []
        workers = await self.refresh()
        async with await self._client(timeout=5.0) as client:
            results = await asyncio.gather(
                *(self._fetch_worker_gpu_stats(client, worker) for worker in workers),
                return_exceptions=True,
            )
        stats: list[GpuStatsResponse] = []
        for worker, result in zip(workers, results):
            if isinstance(result, GpuStatsResponse):
                stats.append(result)
            elif isinstance(result, BaseException):
                stats.append(self._worker_gpu_stats_error(worker, str(result)))
        return stats

    async def _fetch_worker_gpu_stats(
        self,
        client: httpx.AsyncClient,
        worker: WorkerState,
    ) -> GpuStatsResponse:
        response = await client.get(f"{worker.url}/api/internal/worker/gpu-stats")
        response.raise_for_status()
        return GpuStatsResponse.model_validate(response.json())

    async def switch_dit_model(self, model_name: str) -> list[str]:
        return await self._switch_model_on_workers(
            capability=DISTRIBUTED_CAPABILITY_MUSIC,
            endpoint="switch-dit",
            payload={"model_name": model_name},
            label="DiT",
        )

    async def switch_lm_model(self, model_name: str, runtime: str) -> list[str]:
        return await self._switch_model_on_workers(
            capability=DISTRIBUTED_CAPABILITY_ACE_LM,
            endpoint="switch-lm",
            payload={"model_name": model_name, "runtime": runtime},
            label="ACE 5Hz LM",
        )

    async def switch_chat_llm_model(self, model_name: str) -> list[str]:
        return await self._switch_model_on_workers(
            capability=DISTRIBUTED_CAPABILITY_CHAT_LLM,
            endpoint="switch-chat-llm",
            payload={"model_name": model_name},
            label="Chat LLM",
        )

    async def _switch_model_on_workers(
        self,
        *,
        capability: str,
        endpoint: str,
        payload: dict[str, Any],
        label: str,
    ) -> list[str]:
        if not self.enabled:
            raise DistributedGenerationError("Distributed workers are not enabled")
        workers = await self.refresh(force=True)
        targets = [worker for worker in workers if worker.supports(capability)]
        if not targets:
            raise DistributedGenerationError(f"No remote {label} worker is configured")

        async with await self._client() as client:
            results = await asyncio.gather(
                *(
                    self._switch_worker_model(client, worker, endpoint, payload)
                    for worker in targets
                ),
                return_exceptions=True,
            )

        loaded_nodes: list[str] = []
        errors: list[str] = []
        for worker, result in zip(targets, results):
            node_id = worker.node_id or worker.url
            if isinstance(result, BaseException):
                errors.append(f"{node_id}: {result}")
            else:
                loaded_nodes.append(node_id)
        if errors:
            raise DistributedGenerationError("; ".join(errors))
        await self.refresh(force=True)
        return loaded_nodes

    async def _switch_worker_model(
        self,
        client: httpx.AsyncClient,
        worker: WorkerState,
        endpoint: str,
        payload: dict[str, Any],
    ) -> None:
        response = await client.post(
            f"{worker.url}/api/internal/worker/models/{endpoint}",
            json=payload,
        )
        response.raise_for_status()

    @staticmethod
    def _worker_gpu_stats_error(worker: WorkerState, error: str) -> GpuStatsResponse:
        node_id = worker.node_id or worker.url
        return GpuStatsResponse(
            device="unavailable",
            error=f"{node_id}: {error}",
            gpus=[
                GpuDeviceStats(
                    node_id=node_id,
                    label=node_id,
                    error=error,
                    busy=worker.busy,
                )
            ],
        )

    async def select_worker(self, capability: str) -> WorkerState | None:
        workers = await self.refresh()
        candidates = [w for w in workers if w.ready_for(capability)]
        if not candidates:
            return None
        free = [w for w in candidates if not w.busy]
        return random.choice(free or candidates)

    async def prepare_with_remote_ace_lm(
        self,
        params: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not self.enabled or not _needs_ace_lm(params):
            return params

        worker = await self.select_worker(DISTRIBUTED_CAPABILITY_ACE_LM)
        if worker is None:
            raise DistributedGenerationError("No remote ACE 5Hz LM worker is ready")

        if progress_callback is not None:
            progress_callback(0.04, f"Running ACE 5Hz LM on {worker.node_id or worker.url}...")

        async with await self._client() as client:
            response = await client.post(
                f"{worker.url}/api/internal/worker/ace-lm/prepare",
                json={"params": params},
            )
            response.raise_for_status()
            data = response.json()
        if not data.get("success", False):
            raise DistributedGenerationError(data.get("error") or "Remote ACE LM failed")
        prepared = dict(data.get("params") or params)
        prepared["_distributed_ace_lm_worker"] = worker.node_id or worker.url
        return prepared

    async def generate_on_remote_music_worker(
        self,
        params: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
        lyrics_prepared: bool = False,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        music_worker = await self.select_worker(DISTRIBUTED_CAPABILITY_MUSIC)
        if music_worker is None:
            return None

        routed_params = dict(params)
        if lyrics_prepared:
            routed_params["_distributed_lyrics_prepared"] = True
        if not music_worker.supports(DISTRIBUTED_CAPABILITY_ACE_LM):
            routed_params = await self.prepare_with_remote_ace_lm(
                routed_params,
                progress_callback=progress_callback,
            )

        if progress_callback is not None:
            progress_callback(0.05, f"Queued on {music_worker.node_id or music_worker.url}...")

        async with await self._client() as client:
            submit = await client.post(
                f"{music_worker.url}/api/internal/worker/jobs",
                json={
                    "params": routed_params,
                    "lyrics_prepared": lyrics_prepared,
                },
            )
            submit.raise_for_status()
            worker_job_id = submit.json()["job_id"]

            try:
                status = await self._poll_worker_job(
                    client,
                    music_worker,
                    worker_job_id,
                    progress_callback=progress_callback,
                )
            except BaseException:
                await self._cancel_worker_job(client, music_worker, worker_job_id)
                raise

            if status.get("status") != "completed":
                raise DistributedGenerationError(status.get("error") or "Remote generation failed")

            audios = await self._fetch_worker_artifacts(
                client,
                music_worker,
                worker_job_id,
                status.get("results", []),
            )

        return {
            "success": True,
            "audios": audios,
            "status_message": f"Generated on {music_worker.node_id or music_worker.url}",
            "extra_outputs": {
                "worker": music_worker.node_id or music_worker.url,
                "worker_job_id": worker_job_id,
            },
            "timings": status.get("timings", {}),
            "error": None,
        }

    async def _poll_worker_job(
        self,
        client: httpx.AsyncClient,
        worker: WorkerState,
        job_id: str,
        *,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        while True:
            response = await client.get(f"{worker.url}/api/internal/worker/jobs/{job_id}")
            response.raise_for_status()
            status = response.json()
            if progress_callback is not None:
                progress_callback(float(status.get("progress") or 0.0), status.get("stage") or "")
            if status.get("status") in {"completed", "failed", "cancelled"}:
                return status
            await asyncio.sleep(1.0)

    async def _cancel_worker_job(
        self,
        client: httpx.AsyncClient,
        worker: WorkerState,
        job_id: str,
    ) -> None:
        try:
            await client.delete(f"{worker.url}/api/internal/worker/jobs/{job_id}")
        except Exception as exc:
            logger.warning(f"Failed to cancel remote worker job {job_id}: {exc}")

    async def _fetch_worker_artifacts(
        self,
        client: httpx.AsyncClient,
        worker: WorkerState,
        job_id: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        settings.ensure_dirs()
        audios: list[dict[str, Any]] = []
        for index, audio in enumerate(results):
            response = await client.get(
                f"{worker.url}/api/internal/worker/jobs/{job_id}/artifacts/{index}"
            )
            response.raise_for_status()
            filename = self._artifact_filename(audio, response)
            dest = settings.AUDIO_DIR / filename
            await asyncio.to_thread(dest.write_bytes, response.content)
            copied = dict(audio)
            copied["path"] = str(dest)
            copied["remote_worker"] = worker.node_id or worker.url
            copied.pop("tensor", None)
            audios.append(copied)
        return audios

    @staticmethod
    def _artifact_filename(audio: dict[str, Any], response: httpx.Response) -> str:
        path = str(audio.get("path") or "")
        basename = os.path.basename(path)
        if basename:
            return basename
        key = str(audio.get("key") or "remote-audio")
        content_type = response.headers.get("content-type", "")
        ext = ".flac"
        if "mpeg" in content_type:
            ext = ".mp3"
        elif "wav" in content_type:
            ext = ".wav"
        elif "opus" in content_type:
            ext = ".opus"
        elif "aac" in content_type:
            ext = ".aac"
        return f"{key}{ext}"

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        worker = await self.select_worker(DISTRIBUTED_CAPABILITY_CHAT_LLM)
        if worker is None:
            raise DistributedGenerationError("No remote chat LLM worker is available")
        async with await self._client() as client:
            response = await client.post(
                f"{worker.url}/api/internal/worker/chat",
                json={
                    "messages": messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()
        if not data.get("success", False):
            raise DistributedGenerationError(data.get("error") or "Remote chat LLM failed")
        return str(data.get("text") or "")


class RemoteChatRuntime:
    """ChatRuntime-compatible adapter backed by a remote chat worker."""

    def loaded_model_name(self) -> str:
        return ""

    async def is_available(self) -> bool:
        return distributed_cluster.enabled

    def is_model_loadable(self, model_name: str) -> bool:
        return bool(model_name) and distributed_cluster.enabled

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        allow_holders: frozenset[str] | None = None,
    ) -> str:
        _ = allow_holders
        return await distributed_cluster.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )


distributed_cluster = DistributedCluster()
remote_chat_runtime = RemoteChatRuntime()
