import asyncio
import json
import sys
import types
from typing import Any, Callable

import pytest


class _ReadyGenerationBackend:
    def __init__(self) -> None:
        self.last_generate_params: dict[str, Any] | None = None
        self.dit_initialized = True
        self.lm_initialized = False
        self.lm_disabled = True
        self.active_dit_model = "fake-dit"
        self.active_lm_model = ""
        self.dit_handler = None

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return "test"

    @property
    def init_stage(self) -> str:
        return "ready"

    @property
    def init_error(self) -> str:
        return ""

    @property
    def download_progress(self) -> float:
        return 0.0

    async def generate(
        self,
        params_dict: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        self.last_generate_params = dict(params_dict)
        if progress_callback:
            progress_callback(1.0, "done")
        return {
            "success": True,
            "audios": [
                {
                    "path": "fake.flac",
                    "key": "fake.flac",
                    "sample_rate": 48000,
                    "params": dict(params_dict),
                }
            ],
        }


def test_generation_cancelled_bypasses_broad_exception_handlers():
    from bangers.routers.generation import GenerationCancelled

    with pytest.raises(GenerationCancelled):
        try:
            raise GenerationCancelled()
        except Exception as exc:
            pytest.fail(f"Cancellation should not be handled as a generation failure: {exc!r}")


@pytest.mark.asyncio
async def test_ace_lm_initialization_maps_nanovllm_alias_to_vllm(monkeypatch, tmp_path):
    from bangers.backends.ace_step_backend import AceStepBackend
    from bangers.config import settings

    calls: list[dict[str, Any]] = []

    class FakeLLMHandler:
        def initialize(self, **kwargs):
            calls.append(kwargs)
            return "ready", True

    llm_module = types.ModuleType("acestep.llm_inference")
    llm_module.LLMHandler = FakeLLMHandler
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_module)

    old_root = settings.ACESTEP_PROJECT_ROOT
    try:
        settings.ACESTEP_PROJECT_ROOT = str(tmp_path)
        ace = AceStepBackend()

        status, ok = await ace.initialize_lm(
            lm_model_path="acestep-5Hz-lm-0.6B",
            backend="nano-vllm",
            device="cuda",
        )

        assert (status, ok) == ("ready", True)
        assert calls[0]["backend"] == "vllm"
    finally:
        settings.ACESTEP_PROJECT_ROOT = old_root


def test_vllm_eager_wrapper_auto_enables_for_gb10(monkeypatch):
    from bangers.backends.ace_step_backend import _install_vllm_eager_wrapper

    calls: list[dict[str, Any]] = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_device):
            return "NVIDIA GB10"

        @staticmethod
        def get_device_capability(_device):
            return (12, 1)

    class FakeHandler:
        def _initialize_5hz_lm_vllm(self, model_path, enforce_eager=False):
            calls.append({"model_path": model_path, "enforce_eager": enforce_eager})
            return "ready"

    monkeypatch.delenv("BANGERS_VLLM_ENFORCE_EAGER", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))

    handler = FakeHandler()
    _install_vllm_eager_wrapper(handler, "cuda")

    assert handler._initialize_5hz_lm_vllm("model") == "ready"
    assert calls == [{"model_path": "model", "enforce_eager": True}]


@pytest.mark.asyncio
async def test_generate_when_dit_not_loaded(client):
    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "test song",
    })
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_generate_no_dit_selected_returns_explicit_error(client):
    from bangers.backends.ace_step_backend import AceStepBackend
    from bangers.services.generation import generation_service

    ace = AceStepBackend()
    ace.active_dit_model = ""
    ace.dit_initialized = False
    generation_service.backend = ace

    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "test song",
    })
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "no_dit_model_selected"
    assert "Models page" in detail["message"]
    assert detail["missing"] == ["dit_model"]


@pytest.mark.asyncio
async def test_generate_dit_loading_distinct_from_no_selection(client):
    from bangers.backends.ace_step_backend import AceStepBackend
    from bangers.services.generation import generation_service

    ace = AceStepBackend()
    ace.active_dit_model = "some-dit"
    ace.dit_initialized = False
    generation_service.backend = ace

    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "test song",
    })
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "dit_model_not_loaded"
    assert "some-dit" in detail["message"]


@pytest.mark.asyncio
async def test_custom_generation_reviews_user_provided_lyrics_before_backend(client, monkeypatch):
    from bangers.db.connection import get_db
    from bangers.services import lyrics_pipeline
    from bangers.services.generation import generation_service

    backend = _ReadyGenerationBackend()
    generation_service.backend = backend

    async def fake_chat(*_args, **_kwargs):
        return "<guarded_lyrics>\n[verse]\nReviewed custom lyric\n</guarded_lyrics>"

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "custom tab vocal song",
        "lyrics": "[verse]\nRaw custom lyric",
        "instrumental": False,
        "inference_steps": 1,
    })

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(50):
        job_resp = await client.get(f"/api/generate/{job_id}")
        body = job_resp.json()
        if body["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.01)

    assert body["status"] == "completed", body
    assert backend.last_generate_params is not None
    assert backend.last_generate_params["lyrics"] == "[verse]\nReviewed custom lyric"

    db = await get_db()
    cursor = await db.execute(
        "SELECT params_json, result_json FROM generation_history WHERE id = ?",
        (body["history_id"],),
    )
    history = await cursor.fetchone()
    params = json.loads(history["params_json"])
    results = json.loads(history["result_json"])

    assert params["lyrics"] == "[verse]\nReviewed custom lyric"
    assert results[0]["params"]["lyrics"] == "[verse]\nReviewed custom lyric"
    assert all(not key.startswith("_lyrics_pipeline_") for key in params)


@pytest.mark.asyncio
async def test_custom_generation_reports_code_of_conduct_rejected_lyrics(client, monkeypatch):
    from bangers.services import lyrics_pipeline
    from bangers.services.generation import generation_service

    backend = _ReadyGenerationBackend()
    generation_service.backend = backend

    async def fake_chat(*_args, **_kwargs):
        return "<lyrics_rejected>\nSevere Code of Conduct violation.\n</lyrics_rejected>"

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "custom tab vocal song",
        "lyrics": "[verse]\nRejected custom lyric",
        "instrumental": False,
        "inference_steps": 1,
    })

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(50):
        job_resp = await client.get(f"/api/generate/{job_id}")
        body = job_resp.json()
        if body["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.01)

    assert body["status"] == "failed", body
    assert "Code of Conduct" in body["error"]
    assert backend.last_generate_params is None


@pytest.mark.asyncio
async def test_generation_client_id_owns_job_but_not_backend_params(client):
    from bangers.services.generation import generation_service

    backend = _ReadyGenerationBackend()
    generation_service.backend = backend

    resp = await client.post("/api/generate", json={
        "task_type": "text2music",
        "caption": "instrumental test song",
        "instrumental": True,
        "client_id": "browser-session-1",
        "inference_steps": 1,
    })

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(50):
        job_resp = await client.get(f"/api/generate/{job_id}")
        body = job_resp.json()
        if body["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.01)

    assert body["status"] == "completed", body
    assert generation_service.get_job(job_id)["client_id"] == "browser-session-1"
    assert backend.last_generate_params is not None
    assert "client_id" not in backend.last_generate_params


@pytest.mark.asyncio
async def test_get_nonexistent_job(client):
    resp = await client.get("/api/generate/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_format_when_lm_not_loaded(client):
    resp = await client.post("/api/format", json={
        "caption": "test caption",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert "Chat LLM" in body["error"]


@pytest.mark.asyncio
async def test_sample_when_lm_not_loaded(client):
    resp = await client.post("/api/sample", json={
        "query": "happy country song",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["quality_profile"] == "Country"
    assert body["spec_source"].endswith("fallback-lyrics")
    assert "acoustic strumming" in body["caption"]
    assert body["lyrics"].startswith("[verse]")
