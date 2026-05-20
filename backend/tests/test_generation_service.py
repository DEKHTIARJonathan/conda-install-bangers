from typing import Any, Callable, Optional

import pytest

from bangers.services.generation import GenerationService


class _FakeBackend:
    def __init__(self, *, ready: bool = False) -> None:
        self._ready = ready
        self.last_generate_params: dict[str, Any] | None = None
        self.dit_initialized = ready
        self.lm_initialized = False
        self.lm_disabled = False
        self.active_dit_model = "fake-dit" if ready else ""
        self.active_lm_model = ""
        self.dit_handler = None

    async def initialize(self, **_kwargs: Any) -> tuple[str, bool]:
        self._ready = True
        self.dit_initialized = True
        return "ready", True

    async def unload(self) -> None:
        self._ready = False
        self.dit_initialized = False

    async def generate(
        self,
        params_dict: dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        self.last_generate_params = dict(params_dict)
        return {"success": True, "audios": []}

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return "test"

    @property
    def init_stage(self) -> str:
        return "ready" if self._ready else "idle"

    @property
    def init_error(self) -> str:
        return ""

    @property
    def download_progress(self) -> float:
        return 0.0

    async def initialize_dit(self, **_kwargs: Any) -> tuple[str, bool]:
        self._ready = True
        self.dit_initialized = True
        return "ready", True

    async def initialize_lm(self, **_kwargs: Any) -> tuple[str, bool]:
        self.lm_initialized = True
        return "ready", True

    async def create_sample(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True}

    async def format_sample(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True}


@pytest.mark.asyncio
async def test_backend_ready_uses_single_backend_state():
    service = GenerationService()
    service.backend = _FakeBackend(ready=True)

    assert service.backend_ready is True
    assert service.dit_initialized is True


@pytest.mark.asyncio
async def test_generate_rejects_when_backend_not_initialized():
    service = GenerationService()
    service.backend = _FakeBackend(ready=False)

    result = await service.generate({"caption": "x"})
    assert result["success"] is False
    assert "not initialized" in result["error"]


@pytest.mark.asyncio
async def test_loading_state_is_explicit_and_cleared():
    service = GenerationService()

    assert service.loading_state is None
    service._set_loading("dit", "acestep-v15-turbo")
    assert service.loading_state == {
        "kind": "dit",
        "model_name": "acestep-v15-turbo",
        "started_at": service.loading_state["started_at"],
    }
    service._clear_loading()
    assert service.loading_state is None


@pytest.mark.asyncio
async def test_generate_normalizes_ace_params():
    service = GenerationService()
    backend = _FakeBackend(ready=True)
    service.backend = backend

    result = await service.generate({
        "caption": "bright synth pop",
        "instrumental": True,
        "audio_format": "unsupported",
        "vocal_language": "fr",
    })

    assert result["success"] is True
    assert backend.last_generate_params is not None
    assert backend.last_generate_params["caption"] == "bright synth pop"
    assert backend.last_generate_params["audio_format"] == "flac"
    assert backend.last_generate_params["vocal_language"] == "en"
    assert backend.last_generate_params["lyrics"] == ""


@pytest.mark.asyncio
async def test_generate_applies_lyrics_pipeline_before_backend(monkeypatch):
    from bangers.services import generation as generation_module

    service = GenerationService()
    backend = _FakeBackend(ready=True)
    service.backend = backend

    async def fake_prepare(params: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        prepared = dict(params)
        prepared["lyrics"] = "[verse]\nClean lyric"
        return prepared

    monkeypatch.setattr(generation_module, "prepare_generation_params", fake_prepare)

    result = await service.generate({
        "caption": "angry rock",
        "lyrics": "raw lyric",
        "instrumental": False,
    })

    assert result["success"] is True
    assert backend.last_generate_params is not None
    assert backend.last_generate_params["lyrics"] == "[verse]\nClean lyric"


@pytest.mark.asyncio
async def test_generate_does_not_trust_unverified_lyrics_prepared_flag(monkeypatch):
    from bangers.services import generation as generation_module

    service = GenerationService()
    backend = _FakeBackend(ready=True)
    service.backend = backend

    async def fake_prepare(params: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        prepared = dict(params)
        prepared["lyrics"] = "[verse]\nReviewed by guardrail"
        return prepared

    monkeypatch.setattr(generation_module, "prepare_generation_params", fake_prepare)

    result = await service.generate(
        {
            "caption": "custom tab vocal song",
            "lyrics": "[verse]\nRaw user lyric",
            "instrumental": False,
        },
        lyrics_prepared=True,
    )

    assert result["success"] is True
    assert backend.last_generate_params is not None
    assert backend.last_generate_params["lyrics"] == "[verse]\nReviewed by guardrail"


@pytest.mark.asyncio
async def test_generate_strips_verified_lyrics_pipeline_marker_before_backend():
    from bangers.services.lyrics_pipeline import (
        LYRICS_PIPELINE_PREPARED_KEY,
        mark_lyrics_pipeline_prepared,
    )

    service = GenerationService()
    backend = _FakeBackend(ready=True)
    service.backend = backend

    params = mark_lyrics_pipeline_prepared({
        "caption": "clean vocal",
        "lyrics": "[verse]\nAlready reviewed",
        "instrumental": False,
    })

    result = await service.generate(params, lyrics_prepared=True)

    assert result["success"] is True
    assert backend.last_generate_params is not None
    assert backend.last_generate_params["lyrics"] == "[verse]\nAlready reviewed"
    assert LYRICS_PIPELINE_PREPARED_KEY not in backend.last_generate_params


@pytest.mark.asyncio
async def test_generate_delegates_to_remote_workers_when_configured(monkeypatch):
    from bangers.config import settings
    from bangers.services import distributed as distributed_module
    from bangers.services.lyrics_pipeline import mark_lyrics_pipeline_prepared

    service = GenerationService()
    service.backend = _FakeBackend(ready=False)
    captured: dict[str, Any] = {}

    async def fake_remote_generate(
        params: dict[str, Any],
        *,
        progress_callback: Callable | None = None,
        lyrics_prepared: bool = False,
    ) -> dict[str, Any]:
        captured["params"] = dict(params)
        captured["lyrics_prepared"] = lyrics_prepared
        return {"success": True, "audios": [{"path": "remote.flac"}]}

    old_role = settings.DISTRIBUTED_ROLE
    old_workers = settings.DISTRIBUTED_WORKERS
    try:
        settings.DISTRIBUTED_ROLE = "coordinator"
        settings.DISTRIBUTED_WORKERS = ("http://spark-a:8000",)
        monkeypatch.setattr(
            distributed_module.distributed_cluster,
            "generate_on_remote_music_worker",
            fake_remote_generate,
        )

        params = mark_lyrics_pipeline_prepared({
            "caption": "remote clean song",
            "lyrics": "[verse]\nAlready reviewed",
            "instrumental": False,
        })
        result = await service.generate(params, lyrics_prepared=True)

        assert result["success"] is True
        assert captured["lyrics_prepared"] is True
        assert captured["params"]["lyrics"] == "[verse]\nAlready reviewed"
        assert all(not key.startswith("_lyrics_pipeline_") for key in captured["params"])
        assert all(not key.startswith("_distributed_") for key in captured["params"])
    finally:
        settings.DISTRIBUTED_ROLE = old_role
        settings.DISTRIBUTED_WORKERS = old_workers


def test_public_params_strips_distributed_internal_keys():
    service = GenerationService()

    public = service.public_params({
        "caption": "x",
        "_distributed_lyrics_prepared": True,
        "_distributed_ace_lm_worker": "spark-b",
    })

    assert public == {"caption": "x"}


@pytest.mark.asyncio
async def test_generate_fails_closed_when_guardrail_llm_unavailable():
    service = GenerationService()
    service.backend = _FakeBackend(ready=True)

    result = await service.generate({
        "caption": "vocal pop",
        "lyrics": "[verse]\nNeeds review",
        "instrumental": False,
    })

    assert result["success"] is False
    assert "Chat LLM" in result["error"]
