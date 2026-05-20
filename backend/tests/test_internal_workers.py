import pytest


@pytest.mark.asyncio
async def test_internal_worker_status_reports_capability_readiness(client):
    from bangers.backends.ace_step_backend import AceStepBackend
    from bangers.config import DISTRIBUTED_CAPABILITY_MUSIC, settings
    from bangers.services.generation import generation_service

    old_role = settings.DISTRIBUTED_ROLE
    old_capabilities = settings.DISTRIBUTED_CAPABILITIES
    try:
        settings.DISTRIBUTED_ROLE = "worker"
        settings.DISTRIBUTED_CAPABILITIES = frozenset({DISTRIBUTED_CAPABILITY_MUSIC})

        ace = AceStepBackend()
        ace.dit_initialized = True
        ace.active_dit_model = "acestep-v15-turbo"
        generation_service.backend = ace

        resp = await client.get("/api/internal/worker/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "worker"
        assert data["capabilities"] == [DISTRIBUTED_CAPABILITY_MUSIC]
        assert data["ready"] is True
        assert data["music_ready"] is True
        assert data["ace_lm_ready"] is False
    finally:
        settings.DISTRIBUTED_ROLE = old_role
        settings.DISTRIBUTED_CAPABILITIES = old_capabilities


@pytest.mark.asyncio
async def test_internal_worker_status_requires_token_when_configured(client):
    from bangers.config import DISTRIBUTED_CAPABILITY_CHAT_LLM, settings
    from bangers.routers import internal_workers as internal_workers_router

    old_role = settings.DISTRIBUTED_ROLE
    old_capabilities = settings.DISTRIBUTED_CAPABILITIES
    old_token = settings.DISTRIBUTED_TOKEN
    try:
        settings.DISTRIBUTED_ROLE = "worker"
        settings.DISTRIBUTED_CAPABILITIES = frozenset({DISTRIBUTED_CAPABILITY_CHAT_LLM})
        settings.DISTRIBUTED_TOKEN = "secret"

        missing = await client.get("/api/internal/worker/status")
        assert missing.status_code == 401

        ok = await client.get(
            "/api/internal/worker/status",
            headers={"X-Bangers-Worker-Token": "secret"},
        )
        assert ok.status_code == 200
        assert ok.json()["chat_ready"] is True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            internal_workers_router,
            "get_loaded_chat_model_name",
            lambda: "Qwen3-4B-Instruct-2507",
        )
        try:
            loaded = await client.get(
                "/api/internal/worker/status",
                headers={"X-Bangers-Worker-Token": "secret"},
            )
            assert loaded.status_code == 200
            data = loaded.json()
            assert data["chat_ready"] is True
            assert data["chat_llm_model"] == "Qwen3-4B-Instruct-2507"
        finally:
            monkeypatch.undo()
    finally:
        settings.DISTRIBUTED_ROLE = old_role
        settings.DISTRIBUTED_CAPABILITIES = old_capabilities
        settings.DISTRIBUTED_TOKEN = old_token
