import pytest


@pytest.mark.asyncio
async def test_available_models_iterates_curated_registry_only(client):
    """The /models/available list is driven entirely by `bangers.model_registry`,
    not by upstream `acestep.model_downloader.SUBMODEL_REGISTRY`."""
    from bangers.model_registry import (
        ACE_DIT_MODELS,
        ACE_LM_MODELS,
        ACE_MAIN_BUNDLE,
        CHAT_LLM_MODELS,
    )
    from bangers.services.generation import generation_service

    generation_service.model_download_status.clear()

    response = await client.get("/api/models/available")
    assert response.status_code == 200
    models = {model["name"]: model for model in response.json()["models"]}

    # Curated bundle entries appear and report the bundle repo_id.
    assert models["acestep-v15-turbo"]["model_type"] == "dit"
    assert models["acestep-v15-turbo"]["repo_id"] == ACE_MAIN_BUNDLE.repo_id
    assert models["acestep-5Hz-lm-1.7B"]["model_type"] == "lm"
    assert models["acestep-5Hz-lm-1.7B"]["repo_id"] == ACE_MAIN_BUNDLE.repo_id
    assert models["acestep-5Hz-lm-1.7B"]["size_mb"] > 0

    # Every curated ACE entry is present and uses its registry repo_id.
    for ace in ACE_DIT_MODELS + ACE_LM_MODELS:
        assert ace.name in models, f"expected {ace.name} in available models"
        expected_repo = (
            ACE_MAIN_BUNDLE.repo_id if ace.bundled else ace.repo_id
        )
        assert models[ace.name]["repo_id"] == expected_repo
        assert models[ace.name]["model_type"] == ace.model_type
        assert models[ace.name]["size_mb"] == ace.size_mb

    # Upstream-only entries that are NOT in our curated registry must not
    # leak through anymore.
    assert "acestep-v15-turbo-shift3" not in models
    assert "acestep-v15-turbo-shift1" not in models

    # Chat LLM metadata still surfaces from the registry.
    chat_model_metadata = {model.name: model for model in CHAT_LLM_MODELS}
    for name, metadata in chat_model_metadata.items():
        assert models[name]["model_type"] == "chat_llm"
        assert models[name]["repo_id"] == metadata.repo_id
        assert models[name]["size_mb"] > 0
        assert models[name]["compatibility"] == list(metadata.compatible_runtimes)
        assert models[name]["format"] == metadata.format
        assert models[name]["quantization"] == metadata.quantization


@pytest.mark.asyncio
async def test_downloading_default_ace_lm_uses_main_bundle(client, monkeypatch):
    """Asking for a bundled LM (1.7B) routes through `download_main_model`
    and marks both bundle-visible components as done."""
    import sys
    import types

    calls: list = []

    def download_main_model(checkpoint_dir):
        calls.append(checkpoint_dir)
        return True, "main bundle ready"

    # Inject a stub `acestep.model_downloader` (the conftest mocks the
    # `acestep` package itself, so we attach the submodule explicitly).
    downloader = types.ModuleType("acestep.model_downloader")
    downloader.download_main_model = download_main_model
    monkeypatch.setitem(sys.modules, "acestep.model_downloader", downloader)

    from bangers.services.generation import generation_service

    generation_service.model_download_status.clear()

    response = await client.post(
        "/api/models/download",
        json={"model_name": "acestep-5Hz-lm-1.7B"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert calls
    assert generation_service.model_download_status["acestep-5Hz-lm-1.7B"] == "done"
    assert generation_service.model_download_status["acestep-v15-turbo"] == "done"


@pytest.mark.asyncio
async def test_downloading_non_bundled_ace_dit_uses_snapshot_download(client, monkeypatch):
    """Curated ACE entries that aren't part of the main bundle download via
    `huggingface_hub.snapshot_download` directly — never via upstream's
    `download_submodel` (which would reject names not in its own registry).
    """
    import sys
    import types

    from bangers.model_registry import ACE_MODEL_BY_NAME

    snapshot_calls: list[dict] = []

    def fake_snapshot_download(repo_id, local_dir, **_kwargs):
        snapshot_calls.append({"repo_id": repo_id, "local_dir": local_dir})
        return local_dir

    hf_hub = types.ModuleType("huggingface_hub")
    hf_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_hub)

    # Sanity-fail loudly if download_submodel is reached.
    def _boom(*_a, **_kw):
        raise AssertionError("download_submodel must not be called for curated non-bundled entries")

    downloader = types.ModuleType("acestep.model_downloader")
    downloader.download_submodel = _boom
    monkeypatch.setitem(sys.modules, "acestep.model_downloader", downloader)

    from bangers.services.generation import generation_service

    generation_service.model_download_status.clear()

    target = "acestep-v15-base"
    expected_repo = ACE_MODEL_BY_NAME[target].repo_id
    assert ACE_MODEL_BY_NAME[target].bundled is False

    response = await client.post(
        "/api/models/download",
        json={"model_name": target},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "started"}
    assert len(snapshot_calls) == 1
    assert snapshot_calls[0]["repo_id"] == expected_repo
    assert snapshot_calls[0]["local_dir"].endswith(f"checkpoints/{target}")
    assert generation_service.model_download_status[target] == "done"


@pytest.mark.asyncio
async def test_unknown_model_download_is_rejected(client):
    """Names not present in any curated registry are rejected with 400."""
    from bangers.services.generation import generation_service

    generation_service.model_download_status.clear()

    response = await client.post(
        "/api/models/download",
        json={"model_name": "acestep-v15-turbo-shift3"},
    )
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_switch_model_rejected_while_another_model_is_loading(client):
    from bangers.services.generation import generation_service

    generation_service._set_loading("dit", "acestep-v15-base")

    response = await client.post(
        "/api/models/switch-lm",
        json={"model_name": "acestep-5Hz-lm-0.6B"},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "load_in_flight"
    assert detail["loading"]["kind"] == "dit"


@pytest.mark.asyncio
async def test_switch_lm_normalizes_legacy_nanovllm_runtime(client, monkeypatch):
    from bangers.db.connection import get_db
    from bangers.services.generation import generation_service

    calls: list[dict] = []

    async def fake_initialize_lm(**kwargs):
        calls.append(kwargs)
        return "ready", True

    monkeypatch.setattr(generation_service, "initialize_lm", fake_initialize_lm)

    response = await client.post(
        "/api/models/switch-lm",
        json={"model_name": "acestep-5Hz-lm-0.6B", "runtime": "nano-vllm"},
    )

    assert response.status_code == 200
    assert calls[0]["backend"] == "vllm"

    db = await get_db()
    async with db.execute("SELECT value FROM settings WHERE key = 'lm_backend'") as cursor:
        row = await cursor.fetchone()
    assert row["value"] == "vllm"


@pytest.mark.asyncio
async def test_switch_chat_llm_loads_and_persists_model(client, monkeypatch):
    from bangers.db.connection import get_db
    from bangers.routers import models as models_router
    from bangers.services.generation import generation_service

    calls: list[str] = []

    async def fake_switch(model_name: str) -> None:
        calls.append(model_name)
        assert generation_service.loading_state is not None
        assert generation_service.loading_state["kind"] == "chat_llm"
        assert generation_service.loading_state["model_name"] == model_name
        db = await get_db()
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) "
            "VALUES ('dj_model', ?, datetime('now'))",
            (model_name,),
        )
        await db.commit()

    monkeypatch.setattr(models_router, "switch_chat_model", fake_switch)

    response = await client.post(
        "/api/models/switch-chat-llm",
        json={"model_name": "Qwen3-1.7B"},
    )

    assert response.status_code == 200
    assert calls == ["Qwen3-1.7B"]
    assert generation_service.loading_state is None

    db = await get_db()
    cursor = await db.execute("SELECT value FROM settings WHERE key = 'dj_model'")
    row = await cursor.fetchone()
    assert row["value"] == "Qwen3-1.7B"


@pytest.mark.asyncio
async def test_chat_llm_active_status_uses_loaded_runtime_not_persisted_setting(client, monkeypatch):
    from bangers.config import settings
    from bangers.db.connection import get_db
    from bangers.routers import models as models_router

    chat_dir = settings.MODEL_CACHE_DIR / "chat-llm" / "Qwen3-1.7B"
    chat_dir.mkdir(parents=True)
    (chat_dir / "config.json").write_text("{}", encoding="utf-8")

    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) "
        "VALUES ('dj_model', 'Qwen3-1.7B', datetime('now'))"
    )
    await db.commit()

    monkeypatch.setattr(models_router, "get_loaded_chat_model_name", lambda: "")
    response = await client.get("/api/models")
    assert response.status_code == 200
    chat_models = response.json()["chat_llm_models"]
    assert chat_models[0]["name"] == "Qwen3-1.7B"
    assert chat_models[0]["is_active"] is False

    monkeypatch.setattr(models_router, "get_loaded_chat_model_name", lambda: "Qwen3-1.7B")
    response = await client.get("/api/models")
    assert response.status_code == 200
    chat_models = response.json()["chat_llm_models"]
    assert chat_models[0]["is_active"] is True


@pytest.mark.asyncio
async def test_models_marks_current_ace_load(client):
    from bangers.config import settings
    from bangers.services.generation import generation_service

    checkpoint_dir = settings.MODEL_CACHE_DIR / "checkpoints" / "acestep-v15-base"
    checkpoint_dir.mkdir(parents=True)
    generation_service._set_loading("dit", "acestep-v15-base")

    response = await client.get("/api/models")

    assert response.status_code == 200
    dit_models = response.json()["dit_models"]
    assert dit_models[0]["name"] == "acestep-v15-base"
    assert dit_models[0]["is_active"] is False
    assert dit_models[0]["is_loading"] is True
    assert dit_models[0]["loaded_on"] == []


@pytest.mark.asyncio
async def test_models_marks_current_ace_lm_load(client):
    from bangers.config import settings
    from bangers.services.generation import generation_service

    checkpoint_dir = settings.MODEL_CACHE_DIR / "checkpoints" / "acestep-5Hz-lm-0.6B"
    checkpoint_dir.mkdir(parents=True)
    generation_service._set_loading("lm", "acestep-5Hz-lm-0.6B")

    response = await client.get("/api/models")

    assert response.status_code == 200
    lm_models = response.json()["lm_models"]
    assert lm_models[0]["name"] == "acestep-5Hz-lm-0.6B"
    assert lm_models[0]["is_active"] is False
    assert lm_models[0]["is_loading"] is True
    assert lm_models[0]["loaded_on"] == []


@pytest.mark.asyncio
async def test_models_marks_current_chat_llm_load(client):
    from bangers.config import settings
    from bangers.services.generation import generation_service

    chat_dir = settings.MODEL_CACHE_DIR / "chat-llm" / "Qwen3-1.7B"
    chat_dir.mkdir(parents=True)
    (chat_dir / "config.json").write_text("{}", encoding="utf-8")
    generation_service._set_loading("chat_llm", "Qwen3-1.7B")

    response = await client.get("/api/models")

    assert response.status_code == 200
    chat_models = response.json()["chat_llm_models"]
    assert chat_models[0]["name"] == "Qwen3-1.7B"
    assert chat_models[0]["is_active"] is False
    assert chat_models[0]["is_loading"] is True
    assert chat_models[0]["loaded_on"] == []


@pytest.mark.asyncio
async def test_models_active_status_uses_distributed_workers(client, monkeypatch):
    from pathlib import Path

    from bangers.config import settings
    from bangers.routers import models as models_router
    from bangers.services.generation import generation_service

    checkpoints_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "checkpoints"
    chat_dir = Path(settings.ACESTEP_PROJECT_ROOT) / "chat-llm" / "Qwen3-4B-Instruct-2507"
    (checkpoints_dir / "acestep-v15-xl-turbo").mkdir(parents=True)
    (checkpoints_dir / "acestep-5Hz-lm-1.7B").mkdir(parents=True)
    (checkpoints_dir / "acestep-5Hz-lm-4B").mkdir(parents=True)
    chat_dir.mkdir(parents=True)
    (chat_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(settings, "DISTRIBUTED_ROLE", "coordinator")
    monkeypatch.setattr(settings, "DISTRIBUTED_WORKERS", ("http://worker",))
    generation_service.backend.active_lm_model = "acestep-5Hz-lm-4B"

    async def fake_distributed_active_models():
        return {
            "dit": {"acestep-v15-xl-turbo": {"spark-local-music"}},
            "lm": {"acestep-5Hz-lm-1.7B": {"spark-9bb0-lm-chat"}},
            "chat_llm": {"Qwen3-4B-Instruct-2507": {"spark-9bb0-lm-chat"}},
        }

    monkeypatch.setattr(
        models_router,
        "_distributed_active_models",
        fake_distributed_active_models,
    )
    monkeypatch.setattr(models_router, "get_loaded_chat_model_name", lambda: "")

    response = await client.get("/api/models")

    assert response.status_code == 200
    data = response.json()
    dit = {model["name"]: model for model in data["dit_models"]}
    lm = {model["name"]: model for model in data["lm_models"]}
    chat = {model["name"]: model for model in data["chat_llm_models"]}

    assert dit["acestep-v15-xl-turbo"]["is_active"] is True
    assert dit["acestep-v15-xl-turbo"]["is_loading"] is False
    assert dit["acestep-v15-xl-turbo"]["loaded_on"] == ["spark-local-music"]
    assert lm["acestep-5Hz-lm-1.7B"]["is_active"] is True
    assert lm["acestep-5Hz-lm-1.7B"]["is_loading"] is False
    assert lm["acestep-5Hz-lm-1.7B"]["loaded_on"] == ["spark-9bb0-lm-chat"]
    assert lm["acestep-5Hz-lm-4B"]["is_active"] is False
    assert chat["Qwen3-4B-Instruct-2507"]["is_active"] is True
    assert chat["Qwen3-4B-Instruct-2507"]["is_loading"] is False
    assert chat["Qwen3-4B-Instruct-2507"]["loaded_on"] == ["spark-9bb0-lm-chat"]
