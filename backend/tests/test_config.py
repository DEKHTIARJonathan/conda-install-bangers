import pytest


def _reset_settings(monkeypatch):
    for name in [
        "BANGERS_BATCH_SIZE",
        "BANGERS_LM_BACKEND",
        "BANGERS_THINKING",
        "BANGERS_MODEL_CACHE_DIR",
        "BANGERS_DISTRIBUTED_ROLE",
        "BANGERS_NODE_ID",
        "BANGERS_WORKERS",
        "BANGERS_WORKER_CAPABILITIES",
        "BANGERS_WORKER_TOKEN",
        "BANGERS_WORKER_TIMEOUT_SECONDS",
        "ACESTEP_PROJECT_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
    ]:
        monkeypatch.delenv(name, raising=False)

    from bangers.config import settings

    settings.apply_runtime_overrides()


def test_default_settings_have_empty_models():
    from bangers.db.schema import DEFAULT_SETTINGS

    assert DEFAULT_SETTINGS["dit_model"] == ""
    assert DEFAULT_SETTINGS["lm_model"] == ""
    assert "config_profile" not in DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["keep_active_models_resident"] == "true"
    assert DEFAULT_SETTINGS["parallel_pipeline_enabled"] == "false"
    assert DEFAULT_SETTINGS["lyrics_guardrails_enabled"] == "true"


def test_settings_no_profile_attribute(monkeypatch):
    import inspect
    from bangers.config import Settings, settings

    try:
        _reset_settings(monkeypatch)

        assert not hasattr(settings, "CONFIG_PROFILE")
        sig = inspect.signature(Settings.apply_runtime_overrides)
        assert list(sig.parameters) == ["self"]
        # No env vars / settings.* fields exist for model selection - that
        # lives exclusively in the DB rows written by the Models page UI.
        assert not hasattr(settings, "DEFAULT_DIT_MODEL")
        assert not hasattr(settings, "DEFAULT_LM_MODEL")
        assert settings.startup_setting_overrides() == {}
    finally:
        _reset_settings(monkeypatch)


def test_db_default_overrides_does_not_seed_models(monkeypatch):
    """Fresh DBs must not auto-seed dit_model or lm_model."""
    from bangers.config import settings

    try:
        _reset_settings(monkeypatch)

        defaults = settings.db_default_overrides()
        assert "dit_model" not in defaults
        assert "lm_model" not in defaults
        # Other generation defaults still present:
        assert "batch_size" in defaults
        assert "inference_steps" in defaults
        assert defaults["keep_active_models_resident"] == "true"
        assert defaults["parallel_pipeline_enabled"] == "false"
        assert defaults["lyrics_guardrails_enabled"] == "true"
    finally:
        _reset_settings(monkeypatch)


def test_lm_backend_defaults_to_vllm_alias_on_linux(monkeypatch):
    import sys

    from bangers.config import normalize_lm_backend, settings

    try:
        _reset_settings(monkeypatch)

        assert normalize_lm_backend("nano-vllm") == "vllm"
        assert normalize_lm_backend("nano_vllm") == "vllm"
        assert normalize_lm_backend("nanovllm") == "vllm"
        assert normalize_lm_backend("VLLM") == "vllm"
        if sys.platform != "darwin":
            assert settings.DEFAULT_LM_BACKEND == "vllm"
            assert normalize_lm_backend("mlx") == "vllm"

        monkeypatch.setenv("BANGERS_LM_BACKEND", "nano-vllm")
        settings.apply_runtime_overrides()
        assert settings.DEFAULT_LM_BACKEND == "vllm"
        assert settings.startup_setting_overrides()["lm_backend"] == "vllm"
    finally:
        _reset_settings(monkeypatch)


def test_env_vars_cannot_set_models(monkeypatch):
    """BANGERS_DIT_MODEL / BANGERS_LM_MODEL must NOT influence settings."""
    from bangers.config import settings

    try:
        monkeypatch.setenv("BANGERS_DIT_MODEL", "foo-dit")
        monkeypatch.setenv("BANGERS_LM_MODEL", "foo-lm")
        settings.apply_runtime_overrides()

        # Env vars are intentionally ignored - the Models page UI is the
        # single source of truth for which models load.
        overrides = settings.startup_setting_overrides()
        assert "dit_model" not in overrides
        assert "lm_model" not in overrides

        defaults = settings.db_default_overrides()
        assert "dit_model" not in defaults
        assert "lm_model" not in defaults
    finally:
        monkeypatch.delenv("BANGERS_DIT_MODEL", raising=False)
        monkeypatch.delenv("BANGERS_LM_MODEL", raising=False)
        _reset_settings(monkeypatch)


def test_model_cache_dir_sets_acestep_and_hf_cache(monkeypatch, tmp_path):
    from bangers.config import settings

    try:
        cache_dir = tmp_path / "models"
        monkeypatch.delenv("ACESTEP_PROJECT_ROOT", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("BANGERS_MODEL_CACHE_DIR", str(cache_dir))
        settings.apply_runtime_overrides()
        settings.ensure_dirs()

        assert settings.MODEL_CACHE_DIR == cache_dir
        assert settings.ACESTEP_PROJECT_ROOT == str(cache_dir)
        assert (cache_dir / "checkpoints").is_dir()
        assert (cache_dir / "chat-llm").is_dir()
        assert settings.HF_HUB_CACHE_DIR == cache_dir / "huggingface" / "hub"
        assert settings.HF_HUB_CACHE_DIR.is_dir()
    finally:
        _reset_settings(monkeypatch)


def test_default_duration_is_seeded_from_backend_default(monkeypatch):
    from bangers.config import settings
    from bangers.config import DEFAULT_GENERATION_DURATION

    try:
        settings.apply_runtime_overrides()

        assert settings.DEFAULT_DURATION == DEFAULT_GENERATION_DURATION
        assert "default_duration" not in settings.startup_setting_overrides()
        assert settings.db_default_overrides()["default_duration"] == str(DEFAULT_GENERATION_DURATION)
    finally:
        _reset_settings(monkeypatch)


def test_distributed_settings_parse_roles_workers_and_capabilities(monkeypatch):
    from bangers.config import (
        DISTRIBUTED_CAPABILITY_CHAT_LLM,
        DISTRIBUTED_CAPABILITY_MUSIC,
        DISTRIBUTED_CAPABILITIES,
        settings,
    )

    try:
        _reset_settings(monkeypatch)
        assert settings.DISTRIBUTED_ROLE == "standalone"
        assert settings.DISTRIBUTED_CAPABILITIES == DISTRIBUTED_CAPABILITIES
        assert settings.delegates_to_workers is False

        monkeypatch.setenv("BANGERS_DISTRIBUTED_ROLE", "coordinator")
        monkeypatch.setenv("BANGERS_WORKERS", "http://spark-a:8000, http://spark-b:8000")
        settings.apply_runtime_overrides()
        assert settings.DISTRIBUTED_ROLE == "coordinator"
        assert settings.DISTRIBUTED_WORKERS == (
            "http://spark-a:8000",
            "http://spark-b:8000",
        )
        assert settings.DISTRIBUTED_CAPABILITIES == frozenset()
        assert settings.delegates_to_workers is True

        monkeypatch.setenv("BANGERS_DISTRIBUTED_ROLE", "worker")
        monkeypatch.setenv("BANGERS_WORKER_CAPABILITIES", "music,chat_llm,unknown")
        settings.apply_runtime_overrides()
        assert settings.DISTRIBUTED_ROLE == "worker"
        assert settings.DISTRIBUTED_CAPABILITIES == frozenset({
            DISTRIBUTED_CAPABILITY_MUSIC,
            DISTRIBUTED_CAPABILITY_CHAT_LLM,
        })
    finally:
        _reset_settings(monkeypatch)


@pytest.mark.asyncio
async def test_health_treats_disabled_lm_as_ready(client):
    from bangers.backends.ace_step_backend import AceStepBackend
    from bangers.services.generation import generation_service

    ace = AceStepBackend()
    ace.dit_initialized = True
    ace.active_dit_model = "acestep-v15-turbo"
    ace.active_lm_model = "none"
    ace.lm_disabled = True
    generation_service.backend = ace

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["dit_model_loaded"] is True
    assert data["lm_model_loaded"] is True
    assert data["lm_model"] == "none"
