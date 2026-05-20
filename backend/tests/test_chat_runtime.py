"""Tests for the per-model chat runtime selector and DJ 'no model' flow."""

import pytest


def test_chat_runtime_for_unknown_model_defaults_to_transformers():
    from bangers.model_registry import chat_runtime_for

    assert chat_runtime_for("some-random-folder-name") == "transformers"


def test_chat_runtime_for_mlx_model():
    from bangers.model_registry import chat_runtime_for

    # Qwen3-0.6B-4bit is registered with compatible_runtimes=("mlx",)
    assert chat_runtime_for("Qwen3-0.6B-4bit") == "mlx"


def test_chat_runtime_for_transformers_model():
    from bangers.model_registry import chat_runtime_for

    # Nemotron is registered with compatible_runtimes=()  -> Transformers fallback
    assert chat_runtime_for("Llama-3_3-Nemotron-Super-49B-v1_5-FP8") == "transformers"


def test_chat_runtime_for_empty_string_defaults_to_transformers():
    from bangers.model_registry import chat_runtime_for

    assert chat_runtime_for("") == "transformers"


def test_default_settings_seed_dj_model_empty():
    from bangers.db.schema import DEFAULT_SETTINGS

    assert DEFAULT_SETTINGS["dj_model"] == ""


def test_mlx_runtime_supported_does_not_import_mlx_lm(monkeypatch):
    import sys

    from bangers.services import llm_provider
    from bangers.services.llm_provider import MLXChatRuntime

    original = sys.modules.pop("mlx_lm", None)
    monkeypatch.setattr(llm_provider.sys, "platform", "darwin")
    monkeypatch.setattr(
        llm_provider.importlib.util,
        "find_spec",
        lambda name: object() if name == "mlx_lm" else None,
    )
    try:
        assert MLXChatRuntime.runtime_supported() is True
        assert "mlx_lm" not in sys.modules
    finally:
        if original is not None:
            sys.modules["mlx_lm"] = original


@pytest.mark.asyncio
async def test_mlx_chat_defers_while_music_gpu_lock_is_held(monkeypatch):
    from bangers.services.gpu_lock import gpu_lock
    from bangers.services.llm_provider import ChatRuntimeBusy, MLXChatRuntime

    runtime = MLXChatRuntime()
    monkeypatch.setattr(runtime, "_model_installed", lambda _model: True)

    await gpu_lock.await_acquire("generation")
    try:
        with pytest.raises(ChatRuntimeBusy):
            await runtime.chat([{"role": "user", "content": "title"}], "Qwen3-0.6B-4bit")
    finally:
        await gpu_lock.release("generation")


@pytest.mark.asyncio
async def test_mlx_chat_allow_holders_lets_lock_owner_through(monkeypatch):
    """Music services that hold the GPU lock for their job lifetime need to
    drive the chat LLM (titles, lyric specs) without the runtime treating
    them as a competing consumer."""
    from bangers.services.gpu_lock import gpu_lock
    from bangers.services.llm_provider import ChatRuntimeBusy, MLXChatRuntime

    runtime = MLXChatRuntime()
    monkeypatch.setattr(runtime, "_model_installed", lambda _model: True)
    # Skip the heavy model load + generation; we only care about the
    # holder-bypass logic.
    monkeypatch.setattr(runtime, "_generate_sync", lambda *_args, **_kwargs: "ok")

    await gpu_lock.await_acquire("radio")
    try:
        # Without the bypass: still busy.
        with pytest.raises(ChatRuntimeBusy):
            await runtime.chat([{"role": "user", "content": "x"}], "Qwen3-0.6B-4bit")
        # With the bypass for the current holder: passes through.
        result = await runtime.chat(
            [{"role": "user", "content": "x"}],
            "Qwen3-0.6B-4bit",
            allow_holders=frozenset({"radio"}),
        )
        assert result == "ok"
        # Bypass for a different holder is ignored.
        with pytest.raises(ChatRuntimeBusy):
            await runtime.chat(
                [{"role": "user", "content": "x"}],
                "Qwen3-0.6B-4bit",
                allow_holders=frozenset({"dj"}),
            )
    finally:
        await gpu_lock.release("radio")


@pytest.mark.asyncio
async def test_mlx_chat_claims_gpu_lock_when_idle(monkeypatch):
    from bangers.services.gpu_lock import gpu_lock
    from bangers.services.llm_provider import MLXChatRuntime

    runtime = MLXChatRuntime()
    monkeypatch.setattr(runtime, "_model_installed", lambda _model: True)

    holders_seen: list[str | None] = []

    def fake_generate(*_args, **_kwargs):
        holders_seen.append(gpu_lock.holder)
        return "ok"

    monkeypatch.setattr(runtime, "_generate_sync", fake_generate)

    result = await runtime.chat(
        [{"role": "user", "content": "x"}],
        "Qwen3-0.6B-4bit",
    )

    assert result == "ok"
    assert holders_seen == ["chat-llm"]
    assert gpu_lock.is_locked is False


@pytest.mark.asyncio
async def test_mlx_chat_uses_one_worker_thread(monkeypatch):
    import threading

    from bangers.services.llm_provider import MLXChatRuntime

    runtime = MLXChatRuntime()
    monkeypatch.setattr(runtime, "_model_installed", lambda _model: True)

    thread_ids: list[int] = []

    def fake_generate(*_args, **_kwargs):
        thread_ids.append(threading.get_ident())
        return "ok"

    monkeypatch.setattr(runtime, "_generate_sync", fake_generate)

    first = await runtime.chat(
        [{"role": "user", "content": "x"}],
        "Qwen3-0.6B-4bit",
    )
    second = await runtime.chat(
        [{"role": "user", "content": "y"}],
        "Qwen3-0.6B-4bit",
    )

    assert (first, second) == ("ok", "ok")
    assert len(thread_ids) == 2
    assert len(set(thread_ids)) == 1


def test_mlx_chat_rebinds_generation_stream_in_current_thread(monkeypatch):
    import sys
    import types

    from bangers.services.llm_provider import MLXChatRuntime

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.default_device = lambda: "gpu"
    mlx_core.new_stream = lambda device: ("stream", device)
    mlx_pkg.core = mlx_core

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    runtime = MLXChatRuntime()

    def fake_generate():
        return ""

    runtime._ensure_generation_stream_for_current_thread(fake_generate)

    assert fake_generate.__globals__["generation_stream"] == ("stream", "gpu")


@pytest.mark.asyncio
async def test_dj_send_message_returns_no_model_selected_when_unset(client):
    from bangers.db.connection import get_db

    db = await get_db()
    # Seeded default leaves dj_model = ""; make sure it's empty.
    await db.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) "
        "VALUES ('dj_model', '', datetime('now'))"
    )
    await db.commit()

    create_resp = await client.post("/api/dj/conversations", json={"title": "T"})
    conv_id = create_resp.json()["id"]

    resp = await client.post(
        f"/api/dj/conversations/{conv_id}/messages",
        json={"content": "play me something"},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "no_chat_model_selected"
    assert "Models page" in detail["message"]
    assert detail["missing"] == ["dj_model"]


@pytest.mark.asyncio
async def test_dj_info_shape_has_no_provider_field(client):
    resp = await client.get("/api/dj/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_model" in data
    assert "installed_models" in data
    assert "providers" not in data
    assert "active_provider" not in data
