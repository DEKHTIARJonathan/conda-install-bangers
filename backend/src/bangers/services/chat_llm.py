from __future__ import annotations

from loguru import logger

from bangers.config import DISTRIBUTED_CAPABILITY_CHAT_LLM, settings
from bangers.db.connection import get_db
from bangers.services.llm_provider import ChatRuntime, get_chat_runtime, loaded_chat_model_name


CHAT_LLM_SETTING_KEY = "dj_model"


class ChatLlmUnavailable(RuntimeError):
    """Raised when the configured app Chat LLM cannot be used."""


async def get_configured_chat_model_name() -> str:
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT value FROM settings WHERE key = ?",
            (CHAT_LLM_SETTING_KEY,),
        )
        row = await cursor.fetchone()
        return row["value"] if row else ""
    except Exception as exc:
        logger.debug(f"Failed to read configured Chat LLM: {exc}")
        return ""


async def get_configured_chat_runtime() -> tuple[ChatRuntime | None, str]:
    model_name = await get_configured_chat_model_name()
    if not model_name:
        return (None, "")
    if (
        settings.delegates_to_workers
        and not settings.has_distributed_capability(DISTRIBUTED_CAPABILITY_CHAT_LLM)
    ):
        from bangers.services.distributed import remote_chat_runtime

        return (remote_chat_runtime, model_name)
    runtime = get_chat_runtime(model_name)
    if runtime is None or not runtime.is_model_loadable(model_name):
        if settings.delegates_to_workers:
            from bangers.services.distributed import remote_chat_runtime

            return (remote_chat_runtime, model_name)
        return (None, model_name)
    return (runtime, model_name)


def get_loaded_chat_model_name() -> str:
    return loaded_chat_model_name()


async def require_configured_chat_runtime() -> tuple[ChatRuntime, str]:
    runtime, model_name = await get_configured_chat_runtime()
    if runtime is None:
        if model_name:
            raise ChatLlmUnavailable(
                f"Configured Chat LLM '{model_name}' is not installed or cannot run on this machine."
            )
        raise ChatLlmUnavailable("No Chat LLM is loaded. Load one on the Models page.")
    return runtime, model_name


async def chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    allow_holders: frozenset[str] | None = None,
) -> str:
    runtime, model_name = await require_configured_chat_runtime()
    return await runtime.chat(
        messages,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        allow_holders=allow_holders,
    )


async def warm_chat_model(
    model_name: str,
    *,
    allow_holders: frozenset[str] | None = None,
) -> None:
    warmup_messages = [
        {
            "role": "system",
            "content": "You are a model warmup probe. Reply with OK only.",
        },
        {"role": "user", "content": "OK"},
    ]
    if (
        settings.delegates_to_workers
        and not settings.has_distributed_capability(DISTRIBUTED_CAPABILITY_CHAT_LLM)
    ):
        from bangers.services.distributed import remote_chat_runtime

        await remote_chat_runtime.chat(
            warmup_messages,
            model=model_name,
            max_tokens=4,
            temperature=0.0,
        )
        return
    runtime = get_chat_runtime(model_name)
    if runtime is None:
        if settings.delegates_to_workers:
            from bangers.services.distributed import remote_chat_runtime

            await remote_chat_runtime.chat(
                warmup_messages,
                model=model_name,
                max_tokens=4,
                temperature=0.0,
            )
            return
        raise ChatLlmUnavailable(
            f"Chat LLM '{model_name}' requires a runtime that is not available on this machine."
        )
    if not runtime.is_model_loadable(model_name):
        if settings.delegates_to_workers:
            from bangers.services.distributed import remote_chat_runtime

            await remote_chat_runtime.chat(
                warmup_messages,
                model=model_name,
                max_tokens=4,
                temperature=0.0,
            )
            return
        raise ChatLlmUnavailable(
            f"Chat LLM '{model_name}' is not installed. Download it before loading."
        )
    await runtime.chat(
        warmup_messages,
        model=model_name,
        max_tokens=4,
        temperature=0.0,
        allow_holders=allow_holders,
    )


async def persist_chat_model(model_name: str) -> None:
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))",
        (CHAT_LLM_SETTING_KEY, model_name),
    )
    await db.commit()


async def switch_chat_model(model_name: str) -> None:
    await warm_chat_model(model_name)
    await persist_chat_model(model_name)
