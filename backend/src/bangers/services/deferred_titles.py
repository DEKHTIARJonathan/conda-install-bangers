import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from bangers.db.connection import get_db
from bangers.services.gpu_lock import gpu_lock
from bangers.services.llm_provider import ChatRuntimeBusy
from bangers.services.title_generator import generate_random_title, generate_song_title
from bangers.ws.manager import generation_ws_manager

_RETRY_DELAYS_SECONDS = (3.0, 8.0, 20.0, 45.0, 90.0)


async def _run_when_music_idle(
    label: str,
    holder: str,
    work: Callable[[], Awaitable[str]],
) -> str | None:
    for attempt, delay in enumerate(_RETRY_DELAYS_SECONDS, start=1):
        await asyncio.sleep(delay)
        if gpu_lock.is_locked:
            logger.info(
                f"{label}: GPU busy with {gpu_lock.holder}; retrying later "
                f"({attempt}/{len(_RETRY_DELAYS_SECONDS)})"
            )
            continue

        await gpu_lock.await_acquire(holder)
        try:
            return await work()
        except ChatRuntimeBusy as exc:
            logger.info(
                f"{label}: helper LLM busy; retrying later "
                f"({attempt}/{len(_RETRY_DELAYS_SECONDS)}): {exc}"
            )
        except Exception as exc:
            logger.warning(f"{label}: deferred title generation failed: {exc}")
            return None
        finally:
            await gpu_lock.release(holder)

    logger.info(f"{label}: title retry budget exhausted")
    return None


def schedule_history_title_retry(
    *,
    job_id: str,
    history_id: str,
    caption: str,
    fallback: str,
) -> None:
    async def _runner() -> None:
        title = await _run_when_music_idle(
            "history title",
            "title-retry",
            lambda: generate_song_title(
                caption,
                "",
                "",
                fallback,
                allow_holders=frozenset({"title-retry"}),
            ),
        )
        if not title:
            return
        try:
            db = await get_db()
            await db.execute(
                "UPDATE generation_history SET title = ? WHERE id = ?",
                (title, history_id),
            )
            await db.commit()
            from bangers.services.generation import generation_service

            job = generation_service.get_job(job_id) or {}
            client_id = str(job.get("client_id") or "").strip() or None
            await generation_ws_manager.broadcast({
                "type": "title",
                "job_id": job_id,
                "history_id": history_id,
                "title": title,
            }, client_id=client_id)
        except Exception as exc:
            logger.warning(f"history title: failed to persist deferred title: {exc}")

    asyncio.create_task(_runner())


def schedule_radio_title_retry(
    *,
    song_id: str,
    history_id: str,
    avoid_titles: list[str] | None = None,
) -> None:
    async def _runner() -> None:
        title = await _run_when_music_idle(
            "radio title",
            "radio-title-retry",
            lambda: generate_random_title(
                avoid_titles=avoid_titles,
                allow_holders=frozenset({"radio-title-retry"}),
            ),
        )
        if not title:
            return
        try:
            db = await get_db()
            await db.execute(
                "UPDATE generation_history SET title = ? WHERE id = ?",
                (title, history_id),
            )
            await db.execute(
                "UPDATE songs SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, song_id),
            )
            await db.commit()
        except Exception as exc:
            logger.warning(f"radio title: failed to persist deferred title: {exc}")

    asyncio.create_task(_runner())
