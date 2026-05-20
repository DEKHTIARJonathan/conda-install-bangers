"""
AI DJ chat service — translates natural language into music generation.

Inspired by:
- clockworksquirrel/ace-step-apple-silicon (conversational AI DJ interface)
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from bangers.db.connection import get_db
from bangers.config import settings
from bangers.services.duration_settings import get_default_duration
from bangers.services.generation_request_builder import build_text_to_music_params
from bangers.services.lyrics_pipeline import (
    MAX_GENERATED_LYRICS_ATTEMPTS,
    LyricsPipelineError,
    LyricsRejectedError,
    coerce_instrumental,
)
from bangers.services import chat_llm
from bangers.services.llm_provider import installed_chat_models

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an AI DJ assistant for conda install bangers, a local music generation app.

Your job is to help users create music through natural conversation. When a user describes music they want to hear, you should:

1. Respond conversationally and enthusiastically about their music request
2. Include a JSON code block with generation parameters when you want to generate music

The JSON block should use this format:
```json
{{
  "caption": "A description of the music to generate",
  "lyrics": "",
  "instrumental": true,
  "bpm": 120,
  "keyscale": "",
  "timesignature": "4/4",
  "vocal_language": "en"
}}
```

Parameter guidelines:
- caption: Descriptive text about the music style, mood, instruments (be detailed and vivid)
- lyrics: Song lyrics with standard structure. For vocal tracks, write at least 2 verses and a chorus (12+ lines). Use [verse], [chorus], [bridge] tags. Example:
  "[verse]\\nFirst verse lines here\\n[chorus]\\nChorus lines here\\n[verse]\\nSecond verse lines here\\n[chorus]\\nChorus lines here"
- instrumental: true for no vocals, false for vocal tracks
- bpm: Beats per minute (60-200 typical range)
- keyscale: Musical key (e.g., "C major", "A minor", leave empty if unsure)
- timesignature: Time signature (e.g., "4/4", "3/4", "6/8")
- duration: Do not choose duration. The app Settings page controls song length.
- vocal_language: Always set to "en". Lyrics MUST be written in English; do not use any other language.

You can also handle these commands:
- "skip" or "next" → respond with: [ACTION:SKIP]
- "replay" or "play again" → respond with: [ACTION:REPLAY]
- "save" or "save to library" → respond with: [ACTION:SAVE]
- "create a radio station like this" → respond with: [ACTION:CREATE_STATION]

Be creative, knowledgeable about music genres, and enthusiastic. Keep responses concise but engaging."""


def _build_default_system_prompt(default_duration: float) -> str:
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        default_duration=int(default_duration)
    )


DEFAULT_SYSTEM_PROMPT = _build_default_system_prompt(settings.DEFAULT_DURATION)


async def _get_default_system_prompt() -> str:
    return _build_default_system_prompt(await get_default_duration())


def _extract_json_block(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON code block from LLM response text."""
    pattern = r"```json\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON block from DJ response")
    return None


def _extract_action(text: str) -> Optional[str]:
    """Extract an action command from the response."""
    pattern = r"\[ACTION:(\w+)\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1).lower()
    return None


class DJService:
    """Manages AI DJ conversations and music generation from chat."""

    async def list_conversations(self) -> list[dict[str, Any]]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM dj_conversations ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"] or "",
                "updated_at": row["updated_at"] or "",
            }
            for row in rows
        ]

    async def create_conversation(self, title: str = "New Conversation") -> dict[str, Any]:
        db = await get_db()
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        await db.execute(
            "INSERT INTO dj_conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        await db.commit()

        return {
            "id": conv_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    async def get_conversation(self, conv_id: str) -> Optional[dict[str, Any]]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM dj_conversations WHERE id = ?", (conv_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        msg_cursor = await db.execute(
            "SELECT * FROM dj_messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        )
        msg_rows = await msg_cursor.fetchall()

        messages = [
            {
                "id": m["id"],
                "conversation_id": m["conversation_id"],
                "role": m["role"],
                "content": m["content"],
                "generation_params_json": m["generation_params_json"],
                "generation_job_id": m["generation_job_id"],
                "created_at": m["created_at"] or "",
            }
            for m in msg_rows
        ]

        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"] or "",
            "updated_at": row["updated_at"] or "",
            "messages": messages,
        }

    async def delete_conversation(self, conv_id: str) -> bool:
        db = await get_db()
        cursor = await db.execute(
            "SELECT id FROM dj_conversations WHERE id = ?", (conv_id,)
        )
        if await cursor.fetchone() is None:
            return False

        await db.execute("DELETE FROM dj_conversations WHERE id = ?", (conv_id,))
        await db.commit()
        return True

    async def send_message(
        self, conv_id: str, user_content: str, client_id: str | None = None
    ) -> dict[str, Any]:
        """Process a user message through the LLM and optionally trigger generation."""
        db = await get_db()

        # Verify conversation exists
        cursor = await db.execute(
            "SELECT id FROM dj_conversations WHERE id = ?", (conv_id,)
        )
        if await cursor.fetchone() is None:
            return {"error": "Conversation not found"}

        # Save user message
        user_msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """INSERT INTO dj_messages (id, conversation_id, role, content, created_at)
               VALUES (?, ?, 'user', ?, ?)""",
            (user_msg_id, conv_id, user_content, now),
        )

        # Load conversation history
        msg_cursor = await db.execute(
            """SELECT role, content FROM dj_messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (conv_id,),
        )
        msg_rows = await msg_cursor.fetchall()

        # Read custom system prompt from settings
        prompt_cursor = await db.execute(
            "SELECT value FROM settings WHERE key = 'dj_system_prompt'"
        )
        prompt_row = await prompt_cursor.fetchone()
        default_system_prompt = await _get_default_system_prompt()
        system_prompt = prompt_row["value"] if prompt_row else default_system_prompt

        # Build messages for LLM
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for m in msg_rows:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        model_name = await chat_llm.get_configured_chat_model_name()

        if not model_name:
            return {
                "error": {
                    "error": "no_chat_model_selected",
                    "message": "No chat model selected. Pick one on the Models page.",
                    "missing": ["dj_model"],
                },
            }

        try:
            runtime, model_name = await chat_llm.require_configured_chat_runtime()
        except chat_llm.ChatLlmUnavailable:
            return {
                "error": {
                    "error": "chat_model_not_installed",
                    "message": (
                        f"Chat model '{model_name}' is selected but its files are not on disk. "
                        "Download it on the Models page."
                    ),
                    "missing": ["dj_model"],
                },
            }

        fallback_notice: str | None = None

        response_text = ""
        action = None
        queued_gen_params = None

        last_pipeline_error: LyricsPipelineError | None = None
        for attempt in range(1, MAX_GENERATED_LYRICS_ATTEMPTS + 1):
            attempt_messages = llm_messages
            if last_pipeline_error is not None:
                retry_system = (
                    f"{system_prompt}\n\nFresh generation attempt {attempt} of "
                    f"{MAX_GENERATED_LYRICS_ATTEMPTS}: the previous lyrics violated "
                    "the platform Code of Conduct. Generate a completely new lyrical "
                    "premise, new wording, and new imagery while preserving only the "
                    "safe musical intent."
                )
                attempt_messages = [
                    {"role": "system", "content": retry_system},
                    *llm_messages[1:],
                    {
                        "role": "user",
                        "content": (
                            "The previous generated lyrics failed the platform lyrics "
                            f"guardrails: {last_pipeline_error}. Generate a new, complete, "
                            "compliant set of lyrics and do not reuse the rejected lyrics."
                        ),
                    },
                ]

            try:
                response_text = await runtime.chat(attempt_messages, model_name)
            except Exception as e:
                logger.exception("DJ LLM call failed")
                return {
                    "error": {
                        "error": "chat_runtime_error",
                        "message": f"LLM error: {e}",
                    },
                }

            # Strip <think>...</think> blocks from models like Qwen3
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            ).strip()

            # Parse response for generation params or actions
            gen_params = _extract_json_block(response_text)
            action = _extract_action(response_text)
            queued_gen_params = None
            if not gen_params:
                break

            default_duration = await get_default_duration()
            gen_params["duration"] = default_duration
            try:
                from bangers.services.generation import generation_service

                gen_params = await generation_service.prepare_params(gen_params)
                queued_gen_params = generation_service.public_params(gen_params)
                break
            except LyricsRejectedError as exc:
                last_pipeline_error = exc
                logger.warning(
                    "DJ lyrics violated guardrails on attempt {}/{}: {}",
                    attempt,
                    MAX_GENERATED_LYRICS_ATTEMPTS,
                    exc,
                )
                if attempt == MAX_GENERATED_LYRICS_ATTEMPTS:
                    return {
                        "error": {
                            "error": "lyrics_pipeline_error",
                            "message": (
                                "Generated lyrics violated the Code of Conduct "
                                f"{MAX_GENERATED_LYRICS_ATTEMPTS} times. Generation stopped."
                            ),
                        },
                    }
            except LyricsPipelineError as exc:
                last_pipeline_error = exc
                logger.warning(
                    "DJ lyrics failed guardrails on attempt {}/{}: {}",
                    attempt,
                    MAX_GENERATED_LYRICS_ATTEMPTS,
                    exc,
                )
                if attempt == MAX_GENERATED_LYRICS_ATTEMPTS:
                    return {
                        "error": {
                            "error": "lyrics_pipeline_error",
                            "message": str(exc),
                        },
                    }
            except Exception as exc:
                return {
                    "error": {
                        "error": "lyrics_pipeline_error",
                        "message": str(exc),
                    },
                }

        # Clean action tags and JSON code blocks from the display text
        display_text = re.sub(r"\[ACTION:\w+\]", "", response_text)
        display_text = re.sub(r"```json\s*\n?.*?\n?\s*```", "", display_text, flags=re.DOTALL)
        display_text = display_text.strip()

        # Save assistant message
        assistant_msg_id = str(uuid.uuid4())
        assistant_now = datetime.now(timezone.utc).isoformat()
        gen_params_json = json.dumps(queued_gen_params) if queued_gen_params else None
        generation_job_id = None

        # Generate conversation title BEFORE firing generation task to avoid
        # concurrent MLX usage (chat model + ACE-Step LM) which causes a segfault
        title_cursor = await db.execute(
            "SELECT title FROM dj_conversations WHERE id = ?", (conv_id,)
        )
        title_row = await title_cursor.fetchone()
        auto_title = None
        if title_row and title_row["title"] == "New Conversation":
            from bangers.services.title_generator import generate_conversation_title
            auto_title = await generate_conversation_title(user_content)

        # Trigger generation if params were provided
        if queued_gen_params:
            try:
                from bangers.services.generation import generation_service

                if generation_service.backend_ready or settings.delegates_to_workers:
                    if client_id:
                        job_id = generation_service.create_job(client_id=client_id)
                    else:
                        job_id = generation_service.create_job()
                    generation_job_id = job_id

                    # Fire and forget generation
                    import asyncio
                    asyncio.create_task(
                        self._run_dj_generation(job_id, queued_gen_params, model_name)
                    )
                else:
                    fallback_notice = (
                        "ACE-Step backend is not ready for DJ generation."
                    )
            except Exception as e:
                logger.warning(f"Failed to start DJ generation: {e}")

        await db.execute(
            """INSERT INTO dj_messages
               (id, conversation_id, role, content, generation_params_json, generation_job_id, created_at)
               VALUES (?, ?, 'assistant', ?, ?, ?, ?)""",
            (assistant_msg_id, conv_id, display_text, gen_params_json, generation_job_id, assistant_now),
        )

        # Apply the pre-generated title or just update timestamp
        if auto_title is not None:
            await db.execute(
                "UPDATE dj_conversations SET title = ?, updated_at = ? WHERE id = ?",
                (auto_title, assistant_now, conv_id),
            )
        else:
            await db.execute(
                "UPDATE dj_conversations SET updated_at = ? WHERE id = ?",
                (assistant_now, conv_id),
            )
        await db.commit()

        return {
            "message": {
                "id": assistant_msg_id,
                "conversation_id": conv_id,
                "role": "assistant",
                "content": display_text,
                "generation_params_json": gen_params_json,
                "generation_job_id": generation_job_id,
                "created_at": assistant_now,
            },
            "action": action,
            "generation_job_id": generation_job_id,
            "fallback_notice": fallback_notice,
        }

    async def _run_dj_generation(
        self, job_id: str, params: dict[str, Any],
        dj_model: str = "",
    ) -> None:
        """Run generation triggered by DJ chat."""
        from bangers.services.generation import generation_service
        from bangers.services.gpu_lock import gpu_lock
        from bangers.ws.manager import generation_ws_manager

        use_local_gpu_lock = not settings.delegates_to_workers

        def _client_id() -> str | None:
            get_job = getattr(generation_service, "get_job", None)
            if not callable(get_job):
                return None
            job = get_job(job_id) or {}
            value = str(job.get("client_id") or "").strip()
            return value or None

        async def _broadcast(message: dict[str, Any]) -> None:
            client_id = _client_id()
            if client_id:
                await generation_ws_manager.broadcast(message, client_id=client_id)
            else:
                await generation_ws_manager.broadcast(message)

        if use_local_gpu_lock and gpu_lock.is_locked:
            await _broadcast({
                "type": "progress",
                "job_id": job_id,
                "progress": 0.0,
                "stage": f"Waiting for GPU (in use by {gpu_lock.holder})...",
            })
        if use_local_gpu_lock:
            await gpu_lock.await_acquire("dj")

        started_at = datetime.now(timezone.utc)
        history_id = job_id  # Same ID so DJ messages can link to history
        deferred_title_caption: str | None = None

        try:
            generation_service.update_job(job_id, status="running", stage="preparing")

            prompt = str(params.get("caption") or "").strip()
            if not prompt:
                prompt = "A music track"
            raw_lyrics = str(params.get("lyrics") or "")
            default_duration = await get_default_duration()
            params = build_text_to_music_params(
                prompt,
                lyrics=raw_lyrics,
                instrumental=coerce_instrumental(params.get("instrumental", not raw_lyrics.strip())),
                vocal_language="en",
                duration=float(params.get("duration") or default_duration),
                batch_size=1,
                extra=params,
            )

            if generation_service.active_dit_model:
                params["dit_model"] = generation_service.active_dit_model
            if generation_service.active_lm_model:
                params["lm_model"] = generation_service.active_lm_model
            if dj_model:
                params["dj_model"] = dj_model

            if hasattr(generation_service, "prepare_params"):
                allow_holders = frozenset({"dj"}) if use_local_gpu_lock else None
                params = await generation_service.prepare_params(
                    params,
                    allow_holders=allow_holders,
                )

            # Insert generation history record
            db = await get_db()
            history_params = generation_service.public_params(params)
            await db.execute(
                """INSERT INTO generation_history (id, task_type, status, params_json, started_at, created_at)
                   VALUES (?, ?, 'running', ?, ?, ?)""",
                (history_id, params.get("task_type", "text2music"),
                 json.dumps(history_params), started_at.isoformat(), started_at.isoformat()),
            )
            await db.commit()

            result = await generation_service.generate(params, lyrics_prepared=True)

            if result.get("success"):
                audios = result.get("audios", [])
                results = [
                    {
                        "path": a.get("path", ""),
                        "key": a.get("key", ""),
                        "sample_rate": a.get("sample_rate", 48000),
                        "params": {
                            k: v
                            for k, v in (a.get("params") or {}).items()
                            if k != "tensor" and not k.startswith("_")
                        },
                    }
                    for a in audios
                ]

                title_caption = params.get("caption") or None
                generated_title: str | None = None
                if title_caption:
                    try:
                        from bangers.services.title_generator import generate_song_title
                        allow_holders = frozenset({"dj"}) if use_local_gpu_lock else None
                        generated_title = await generate_song_title(
                            title_caption, "", "", "DJ Generation",
                            allow_holders=allow_holders,
                        )
                    except Exception as e:
                        logger.debug(f"DJ title generation failed: {e}")

                generation_service.update_job(
                    job_id, status="completed", progress=1.0, results=results
                )

                # Update history as completed (title included if generated).
                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                db = await get_db()
                if generated_title is not None:
                    await db.execute(
                        """UPDATE generation_history
                           SET status = 'completed', result_json = ?, audio_count = ?,
                               completed_at = ?, duration_ms = ?, title = ?
                           WHERE id = ?""",
                        (json.dumps(results), len(results),
                         completed_at.isoformat(), duration_ms,
                         generated_title, history_id),
                    )
                else:
                    await db.execute(
                        """UPDATE generation_history
                           SET status = 'completed', result_json = ?, audio_count = ?,
                               completed_at = ?, duration_ms = ?
                           WHERE id = ?""",
                        (json.dumps(results), len(results),
                         completed_at.isoformat(), duration_ms, history_id),
                    )
                await db.commit()

                if generated_title is not None:
                    await _broadcast({
                        "type": "title",
                        "job_id": job_id,
                        "history_id": history_id,
                        "title": generated_title,
                    })

                await _broadcast({
                    "type": "completed",
                    "job_id": job_id,
                    "history_id": history_id,
                    "results": results,
                })
            else:
                error = result.get("error", "Generation failed")
                generation_service.update_job(job_id, status="failed", error=error)
                await _broadcast({
                    "type": "failed",
                    "job_id": job_id,
                    "error": error,
                })

                db = await get_db()
                await db.execute(
                    """UPDATE generation_history SET status = 'failed', error_message = ? WHERE id = ?""",
                    (error, history_id),
                )
                await db.commit()
        except Exception as e:
            logger.exception(f"DJ generation {job_id} failed")
            generation_service.update_job(job_id, status="failed", error=str(e))
            await _broadcast({
                "type": "failed",
                "job_id": job_id,
                "error": str(e),
            })

            try:
                db = await get_db()
                await db.execute(
                    """UPDATE generation_history SET status = 'failed', error_message = ? WHERE id = ?""",
                    (str(e), history_id),
                )
                await db.commit()
            except Exception:
                pass
        finally:
            if use_local_gpu_lock:
                await gpu_lock.release("dj")

        if deferred_title_caption:
            from bangers.services.deferred_titles import schedule_history_title_retry

            schedule_history_title_retry(
                job_id=job_id,
                history_id=history_id,
                caption=deferred_title_caption,
                fallback="DJ Generation",
            )

    async def get_dj_info(self) -> dict[str, Any]:
        """Return current DJ chat-LLM state.

        The user picks a chat model on the Models page; this endpoint
        reports which one is currently selected, what other models are
        installed, and the system prompt customization (if any).
        """
        db = await get_db()
        cursor = await db.execute(
            "SELECT key, value FROM settings WHERE key IN ('dj_model', 'dj_system_prompt')"
        )
        rows = await cursor.fetchall()
        active_model = ""
        custom_system_prompt = ""
        for r in rows:
            if r["key"] == "dj_model":
                active_model = r["value"]
            elif r["key"] == "dj_system_prompt":
                custom_system_prompt = r["value"]

        return {
            "active_model": active_model,
            "loaded_model": chat_llm.get_loaded_chat_model_name(),
            "installed_models": installed_chat_models(),
            "system_prompt": custom_system_prompt,
            "default_system_prompt": await _get_default_system_prompt(),
        }

    async def update_settings(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        db = await get_db()
        if model is not None:
            await chat_llm.switch_chat_model(model)
        if system_prompt is not None:
            default_system_prompt = await _get_default_system_prompt()
            if system_prompt.strip() == "" or system_prompt.strip() == default_system_prompt.strip():
                await db.execute(
                    "DELETE FROM settings WHERE key = 'dj_system_prompt'"
                )
            else:
                await db.execute(
                    "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES ('dj_system_prompt', ?, datetime('now'))",
                    (system_prompt,),
                )
        await db.commit()
        return await self.get_dj_info()

    async def rename_conversation(self, conv_id: str, title: str) -> bool:
        db = await get_db()
        cursor = await db.execute(
            "SELECT id FROM dj_conversations WHERE id = ?", (conv_id,)
        )
        if await cursor.fetchone() is None:
            return False
        await db.execute(
            "UPDATE dj_conversations SET title = ? WHERE id = ?",
            (title, conv_id),
        )
        await db.commit()
        return True


dj_service = DJService()
