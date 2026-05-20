"""
Radio station auto-generation service.

Inspired by:
- nalexand/ACE-Step-1.5-OPTIMIZED (MusicBox jukebox)
- PasiKoodaa/ACE-Step-RADIO (radio station mode with memory optimization)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from bangers.config import settings
from bangers.db.connection import get_db
from bangers.models.radio import StationResponse
from bangers.services.duration_settings import get_default_duration
from bangers.services.generation import generation_service
from bangers.services.generation_request_builder import build_text_to_music_params
from bangers.services.gpu_lock import gpu_lock
from bangers.services import chat_llm
from bangers.services.llm_provider import ChatRuntime, installed_chat_models
from bangers.services.music_specs import build_music_spec

RADIO_DEFAULT_SYSTEM_PROMPT = """You are a music caption generator. Given station parameters, write a creative, \
detailed caption for an AI music generator. Describe instrumentation, texture, \
atmosphere, and sonic qualities. Be specific and varied — each caption should \
feel unique. Output ONLY the caption text, nothing else."""


def _row_to_station(row) -> StationResponse:
    return StationResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"] or "",
        is_preset=bool(row["is_preset"]),
        caption_template=row["caption_template"] or "",
        genre=row["genre"] or "",
        mood=row["mood"] or "",
        instrumental=bool(row["instrumental"]),
        vocal_language=row["vocal_language"] or "unknown",
        bpm_min=row["bpm_min"],
        bpm_max=row["bpm_max"],
        keyscale=row["keyscale"] or "",
        timesignature=row["timesignature"] or "",
        advanced_params_json=row["advanced_params_json"] or "{}",
        total_plays=row["total_plays"] or 0,
        last_played_at=row["last_played_at"],
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
    )


class RadioService:
    """Manages radio stations and auto-generation of tracks."""

    async def list_stations(self) -> list[StationResponse]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM radio_stations ORDER BY is_preset DESC, name ASC"
        )
        rows = await cursor.fetchall()
        return [_row_to_station(row) for row in rows]

    async def get_station(self, station_id: str) -> Optional[StationResponse]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM radio_stations WHERE id = ?", (station_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_station(row)

    async def create_station(self, data: dict[str, Any]) -> StationResponse:
        db = await get_db()
        station_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        default_duration = await get_default_duration()

        await db.execute(
            """INSERT INTO radio_stations (
                id, name, description, is_preset, caption_template,
                genre, mood, instrumental, vocal_language,
                bpm_min, bpm_max, keyscale, timesignature,
                duration_min, duration_max, advanced_params_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                station_id,
                data.get("name", "Custom Station"),
                data.get("description", ""),
                data.get("caption_template", ""),
                data.get("genre", ""),
                data.get("mood", ""),
                1 if data.get("instrumental", True) else 0,
                "en",
                data.get("bpm_min"),
                data.get("bpm_max"),
                data.get("keyscale", ""),
                data.get("timesignature", ""),
                default_duration,
                default_duration,
                data.get("advanced_params_json", "{}"),
                now,
                now,
            ),
        )
        await db.commit()

        station = await self.get_station(station_id)
        assert station is not None
        return station

    async def update_station(
        self, station_id: str, updates: dict[str, Any]
    ) -> Optional[StationResponse]:
        db = await get_db()

        # Check station exists and is not a preset (presets can't be edited)
        station = await self.get_station(station_id)
        if station is None:
            return None

        if not updates:
            return station

        # Presets are mostly read-only, but their lyrics toggle maps to the
        # existing instrumental bit and is allowed to persist.
        if station.is_preset:
            updates = {
                k: v
                for k, v in updates.items()
                if k == "instrumental"
            }
            if not updates:
                return station

        # Convert bool to int for SQLite
        if "instrumental" in updates:
            updates["instrumental"] = 1 if updates["instrumental"] else 0

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(station_id)

        await db.execute(
            f"UPDATE radio_stations SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
            values,
        )
        await db.commit()

        return await self.get_station(station_id)

    async def delete_station(self, station_id: str) -> bool:
        db = await get_db()
        cursor = await db.execute(
            "SELECT is_preset FROM radio_stations WHERE id = ?", (station_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        if bool(row["is_preset"]):
            return False

        await db.execute("DELETE FROM radio_stations WHERE id = ?", (station_id,))
        await db.commit()
        return True

    async def create_station_from_song(
        self, song_id: str, name: Optional[str] = None
    ) -> Optional[StationResponse]:
        db = await get_db()
        cursor = await db.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
        row = await cursor.fetchone()
        if row is None:
            return None

        station_name = name or f"Station from {row['title']}"
        data = {
            "name": station_name,
            "description": f"Created from: {row['title']}",
            "genre": "",
            "mood": "",
            "instrumental": bool(row["instrumental"]),
            "vocal_language": "en",
            "bpm_min": row["bpm"] - 10 if row["bpm"] else None,
            "bpm_max": row["bpm"] + 10 if row["bpm"] else None,
            "keyscale": row["keyscale"] or "",
            "timesignature": row["timesignature"] or "",
            "caption_template": row["caption"] or "",
        }

        return await self.create_station(data)

    async def get_settings(self) -> dict[str, Any]:
        """Return current radio chat-LLM state.

        ``active_model`` is the empty string when no caption LLM is selected
        (radio falls back to template captions in that case).
        """
        db = await get_db()
        cursor = await db.execute(
            "SELECT key, value FROM settings WHERE key = 'radio_system_prompt'"
        )
        rows = await cursor.fetchall()
        active_model = await chat_llm.get_configured_chat_model_name()
        custom_system_prompt = ""
        for r in rows:
            if r["key"] == "radio_system_prompt":
                custom_system_prompt = r["value"]

        return {
            "active_model": active_model,
            "loaded_model": chat_llm.get_loaded_chat_model_name(),
            "installed_models": installed_chat_models(),
            "system_prompt": custom_system_prompt,
            "default_system_prompt": RADIO_DEFAULT_SYSTEM_PROMPT,
        }

    async def update_settings(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update radio caption-LLM settings."""
        db = await get_db()
        if model is not None:
            await chat_llm.switch_chat_model(model)
        if system_prompt is not None:
            if system_prompt.strip() == "" or system_prompt.strip() == RADIO_DEFAULT_SYSTEM_PROMPT.strip():
                await db.execute(
                    "DELETE FROM settings WHERE key = 'radio_system_prompt'"
                )
            else:
                await db.execute(
                    "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES ('radio_system_prompt', ?, datetime('now'))",
                    (system_prompt,),
                )
        await db.commit()
        return await self.get_settings()

    async def _get_radio_llm(self) -> tuple[ChatRuntime | None, str, str]:
        """Resolve the radio caption LLM.

        Returns ``(runtime, model_name, system_prompt)``. ``runtime`` is None
        when no model is selected or the chosen model isn't loadable on
        this machine — radio then falls back to its caption template.
        """
        db = await get_db()
        cursor = await db.execute(
            "SELECT key, value FROM settings WHERE key = 'radio_system_prompt'"
        )
        rows = await cursor.fetchall()
        custom_prompt = ""
        for r in rows:
            if r["key"] == "radio_system_prompt":
                custom_prompt = r["value"]

        system_prompt = custom_prompt if custom_prompt.strip() else RADIO_DEFAULT_SYSTEM_PROMPT
        runtime, model_name = await chat_llm.get_configured_chat_runtime()
        if runtime is None:
            logger.info("Radio LLM: no app Chat LLM is loadable; using template caption")
            return (None, model_name, system_prompt)
        return (runtime, model_name, system_prompt)

    async def generate_next_track(self, station_id: str) -> dict[str, Any]:
        """Generate the next track for a radio station.

        Returns the generation result dict with song info on success,
        or an error dict on failure.
        """
        station = await self.get_station(station_id)
        if station is None:
            return {"success": False, "error": "Station not found"}

        logger.info(f"Radio: generating next track for station '{station.name}' (id={station_id})")

        if not generation_service.backend_ready and not settings.delegates_to_workers:
            return {
                "success": False,
                "error": "ACE-Step backend not loaded",
            }

        # Standalone/local generation holds the GPU lock across all MLX and
        # ACE-Step work. Coordinator mode delegates the heavy work to workers.
        use_local_gpu_lock = not settings.delegates_to_workers
        if use_local_gpu_lock:
            acquired = await gpu_lock.await_acquire("radio")
            if not acquired:
                return {
                    "success": False,
                    "error": "GPU lock timeout. Please try again.",
                }

        history_id: str | None = None
        song_title: str | None = None
        deferred_recent_titles: list[str] | None = None
        try:
            duration = await get_default_duration()
            _runtime, _model_name, radio_system_prompt = await self._get_radio_llm()
            spec = await build_music_spec(
                prompt=" ".join(
                    part
                    for part in (station.name, station.description)
                    if part
                ),
                instrumental=station.instrumental,
                genre=station.genre,
                mood=station.mood,
                caption_template="" if station.is_preset else station.caption_template,
                bpm_min=station.bpm_min,
                bpm_max=station.bpm_max,
                keyscale=station.keyscale,
                timesignature=station.timesignature,
                duration=duration,
                source="radio",
                system_prompt=radio_system_prompt,
                allow_holders=frozenset({"radio"}) if use_local_gpu_lock else None,
            )
            caption = str(spec.get("caption") or station.caption_template or "A music track")
            lyrics = str(spec.get("lyrics") or "")
            bpm = spec.get("bpm") if isinstance(spec.get("bpm"), int) else None

            # Parse advanced params
            advanced = {}
            if station.advanced_params_json:
                try:
                    advanced = json.loads(station.advanced_params_json)
                except json.JSONDecodeError:
                    pass

            params_dict = build_text_to_music_params(
                caption,
                lyrics=lyrics,
                instrumental=station.instrumental,
                vocal_language="en",
                duration=duration,
                batch_size=1,
                extra=advanced,
            )
            if bpm is not None:
                params_dict["bpm"] = bpm
            if station.keyscale:
                params_dict["keyscale"] = station.keyscale
            if station.timesignature:
                params_dict["timesignature"] = station.timesignature
            for key in ("quality_profile", "spec_source", "source_prompt"):
                if spec.get(key):
                    params_dict[key] = spec[key]
            if generation_service.active_dit_model:
                params_dict["dit_model"] = generation_service.active_dit_model
            if generation_service.active_lm_model:
                params_dict["lm_model"] = generation_service.active_lm_model

            if hasattr(generation_service, "prepare_params"):
                allow_holders = frozenset({"radio"}) if use_local_gpu_lock else None
                params_dict = await generation_service.prepare_params(
                    params_dict,
                    allow_holders=allow_holders,
                )

            # Create generation history entry
            history_id = str(uuid.uuid4())
            started_at = datetime.now(timezone.utc)
            db = await get_db()
            history_params = generation_service.public_params(params_dict)
            await db.execute(
                """INSERT INTO generation_history (id, task_type, status, params_json, started_at, created_at)
                   VALUES (?, 'text2music', 'running', ?, ?, ?)""",
                (
                    history_id,
                    json.dumps(history_params),
                    started_at.isoformat(),
                    started_at.isoformat(),
                ),
            )
            await db.commit()

            result = await generation_service.generate(params_dict, lyrics_prepared=True)

            if result.get("success") and result.get("audios") and song_title is None:
                from bangers.services.title_generator import pick_random_title

                db = await get_db()
                recent_cursor = await db.execute(
                    """SELECT s.title FROM songs s
                       JOIN radio_station_songs rs ON s.id = rs.song_id
                       WHERE rs.station_id = ?
                       ORDER BY rs.generated_at DESC LIMIT 10""",
                    (station_id,),
                )
                recent_rows = await recent_cursor.fetchall()
                recent_titles = [r["title"] for r in recent_rows if r["title"]]
                song_title = pick_random_title(recent_titles)
                deferred_recent_titles = recent_titles
        except Exception as e:
            logger.exception(f"Radio generation failed for station {station_id}")
            if history_id is not None:
                try:
                    db = await get_db()
                    await db.execute(
                        "UPDATE generation_history SET status = 'failed', error_message = ? WHERE id = ?",
                        (str(e), history_id),
                    )
                    await db.commit()
                except Exception:
                    pass
            return {"success": False, "error": str(e)}
        finally:
            # Release after the music-critical stage. Title generation and
            # DB/file work run outside the music lock.
            if use_local_gpu_lock:
                await gpu_lock.release("radio")

        # --- Post-generation work (DB writes, file copies) runs WITHOUT the GPU lock ---
        try:
            if result.get("success"):
                audios = result.get("audios", [])
                if audios:
                    audio = audios[0]
                    completed_at = datetime.now(timezone.utc)
                    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                    audio_results = [
                        {
                            "path": a.get("path", ""),
                            "key": a.get("key", ""),
                            "sample_rate": a.get("sample_rate", 48000),
                            "params": {
                                k: v
                                for k, v in (a.get("params") or params_dict).items()
                                if k != "tensor" and not k.startswith("_")
                            },
                        }
                        for a in audios
                    ]
                    db = await get_db()

                    # Auto-save to library
                    from bangers.routers.songs import _row_to_song
                    from bangers.models.common import SongResponse as SongResp

                    song_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc).isoformat()
                    audio_path = audio.get("path", "")
                    file_format = audio_path.rsplit(".", 1)[-1] if "." in audio_path else "flac"

                    # Title generation happens after the music lock is released.
                    if song_title is None:
                        song_title = f"Radio Track {datetime.now(timezone.utc).strftime('%H:%M')}"

                    # Update history as completed (now includes title)
                    await db.execute(
                        """UPDATE generation_history
                           SET status = 'completed', title = ?, result_json = ?,
                               audio_count = ?, completed_at = ?, duration_ms = ?
                           WHERE id = ?""",
                        (song_title, json.dumps(audio_results), len(audios),
                         completed_at.isoformat(), duration_ms, history_id),
                    )
                    await db.commit()

                    # Copy to library
                    import shutil
                    from pathlib import Path

                    src = Path(audio_path)
                    dest_filename = f"{song_id}.{file_format}"
                    dest = settings.AUDIO_DIR / dest_filename

                    if src.exists():
                        shutil.copy2(str(src), str(dest))
                        file_size = dest.stat().st_size
                    else:
                        file_size = 0

                    db = await get_db()
                    await db.execute(
                        """INSERT INTO songs (
                            id, title, file_path, file_format, duration_seconds,
                            caption, lyrics, bpm, keyscale, timesignature,
                            vocal_language, instrumental, generation_history_id,
                            tags, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            song_id,
                            song_title,
                            dest_filename,
                            file_format,
                            duration,
                            caption,
                            lyrics,
                            bpm,
                            station.keyscale,
                            station.timesignature,
                            "en",
                            1 if station.instrumental else 0,
                            history_id,
                            "radio",
                            now,
                            now,
                        ),
                    )

                    # Link to station
                    link_id = str(uuid.uuid4())
                    await db.execute(
                        """INSERT INTO radio_station_songs (id, station_id, song_id, position, generated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (link_id, station_id, song_id, 0, now),
                    )

                    # Update station play count
                    await db.execute(
                        """UPDATE radio_stations
                           SET total_plays = total_plays + 1, last_played_at = ?
                           WHERE id = ?""",
                        (now, station_id),
                    )
                    await db.commit()

                    if deferred_recent_titles is not None and history_id is not None:
                        from bangers.services.deferred_titles import schedule_radio_title_retry

                        schedule_radio_title_retry(
                            song_id=song_id,
                            history_id=history_id,
                            avoid_titles=deferred_recent_titles,
                        )

                    song_response = SongResp(
                        id=song_id,
                        title=song_title,
                        file_path=dest_filename,
                        file_format=file_format,
                        duration_seconds=duration,
                        caption=caption,
                        lyrics=lyrics,
                        bpm=bpm,
                        keyscale=station.keyscale,
                        timesignature=station.timesignature,
                        vocal_language="en",
                        instrumental=station.instrumental,
                        created_at=now,
                        updated_at=now,
                    )

                    return {
                        "success": True,
                        "song": song_response.model_dump(),
                    }

            error = result.get("error", "Generation failed")
            db = await get_db()
            await db.execute(
                "UPDATE generation_history SET status = 'failed', error_message = ? WHERE id = ?",
                (error, history_id),
            )
            await db.commit()
            return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Radio post-generation failed for station {station_id}")
            return {"success": False, "error": str(e)}

    async def get_station_songs(
        self, station_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        db = await get_db()
        cursor = await db.execute(
            """SELECT s.* FROM songs s
               JOIN radio_station_songs rs ON s.id = rs.song_id
               WHERE rs.station_id = ?
               ORDER BY rs.generated_at DESC
               LIMIT ?""",
            (station_id, limit),
        )
        rows = await cursor.fetchall()
        from bangers.routers.songs import _row_to_song

        return [_row_to_song(row).model_dump() for row in rows]


radio_service = RadioService()
