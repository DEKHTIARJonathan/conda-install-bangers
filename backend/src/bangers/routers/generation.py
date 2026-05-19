import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from bangers.models.generation import (
    GenerateRequest,
    GenerateResponse,
    JobStatusResponse,
    FormatRequest,
    FormatResponse,
    SampleRequest,
    SampleResponse,
    GenerateTitleRequest,
    GenerateTitleResponse,
)
from bangers.config import settings
from bangers.db.connection import get_db
from bangers.services.duration_settings import coerce_duration, get_default_duration
from bangers.services.generation import generation_service
from bangers.ws.manager import generation_ws_manager
from bangers.services.gpu_lock import gpu_lock

class GenerationCancelled(BaseException):
    """Raised inside progress_callback when a job has been cancelled.

    This intentionally mirrors asyncio.CancelledError and inherits from
    BaseException so dependency code with broad ``except Exception`` handlers
    cannot convert a cooperative cancellation into a noisy generation failure.
    """
    pass


class _LMProgressInterpolator:
    """Wraps a progress callback to provide smooth interpolation during blocking LM phases.

    ACE-Step's LLM handler only emits two sparse progress updates: 0.1 (Phase 1 start)
    and 0.5 (Phase 2 start). Between these, generation blocks for seconds to minutes
    with zero feedback. This wrapper intercepts those values, remaps them to compressed
    ranges, and runs timer threads that smoothly fill the gaps:

      ACE-Step 0.1 → emit 0.05, timer fills 0.05→0.09 (Phase 1: CoT metadata)
      ACE-Step 0.5 → emit 0.1,  timer fills 0.1→0.49  (Phase 2: audio codes)
      Next value >0.5 → emit 0.5 "LM complete", pass through (DiT phase)

    The timer uses a logarithmic curve that asymptotically approaches the target,
    so the progress bar always keeps moving regardless of generation duration.
    """

    # ACE-Step LM milestones to intercept
    _PHASE1_VALUE = 0.1
    _PHASE2_VALUE = 0.5

    def __init__(self, real_callback):
        self._real_callback = real_callback
        self._stop_event = None
        self._thread = None
        self._last_lm_value = 0.0
        self._lm_complete = False

    def __call__(self, value, desc=""):
        # Once LM is done, pass everything through unchanged
        if self._lm_complete:
            self._real_callback(value, desc)
            return

        # Intercept Phase 1 start (0.1 from ACE-Step)
        if abs(value - self._PHASE1_VALUE) < 0.01 and self._last_lm_value < self._PHASE1_VALUE:
            self._stop_timer()
            self._last_lm_value = value
            self._real_callback(0.05, desc)
            self._start_timer(0.05, 0.09, desc, half_life=10.0)
            return

        # Intercept Phase 2 start (0.5 from ACE-Step)
        if abs(value - self._PHASE2_VALUE) < 0.01 and self._last_lm_value < self._PHASE2_VALUE:
            self._stop_timer()
            self._last_lm_value = value
            self._real_callback(0.1, desc)
            self._start_timer(0.1, 0.49, desc, half_life=45.0)
            return

        # First value after Phase 2 (>0.5) means LM is done, DiT is starting
        if value > self._PHASE2_VALUE and self._last_lm_value >= self._PHASE2_VALUE:
            self._stop_timer()
            self._lm_complete = True
            self._real_callback(0.5, "LM generation complete")
            self._real_callback(value, desc)
            return

        # Pass through any other values (e.g. values before LM starts)
        self._last_lm_value = value
        self._real_callback(value, desc)

    def _start_timer(self, start_val, end_val, desc, half_life):
        self._stop_timer()
        evt = threading.Event()

        def _runner():
            t0 = time.time()
            while not evt.is_set():
                elapsed = time.time() - t0
                frac = 1.0 - 1.0 / (1.0 + elapsed / half_life)
                value = start_val + (end_val - start_val) * frac
                try:
                    self._real_callback(value, desc)
                except GenerationCancelled:
                    break
                except Exception:
                    break
                evt.wait(1.0)

        thread = threading.Thread(target=_runner, name="lm-progress", daemon=True)
        thread.start()
        self._stop_event = evt
        self._thread = thread

    def _stop_timer(self):
        if self._stop_event is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            self._stop_event = None
            self._thread = None

    def cleanup(self):
        """Stop any running timer thread."""
        self._stop_timer()


router = APIRouter(prefix="/generate", tags=["generation"])


def _timing_bucket(desc: str) -> str:
    """Map a progress callback description to a stage timing bucket.

    Rules are mutually exclusive and ordered most-specific-first so that
    overlapping substrings (``"caption"`` appears in both LM and conditioning
    stages, ``"lm"`` appears inside ``"calm"``) don't cross-contaminate.
    """
    lowered = (desc or "").lower()
    if not lowered:
        return "backend_other"

    def has_word(word: str) -> bool:
        # word-boundary-ish check: surrounded by non-letter or string ends.
        idx = lowered.find(word)
        while idx != -1:
            before_ok = idx == 0 or not lowered[idx - 1].isalpha()
            after_ok = (
                idx + len(word) == len(lowered)
                or not lowered[idx + len(word)].isalpha()
            )
            if before_ok and after_ok:
                return True
            idx = lowered.find(word, idx + 1)
        return False

    # VAE / decoding wins over generic "decode caption" matches because VAE
    # never overlaps with text encoder paths.
    if "vae" in lowered or "decode_audio" in lowered or "audio decode" in lowered:
        return "vae"

    # ACE 5Hz LM stage: explicit markers from the LM handler.
    if (
        "5hz" in lowered
        or has_word("lm")
        or "language model" in lowered
        or "metadata" in lowered
        or "lyrics" in lowered
        or "generating caption" in lowered
        or "phase 1" in lowered
        or "phase 2" in lowered
    ):
        return "ace_5hz_lm"

    # Text conditioning: only count this when the LM stage is not itself
    # producing the caption (handled above).
    if (
        "condition" in lowered
        or "text encoder" in lowered
        or "encode text" in lowered
        or "encode caption" in lowered
    ):
        return "text_conditioning"

    if (
        "dit" in lowered
        or "diffusion" in lowered
        or "sampling" in lowered
        or "denois" in lowered
        or "step " in lowered
    ):
        return "dit"

    if "normaliz" in lowered or "loudness" in lowered:
        return "normalization"

    if (
        "saving" in lowered
        or has_word("save")
        or "writing" in lowered
        or has_word("write")
        or "export" in lowered
    ):
        return "audio_save"

    return "backend_other"


_GENERATION_SETTING_FIELDS = {
    "audio_format": ("audio_format", str),
    "batch_size": ("batch_size", int),
    "default_duration": ("duration", float),
    "inference_steps": ("inference_steps", int),
    "guidance_scale": ("guidance_scale", float),
    "thinking": ("thinking", "bool"),
}


def _request_fields_set(request: GenerateRequest) -> set[str]:
    model_fields_set = getattr(request, "model_fields_set", None)
    if model_fields_set is not None:
        return set(model_fields_set)
    return set(getattr(request, "__fields_set__", set()))


def _coerce_saved_setting(value: str, caster):
    if caster == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return caster(value)


def _request_client_id(request: GenerateRequest) -> str | None:
    client_id = request.client_id.strip()
    return client_id or None


def _job_client_id(job_id: str) -> str | None:
    job = generation_service.get_job(job_id) or {}
    client_id = str(job.get("client_id") or "").strip()
    return client_id or None


async def _broadcast_job_event(job_id: str, message: dict) -> None:
    await generation_ws_manager.broadcast(message, client_id=_job_client_id(job_id))


async def _apply_saved_generation_defaults(request: GenerateRequest) -> dict[str, str]:
    """Apply DB/profile-backed defaults to omitted generation request fields.

    Returns the saved settings map so callers can reuse it without a second DB
    roundtrip.
    """
    fields_set = _request_fields_set(request)
    try:
        db = await get_db()
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
    except Exception as exc:
        logger.debug(f"Could not load generation defaults from settings: {exc}")
        return {}

    saved_settings = {row["key"]: row["value"] for row in rows}
    for setting_key, (request_field, caster) in _GENERATION_SETTING_FIELDS.items():
        if setting_key not in saved_settings:
            continue
        if setting_key == "default_duration":
            setattr(request, request_field, coerce_duration(saved_settings[setting_key]))
            continue
        if request_field in fields_set:
            continue
        try:
            setattr(request, request_field, _coerce_saved_setting(saved_settings[setting_key], caster))
        except (TypeError, ValueError) as exc:
            logger.warning(f"Ignoring invalid saved setting {setting_key}={saved_settings[setting_key]!r}: {exc}")
    return saved_settings


@router.get("/gpu-status")
async def get_gpu_status() -> dict:
    return {"locked": gpu_lock.is_locked, "holder": gpu_lock.holder}


@router.post("", response_model=GenerateResponse)
async def submit_generation(request: GenerateRequest) -> GenerateResponse:
    svc = generation_service

    await _apply_saved_generation_defaults(request)

    if not settings.delegates_to_workers:
        if not svc.active_dit_model:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "no_dit_model_selected",
                    "message": "No ACE-Step DiT model selected. Choose one on the Models page.",
                    "missing": ["dit_model"],
                },
            )
        if not svc.backend_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "dit_model_not_loaded",
                    "message": (
                        f"DiT model '{svc.active_dit_model}' is still loading or "
                        "failed to load. Check server logs and try again shortly."
                    ),
                },
            )

    job_id = svc.create_job(client_id=_request_client_id(request))

    asyncio.create_task(_run_generation(job_id, request))

    return GenerateResponse(job_id=job_id, status="queued")


async def _run_generation(job_id: str, request: GenerateRequest) -> None:
    svc = generation_service
    timings: dict[str, float] = {}
    queue_wait_started = time.perf_counter()
    deferred_title_source: str | None = None
    deferred_title_history_id: str | None = None
    use_local_gpu_lock = not settings.delegates_to_workers

    # Check cancellation before waiting for GPU
    if svc.is_cancelled(job_id):
        logger.info(f"Job {job_id} cancelled before GPU acquire")
        await _broadcast_job_event(job_id, {
            "type": "failed", "job_id": job_id, "error": "Cancelled by user",
        })
        return

    current_holder = gpu_lock.holder
    if use_local_gpu_lock and gpu_lock.is_locked:
        await _broadcast_job_event(job_id, {
            "type": "progress",
            "job_id": job_id,
            "progress": 0.0,
            "stage": f"Waiting for GPU (in use by {current_holder})...",
        })

    if use_local_gpu_lock:
        await gpu_lock.await_acquire("generation")
        timings["queue_wait_ms"] = round((time.perf_counter() - queue_wait_started) * 1000, 2)
    else:
        timings["queue_wait_ms"] = 0.0
    svc.update_job(job_id, timings=timings)

    # Check cancellation after acquiring GPU
    if svc.is_cancelled(job_id):
        logger.info(f"Job {job_id} cancelled after GPU acquire")
        if use_local_gpu_lock:
            await gpu_lock.release("generation")
        await _broadcast_job_event(job_id, {
            "type": "failed", "job_id": job_id, "error": "Cancelled by user",
        })
        return

    started_at = datetime.now(timezone.utc)
    history_id = str(uuid.uuid4())
    params_dict = request.model_dump(exclude={"client_id"})
    # Force English vocals for all generation requests.
    params_dict["vocal_language"] = "en"

    # Inject active model names so history records which models were used
    if svc.active_dit_model:
        params_dict["dit_model"] = svc.active_dit_model
    if svc.active_lm_model:
        params_dict["lm_model"] = svc.active_lm_model

    try:
        lyrics_started = time.perf_counter()
        allow_holders = frozenset({"generation"}) if use_local_gpu_lock else None
        params_dict = await svc.prepare_params(
            params_dict,
            allow_holders=allow_holders,
        )
        timings["lyrics_pipeline_ms"] = round(
            (time.perf_counter() - lyrics_started) * 1000,
            2,
        )
        prepared_public_params = svc.public_params(params_dict)
        svc.update_job(job_id, prepared_params=prepared_public_params)
        await _broadcast_job_event(job_id, {
            "type": "prepared",
            "job_id": job_id,
            "params": prepared_public_params,
        })
    except Exception as e:
        logger.warning(f"Generation lyrics pipeline failed: {e}")
        svc.update_job(
            job_id,
            status="failed",
            error=str(e),
            timings=timings,
            history_id=history_id,
        )
        if use_local_gpu_lock:
            await gpu_lock.release("generation")
        await _broadcast_job_event(job_id, {
            "type": "failed",
            "job_id": job_id,
            "error": str(e),
        })
        return

    # Insert initial history record
    try:
        db = await get_db()
        history_params = svc.public_params(params_dict)
        await db.execute(
            """INSERT INTO generation_history (id, task_type, status, params_json, started_at, created_at)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (history_id, request.task_type, json.dumps(history_params),
             started_at.isoformat(), started_at.isoformat()),
        )
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to insert generation history: {e}")

    svc.update_job(job_id, status="running", stage="preparing")
    progress_msg: dict = {
        "type": "progress",
        "job_id": job_id,
        "progress": 0.0,
        "stage": "preparing",
    }
    progress_msg["step"] = 0
    progress_msg["total_steps"] = request.inference_steps
    await _broadcast_job_event(job_id, progress_msg)

    # Check cancellation before generation
    if svc.is_cancelled(job_id):
        logger.info(f"Job {job_id} cancelled before generation")
        if use_local_gpu_lock:
            await gpu_lock.release("generation")
        await _broadcast_job_event(job_id, {
            "type": "failed", "job_id": job_id, "error": "Cancelled by user",
        })
        return

    svc.update_job(job_id, progress=0.02, stage="Starting generation...")
    await _broadcast_job_event(job_id, {
        "type": "progress",
        "job_id": job_id,
        "progress": 0.02,
        "stage": "Starting generation...",
    })

    loop = asyncio.get_running_loop()

    last_broadcast_time = 0.0
    current_stage_bucket: str | None = None
    current_stage_started = time.perf_counter()

    def _mark_stage_timing(desc: str) -> None:
        nonlocal current_stage_bucket, current_stage_started
        if not desc:
            return
        bucket = _timing_bucket(desc)
        now = time.perf_counter()
        if current_stage_bucket is None:
            current_stage_bucket = bucket
            current_stage_started = now
            return
        if bucket == current_stage_bucket:
            return
        key = f"stage_{current_stage_bucket}_ms"
        timings[key] = round(timings.get(key, 0.0) + (now - current_stage_started) * 1000, 2)
        current_stage_bucket = bucket
        current_stage_started = now

    def progress_callback(progress_value: float, desc: str = "") -> None:
        nonlocal last_broadcast_time
        if svc.is_cancelled(job_id):
            logger.info(f"Job {job_id} cancelled during generation (progress_callback)")
            raise GenerationCancelled()
        _mark_stage_timing(desc)
        now = time.time()
        svc.update_job(job_id, progress=progress_value, stage=desc)
        # Throttle WebSocket broadcasts to max 1 per 1.5s (except near completion)
        if progress_value < 0.99 and (now - last_broadcast_time) < 1.5:
            return
        last_broadcast_time = now
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            _broadcast_job_event(job_id, {
                "type": "progress",
                "job_id": job_id,
                "progress": progress_value,
                "stage": desc,
            }),
        )

    # Wrap callback with LM progress interpolator for smooth updates during blocking LM phases
    lm_interpolator = _LMProgressInterpolator(progress_callback)

    try:
        try:
            generation_started = time.perf_counter()
            result = await svc.generate(
                params_dict,
                progress_callback=lm_interpolator,
                lyrics_prepared=True,
            )
            timings["music_generation_ms"] = round((time.perf_counter() - generation_started) * 1000, 2)
            if current_stage_bucket is not None:
                key = f"stage_{current_stage_bucket}_ms"
                timings[key] = round(
                    timings.get(key, 0.0) + (time.perf_counter() - current_stage_started) * 1000,
                    2,
                )
            if isinstance(result.get("timings"), dict):
                for key, value in result["timings"].items():
                    try:
                        timings[f"backend_{key}"] = round(float(value), 2)
                    except (TypeError, ValueError):
                        pass
            svc.update_job(job_id, timings=timings)

            if result.get("success"):
                audios = result.get("audios", [])
                results = []
                for audio in audios:
                    audio_info = {
                        "path": audio.get("path", ""),
                        "key": audio.get("key", ""),
                        "sample_rate": audio.get("sample_rate", 48000),
                        "params": {
                            k: v for k, v in (audio.get("params") or {}).items()
                            if k != "tensor" and not k.startswith("_")
                        },
                    }
                    results.append(audio_info)

                # When generating locally, keep the title inside the music
                # lock so the next job cannot jump ahead of it.
                title_source = request.caption
                generated_title: str | None = None
                if request.auto_title and title_source:
                    try:
                        from bangers.services.title_generator import generate_song_title
                        title_started = time.perf_counter()
                        allow_holders = frozenset({"generation"}) if use_local_gpu_lock else None
                        generated_title = await generate_song_title(
                            title_source, "", "", "Untitled",
                            allow_holders=allow_holders,
                        )
                        timings["title_llm_ms"] = round(
                            (time.perf_counter() - title_started) * 1000, 2
                        )
                    except Exception as exc:
                        logger.warning(f"Title generation failed: {exc}")

                # Persist completion + title in a single DB write.
                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                db_write_started = time.perf_counter()
                try:
                    db = await get_db()
                    if generated_title is not None:
                        await db.execute(
                            """UPDATE generation_history
                               SET status='completed', result_json=?, audio_count=?,
                                   completed_at=?, duration_ms=?, title=?
                               WHERE id=?""",
                            (json.dumps(results), len(results),
                             completed_at.isoformat(), duration_ms,
                             generated_title, history_id),
                        )
                    else:
                        await db.execute(
                            """UPDATE generation_history
                               SET status='completed', result_json=?, audio_count=?,
                                   completed_at=?, duration_ms=?
                               WHERE id=?""",
                            (json.dumps(results), len(results),
                             completed_at.isoformat(), duration_ms, history_id),
                        )
                    await db.commit()
                except Exception as e:
                    logger.warning(f"Failed to update generation history: {e}")
                timings["db_write_ms"] = round((time.perf_counter() - db_write_started) * 1000, 2)

                svc.update_job(
                    job_id,
                    status="completed",
                    progress=1.0,
                    results=results,
                    timings=timings,
                    history_id=history_id,
                )

                # Send the title BEFORE completed so the frontend has it when
                # the auto-save side effects run.
                if generated_title is not None:
                    await _broadcast_job_event(job_id, {
                        "type": "title",
                        "job_id": job_id,
                        "history_id": history_id,
                        "title": generated_title,
                    })

                await _broadcast_job_event(job_id, {
                    "type": "completed",
                    "job_id": job_id,
                    "history_id": history_id,
                    "results": results,
                })
            else:
                error = result.get("error", "Unknown error")
                svc.update_job(job_id, status="failed", error=error, timings=timings, history_id=history_id)

                # Log failure to history
                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                try:
                    db = await get_db()
                    await db.execute(
                        """UPDATE generation_history
                           SET status='failed', error_message=?,
                               completed_at=?, duration_ms=?
                           WHERE id=?""",
                        (error, completed_at.isoformat(), duration_ms, history_id),
                    )
                    await db.commit()
                except Exception as e:
                    logger.warning(f"Failed to update generation history: {e}")

                await _broadcast_job_event(job_id, {
                    "type": "failed",
                    "job_id": job_id,
                    "error": error,
                })

        except GenerationCancelled:
            logger.info(f"Generation job {job_id} cancelled by user")
            svc.update_job(
                job_id,
                status="cancelled",
                error="Cancelled by user",
                history_id=history_id,
            )

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            try:
                db = await get_db()
                await db.execute(
                    """UPDATE generation_history
                       SET status='cancelled', error_message=?,
                           completed_at=?, duration_ms=?
                       WHERE id=?""",
                    ("Cancelled by user", completed_at.isoformat(), duration_ms, history_id),
                )
                await db.commit()
            except Exception as he:
                logger.warning(f"Failed to update generation history: {he}")

            await _broadcast_job_event(job_id, {
                "type": "failed",
                "job_id": job_id,
                "error": "Cancelled by user",
            })

        except Exception as e:
            logger.exception(f"Generation job {job_id} failed")
            svc.update_job(job_id, status="failed", error=str(e), history_id=history_id)

            # Log exception to history
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            try:
                db = await get_db()
                await db.execute(
                    """UPDATE generation_history
                       SET status='failed', error_message=?,
                           completed_at=?, duration_ms=?
                       WHERE id=?""",
                    (str(e), completed_at.isoformat(), duration_ms, history_id),
                )
                await db.commit()
            except Exception as he:
                logger.warning(f"Failed to update generation history: {he}")

            await _broadcast_job_event(job_id, {
                "type": "failed",
                "job_id": job_id,
                "error": str(e),
            })
    finally:
        lm_interpolator.cleanup()
        if use_local_gpu_lock:
            await gpu_lock.release("generation")

    if deferred_title_source and deferred_title_history_id:
        from bangers.services.deferred_titles import schedule_history_title_retry

        schedule_history_title_retry(
            job_id=job_id,
            history_id=deferred_title_history_id,
            caption=deferred_title_source,
            fallback="Untitled",
        )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, str]:
    cancelled = generation_service.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Job not found")
    logger.info(f"Job {job_id} marked for cancellation")
    return {"message": "Cancellation requested"}


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = generation_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job.get("stage", ""),
        results=job.get("results", []),
        error=job.get("error"),
        timings=job.get("timings", {}),
        history_id=job.get("history_id"),
    )


format_router = APIRouter(tags=["generation"])


@format_router.post("/format", response_model=FormatResponse)
async def format_caption(request: FormatRequest) -> FormatResponse:
    svc = generation_service

    await gpu_lock.await_acquire("format")

    try:
        user_metadata = {}
        if request.bpm is not None:
            user_metadata["bpm"] = request.bpm
        if request.keyscale:
            user_metadata["keyscale"] = request.keyscale
        if request.timesignature:
            user_metadata["timesignature"] = request.timesignature
        user_metadata["duration"] = await get_default_duration()
        user_metadata["language"] = "en"

        try:
            result = await svc.format_sample(
                caption=request.caption,
                lyrics=request.lyrics,
                user_metadata=user_metadata or None,
                allow_holders=frozenset({"format"}),
            )
        except Exception as exc:
            result = {
                "caption": request.caption,
                "lyrics": request.lyrics,
                "success": False,
                "error": str(exc),
            }

        return FormatResponse(**result)
    finally:
        await gpu_lock.release("format")


@format_router.post("/generate-title", response_model=GenerateTitleResponse)
async def generate_title(request: GenerateTitleRequest) -> GenerateTitleResponse:
    from bangers.services.title_generator import generate_song_title

    # Acquire GPU lock to prevent MLX chat model from conflicting with DiT on Metal
    acquired = await gpu_lock.acquire("title-generation")
    if not acquired:
        fallback = request.caption[:60] if request.caption else request.fallback
        return GenerateTitleResponse(title=fallback)

    try:
        title = await generate_song_title(
            request.caption,
            request.genre,
            request.mood,
            request.fallback,
            allow_holders=frozenset({"title-generation"}),
        )
        return GenerateTitleResponse(title=title)
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        fallback = request.caption[:60] if request.caption else request.fallback
        return GenerateTitleResponse(title=fallback, success=False, error=str(e))
    finally:
        await gpu_lock.release("title-generation")


@format_router.post("/sample", response_model=SampleResponse)
async def create_sample(request: SampleRequest) -> SampleResponse:
    svc = generation_service

    await gpu_lock.await_acquire("sample")

    try:
        try:
            result = await svc.create_sample(
                query=request.query,
                instrumental=request.instrumental,
                vocal_language="en",
                temperature=request.temperature,
                allow_holders=frozenset({"sample"}),
            )
        except Exception as exc:
            result = {
                "caption": request.query,
                "lyrics": "",
                "instrumental": request.instrumental,
                "success": False,
                "error": str(exc),
            }
        result["duration"] = await get_default_duration()

        return SampleResponse(**result)
    finally:
        await gpu_lock.release("sample")


ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/ws/generate")
async def websocket_generation(websocket: WebSocket) -> None:
    client_id = websocket.query_params.get("client_id") or None
    await generation_ws_manager.connect(websocket, client_id=client_id)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        await generation_ws_manager.disconnect(websocket)
