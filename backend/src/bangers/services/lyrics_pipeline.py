from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from bangers.services import chat_llm
from bangers.services.runtime_settings import lyrics_guardrails_enabled


class LyricsPipelineError(RuntimeError):
    """Raised when the unified lyrics pipeline cannot produce safe lyrics."""


class LyricsRejectedError(LyricsPipelineError):
    """Raised when lyrics violate the Code of Conduct and cannot be rewritten."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason.strip()
        message = "Lyrics violate the Code of Conduct and cannot be processed."
        if self.reason:
            message = f"{message} {self.reason}"
        super().__init__(message)


_GUARDED_LYRICS_RE = re.compile(
    r"<guarded_lyrics>\s*(.*?)\s*</guarded_lyrics\s*>",
    re.IGNORECASE | re.DOTALL,
)
_LYRICS_REJECTED_RE = re.compile(
    r"<lyrics_rejected>\s*(.*?)\s*</lyrics_rejected\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_SECTION_TAG_RE = re.compile(r"\[[^\]\n]+\]")
_PLACEHOLDER_RE = re.compile(
    r"(\.{3}|…|\btbd\b|\btodo\b|\bplaceholder\b|lyrics here|"
    r"reviewed or rewritten lyrics here)",
    re.IGNORECASE,
)
MAX_GENERATED_LYRICS_ATTEMPTS = 3
LYRICS_PIPELINE_PREPARED_KEY = "_lyrics_pipeline_prepared"
_LYRICS_PIPELINE_INTERNAL_PREFIX = "_lyrics_pipeline_"
_LYRICS_PIPELINE_PREPARED_VALUE = object()


def _guardrails_path() -> Path:
    return Path(__file__).resolve().parents[1] / "lyrics_guardrails.md"


def load_guardrails() -> str:
    return _guardrails_path().read_text(encoding="utf-8")


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    return _UNCLOSED_THINK_RE.sub("", text).strip()


def _looks_like_placeholder_lyrics(lyrics: str) -> bool:
    content = _SECTION_TAG_RE.sub("", lyrics).strip()
    content = _PLACEHOLDER_RE.sub("", content)
    content = re.sub(r"[\W_]+", "", content, flags=re.UNICODE)
    return len(content) < 4


def _looks_like_placeholder_caption(caption: str) -> bool:
    normalized = re.sub(r"\s+", " ", caption).strip().lower()
    return normalized in {
        "",
        "detailed music caption",
        "complete caption",
        "complete detailed caption",
    }


def coerce_instrumental(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def mark_lyrics_pipeline_prepared(params: dict[str, Any]) -> dict[str, Any]:
    params[LYRICS_PIPELINE_PREPARED_KEY] = _LYRICS_PIPELINE_PREPARED_VALUE
    return params


def is_lyrics_pipeline_prepared(params: dict[str, Any]) -> bool:
    return params.get(LYRICS_PIPELINE_PREPARED_KEY) is _LYRICS_PIPELINE_PREPARED_VALUE


def strip_lyrics_pipeline_internal_keys(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if not key.startswith(_LYRICS_PIPELINE_INTERNAL_PREFIX)
    }


def extract_guarded_lyrics(raw: str) -> str:
    cleaned = _strip_thinking(raw)
    rejection = _LYRICS_REJECTED_RE.fullmatch(cleaned)
    if rejection is not None:
        reason = rejection.group(1).strip()
        raise LyricsRejectedError(reason)
    match = _GUARDED_LYRICS_RE.fullmatch(cleaned)
    if match is None:
        raise LyricsPipelineError(
            "Guardrail reviewer did not return <guarded_lyrics> or "
            "<lyrics_rejected> output."
        )
    lyrics = match.group(1).strip()
    if not lyrics:
        raise LyricsPipelineError("Guardrail reviewer returned empty lyrics.")
    if _looks_like_placeholder_lyrics(lyrics):
        raise LyricsPipelineError("Guardrail reviewer returned placeholder lyrics.")
    return lyrics


def _json_from_text(raw: str) -> dict[str, Any]:
    cleaned = _strip_thinking(raw)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


async def review_lyrics(
    lyrics: str,
    *,
    caption: str = "",
    allow_holders: frozenset[str] | None = None,
) -> str:
    if not lyrics.strip():
        return ""

    logger.info(
        "Lyrics guardrails: reviewing {} lyric chars with fresh LLM context",
        len(lyrics),
    )
    logger.info("Lyrics guardrails input lyrics:\n{}", lyrics)
    guardrails = load_guardrails()
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final lyrics guardrail reviewer for a music generation app. "
                "Use a fresh context, follow the provided guardrails exactly, and output "
                "only one required block: <guarded_lyrics> or <lyrics_rejected>."
            ),
        },
        {
            "role": "user",
            "content": (
                "Full lyrics guardrails:\n"
                f"{guardrails}\n\n"
                "Creative caption/context:\n"
                f"{caption or '(none)'}\n\n"
                "Lyrics to review:\n"
                "<lyrics>\n"
                f"{lyrics}\n"
                "</lyrics>"
            ),
        },
    ]

    raw = await chat_llm.chat(
        messages,
        max_tokens=4096,
        temperature=0.0,
        allow_holders=allow_holders,
    )
    logger.info("Lyrics guardrails model raw output:\n{}", raw)
    guarded = extract_guarded_lyrics(raw)
    logger.info(
        "Lyrics guardrails: accepted {} lyric chars (changed={})",
        len(guarded),
        guarded.strip() != lyrics.strip(),
    )
    return guarded


async def review_lyrics_if_enabled(
    lyrics: str,
    *,
    caption: str = "",
    allow_holders: frozenset[str] | None = None,
) -> str:
    if not lyrics.strip():
        return ""
    if not await lyrics_guardrails_enabled():
        logger.info("Lyrics guardrails: disabled; returning lyrics without review")
        return lyrics
    return await review_lyrics(lyrics, caption=caption, allow_holders=allow_holders)


async def generate_song_spec(
    prompt: str,
    *,
    instrumental: bool = False,
    user_metadata: dict[str, Any] | None = None,
    allow_holders: frozenset[str] | None = None,
) -> dict[str, Any]:
    instrumental = coerce_instrumental(instrumental)
    metadata = user_metadata or {}
    section_instruction = (
        "If instrumental is false, write original English lyrics with [verse], "
        "[chorus], and optional [bridge]/[outro] tags. If instrumental is true, "
        "lyrics must be an empty string."
    )
    last_error: Exception | None = None

    for attempt in range(1, MAX_GENERATED_LYRICS_ATTEMPTS + 1):
        fresh_attempt_instruction = ""
        if attempt > 1:
            fresh_attempt_instruction = (
                f"\nFresh generation attempt {attempt} of "
                f"{MAX_GENERATED_LYRICS_ATTEMPTS}: choose a new lyrical premise, "
                "new wording, and new imagery that still fits the request."
            )
        retry_instruction = ""
        if last_error is not None:
            retry_instruction = (
                "\n\nPrevious generated lyrics failed the lyrics guardrail review. "
                f"Reason: {last_error}. Generate a new, complete, compliant lyric set. "
                "Do not reuse the rejected lyrics."
            )
            logger.info(
                "Lyrics generator: retrying after guardrail rejection ({}/{})",
                attempt,
                MAX_GENERATED_LYRICS_ATTEMPTS,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the single Chat LLM used by this app for music prompts and lyrics. "
                    "Return valid JSON only. Do not wrap it in markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a music generation spec from this request.\n"
                    f"Request: {prompt}\n"
                    f"Instrumental: {'true' if instrumental else 'false'}\n"
                    f"User metadata JSON: {json.dumps(metadata, ensure_ascii=False)}\n\n"
                    f"{section_instruction}\n"
                    "Return only a valid JSON object with these keys: caption, lyrics, bpm, "
                    "duration, keyscale, language, timesignature, instrumental.\n"
                    "- caption: complete detailed music caption. Never return the literal "
                    "'detailed music caption'.\n"
                    "- lyrics: for vocal songs, complete original lyrics with at least [verse] "
                    "and [chorus] sections. Never use ellipses, TBD, placeholders, or schema "
                    "example text. Use newline escapes inside the JSON string.\n"
                    "- bpm and duration: numbers or null.\n"
                    "- keyscale and timesignature: strings, empty if unknown.\n"
                    "- language: 'en'.\n"
                    "- instrumental: boolean."
                    f"{fresh_attempt_instruction}"
                    f"{retry_instruction}"
                ),
            },
        ]

        try:
            raw = await chat_llm.chat(
                messages,
                max_tokens=3072,
                temperature=0.85,
                allow_holders=allow_holders,
            )
            logger.info("Lyrics generator raw output:\n{}", raw)
            parsed = _json_from_text(raw)
        except Exception as exc:
            last_error = LyricsPipelineError(f"Chat LLM failed to generate lyrics: {exc}")
            if attempt == MAX_GENERATED_LYRICS_ATTEMPTS:
                raise last_error from exc
            continue

        try:
            caption = str(parsed.get("caption") or prompt).strip()
            lyrics = "" if instrumental else str(parsed.get("lyrics") or "").strip()
            logger.info("Lyrics generator parsed caption:\n{}", caption)
            if not instrumental:
                logger.info("Lyrics generator parsed lyrics before guardrails:\n{}", lyrics)
            if _looks_like_placeholder_caption(caption):
                raise LyricsPipelineError("Chat LLM returned placeholder caption.")
            if not instrumental and not lyrics:
                raise LyricsPipelineError("Chat LLM did not return lyrics for a vocal generation.")
            if not instrumental and _looks_like_placeholder_lyrics(lyrics):
                raise LyricsPipelineError("Chat LLM returned placeholder lyrics.")

            guarded = await review_lyrics_if_enabled(
                lyrics,
                caption=caption,
                allow_holders=allow_holders,
            )

            return {
                "caption": caption,
                "lyrics": guarded,
                "bpm": parsed.get("bpm"),
                "duration": parsed.get("duration"),
                "keyscale": str(parsed.get("keyscale") or ""),
                "language": "en",
                "timesignature": str(parsed.get("timesignature") or ""),
                "instrumental": instrumental,
                "success": True,
            }
        except LyricsPipelineError as exc:
            last_error = exc
            if attempt == MAX_GENERATED_LYRICS_ATTEMPTS:
                raise

    raise LyricsPipelineError("Chat LLM failed to generate compliant lyrics.")


async def format_song_spec(
    caption: str,
    lyrics: str,
    *,
    user_metadata: dict[str, Any] | None = None,
    allow_holders: frozenset[str] | None = None,
) -> dict[str, Any]:
    metadata = user_metadata or {}
    instrumental = coerce_instrumental(metadata.get("instrumental", False))
    messages = [
        {
            "role": "system",
            "content": (
                "You are the single Chat LLM used by this app to format music generation inputs. "
                "Return valid JSON only. Preserve user-provided musical intent."
            ),
        },
        {
            "role": "user",
            "content": (
                "Format this music generation request for ACE-Step.\n"
                f"Caption: {caption}\n"
                f"Lyrics: {lyrics}\n"
                f"Metadata JSON: {json.dumps(metadata, ensure_ascii=False)}\n\n"
                "Return only a valid JSON object with these keys: caption, lyrics, bpm, "
                "duration, keyscale, language, timesignature.\n"
                "- caption: complete detailed music caption. Never return the literal "
                "'detailed music caption'.\n"
                "- lyrics: preserve and format the provided lyrics. Never use ellipses, "
                "TBD, placeholders, or schema example text.\n"
                "- bpm and duration: numbers or null.\n"
                "- keyscale and timesignature: strings, empty if unknown.\n"
                "- language: 'en'."
            ),
        },
    ]

    try:
        raw = await chat_llm.chat(
            messages,
            max_tokens=3072,
            temperature=0.4,
            allow_holders=allow_holders,
        )
        logger.info("Lyrics formatter raw output:\n{}", raw)
        parsed = _json_from_text(raw)
    except Exception as exc:
        logger.warning(f"Chat LLM format failed: {exc}")
        return {
            "caption": caption,
            "lyrics": lyrics,
            "bpm": metadata.get("bpm"),
            "duration": metadata.get("duration"),
            "keyscale": metadata.get("keyscale", ""),
            "language": "en",
            "timesignature": metadata.get("timesignature", ""),
            "success": False,
            "error": str(exc),
        }

    formatted_caption = str(parsed.get("caption") or caption).strip()
    formatted_lyrics = "" if instrumental else str(parsed.get("lyrics") or lyrics).strip()
    if _looks_like_placeholder_caption(formatted_caption):
        formatted_caption = caption
    if formatted_lyrics and _looks_like_placeholder_lyrics(formatted_lyrics):
        return {
            "caption": formatted_caption,
            "lyrics": lyrics,
            "bpm": parsed.get("bpm") or metadata.get("bpm"),
            "duration": parsed.get("duration") or metadata.get("duration"),
            "keyscale": str(parsed.get("keyscale") or metadata.get("keyscale", "")),
            "language": "en",
            "timesignature": str(parsed.get("timesignature") or metadata.get("timesignature", "")),
            "success": False,
            "error": "Chat LLM returned placeholder lyrics.",
        }
    try:
        if formatted_lyrics:
            formatted_lyrics = await review_lyrics_if_enabled(
                formatted_lyrics,
                caption=formatted_caption,
                allow_holders=allow_holders,
            )
    except Exception as exc:
        return {
            "caption": formatted_caption,
            "lyrics": lyrics,
            "bpm": parsed.get("bpm") or metadata.get("bpm"),
            "duration": parsed.get("duration") or metadata.get("duration"),
            "keyscale": str(parsed.get("keyscale") or metadata.get("keyscale", "")),
            "language": "en",
            "timesignature": str(parsed.get("timesignature") or metadata.get("timesignature", "")),
            "success": False,
            "error": str(exc),
        }

    return {
        "caption": formatted_caption,
        "lyrics": formatted_lyrics,
        "bpm": parsed.get("bpm") or metadata.get("bpm"),
        "duration": parsed.get("duration") or metadata.get("duration"),
        "keyscale": str(parsed.get("keyscale") or metadata.get("keyscale", "")),
        "language": "en",
        "timesignature": str(parsed.get("timesignature") or metadata.get("timesignature", "")),
        "success": True,
        "error": None,
    }


async def prepare_generation_params(
    params: dict[str, Any],
    *,
    allow_holders: frozenset[str] | None = None,
) -> dict[str, Any]:
    prepared = dict(params)
    instrumental = coerce_instrumental(prepared.get("instrumental", False))
    prepared["instrumental"] = instrumental
    caption = str(prepared.get("caption") or "").strip()

    if instrumental:
        logger.info("Lyrics guardrails: skipped for instrumental request")
        prepared["lyrics"] = ""
        return mark_lyrics_pipeline_prepared(prepared)

    lyrics = str(prepared.get("lyrics") or "").strip()
    if not lyrics:
        spec = await generate_song_spec(
            caption or "Write an original vocal song.",
            instrumental=False,
            user_metadata={
                "bpm": prepared.get("bpm"),
                "keyscale": prepared.get("keyscale"),
                "timesignature": prepared.get("timesignature"),
                "duration": prepared.get("duration"),
            },
            allow_holders=allow_holders,
        )
        prepared["caption"] = spec.get("caption") or caption
        prepared["lyrics"] = spec["lyrics"]
        for key in ("bpm", "keyscale", "timesignature"):
            if not prepared.get(key) and spec.get(key):
                prepared[key] = spec[key]
        return mark_lyrics_pipeline_prepared(prepared)

    prepared["lyrics"] = await review_lyrics_if_enabled(
        lyrics,
        caption=caption,
        allow_holders=allow_holders,
    )
    return mark_lyrics_pipeline_prepared(prepared)
