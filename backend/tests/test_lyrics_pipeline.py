import pytest


def test_extract_guarded_lyrics_accepts_tagged_output():
    from bangers.services.lyrics_pipeline import extract_guarded_lyrics

    raw = "<guarded_lyrics>\n[verse]\nClean line\n</guarded_lyrics>"

    assert extract_guarded_lyrics(raw) == "[verse]\nClean line"


def test_extract_guarded_lyrics_rejects_malformed_output():
    from bangers.services.lyrics_pipeline import LyricsPipelineError, extract_guarded_lyrics

    with pytest.raises(LyricsPipelineError):
        extract_guarded_lyrics("[verse]\nNo wrapper")


def test_extract_guarded_lyrics_rejects_extra_text():
    from bangers.services.lyrics_pipeline import LyricsPipelineError, extract_guarded_lyrics

    raw = "Here are the revised lyrics:\n<guarded_lyrics>\n[verse]\nClean line\n</guarded_lyrics>"

    with pytest.raises(LyricsPipelineError):
        extract_guarded_lyrics(raw)


def test_extract_guarded_lyrics_rejects_placeholder_output():
    from bangers.services.lyrics_pipeline import LyricsPipelineError, extract_guarded_lyrics

    raw = "<guarded_lyrics>\n[verse]\n...\n</guarded_lyrics>"

    with pytest.raises(LyricsPipelineError):
        extract_guarded_lyrics(raw)


def test_extract_guarded_lyrics_detects_guardrail_refusal():
    from bangers.services.lyrics_pipeline import LyricsRejectedError, extract_guarded_lyrics

    raw = "<lyrics_rejected>\nSevere targeted abuse.\n</lyrics_rejected>"

    with pytest.raises(LyricsRejectedError, match="Code of Conduct"):
        extract_guarded_lyrics(raw)


@pytest.mark.asyncio
async def test_review_lyrics_logs_full_guardrail_model_output(monkeypatch):
    from bangers.services import lyrics_pipeline

    raw = "<guarded_lyrics>\n[verse]\nFull logged line\n</guarded_lyrics>"
    log_calls: list[tuple[str, tuple]] = []

    async def fake_chat(*_args, **_kwargs):
        return raw

    def fake_info(message, *args, **_kwargs):
        log_calls.append((message, args))

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)
    monkeypatch.setattr(lyrics_pipeline.logger, "info", fake_info)

    result = await lyrics_pipeline.review_lyrics("[verse]\nRaw line")

    assert result == "[verse]\nFull logged line"
    assert ("Lyrics guardrails input lyrics:\n{}", ("[verse]\nRaw line",)) in log_calls
    assert ("Lyrics guardrails model raw output:\n{}", (raw,)) in log_calls


@pytest.mark.asyncio
async def test_generate_song_spec_rejects_placeholder_llm_output(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_chat(*_args, **_kwargs):
        return (
            '{"caption":"detailed music caption","lyrics":"[verse]\\n...",'
            '"bpm":120,"duration":null,"keyscale":"","language":"en",'
            '"timesignature":"","instrumental":false}'
        )

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    with pytest.raises(lyrics_pipeline.LyricsPipelineError, match="placeholder caption|placeholder lyrics"):
        await lyrics_pipeline.generate_song_spec("blues vocal", instrumental=False)


@pytest.mark.asyncio
async def test_generate_song_spec_retries_generated_lyrics_after_guardrail_rejection(monkeypatch):
    from bangers.services import lyrics_pipeline

    outputs = [
        (
            '{"caption":"A bright pop song","lyrics":"[verse]\\nFirst rejected line",'
            '"bpm":120,"duration":null,"keyscale":"","language":"en",'
            '"timesignature":"","instrumental":false}'
        ),
        "<lyrics_rejected>\nToo unsafe to rewrite.\n</lyrics_rejected>",
        (
            '{"caption":"A bright pop song","lyrics":"[verse]\\nSecond clean line",'
            '"bpm":120,"duration":null,"keyscale":"","language":"en",'
            '"timesignature":"","instrumental":false}'
        ),
        "<guarded_lyrics>\n[verse]\nSecond clean line\n</guarded_lyrics>",
    ]

    async def fake_chat(*_args, **_kwargs):
        return outputs.pop(0)

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    result = await lyrics_pipeline.generate_song_spec("bright pop vocal", instrumental=False)

    assert result["lyrics"] == "[verse]\nSecond clean line"
    assert outputs == []


@pytest.mark.asyncio
async def test_format_song_spec_allows_empty_lyrics_for_instrumental(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_chat(*_args, **_kwargs):
        return (
            '{"caption":"Energetic synthwave instrumental","lyrics":"",'
            '"bpm":128,"duration":200,"keyscale":"","language":"en",'
            '"timesignature":"4/4"}'
        )

    async def fail_review(*_args, **_kwargs):
        raise AssertionError("instrumental lyrics should not be reviewed")

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)
    monkeypatch.setattr(lyrics_pipeline, "review_lyrics_if_enabled", fail_review)

    result = await lyrics_pipeline.format_song_spec(
        "synthwave instrumental",
        "",
        user_metadata={"instrumental": True},
    )

    assert result["success"] is True
    assert result["lyrics"] == ""
    assert result["error"] is None


@pytest.mark.asyncio
async def test_format_song_spec_allows_empty_lyrics_for_vocal_formatting(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_chat(*_args, **_kwargs):
        return (
            '{"caption":"Energetic synthwave vocal","lyrics":"",'
            '"bpm":128,"duration":200,"keyscale":"","language":"en",'
            '"timesignature":"4/4"}'
        )

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    result = await lyrics_pipeline.format_song_spec(
        "synthwave vocal",
        "",
        user_metadata={"instrumental": False},
    )

    assert result["success"] is True
    assert result["lyrics"] == ""
    assert result["error"] is None


@pytest.mark.asyncio
async def test_format_song_spec_rejects_placeholder_lyrics_text(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_chat(*_args, **_kwargs):
        return (
            '{"caption":"Energetic synthwave vocal","lyrics":"[verse]\\n...",'
            '"bpm":128,"duration":200,"keyscale":"","language":"en",'
            '"timesignature":"4/4"}'
        )

    monkeypatch.setattr(lyrics_pipeline.chat_llm, "chat", fake_chat)

    result = await lyrics_pipeline.format_song_spec(
        "synthwave vocal",
        "",
        user_metadata={"instrumental": False},
    )

    assert result["success"] is False
    assert result["error"] == "Chat LLM returned placeholder lyrics."


@pytest.mark.asyncio
async def test_prepare_generation_params_generates_missing_vocal_lyrics(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_generate_song_spec(*_args, **_kwargs):
        return {
            "caption": "A clean pop track",
            "lyrics": "[verse]\nClean generated line",
        }

    monkeypatch.setattr(lyrics_pipeline, "generate_song_spec", fake_generate_song_spec)

    result = await lyrics_pipeline.prepare_generation_params({
        "caption": "pop",
        "instrumental": False,
        "lyrics": "",
    })

    assert result["caption"] == "A clean pop track"
    assert result["lyrics"] == "[verse]\nClean generated line"


@pytest.mark.asyncio
async def test_prepare_generation_params_reviews_string_false_instrumental(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fake_review(lyrics: str, **_kwargs):
        assert lyrics == "[verse]\nNeeds review"
        return "[verse]\nReviewed lyric"

    monkeypatch.setattr(lyrics_pipeline, "review_lyrics_if_enabled", fake_review)

    result = await lyrics_pipeline.prepare_generation_params({
        "caption": "custom tab vocal song",
        "instrumental": "false",
        "lyrics": "[verse]\nNeeds review",
    })

    assert result["instrumental"] is False
    assert result["lyrics"] == "[verse]\nReviewed lyric"
    assert lyrics_pipeline.is_lyrics_pipeline_prepared(result) is True
    assert lyrics_pipeline.LYRICS_PIPELINE_PREPARED_KEY not in (
        lyrics_pipeline.strip_lyrics_pipeline_internal_keys(result)
    )


@pytest.mark.asyncio
async def test_prepare_generation_params_skips_instrumentals(monkeypatch):
    from bangers.services import lyrics_pipeline

    async def fail_review(*_args, **_kwargs):
        raise AssertionError("instrumental lyrics should not be reviewed")

    monkeypatch.setattr(lyrics_pipeline, "review_lyrics_if_enabled", fail_review)

    result = await lyrics_pipeline.prepare_generation_params({
        "caption": "ambient",
        "instrumental": True,
        "lyrics": "remove me",
    })

    assert result["lyrics"] == ""
