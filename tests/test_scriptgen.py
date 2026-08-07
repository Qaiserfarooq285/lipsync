import pytest

from core.scriptgen import chunk_for_tts, estimate_duration, normalize_for_speech


def test_normalize_strips_markdown():
    raw = "# Heading\n\nThis is **bold** and *italic* and `code`.\n\n- bullet one\n- bullet two"
    text = normalize_for_speech(raw)
    assert "#" not in text
    assert "**" not in text
    assert "`" not in text
    assert "bold" in text and "italic" in text
    assert "bullet one" in text


def test_normalize_folds_smart_punctuation():
    raw = "It’s a “test” — really…"
    text = normalize_for_speech(raw)
    assert "’" not in text and "“" not in text and "—" not in text
    assert "It's" in text


def test_normalize_strips_stage_directions():
    raw = "Hello there. [pause] This is fine. (laughs) Great."
    text = normalize_for_speech(raw)
    assert "pause" not in text.lower()
    assert "laughs" not in text.lower()


def test_chunk_for_tts_respects_max_chars():
    text = " ".join(["This is sentence number %d." % i for i in range(30)])
    chunks = chunk_for_tts(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).replace("  ", " ").strip() != ""


def test_chunk_for_tts_short_text_single_chunk():
    assert chunk_for_tts("Hello world.", max_chars=280) == ["Hello world."]


def test_chunk_for_tts_empty():
    assert chunk_for_tts("", max_chars=280) == []


def test_chunk_for_tts_splits_very_long_sentence():
    long_sentence = "word " * 200 + "."
    chunks = chunk_for_tts(long_sentence.strip(), max_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_estimate_duration_scales_with_words():
    short = estimate_duration("one two three")
    long = estimate_duration(" ".join(["word"] * 150))
    assert long > short
    assert estimate_duration(" ".join(["word"] * 150), wpm=150) == pytest.approx(60.0)
