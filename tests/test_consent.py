from pathlib import Path

import core.consent as consent
from core.config import REPO_ROOT, config_from_dict


def test_selftest_assets_are_exempt():
    placeholder = REPO_ROOT / "assets" / "samples" / "placeholder_presenter.mp4"
    assert consent.is_exempt(placeholder)


def test_real_looking_path_is_not_exempt():
    assert not consent.is_exempt(REPO_ROOT / "assets" / "presenter" / "clip.mp4")


def test_consent_status_not_granted_by_default():
    granted, problems = consent.consent_status()
    assert granted is False
    assert problems


def test_assert_allowed_passes_for_placeholder_job():
    cfg = config_from_dict({
        "voice": {"reference_audio": "assets/samples/placeholder_voice.wav"},
        "video": {"presenter_clip": "assets/samples/placeholder_presenter.mp4"},
    })
    consent.assert_allowed(cfg)  # must not raise


def test_assert_allowed_blocks_real_media_without_consent():
    cfg = config_from_dict({
        "voice": {"reference_audio": "assets/presenter/voice.wav"},
        "video": {"presenter_clip": "assets/presenter/clip.mp4"},
    })
    try:
        consent.assert_allowed(cfg)
        raised = False
    except consent.ConsentError:
        raised = True
    assert raised


def test_granted_status_requires_filled_fields(tmp_path, monkeypatch):
    fake = tmp_path / "CONSENT.md"
    fake.write_text(
        "**STATUS: GRANTED**\n\n"
        "- **Presenter name:** `<full name>`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(consent, "CONSENT_FILE", fake)
    granted, problems = consent.consent_status()
    assert granted is False
    assert any("Presenter name" in p or "Consent obtained" in p for p in problems)
