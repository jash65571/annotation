"""Reviewer honesty and web-app guardrails.

The reviewer pass previously returned the unreviewed machine caption as the
"final RTD caption" whenever the model failed, and wrote whatever the model
produced straight to disk without revalidating it.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoscribe import review as review_mod
from autoscribe import webapp
from autoscribe.blockers import BlockerLog

FRESH = """[Overview]
Cast:
C1: A person in a red jacket.

Scene: A domestic kitchen shot along its length. In the foreground a rectangular \
oak table with turned legs occupies the lower third, its surface unvarnished. In \
the middle ground C1 stands behind the table facing camera, with open pine \
shelving on screen-left and a steel refrigerator on screen-right. The background \
is a plastered wall with a deep sash window above the counter run, beyond which a \
brick garden wall is visible.

Style: Daylight key from the sash window at screen-left with soft overhead fill; \
shadows are soft-edged and shallow. Colour temperature is cool toward the window \
and warmer near the shelving. Shallow depth of field, digital capture, no \
non-standard aspect ratio.

Audio: Music throughout.

Visual Concerns: None.
Audio Concerns: None.

[Shot 1: 0.0s–2.0s]
Cut: Opening shot.
Camera: Medium, eye-level, static.
Scene: No changes from overview.
Action & Audio:
(0.0s–1.0s) C1 raises the right hand.
Playback Speed: regular.
"""


class _FailingBackend:
    def complete(self, *_a: Any, **_k: Any) -> str:
        raise ConnectionError("model unavailable")


class _EmptyBackend:
    def complete(self, *_a: Any, **_k: Any) -> str:
        return ""


class _BadCaptionBackend:
    """Returns a caption containing defects the reviewer must not launder."""

    def complete(self, *_a: Any, **_k: Any) -> str:
        import json

        broken = FRESH.replace(
            "(0.0s–1.0s) C1 raises the right hand.",
            '(0.0s–9.0s) C4 raises his right hand',
        )
        return json.dumps({
            "verdict": "FIX / ENRICH", "score": 3, "score_reason": "ok",
            "issues": [], "unresolved": [], "feedback": "-",
            "final_caption": broken,
        })


def test_review_failure_is_recorded_not_laundered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _FailingBackend())
    log = BlockerLog()
    result = review_mod.review(FRESH, "seed caption text", blockers=log)
    assert result["final_caption"] == FRESH, "fresh draft should be preserved"
    assert any(b.code == "REVIEW_FAILED" for b in log.blocking), (
        "a failed review must not be presented as a completed review"
    )
    assert result["ready"] is False


def test_empty_model_response_is_a_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _EmptyBackend())
    result = review_mod.review(FRESH, "seed")
    assert result["ready"] is False
    assert any(b["code"] == "REVIEW_FAILED" for b in result["blockers"])


def test_reviewer_output_is_revalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model-introduced ghost ID / pronoun / bad range must be caught."""
    monkeypatch.setattr(
        review_mod, "OpenAIVisionBackend", lambda *a, **k: _BadCaptionBackend()
    )
    result = review_mod.review(FRESH, "seed")
    codes = {b["code"] for b in result["blockers"]}
    assert "GHOST_ID" in codes
    assert "PRONOUN_OUTSIDE_QUOTES" in codes
    assert "TIMESTAMP_OUTSIDE_SHOT" in codes
    assert result["ready"] is False


def test_reviewer_receives_frames_when_available(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source-of-truth §1 ranks the media above every caption. A reviewer given
    only prose cannot apply that rule at all."""
    import json

    seen: dict[str, Any] = {}

    class _Recording:
        def complete(self, content: list[dict[str, Any]], **_k: Any) -> str:
            seen["types"] = [c.get("type") for c in content]
            return json.dumps({
                "verdict": "KEEP", "score": 5, "score_reason": "-",
                "issues": [], "unresolved": [], "feedback": "-",
                "final_caption": FRESH,
            })

    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _Recording())
    frame = tmp_path / "f.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = review_mod.review(FRESH, "seed", frames=[(4.25, frame)])

    assert "image_url" in seen["types"], "frames were not sent to the reviewer"
    assert not any(b["code"] == "REVIEW_WITHOUT_PICTURE" for b in result["blockers"])


def test_reviewer_frames_are_labelled_with_timestamps(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlabelled still can settle 'is there a red jacket?' but not 'does C1
    raise a hand at 4.2s' — which is most of what a caption review disputes."""
    import json

    seen: dict[str, Any] = {}

    class _Recording:
        def complete(self, content: list[dict[str, Any]], **_k: Any) -> str:
            seen["text"] = " ".join(
                str(c.get("text", "")) for c in content if c.get("type") == "text"
            )
            return json.dumps({
                "verdict": "KEEP", "score": 5, "score_reason": "-",
                "issues": [], "unresolved": [], "feedback": "-",
                "final_caption": FRESH,
            })

    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _Recording())
    frame = tmp_path / "f.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")

    review_mod.review(FRESH, "seed", frames=[(4.25, frame), (9.5, frame)])

    assert "t=4.25s" in seen["text"], "frames carry no timestamp label"
    assert "t=9.50s" in seen["text"]


def test_review_without_frames_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    class _Clean:
        def complete(self, *_a: Any, **_k: Any) -> str:
            return json.dumps({
                "verdict": "KEEP", "score": 5, "score_reason": "-",
                "issues": [], "unresolved": [], "feedback": "-",
                "final_caption": FRESH,
            })

    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _Clean())
    result = review_mod.review(FRESH, "seed")
    assert any(b["code"] == "REVIEW_WITHOUT_PICTURE" for b in result["blockers"])


def test_reviewer_rewrite_keeps_the_language_evidence_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The draft was validated with speech_present/language_confident, but the
    reviewer's rewrite was not — so a reviewer could delete the required
    'a foreign language' declaration and final validation recorded nothing."""
    import json

    stripped = FRESH.replace(
        "Audio: Music throughout.", "Audio: Music throughout."
    )

    class _Stripping:
        def complete(self, *_a: Any, **_k: Any) -> str:
            return json.dumps({
                "verdict": "KEEP", "score": 5, "score_reason": "-",
                "issues": [], "unresolved": [], "feedback": "-",
                "final_caption": stripped,
            })

    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _Stripping())
    result = review_mod.review(
        FRESH, "seed", speech_present=True, language_confident=False,
    )
    assert any(b["code"] == "LANGUAGE_NOT_DECLARED" for b in result["blockers"]), (
        "the rewrite was revalidated without the evidence context"
    )


def test_review_is_never_ready_even_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    class _CleanBackend:
        def complete(self, *_a: Any, **_k: Any) -> str:
            return json.dumps({
                "verdict": "KEEP", "score": 5, "score_reason": "clean",
                "issues": [], "unresolved": [], "feedback": "-",
                "final_caption": FRESH,
            })

    monkeypatch.setattr(review_mod, "OpenAIVisionBackend", lambda *a, **k: _CleanBackend())
    result = review_mod.review(FRESH, "seed")
    assert result["ready"] is False, "RTD requires a human, not a clean validator run"
    assert "human" in result["readiness_reason"].lower()


# --------------------------------------------------------------------------
# web app
# --------------------------------------------------------------------------
def test_privacy_notice_discloses_cloud_upload_in_structured_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOSCRIBE_MODE", "structured")
    notice = webapp.privacy_notice()
    assert "CLOUD" in notice
    assert "OpenAI" in notice


def test_privacy_notice_reports_local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOSCRIBE_MODE", "flat")
    monkeypatch.setenv("AUTOSCRIBE_VISION", "ollama")
    assert "LOCAL MODE" in webapp.privacy_notice()


def test_upload_and_rate_limits_are_bounded() -> None:
    assert webapp.MAX_UPLOAD_BYTES > 0
    assert webapp.MIN_HZ > 0 and webapp.MAX_HZ >= webapp.MIN_HZ
    assert webapp.MAX_CONCURRENT_JOBS >= 1


def test_cleanup_removes_every_artefact_except_declared_results(
    tmp_path: Any,
) -> None:
    """Deny-listing failed twice: by directory name (`dense/` escaped) and by
    suffix (the upload name is user-controlled, so `upload.txt` or a bare
    `upload` survived while FFmpeg happily processed their MP4 content)."""
    workspace = tmp_path / "job"
    clip_dir = workspace / "out" / "clip"
    frames_dir = clip_dir / "frames"
    dense_dir = clip_dir / "dense"
    for d in (frames_dir, dense_dir):
        d.mkdir(parents=True)
    (frames_dir / "g000000.png").write_bytes(b"x")
    (dense_dir / "d000000.png").write_bytes(b"x")
    (clip_dir / "audio.wav").write_bytes(b"x")
    # Names an attacker or a careless user controls.
    disguised = [
        workspace / "upload.mp4",
        workspace / "upload.txt",
        workspace / "upload.ts",
        workspace / "upload",
    ]
    for p in disguised:
        p.write_bytes(b"x")
    caption = workspace / "out" / "clip.manuscript.md"
    caption.write_text("keep me")

    webapp._cleanup_workspace(workspace, keep={caption})

    assert not frames_dir.exists(), "extracted frames must not be left behind"
    assert not dense_dir.exists(), "boundary-adjacent frames must not survive"
    assert not (clip_dir / "audio.wav").exists(), "extracted audio must not survive"
    for p in disguised:
        assert not p.exists(), f"{p.name} survived cleanup"
    assert caption.read_text() == "keep me", "declared results must survive"


def test_cleanup_keeps_only_what_was_declared(tmp_path: Any) -> None:
    """An undeclared .md is an intermediate, not a result, and goes too."""
    workspace = tmp_path / "job"
    workspace.mkdir()
    declared = workspace / "final.md"
    declared.write_text("keep")
    stray = workspace / "scratch.md"
    stray.write_text("drop")

    webapp._cleanup_workspace(workspace, keep={declared})

    assert declared.exists()
    assert not stray.exists()


def test_cleanup_is_safe_on_a_missing_workspace(tmp_path: Any) -> None:
    webapp._cleanup_workspace(tmp_path / "never_created")


def test_unclear_token_defaults_to_current_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoscribe.structured import _unclear_token

    monkeypatch.delenv("AUTOSCRIBE_UNCLEAR_TOKEN", raising=False)
    assert _unclear_token() == "<unintelligible>"
    monkeypatch.setenv("AUTOSCRIBE_UNCLEAR_TOKEN", "[inaudible]")
    assert _unclear_token() == "[inaudible]"


def test_prompts_forbid_lip_reading_and_protected_traits() -> None:
    from autoscribe.structured import _CAST_PROMPT, _SHOT_PROMPT

    assert "NEVER LIP-READ" in _SHOT_PROMPT
    assert "lip-read — otherwise" not in _SHOT_PROMPT
    assert "PROTECTED TRAITS ARE FORBIDDEN" in _CAST_PROMPT
    assert "race/ethnicity" not in _CAST_PROMPT
    assert "CLOTHING LOCK" not in _CAST_PROMPT
    assert "[mid-sentence cut]" in _SHOT_PROMPT


def test_prompt_allows_genuinely_simultaneous_events() -> None:
    """The old rule 2 demanded every range be distinct, which forced the model
    to falsify timings for events that truly co-occur."""
    from autoscribe.structured import _SHOT_PROMPT

    assert "SEPARATE lines with the SAME true range" in _SHOT_PROMPT
    assert "Every timestamp range must be DISTINCT" not in _SHOT_PROMPT
