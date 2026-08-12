"""Real local ASR integration tests (marked; excluded from normal CI runs).

Run with:  uv run pytest -m asr_integration --no-header
Requires: bootstrapped worker envs + a local Whisper model (tiny is enough to
verify the plumbing; transcript exactness is NOT graded as universal accuracy).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from manuscript_reviewer.audio.asr.runtime import FW_ENV, WX_ENV, ASRConfig
from manuscript_reviewer.models.audio import AlignmentStatus, ASRStatus
from tests.conftest import requires_ffmpeg
from tests.test_audio_truth import _analyze

pytestmark = [pytest.mark.asr_integration, requires_ffmpeg]

TTS_VIDEO = Path(__file__).parent / "fixtures" / "tts_video.mp4"


@pytest.mark.skipif(not TTS_VIDEO.exists(), reason="local TTS fixture not generated")
@pytest.mark.skipif(not (FW_ENV / ".venv").exists(), reason="fw worker env not bootstrapped")
def test_real_faster_whisper_and_whisperx(tmp_path: Path) -> None:
    output = _analyze(
        TTS_VIDEO, tmp_path, with_shots=False, asr_enabled=True,
        asr_config=ASRConfig(model="tiny", device="cpu", compute_type="int8"),
    )
    qc = output.result
    assert qc is not None
    assert qc.asr_status == ASRStatus.PASS

    assert qc.speech_region_count >= 1
    region = qc.speech_regions[0]
    assert region.text_candidate is not None
    assert "fox" in region.text_candidate.lower()

    asr_dir = tmp_path / TTS_VIDEO.stem / "audio" / "asr"
    words_csv = (asr_dir / "words_best.csv").read_text(encoding="utf-8").splitlines()
    assert len(words_csv) > 5  # header + words
    starts = [float(line.split(",")[2]) for line in words_csv[1:]]
    assert all(b >= a for a, b in itertools.pairwise(starts)), "word timestamps unordered"

    runtime = (asr_dir / "runtime.json").read_text(encoding="utf-8")
    assert "faster_whisper" in runtime and "tiny" in runtime

    if (WX_ENV / ".venv").exists():
        assert qc.alignment_status in (
            AlignmentStatus.ALIGNED,
            AlignmentStatus.TEXT_MISMATCH,
            AlignmentStatus.FAILED,
        )
        if qc.alignment_status == AlignmentStatus.ALIGNED:
            assert "whisperx" in runtime
