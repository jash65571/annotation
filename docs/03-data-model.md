# 03 — Data Model

All persisted structures are Pydantic v2 models (`extra="forbid"`). Exact time is a
`Fraction` serialized losslessly as `"num/den"` (`models/common.py::ExactFraction`).
Floats never carry authoritative time.

## Populated in Phase 1

| Model | File | Purpose |
|---|---|---|
| `MediaInfo` | `models/media.py` | Container-level facts + per-stream lists; declared values kept separate from measured ones so discrepancies stay visible |
| `VideoStreamInfo` | `models/media.py` | codec, profile, pix_fmt, geometry, SAR/DAR, nominal (`r_frame_rate`) vs average (`avg_frame_rate`) rate, exact `time_base`, start_pts, declared duration, declared `nb_frames` |
| `AudioStreamInfo` | `models/media.py` | codec, sample rate, channels/layout, time_base, start_pts, declared duration |
| `FrameRecord` | `models/frame.py` | frame_index, pts, exact pts_time, dts, duration (+exact seconds), key_frame, pict_type, width, height |
| `FrameLedger` | `models/frame.py` | stream_index + time_base + every FrameRecord in presentation order |
| `ValidatorIssue` | `models/validation.py` | rule_id (P1-*/M2-*), severity (FAIL/WARN/INFO), location, message, suggested fix |
| `FrameCountSignal` | `models/validation.py` | one independent frame-count measurement in the cross-check |
| `QCReport` | `models/validation.py` | status (PASS/PARTIAL/FAILED), issues, signals, checks run |
| `RunManifest` / `ArtifactEntry` | `models/run.py` | run_id, source SHA-256 (video + seed), app/rules/ffmpeg/ffprobe versions, start/end, stage timings, artifact hashes, validation status |
| `ProjectRun` | `models/run.py` | in-memory run parameters |

`FrameRecord` intentionally omits fields ffprobe cannot reliably provide; `pts` is
`None` only when the codec truly reports none (run becomes PARTIAL at best).

## Defined now, populated in later phases

| Model | Notes |
|---|---|
| `EvidenceReference` | typed evidence pointer: FRAME, FRAME_RANGE, FRAME_STRIP, AUDIO_RANGE, WAVEFORM_RANGE, SPECTROGRAM_RANGE, ASR_SEGMENT, OCR_RESULT, MODEL_OBSERVATION, HUMAN_VERIFICATION. `is_factual` excludes ASR/OCR/model output — **AI confidence alone never counts as factual evidence** |
| `Shot`, `Transition` | PTS-exact boundaries; transition types validated against the rule file |
| `Character`, `ObjectEntity` | C#/O# identity maps with first-appearance anchors |
| `CameraEvent` | timestamped movement phase, "reveals or hides" field |
| `ActionAudioEvent` | one sentence, one event, exact window, C/O references |
| `SpeechEvent` | verbatim text, speaker, tone, off-screen flag, `contains_inaudible` |
| `SoundEvent`, `OnScreenTextEvent` | timestamped audio/text events |
| `SeedClaim` | one hypothesis extracted from the seed, with KEEP/FIX_ENRICH/REDO_REBUILD outcome |
| `ConfidenceAssessment` | uncertainty tracking; `requires_human_verification` defaults to True |
| `ReviewDecision` | recorded human decision with evidence |
| `ReviewedCaption` | the assembled final caption structure |
| `ExactTimeRange` | shared exact window type (PTS + Fraction + frame anchors) |

## Evidence-first contract

A future statement like "C1 raises the right hand" is an `ActionAudioEvent` whose
`time_range` carries start/end frame + PTS anchors and whose `evidence` list points at
frame images, audio ranges, model observations, and the reviewer decision. Claims
without factual evidence types cannot pass the future validator.
