"""Shot/cut detection: sensitive candidate detection + VLM verification.

Getting the SHOT COUNT right is critical. Two failure modes:
  - Over-counting: fast motion / strobing lights fool a naive detector into
    reporting cuts that aren't there (ContentDetector reports 6 on a clip that is
    really 1 continuous handheld take).
  - Under-counting: a robust detector might smooth over a real quick cut.

Strategy: collect candidate boundaries from BOTH a motion-robust detector
(AdaptiveDetector) and a sensitive one (ContentDetector) so nothing is missed,
then have the vision model look at the frames straddling each candidate and
confirm whether it is a genuine shot change (and, if so, its cut type).

Short shots are structurally preserved. The previous version merged candidates
within 0.4 s, discarded anything within 0.3 s of either end, and de-duplicated
confirmed cuts within 0.3 s — so a genuine 0.25 s shot, a fast montage, or a
brief opening title could never survive no matter what the footage showed.
Clustering now happens at frame resolution only, which removes duplicate reports
of the SAME boundary from different detectors without ever erasing two distinct
boundaries that are genuinely close together.

Picture boundaries and audio-over-picture transitions (L-cut / J-cut) are
separate questions and are resolved from separate evidence: the vision model
only ever sees images, so it is never asked to name a transition that is defined
by sound. When measured audio runs across a confirmed picture boundary, the
boundary is flagged UNRESOLVED for a human rather than silently called a hard cut.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from .audio_timeline import AudioSpan
from .blockers import WARNING, BlockerLog
from .frames import GridFrame, extract_indices
from .vision import OpenAIVisionBackend, image_content, text_content

#: Transitions decided from PICTURE evidence alone — what the VLM may choose.
PICTURE_CUT_TYPES = (
    "Hard cut", "Cross dissolve", "Fade in", "Fade out",
    "Match cut", "Jump cut", "Smash cut", "Wipe", "Iris",
    "Whip pan", "Swish pan",
)
#: Transitions defined by audio crossing a picture boundary. Never selectable
#: from frames; only a human (or audio evidence + review) may assign these.
AUDIO_CUT_TYPES = ("L-cut", "J-cut")

CUT_TYPES = ("Opening shot", *PICTURE_CUT_TYPES, *AUDIO_CUT_TYPES)

#: Fallback used only when the real frame period is unknown (~2 frames at 25fps).
FRAME_EPSILON = 0.08


def frame_epsilon(frame_times: list[float]) -> float:
    """The smallest gap that can still be TWO boundaries rather than one.

    ``frame_times`` MUST be the full encoded-frame ledger, not the sampled
    grid. The sampled grid only contains the frames chosen for the vision model
    (~10 Hz), so its smallest gap reflects the *sampling* rate, not the frame
    rate: on 25 fps footage sampled at 10 Hz it yields 0.072 s, which still
    erases the one-frame (0.040 s) shots this function exists to protect.

    Two candidates closer together than a single encoded frame cannot be
    distinct boundaries; anything a frame apart or more is a real shot.
    """
    times = sorted(set(frame_times))
    if len(times) < 3:
        return FRAME_EPSILON
    gaps = sorted(b - a for a, b in pairwise(times) if b > a)
    if not gaps:
        return FRAME_EPSILON
    # Use a low percentile rather than the strict minimum so one anomalous
    # short gap (container jitter, a duplicated timestamp) cannot collapse the
    # threshold to near zero.
    period = gaps[max(0, len(gaps) // 20)]
    if not 0.0 < period < 0.5:
        return FRAME_EPSILON
    return max(period * 0.9, 0.005)


#: PySceneDetect defaults every detector to min_scene_len=15 FRAMES — 0.6 s at
#: 25 fps. That floor silently makes short shots undetectable at the source:
#: a genuine one-frame white flash produces NO candidate at all, so no amount of
#: careful de-duplication downstream can recover it. Detection must be allowed
#: to propose short shots; rejecting them is the verifier's job, not the
#: detector's.
MIN_SCENE_LEN = 1


def _adaptive(video: Path) -> list[float]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(video), AdaptiveDetector(min_scene_len=MIN_SCENE_LEN))
    return [round(s.seconds, 3) for i, (s, _e) in enumerate(scenes) if i > 0]


def _content(video: Path, threshold: float = 27.0) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(
        str(video), ContentDetector(threshold=threshold, min_scene_len=MIN_SCENE_LEN)
    )
    return [round(s.seconds, 3) for i, (s, _e) in enumerate(scenes) if i > 0]


def _fades(video: Path) -> list[float]:
    """Fade to/from black boundaries (ThresholdDetector) — gradual transitions
    that content detectors and the motion-robust detector both miss."""
    from scenedetect import ThresholdDetector, detect

    scenes = detect(str(video), ThresholdDetector(min_scene_len=MIN_SCENE_LEN))
    return [round(s.seconds, 3) for i, (s, _e) in enumerate(scenes) if i > 0]


def candidate_boundaries(video: Path, min_gap: float = FRAME_EPSILON) -> list[float]:
    """Union of robust + sensitive + extra-sensitive + fade detectors.

    The extra-sensitive pass (threshold 12) exists for GRADUAL masked
    transitions — an expanding circular iris changes only ~15% of the frame per
    step and slips under the default threshold entirely (verified miss: a real
    Iris at 3.2s produced no candidate).

    ``min_gap`` defaults to one frame period, NOT a shot-length heuristic: its
    only job is to collapse the same boundary reported by several detectors.
    Over-candidates are cheap — every candidate is VLM-verified before it
    becomes a shot — whereas a dropped candidate is a silently deleted shot.
    """
    raw = sorted(set(
        _adaptive(video) + _content(video) + _content(video, threshold=12.0) + _fades(video)
    ))
    merged: list[float] = []
    for t in raw:
        if not merged or t - merged[-1] > min_gap:
            merged.append(t)
    return merged


def _near(
    grid: list[GridFrame], t: float, *, before: bool, n: int = 3, span: float = 0.4
) -> list[GridFrame]:
    if before:
        fs = [f for f in grid if t - span <= f.time_seconds < t]
        return fs[-n:]
    fs = [f for f in grid if t <= f.time_seconds <= t + span]
    return fs[:n]


def densify(
    video: Path,
    grid: list[GridFrame],
    boundaries: list[float],
    frame_times: list[float],
    work_dir: Path,
    radius: int = 3,
    blockers: BlockerLog | None = None,
) -> list[GridFrame]:
    """Grid plus the ACTUAL source frames straddling each candidate boundary.

    A ~10 Hz review grid cannot contain a one-frame event on 25 fps footage, so
    verifying such a candidate against the grid means showing the model frames
    that never contained the shot — and it correctly answers "not a cut". The
    neighbouring encoded frames are pulled on demand so the verifier sees what
    the detector saw.
    """
    if not frame_times or not boundaries:
        return grid
    have = {f.source_index for f in grid if f.source_index >= 0}
    wanted: set[int] = set()
    for t in boundaries:
        nearest = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - t))
        for i in range(nearest - radius, nearest + radius + 1):
            if 0 <= i < len(frame_times) and i not in have:
                wanted.add(i)
    if not wanted:
        return grid
    extra = extract_indices(
        video, work_dir, sorted(wanted), frame_times, blockers=blockers,
    )
    return sorted([*grid, *extra], key=lambda f: f.time_seconds)


_VERIFY = (
    "These frames straddle a POSSIBLE shot boundary in a video. The first images are the "
    "last frames before it; the later images are the first frames after it.\n"
    "Decide: is this a GENUINE shot boundary?\n"
    "It IS a boundary when the content changes to a different shot. This includes:\n"
    "- an instant change of scene, framing, or subjects (hard cut, jump cut);\n"
    "- a GRADUAL transition: fade to/from black or white, cross dissolve, wipe, or a "
    "MASKED reveal — e.g. a circular window (iris) or sliding panel that expands until "
    "new footage replaces the old (label it Iris for a circular mask, Wipe for a "
    "sliding edge/panel);\n"
    "- live footage replaced by (or emerging from) a GRAPHIC: a logo, title card, "
    "full-screen animation, scoreboard or branding screen. A footage-to-graphic "
    "transition is ALWAYS a boundary, even when it happens gradually.\n"
    "A boundary may be VERY brief — two boundaries a fraction of a second apart are "
    "normal in a fast montage. Never reject a boundary for being close to another one.\n"
    "It is NOT a boundary when it is the same continuous shot with only camera or "
    "subject motion (pans, fast movement of the same people/scene) or lighting/strobe "
    "changes over the same scene.\n"
    "If it is a boundary, also give the cut type from: Hard cut, Cross dissolve, "
    "Fade in, Fade out, Match cut, Jump cut, Smash cut, Wipe, Iris, Whip pan, Swish pan "
    "(use 'Hard cut' if unsure). You are seeing IMAGES ONLY, so never answer L-cut or "
    "J-cut — those are defined by sound and are decided elsewhere.\n"
    "If the frames genuinely do not let you decide, answer "
    '{"is_cut": null} rather than guessing.\n'
    'Return JSON {"is_cut": true|false|null, "cut": "<type>"}.'
)


def _verify(
    backend: OpenAIVisionBackend, before: list[GridFrame], after: list[GridFrame],
    blockers: BlockerLog | None = None, at: float | None = None,
) -> tuple[bool | None, str]:
    """(is_cut, cut_type). ``None`` means UNRESOLVED — never silently 'not a cut'.

    A model/network error used to return False here, which quietly reduced the
    shot count and produced a caption that looked complete.
    """
    if not before or not after:
        if blockers is not None:
            blockers.add(
                "CUT_UNVERIFIABLE",
                "No frames available on one side of a candidate boundary.",
                start=at,
            )
        return None, "Hard cut"
    content: list[dict[str, object]] = [text_content(_VERIFY), text_content("Before:")]
    content += [image_content(f.path) for f in before]
    content.append(text_content("After:"))
    content += [image_content(f.path) for f in after]
    try:
        data = json.loads(backend.complete(content, json_mode=True, max_tokens=60))
    except Exception as exc:
        if blockers is not None:
            blockers.add_exception("CUT_VERIFY_FAILED", exc, start=at)
        return None, "Hard cut"
    value = data.get("is_cut")
    if value is None:
        if blockers is not None:
            blockers.add(
                "CUT_UNDECIDED",
                "The vision model could not decide whether this is a shot boundary.",
                start=at,
            )
        return None, "Hard cut"
    cut = str(data.get("cut", "Hard cut")).strip()
    return bool(value), (cut if cut in PICTURE_CUT_TYPES else "Hard cut")


def snap_span(candidates: list[float], index: int, epsilon: float,
              default: float = 0.45) -> float:
    """How far a boundary may be snapped without swallowing its neighbour.

    Snapping searches for the biggest pixel change near a candidate. With a
    fixed ±0.45 s window, the EXIT of a one-frame shot at 1.04 s sees the
    entry's red-to-white change at 1.00 s, snaps backward onto it, and the two
    boundaries then de-duplicate into one — deleting the very shot the rest of
    this module works to preserve. A boundary may never snap onto or past an
    adjacent candidate.
    """
    span = default
    t = candidates[index]
    if index > 0:
        span = min(span, max((t - candidates[index - 1]) / 2, epsilon))
    if index + 1 < len(candidates):
        span = min(span, max((candidates[index + 1] - t) / 2, epsilon))
    return span


def snap_boundary(grid: list[GridFrame], t: float, span: float = 0.45) -> float:
    """Snap a confirmed boundary to the FIRST CHANGED grid frame near ``t``.

    Verified failure: the detector reported an iris at 3.1s, but frame 3.1s is
    still pure collage — the first changed frame is 3.167s. Strategy:
    inter-frame pixel differences in a small window; baseline noise = median;
    the boundary is the first frame whose difference clearly exceeds the
    baseline (fallback: the largest single change)."""
    try:
        import cv2  # bundled with scenedetect's opencv dependency
    except ImportError:
        return t
    fs = [f for f in grid if t - span <= f.time_seconds <= t + span]
    if len(fs) < 4:
        return t
    loaded = [cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE) for f in fs]
    if any(img is None for img in loaded):
        return t
    imgs = [img for img in loaded if img is not None]
    diffs = [float(cv2.absdiff(a, b).mean()) for a, b in pairwise(imgs)]
    med = sorted(diffs)[len(diffs) // 2]
    mx = max(diffs)
    if mx < max(4.0 * med, 2.0):
        return t  # no decisive single-frame spike — gradual transition, keep t
    return fs[diffs.index(mx) + 1].time_seconds


_GRADUAL = {"Iris", "Wipe", "Cross dissolve", "Fade in", "Fade out", "Whip pan", "Swish pan"}

_SNAP_PROMPT = (
    "These are consecutive labelled frames straddling a {ctype} transition in a video. "
    "Identify the FIRST frame in which the incoming transition is actually visible: the "
    "first appearance of the circular mask for an Iris, the first sliding edge or panel "
    "of a Wipe, the first blended double-image of a Cross dissolve, the first darkened/"
    "brightened frame of a Fade, or the first strongly blurred frame of a Whip pan. "
    'Return STRICT JSON: {{"time": <the exact labelled time in seconds of that frame>}}.'
)


def snap_gradual(
    backend: OpenAIVisionBackend, grid: list[GridFrame], t: float, ctype: str,
    span: float = 0.4,
) -> float:
    """Verified failure: an iris was stamped 3.1s but the first mask pixel
    appears at 3.167s. Pixel differencing cannot separate a gradual mask onset
    from in-shot motion, so the model picks the first frame showing the incoming
    transition from a labelled strip."""
    from .vision import image_content, text_content

    fs = [f for f in grid if t - span <= f.time_seconds <= t + span]
    if len(fs) < 3:
        return t
    content: list[dict[str, object]] = [text_content(_SNAP_PROMPT.format(ctype=ctype))]
    for f in fs:
        content.append(text_content(f"t={f.time_seconds:.2f}s:"))
        content.append(image_content(f.path))
    try:
        data = json.loads(backend.complete(content, json_mode=True, max_tokens=40))
        tt = float(data["time"])
        if t - span - 1e-6 <= tt <= t + span + 1e-6:
            return tt
    except Exception:
        pass
    return t


def audio_crosses(spans: list[AudioSpan], t: float, margin: float = 0.3) -> bool:
    """Does a continuous audible span run across the picture boundary at ``t``?

    An L-cut (outgoing audio continues under the incoming picture) and a J-cut
    (incoming audio starts under the outgoing picture) both look like this from
    the timeline alone. Which one it is depends on which SCENE the sound belongs
    to, which the signal cannot answer — so this only reports that the question
    exists, and the boundary is marked unresolved for a human.
    """
    for sp in spans:
        if sp.label == "quiet":
            continue
        if sp.start <= t - margin and sp.end >= t + margin:
            return True
    return False


def resolve_shots(
    backend: OpenAIVisionBackend,
    video: Path,
    grid: list[GridFrame],
    duration: float,
    blockers: BlockerLog | None = None,
    audio_spans: list[AudioSpan] | None = None,
    frame_times: list[float] | None = None,
    work_dir: Path | None = None,
) -> list[tuple[float, float, str]]:
    """Return verified shots as (start, end, cut_type). Always >= 1 shot.

    Candidates that could not be verified are NOT silently dropped: each one
    becomes a blocking unresolved item, because "we could not tell" and "there
    is no cut here" are different facts and only one of them is safe to render.
    """
    # The full ledger when available; the sampled grid only as a last resort,
    # and then it is recorded because boundary resolution is degraded.
    if frame_times:
        epsilon = frame_epsilon(frame_times)
    else:
        epsilon = frame_epsilon([f.time_seconds for f in grid])
        if blockers is not None:
            blockers.add(
                "BOUNDARY_RESOLUTION_DEGRADED",
                f"The encoded-frame ledger was unavailable, so boundary resolution "
                f"fell back to the sampled grid ({epsilon:.3f}s). Shots shorter than "
                f"that cannot be detected.",
                severity=WARNING,
            )
    candidates = candidate_boundaries(video, min_gap=epsilon)
    # Verify against the real frames straddling each candidate, not against a
    # sampled grid that may never have contained the shot.
    verify_grid = grid
    if frame_times and work_dir is not None:
        verify_grid = densify(
            video, grid, candidates, frame_times, work_dir / "dense",
            blockers=blockers,
        )

    confirmed: list[tuple[float, str]] = []
    for candidate_index, t in enumerate(candidates):
        # Only a boundary at or past the very first/last frame is meaningless.
        # A real 0.2 s opening title is a shot and must survive.
        if t <= epsilon or t >= duration - epsilon:
            continue
        is_cut, ctype = _verify(
            backend, _near(verify_grid, t, before=True),
            _near(verify_grid, t, before=False),
            blockers=blockers, at=t,
        )
        if is_cut is None:
            if blockers is not None:
                blockers.add(
                    "SHOT_BOUNDARY_UNRESOLVED",
                    f"A candidate shot boundary at {t:.2f}s could not be verified. The "
                    "shot list below may be missing a shot.",
                    start=t,
                )
            continue
        if not is_cut:
            continue
        # Snap against the dense frames too — snapping to a sampled grid would
        # re-round a boundary the dense pass just located precisely — and never
        # further than half-way to the neighbouring candidate.
        allowed = snap_span(candidates, candidate_index, epsilon)
        snapped = (snap_gradual(backend, verify_grid, t, ctype, span=allowed)
                   if ctype in _GRADUAL
                   else snap_boundary(verify_grid, t, span=allowed))
        if epsilon < snapped < duration - epsilon:
            confirmed.append((round(snapped, 3), ctype))

    confirmed = sorted(set(confirmed))
    # De-duplicate only true duplicates of the SAME boundary (frame resolution).
    # Candidates were already separated by >= epsilon before verification, so
    # two confirmed boundaries closer than that can ONLY be the result of
    # snapping having pulled them together — i.e. a verified cut is about to be
    # discarded. That is a lost shot and must never happen silently.
    deduped: list[tuple[float, str]] = []
    for c in confirmed:
        if deduped and c[0] - deduped[-1][0] < epsilon:
            if blockers is not None:
                blockers.add(
                    "SHOT_BOUNDARY_COLLAPSED",
                    f"A verified boundary at {c[0]:.3f}s was dropped because snapping "
                    f"moved it within {epsilon:.3f}s of the boundary at "
                    f"{deduped[-1][0]:.3f}s. A short shot may have been lost here.",
                    start=c[0],
                )
            continue
        deduped.append(c)

    if audio_spans:
        for t, ctype in deduped:
            if audio_crosses(audio_spans, t) and blockers is not None:
                blockers.add(
                    "TRANSITION_TYPE_UNRESOLVED",
                    f"Audio runs continuously across the picture boundary at {t:.2f}s, "
                    f"which is the signature of an L-cut or J-cut. Frames alone cannot "
                    f"tell them apart; rendered as '{ctype}' pending human confirmation.",
                    start=t,
                )

    marks = [0.0, *[t for t, _ in deduped], round(duration, 3)]
    shots: list[tuple[float, float, str]] = []
    for i in range(len(marks) - 1):
        cut = "Opening shot" if i == 0 else deduped[i - 1][1]
        shots.append((marks[i], marks[i + 1], cut))
    if blockers is not None:
        for s, e, _c in shots:
            if e - s < 0.4:
                blockers.add(
                    "SHORT_SHOT",
                    f"Shot of {e - s:.2f}s at {s:.2f}s is very short — confirm it is a "
                    "real shot and not a duplicated boundary.",
                    severity=WARNING, start=s, end=e,
                )
    return shots
