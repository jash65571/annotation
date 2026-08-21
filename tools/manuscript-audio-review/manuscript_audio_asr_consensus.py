"""Multi-model ASR consensus (Phase 3A, hardened in 3A.1).

Compares the primary transcript (WhisperX large-v3 + forced alignment, run in
Phase 0 of the pipeline) against an independent secondary pass
(faster-whisper large-v3-turbo, run directly with no alignment) so that words
or whole late-speech regions missed by one model do not silently disappear.

3A.1 hardening (same evidence conclusions, much less CPU time):
- The secondary model is loaded ONCE in a persistent worker subprocess and
  reused for the full-clip pass and every targeted rerun (was: one full
  model reload per rerun).
- Rerun windows within ~0.5s of each other are merged into a single
  inference call; results are mapped back to each original source window so
  provenance is preserved.
- Tiny, uncorroborated secondary-only gaps no longer trigger an automatic
  rerun (alignment jitter was burning model time for no evidence gain). They
  stay fully visible in word_consensus / secondary_only_words either way --
  only the auto-rerun trigger got stricter, not recall.
- Added advisory hallucination_risk and proper_noun_risk signals, computed
  from evidence already collected (no new model).

Pure standard library except for the subprocess call into `.venv-whisperx`
(design rule 5: heavy ASR stays in its own worker; this module never imports
torch/faster-whisper itself, so it also runs fine under `.venv-review` or the
base interpreter). If the secondary pass cannot run (worker missing, no
ffmpeg, model load failure, timeout), every function here still returns a
well-formed `status: "unavailable"`/"failed" result -- the base packet must
keep generating (non-negotiable rule 4).

This module never turns a model disagreement into a transcript decision. It
only classifies each primary word's cross-model support and lists windows a
human should listen to.
"""

from pathlib import Path
import json
import queue
import re
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "manuscript_audio_asr_worker.py"

SECONDARY_MODEL = "large-v3-turbo"
LOW_CONFIDENCE_SCORE = 0.5
UNRECOVERABLE_SCORE = 0.15
MAX_AUTO_RERUNS = 6
RERUN_PADDING_SEC = 0.4
WORKER_READY_TIMEOUT_SEC = 180
WORKER_REQUEST_TIMEOUT_SEC = 180

# 3.5 hardening: cross-model matching is sequence-aware, not strict
# overlap-only. The same lexical word can drift by a few hundred ms between
# models, so a word pair is aligned when the normalized text matches and the
# center-times are within tolerance, even if the raw intervals do not touch.
# MATCH_CENTER_TOLERANCE_SEC is the primary 300-400ms window the audit asked
# for; MATCH_CENTER_DRIFT_SEC allows a wider (but still bounded) match for
# identical text so a single drifted word is not misread as a coverage gap.
MATCH_CENTER_TOLERANCE_SEC = 0.35
MATCH_CENTER_DRIFT_SEC = 0.9
MATCH_SAME_TEXT_SCORE = 3.0
MATCH_SAME_TEXT_DRIFTED_SCORE = 1.0
MATCH_DIFF_TEXT_PENALTY = -0.5
MATCH_FAR_PENALTY = -2.0
# Two gaps (-1.6) must always beat one far/apart match (-2.0) so unrelated
# words are never stretched into an alignment; a real match is still
# strongly preferred over gapping.
ALIGN_GAP_PENALTY = -0.8

# 3A.1: merge rerun candidates within this gap into one inference call.
MERGE_GAP_SEC = 0.5
# 3A.1: a secondary-only gap must clear one of these bars to earn an
# automatic rerun; otherwise it stays evidence-only (word_consensus /
# secondary_only_words), never silently dropped, just not auto-reran.
MIN_AUTO_RERUN_GAP_SEC = 0.5
CLIP_BOUNDARY_TRIGGER_SEC = 0.5

# Advisory-only risk thresholds (spec 3A.1-4/5). Never promoted past MEDIUM;
# these flag review candidates, they do not decide anything.
HALLUCINATION_LOW_WORD_SCORE = 0.25
HALLUCINATION_LOW_AVG_LOGPROB = -0.6
HALLUCINATION_HIGH_NO_SPEECH_PROB = 0.5
HALLUCINATION_HIGH_COMPRESSION_RATIO = 2.4
PROPER_NOUN_MIN_LENGTH = 3
PROPER_NOUN_LOW_CONFIDENCE = 0.4

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because",
    "of", "to", "in", "on", "at", "for", "with", "is", "are", "was", "were",
    "be", "been", "i", "you", "he", "she", "it", "we", "they", "this",
    "that", "these", "those", "my", "your", "his", "her", "its", "our",
    "their", "not", "no", "yes", "do", "does", "did", "just", "there",
    "here", "go", "going", "gonna", "okay", "what", "who", "how", "why",
    "when", "where", "um", "uh", "like", "well", "now", "get", "got",
}


# ---------------------------------------------------------------------------
# Loading / normalizing word lists
# ---------------------------------------------------------------------------

def load_primary_words(whisperx_json_path):
    """Flatten WhisperX's raw output/VIDEO.json into a flat word list.

    Uses the raw WhisperX JSON (not manuscript_audio_evidence.json) because
    the evidence file only carries low-confidence words, not every word.
    """
    path = Path(whisperx_json_path)

    if not path.exists():
        return [], {}

    with path.open("r", encoding="utf-8-sig") as f:
        transcript = json.load(f)

    words = []
    segment_avg_logprob = {}

    for seg_index, segment in enumerate(transcript.get("segments", [])):
        if segment.get("avg_logprob") is not None:
            segment_avg_logprob[seg_index] = float(segment["avg_logprob"])

        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")

            if start is None or end is None:
                # WhisperX sometimes drops timing on stray punctuation-only
                # tokens; they cannot be time-aligned, so they are skipped
                # here rather than guessed.
                continue

            words.append({
                "word": word.get("word", ""),
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "score": (
                    round(float(word["score"]), 3)
                    if word.get("score") is not None
                    else None
                ),
                "segment": seg_index,
            })

    words.sort(key=lambda w: w["start"])
    return words, segment_avg_logprob


def normalize_word(word):
    return re.sub(r"[^\w']", "", word.lower()).strip()


# ---------------------------------------------------------------------------
# Persistent secondary-ASR worker client (3A.1: one model load per run)
# ---------------------------------------------------------------------------

class WorkerUnavailable(Exception):
    pass


class AsrWorkerClient:
    """Talks to one long-lived `manuscript_audio_asr_worker.py --serve`
    subprocess so the secondary model is loaded exactly once per pipeline
    run, no matter how many windows get transcribed.
    """

    def __init__(
        self,
        whisperx_python,
        model=SECONDARY_MODEL,
        compute_type="int8",
        ready_timeout=WORKER_READY_TIMEOUT_SEC,
    ):
        whisperx_python = Path(whisperx_python)

        if not whisperx_python.exists():
            raise WorkerUnavailable(
                f"WhisperX environment not found: {whisperx_python}"
            )
        if not WORKER.exists():
            raise WorkerUnavailable(f"ASR worker script not found: {WORKER}")

        self.model = model
        self._proc = subprocess.Popen(
            [
                str(whisperx_python),
                str(WORKER),
                "--serve",
                "--model",
                model,
                "--compute-type",
                compute_type,
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        ready_line = self._read_line(timeout=ready_timeout)

        if ready_line is None:
            stderr = self._drain_stderr()
            self.close(force=True)
            raise WorkerUnavailable(
                f"secondary ASR worker did not become ready within "
                f"{ready_timeout}s: {stderr}"
            )

        payload = json.loads(ready_line)

        if payload.get("status") != "ready":
            self.close(force=True)
            raise WorkerUnavailable(
                f"secondary ASR worker failed to initialize: {payload}"
            )

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_line(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return line

    def _drain_stderr(self):
        try:
            return (self._proc.stderr.read() or "")[-2000:]
        except Exception:  # noqa: BLE001
            return ""

    def transcribe(self, audio_path, start=None, end=None, timeout=WORKER_REQUEST_TIMEOUT_SEC):
        request = {"audio_path": str(audio_path)}
        if start is not None and end is not None:
            request["start"] = float(start)
            request["end"] = float(end)

        started = time.time()

        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return {
                "status": "failed",
                "error": f"worker pipe closed: {exc}",
                "segments": [],
            }

        line = self._read_line(timeout=timeout)

        if line is None:
            return {
                "status": "failed",
                "error": f"worker did not respond within {timeout}s",
                "segments": [],
            }

        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "status": "failed",
                "error": f"worker response was not valid JSON: {exc}",
                "segments": [],
            }

        result["runtime_sec"] = round(time.time() - started, 2)
        return result

    def close(self, force=False):
        if self._proc.poll() is not None:
            return

        if not force:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass

        try:
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._proc.kill()


def flatten_secondary_words(secondary_result):
    words = []

    for segment in secondary_result.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "word": word.get("word", ""),
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "score": round(float(word.get("score", 0.0)), 3),
            })

    words.sort(key=lambda w: w["start"])
    return words


def find_covering_segment(segments, start, end):
    best = None
    best_overlap = 0.0

    for seg in segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg

    return best


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals, join_gap=0.15):
    intervals = sorted(
        (float(a), float(b)) for a, b in intervals if float(b) > float(a)
    )
    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(a, b) for a, b in merged]


# ---------------------------------------------------------------------------
# Word-level consensus (3.5: sequence-aware alignment)
# ---------------------------------------------------------------------------

def _word_center(w):
    return (float(w["start"]) + float(w["end"])) / 2.0


def _pair_score(pw, sw):
    """Score aligning primary word pw with secondary word sw.

    Same normalized text within the 300-400ms center tolerance is the best
    match (drift-tolerant). Identical text that drifted further still beats a
    gap, so one misaligned word is not misread as a coverage gap. Different
    text with close center times is a genuine model disagreement (conflict
    candidate). Everything else is strongly discouraged so the alignment does
    not stretch across unrelated words.
    """
    norm_p = normalize_word(pw["word"])
    norm_s = normalize_word(sw["word"])
    center_dist = abs(_word_center(pw) - _word_center(sw))

    if norm_p and norm_p == norm_s:
        if center_dist <= MATCH_CENTER_TOLERANCE_SEC:
            return MATCH_SAME_TEXT_SCORE
        if center_dist <= MATCH_CENTER_DRIFT_SEC:
            return MATCH_SAME_TEXT_DRIFTED_SCORE
        return MATCH_FAR_PENALTY

    if center_dist <= MATCH_CENTER_TOLERANCE_SEC:
        return MATCH_DIFF_TEXT_PENALTY

    return MATCH_FAR_PENALTY


def _align_word_sequences(primary_words, secondary_words):
    """Needleman-Wunsch global alignment over normalized text, scored by
    center-time distance. Returns (aligned_pairs, primary_gapped,
    secondary_gapped) where aligned_pairs is a list of (p_index, s_index)
    tuples. Sequence context makes the match robust to both timing drift and
    small insertions/deletions between the two models (3.5 hardening).
    """
    n = len(primary_words)
    m = len(secondary_words)

    if n == 0 or m == 0:
        return [], set(range(n)), set(range(m))

    # DP table with traceback. Row-major; dp[i][j] is the best score for
    # prefix primary[:i] vs secondary[:j].
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag, 2=up(gap p), 3=left(gap s)

    for i in range(1, n + 1):
        dp[i][0] = ALIGN_GAP_PENALTY * i
        trace[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = ALIGN_GAP_PENALTY * j
        trace[0][j] = 3

    for i in range(1, n + 1):
        pw = primary_words[i - 1]
        for j in range(1, m + 1):
            sw = secondary_words[j - 1]
            diag = dp[i - 1][j - 1] + _pair_score(pw, sw)
            up = dp[i - 1][j] + ALIGN_GAP_PENALTY
            left = dp[i][j - 1] + ALIGN_GAP_PENALTY

            if diag >= up and diag >= left:
                dp[i][j] = diag
                trace[i][j] = 1
            elif up >= left:
                dp[i][j] = up
                trace[i][j] = 2
            else:
                dp[i][j] = left
                trace[i][j] = 3

    aligned = []
    primary_gapped = set()
    secondary_gapped = set()
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 1:
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 2:
            primary_gapped.add(i - 1)
            i -= 1
        else:
            if j > 0:
                secondary_gapped.add(j - 1)
            j -= 1

    aligned.reverse()
    return aligned, primary_gapped, secondary_gapped


def build_word_consensus(primary_words, secondary_words):
    """Classify each primary word by whether the secondary pass agrees.

    Consensus states (spec 3A-C): confirmed / probable / conflicting /
    missing_from_one_model / unrecoverable. `needs_listen` is a derived flag,
    not a separate bucket, so every word still gets exactly one state.

    3.5 hardening: matching is a global sequence alignment (normalized-text
    equality + center-time distance), so the same word heard by both models
    a few hundred ms apart counts as agreement instead of two unrelated weak
    words (the `Go!` regression) and tiny drift no longer fabricates
    coverage gaps.

    Also returns secondary_only_words: words the secondary model produced
    with no aligned primary word at all. This is the mechanism that catches
    the exact regression this phase exists for -- primary ASR stopping early
    while speech continues.
    """
    aligned, primary_gapped, secondary_gapped = _align_word_sequences(
        primary_words, secondary_words
    )

    consensus = []

    for i, pw in enumerate(primary_words):
        primary_score = pw["score"] if pw["score"] is not None else 1.0

        matched = next(
            (j for pi, j in aligned if pi == i),
            None,
        )

        if matched is None:
            state = (
                "unrecoverable"
                if primary_score < UNRECOVERABLE_SCORE
                else "missing_from_one_model"
            )
            consensus.append({
                "word": pw["word"],
                "start": pw["start"],
                "end": pw["end"],
                "segment": pw.get("segment"),
                "primary_score": pw["score"],
                "state": state,
                "secondary_word": None,
                "secondary_score": None,
                "overlap_ratio": 0.0,
                "center_distance_sec": None,
                "missing_from": "secondary",
                "needs_listen": True,
            })
            continue

        sw = secondary_words[matched]
        same_text = normalize_word(pw["word"]) == normalize_word(sw["word"])

        if same_text:
            state = (
                "confirmed"
                if primary_score >= LOW_CONFIDENCE_SCORE
                else "probable"
            )
        else:
            state = "conflicting"

        entry = {
            "word": pw["word"],
            "start": pw["start"],
            "end": pw["end"],
            "segment": pw.get("segment"),
            "primary_score": pw["score"],
            "state": state,
            "secondary_word": sw["word"],
            "secondary_score": sw["score"],
            "overlap_ratio": round(
                overlap_ratio(pw["start"], pw["end"], sw["start"], sw["end"]),
                3,
            ),
            "center_distance_sec": round(
                abs(_word_center(pw) - _word_center(sw)), 3
            ),
            "needs_listen": state in ("conflicting", "probable"),
        }

        consensus.append(entry)

    secondary_only_words = [
        {
            "word": sw["word"],
            "start": sw["start"],
            "end": sw["end"],
            "secondary_score": sw["score"],
            "state": "missing_from_one_model",
            "missing_from": "primary",
            "needs_listen": True,
        }
        for j, sw in enumerate(secondary_words)
        if j in secondary_gapped
    ]

    return consensus, secondary_only_words


def build_conflicts(word_consensus):
    return [
        {
            "start": w["start"],
            "end": w["end"],
            "primary_word": w["word"],
            "secondary_word": w["secondary_word"],
            "primary_score": w["primary_score"],
            "secondary_score": w["secondary_score"],
        }
        for w in word_consensus
        if w["state"] == "conflicting"
    ]


# ---------------------------------------------------------------------------
# Advisory risk signals (3A.1-4/5): flag-for-review only, never a fact.
# ---------------------------------------------------------------------------

def compute_hallucination_risk(
    word_entry,
    primary_segment_avg_logprob,
    secondary_segments,
    independent_speech_regions,
):
    """Advisory only. Never returns a tier above MEDIUM; a hallucination is
    a hypothesis for the reviewer to check by listening, not a fact.
    """
    reasons = []
    score = 0.0

    score_val = word_entry.get("primary_score")
    if score_val is not None and score_val < HALLUCINATION_LOW_WORD_SCORE:
        score += 0.3
        reasons.append(f"very low primary word confidence ({score_val})")

    state = word_entry.get("state")
    if state == "conflicting":
        score += 0.25
        reasons.append("model disagreement on this word")
    elif state in ("missing_from_one_model", "unrecoverable"):
        score += 0.15
        reasons.append(f"state={state}")

    if primary_segment_avg_logprob is not None and (
        primary_segment_avg_logprob < HALLUCINATION_LOW_AVG_LOGPROB
    ):
        score += 0.2
        reasons.append(
            f"low segment avg_logprob ({round(primary_segment_avg_logprob, 3)})"
        )

    sec_seg = find_covering_segment(
        secondary_segments, word_entry["start"], word_entry["end"]
    )
    if sec_seg:
        no_speech = sec_seg.get("no_speech_prob")
        if no_speech is not None and no_speech > HALLUCINATION_HIGH_NO_SPEECH_PROB:
            score += 0.25
            reasons.append(f"secondary segment no_speech_prob={no_speech}")

        compression = sec_seg.get("compression_ratio")
        if (
            compression is not None
            and compression > HALLUCINATION_HIGH_COMPRESSION_RATIO
        ):
            score += 0.2
            reasons.append(
                f"high compression_ratio={compression} (possible repetition)"
            )

    if independent_speech_regions:
        covered = any(
            min(word_entry["end"], r_end) - max(word_entry["start"], r_start) > 0.02
            for r_start, r_end in independent_speech_regions
        )
        if not covered:
            score += 0.2
            reasons.append(
                "no independent VAD/diarization speech signal at this time"
            )

    if not reasons:
        return None

    score = round(min(score, 1.0), 2)
    # Advisory cap: hallucination risk never reaches STRONG/CONFLICT tiers on
    # its own -- it is a lead, not a transcript verdict.
    tier = "MEDIUM" if score >= 0.5 else "WEAK"

    return {
        "word": word_entry["word"],
        "start": word_entry["start"],
        "end": word_entry["end"],
        "score": score,
        "tier": tier,
        "reasons": reasons,
    }


def build_hallucination_risk(
    word_consensus,
    primary_segment_avg_logprob,
    secondary_segments,
    independent_speech_regions,
):
    risks = []

    for entry in word_consensus:
        risk = compute_hallucination_risk(
            entry,
            primary_segment_avg_logprob.get(entry.get("segment")),
            secondary_segments,
            independent_speech_regions,
        )
        if risk:
            risks.append(risk)

    return risks


def compute_proper_noun_risk(word_entry):
    """Conservative: flags a token worth checking, never guesses the name.

    Trigger bar: not a common function/filler word, AND (model disagreement
    OR low primary confidence). Purely a listening cue.
    """
    norm = normalize_word(word_entry["word"])

    if len(norm) < PROPER_NOUN_MIN_LENGTH or norm in _STOPWORDS:
        return None

    state = word_entry.get("state")
    score_val = word_entry.get("primary_score")
    weak_confidence = score_val is not None and score_val < PROPER_NOUN_LOW_CONFIDENCE

    if state != "conflicting" and not weak_confidence:
        return None

    reasons = []
    if state == "conflicting":
        reasons.append(
            f"model disagreement (secondary heard "
            f"\"{word_entry.get('secondary_word')}\")"
        )
    if weak_confidence:
        reasons.append(f"low primary confidence ({score_val})")

    return {
        "word": word_entry["word"],
        "start": word_entry["start"],
        "end": word_entry["end"],
        "reasons": reasons,
    }


def build_proper_noun_risk(word_consensus):
    risks = []

    for entry in word_consensus:
        risk = compute_proper_noun_risk(entry)
        if risk:
            risks.append(risk)

    return risks


# ---------------------------------------------------------------------------
# Coverage / agreement stats
# ---------------------------------------------------------------------------

def build_coverage_stats(
    primary_words,
    secondary_only_words,
    word_consensus,
    duration_sec,
):
    primary_spans = merge_intervals(
        (w["start"], w["end"]) for w in primary_words
    )
    secondary_only_spans = merge_intervals(
        (w["start"], w["end"]) for w in secondary_only_words
    )

    asr_covered = sum(b - a for a, b in primary_spans)
    uncovered_recovered = sum(b - a for a, b in secondary_only_spans)

    matched = [w for w in word_consensus if w["secondary_word"] is not None]
    agreeing = [w for w in matched if w["state"] == "confirmed"]

    longest_uncovered = (
        round(max(b - a for a, b in secondary_only_spans), 3)
        if secondary_only_spans
        else 0.0
    )

    return {
        "duration_sec": round(float(duration_sec), 3) if duration_sec else None,
        "asr_coverage_pct": (
            round(asr_covered / duration_sec, 3) if duration_sec else None
        ),
        "uncovered_speech_duration_sec": round(uncovered_recovered, 3),
        "longest_uncovered_region_sec": longest_uncovered,
        "model_agreement_pct": (
            round(len(agreeing) / len(matched), 3) if matched else None
        ),
        "word_disagreement_count": sum(
            1 for w in word_consensus if w["state"] == "conflicting"
        ),
        "missing_from_primary_count": len(secondary_only_words),
        "missing_from_secondary_count": sum(
            1
            for w in word_consensus
            if w["state"] in ("missing_from_one_model", "unrecoverable")
        ),
        "unrecoverable_count": sum(
            1 for w in word_consensus if w["state"] == "unrecoverable"
        ),
    }


# ---------------------------------------------------------------------------
# Rerun-window identification (spec 3A-E/F, hardened in 3A.1-2/3)
# ---------------------------------------------------------------------------

def _secondary_only_trigger(start, end, independent_speech_regions, duration_sec):
    """Is this secondary-only gap worth an automatic model rerun?

    Small gaps stay in the evidence (word_consensus / secondary_only_words /
    REVIEW_ME coverage-gap list) regardless -- this only gates whether it
    also gets a rerun, so tiny alignment jitter stops burning CPU time
    without losing recall.
    """
    length = end - start

    if length >= MIN_AUTO_RERUN_GAP_SEC:
        return "gap_duration_meets_threshold"

    if independent_speech_regions:
        for r_start, r_end in independent_speech_regions:
            if min(end, r_end) - max(start, r_start) > 0.05:
                return "independent_speech_signal_present"

    if duration_sec:
        if start <= CLIP_BOUNDARY_TRIGGER_SEC or duration_sec - end <= CLIP_BOUNDARY_TRIGGER_SEC:
            return "near_clip_boundary"

    return None


def identify_rerun_windows(
    primary_words,
    secondary_only_words,
    word_consensus,
    duration_sec,
    independent_speech_regions=None,
):
    """Windows worth a short, targeted secondary rerun.

    Triggers: secondary caught speech primary missed entirely (if it clears
    the strength bar in `_secondary_only_trigger`), two models disagree
    materially (always -- a conflict is inherently strong evidence), speech
    reaches clip end but the transcript ends early, or (when given) an
    independent VAD/diarization speech region has no primary ASR word at
    all. Nearby candidates are merged into one inference call; each merged
    window keeps its original `source_windows` for provenance.
    """
    raw_candidates = []

    # Contiguous secondary-only words are grouped ONLY to judge trigger
    # strength as one missed utterance (a multi-word gap should be judged as
    # a whole, per spec 3A.1-3). Each underlying word still becomes its own
    # raw candidate so provenance/attribution below stays word-level and
    # does not collapse distinct words into one blurred span.
    if secondary_only_words:
        sorted_gap_words = sorted(secondary_only_words, key=lambda w: w["start"])
        gap_regions = merge_intervals(
            (w["start"], w["end"]) for w in sorted_gap_words
        )

        for region_start, region_end in gap_regions:
            trigger = _secondary_only_trigger(
                region_start, region_end, independent_speech_regions, duration_sec
            )
            if not trigger:
                continue

            for w in sorted_gap_words:
                if w["start"] >= region_start - 1e-6 and w["end"] <= region_end + 1e-6:
                    raw_candidates.append({
                        "start": w["start"], "end": w["end"],
                        "reason": "secondary_only_speech",
                        "trigger": trigger,
                    })

    # Each conflicting word is its own raw candidate (not pre-merged) so a
    # transcript conflict never loses its individual identity -- the outer
    # merge step below is the only place nearby candidates combine.
    for w in word_consensus:
        if w["state"] == "conflicting":
            raw_candidates.append({
                "start": w["start"], "end": w["end"],
                "reason": "model_disagreement",
                "trigger": "transcript_conflict",
            })

    if primary_words and duration_sec:
        last_end = max(w["end"] for w in primary_words)
        if duration_sec - last_end > 0.5:
            raw_candidates.append({
                "start": last_end, "end": duration_sec,
                "reason": "clip_end_gap",
                "trigger": "speech_may_continue_past_transcript_end",
            })

    if independent_speech_regions:
        primary_spans = merge_intervals(
            (w["start"], w["end"]) for w in primary_words
        )
        for region_start, region_end in independent_speech_regions:
            has_primary = any(
                min(region_end, b) - max(region_start, a) > 0.2
                for a, b in primary_spans
            )
            if not has_primary:
                raw_candidates.append({
                    "start": region_start, "end": region_end,
                    "reason": "independent_signal_gap",
                    "trigger": "vad_or_diarization_speech_with_no_asr",
                })

    raw_candidates.sort(key=lambda c: c["start"])

    # 3A.1-2: merge candidates within MERGE_GAP_SEC into one window, keeping
    # each original candidate as a source_window for provenance.
    merged = []
    for c in raw_candidates:
        if merged and c["start"] <= merged[-1]["end"] + MERGE_GAP_SEC:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            merged[-1]["source_windows"].append(c)
            if c["reason"] not in merged[-1]["reasons"]:
                merged[-1]["reasons"].append(c["reason"])
        else:
            merged.append({
                "start": c["start"],
                "end": c["end"],
                "reasons": [c["reason"]],
                "source_windows": [c],
            })

    windows = []

    for m in merged:
        padded_start = max(0.0, m["start"] - RERUN_PADDING_SEC)
        padded_end = m["end"] + RERUN_PADDING_SEC
        if duration_sec:
            padded_end = min(duration_sec, padded_end)

        windows.append({
            "start": round(padded_start, 3),
            "end": round(padded_end, 3),
            "reasons": m["reasons"],
            "source_windows": [
                {
                    "start": round(sw["start"], 3),
                    "end": round(sw["end"], 3),
                    "reason": sw["reason"],
                    "trigger": sw["trigger"],
                }
                for sw in m["source_windows"]
            ],
        })

    return windows


def map_rerun_to_source_windows(rerun_result, source_windows):
    """Attribute the merged rerun's recovered words back to each original
    (unmerged) source window, so a wider inference call does not blur which
    specific gap/conflict it resolved (spec 3A.1-2).
    """
    all_words = [
        w
        for seg in rerun_result.get("segments", [])
        for w in seg.get("words", [])
    ]

    mapped = []

    for sw in source_windows:
        s, e = sw["start"], sw["end"]
        words_in = [w for w in all_words if w["end"] > s and w["start"] < e]
        text = " ".join(w["word"] for w in words_in).strip()

        mapped.append({
            **sw,
            "recovered_text": text or None,
        })

    return mapped


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def _unavailable_result(error):
    return {
        "status": "unavailable",
        "error": error,
        "secondary_model": SECONDARY_MODEL,
        "coverage": None,
        "word_consensus": [],
        "conflicts": [],
        "rerun_windows": [],
        "reruns_executed": [],
        "hallucination_risk_words": [],
        "proper_noun_risk_words": [],
        "policy": (
            "Secondary ASR pass did not complete; the base packet still "
            "generates. Transcript was NOT cross-checked. Treat every "
            "word as needing the same listening discipline as before "
            "Phase 3A."
        ),
    }


def write_asr_consensus_evidence(
    whisperx_json_path,
    audio_path,
    whisperx_python,
    output_path,
    duration_sec=None,
    independent_speech_regions=None,
    max_auto_reruns=MAX_AUTO_RERUNS,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary_words, primary_segment_avg_logprob = load_primary_words(
        whisperx_json_path
    )

    started = time.time()

    try:
        client = AsrWorkerClient(whisperx_python)
    except WorkerUnavailable as exc:
        result = _unavailable_result(str(exc))
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("ASR CONSENSUS: UNAVAILABLE |", result["error"])
        return result

    model_load_sec = round(time.time() - started, 2)

    try:
        secondary_full = client.transcribe(audio_path)

        if secondary_full.get("status") != "complete":
            result = _unavailable_result(
                secondary_full.get("error", "secondary pass failed")
            )
            result["status"] = "failed"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("ASR CONSENSUS: FAILED |", result["error"])
            return result

        secondary_words = flatten_secondary_words(secondary_full)
        secondary_segments = secondary_full.get("segments", [])

        word_consensus, secondary_only_words = build_word_consensus(
            primary_words, secondary_words
        )

        conflicts = build_conflicts(word_consensus)

        coverage = build_coverage_stats(
            primary_words, secondary_only_words, word_consensus, duration_sec
        )

        rerun_windows = identify_rerun_windows(
            primary_words,
            secondary_only_words,
            word_consensus,
            duration_sec,
            independent_speech_regions,
        )

        hallucination_risk_words = build_hallucination_risk(
            word_consensus,
            primary_segment_avg_logprob,
            secondary_segments,
            independent_speech_regions,
        )

        proper_noun_risk_words = build_proper_noun_risk(word_consensus)

        reruns_executed = []

        for window in rerun_windows[:max_auto_reruns]:
            rerun_result = client.transcribe(
                audio_path, start=window["start"], end=window["end"]
            )
            per_source = (
                map_rerun_to_source_windows(
                    rerun_result, window["source_windows"]
                )
                if rerun_result.get("status") == "complete"
                else [
                    {**sw, "recovered_text": None}
                    for sw in window["source_windows"]
                ]
            )
            reruns_executed.append({
                "window": [window["start"], window["end"]],
                "reasons": window["reasons"],
                "status": rerun_result.get("status"),
                "recovered_text": " ".join(
                    seg.get("text", "")
                    for seg in rerun_result.get("segments", [])
                ).strip()
                or None,
                "source_windows": per_source,
                "runtime_sec": rerun_result.get("runtime_sec"),
            })

        skipped = len(rerun_windows) - len(reruns_executed)

        result = {
            "status": "complete",
            "secondary_model": SECONDARY_MODEL,
            "model_load_sec": model_load_sec,
            "secondary_runtime_sec": secondary_full.get("runtime_sec"),
            "primary_word_count": len(primary_words),
            "secondary_word_count": len(secondary_words),
            "coverage": coverage,
            "word_consensus": word_consensus,
            "secondary_only_words": secondary_only_words,
            "conflicts": conflicts,
            "rerun_windows": rerun_windows,
            "reruns_executed": reruns_executed,
            "reruns_skipped_count": max(0, skipped),
            "hallucination_risk_words": hallucination_risk_words,
            "proper_noun_risk_words": proper_noun_risk_words,
            "policy": (
                "Cross-model comparison is evidence, not a transcript edit. "
                "Never auto-insert [uncertain]/[unintelligible]/[inaudible] "
                "-- those remain reviewer decisions. hallucination_risk and "
                "proper_noun_risk are advisory leads only, capped at MEDIUM."
            ),
        }
    finally:
        client.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        "ASR CONSENSUS: PASS |",
        f"model_load={model_load_sec}s |",
        f"primary={len(primary_words)} words |",
        f"secondary={len(secondary_words)} words |",
        f"agreement={coverage['model_agreement_pct']} |",
        f"conflicts={len(conflicts)} |",
        f"recovered_gap_sec={coverage['uncovered_speech_duration_sec']} |",
        f"reruns={len(reruns_executed)}/{len(rerun_windows)}",
    )

    if skipped > 0:
        print(
            f"NOTE: {skipped} rerun window(s) were identified but not "
            "executed (max_auto_reruns cap); see rerun_windows for the full "
            "list."
        )

    return result


def main():
    if len(sys.argv) < 4:
        print(
            "usage: manuscript_audio_asr_consensus.py "
            "WHISPERX_JSON AUDIO_WAV WHISPERX_PYTHON [OUTPUT_JSON]"
        )
        sys.exit(1)

    whisperx_json_path = sys.argv[1]
    audio_path = sys.argv[2]
    whisperx_python = sys.argv[3]
    output_path = (
        sys.argv[4]
        if len(sys.argv) > 4
        else ROOT / "analysis" / "asr_consensus_evidence.json"
    )

    write_asr_consensus_evidence(
        whisperx_json_path, audio_path, whisperx_python, output_path
    )


if __name__ == "__main__":
    main()
