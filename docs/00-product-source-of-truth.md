# 00 — Product Source of Truth

Converted from the Manuscript II reference material into software requirements.

**Sources** (in `./references/`):

| ID | File | Role |
|---|---|---|
| MASTER-REF | `manuscript-ii-master-reference.md` | Master Reference compiled from the Learning Hub + tool walkthrough (newest live-tool behavior) |
| HANDBOOK | `MANUSCRIPT-II-REVIEWER-HANDBOOK-v1.2 (1).md` | Reviewer Handbook v1.2 (2026-08-10), includes complete Golden Examples coverage |
| GOLDEN-PDF | `Manuscript II.pdf` | 31-page Golden Examples export (game trailer + crayfish clip), frame-matched entries |

> Note: the supplied `Manuscript II.pdf` **is** the Golden Examples document. The
> current controlling source set also names four documents **not yet locally
> supplied** — Master Frame Audit Protocol v1.5, MANUSCRIPT-II-PROJECT-RULES-CURRENT.md,
> MANUSCRIPT_AUDIT_README-v3.2.md, MANUSCRIPT-II-CURRENT-SOURCES.md. Their known
> requirements are encoded in `manuscript_v1.yaml` (v1.3.0) with TASK-FEED provenance;
> when the documents arrive they outrank this file per docs/01. Raw source documents
> are local-only and gitignored (see `references/README.md`). **Phase 3.1 note:** the
> four named controlling files are still absent, so their provenance re-mapping is
> deferred — `TASK-FEED` entries and `supplied_locally: false` stand until they land.

## Additional controlling rules (rules v1.3.0, TASK-FEED provenance)

- Newest official/task-specific workflow material outranks all older material;
  actual media remains factual truth; Golden Examples are floor, not ceiling.
- **Action & Audio strict atomicity**: one line = one independently defensible
  event/source unless truly inseparable; no hidden second action behind
  "and/then/while/as"; no multiple speech acts per line; no visual action plus a
  separate popup/sound/object event in one line.
- **No fake timestamp nudges** — ever; re-measure real boundaries instead.
- **Final object state must be checked** at shot end.
- **An incorrect annotation endpoint is a permanent failure class** — annotation
  interval endpoints come from verified media evidence, never from the final
  frame's start PTS.
- Our machine validator is a pre-check; it does not replace platform-semantic
  validation.
- **Descript is disabled by default**; local ASR failure never authorizes cloud or
  Descript fallback; ASR is evidence only; the full audit continues after a local
  ASR failure.

## Audio workflow (rules v1.3.0, AUDIT-NOTES/QC-REPORT provenance)

The approved local audio workflow is
`FFmpeg → faster-whisper (large-v3-turbo) → WhisperX forced alignment` in
isolated uv-bootstrapped environments (AUDIT-NOTES §5). Transcription only —
never translation. Transcripts are wording leads, not frame evidence
(QC-REPORT). When ASR cannot run, waveform/spectrogram/10 ms-energy evidence
still completes and dialogue remains provisional until source-audio
verification — never export-ready on ASR output alone.

**Phase 3.1 audio evidence integrity.** ASR speech (text, word timing, language,
speaker) is *unverified evidence*: every machine speech region carries
`source_verification_status = UNVERIFIED` and a mandatory
`MANDATORY_SOURCE_AUDIO_VERIFICATION` review item — a clean, high-confidence,
fully-aligned result never bypasses a human listen, and top-level audio PASS is
forbidden while any speech is source-unverified. Machine ASR text is never
caption-eligible (`caption_text_eligible` requires human verification/correction),
and a machine language guess is never a caption language claim. WhisperX supplies
timing only: reconciliation yields exactly one best word per faster-whisper
source word (never dropped, invented, or re-worded), missing timing stays missing
(never 0 s), and partial alignment is recorded with coverage. Boundary audio
continuity is conservative — energy on both sides never proves one source crosses
a cut; CONTINUOUS needs a crossing word or proven spectral continuity, and L/J
transitions are never auto-selected.

## 1. Product purpose
Produce extremely precise, evidence-backed Manuscript II reviewer captions from short
videos (~10–20 s) and seeded caption data, reducing a ~40-minute manual review to a
5–10-minute human verification workflow. Captions are training data for
video-understanding models: the bar is **precision and verifiability, not creativity**
(MASTER-REF §1).

## 2. Reviewer role
We review the pre-filled caption against the actual video but are not required to
preserve it (HANDBOOK §1). The reviewer watches the clip with sound, checks every
factual claim, adds missing detail, corrects errors, rebuilds unreliable sections, and
marks reviewed only when the caption meets the 5/5 standard (HANDBOOK §29, §33 Rule L).
The application assists; **the human remains final** — no autonomous platform
submission, no task claiming, no credential handling.

## 3. Source hierarchy
When sources disagree (HANDBOOK §4, project brief):
1. Current official workflow / live-tool rules / task-specific feedback
2. Actual video frames and audio
3. Newest Master Frame Audit Protocol (when supplied)
4. Current Reviewer Handbook / current project rules
5. Golden Examples (quality level)
6. Seed caption
7. AI / evaluator suggestions

An older example never overrides a newer live-tool rule.

## 4. KEEP / FIX_ENRICH / REDO_REBUILD
Every seeded section gets exactly one outcome (HANDBOOK §1, §5 step 3A):
- **KEEP** — structure and facts already correct.
- **FIX_ENRICH** — mostly usable; needs detail, tighter timing, corrections.
- **REDO_REBUILD** — wrong shot count/boundaries, mixed-up identities, broadly
  missing dialogue, consistently broad timestamps, duplicated entries, or when
  patching preserves more mistakes than rebuilding. **Do not patch a bad foundation.**

## 5. No omissions
Every major visible or audible thing a viewer would notice must be captioned
(MASTER-REF §1 "the two laws"). Includes reactions, micro-expressions, ambient sound,
on-screen text, camera movement, and every audible speech line.

## 6. No hallucinations
Nothing enters the caption unless actually seeable or hearable. Never invent dialogue,
speaker identity, hand assignment, emotion, accent, race, age, camera movement, sound,
cut type, or playback change (HANDBOOK §3). Uncertainty lowers specificity — omit
uncertain traits entirely; never write hedges (MASTER-REF check 6). Emotions are
hallucinations; observables are captions (MASTER-REF §11).

## 7. 0.1-second final timing standard
Timestamps accurate to within 0.1 s of the actual occurrence — graded literally
(MASTER-REF §1, §5; HANDBOOK §20 Rule 1). Display format: block `[start–end]` for
shots, inline `(start–end)` / `[Xs-Ys]` before each entry.

## 8. Exact internal frame timing
Internally the engine preserves source timing exactly: integer PTS × rational
time_base, held as `Fraction`. The 0.1 s value is a presentation-layer projection
(`to_manuscript_display`), never internal state. AI never owns the clock.

## 9. Every-frame accounting
If ffprobe reports N encoded video frames, the ledger must contain exactly N records.
If reliable enumeration fails, the run is FAILED or PARTIAL — never a silently
incomplete ledger. The Golden Examples pull evidence stills "straight from the video
with ffmpeg" (GOLDEN-PDF p.1) — frame-level evidence is the grading standard.

## 10. Shot and cut verification
Check every suspected boundary frame by frame (HANDBOOK §5 step 5, §14). Do not trust
seeded shot counts. Very short shots (0.4 s) can be real (MASTER-REF §8). Flash frames
require adjacent-frame review (MASTER-REF check 11). Fade content belongs to the
outgoing shot (check 12). Camera movement is never a cut (GOLDEN-PDF example 2).
Jump cuts are graded hotspots — find, timestamp, and label every one (MASTER-REF §9).

## 11. Character continuity
C1, C2, … in order of appearance; once defined, refer by ID only (MASTER-REF §4).
Full head-to-toe description; literal sentence "Lower body and shoes are not visible."
when applicable. Off-screen voices are characters. The filmer gets a normal C-ID when
relevant (check 2). No ghost characters (check 1). Descriptions must stay consistent
across shots (HANDBOOK §31).

## 12. Object continuity
O-IDs for prominent, recurring objects; small common items described inline
(MASTER-REF §4). Object contact, release, and transfer are real state changes
(HANDBOOK §28C). No duplicate or unused O-IDs; color/type/ownership must not conflict
across shots (HANDBOOK §7).

## 13. Camera requirements
Camera field: shot size + angle + framing + one primary movement with a rhythm word
(MASTER-REF §4). Camera Movements: timestamped entries per direction phase, stating
what becomes visible or hidden; optional (empty) for static shots (MASTER-REF §8,
HANDBOOK §15). Camera movement never goes in Action & Audio.

## 14. Action & Audio requirements
The live tool combines action and sound into Action & Audio: one line per sentence,
each with its own start/end (MASTER-REF §8). Granularity follows the footage — 0.1 s
entries are valid (GOLDEN-PDF 7.3s–7.4s). Capture entries character by character in
dense shots; never skip reactions (MASTER-REF §4; HANDBOOK §16).

## 15. Dialogue rules
Verbatim transcription with speaker C-ID, tone/delivery, on/off-screen status.
Preserve stutters, repeats, cut-offs. Split at pauses > 0.5 s. Overlapping speakers
get overlapping separate entries. Never lip-read a quote. Wrong attribution triggers
the diarization cascade (fix timestamps, gestures, actions too) (MASTER-REF §4/§5,
checks 7/8/15; HANDBOOK §17).

## 16. Audio requirements
All audible sound is captioned: music, ambience, impacts, footsteps, wind, mechanical
noise — labeled and timestamped (MASTER-REF §4, check 9; HANDBOOK §18). Human
off-screen singing is a voice, not generic "music" (check 10). Overview Audio holds
recurring sound only; speech never lives there.

## 17. On-screen text rules
Visible text is its own timestamped line quoting the text exactly
(e.g. `On-screen text reads "I COULDN'T FORGET HER"`); multi-text shots list every
string; simultaneous text blocks stay together (MASTER-REF §8; HANDBOOK §19).

## 18. Playback speed
Per-shot required value: slow_motion / regular / accelerated (use the tool's exact
labels). Mid-shot changes get timestamped Speed Change entries. Fast motion ≠
accelerated playback; motion blur ≠ slow motion (MASTER-REF §4; HANDBOOK §23).

## 19. Visual concerns
Real recording/processing defects only: watermarks, platform logos, burned-in
subtitles, letterbox bars, shake, compression artifacts, ghosting, blowouts — with
frame position. Else literally "None." Never reviewer notes (HANDBOOK §11).

## 20. Audio concerns
Silence, clipping, hum, echo, wind, dropouts, volume shifts, muffled speech,
first-word cuts, overlap that blocks words, hedged attribution context. Else "None."
Never a reviewer scratchpad (MASTER-REF §4; HANDBOOK §12).

## 21. Current pronoun rule
**No pronouns outside quoted dialogue** (he/she/they/his/her/him/them). Use C-IDs;
"C1 raises the right hand", not "his hand". Live tool flags violations. Older Golden
Example wording with pronouns must not be copied (MASTER-REF §8; HANDBOOK §21, §36).

## 22. Current [inaudible] rule
`[inaudible]` is the current token; `<unintelligible>` is deprecated older style.
Transcribe what is audible, mark only the missing portion, never guess, use sparingly,
and note the difficulty in Audio concerns (MASTER-REF check 4, §13; HANDBOOK §36).

## 23. Current Action & Audio rule
Combined Action & Audio field (not separate Movements + Audio), one sentence per line,
each line individually timestamped; speech is mentioned in the action stream when a
character speaks AND transcribed verbatim (MASTER-REF §4 CHANGE 2, §13; HANDBOOK §36).

## 24. Unique timestamp pair rule
Within a shot, no two entries may share the exact same start AND end. Sharing one
endpoint is fine. Never fake a 0.1 s shift to satisfy the validator — re-measure the
real boundaries (MASTER-REF §8; HANDBOOK §20 Rule 4, §34 examples 4–5).

## 25. Timestamp-inside-shot rule
Every entry's window must fall inside its shot window; shot-spanning ambience is
clipped per shot (MASTER-REF §8; HANDBOOK §20 Rule 3).

## 26. Overlap rules
Overlap is allowed and often required: overlapping speakers, action under narration,
sound vs visual with different ranges (explosion visual ≠ explosion audio). Never
flatten real overlap (MASTER-REF §5; HANDBOOK §20 Rule 5; GOLDEN-PDF shot 3).

## 27. Golden Example quality gate
Golden Examples are the floor, not the ceiling (HANDBOOK §1A). Compare behaviors, not
word count: event awareness, specificity, true timing boundaries, overlap fidelity,
camera separation, audio completeness, speech completeness, object/character
continuity, scene-reconstruction value, absence of unsupported claims.

## 28. Reviewer checklist
The 9 official questions (detail, timestamp accuracy, duplicate windows, character
labels, playback speed, camera movement, redundancy, completeness, overlapping speech
— HANDBOOK §24) plus the 16 Additional Checks (ghost IDs, cameraman, watermarks,
[inaudible], concerns test, no hedging, ghost quotes, missed speech, ambient audio,
vocal profiling, frame-level cut QA, fade attribution, camera accuracy, major-action
completeness, diarization cascade, timestamp placement — MASTER-REF §7).

## 29. Validator requirements
The future caption validator (Phase 3+) must detect at minimum: duplicate full
timestamp pairs; timestamps outside shots; improper gaps/overlap in shot layout;
camera movement inside Action & Audio; dynamic action in Scene; pronouns outside
dialogue; undefined/ghost C#/O#; malformed quotes; empty-audio fillers ("No music",
"No speech", "No dialogue"); unsupported transition types; Shot 1 not Opening shot /
later shot marked Opening; untimestamped speech or on-screen text; invalid playback
speed; contradictory visibility; object-continuity conflicts; shot final-state gaps;
reviewer commentary in caption fields. Rule IDs use the `M2-<AREA>-<NNN>` registry
(e.g. `M2-TIME-001`); Phase 1 pipeline checks use `P1-<AREA>-<NNN>`. Output:
severity, rule, location, message, suggested fix.

## 30. First-pass export-ready quality goal
The finished caption must survive the live tool's submit gates first try: every field
reviewed, all flags at zero, Final Review clean, result code copied verbatim
(MASTER-REF §8, §12). Target score: 5/5 — "when in doubt, add the detail and verify
the stamp" (MASTER-REF §10).

## Documented source conflicts
See `engine/manuscript_reviewer/rules/manuscript_v1.yaml → known_conflicts`:
[inaudible] vs `<unintelligible>`; Movements+Audio vs Action & Audio; pronouns in old
Golden Examples vs current C-ID rule; playback-speed label variants. In every case the
newer live-tool rule wins.
