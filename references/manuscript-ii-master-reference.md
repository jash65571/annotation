# Manuscript II — Master Reference
Compiled from the Learning Hub (Differences vs Manuscript I · Assessment Instructions · Task Overview · Additional Checks) + annotation-tool walkthrough screenshots. Keep this open while tasking.

---

## 1. Project at a glance
- **What it is:** Review + enrich pre-filled AI-generated captions for short clips (~10–20s). Captions become training data for video-understanding models → the bar is **precision and verifiability, not creativity**.
- **Pay:** $30/hr, billed via the annotation platform (time caps per task, based on average completion times).
- **Core change vs Manuscript I:** No more error-code flagging. You **enrich, not audit** — add detail + inline timestamps everywhere.
- **The two laws:** NO OMISSIONS (every major visible/audible thing a viewer would notice must be captioned) and NO HALLUCINATIONS (nothing you can't actually see or hear).
- **Timestamps: accurate to within 0.1s** of the actual occurrence. This is graded literally.

## 2. Path to paid work
1. Standard onboarding (identity, payment/tax, Project Terms)
2. Slack → **#manuscript-tasking**
3. Assessment: ~30 min, auto-graded, **open book** (keep Learning Hub + this doc open). Unpaid, completed inline (no result-code flow).
4. Tasking at $30/hr.
- Free practice sandbox: manuscript-annotator-two.learn.joinhandshake.com/walkthrough

## 3. Task workflow (end-to-end)
1. Claim task on annotation platform → click link → task tool opens in browser (video + pre-filled caption loaded, nothing to install).
2. **First watch is locked**: caption stays locked until ~90% watched; scrubbing/skip disabled — only play, pause, speed. ("Skip — I've already watched it" link exists for re-runs.)
3. Read the pre-filled caption end-to-end.
4. Watch again against the caption; note everything missing (details, characters, objects, scene elements, actions, sounds, on-screen text).
5. Edit in place: add detail + inline timestamps.
6. Re-read the finished caption against the video one final time (nothing missed, nothing invented).
7. Click **Copy** → produces a **result code** (long encoded multi-line gibberish string). Paste it into the platform submission field **exactly as copied — never edit, trim, reformat, or remove line breaks** — then submit there.
8. Timing/payment handled by the platform; no timer in the tool.
- ⚠️ **Wrong-video captions (code video-id ≠ task video-id) are an AUTO-REJECT.** First sanity check on every task: does the caption describe the video you're watching?

## 4. Caption format

### Part 1 — [Overview] block (true of the whole clip)
**Characters & Objects**
- Every person = C1, C2, C3… in order of appearance. Prominent objects = O1, O2…
- Required for every human: gender · approximate age · ethnicity/race · build · hair (color, length, style) · clothing head-to-toe · distinguishing features (scars, tattoos, glasses, beard).
- If lower body never visible, end with the literal sentence: **"Lower body and shoes are not visible."**
- **Off-screen voices ARE characters** — own C-ID + voice description (e.g., "C3: Off-screen female narrator, warm mid-range voice. Never visible on camera."). Demographic fields don't apply — describe the voice. Tool also prompts for nationality/ethnicity **or accent** (e.g., "British, soft-spoken" / "Southern drawl") when determinable.
- Once defined with an ID, shots refer by ID only — no re-describing.
- Small common objects (a knife, a fruit slice): skip formal O-labels, describe inline in Scene. Formal O-labels are for large/prominent, repeatedly referenced objects.

**Scene** — Comprehensive enough to reconstruct composition without the video: spatial positioning of characters relative to each other and camera, ground/vegetation texture, lighting quality (dappled/overexposed/shadowed), background elements, depth layering. Static initial state lives here: what each character HOLDS and their posture — described **per-character inline** (never shared generic labels → hand-assignment ambiguity). Changes-over-time go in per-shot Movements.

**Style** — Lighting, color, depth of field, film look, aspect ratio (e.g., "Handheld documentary style… naturalistic color, no visible grading, shallow-to-moderate DOF, no non-standard aspect ratio").

**Audio** — Background audio, music, ambient noise only. Document ALL audible sound incl. subtle ambient (wind, rustling, distant machinery) and label it as such. **Speech NEVER goes here** — speech lives in shots.

**Visual concerns** — letterbox bars, watermarks, platform bugs (TikTok/IG), burned-in text/subtitles, camera shake, compression artifacts, ghosting, exposure blowouts. Goes here, NOT in Scene; note what + where on frame. Else "None."

**Audio concerns** — silence, clipping, hum, echo, wind, dropouts, sudden volume shifts, muffled/unclear speech, first-word cuts, best-effort-guess transcriptions, hedged speaker attribution ("either C2 or C3"). Else "None."

### Part 2 — one [Shot N: start–end] block per cut
- **Cut** — how the shot begins (see §6). Shot 1 is ALWAYS "Opening shot."
- **Camera** — transition + shot type + angle + ONE primary movement with a rhythm word (e.g., "Medium close-up, eye-level, slow dolly push-in"). ALL camera moves (pans, tilts, zooms, drifts, rack focus) get timestamps + **what becomes visible or hidden** ("Camera pans left; only C1 and the right arm of C2 are visible").
- **Scene** — only what's NEW or CHANGED. Otherwise literally "No changes from overview."
- **Movements / Action & Audio** — what characters do, how they move, micro-expressions. Every significant motion gets an inline timestamp. Never skip facial expressions/reactions (smiles, nods, lip movement, gaze shifts) — **especially reactions to speech**. When a character speaks, mention the speech action here too (CHANGE 2 rule: "C1 sets down the mug and says 'Did you finish the report?'") in addition to the verbatim Audio line.
- **Video playback speed** — one of: slow_motion / regular / accelerated. Mid-shot speed changes (slow-mo, speed ramp, freeze frame) noted with start/end timestamps + effect. Never call a sped-up shot "regular."
- **Audio** — verbatim quoted speech with attribution + tone/delivery ("C1 says in a relaxed, slightly muffled tone, mouth partly full: '…'"), plus SFX and music changes. Every speech segment timestamped. Preserve stutters/hesitations ("My wife, Trudy. Uh—I'm Shane."). Off-screen speech marked ("C3 says off-screen…").
- Either Camera or Movements can carry a given action — but it must exist in one of them.

## 5. Timestamp rules
- **0.1s precision, always.**
- Block timestamps `[start–end]` = overall shot range / group of related events. Inline `(start–end)` = individual action, speech, or sound within a block.
- Timestamps go **BEFORE** the thing they describe — never mid-line.
- All speech + all important visible actions must be timestamped. Full-shot ambient audio may ride the block timestamp.
- Speech pause > **0.5s** → split into separate timestamped segments at the pause boundary; each covers only spoken words, excluding silence.
- **Overlapping speakers**: each gets their own timestamp — timestamps may overlap.
- Don't cut lines off early: a line labeled (1.5s–3.0s) that ends at 3.4s must say 3.4s.

## 6. Cut types (fixed menu)
| Type | Meaning / example |
|---|---|
| Opening shot | Shot 1 only, always, never elsewhere |
| **Hard cut** | Instant change, no blending. **The overwhelming default — when in doubt, it's a hard cut** |
| Cross dissolve | Shots overlap/fade into each other (montages, time passing) |
| Fade in / Fade out | From/to black or white (open/close scenes) |
| Match cut | Hard cut where shape/composition/motion matches across the cut (bone→spaceship) |
| Jump cut | Same subject, same framing, intermediate motion removed — subject "jumps" (vlogger edits) |
| Smash cut | Abrupt contrast in energy/sound/subject — jarring on purpose |
| Wipe | New shot pushes old off-screen along a line (Star Wars) |
| Iris | Circular wipe opening/closing (old cartoons) |
| L-cut | PREVIOUS shot's audio continues over next shot's picture |
| J-cut | NEXT shot's audio starts before its picture |
| Whip pan / Swish pan | Fast blurred pan as the transition — tag either as the same type |
- Jump cut ≠ match cut — don't confuse them. Specialty cuts only when the defining property is clearly present.
- **Recent reviewer feedback:** jump cuts get missed and mislabeled — scan for EVERY cut, timestamp each, label jump cuts explicitly.

## 7. The 16 Additional Checks (pre-submit pass)
1. **Ghost characters** — C#/O# defined but never referenced in any Scene/Movements/Audio line (esp. the off-screen speaker who never speaks). Remove or place them. Every entry needs full head-to-toe + lower-body sentence.
2. **Cameraman/off-screen filmer** — hand/arm/shadow/voice of the person filming = a normal C# character; speech gets the off-screen marker. No special "cameraman" field exists.
3. **Watermarks/visual artifacts** → Visual concerns (not Scene), with what + rough frame position. "None" only if genuinely clean.
4. **Unintelligible audio** — transcribe what IS audible, mark lost portion inline with **[inaudible]** (spec convention — NOT `<unintelligible>`; the reference caption's `<unintelligible>` is the older style). Never guess — a guess = hallucinated quote. Flag in Audio concerns. Hedged attribution also → Audio concerns.
5. **Concerns test** — is it a technical/processing defect of the recording? Audio bucket: silence, clipping, hum, echo, wind, muffled, dropouts, volume shifts. Visual bucket: letterbox, watermarks, burned-in text, shake, artifacts, ghosting, blowouts. Fit → name it; else "None."
6. **Don't hedge vocal-profile attributes** — if accent/apparent race/etc. can't be confidently determined, OMIT entirely. No filler like "accent not confidently identifiable."
7. **Ghost quotes** — (a) fabricated/inaudible/lip-read/carried-over quotes: delete or repair; every quote must match audible speech within 0.1s. (b) malformed punctuation: unclosed ", doubled "", split quotes — balance every line.
8. **Missed speech** — capture ALL audible dialogue, each attributed to a specific C#, timestamped to 0.1s.
9. **Ambient audio** — always label + describe the block. Never "[no audio]" before an audible sound, never empty Audio field, no filler like "No speech." Genuine silence → Audio concerns as silence.
10. **Vocal profiling** — any human sound (speech, singing, laughter): detectable gender, approx. age, accent, voice quality — especially off-screen. Human off-screen singer ≠ generic "music."
11. **Frame-level cut QA** — don't default to Hard cut without checking; a brief flash-frame is its OWN shot, not folded into a neighbor.
12. **Fade-frame attribution** — content visible during a fade/cross-fade belongs to the OUTGOING shot.
13. **Camera-movement accuracy** — each direction segment (pan-left → pan-right → tilt-up) gets its own 0.1s timestamp; sustained direction in Camera; tilts that reveal new objects described.
14. **Major-action completeness** — state direction, intent, affected object, character reaction ("C1 walks stage-right while moving O3 partially off-screen"). Include big reactions (a laugh), not just the object move.
15. **Diarization cascade** — fixing a speaker means also fixing that speaker's timestamps, gestures, and action descriptions — not just the transcript.
16. **Timestamp & field placement** — timestamps before descriptions, never mid-line; speech always in a timestamped Action & Audio entry, never untimestamped inside Scene.

## 8. Tool validation rules (from the walkthrough — not in the docs!)
These are hard blockers/flags the editor enforces; "N to resolve" counts them:
- **Unique start/end PAIR per action line within a shot.** Two lines may share a start OR an end, but never both → "Duplicate window" flag. Stagger honestly (e.g., 0.0–1.0 / 0.2–1.1), don't stamp everything 0.0–0.4.
- **Action timestamps must fall INSIDE the shot window** → "Timestamps fall outside the shot window" flag. Shot-spanning ambient must be clipped per shot (e.g., shot = 0.4–8.1 → ambient line ends 8.1, not 8.5).
- **Character IDs, not pronouns.** "He turns back…" gets flagged: "Add a character ID: He, his — replace with the character label (@C1, @C2…). Always refer to characters by ID unless inside quoted dialogue."
- **One line per sentence** in Action & Audio; each line gets its own start/end. Include conversation captions AND background/prominent audio as lines.
- **On-screen text is captioned as its own timestamped line** (e.g., `On-screen text reads "I COULDN'T FORGET HER"` — and multi-text shots list every string).
- **Camera Movements sub-section is OPTIONAL** — only add if the camera actually moves (pan/tilt/zoom/dolly…) with start+end; leave empty for static shots.
- **Playback speed** is its own per-shot field; speed-change entries only if speed shifts mid-shot.
- Shots can be **inserted, split, resized (start/end fields), and deleted**; very short shots (even 0.4s) still stand alone ("short shot" badge is informational — see check 11).
- **Off-screen character entries prompt for nationality/ethnicity or accent** — add if determinable, omit if not (check 6).
- Submit gates: **Watch the full video** ✓ → **mark Overview as reviewed** → **mark every Shot as reviewed** → resolve all flags → Final Review → **Copy** result code.
- Player tools: −Frame/+Frame stepping, 0.1x / 0.25x / 0.5x / 1x speeds, loop toggle, "Follow video" mode, click a shot to play it on loop. **Use frame-step + 0.1x–0.25x to pin every timestamp.**

## 9. Recent reviewer-feedback hotspots (graded hard)
1. **Speaker attribution + timestamp specificity** — name the C# ("C3 laughs"), never "woman laughing," whenever visually/audibly clear; tighten endpoints to the true word boundary; keep stutters/hesitations.
2. **Scene cuts & jump cuts** — find every cut, timestamp it, label jump cuts explicitly as jump cuts, don't confuse jump vs match.

## 10. Scoring rubric (how QA grades you)
- **5** — No issues. Everything captured; all timestamps within 0.1s.
- **4** — ONE minor issue (one subtle gesture missed, one timestamp off by >0.1s, minor phrasing).
- **3** — Two+ minor issues stacked (missed micro-expressions + under-detailed lighting + slightly-off stamps).
- **2** — ONE major issue (undocumented framing-changing pan, wrong hand/character assignment, missing speech line) OR many minors.
- **1** — Multiple major issues (missing timestamps throughout, wrong labels/objects, omitted actions).
- Reviewer motto: **"When in doubt, score down — precision is the point."** → your motto: when in doubt, add the detail and verify the stamp.

## 11. Detail-quality bar (phrasing standards)
- ❌ "Holding a mug" → ✅ "holding a chipped white ceramic mug in their right hand" (object = name, color, material, condition, how handled).
- ❌ "A man walks in" → ✅ "C2 (tall, dark coat) enters from the left at (00:03.4)."
- ❌ "a kitchen" → ✅ what's on the counter, on the walls, where each character stands.
- Emotions are hallucinations; observables are captions: ❌ "he's happy" → ✅ "(4.8s–7.9s) C3 smiles, opening her mouth and showing her teeth."
- C-ID + physical anchor on references: "C2 (the blonde bride in the bolero)," not "the woman."
- Target: someone who never saw the video could reconstruct it from your caption.

## 12. Final pre-submit checklist (merged from all docs)
- [ ] Caption matches THIS video (video-id sanity check — auto-reject otherwise)
- [ ] Watched full clip cold (sound ON), then again against the caption
- [ ] Every field touched — Characters, Scene, Style, Audio, both Concerns, every Shot
- [ ] Every character: full required fields + lower-body sentence; off-screen voices have C-IDs
- [ ] No ghost characters, no ghost quotes, no missed speech, no missed cuts
- [ ] Every major motion/action/reaction/camera move/speech line has an inline timestamp to 0.1s
- [ ] Unique start/end pairs; all stamps inside shot windows; stamps before descriptions
- [ ] C-IDs everywhere (no bare he/she outside quotes)
- [ ] Playback speed correct on every shot
- [ ] Nothing invented — every claim seeable/hearable
- [ ] Ran the 16 Additional Checks
- [ ] Overview + all shots marked reviewed; flags at 0; Final Review clean
- [ ] Result code copied VERBATIM into the platform and submitted there

## 13. Known inconsistencies to watch (ask in Slack if graded on them)
- **[inaudible] vs `<unintelligible>`** — Additional Checks says `[inaudible]` is the spec convention; the reference caption uses `<unintelligible>…>`. Follow **[inaudible]** and note the guess in Audio concerns; use it extremely sparingly.
- **"Movements"+"Audio" (assessment wedding example) vs combined "Action & Audio" (reference caption + live tool)** — the tool uses **Action & Audio, one line per sentence**; follow the tool.
- The sandbox loads mismatched sample data (food-hall caption over different footage) — that's placeholder, but it's also exactly what the wrong-video auto-reject looks like in the wild.

## 14. Ops notes
- Time caps = billable ceiling per task; questions → Slack **#manuscript-tasking** ("no dumb questions").
- Reading materials + assessment = unpaid; walkthrough practice = unpaid; real tasks = paid.
- Quality on early tasks determines whether you keep the queue. Precision > throughput.
