# Manuscript II Reviewer Handbook

**Version 1.2**  
**Role: Reviewer**  
**Built from the current Manuscript II Task Overview, Reviewer Checklist, Additional Checks, workflow walkthrough, Master Reference, and Golden Examples supplied on August 10, 2026.**

---

# 1. The Role: We Are Reviewers

The most important rule is simple:

> **We review the pre-filled caption against the actual video, but we are not required to preserve it.**

A reviewer has three valid outcomes:

1. **KEEP** — preserve a seeded field or entry when it is already accurate and detailed enough.
2. **FIX / ENRICH** — correct or expand a seed that is mostly usable.
3. **REDO / REBUILD** — replace the seeded Overview, shot structure, Cast, Objects, Camera, Action & Audio, dialogue, or even the whole caption when the seed is too wrong, incomplete, or structurally unreliable.

The task is not to protect the seed. The task is to produce the strongest caption supported by the video and audio.

A reviewer must:

1. Watch the actual clip with sound.
2. Read the pre-filled caption.
3. Decide whether each part should be kept, fixed, or rebuilt.
4. Check every factual claim against the video.
5. Add missing detail.
6. Correct wrong detail.
7. Rebuild sections or the full caption when patching would preserve errors.
8. Correct shot boundaries and transition types.
9. Correct speaker attribution.
10. Correct action and speech timing.
11. Remove duplicate or unsupported content.
12. Check all tool warnings and Final Review suggestions.
13. Use judgment on suggestions instead of accepting them automatically.
14. Mark the task reviewed only when the finished caption is strong enough to submit.

The finished caption should let a person reconstruct the clip without watching it.

## Reviewer mindset

A reviewer asks:

- Is this actually visible?
- Is this actually audible?
- Is the correct character named?
- Is the correct object named?
- Does this event really start here?
- Does it really end here?
- Is a cut missing?
- Is this really a hard cut?
- Is this camera movement or character movement?
- Is this line of speech complete?
- Is the speaker correct?
- Is the timestamp inside the shot?
- Does another line use the same full timestamp pair?
- Is any important action, reaction, sound, or text missing?
- Is any sentence claiming more than the clip proves?

**No omissions. No hallucinations.**


---

# 1A. Mandatory Golden Example Standard

The Golden Examples are the **required quality and writing benchmark** for every review task.

Before editing a task, remember how the Golden Examples are written:

- rich but factual Overview descriptions;
- clear C# and O# identity use;
- detailed Scene descriptions that let someone reconstruct the view;
- camera movement separated from Action & Audio;
- exact start and end timing for each event;
- overlapping events kept as overlapping entries;
- speech transcribed verbatim with speaker and delivery;
- audible music, ambience, impacts, and effects included;
- fine-grained actions split when the footage supports them;
- no important visible or audible event silently omitted.

The Golden Examples show that **entry count follows the footage, not a quota**. A short action can deserve a 0.1-second entry. A dense shot can need many Action & Audio entries.

## Mandatory Golden Example comparison

For every task, compare the final caption against the Golden Examples before marking it reviewed.

Ask:

- Is our Overview at least as useful and reconstructable?
- Are our character and object descriptions at the same level of useful detail?
- Are our action entries as precise?
- Are our timestamps tied to the true event boundaries?
- Is camera movement separated and timed correctly?
- Did we capture all meaningful dialogue and audio?
- Did we preserve real overlap instead of flattening it?
- Did we describe visible physical changes rather than vague interpretations?
- Would this caption look at home beside a Golden Example?

**Golden Examples are the floor, not the ceiling.**

Current tool rules override older Golden Example syntax when they conflict, but the Golden Examples remain the minimum standard for detail, timing, structure, and caption-writing quality.

---

# 2. What Manuscript II Review Work Is

Manuscript II uses short video clips, often around 10 to 20 seconds.

The tool normally opens with:

- the video;
- a pre-filled caption;
- an Overview;
- one or more Shots;
- character and object IDs;
- scene, style, audio, and concerns;
- camera fields;
- Action & Audio entries;
- playback-speed fields.

The pre-filled caption is a starting point.

The reviewer is responsible for turning it into a precise, detailed, timestamped caption.

The quality target is not a loose summary.

The quality target is a **frame-matched event record**.

---

# 3. The Reviewer Standard

A strong reviewed caption must meet all of these standards.

## Detail

Capture meaningful:

- people;
- objects;
- positioning;
- posture;
- actions;
- reactions;
- facial changes;
- speech;
- music;
- sound effects;
- ambient sound;
- camera movement;
- visible text;
- visual defects.

Do not add detail just to make a caption longer.

Useful detail is visible, audible, and relevant.

## Timing

Every important action, speech segment, sound, camera move, or text event needs an accurate time range.

**Target: within 0.1 seconds of the actual event.**

The Golden Examples show the expected standard: the start frame should genuinely show the event beginning, and the end frame should genuinely show it ending.

## Truthfulness

Never invent:

- dialogue;
- speaker identity;
- object ownership;
- hand assignment;
- facial emotion;
- accent;
- race or ethnicity;
- age;
- camera movement;
- sound;
- cut type;
- playback-speed change.

If something cannot be supported, remove it or describe only what can be supported.

## Consistency

The final caption must not contain:

- ghost characters;
- unused objects;
- conflicting character descriptions;
- conflicting object descriptions;
- conflicting vehicle names;
- impossible hand assignments;
- duplicate speech;
- duplicated timestamp windows;
- speech assigned to two different people without evidence;
- Scene text that conflicts with Action & Audio.

## Export safety

A factually good caption is still unfinished if the tool cannot export it.

All fields must be reviewed.

All blocking flags must be handled.

The final result code must remain intact.

---

# 4. Reviewer Source Priority

When sources disagree, use this order:

1. **Current task instructions and live tool rules**
2. **Actual video and audio**
3. **Current reviewer guidance and Additional Checks**
4. **Golden Examples**
5. **Pre-filled caption**
6. **AI or Final Review suggestions**

The Golden Examples teach the quality bar.

They are not permission to copy older syntax that the live tool now rejects.

The pre-filled caption and automated suggestions are leads, not truth.

---

# 5. End-to-End Reviewer Workflow

## Step 1: Confirm the task

Before editing:

- confirm the video ID;
- confirm the caption describes the same video;
- confirm the clip duration;
- note the current number of shots.

A wrong-video caption is an auto-reject risk.

## Step 2: Watch the full clip cold

Watch once with sound before making major edits.

During this watch, note:

- setting;
- characters;
- objects;
- speech;
- music;
- ambient sound;
- visible text;
- likely cuts;
- camera motion;
- speed changes.

Do not rely on the pre-filled caption during the first understanding pass.

## Step 3: Read the pre-filled caption fully

Read:

- every Cast entry;
- every Object entry;
- Scene;
- Style;
- Audio;
- Visual concerns;
- Audio concerns;
- every Shot;
- every Camera entry;
- every Action & Audio line;
- every playback-speed value.

Do not review only the fields already flagged.

## Step 3A: Decide whether to keep, fix, or rebuild

Before patching individual lines, judge the seed as a whole.

### Keep
Use the seed when the structure and facts are already correct and only normal review is needed.

### Fix / enrich
Use this when the caption is generally right but needs better detail, tighter timing, missing audio, corrected wording, or a few structural changes.

### Redo / rebuild
Redo the affected section, shot, or full caption when:

- the shot count is wrong;
- many shot boundaries are wrong;
- character identities are mixed up;
- object IDs are unreliable;
- dialogue is broadly missing or misattributed;
- timestamps are consistently broad or incorrect;
- camera movement is mixed into the wrong fields;
- many entries are duplicated;
- the Overview conflicts with the actual clip;
- patching the seed would take more work or preserve more mistakes than rebuilding it.

**Do not patch a bad foundation.**

If the seed is unreliable, rebuild from the video and audio.

## Step 4: Watch again against the caption

Now compare the caption to the clip.

Look for:

- missing people;
- wrong people;
- missing objects;
- wrong objects;
- missing speech;
- wrong speaker;
- wrong speech text;
- missing actions;
- broad action ranges;
- bad shot boundaries;
- missed jump cuts;
- incorrect transition labels;
- missed camera movement;
- wrong playback speed;
- missing audio;
- missing watermarks;
- missing on-screen text.

## Step 5: Map the true shots

Check each suspected boundary frame by frame.

For every shot determine:

- start;
- end;
- transition type;
- whether the boundary is real.

Do not preserve seeded shot boundaries just because they are already there.

Very short shots can still be real shots.

## Step 6: Review the Overview

Verify:

- Characters;
- Objects;
- Scene;
- Style;
- Audio;
- Visual concerns;
- Audio concerns.

Remove unused IDs.

Add missing IDs.

Correct descriptions that conflict with the video.

## Step 7: Review every shot

For each shot check:

- Cut;
- Camera;
- Camera Movements;
- Scene;
- Action & Audio;
- Playback speed;
- Speed changes.

## Step 8: Pin timestamps

Use:

- frame stepping;
- 0.1x playback;
- 0.25x playback;
- looping;
- follow-video mode.

Check the first and last moment of every event.

## Step 9: Compare against the Golden Examples

Before validator cleanup, compare the draft to the Golden Examples.

Check:

- Overview detail level;
- Cast/Object description quality;
- Scene reconstruction value;
- Action & Audio granularity;
- speech detail;
- ambient and sound-effect coverage;
- camera separation;
- timestamp precision;
- truthful overlap;
- overall readability.

If our caption is clearly thinner, broader, or less precise than the Golden Examples, continue reviewing.

## Step 10: Run the Reviewer Checklist

Run the 9 official reviewer questions.

Then run all 16 Additional Checks.

## Step 11: Review tool flags

Resolve:

- duplicate windows;
- timestamps outside shots;
- pronoun warnings;
- missing required fields;
- malformed quotes;
- unsupported transitions;
- other export blockers.

Do not fix warnings with fake timing.

## Step 12: Judge Final Review suggestions

Final Review suggestions are **review leads**, not commands.

For each suggestion:

1. Read the issue.
2. Go back to the relevant video range.
3. Check the actual evidence.
4. Resolve it if the suggestion is correct.
5. Ignore it if the suggestion is wrong.
6. Never make a change only because the suggestion exists.

## Step 13: Final watch

Watch the complete clip one more time against the finished caption.

Ask:

- Did we miss anything?
- Did we invent anything?
- Is any speaker wrong?
- Is any important event mistimed?
- Is any cut missing?
- Does each field contain the right type of information?

## Step 14: Submit safely

Mark:

- Overview reviewed;
- every Shot reviewed;
- all flags handled.

Run Final Review.

Copy the result code exactly.

Never:

- trim it;
- edit it;
- reformat it;
- remove line breaks.

If possible, test the copied code in a fresh tool tab before submission.

---

# 6. Overview Review Rules

# Characters

Every important person needs a C-ID.

Use:

- C1;
- C2;
- C3;
- and so on.

Character IDs should follow order of appearance where possible.

Once an ID exists, use that ID later.

Do not switch between:

- C2;
- "the man";
- "the woman";
- "the person";

when the person is already identified.

## Character description

Describe visible traits that help identify the person:

- apparent gender presentation when supported;
- approximate age when support is strong;
- build;
- hair;
- facial hair;
- glasses;
- hat;
- top;
- jacket;
- trousers;
- shoes;
- visible jewelry;
- notable accessories.

If the lower body never appears, use:

> **Lower body and shoes are not visible.**

Do not force uncertain traits.

If race, ethnicity, accent, or another profile trait cannot be supported, omit it rather than writing a hedge.

## Off-screen voices

An off-screen narrator or speaker can be a C-ID.

Describe:

- voice type;
- apparent gender when clear;
- age range when clear;
- accent when clear;
- tone;
- fact that the speaker is never visible, when applicable.

Do not create an off-screen character and then never reference that character again.

That is a ghost character.

## Cameraman or filmer

If the person holding the camera becomes relevant through:

- a visible hand;
- a visible arm;
- a shadow;
- speech;
- interaction;

the filmer may need a normal C-ID.

There is no special "cameraman" field.

---

# 7. Object Review Rules

Use O-IDs for prominent objects that are reused or matter to the clip.

Examples:

- vehicle;
- cooler;
- weapon;
- sign;
- major tool;
- recurring device.

Small common items can be described inline instead.

Examples:

- fruit slice;
- small knife;
- napkin;
- spoon.

## Object checks

For each O-ID ask:

- Is this one real physical object?
- Is it used in later shots?
- Is another ID describing the same object?
- Does the color match?
- Does the type match?
- Does the object stay with the same character?
- Does the object description conflict with later scenes?

Avoid vague forward references.

Prefer a self-contained description.

Example:

Bad:

> O3: The bumper of O4.

Better:

> O3: A chrome front bumper with a yellow license plate, mounted on the black mini-bus O4.

---

# 8. Scene Review Rules

Scene describes the space and stable state.

Include:

- location;
- background;
- foreground;
- surfaces;
- furniture;
- vegetation;
- major objects;
- lighting;
- depth;
- character position;
- initial posture;
- what characters are holding at the stable starting state.

A good Scene lets a viewer reconstruct the composition.

Do not use Scene for changing motion.

Changing action belongs in Action & Audio.

Camera movement belongs in Camera Movements.

Speech belongs in Action & Audio.

---

# 9. Style Review Rules

Style describes the overall visual treatment.

Useful details:

- natural daylight;
- indoor artificial light;
- cool or warm color;
- saturated or muted color;
- high contrast;
- shallow depth of field;
- moderate depth of field;
- handheld documentary look;
- game-engine render;
- heavy bloom;
- lens flare;
- standard 16:9;
- letterbox bars, when relevant.

Do not repeat the same defect in both Style and Visual concerns unless the two fields serve different purposes.

---

# 10. Overview Audio Review Rules

Overview Audio describes recurring sound across the clip.

Examples:

- orchestral score;
- electronic music;
- room ambience;
- wind;
- road noise;
- narration;
- recurring speaker;
- game sound effects.

Detailed speech transcription belongs in the shots.

Overview Audio may summarize who speaks and the general sound design.

Do not leave Audio empty if strong recurring sound is clearly present.

Do not invent ambient sound because the visual suggests it should exist.

---

# 11. Visual Concerns

Use Visual concerns for real visual recording or processing issues.

Examples:

- watermark;
- platform logo;
- burned-in subtitles;
- letterbox bars;
- severe camera shake;
- compression artifacts;
- double exposure;
- ghosting;
- heavy overexposure;
- glare that blocks detail.

If none apply, write:

> None.

Do not use Visual concerns for internal reviewer notes.

Bad:

> Need to verify whether C2 is the person at left.

That is reviewer process text, not a customer-facing concern.

---

# 12. Audio Concerns

Use Audio concerns for real audio problems.

Examples:

- silence;
- clipping;
- hum;
- echo;
- wind;
- muffled audio;
- dropout;
- sudden volume change;
- overlapping speech that blocks words;
- first-word cut;
- unclear portion of speech.

If none apply, write:

> None.

Do not use Audio concerns as a reviewer scratchpad.

Bad:

> Speaker still needs verification.

Better, when the source itself is unclear:

> Overlapping speech from C2 and C3 makes several words difficult to distinguish.

---

# 13. Shot Review Structure

Each real shot should contain:

1. Start
2. End
3. Cut
4. Camera
5. Camera Movements, only when present
6. Scene
7. Action & Audio
8. Playback speed
9. Speed Changes, only when present

---

# 14. Shot Boundary Review

Do not trust the existing shot count.

Check every possible edit.

## Opening shot

Only Shot 1.

## Hard cut

Instant image change with no blend.

## Jump cut

Same subject or very similar framing, but time is removed and the subject visibly jumps.

Do not confuse a jump cut with a match cut.

## Cross dissolve

Both shots overlap while one fades out and the next fades in.

## Fade in

Image appears from black or white.

## Fade out

Image disappears toward black or white.

## Match cut

A hard cut linked through matched shape, framing, or motion.

## Smash cut

A deliberately jarring abrupt contrast.

## Wipe

The new image pushes the old image away.

## Iris

Circular opening or closing transition.

## L-cut

Previous shot audio continues after the visual cut.

## J-cut

Next shot audio begins before its picture appears.

## Whip pan / Swish pan

Fast blurred camera pan used as the transition.

## Flash-frame review

A bright flash can be:

- a real edited flash frame;
- part of a fade;
- part of a wipe;
- an exposure effect inside one shot.

Inspect adjacent frames.

Do not create a shot only because one frame is bright.

If the flash is truly a separate edited frame or transition element, treat it according to what the tool supports.

## Fade-frame attribution

Content still visible during the outgoing fade belongs to the outgoing shot.

---

# 15. Camera Review

Camera description and camera movement are not the same thing.

## Camera field

Describe:

- shot size;
- angle;
- framing;
- general handheld/static character.

Examples:

- Medium shot, eye-level.
- Wide shot, low angle.
- Close-up, high angle, handheld.

## Camera Movements

Use timestamped movement entries for:

- pan;
- tilt;
- push-in;
- pull-back;
- zoom;
- tracking;
- dolly;
- drift;
- rack focus;
- major handheld move.

Each distinct direction can need its own entry.

Example:

- 0.0-1.2: Camera pans screen-left.
- 1.2-2.0: Camera reverses and pans screen-right.
- 2.0-2.5: Camera tilts upward.

When useful, state what the movement reveals or hides.

Example:

> Camera pans screen-left, moving C2 out of view while bringing C3 into frame.

Do not put camera movement in Action & Audio.

Leave the Camera Movements section empty for a truly static shot.

---

# 16. Action & Audio Review

The live tool combines dynamic visual action and sound into **Action & Audio** entries.

Each entry needs:

- a start;
- an end;
- one clear sentence;
- one clear event or tightly linked event;
- correct C/O labels;
- a range inside the current shot.

## Granularity follows the footage

An entry can last:

- 0.1 seconds;
- 0.6 seconds;
- 3 seconds;
- 7 seconds.

There is no quota.

Use the real event boundary.

## Good action detail

Bad:

> C1 moves.

Better:

> C1 raises the right hand from waist height to the forehead.

Bad:

> C2 looks happy.

Better:

> C2 smiles, opens the mouth, and shows the teeth.

Describe observables instead of inferred emotion.

## Major actions

Capture:

- entry;
- exit;
- turn;
- reach;
- release;
- pickup;
- drop;
- bite;
- clap;
- laugh;
- nod;
- smile;
- hand raise;
- hand lower;
- head turn;
- gaze change;
- body lean;
- object movement;
- reaction.

Do not skip reactions just because another event is more important.

---

# 17. Speech Review

Every audible speech segment must be captured.

For every line verify:

1. speaker;
2. exact wording;
3. tone;
4. start;
5. end;
6. whether the speaker is on-screen or off-screen.

## Verbatim speech

Preserve:

- hesitations;
- repeated words;
- stutters;
- cut-off words.

Example:

> C1 says, "I'm gonna, I'm just gonna put them in here where they--"

Do not silently clean speech into better grammar.

## Speaker attribution

Use the correct C-ID.

Avoid:

> A woman says...

when the clip supports:

> C3 says...

Wrong speaker attribution is a major error because it can also break related gestures and timestamps.

When correcting a speaker, review the full surrounding sequence.

This is the **diarization cascade check**.

## Off-screen speech

Mark it.

Example:

> C4 says off-screen in a calm tone: "..."

## Inaudible speech

Use **[inaudible]** for a genuinely missing portion.

Example:

> C2 says, "[inaudible] already."

Use it sparingly.

Never invent a word to avoid [inaudible].

Flag meaningful audio difficulty in Audio concerns.

## Overlapping speech

Each speaker keeps a separate entry and separate range.

Example:

- 10.1-10.5: C4 says, "Jackfruit?"
- 10.2-10.6: C3 says, "a lot."

The overlap is correct.

Do not collapse both speakers into one line.

## Long pauses

If one speaker pauses for more than about 0.5 seconds, split the speech into separate ranges.

The timestamp should cover speech, not the silence between phrases.

---

# 18. Ambient Audio and Sound Effects

Capture audible:

- music;
- wind;
- room tone;
- traffic;
- machinery;
- footsteps;
- impacts;
- explosions;
- clatter;
- scraping;
- rustling;
- crowd sound;
- mechanical noise.

Give sound its own timestamped entry when it matters.

A long music bed can use a long range.

A short impact can use a short range.

Do not write filler such as:

- No speech.
- No dialogue.
- No music.

If the shot is truly silent, that belongs in Audio concerns.

---

# 19. On-Screen Text

Visible text can be an event.

Examples:

- title;
- subtitle;
- sign;
- game banner;
- UI message;
- lower third;
- burned-in caption.

Add a timestamped Action & Audio line when the text appears or changes.

Quote the visible text.

Example:

> On-screen text reads "I COULDN'T FORGET HER."

If several lines are one simultaneous text block, keep them together clearly rather than creating misleading separate speech-like quotes.

---

# 20. Timestamp Rules

## Rule 1: Accuracy to 0.1 seconds

Every important event must start and end within 0.1 seconds of the real event.

## Rule 2: Timestamps come first

Correct:

> [4.3s-5.5s] C2 says off-screen, "Have we got another bucket?"

Wrong:

> C2 says off-screen [4.3s-5.5s], "Have we got another bucket?"

## Rule 3: Stay inside the shot

An event cannot start before its shot begins.

An event cannot end after its shot ends.

Clip long music or ambience to shot boundaries if needed.

## Rule 4: No duplicate full windows

Within a shot, avoid two entries with the exact same start and end if the live validator rejects duplicate windows.

Two lines may share:

- the same start;
- or the same end;

but a duplicated full pair can trigger a blocker.

Never fix this with fake 0.1-second shifts.

Recheck the real boundaries.

## Rule 5: Overlap is allowed

Different events can overlap.

The footage decides.

## Rule 6: Do not force "nice" timestamps

Do not choose 2.0-3.0 because it looks clean.

Choose the actual event range.

---

# 21. Character-ID and Pronoun Rule

Current critical rule:

> **No pronouns outside quoted dialogue.**

Avoid:

- he;
- she;
- they;
- his;
- her;
- him;
- them.

Use IDs.

Bad:

> C1 raises his hand.

Preferred live-tool style:

> C1 raises the right hand.

Bad:

> He turns toward C2.

Good:

> C1 turns toward C2.

Quoted dialogue preserves the speaker's exact words.

Older Golden Examples may still contain pronouns. Do not copy that older syntax when the live tool flags it.

---

# 22. Left and Right

For position in the image, use:

- screen-left;
- screen-right.

Example:

> C2 stands screen-left of C3.

Use a character's own left/right only when the body side is clear and that detail matters.

Avoid bare:

> C2 moves left.

That can be ambiguous.

---

# 23. Playback Speed

Every shot needs a playback-speed value.

Use the exact option offered by the current tool.

Conceptually:

- regular;
- slow motion;
- accelerated / fast.

Do not confuse fast movement with accelerated playback.

Do not confuse motion blur with slow motion.

If speed changes inside a shot, add a Speed Change entry with start and end.

---

# 24. The 9 Reviewer Checklist Questions

Before marking a task reviewed, answer all nine.

## 1. Detail

Does the caption capture enough scene, object, motion, and speech detail for someone to reconstruct the clip?

## 2. Timestamp accuracy

Is every inline timestamp within 0.1 seconds of the real event?

## 3. Duplicate windows

Is the caption free of duplicate identical timestamp ranges?

## 4. Character labels

Does the caption use C-IDs instead of pronouns where required?

## 5. Playback speed

Are playback-speed values and speed changes correct?

## 6. Camera movement

Are camera moves complete, accurate, timestamped, and placed in Camera Movements?

## 7. Redundancy

Have repeated or duplicate lines been removed?

## 8. Completeness

Are all character actions and all audible dialogue captured?

## 9. Overlapping speech

Do overlapping speakers have explicitly overlapping ranges?

A reviewer should not mark the caption reviewed until all nine pass.

---

# 25. The 16 Additional Reviewer Checks

## Check 1: Ghost characters and objects

Every C-ID and O-ID should be used later.

Remove unused IDs or add the missing real reference.

## Check 2: Cameraman / off-screen filmer

If the filmer becomes visible or audible, consider whether a normal C-ID is needed.

## Check 3: Watermarks and visual artifacts

Put watermarks and recording defects in Visual concerns.

## Check 4: Unintelligible audio

Transcribe what is audible.

Use [inaudible] only for the missing portion.

Do not guess.

## Check 5: Concern-field test

Concerns should describe real recording or processing issues.

They should not contain reviewer notes.

## Check 6: Vocal-profile hedging

If accent, race, or another trait is uncertain, omit it.

Do not write empty hedges.

## Check 7: Ghost quotes

Every quote must be audible and properly punctuated.

Delete lip-read or carried-over speech that is not actually heard.

## Check 8: Missed speech

Capture every audible line.

## Check 9: Ambient audio

Do not leave strong ambient sound or music uncaptioned.

## Check 10: Vocal profiling

For human voice, add useful tone, age, accent, or voice quality when clear.

## Check 11: Frame-level transition QA

Check every cut.

Do not auto-label all transitions Hard cut.

## Check 12: Fade-frame attribution

Visible outgoing content during a fade belongs to the outgoing shot.

## Check 13: Camera accuracy

Split direction changes and timestamp them.

## Check 14: Major-action completeness

Capture the meaningful motion, affected object, direction, and visible reaction.

## Check 15: Diarization cascade

When the speaker changes, review the related gesture, timing, and action description too.

## Check 16: Timestamp and field placement

Timestamps go before descriptions.

Speech belongs in timestamped Action & Audio entries.

---

# 26. Final Review Suggestions: Reviewer Decision Rule

The system may show suggestions such as:

- missing detail;
- wrong attribution;
- unclear wording;
- duplicate range;
- wrong camera placement;
- vague character;
- missing dialogue;
- possible transition error.

These are **suggestions**.

Do not treat them as source truth.

For every suggestion:

### Resolve

Use Resolve when the video supports the criticism.

Examples:

- a real speech line is missing;
- C3 is clearly the speaker, not C2;
- "left" should be "screen-left";
- two entries really duplicate the same event;
- a shot boundary is wrong.

### Ignore

Use Ignore when the suggestion is not supported by the clip.

Examples:

- the evaluator asks to identify a hand, but the hand cannot be linked to a character;
- it requests a sound that is not audible;
- it suggests a speaker only because a mouth is moving;
- it assumes a transition that frame review disproves.

### Never force a fix

Do not invent:

- a character ID;
- a word;
- a speaker;
- an object;
- a timestamp;
- a sound;

just to satisfy a suggestion.

The reviewer is the final human evidence check.

---

# 27. Common Reviewer Failure Modes

## Blindly trusting the seed

A pre-filled caption can be wrong.

Check it.

## Blindly accepting Final Review

Suggestions can also be wrong.

Check them.

## Missing speech

This is a major issue.

Always review audio.

## Wrong speaker

A speaker mistake can break several linked actions.

Fix the whole sequence.

## Lip-reading dialogue

Visible mouth movement is not enough to create a quote.

Use actual audio.

## Wrong hand

Do not assign right/left or ownership from one unclear frame.

## Missed jump cut

Same person and setting does not mean one continuous shot.

Look for a position jump.

## Fake hard cut

A bright flash, pan, obstruction, or exposure shift can look like a cut.

Check adjacent frames.

## Camera movement inside Action & Audio

Move it to Camera Movements.

## Duplicate Action & Audio line

Remove repeated text even when the timestamp differs unless two real separate events exist.

## Duplicate timestamp pair

Re-measure the true boundaries.

Do not fake a shift.

## Empty audio filler

Delete:

- No speech.
- No dialogue.
- No ambient sound.

Describe real sound or omit the absence.

## Internal reviewer notes in customer fields

Never leave:

- needs verification;
- seed says;
- evaluator flagged;
- likely wrong;
- provisional;
- review later.

The final caption must read like finished caption data.

---

# 28. Golden Example Lessons

The Golden Examples set the standard for **how we write the caption**, not only timing and event coverage.

Every reviewer should actively use them as a reference while reviewing. When unsure how much detail, how to split actions, how to represent overlap, how to describe a scene, or how to structure audio, compare the task to the Golden Examples.

Key lessons:

## Frame-matched timing

The example pages show start and end frames beside every entry.

The event truly begins on the start frame.

The event truly ends on the end frame.

That is the timestamp standard.

## Overlap is normal

Multiple events can occur at once.

Do not merge them merely because they overlap.

## Camera is separate

Camera movement gets its own timed entry.

## Audio is real event data

Music, impacts, whooshes, footsteps, scraping, wind, and dialogue are all captionable events.

## Granularity follows the video

The one-shot crayfish example contains a 0.1-second hand-release entry.

That is acceptable when the action itself lasts 0.1 seconds.

## Dense shots need multiple entries

The game trailer uses many entries during one continuous tracking shot because different characters act at different times.

Shot count does not control entry count.

---


---

# 28A. Complete Golden Examples PDF Coverage

This handbook is based on the **entire 31-page Golden Examples PDF**, not only the first page or a few sample entries.

The PDF contains three layers that reviewers must learn:

1. the general Golden Example rules on pages 1-2;
2. Golden Example 1, the dense three-shot game trailer, across pages 2-16;
3. Golden Example 2, the one-shot handheld crayfish clip, across pages 16-30.

Page 31 closes the source.

The goal is not to memorize the example wording. The goal is to internalize the **caption-writing behavior** shown across the complete examples.

## A. Pages 1-2: core Golden Example rules

The opening pages establish these standards:

- every timestamp is matched to the real start and end frames;
- timestamps are not rough estimates;
- camera movement is kept separate from character action;
- overlapping events keep overlapping ranges;
- audio gets its own timed entries;
- granularity follows the footage, not a target entry count;
- every referenced C# and O# is defined in Overview;
- speech is transcribed rather than summarized;
- unclear speech is flagged instead of silently dropped.

Reviewer lesson:

> A caption is an event ledger, not a paragraph summary.

## B. Golden Example 1: game trailer

**Clip structure:** three shots, about fifteen seconds, 27 timestamped entries.

This example teaches how to review a visually dense montage with many characters, effects, sounds, and camera movement.

### Overview lesson

The Overview defines:

- C1 through C8 as visible game characters;
- C9 as an unseen narrator;
- O1 as a floating digital card;
- O2 as C4's hovering support device;
- O3 as C5's waist-height device;
- O4 as the yellow zipline.

The Overview also establishes:

- the abstract opening space;
- the ruined futuristic street;
- the final industrial exterior;
- saturated cyan/orange visual treatment;
- electronic music;
- narrator presence;
- C8's later spoken line.

Reviewer lesson:

> In a dense clip, Overview must create a reliable identity map before the shot entries begin.

### Shot 1 lesson: separate visual motion, object action, and sound

Shot 1 is 0.0-3.2.

The example separates:

- camera push-in across the shot;
- opening synthesized impact/rise;
- O1's rotation;
- the electronic music bed.

These overlap but remain separate entries because they are different event sources.

Reviewer lesson:

> Do not merge camera movement, object movement, music, and sound effects merely because they happen together.

### Shot 2 lesson: one long tracking shot can contain many events

Shot 2 is 3.2-13.6.

Camera movement is divided into three phases:

- entry through the teal energy transition;
- long forward/rightward tracking past the lineup;
- final rapid push toward C8.

The shot then uses separate Action & Audio entries for:

- C1 appearing during the energy reveal;
- transition whoosh;
- continuous music;
- C2's pose/hand movement;
- C9 narration;
- C3's pose;
- C4 hovering and raising both hands;
- C5 raising an arm;
- C6 dropping and landing;
- C7 advancing;
- C8 spreading the arms;
- rising electronic crackle;
- C8's spoken line.

Reviewer lesson:

> One continuous shot can need many entries. Shot count does not control detail level.

### Character sequencing lesson

As the camera reaches each character, the example gives that character an individual timed entry.

This prevents a vague line such as:

> "The camera passes several characters."

Instead, the caption records what each character does when the camera reaches that character.

Reviewer lesson:

> Dense lineup shots should be reviewed character by character.

### Narration lesson

C9's narration receives its own long speech range while other character actions occur underneath it.

Reviewer lesson:

> Long narration does not replace the need to caption simultaneous visual actions.

### Shot 3 lesson: action, camera, music, mechanical sound, explosion, and character motion all coexist

Shot 3 is 13.6-15.0.

The example separately captures:

- rapid camera tracking;
- C2 entering on O4;
- continuing music;
- zipline/mechanical wind sound;
- explosion visual;
- explosion sound;
- C1 following on O4.

Reviewer lesson:

> Visual explosion and explosion sound can have different ranges. Treat them as separate events when their timing differs.

### Golden Example 1 reviewer takeaways

Before approving a similar dense clip, verify:

- every major visible character has an individual event where needed;
- narrator speech does not hide visual events;
- transition SFX are not merged into camera movement;
- music beds are timed independently;
- camera movement is segmented when its phase changes;
- explosion visual and explosion sound are timed independently when needed;
- one shot can contain many overlapping event layers.

---

## C. Golden Example 2: one continuous handheld crayfish clip

**Clip structure:** one shot for the full sixteen seconds, with 27 total timed entries across camera and Action & Audio.

This example is crucial because it proves that **camera changes do not automatically create new shots**.

The camera changes:

- direction;
- framing;
- height;
- angle;
- subject emphasis.

Yet the footage stays one continuous shot.

Reviewer lesson:

> A framing change is not a cut. A camera rotation is not a cut. A tilt is not a cut.

### Overview lesson

The example defines:

- C1 as the visible adult;
- C2 as the off-screen child speaker;
- O1 as the bucket;
- O2 as the crayfish;
- O3 as the pickup truck;
- O4 as the cooler.

The Overview describes:

- the canal-side dirt road;
- fields and vegetation;
- truck placement;
- sunlight;
- overall handheld style;
- recurring speech;
- recurring handling sounds;
- accent/overlap difficulty in Audio concerns.

Reviewer lesson:

> Overview Audio and Audio concerns can summarize recurring speech/audio conditions, while exact dialogue remains in timed entries.

### Camera lesson

The single shot includes multiple Camera Movement entries.

The example times distinct movement phases such as:

- moving right;
- moving upward and forward;
- rotating right;
- moving downward;
- another downward move near the end.

Reviewer lesson:

> One continuous shot may have several camera entries. Split camera phases instead of inventing shot boundaries.

### Micro-action lesson

The strongest lesson in the second example is granularity.

The example includes a **0.1-second hand-release event**.

It separately records:

- holding the bucket;
- releasing one hand;
- walking while holding with the other hand;
- raising the right arm;
- lowering/swinging the right arm;
- reaching toward the cooler;
- opening the cooler;
- releasing the cooler lid;
- reaching back toward the bucket;
- changing hand placement on the bucket;
- pouring crayfish;
- reaching toward a crayfish near the end.

Reviewer lesson:

> Small hand-state changes matter when they change object contact or action state.

### Object continuity lesson

The example carefully tracks:

- O1 bucket;
- O2 crayfish;
- O3 truck;
- O4 cooler.

The caption records when C1:

- holds O1;
- opens O4;
- repositions grip on O1;
- pours O2 from O1 into O4;
- reaches into O4.

Reviewer lesson:

> Object ownership, contact, release, and transfer should be treated as real state changes.

### Speech lesson

The example includes:

- overlapping speech near the beginning;
- off-screen speech from C2;
- cut-off speech from C1;
- repeated/filler words;
- one brief line during object handling;
- later overlap where C1 is difficult to hear because C2 speaks over C1.

Reviewer lesson:

> Do not clean up natural speech. Preserve repeats, cutoffs, overlaps, and unclear portions.

### Inaudible speech lesson

The example shows partial recovery rather than deleting a whole line.

When part of speech cannot be recovered:

- keep the audible portion;
- mark the missing portion as [inaudible] under current rules;
- document the real cause in Audio concerns if useful.

Reviewer lesson:

> Missing one word does not justify dropping an entire audible utterance.

### Ambient and object-sound lesson

The example separately captures:

- bucket/plastic scraping;
- wind;
- footsteps on dirt;
- crayfish/plastic/claw sounds.

Reviewer lesson:

> Natural sound is part of the event record, not decoration.

### Long action vs short action lesson

Some entries last several seconds because the action truly continues.

Other entries last only 0.1 seconds because the state change is brief.

Reviewer lesson:

> Never split just to make entries short. Never merge just to make the caption simple.

The footage decides.

### Golden Example 2 reviewer takeaways

Before approving a similar handheld clip, verify:

- no false cuts were created from camera movement;
- camera phases are timed separately;
- hand contact and release are precise;
- object transfer is fully tracked;
- off-screen speech has a real C-ID;
- overlapping speech keeps separate ranges;
- speech cutoffs and repeats are preserved;
- ambient sound is not omitted;
- micro-actions are not lost inside broad lines.

---

## D. Complete Golden Example behavior checklist

The entire PDF teaches the reviewer to expect all of these behaviors:

### Overview
- Complete identity map.
- Useful object map.
- Reconstructable Scene.
- Visual Style.
- Recurring Audio.
- Real Visual concerns.
- Real Audio concerns.

### Shots
- Correct shot boundaries.
- Correct transitions.
- Detailed Camera field.
- Timed Camera Movements.
- Static Scene facts.
- Dynamic Action & Audio entries.
- Playback speed.

### Actions
- Character-specific.
- Object-specific.
- Body-part specific when useful.
- Contact/release states captured.
- Direction clear.
- Real start/end.

### Audio
- Dialogue verbatim.
- Speaker identified.
- Off-screen status used.
- Tone only when audible.
- Overlap preserved.
- Inaudible portions flagged.
- Music timed.
- Ambience timed.
- SFX timed.

### Camera
- Separate from character action.
- Split by real movement phase.
- Do not confuse movement with editing.

### Timing
- Start on the true beginning.
- End on the true ending.
- Allow overlap.
- Use short entries when the event is short.
- Use long entries when the event is long.

### Review judgment
- Do not force every clip to look like one Golden Example.
- Match the **level of evidence and granularity**, not the exact number of entries.
- Use the dense game example for complex multi-event footage.
- Use the handheld example for continuous action, object handling, overlapping speech, and camera movement without cuts.

---

# 28B. Golden Example Rule for Future Reviewer Tasks

For every future Manuscript II reviewer task:

1. Read the task video and seed.
2. Decide KEEP, FIX, or REBUILD.
3. Identify which Golden Example pattern is most useful:
   - dense multi-shot / multi-character;
   - continuous handheld / object-handling;
   - or a mix.
4. Use that Golden Example as the writing benchmark.
5. Follow current live-tool syntax when it differs from older example syntax.
6. Before marking reviewed, compare the finished caption against the full Golden Example standard.

The question is not:

> "Did we write enough?"

The question is:

> "Does this caption show the same level of event awareness, timing discipline, audio coverage, object continuity, and camera separation as the Golden Examples?"

If not, continue reviewing.


# 29. Reviewer Scoring Mindset

The supplied reviewer rubric uses:

## Score 5

No issues.

Everything important is captured.

Timestamps are accurate.

## Score 4

One minor issue.

Example:

- one subtle gesture missed;
- one small timestamp error;
- minor wording issue.

## Score 3

Several minor issues together.

## Score 2

One major issue or many minor issues.

Examples:

- missed framing-changing camera move;
- wrong hand;
- wrong character;
- missing speech.

## Score 1

Multiple major issues.

As the reviewer, the goal is not to submit a 3 or 4.

The goal is to **fix the caption until it meets the 5-level standard**.

---

# 30. Reviewer Quick Pass for Every Shot

For each shot ask:

### Boundary
- Is the start correct?
- Is the end correct?
- Is the cut type correct?

### Camera
- Is framing correct?
- Is angle correct?
- Does the camera move?
- Is every camera movement timestamped?

### Scene
- Is the setting complete?
- Are character positions correct?
- Are held objects correct?
- Is Scene free of dynamic action?

### Action
- Is every major movement included?
- Are reactions included?
- Are hands and objects assigned correctly?
- Is left/right clear?

### Speech
- Is every audible line included?
- Is the speaker correct?
- Is the wording verbatim?
- Is tone useful?
- Is off-screen status correct?
- Are overlaps preserved?

### Sound
- Music?
- Ambient sound?
- SFX?
- Impact?
- Silence/dropout?

### Timing
- Does every entry start correctly?
- Does every entry end correctly?
- Is every range within the shot?
- Are any full ranges duplicated?

### Playback
- Regular?
- Slow motion?
- Accelerated?
- Any mid-shot change?

---

# 31. Reviewer Quick Pass for the Overview

## Characters
- All real people defined?
- All off-screen speakers defined?
- Any ghost characters?
- Description consistent across shots?

## Objects
- All prominent recurring objects defined?
- Any duplicate IDs?
- Any unused IDs?

## Scene
- Enough detail to reconstruct the settings?
- Stable facts only?

## Style
- Lighting?
- color?
- depth?
- camera look?
- aspect ratio?

## Audio
- recurring music?
- recurring speakers?
- recurring ambience?

## Visual concerns
- watermark?
- subtitles?
- bars?
- shake?
- blowout?
- artifacts?

## Audio concerns
- unclear words?
- clipping?
- silence?
- wind?
- overlap?
- dropout?

---

# 32. Pre-Submit Reviewer Checklist

Do not submit until all are true.

- [ ] Correct video ID.
- [ ] Full clip watched with sound.
- [ ] Pre-filled caption read end to end.
- [ ] Every real shot checked.
- [ ] Every transition checked.
- [ ] Every character checked.
- [ ] Every object checked.
- [ ] No ghost C-IDs.
- [ ] No ghost O-IDs.
- [ ] Every major action captured.
- [ ] Every reaction captured.
- [ ] Every audible speech line captured.
- [ ] Every speaker attribution checked.
- [ ] Every quote matches audible speech.
- [ ] [inaudible] used only when needed.
- [ ] Every important sound captured.
- [ ] Every camera movement is in Camera Movements.
- [ ] Every camera movement has a real timestamp.
- [ ] Every action and speech range is within 0.1s.
- [ ] No duplicate full timestamp pairs.
- [ ] No timestamps outside shot boundaries.
- [ ] No vague left/right.
- [ ] No prohibited pronouns outside quoted dialogue.
- [ ] Every playback-speed value checked.
- [ ] Every on-screen text event checked.
- [ ] Watermarks and visual defects are in Visual concerns.
- [ ] Audio defects are in Audio concerns.
- [ ] No reviewer notes remain in caption fields.
- [ ] No duplicate or redundant lines remain.
- [ ] All 9 Reviewer Checklist items pass.
- [ ] All 16 Additional Checks pass.
- [ ] Overview marked reviewed.
- [ ] Every Shot marked reviewed.
- [ ] Tool flags resolved.
- [ ] Final Review suggestions judged against the video.
- [ ] Result code copied without edits.
- [ ] Result code tested in a fresh tab when possible.

---

# 33. Rules to Remember During Review

## Rule A

**The video and audio decide what happened.**

## Rule B

**The reviewer decides whether a suggestion is valid.**

## Rule C

**Every major event needs a timestamp.**

## Rule D

**Accuracy matters more than neat-looking ranges.**

## Rule E

**Overlap is allowed and often required.**

## Rule F

**No important speech can be missing.**

## Rule G

**No speaker should be guessed.**

## Rule H

**Camera movement has one home: Camera Movements.**

## Rule I

**Do not leave duplicate lines.**

## Rule J

**Do not leave internal QA notes in the final caption.**

## Rule K

**If a character/object ID is defined, it must be used.**

## Rule L

**Do not submit until you would personally give the caption a 5/5.**

---

# 34. Reviewer Decision Examples

## Example 1: Final Review says identify an arm

The arm is cropped and cannot be linked to C1 or C2.

**Reviewer decision:** Ignore the forced attribution. Keep the description neutral or remove the weak detail.

## Example 2: Final Review says dialogue is missing

The audio clearly contains speech.

**Reviewer decision:** Resolve. Add the exact words, correct speaker, tone, and real speech window.

## Example 3: Final Review says C2 is speaking because C2's mouth moves

The audio is clearly C3.

**Reviewer decision:** Keep C3 as speaker. Mouth movement alone is not enough.

## Example 4: Two lines have the same 0.0-1.0 range

The actions actually start and stop at different times.

**Reviewer decision:** Re-measure and use real ranges.

## Example 5: Two events really do share the same visible range

The live tool blocks duplicate windows.

**Reviewer decision:** Recheck whether both lines are independently needed. Merge only if they are one true event. Otherwise measure their real distinct boundaries. Never invent a fake offset.

## Example 6: Seed says Hard cut

Frames show both scenes blended.

**Reviewer decision:** Change to Cross dissolve.

## Example 7: Seed defines C5 but C5 never appears or speaks

**Reviewer decision:** Remove C5 unless the video proves C5 belongs.

---

# 35. What We Should Give ChatGPT for Future Reviewer Tasks

For a new Manuscript II reviewer task, provide:

1. actual video;
2. pre-filled Overview and Shots;
3. Final Review suggestions, if any;
4. validator or export blockers, if any;
5. any task-specific notes.

The review should then focus on:

- real shot structure;
- cast;
- objects;
- actions;
- camera;
- speech;
- ambient sound;
- exact timestamps;
- validator safety.

The final response should tell the reviewer exactly what to enter or change.

Do not replace evidence review with generic advice.

---

# 36. Source Notes and Known Documentation Conflicts

The supplied materials contain a few older/newer format differences.

## [inaudible] vs `<unintelligible>`

The Additional Checks use **[inaudible]** as the current convention.

Use **[inaudible]**.

Older examples showing `<unintelligible>` should not override the newer check.

## Movements + Audio vs Action & Audio

Some older pages describe separate Movements and Audio fields.

The current live tool uses **Action & Audio**.

Follow the current live tool.

## Pronouns in old examples

Some Golden Example wording uses pronouns.

The current critical rule says to use C-IDs and avoid pronouns outside quoted dialogue.

Follow the current rule.

## Playback-speed labels

Documentation may show `slow_motion`, `regular`, and `accelerated`, while the UI may display human-readable labels.

Use the exact option provided by the current tool.

---

# 37. Source Basis

This handbook was compiled from the material supplied for Manuscript II, including:

- Manuscript II project introduction and onboarding text;
- Differences vs Manuscript I;
- Reviewer Checklist;
- Task Overview;
- Video workflow walkthrough;
- Examples & Video Walkthrough;
- Additional Checks;
- current Master Reference;
- Golden Examples PDF, including the dense game trailer example;
- Golden Examples PDF, including the one-shot crayfish example.

The Golden Examples establish the frame-level timing standard.

The Reviewer Checklist and Additional Checks establish the pre-submit review duties.

The live tool behavior controls fields and validator handling.

---

# Final Reviewer Rule

> **Do not ask whether the seed can be repaired. Ask what the best caption for the actual video should be.**

Keep correct seed content. Fix usable content. Rebuild bad content.

Then compare the finished result to the Golden Examples.

> **Do not ask whether the caption looks good. Ask whether every important line can survive comparison with the actual video and audio.**

If not, keep reviewing.
