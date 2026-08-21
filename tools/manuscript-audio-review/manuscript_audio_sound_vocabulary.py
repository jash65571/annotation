"""Phase 3C.1: controlled Manuscript sound vocabulary.

Raw model output (PANNs AudioSet-527 labels, CLAP zero-shot prompts) is
never shown to the reviewer directly. This module is the single mapping
layer between "whatever a classifier said" and the small, reviewer-facing
Manuscript candidate-class vocabulary.

Pure standard library. No model dependency -- fully unit-testable without
PANNs/CLAP installed.
"""

# ---------------------------------------------------------------------------
# Controlled vocabulary groups (spec 3C.1)
# ---------------------------------------------------------------------------

HUMAN_NONVERBAL = "human_nonverbal"
AMBIENCE = "ambience"
OBJECT_SFX = "object_sfx"
MUSIC = "music"

GROUPS = (HUMAN_NONVERBAL, AMBIENCE, OBJECT_SFX, MUSIC)

CANDIDATE_CLASSES = {
    HUMAN_NONVERBAL: (
        "applause", "clapping", "cheering", "whoop", "laughter", "chuckle",
        "giggle", "gasp", "sigh", "groan", "scream", "cough",
        "throat_clearing", "humming", "wordless_vocalization", "crowd_noise",
        "speech_babble",
    ),
    AMBIENCE: (
        "room_ambience", "outdoor_ambience", "traffic", "wind", "ocean",
        "waves", "water", "splashing", "rain", "birds", "crowd_ambience",
        "boat_engine", "vehicle_engine", "machinery",
    ),
    OBJECT_SFX: (
        "impact", "metallic_impact", "metallic_click", "click", "beep",
        "electronic_tone", "hiss", "spray", "footsteps",
        # 3.5: door classes are NEVER collapsed into one generic "door". A
        # physical door open/close, an electronic doorbell chime, a latch
        # click, and a knock are different acoustic sources (O1 can be the
        # physical door while the chime is a different source).
        "door_open_close", "doorbell_chime", "door_latch_click", "door_knock",
        "handling_noise",
    ),
    MUSIC: (
        "music", "background_music", "instrumental_music",
        "singing_vocals_in_music",
    ),
}

# A candidate class this pipeline never treats as speech, no matter how much
# vocal energy a diarization cluster carries (spec 3C.10).
NONVERBAL_CLASSES = frozenset(CANDIDATE_CLASSES[HUMAN_NONVERBAL])
MUSIC_CLASSES = frozenset(CANDIDATE_CLASSES[MUSIC])

# UI source category a candidate class may become. Only genuinely indoor
# room tone maps to a named category; every other ambience defaults to
# "Unidentified sound" -- the live UI's source list is small, and forcing an
# outdoor clip into "Room ambience" would be a false claim (spec 3C.7).
UI_SOURCE_MAP = {
    "room_ambience": "Room ambience",
}
DEFAULT_UI_SOURCE = "Unidentified sound"

# ---------------------------------------------------------------------------
# Raw label -> candidate class mapping (spec 3C.2)
#
# Each entry: (substring-to-match-in-lowercased-raw-label, group, class).
# Order matters: first match wins, so more specific substrings are listed
# before broader ones (e.g. "background music" before "music").
# ---------------------------------------------------------------------------

RAW_LABEL_MAP = (
    # --- human nonverbal ---
    ("applause", HUMAN_NONVERBAL, "applause"),
    ("clapping", HUMAN_NONVERBAL, "clapping"),
    ("cheering", HUMAN_NONVERBAL, "cheering"),
    ("whoop", HUMAN_NONVERBAL, "whoop"),
    ("chuckle", HUMAN_NONVERBAL, "chuckle"),
    ("chortle", HUMAN_NONVERBAL, "chuckle"),
    ("giggle", HUMAN_NONVERBAL, "giggle"),
    ("laughter", HUMAN_NONVERBAL, "laughter"),
    ("laugh", HUMAN_NONVERBAL, "laughter"),
    ("gasp", HUMAN_NONVERBAL, "gasp"),
    ("sigh", HUMAN_NONVERBAL, "sigh"),
    ("groan", HUMAN_NONVERBAL, "groan"),
    ("scream", HUMAN_NONVERBAL, "scream"),
    ("yell", HUMAN_NONVERBAL, "scream"),
    ("shout", HUMAN_NONVERBAL, "scream"),
    ("cough", HUMAN_NONVERBAL, "cough"),
    ("throat clearing", HUMAN_NONVERBAL, "throat_clearing"),
    ("humming", HUMAN_NONVERBAL, "humming"),
    ("hum", HUMAN_NONVERBAL, "humming"),
    ("babble", HUMAN_NONVERBAL, "speech_babble"),
    ("chatter", HUMAN_NONVERBAL, "speech_babble"),
    ("hubbub", HUMAN_NONVERBAL, "crowd_noise"),
    ("crowd", HUMAN_NONVERBAL, "crowd_noise"),

    # --- music (before generic "music" to catch specific variants first) ---
    ("background music", MUSIC, "background_music"),
    ("instrumental", MUSIC, "instrumental_music"),
    ("singing", MUSIC, "singing_vocals_in_music"),
    ("music", MUSIC, "music"),

    # --- ambience / environment ---
    ("inside, small room", AMBIENCE, "room_ambience"),
    ("inside, large room", AMBIENCE, "room_ambience"),
    ("room tone", AMBIENCE, "room_ambience"),
    ("reverberation", AMBIENCE, "room_ambience"),
    ("outside, urban", AMBIENCE, "outdoor_ambience"),
    ("outside, rural", AMBIENCE, "outdoor_ambience"),
    ("traffic", AMBIENCE, "traffic"),
    ("roadway", AMBIENCE, "traffic"),
    ("wind noise", AMBIENCE, "wind"),
    ("wind", AMBIENCE, "wind"),
    ("ocean", AMBIENCE, "ocean"),
    ("waves", AMBIENCE, "waves"),
    ("surf", AMBIENCE, "waves"),
    ("splash", AMBIENCE, "splashing"),
    ("water", AMBIENCE, "water"),
    ("rain", AMBIENCE, "rain"),
    ("bird", AMBIENCE, "birds"),
    ("motorboat", AMBIENCE, "boat_engine"),
    ("speedboat", AMBIENCE, "boat_engine"),
    ("boat", AMBIENCE, "boat_engine"),
    ("engine", AMBIENCE, "vehicle_engine"),
    ("vehicle", AMBIENCE, "vehicle_engine"),
    ("machinery", AMBIENCE, "machinery"),
    ("machine", AMBIENCE, "machinery"),

    # --- object / SFX ---
    ("metallic click", OBJECT_SFX, "metallic_click"),
    ("metallic", OBJECT_SFX, "metallic_impact"),
    ("clang", OBJECT_SFX, "metallic_impact"),
    ("thud", OBJECT_SFX, "impact"),
    ("bang", OBJECT_SFX, "impact"),
    ("smash", OBJECT_SFX, "impact"),
    ("impact", OBJECT_SFX, "impact"),
    ("beep", OBJECT_SFX, "beep"),
    ("bleep", OBJECT_SFX, "beep"),
    ("electronic tone", OBJECT_SFX, "electronic_tone"),
    ("click", OBJECT_SFX, "click"),
    ("hiss", OBJECT_SFX, "hiss"),
    ("spray", OBJECT_SFX, "spray"),
    ("footstep", OBJECT_SFX, "footsteps"),
    ("walk,", OBJECT_SFX, "footsteps"),
    # 3.5 door split: specific substrings first so "doorbell" never falls
    # through to the generic door mapping.
    ("doorbell", OBJECT_SFX, "doorbell_chime"),
    ("bell", OBJECT_SFX, "doorbell_chime"),
    ("door open", OBJECT_SFX, "door_open_close"),
    ("opening door", OBJECT_SFX, "door_open_close"),
    ("door close", OBJECT_SFX, "door_open_close"),
    ("closing door", OBJECT_SFX, "door_open_close"),
    ("door slam", OBJECT_SFX, "door_open_close"),
    ("slamming", OBJECT_SFX, "door_open_close"),
    ("slam", OBJECT_SFX, "door_open_close"),
    ("door latch", OBJECT_SFX, "door_latch_click"),
    ("latch", OBJECT_SFX, "door_latch_click"),
    ("doorknob", OBJECT_SFX, "door_latch_click"),
    ("knock", OBJECT_SFX, "door_knock"),
    ("door", OBJECT_SFX, "door_open_close"),
    ("rustle", OBJECT_SFX, "handling_noise"),
    ("rustling", OBJECT_SFX, "handling_noise"),
)


def map_raw_label(raw_label):
    """Map one raw model label (PANNs AudioSet label, CLAP prompt topic) to
    a (group, candidate_class) pair, or None if it names nothing in the
    controlled vocabulary.

    Never invents a class not present in CANDIDATE_CLASSES.
    """
    if not raw_label:
        return None

    name = raw_label.strip().lower()

    for substring, group, candidate_class in RAW_LABEL_MAP:
        if substring in name:
            return (group, candidate_class)

    return None


# ---------------------------------------------------------------------------
# CLAP zero-shot prompts (spec 3C.3) -- deterministic, versioned. Each
# worker run must use exactly this list so results are reproducible and
# comparable across runs.
# ---------------------------------------------------------------------------

CLAP_PROMPT_SET_VERSION = "3c-prompts-v2"

CLAP_PROMPTS = (
    {"prompt": "people clapping", "group": HUMAN_NONVERBAL, "candidate_class": "clapping"},
    {"prompt": "an audience applauding", "group": HUMAN_NONVERBAL, "candidate_class": "applause"},
    {"prompt": "a crowd cheering", "group": HUMAN_NONVERBAL, "candidate_class": "cheering"},
    {"prompt": "a person whooping", "group": HUMAN_NONVERBAL, "candidate_class": "whoop"},
    {"prompt": "a person laughing", "group": HUMAN_NONVERBAL, "candidate_class": "laughter"},
    {"prompt": "a person chuckling softly", "group": HUMAN_NONVERBAL, "candidate_class": "chuckle"},
    {"prompt": "a person gasping", "group": HUMAN_NONVERBAL, "candidate_class": "gasp"},
    {"prompt": "a person sighing", "group": HUMAN_NONVERBAL, "candidate_class": "sigh"},
    {"prompt": "a person coughing", "group": HUMAN_NONVERBAL, "candidate_class": "cough"},
    {"prompt": "a crowd of people talking at once", "group": HUMAN_NONVERBAL, "candidate_class": "speech_babble"},
    {"prompt": "wind noise", "group": AMBIENCE, "candidate_class": "wind"},
    {"prompt": "ocean waves and water", "group": AMBIENCE, "candidate_class": "waves"},
    {"prompt": "rain falling", "group": AMBIENCE, "candidate_class": "rain"},
    {"prompt": "birds chirping outdoors", "group": AMBIENCE, "candidate_class": "birds"},
    {"prompt": "city traffic noise", "group": AMBIENCE, "candidate_class": "traffic"},
    {"prompt": "a boat engine running", "group": AMBIENCE, "candidate_class": "boat_engine"},
    {"prompt": "a quiet indoor room tone", "group": AMBIENCE, "candidate_class": "room_ambience"},
    {"prompt": "a metallic click or clink", "group": OBJECT_SFX, "candidate_class": "metallic_click"},
    {"prompt": "footsteps", "group": OBJECT_SFX, "candidate_class": "footsteps"},
    # 3.5: door classes are split -- the physical door open/close, the
    # electronic doorbell chime, a latch click, and a knock are separate
    # candidate classes, never one generic "door".
    {"prompt": "a door opening or closing", "group": OBJECT_SFX, "candidate_class": "door_open_close"},
    {"prompt": "a doorbell chime ringing", "group": OBJECT_SFX, "candidate_class": "doorbell_chime"},
    {"prompt": "a door latch clicking", "group": OBJECT_SFX, "candidate_class": "door_latch_click"},
    {"prompt": "someone knocking on a door", "group": OBJECT_SFX, "candidate_class": "door_knock"},
    {"prompt": "background music", "group": MUSIC, "candidate_class": "background_music"},
    {"prompt": "instrumental music", "group": MUSIC, "candidate_class": "instrumental_music"},
    {"prompt": "singing with music", "group": MUSIC, "candidate_class": "singing_vocals_in_music"},
    # Deliberately included as competing / exclusionary prompts for the
    # music-consensus CONFLICT check (spec 3C.6) -- if one of these wins by
    # a wide margin over the music prompts, that is evidence AGAINST music.
    {"prompt": "speech without music", "group": None, "candidate_class": "speech_no_music"},
    {"prompt": "silence or near silence", "group": None, "candidate_class": "silence"},
)

NON_MUSIC_EXCLUSION_CLASSES = frozenset({"speech_no_music", "silence"})
