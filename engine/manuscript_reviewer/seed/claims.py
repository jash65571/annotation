"""Atomic seed-claim extraction.

Turns a parsed :class:`SeedDocument` into a list of *atomic* claims (the
extended :class:`~manuscript_reviewer.models.caption.SeedClaim`). A paragraph is
never one giant claim: each defined entity, each shot boundary, each transition,
each timestamped action/overlay/speed field becomes its own independently
reviewable claim.

Claim *importance* (FOUNDATIONAL vs LOCAL) is set here; ``evidence_status`` is
left ``None`` and is filled in later by comparison/matrix building. The two are
different concepts and are never conflated.
"""

from __future__ import annotations

import re

from ..models.caption import ExactTimeRange, SeedClaim
from ..models.review_intelligence import (
    ClaimImportance,
    ClaimReviewStatus,
    SeedClaimType,
    SeedDocument,
    SeedEntry,
    SeedFieldKind,
    SeedSection,
    SeedSectionKind,
)

# Character/object definition lines look like "C1: description" / "O2 - ...".
_DEF_RE = re.compile(r"^\s*@?(?P<id>[CO]\d+)\s*[:\-–—)]", re.IGNORECASE)  # noqa: RUF001

# Protected / unsupported traits: captured so they can be flagged unsupported,
# NEVER visually inferred. Detection is conservative and additive.
_GENDER_WORDS = re.compile(
    r"\b(man|woman|male|female|boy|girl|lady|gentleman|guys?|men|women)\b", re.IGNORECASE
)
_AGE_WORDS = re.compile(
    r"\b(\d+\s*[-\s]?year[-\s]?old|aged?\s+\d+|young|old|elderly|middle[-\s]?aged|teen(?:ager)?|"
    r"child|adult|twenties|thirties|forties|fifties|sixties)\b",
    re.IGNORECASE,
)
_ACCENT_WORDS = re.compile(r"\baccent(?:ed)?\b", re.IGNORECASE)
_EXPLICIT_PROTECTED = re.compile(r"\b(nationality|ethnicity|ethnic|race|racial)\b", re.IGNORECASE)
#: A modest nationality/ethnicity adjective set (extend as needed). Deliberately
#: small — the real guarantee is that the CV side never *infers* these.
_NATIONALITY = re.compile(
    r"\b(American|British|English|Irish|Scottish|French|German|Spanish|Italian|"
    r"Portuguese|Russian|Chinese|Japanese|Korean|Indian|Pakistani|Nigerian|"
    r"Mexican|Brazilian|Canadian|Australian|Dutch|Swedish|Norwegian|Polish|"
    r"Greek|Turkish|Arab|Arabic|African|Asian|European|Latino|Latina|Hispanic|"
    r"Caucasian|Black|White)\b"
)


def _time_range(entry: SeedEntry) -> ExactTimeRange | None:
    start = entry.parsed_start_exact
    end = entry.parsed_end_exact
    if start is None and end is None:
        return None
    if end is None:
        end = start
    if start is None:
        start = end
    # Both are guaranteed non-None here.
    assert start is not None and end is not None
    return ExactTimeRange(start_seconds=start, end_seconds=end)


def _protected_trait_hits(text: str) -> list[str]:
    hits: list[str] = []
    if _EXPLICIT_PROTECTED.search(text):
        hits.append("nationality/ethnicity/race")
    if _NATIONALITY.search(text):
        hits.append("nationality/ethnicity")
    if _GENDER_WORDS.search(text):
        hits.append("gender")
    if _AGE_WORDS.search(text):
        hits.append("age")
    if _ACCENT_WORDS.search(text):
        hits.append("accent")
    return hits


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def next(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}-{self.value:04d}"


def extract_claims(doc: SeedDocument) -> list[SeedClaim]:
    """Extract atomic claims from a parsed seed document."""
    claims: list[SeedClaim] = []
    counter = _Counter()

    # --- MEDIA_ID (foundational) ---
    if doc.video_id is not None:
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field="video_id",
                text=f"Video ID: {doc.video_id}",
                claim_type=SeedClaimType.MEDIA_ID,
                seed_source_line=_video_id_line(doc),
                importance=ClaimImportance.FOUNDATIONAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )

    # --- SHOT_COUNT (foundational) ---
    seed_shots = doc.seed_shot_count
    if seed_shots is not None:
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field="shot_structure",
                text=f"Seed describes {seed_shots} shot(s).",
                claim_type=SeedClaimType.SHOT_COUNT,
                importance=ClaimImportance.FOUNDATIONAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )

    for section in doc.sections:
        if section.kind == SeedSectionKind.CAST:
            claims.extend(_cast_claims(section, counter))
        elif section.kind == SeedSectionKind.OBJECTS:
            claims.extend(_object_claims(section, counter))
        elif section.kind in (SeedSectionKind.SCENE, SeedSectionKind.OVERVIEW):
            claims.extend(_generic_claims(section, counter, SeedClaimType.SCENE_STATE))
        elif section.kind == SeedSectionKind.STYLE:
            claims.extend(_generic_claims(section, counter, SeedClaimType.STYLE_STATE))
        elif section.kind == SeedSectionKind.AUDIO:
            claims.extend(_generic_claims(section, counter, SeedClaimType.SOUND))
        elif section.kind == SeedSectionKind.VISUAL_CONCERNS:
            claims.extend(_generic_claims(section, counter, SeedClaimType.VISUAL_CONCERN))
        elif section.kind == SeedSectionKind.AUDIO_CONCERNS:
            claims.extend(_generic_claims(section, counter, SeedClaimType.AUDIO_CONCERN))
        elif section.kind == SeedSectionKind.SHOT:
            claims.extend(_shot_claims(section, counter))

    return claims


def _video_id_line(doc: SeedDocument) -> int | None:
    for section in doc.sections:
        if section.kind == SeedSectionKind.VIDEO_ID:
            return section.source_line
    return None


def _cast_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        match = _DEF_RE.match(text)
        char_id = match.group("id").upper().lstrip("@") if match else None
        subject = [char_id] if char_id else entry.referenced_character_ids
        if char_id is not None:
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Characters",
                    text=text,
                    claim_type=SeedClaimType.CHARACTER_EXISTS,
                    subject_ids=[char_id],
                    shot_number=None,
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.FOUNDATIONAL,
                    review_status=ClaimReviewStatus.MACHINE_ONLY,
                )
            )
        for trait in _protected_trait_hits(text):
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Characters",
                    text=f"{subject or ['?']}: {trait} trait — {text}",
                    claim_type=SeedClaimType.PROTECTED_TRAIT,
                    subject_ids=list(subject),
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.LOCAL,
                    review_status=ClaimReviewStatus.REVIEW_REQUIRED,
                )
            )
    return claims


def _object_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        match = _DEF_RE.match(text)
        obj_id = match.group("id").upper().lstrip("@") if match else None
        if obj_id is not None:
            claims.append(
                SeedClaim(
                    claim_id=counter.next("CLM"),
                    source_field="Objects",
                    text=text,
                    claim_type=SeedClaimType.OBJECT_EXISTS,
                    object_ids=[obj_id],
                    seed_source_line=entry.source_line,
                    seed_entry_id=entry.entry_id,
                    importance=ClaimImportance.FOUNDATIONAL,
                    review_status=ClaimReviewStatus.MACHINE_ONLY,
                )
            )
    return claims


def _generic_claims(
    section: SeedSection, counter: _Counter, claim_type: SeedClaimType
) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    for entry in section.entries:
        text = entry.value_text or entry.raw_line
        if not text.strip():
            continue
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=section.kind.value,
                text=text,
                claim_type=claim_type,
                subject_ids=entry.referenced_character_ids,
                object_ids=entry.referenced_object_ids,
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                importance=ClaimImportance.LOCAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


_SPEECH_RE = re.compile(r"\b(say|says|said|speak|speaks|shout|shouts|asks?|replies|whisper)\b",
                        re.IGNORECASE)
_TEXT_RE = re.compile(r"\b(on[-\s]?screen text|text reads?|caption reads?|overlay|subtitle)\b",
                      re.IGNORECASE)
_SOUND_RE = re.compile(r"\b(sound|music|noise|beeps?|thud|bang|ambient|sfx)\b", re.IGNORECASE)


def _classify_action_audio(entry: SeedEntry) -> SeedClaimType:
    text = entry.value_text or entry.raw_line
    if _TEXT_RE.search(text):
        return SeedClaimType.ON_SCREEN_TEXT
    if entry.quoted_strings and _SPEECH_RE.search(text):
        return SeedClaimType.SPEECH
    if _SOUND_RE.search(text):
        return SeedClaimType.SOUND
    return SeedClaimType.ACTION


def _shot_claims(section: SeedSection, counter: _Counter) -> list[SeedClaim]:
    claims: list[SeedClaim] = []
    shot = section.shot_number

    # Shot boundary claim from the header (foundational).
    header = next((e for e in section.entries if e.field_label == "Shot header"), None)
    if header is not None and (
        header.parsed_start_exact is not None or header.parsed_end_exact is not None
    ):
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field="Shot boundary",
                text=header.value_text or header.raw_line,
                claim_type=SeedClaimType.SHOT_BOUNDARY,
                shot_number=shot,
                seed_source_line=header.source_line,
                seed_entry_id=header.entry_id,
                seed_time_range=_time_range(header),
                importance=ClaimImportance.FOUNDATIONAL,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )

    for entry in section.entries:
        if entry.field_label == "Shot header":
            continue
        claim_type, importance = _shot_entry_type(entry)
        if claim_type is None:
            continue
        quoted = entry.quoted_strings[0] if entry.quoted_strings else None
        claims.append(
            SeedClaim(
                claim_id=counter.next("CLM"),
                source_field=f"Shot {shot} {entry.field.value}" if shot else entry.field.value,
                text=entry.value_text or entry.raw_line,
                claim_type=claim_type,
                subject_ids=entry.referenced_character_ids,
                object_ids=entry.referenced_object_ids,
                shot_number=shot,
                seed_source_line=entry.source_line,
                seed_entry_id=entry.entry_id,
                seed_time_range=_time_range(entry),
                quoted_text=quoted,
                importance=importance,
                review_status=ClaimReviewStatus.MACHINE_ONLY,
            )
        )
    return claims


def _shot_entry_type(entry: SeedEntry) -> tuple[SeedClaimType | None, ClaimImportance]:
    field = entry.field
    if field == SeedFieldKind.TRANSITION:
        return SeedClaimType.TRANSITION, ClaimImportance.FOUNDATIONAL
    if field == SeedFieldKind.SHOT_START or field == SeedFieldKind.SHOT_END:
        return SeedClaimType.SHOT_BOUNDARY, ClaimImportance.FOUNDATIONAL
    if field == SeedFieldKind.CAMERA:
        return SeedClaimType.CAMERA_FRAMING, ClaimImportance.LOCAL
    if field == SeedFieldKind.CAMERA_MOVEMENTS:
        return SeedClaimType.CAMERA_MOVEMENT, ClaimImportance.LOCAL
    if field == SeedFieldKind.SCENE:
        return SeedClaimType.SCENE_STATE, ClaimImportance.LOCAL
    if field == SeedFieldKind.PLAYBACK_SPEED or field == SeedFieldKind.SPEED_CHANGES:
        return SeedClaimType.PLAYBACK_SPEED, ClaimImportance.LOCAL
    if field == SeedFieldKind.ACTION_AUDIO:
        return _classify_action_audio(entry), ClaimImportance.LOCAL
    # FREEFORM (and any future field) produces no atomic claim on its own.
    return None, ClaimImportance.LOCAL
