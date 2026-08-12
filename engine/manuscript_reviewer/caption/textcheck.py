"""Robust text checks shared by the M2 and platform-semantic validators.

Real parsers, not naive counts (§79/§80):

* Sentence splitting ignores decimal numbers, common abbreviations, and
  everything inside quoted spans (a quote may contain many sentences and still
  be ONE speech act).
* Quote-span detection pairs double quotes (straight or curly) only —
  apostrophes/contractions ("I've") are never treated as quote delimiters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "st", "vs", "etc", "approx", "no", "e.g", "i.e",
        "jr", "sr", "prof", "fig", "min", "max", "sec",
    }
)

_QUOTE_OPENERS = {'"': '"', "“": "”"}


@dataclass(frozen=True)
class QuoteSpan:
    start: int  # index of the opening quote character
    end: int  # index one past the closing quote character
    text: str  # content between the quotes


def find_quote_spans(text: str) -> list[QuoteSpan]:
    """Double-quote spans only. An unclosed quote yields a span to the end of
    the line (callers may flag it as malformed separately)."""
    spans: list[QuoteSpan] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _QUOTE_OPENERS:
            closer = _QUOTE_OPENERS[ch]
            j = text.find(closer, i + 1)
            if ch == '"' and j == -1:
                j = len(text)
                spans.append(QuoteSpan(i, j, text[i + 1 :]))
                break
            if j == -1:
                spans.append(QuoteSpan(i, len(text), text[i + 1 :]))
                break
            spans.append(QuoteSpan(i, j + 1, text[i + 1 : j]))
            i = j + 1
            continue
        i += 1
    return spans


def strip_quotes(text: str) -> str:
    """Replace each quoted span with a neutral token so sentence/pronoun/verb
    checks never fire inside quoted dialogue."""
    spans = find_quote_spans(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        out.append("QUOTED")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


_TERMINATOR = re.compile(r"[.!?]+")


def _is_abbreviation(text: str, index: int) -> bool:
    """Is the '.' at ``index`` part of an abbreviation or initialism?"""
    word_start = index - 1
    while word_start >= 0 and (text[word_start].isalpha() or text[word_start] == "."):
        word_start -= 1
    word = text[word_start + 1 : index].lower().rstrip(".")
    if word in _ABBREVIATIONS:
        return True
    # Single-letter initialism like "U.S." / "C." — never a sentence break.
    return len(word) == 1


def split_sentences(text: str) -> list[str]:
    """Split prose (with quotes already excluded by the caller when needed)
    into sentences, defending against decimals and abbreviations."""
    working = strip_quotes(text)
    breaks: list[int] = []
    for match in _TERMINATOR.finditer(working):
        idx = match.start()
        end = match.end()
        if working[idx] == ".":
            before = working[idx - 1] if idx > 0 else ""
            after = working[end] if end < len(working) else ""
            if before.isdigit() and after.isdigit():
                continue  # decimal number: 4.3
            if _is_abbreviation(working, idx):
                continue
        # A terminator ends a sentence only when followed by whitespace + an
        # uppercase/idental start, or end of line.
        rest = working[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'“CO["):
            continue
        breaks.append(end)
    sentences: list[str] = []
    cursor = 0
    for brk in breaks:
        chunk = working[cursor:brk].strip()
        if chunk:
            sentences.append(chunk)
        cursor = brk
    tail = working[cursor:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def sentence_count(text: str) -> int:
    return len(split_sentences(text))


def pronoun_hits(text: str, blocklist: list[str]) -> list[str]:
    """Blocked pronouns OUTSIDE quoted dialogue. Never falsely triggers inside
    quotes (§66)."""
    working = strip_quotes(text)
    hits: list[str] = []
    for pronoun in blocklist:
        if re.search(rf"\b{re.escape(pronoun)}\b", working, flags=re.IGNORECASE):
            hits.append(pronoun)
    return hits


_SUBJECT_VERB = re.compile(r"\b(C\d+)\s+(?:[a-z]+ly\s+)?[a-z]+(?:s|es)\b")


def subject_verb_subjects(text: str) -> list[str]:
    """Distinct C-IDs that appear as sentence subjects with a finite verb —
    used to catch two independent subjects hidden in one line (§41/§78)."""
    working = strip_quotes(text)
    seen: list[str] = []
    for match in _SUBJECT_VERB.finditer(working):
        cid = match.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


_CONNECTIVES = re.compile(r"\b(and|then|while|as)\b", re.IGNORECASE)


def hidden_second_action(text: str) -> bool:
    """Connective defense (§41): a connective plus a second independent
    C-subject finite verb suggests two events in one sentence. The words
    themselves are not banned — "holds O1 with both hands" passes."""
    working = strip_quotes(text)
    if not _CONNECTIVES.search(working):
        return False
    return len(subject_verb_subjects(working)) >= 2
