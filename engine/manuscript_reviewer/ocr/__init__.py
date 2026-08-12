"""OCR subsystem (Phase 4): machine on-screen-text *evidence* only.

Local-only, no cloud OCR. OCR is never truth: a single read never becomes final
on-screen text evidence, machine OCR text is never caption-eligible without human
source verification, and Unicode is preserved (never normalized to ASCII).

The Tesseract adapter is optional — if Tesseract is absent, OCR status is
UNAVAILABLE and deterministic text-region detection still works.
"""
