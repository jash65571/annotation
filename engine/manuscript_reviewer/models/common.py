"""Shared model plumbing: exact-time serialization helpers and base config."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


def _fraction_from_any(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, list | tuple) and len(value) == 2:
        return Fraction(int(value[0]), int(value[1]))
    raise TypeError(f"Cannot interpret {value!r} as an exact rational")


#: Exact rational serialized as the lossless string "num/den" (never a float).
ExactFraction = Annotated[
    Fraction,
    BeforeValidator(_fraction_from_any),
    PlainSerializer(lambda f: f"{f.numerator}/{f.denominator}", return_type=str),
]


class StrictModel(BaseModel):
    """Base for all persisted models: no silent extra fields, exact types."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
