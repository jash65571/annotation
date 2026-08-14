"""Unresolved-state ledger: the thing that stops a draft becoming an RTD caption.

AutoScribe used to swallow every failure — an audio-extraction exception became
"no speech", a cut-verification API error became "not a boundary", a reviewer
model returning nothing became "the fresh caption is final". Each of those turns
a *failure to observe* into a *positive factual claim*, which is the single most
dangerous thing a captioning tool can do.

Every such event is now recorded here instead. A run that carries any BLOCKING
entry is a draft, never a deliverable: `readiness()` is what the UI, the CLI and
the tests ask, and it answers "no" until a human clears the list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Blocking = a claim in the caption may be wrong or missing because something
#: could not be observed. Warning = degraded confidence, caption still complete.
BLOCKING = "blocking"
WARNING = "warning"


@dataclass(frozen=True)
class Blocker:
    code: str
    detail: str
    severity: str = BLOCKING
    #: Where in the media this is unresolved, when it is localized.
    start: float | None = None
    end: float | None = None

    def describe(self) -> str:
        where = ""
        if self.start is not None:
            where = (
                f" [{self.start:.1f}s-{self.end:.1f}s]"
                if self.end is not None
                else f" [{self.start:.1f}s]"
            )
        return f"{self.severity.upper()} {self.code}{where}: {self.detail}"


@dataclass
class BlockerLog:
    """Collects unresolved state across a run. Never raises — the caller decides
    what to do — but a non-empty blocking list must prevent RTD."""

    entries: list[Blocker] = field(default_factory=list)

    def add(
        self,
        code: str,
        detail: str,
        *,
        severity: str = BLOCKING,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        self.entries.append(Blocker(code, detail, severity, start, end))

    def add_exception(
        self,
        code: str,
        exc: BaseException,
        *,
        severity: str = BLOCKING,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        self.add(
            code,
            f"{type(exc).__name__}: {exc}",
            severity=severity,
            start=start,
            end=end,
        )

    def extend(self, other: BlockerLog) -> None:
        self.entries.extend(other.entries)

    @property
    def blocking(self) -> list[Blocker]:
        return [b for b in self.entries if b.severity == BLOCKING]

    @property
    def warnings(self) -> list[Blocker]:
        return [b for b in self.entries if b.severity == WARNING]

    def readiness(self) -> tuple[bool, str]:
        """(ready_to_deliver, human-readable reason).

        Ready is ALWAYS False here: AutoScribe output is a draft by
        construction, because no human has approved it yet. Clearing the
        blocking list is necessary, not sufficient — `ready` becomes a
        meaningful signal only once a reviewer signoff is recorded upstream.
        """
        if self.blocking:
            return False, f"{len(self.blocking)} unresolved blocking item(s)"
        return False, "awaiting human review signoff"

    def as_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "code": b.code,
                "detail": b.detail,
                "severity": b.severity,
                "start": b.start,
                "end": b.end,
            }
            for b in self.entries
        ]

    def describe(self) -> str:
        if not self.entries:
            return "No unresolved items."
        return "\n".join(f"  {b.describe()}" for b in self.entries)
