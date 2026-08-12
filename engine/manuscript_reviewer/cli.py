"""Manuscript Reviewer CLI. Entry point: ``manuscript-reviewer audit VIDEO``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .models.validation import RunStatus, Severity
from .pipeline import run_audit

app = typer.Typer(
    name="manuscript-reviewer",
    help="Frame-accurate Manuscript II review and evidence engine.",
    no_args_is_help=True,
)
console = Console()


def _status_style(status: RunStatus) -> str:
    return {
        RunStatus.PASS: "bold green",
        RunStatus.REVIEW_REQUIRED: "bold yellow",
        RunStatus.PARTIAL: "bold yellow",
        RunStatus.FAILED: "bold red",
    }[status]


@app.command()
def audit(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Video to audit")],
    seed: Annotated[
        Path | None,
        typer.Option(
            "--seed",
            exists=True,
            dir_okay=False,
            help="Seed caption file (hashed and copied into the run; not parsed in Phase 1)",
        ),
    ] = None,
    extract_frames: Annotated[
        bool,
        typer.Option("--extract-frames", help="Extract every frame as a PNG evidence image"),
    ] = False,
    artifacts_root: Annotated[
        Path,
        typer.Option("--artifacts-root", help="Root folder for run artifacts"),
    ] = Path("artifacts"),
    shot_analysis: Annotated[
        bool,
        typer.Option(
            "--shot-analysis/--no-shot-analysis",
            help="Run the Phase 2 Shot Truth Engine (adjacent metrics, cut candidates, shots)",
        ),
    ] = True,
    candidate_sensitivity: Annotated[
        str,
        typer.Option(
            "--candidate-sensitivity",
            help="Candidate generation sensitivity: high (max recall) / normal / low",
        ),
    ] = "normal",
    extract_shot_evidence: Annotated[
        bool,
        typer.Option(
            "--extract-shot-evidence/--no-extract-shot-evidence",
            help="Render labeled evidence bundles for supported/review candidates",
        ),
    ] = True,
    scdet: Annotated[
        bool,
        typer.Option("--scdet/--no-scdet", help="Include the ffmpeg scdet evidence pass"),
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Produce a provably correct frame ledger and audit artifacts for VIDEO."""
    console_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(console_level)

    console.print("\n[bold]MANUSCRIPT REVIEWER[/bold]")
    console.print("-" * 19)

    result = run_audit(
        video_path=video,
        artifacts_root=artifacts_root,
        seed_path=seed,
        extract_frames=extract_frames,
        shot_analysis=shot_analysis,
        shot_sensitivity=candidate_sensitivity,
        extract_shot_evidence=extract_shot_evidence,
        use_scdet=scdet,
    )

    if result.fatal_error:
        console.print(f"[bold red]FATAL:[/bold red] {result.fatal_error}")
        raise typer.Exit(code=2)

    media = result.media
    if media is not None and media.video_streams:
        v = media.video_streams[0]
        console.print("\n[green]Media verified[/green]")
        console.print(f"Resolution: {v.width}x{v.height}")
        console.print(f"Codec: {v.codec_name}" + (f" ({v.profile})" if v.profile else ""))
        if v.nominal_frame_rate is not None:
            console.print(f"FPS: {v.nominal_frame_rate}")
        if media.container_duration_seconds is not None:
            console.print(f"Duration: {float(media.container_duration_seconds):.6f}s")
        declared = v.declared_frame_count
        console.print(f"Frames declared: {declared if declared is not None else 'n/a'}")
        if result.ledger is not None:
            console.print(f"Frames enumerated: {result.ledger.frame_count}")
        for a in media.audio_streams:
            layout = a.channel_layout or (f"{a.channels}ch" if a.channels else "?")
            rate = f"{a.sample_rate}Hz" if a.sample_rate else "?"
            console.print(f"Audio: {a.codec_name.upper()} {rate} {layout}")
        if not media.audio_streams:
            console.print("Audio: none")

    if result.qc is not None:
        qc = result.qc
        console.print()
        checks = {
            "Frame ledger": "frame_enumeration" in qc.checks_run
            and not any(
                i.rule_id.startswith("P1-LEDGER-00") and i.severity == Severity.FAIL
                for i in qc.issues
            ),
            "Timestamp monotonicity": not any(i.rule_id == "P1-LEDGER-005" for i in qc.issues),
            "Frame accounting": not any(
                i.rule_id in ("P1-COUNT-002",) for i in qc.issues
            ),
        }
        for name, ok in checks.items():
            mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            console.print(f"{name}: {mark}")

        if qc.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for issue in qc.warnings:
                console.print(f"  [yellow]{issue.rule_id}[/yellow] {issue.message}")
        if qc.failures:
            console.print("\n[red]Failures:[/red]")
            for issue in qc.failures:
                console.print(f"  [red]{issue.rule_id}[/red] {issue.message}")

        if qc.frame_count_signals:
            table = Table(title="Frame count signals", show_edge=False)
            table.add_column("Method")
            table.add_column("Count", justify="right")
            for signal in qc.frame_count_signals:
                table.add_row(
                    signal.method,
                    str(signal.count) if signal.count is not None else "n/a",
                )
            console.print()
            console.print(table)

    shot = result.shot_truth
    if shot is not None:
        from .models.shot_truth import CandidateStatus

        console.print("\n[bold]SHOT TRUTH[/bold]")
        console.print("-" * 10)
        console.print(f"\nFrames: {shot.frame_count}")
        console.print(f"Adjacent pairs analyzed: {shot.adjacent_pair_count}")
        console.print(f"\nRaw candidates: {shot.raw_candidate_count}")
        console.print(f"Merged candidates: {shot.merged_candidate_count}")
        console.print(f"\nSupported boundaries: {shot.supported_count}")
        console.print(f"Rejected false positives: {shot.rejected_count}")
        console.print(f"Review required: {shot.review_required_count}")

        def _describe(candidate_status: CandidateStatus, title: str, style: str) -> None:
            selected = [c for c in shot.candidates if c.status == candidate_status]
            if not selected:
                return
            console.print(f"\n[{style}]{title}:[/{style}]")
            for c in selected:
                time_str = (
                    f"{float(c.boundary_time_exact):.6f}s"
                    if c.boundary_time_exact is not None
                    else "unknown time"
                )
                console.print(f"F{c.left_frame_index} -> F{c.right_frame_index}   {time_str}")
                if candidate_status == CandidateStatus.SUPPORTED and c.transition is not None:
                    label = c.transition.manuscript_type or "unresolved"
                    console.print(f"  Likely transition: {label}")
                reasons = ", ".join(r.value for r in c.reason_codes)
                console.print(f"  Reasons: {reasons}")

        _describe(CandidateStatus.SUPPORTED, "Supported", "green")
        _describe(CandidateStatus.REJECTED, "Rejected", "dim")
        _describe(CandidateStatus.REVIEW_REQUIRED, "Review required", "yellow")

        console.print(f"\nProvisional shots: {shot.proposed_shot_count}")
        shot_style = {
            "PASS": "bold green",
            "REVIEW_REQUIRED": "bold yellow",
            "FAILED": "bold red",
        }.get(shot.overall_status, "bold")
        console.print(
            f"\nOverall shot status: [{shot_style}]{shot.overall_status}[/{shot_style}]"
        )

    console.print(f"\nArtifacts:\n{result.run_dir}")
    style = _status_style(result.status)
    status_label = "Overall audit status" if shot_analysis else "Overall Phase 1 status"
    console.print(f"\n{status_label}: [{style}]{result.status.value}[/{style}]\n")

    if result.status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print engine and rules versions."""
    from . import __version__
    from .rules.loader import load_rules

    console.print(f"manuscript-reviewer {__version__}")
    console.print(f"rules {load_rules().version}")


if __name__ == "__main__":
    app()
