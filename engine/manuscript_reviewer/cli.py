"""Manuscript Reviewer CLI. Entry point: ``manuscript-reviewer audit VIDEO``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .audio.asr.runtime import ASRConfig
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
    audio_analysis: Annotated[
        bool,
        typer.Option(
            "--audio-analysis/--no-audio-analysis",
            help="Run the Phase 3 Audio Truth Engine",
        ),
    ] = True,
    asr: Annotated[
        bool,
        typer.Option(
            "--asr/--no-asr",
            help="Run local ASR (waveform/energy/spectrogram analysis still runs without it)",
        ),
    ] = True,
    asr_bootstrap: Annotated[
        bool,
        typer.Option(
            "--asr-bootstrap/--no-asr-bootstrap",
            help="Allow uv to create ASR worker environments / download models",
        ),
    ] = True,
    asr_model: Annotated[
        str, typer.Option("--asr-model", help="faster-whisper model")
    ] = "large-v3-turbo",
    asr_language: Annotated[
        str | None,
        typer.Option("--asr-language", help="Force transcription language (default: detect)"),
    ] = None,
    asr_device: Annotated[
        str, typer.Option("--asr-device", help="auto|cpu|cuda")
    ] = "auto",
    asr_compute_type: Annotated[
        str, typer.Option("--asr-compute-type", help="auto|int8|float16|...")
    ] = "auto",
    visual_intelligence: Annotated[
        bool,
        typer.Option(
            "--visual-intelligence/--no-visual-intelligence",
            help="Run the Phase 4 Visual Review Intelligence stage (seed comparison)",
        ),
    ] = True,
    feedback: Annotated[
        Path | None,
        typer.Option(
            "--feedback",
            exists=True,
            dir_okay=False,
            help="Task-specific evaluator feedback (snapshotted and structured; higher "
            "priority than the seed)",
        ),
    ] = None,
    ocr: Annotated[
        bool,
        typer.Option("--ocr/--no-ocr", help="Enable local OCR text evidence (degrades safely)"),
    ] = True,
    visual_anchors: Annotated[
        Path | None,
        typer.Option(
            "--visual-anchors",
            exists=True,
            dir_okay=False,
            help="Human/detector anchors JSON for assisted local tracking "
            "(reserved; tracking slice not yet implemented)",
        ),
    ] = None,
    review_decisions: Annotated[
        Path | None,
        typer.Option(
            "--review-decisions",
            exists=True,
            dir_okay=False,
            help="Human review-decisions JSON (bound to video/rules; stale ones are skipped)",
        ),
    ] = None,
    extract_visual_evidence: Annotated[
        bool,
        typer.Option(
            "--extract-visual-evidence",
            help="Render local visual evidence bundles "
            "(reserved; evidence-bundle slice not yet implemented)",
        ),
    ] = False,
    caption_brain: Annotated[
        bool,
        typer.Option(
            "--caption-brain/--no-caption-brain",
            help="Run the Phase 5 Caption Brain (facts, plan, render, M2/platform/"
            "Golden gates, readiness) from the run's evidence artifacts",
        ),
    ] = True,
    human_facts: Annotated[
        Path | None,
        typer.Option(
            "--human-facts",
            exists=True,
            dir_okay=False,
            help="Human-added caption facts JSON (bound to video/rules; stale "
            "entries are rejected; machine code never creates these)",
        ),
    ] = None,
    final_review: Annotated[
        Path | None,
        typer.Option(
            "--final-review",
            exists=True,
            dir_okay=False,
            help="Manual final-review signoff JSON (required for READY_TO_ENTER; "
            "machine code never fabricates it)",
        ),
    ] = None,
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
        audio_analysis=audio_analysis,
        asr_enabled=asr,
        asr_config=ASRConfig(
            model=asr_model,
            device=asr_device,
            compute_type=asr_compute_type,
            language=asr_language,
            bootstrap=asr_bootstrap,
        ),
        visual_intelligence=visual_intelligence,
        feedback_path=feedback,
        visual_anchors_path=visual_anchors,
        review_decisions_path=review_decisions,
        ocr_enabled=ocr,
        extract_visual_evidence=extract_visual_evidence,
        caption_brain=caption_brain,
        human_facts_path=human_facts,
        final_review_path=final_review,
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

    audio = result.audio_truth
    if audio is not None:
        console.print("\n[bold]AUDIO TRUTH[/bold]")
        console.print("-" * 11)
        if audio.audio_status.value == "NO_AUDIO_STREAM":
            console.print("\nAudio stream: none (valid; no audio analysis performed)")
        else:
            timeline = audio.timeline
            if timeline is not None:
                console.print(
                    f"\nEvidence PCM: {timeline.evidence_sample_rate} Hz "
                    f"{timeline.evidence_channels}ch"
                )
                console.print(f"Samples: {timeline.evidence_sample_count}")
                console.print(
                    f"Timeline offset: {float(timeline.annotation_audio_offset):.6f}s"
                )
            console.print(f"\nEnergy bins: {audio.energy_bin_count}")
            console.print(f"Signal regions: {audio.region_count}")
            console.print(f"Transient candidates: {audio.transient_count}")
            console.print(f"\nfaster-whisper: {audio.asr_status.value}")
            if audio.language is not None and audio.language.language_candidate:
                console.print(
                    f"Language candidate: {audio.language.language_candidate} "
                    f"(machine evidence only, "
                    f"{audio.language.language_review_status.value})"
                )
            console.print(f"WhisperX alignment: {audio.alignment_status.value}")
            console.print(f"\nSpeech regions: {audio.speech_region_count}")
            console.print(f"Review required: {audio.review_item_count}")
            console.print(
                f"Visual boundaries checked for audio continuity: "
                f"{audio.boundaries_checked}"
            )
            audio_style = {
                "PASS": "bold green",
                "REVIEW_REQUIRED": "bold yellow",
                "FAILED": "bold red",
            }.get(audio.overall_status, "bold")
            console.print(
                f"\nOverall audio status: "
                f"[{audio_style}]{audio.overall_status}[/{audio_style}]"
            )

    vi = result.visual_intelligence
    if vi is not None and vi.seed_present:
        console.print("\n[bold]VISUAL REVIEW INTELLIGENCE[/bold]")
        console.print("-" * 26)
        console.print(f"\nSeed parsed: {'PASS' if vi.seed_parsed else 'FAIL'}")
        console.print(f"Seed claims: {vi.seed_claim_count}")
        console.print(f"Frame observations: {vi.frame_observation_count}")
        console.print(f"OCR: {vi.ocr_status.value} (tracks: {vi.ocr_track_count})")
        console.print(
            f"Camera phases: {vi.camera_phase_count} "
            f"(direction reversals: {vi.camera_direction_reversals})"
        )
        console.print("\nShot foundation:")
        console.print(f"  {vi.foundation_status.value}")
        if vi.seed_shot_count is not None or vi.verified_shot_count is not None:
            console.print(f"  Seed shots: {vi.seed_shot_count}")
            console.print(f"  Verified shots: {vi.verified_shot_count}")
        if vi.triage is not None:
            console.print(f"  Triage strategy: {vi.triage.suggested_strategy.value}")
        counts = vi.proposal_counts
        if counts:
            console.print("\nSeed claim proposals:")
            for outcome in ("KEEP", "FIX_ENRICH", "REDO_REBUILD", "HUMAN_DECISION_REQUIRED"):
                console.print(f"  {outcome}: {counts.get(outcome, 0)}")
        console.print(f"\nReview items: {vi.review_item_count}")
        console.print(f"Critical review items: {vi.critical_review_item_count}")
        vi_style = {
            "PASS": "bold green",
            "REVIEW_REQUIRED": "bold yellow",
            "FAILED": "bold red",
        }.get(vi.overall_status, "bold")
        console.print(
            f"\nOverall visual intelligence: [{vi_style}]{vi.overall_status}[/{vi_style}]"
        )

    if result.caption_brain is not None:
        _print_caption_brain_report(result.caption_brain)

    console.print(f"\nArtifacts:\n{result.run_dir}")
    style = _status_style(result.status)
    status_label = "Overall audit status" if shot_analysis else "Overall Phase 1 status"
    console.print(f"\n{status_label}: [{style}]{result.status.value}[/{style}]\n")

    if result.status == RunStatus.FAILED:
        raise typer.Exit(code=1)


def _print_caption_brain_report(output: object) -> None:
    """§97 CLI report. Never prints EXPORT READY unless READY_TO_ENTER."""
    from .caption_brain import CaptionBrainOutput

    assert isinstance(output, CaptionBrainOutput)
    r = output.result
    console.print("\n[bold]CAPTION BRAIN[/bold]")
    console.print("-" * 13)
    console.print(f"\nCaption facts: {r.fact_count}")
    console.print(f"\nEligible: {r.eligible_count}")
    console.print(f"Review required: {r.review_required_count}")
    console.print(f"Ineligible: {r.ineligible_count + r.rejected_count}")
    console.print(f"\nShots: {r.shot_count}")
    console.print(f"Characters: {r.character_count}")
    console.print(f"Objects: {r.object_count}")
    console.print("\nSpeech acts:")
    console.print(f"  Verified: {r.speech_verified_count}")
    console.print(f"  Blocked: {r.speech_blocked_count}")
    console.print("On-screen text:")
    console.print(f"  Verified: {r.text_verified_count}")
    console.print(f"  Blocked: {r.text_blocked_count}")
    console.print("Visual actions:")
    console.print(f"  Verified: {r.action_verified_count}")
    console.print(f"  Blocked: {r.action_blocked_count}")
    console.print("\nM2 validator:")
    console.print(f"  {r.m2_fail_count} FAIL")
    console.print(f"  {r.m2_review_count} REVIEW REQUIRED")
    console.print(f"\nPlatform-semantic: {r.platform_semantic_status}")
    console.print(f"Golden gate: {r.golden_gate_status}")
    if r.unresolved_feedback_high:
        console.print(f"Unresolved HIGH feedback: {r.unresolved_feedback_high}")
    if r.signoff_present:
        console.print(
            f"Final-review signoff: {'STALE' if r.signoff_stale else 'present'}"
        )
    readiness_style = {
        "READY_TO_ENTER": "bold green",
        "READY_FOR_FINAL_REVIEW": "bold cyan",
        "REVIEW_REQUIRED": "bold yellow",
        "BLOCKED": "bold red",
    }.get(r.readiness.value, "bold")
    console.print(
        f"\nFinal: [{readiness_style}]{r.readiness.value}[/{readiness_style}]"
    )
    if r.blockers:
        console.print("\nRemaining before ready:")
        for blocker in r.blockers[:20]:
            console.print(f"  - {blocker}")
        if len(r.blockers) > 20:
            console.print(f"  ... and {len(r.blockers) - 20} more (see review_report.md)")


@app.command()
def finalize(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Existing audit run directory"),
    ],
    review_decisions: Annotated[
        Path | None,
        typer.Option("--review-decisions", exists=True, dir_okay=False),
    ] = None,
    human_facts: Annotated[
        Path | None,
        typer.Option("--human-facts", exists=True, dir_okay=False),
    ] = None,
    final_review: Annotated[
        Path | None,
        typer.Option("--final-review", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Fast re-finalization: rebuild facts/plan/caption/gates from existing
    Phase 1-4 evidence + current human review inputs. Never re-runs media
    analysis."""
    from .caption_brain import CaptionBrainError, finalize_run

    console.print("\n[bold]MANUSCRIPT REVIEWER — FINALIZE[/bold]")
    try:
        output = finalize_run(
            run_dir,
            review_decisions_path=review_decisions,
            human_facts_path=human_facts,
            final_review_path=final_review,
        )
    except CaptionBrainError as exc:
        console.print(f"[bold red]FATAL:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
    _print_caption_brain_report(output)
    total = output.result.stage_timings_seconds.get("caption_brain_total")
    if total is not None:
        console.print(f"\nRe-finalization time: {total:.3f}s")
    console.print(f"\nArtifacts:\n{run_dir / 'caption'}\n")
    if output.result.readiness.value == "BLOCKED":
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
