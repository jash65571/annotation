/**
 * Waveform context panel: shows the engine's waveform PNG with speech-region
 * markers (rendered defensively from unknown-shaped timeline JSON) plus a
 * context audio element. This is orientation audio only — factual speech
 * review uses the engine's exact review clips, never this player.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getWaveformMetadata, readRunFile } from "../../api/bridge";
import { parseExactSeconds } from "../../lib/exactTime";

export interface WaveformPanelProps {
  runDir: string;
}

interface SpeechRegion {
  start: number;
  end: number;
}

const RATES = [0.5, 1, 2] as const;

/** Coerce an unknown JSON value into seconds: numbers pass through, strings
 * try exact "num/den" first, then plain decimal text. */
function asSeconds(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const exact = parseExactSeconds(value);
    if (Number.isFinite(exact)) return exact;
    const decimal = Number(value);
    if (Number.isFinite(decimal)) return decimal;
  }
  return null;
}

function regionFrom(value: unknown): SpeechRegion | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const start =
    asSeconds(record["start_exact"]) ??
    asSeconds(record["start_seconds"]) ??
    asSeconds(record["start_time_seconds"]) ??
    asSeconds(record["start"]);
  const end =
    asSeconds(record["end_exact"]) ??
    asSeconds(record["end_seconds"]) ??
    asSeconds(record["end_time_seconds"]) ??
    asSeconds(record["end"]);
  if (start === null || end === null || end < start) return null;
  return { start, end };
}

/** Dig speech regions out of an unknown-shaped timeline JSON blob. */
function extractRegions(timeline: unknown): SpeechRegion[] {
  if (typeof timeline !== "object" || timeline === null) return [];
  const record = timeline as Record<string, unknown>;
  for (const key of ["speech_regions", "regions", "speech"]) {
    const candidate = record[key];
    if (Array.isArray(candidate)) {
      const regions: SpeechRegion[] = [];
      for (const item of candidate) {
        const region = regionFrom(item);
        if (region !== null) regions.push(region);
      }
      return regions;
    }
  }
  return [];
}

/** Best-effort duration from unknown timeline JSON. */
function extractDuration(timeline: unknown): number | null {
  if (typeof timeline !== "object" || timeline === null) return null;
  const record = timeline as Record<string, unknown>;
  for (const key of ["duration_exact", "duration_seconds", "duration"]) {
    const seconds = asSeconds(record[key]);
    if (seconds !== null && seconds > 0) return seconds;
  }
  return null;
}

export function WaveformPanel({ runDir }: WaveformPanelProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [waveformUrl, setWaveformUrl] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [regions, setRegions] = useState<SpeechRegion[]>([]);
  const [metaDuration, setMetaDuration] = useState<number | null>(null);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rate, setRate] = useState<number>(1);

  useEffect(() => {
    let cancelled = false;
    setWaveformUrl(null);
    setAudioUrl(null);
    setRegions([]);
    setMetaDuration(null);
    setError(null);
    (async () => {
      try {
        const meta = await getWaveformMetadata(runDir);
        if (cancelled) return;
        setRegions(extractRegions(meta.timeline));
        setMetaDuration(extractDuration(meta.timeline));
        if (meta.waveform_png !== null) {
          const png = await readRunFile(runDir, meta.waveform_png);
          if (!cancelled) setWaveformUrl(png);
        }
        if (meta.source_wav !== null) {
          const wav = await readRunFile(runDir, meta.source_wav);
          if (!cancelled) setAudioUrl(wav);
        }
      } catch (raw) {
        if (!cancelled) setError(String(raw));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runDir]);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio !== null) audio.playbackRate = rate;
  }, [rate, audioUrl]);

  const duration = metaDuration ?? mediaDuration;

  const onWaveformClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const audio = audioRef.current;
      if (audio === null || duration === null || duration <= 0) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (rect.width <= 0) return;
      const fraction = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
      audio.currentTime = fraction * duration;
    },
    [duration],
  );

  return (
    <div className="col panel" role="group" aria-label="Context audio waveform">
      <div className="row">
        <span className="badge machine">
          Context audio — factual speech review uses exact review clips
        </span>
      </div>

      {error !== null ? (
        <div className="muted mono" role="alert">
          Waveform unavailable: {error}
        </div>
      ) : null}

      {waveformUrl !== null ? (
        <div
          onClick={onWaveformClick}
          role="presentation"
          style={{ position: "relative", cursor: "pointer" }}
          title="Click to seek context audio (proportional; waveform spans the full duration)"
        >
          <img
            src={waveformUrl}
            alt="Audio waveform (full duration)"
            style={{ display: "block", width: "100%" }}
          />
          {duration !== null && duration > 0
            ? regions.map((region, index) => {
                const left = Math.min(Math.max((region.start / duration) * 100, 0), 100);
                const width = Math.max(
                  Math.min(((region.end - region.start) / duration) * 100, 100 - left),
                  0.2,
                );
                return (
                  <div
                    key={`${region.start}-${region.end}-${index}`}
                    title={`Speech region ${region.start.toFixed(2)}s – ${region.end.toFixed(2)}s`}
                    style={{
                      position: "absolute",
                      left: `${left}%`,
                      width: `${width}%`,
                      top: 0,
                      bottom: 0,
                      background: "var(--status-pass)",
                      opacity: 0.22,
                      borderLeft: "1px solid var(--status-pass)",
                      borderRight: "1px solid var(--status-pass)",
                      pointerEvents: "none",
                    }}
                  />
                );
              })
            : null}
        </div>
      ) : error === null ? (
        <div className="faint">Loading waveform…</div>
      ) : null}

      {audioUrl !== null ? (
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          style={{ width: "100%" }}
          aria-label="Context audio player"
          onLoadedMetadata={(event) => {
            const seconds = event.currentTarget.duration;
            if (Number.isFinite(seconds) && seconds > 0) setMediaDuration(seconds);
          }}
        />
      ) : error === null ? (
        <div className="faint">Loading context audio…</div>
      ) : null}

      <div className="row">
        <span className="muted">Rate:</span>
        {RATES.map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setRate(value)}
            disabled={audioUrl === null}
            aria-label={`Set context audio rate to ${value}x`}
            aria-pressed={rate === value}
            className={rate === value ? "primary" : undefined}
          >
            {value}x
          </button>
        ))}
        {regions.length > 0 ? (
          <span className="faint">
            {regions.length} machine-lead speech region{regions.length === 1 ? "" : "s"} shown
            (candidates, not facts)
          </span>
        ) : null}
      </div>
    </div>
  );
}
