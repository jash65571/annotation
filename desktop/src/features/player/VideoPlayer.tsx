/**
 * Context-only playback of the source video. This player exists so the
 * reviewer can orient themselves — it is NEVER a source of frame truth, and
 * nothing here ever emits a timestamp as a decision. Frame-accurate review
 * happens in the ExactFrameViewer against the ledger.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";

export interface VideoPlayerProps {
  videoPath: string;
  /** Imperative context seek: the player seeks whenever this value changes.
   * Orientation only — never recorded as a decision. */
  seekSeconds?: number;
}

const RATES = [0.5, 1, 2] as const;

export function VideoPlayer({ videoPath, seekSeconds }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<number>(1);

  useEffect(() => {
    let cancelled = false;
    setSrc(null);
    setError(null);
    (async () => {
      try {
        const allowed = await invoke<string>("allow_video_playback", { path: videoPath });
        if (!cancelled) setSrc(convertFileSrc(allowed));
      } catch (raw) {
        if (!cancelled) setError(String(raw));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [videoPath]);

  // Imperative context seek: react to changes of seekSeconds.
  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;
    if (typeof seekSeconds === "number" && Number.isFinite(seekSeconds) && seekSeconds >= 0) {
      video.currentTime = seekSeconds;
    }
  }, [seekSeconds]);

  useEffect(() => {
    const video = videoRef.current;
    if (video !== null) video.playbackRate = rate;
  }, [rate, src]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (video === null) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, []);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.code === "Space" || event.key === " ") {
        event.preventDefault();
        togglePlay();
      }
    },
    [togglePlay],
  );

  return (
    <div
      className="col panel"
      tabIndex={0}
      role="group"
      aria-label="Context video player — not frame truth"
      onKeyDown={onKeyDown}
    >
      <div className="row">
        <span className="badge machine">CONTEXT PLAYBACK — not frame truth</span>
        <span className="faint">Frame truth lives in the exact-frame viewer.</span>
      </div>

      {error !== null ? (
        <div className="muted mono" role="alert">
          Playback unavailable: {error}
        </div>
      ) : src === null ? (
        <div className="faint">Preparing context playback…</div>
      ) : (
        <video
          ref={videoRef}
          src={src}
          style={{ width: "100%", maxHeight: "60vh", background: "var(--bg-inset)" }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          preload="metadata"
        />
      )}

      <div className="row">
        <button
          type="button"
          onClick={togglePlay}
          disabled={src === null}
          aria-label={playing ? "Pause context playback" : "Play context playback"}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <span className="faint">Space plays/pauses when the player is focused.</span>
        <span className="muted">Rate:</span>
        {RATES.map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setRate(value)}
            disabled={src === null}
            aria-label={`Set context playback rate to ${value}x`}
            aria-pressed={rate === value}
            className={rate === value ? "primary" : undefined}
          >
            {value}x
          </button>
        ))}
      </div>
    </div>
  );
}
