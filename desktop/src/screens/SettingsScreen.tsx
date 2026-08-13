/** Settings + About/Privacy (spec §73–74). Small on purpose. */

import { useEffect, useState } from "react";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import { health } from "../api/bridge";
import type { HealthPayload } from "../api/types";

export function SettingsScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const [info, setInfo] = useState<HealthPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  useApp();

  useEffect(() => {
    health()
      .then(setInfo)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="col" style={{ padding: "var(--gap-lg)", maxWidth: 860, margin: "0 auto" }}>
      <h1 style={{ fontSize: 18, margin: 0 }}>Settings &amp; system status</h1>

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>System status</h2>
        {error && <span className="badge fail">Engine unavailable: {error}</span>}
        {info && (
          <dl className="col" style={{ gap: 4, margin: 0 }}>
            <Row label="Engine version" value={info.engine_version} />
            <Row label="Rules version" value={info.rules_version} />
            <Row label="Bridge protocol" value={String(info.protocol_version)} />
            <Row
              label="FFmpeg"
              value={
                info.ffmpeg.available
                  ? `${info.ffmpeg.version ?? "available"}`
                  : "UNAVAILABLE"
              }
            />
            <Row
              label="ASR workers"
              value={
                info.asr_worker_envs.fw_env && info.asr_worker_envs.wx_env
                  ? "cached locally"
                  : "will bootstrap on first use"
              }
            />
            <Row
              label="OCR (Tesseract)"
              value={
                info.ocr.tesseract_on_path || info.ocr.tesseract_dir_env
                  ? "available"
                  : "OCR unavailable — text review remains manual"
              }
            />
          </dl>
        )}
      </section>

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>About &amp; privacy</h2>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>Processing is local. Task media is never uploaded by this app.</li>
          <li>
            Local ASR bootstrap may download packages/models from package hosts; task audio
            itself never leaves this machine.
          </li>
          <li>No telemetry. No account. No platform credentials are stored or read.</li>
          <li>
            This app never claims tasks, never submits, and never automates the live tool.
            The final human remains the submission authority.
          </li>
        </ul>
      </section>

      <button onClick={() => onNavigate("home")} style={{ alignSelf: "flex-start" }}>
        Back
      </button>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="row">
      <dt className="muted" style={{ width: 140 }}>
        {label}
      </dt>
      <dd className="mono" style={{ margin: 0 }}>
        {value}
      </dd>
    </div>
  );
}
