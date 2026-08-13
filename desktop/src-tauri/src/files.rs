//! Validated file services: run-scoped file reads (served as data URLs) and
//! the app-local recent-runs index. The run directory stays the source of
//! truth — the index is display cache only (spec §28).

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::errors::{BridgeError, BridgeResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentRun {
    pub run_dir: String,
    pub video_name: String,
    pub last_opened_utc: String,
    pub display_status: String,
    pub readiness: String,
}

const MAX_RECENT: usize = 20;

pub fn recent_runs_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("recent_runs.json")
}

pub fn load_recent_runs(app_data_dir: &Path) -> Vec<RecentRun> {
    let path = recent_runs_path(app_data_dir);
    let Ok(raw) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut runs: Vec<RecentRun> = serde_json::from_str(&raw).unwrap_or_default();
    // A run whose directory disappeared is still listed but surfaced as
    // incomplete/unknown by the frontend; a run with no manifest is INCOMPLETE.
    for run in &mut runs {
        let dir = Path::new(&run.run_dir);
        if !dir.is_dir() {
            run.readiness = "UNKNOWN".into();
            run.display_status = "MISSING".into();
        } else if !dir.join("manifest.json").exists() {
            run.readiness = "INCOMPLETE".into();
            run.display_status = "INCOMPLETE".into();
        }
    }
    runs
}

pub fn remember_recent_run(app_data_dir: &Path, entry: RecentRun) -> BridgeResult<()> {
    fs::create_dir_all(app_data_dir).map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    let mut runs = load_recent_runs(app_data_dir);
    runs.retain(|r| r.run_dir != entry.run_dir);
    runs.insert(0, entry);
    runs.truncate(MAX_RECENT);
    let raw = serde_json::to_string_pretty(&runs)
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    let tmp = recent_runs_path(app_data_dir).with_extension("json.tmp");
    fs::write(&tmp, raw).map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    fs::rename(&tmp, recent_runs_path(app_data_dir))
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    Ok(())
}

pub fn forget_recent_run(app_data_dir: &Path, run_dir: &str) -> BridgeResult<()> {
    let mut runs = load_recent_runs(app_data_dir);
    runs.retain(|r| r.run_dir != run_dir);
    let raw = serde_json::to_string_pretty(&runs)
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    fs::write(recent_runs_path(app_data_dir), raw)
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    Ok(())
}

/// Read a validated run-scoped file as a data URL for direct display.
pub fn read_as_data_url(path: &Path) -> BridgeResult<String> {
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "wav" => "audio/wav",
        "json" | "jsonl" | "csv" | "md" | "txt" | "log" => "text/plain",
        other => {
            return Err(BridgeError::invalid_input(format!(
                "File type .{other} is not served to the UI"
            )))
        }
    };
    let bytes = fs::read(path).map_err(|e| {
        BridgeError::with_detail(
            "ARTIFACT_NOT_FOUND",
            "File could not be read",
            e.to_string(),
        )
    })?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:{mime};base64,{encoded}"))
}
