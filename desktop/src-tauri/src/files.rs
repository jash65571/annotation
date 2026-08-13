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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mr-files-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(run_dir: &str) -> RecentRun {
        RecentRun {
            run_dir: run_dir.to_string(),
            video_name: "clip.mp4".into(),
            last_opened_utc: "2026-08-12T00:00:00Z".into(),
            display_status: "REVIEW_REQUIRED".into(),
            readiness: "REVIEW_REQUIRED".into(),
        }
    }

    #[test]
    fn remember_dedupes_and_truncates() {
        let dir = temp_dir("recent");
        for i in 0..25 {
            remember_recent_run(&dir, entry(&format!("C:/runs/{i}"))).unwrap();
        }
        remember_recent_run(&dir, entry("C:/runs/24")).unwrap();
        let runs = load_recent_runs(&dir);
        assert_eq!(runs.len(), MAX_RECENT);
        assert_eq!(runs[0].run_dir, "C:/runs/24");
        assert_eq!(runs.iter().filter(|r| r.run_dir == "C:/runs/24").count(), 1);
    }

    #[test]
    fn forget_removes_entry() {
        let dir = temp_dir("forget");
        remember_recent_run(&dir, entry("C:/runs/a")).unwrap();
        remember_recent_run(&dir, entry("C:/runs/b")).unwrap();
        forget_recent_run(&dir, "C:/runs/a").unwrap();
        let runs = load_recent_runs(&dir);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_dir, "C:/runs/b");
    }

    #[test]
    fn missing_run_dirs_are_marked_not_verified() {
        let dir = temp_dir("marks");
        let real_run = dir.join("incomplete-run");
        fs::create_dir_all(&real_run).unwrap();
        remember_recent_run(&dir, entry(&real_run.to_string_lossy())).unwrap();
        let runs = load_recent_runs(&dir);
        // A run directory without a manifest is INCOMPLETE, never verified.
        assert_eq!(runs[0].readiness, "INCOMPLETE");
    }

    #[test]
    fn data_url_serves_known_types_only() {
        let dir = temp_dir("dataurl");
        let png = dir.join("f.png");
        fs::write(&png, [137u8, 80, 78, 71]).unwrap();
        let url = read_as_data_url(&png).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        let exe = dir.join("evil.exe");
        fs::write(&exe, b"MZ").unwrap();
        assert!(read_as_data_url(&exe).is_err());
    }
}
