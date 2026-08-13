//! The complete typed command surface exposed to the frontend. There is no
//! generic run_command / shell escape — every command validates its inputs.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use tauri::{AppHandle, Manager, State};

use crate::engine::EngineWorker;
use crate::errors::{BridgeError, BridgeResult};
use crate::files::{self, RecentRun};
use crate::state::BackendState;

/// Engine commands the frontend may proxy through the query worker.
const ALLOWED_ENGINE_COMMANDS: &[&str] = &[
    "health",
    "engine_info",
    "get_rules",
    "load_run",
    "get_run_summary",
    "get_review_queue",
    "get_review_item",
    "get_shots",
    "get_frame_record",
    "get_exact_frame",
    "get_evidence_bundle",
    "get_audio_review_clip",
    "get_waveform_metadata",
    "save_review_decisions",
    "save_human_facts",
    "finalize",
    "get_caption_state",
    "create_final_signoff",
    "validate_final_signoff",
    "export_caption",
];

fn with_worker<T>(
    state: &BackendState,
    f: impl FnOnce(&mut EngineWorker) -> BridgeResult<T>,
) -> BridgeResult<T> {
    let mut guard = state
        .worker
        .lock()
        .map_err(|_| BridgeError::engine_crash("worker state poisoned"))?;
    if guard.is_none() {
        let mut worker = EngineWorker::spawn(&state.launch)?;
        worker.handshake()?;
        *guard = Some(worker);
    }
    let worker = guard.as_mut().expect("worker present");
    let result = f(worker);
    if let Err(error) = &result {
        // A crashed worker is discarded so the next request restarts cleanly.
        if error.code == "ENGINE_CRASH" {
            if let Some(mut dead) = guard.take() {
                dead.kill();
            }
        }
    }
    result
}

#[tauri::command]
pub fn engine_request(
    state: State<'_, BackendState>,
    command: String,
    payload: Map<String, Value>,
) -> BridgeResult<Value> {
    if !ALLOWED_ENGINE_COMMANDS.contains(&command.as_str()) {
        return Err(BridgeError::new(
            "INVALID_COMMAND",
            format!("Command {command} is not exposed to the UI"),
        ));
    }
    let result = with_worker(&state, |worker| {
        worker.request(&command, Value::Object(payload.clone()))
    })?;
    // Any run dir the engine successfully served becomes session-allowed for
    // run-scoped file reads.
    if let Some(run_dir) = payload.get("run_dir").and_then(Value::as_str) {
        state.allow_run_dir(Path::new(run_dir));
    }
    if let Some(run_dir) = result.get("run_dir").and_then(Value::as_str) {
        state.allow_run_dir(Path::new(run_dir));
    }
    Ok(result)
}

#[tauri::command]
pub fn start_analysis(
    app: AppHandle,
    state: State<'_, BackendState>,
    request: Map<String, Value>,
) -> BridgeResult<String> {
    let video = request
        .get("video_path")
        .and_then(Value::as_str)
        .ok_or_else(|| BridgeError::invalid_input("video_path is required"))?;
    let video_path = Path::new(video);
    let ext_ok = matches!(
        video_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v"
    );
    if !video_path.is_file() || !ext_ok {
        return Err(BridgeError::invalid_input(
            "The selected video path is not a supported video file",
        ));
    }
    let mut payload = request;
    if !payload.contains_key("artifacts_root") {
        let artifacts = default_artifacts_root(&app)?;
        payload.insert(
            "artifacts_root".into(),
            Value::String(artifacts.to_string_lossy().to_string()),
        );
    }
    state
        .jobs
        .start_analysis(app, &state.launch, Value::Object(payload))
}

#[tauri::command]
pub fn cancel_analysis(state: State<'_, BackendState>) -> BridgeResult<()> {
    state.jobs.cancel()
}

#[tauri::command]
pub fn analysis_running(state: State<'_, BackendState>) -> bool {
    state.jobs.is_running()
}

#[tauri::command]
pub fn read_run_file(
    state: State<'_, BackendState>,
    run_dir: String,
    path: String,
) -> BridgeResult<String> {
    let file = if Path::new(&path).is_absolute() {
        PathBuf::from(&path)
    } else {
        Path::new(&run_dir).join(&path)
    };
    let validated = state.check_path_allowed(Path::new(&run_dir), &file)?;
    files::read_as_data_url(&validated)
}

#[tauri::command]
pub fn list_recent_runs(app: AppHandle, state: State<'_, BackendState>) -> Vec<RecentRun> {
    let Ok(dir) = app_data_dir(&app) else {
        return Vec::new();
    };
    let runs = files::load_recent_runs(&dir);
    for run in &runs {
        // Recent runs the user recorded earlier are session-allowed again.
        state.allow_run_dir(Path::new(&run.run_dir));
    }
    runs
}

#[tauri::command]
pub fn remember_recent_run(app: AppHandle, entry: RecentRun) -> BridgeResult<()> {
    let dir = app_data_dir(&app)?;
    files::remember_recent_run(&dir, entry)
}

#[tauri::command]
pub fn forget_recent_run(app: AppHandle, run_dir: String) -> BridgeResult<()> {
    let dir = app_data_dir(&app)?;
    files::forget_recent_run(&dir, &run_dir)
}

#[tauri::command]
pub fn open_artifact_folder(state: State<'_, BackendState>, run_dir: String) -> BridgeResult<()> {
    let canonical = state.check_path_allowed(Path::new(&run_dir), Path::new(&run_dir))?;
    tauri_plugin_opener::open_path(canonical.to_string_lossy().to_string(), None::<&str>).map_err(
        |e| BridgeError::with_detail("INVALID_INPUT", "Folder could not be opened", e.to_string()),
    )
}

/// Allow the webview to stream the ONE selected source video via the asset
/// protocol (context playback only — never timing authority).
#[tauri::command]
pub fn allow_video_playback(app: AppHandle, path: String) -> BridgeResult<String> {
    let video = Path::new(&path);
    let ext_ok = matches!(
        video
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v"
    );
    if !video.is_file() || !ext_ok {
        return Err(BridgeError::invalid_input("Not a supported video file"));
    }
    let canonical = video
        .canonicalize()
        .map_err(|_| BridgeError::invalid_input("Video not found"))?;
    app.asset_protocol_scope()
        .allow_file(&canonical)
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    Ok(canonical.to_string_lossy().to_string())
}

/// Save exact text to a user-chosen destination (from the native save
/// dialog). Used only for "Save caption as…" — the contents are the exact
/// ready artifact bytes, never re-rendered.
#[tauri::command]
pub fn save_text_file(path: String, contents: String) -> BridgeResult<()> {
    let target = Path::new(&path);
    match target
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "md" | "txt" => {}
        other => {
            return Err(BridgeError::invalid_input(format!(
                "Refusing to write .{other} — captions save as .md or .txt"
            )))
        }
    }
    std::fs::write(target, contents).map_err(|e| {
        BridgeError::with_detail("INVALID_INPUT", "File could not be saved", e.to_string())
    })
}

fn app_data_dir(app: &AppHandle) -> BridgeResult<PathBuf> {
    app.path()
        .app_data_dir()
        .map_err(|e| BridgeError::engine_crash(e.to_string()))
}

fn default_artifacts_root(app: &AppHandle) -> BridgeResult<PathBuf> {
    let dir = app_data_dir(app)?.join("artifacts");
    std::fs::create_dir_all(&dir).map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    Ok(dir)
}
