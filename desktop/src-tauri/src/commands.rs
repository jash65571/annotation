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
    "get_review_inputs",
    "save_visual_anchors",
    "get_review_resolution",
    "get_media_dimensions",
    "save_review_inputs",
    "append_audit_history",
    "get_audit_history",
    "save_ui_state",
    "get_ui_state",
    "export_draft",
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
    let log_path = state.diagnostics_log_path();
    if guard.is_none() {
        let mut worker = EngineWorker::spawn_with_diagnostics(&state.launch, log_path.as_deref())?;
        worker.handshake()?;
        *guard = Some(worker);
    }
    let worker = guard.as_mut().expect("worker present");
    let mut result = f(worker);
    if let Err(error) = &mut result {
        // A crashed worker is discarded so the next request restarts cleanly;
        // the last diagnostic lines ride along behind "Details".
        if error.code == "ENGINE_CRASH" {
            if let Some(mut dead) = guard.take() {
                dead.kill();
            }
            if error.detail.is_none() {
                error.detail = crate::engine::diagnostics_tail(log_path.as_deref());
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
    // run-scoped file reads; its recorded source video becomes playable.
    if let Some(run_dir) = payload.get("run_dir").and_then(Value::as_str) {
        state.allow_run_dir(Path::new(run_dir));
    }
    if let Some(run_dir) = result.get("run_dir").and_then(Value::as_str) {
        state.allow_run_dir(Path::new(run_dir));
    }
    if let Some(source) = result
        .get("manifest")
        .and_then(|m| m.get("source_video_path"))
        .and_then(Value::as_str)
    {
        state.allow_video(Path::new(source));
    }
    Ok(result)
}

fn is_supported_video(path: &Path) -> bool {
    path.is_file()
        && matches!(
            path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str(),
            "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v"
        )
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
        .map(str::to_owned)
        .ok_or_else(|| BridgeError::invalid_input("video_path is required"))?;
    let video_path = PathBuf::from(&video);
    if !is_supported_video(&video_path) {
        return Err(BridgeError::invalid_input(
            "The selected video path is not a supported video file",
        ));
    }
    // The renderer never chooses where artifacts go or which anchors file to
    // read (spec §Phase 6.1-9/10): the artifacts root is app-managed and
    // anchor reruns go through start_rerun_with_anchors.
    let mut payload = request;
    if payload.contains_key("visual_anchors_path") {
        return Err(BridgeError::invalid_input(
            "Anchor re-runs must use the dedicated re-run command",
        ));
    }
    let artifacts = default_artifacts_root(&app)?;
    payload.insert(
        "artifacts_root".into(),
        Value::String(artifacts.to_string_lossy().to_string()),
    );
    // The analyzed video becomes the current playable intake video.
    if let Ok(canonical) = video_path.canonicalize() {
        if let Ok(mut guard) = state.intake_video.lock() {
            *guard = Some(canonical);
        }
    }
    state.jobs.start_analysis_with_diagnostics(
        app,
        &state.launch,
        Value::Object(payload),
        state.diagnostics_log_path().as_deref(),
    )
}

/// RE-RUN VISUAL ANALYSIS WITH ANCHORS: every input path is resolved by the
/// ENGINE from the verified run's provenance — the renderer supplies only the
/// run directory (spec §Phase 6.1-2).
#[tauri::command]
pub fn start_rerun_with_anchors(
    app: AppHandle,
    state: State<'_, BackendState>,
    run_dir: String,
) -> BridgeResult<String> {
    if !state.is_run_dir_allowed(Path::new(&run_dir)) {
        return Err(BridgeError::invalid_input(
            "Run directory has not been opened in this session",
        ));
    }
    let resolved = with_worker(&state, |worker| {
        worker.request(
            "resolve_rerun_request",
            serde_json::json!({ "run_dir": run_dir }),
        )
    })?;
    let request = resolved
        .get("request")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| BridgeError::engine_crash("rerun resolution returned no request"))?;
    state.jobs.start_analysis_with_diagnostics(
        app,
        &state.launch,
        Value::Object(request),
        state.diagnostics_log_path().as_deref(),
    )
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

/// The recent-runs index is DISPLAY DATA only: listing it never grants
/// filesystem trust. A run becomes readable only after the engine validates
/// it again (get_run_summary) in this session (spec §Phase 6.1-6).
#[tauri::command]
pub fn list_recent_runs(app: AppHandle) -> Vec<RecentRun> {
    let Ok(dir) = app_data_dir(&app) else {
        return Vec::new();
    };
    files::load_recent_runs(&dir)
}

#[tauri::command]
pub fn remember_recent_run(
    app: AppHandle,
    state: State<'_, BackendState>,
    entry: RecentRun,
) -> BridgeResult<()> {
    // Only engine-validated (session-allowed) run directories may enter the
    // index — an arbitrary renderer-supplied path is refused.
    if !state.is_run_dir_allowed(Path::new(&entry.run_dir)) {
        return Err(BridgeError::invalid_input(
            "Only runs validated by the engine this session can be remembered",
        ));
    }
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

/// Register the video the reviewer just picked in the native dialog as the
/// current intake video (the ONLY new-task video eligible for playback).
#[tauri::command]
pub fn register_intake_video(state: State<'_, BackendState>, path: String) -> BridgeResult<String> {
    let video = Path::new(&path);
    if !is_supported_video(video) {
        return Err(BridgeError::invalid_input("Not a supported video file"));
    }
    let canonical = video
        .canonicalize()
        .map_err(|_| BridgeError::invalid_input("Video not found"))?;
    let mut guard = state
        .intake_video
        .lock()
        .map_err(|_| BridgeError::engine_crash("state poisoned"))?;
    *guard = Some(canonical.clone());
    Ok(canonical.to_string_lossy().to_string())
}

/// Allow the webview to stream ONE trusted video via the asset protocol:
/// either the current intake video or the recorded source video of an
/// engine-validated run. The webview gets no general local-video oracle.
#[tauri::command]
pub fn allow_video_playback(
    app: AppHandle,
    state: State<'_, BackendState>,
    path: String,
) -> BridgeResult<String> {
    let video = Path::new(&path);
    if !is_supported_video(video) {
        return Err(BridgeError::invalid_input("Not a supported video file"));
    }
    if !state.is_video_allowed(video) {
        return Err(BridgeError::invalid_input(
            "This video is not the current task video or a validated run's source",
        ));
    }
    let canonical = video
        .canonicalize()
        .map_err(|_| BridgeError::invalid_input("Video not found"))?;
    app.asset_protocol_scope()
        .allow_file(&canonical)
        .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
    Ok(canonical.to_string_lossy().to_string())
}

fn checked_caption_destination(path: &str) -> BridgeResult<PathBuf> {
    let target = PathBuf::from(path);
    match target
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "md" | "txt" => Ok(target),
        other => Err(BridgeError::invalid_input(format!(
            "Refusing to write .{other} — captions save as .md or .txt"
        ))),
    }
}

/// Save the READY caption: Rust obtains the engine's export (which itself is
/// gated on READY_TO_ENTER) and writes those exact bytes. The renderer never
/// supplies caption content to a privileged save (spec §Phase 6.1-8).
#[tauri::command]
pub fn save_ready_caption(
    state: State<'_, BackendState>,
    run_dir: String,
    destination: String,
) -> BridgeResult<()> {
    let target = checked_caption_destination(&destination)?;
    let exported = with_worker(&state, |worker| {
        worker.request("export_caption", serde_json::json!({ "run_dir": run_dir }))
    })?;
    let markdown = exported
        .get("markdown")
        .and_then(Value::as_str)
        .ok_or_else(|| BridgeError::engine_crash("export returned no caption text"))?;
    std::fs::write(&target, markdown.as_bytes()).map_err(|e| {
        BridgeError::with_detail("INVALID_INPUT", "File could not be saved", e.to_string())
    })
}

/// Save the current DRAFT (clearly a draft) from draft_review_only.md.
#[tauri::command]
pub fn save_review_draft(
    state: State<'_, BackendState>,
    run_dir: String,
    destination: String,
) -> BridgeResult<()> {
    let target = checked_caption_destination(&destination)?;
    let exported = with_worker(&state, |worker| {
        worker.request("export_draft", serde_json::json!({ "run_dir": run_dir }))
    })?;
    let markdown = exported
        .get("markdown")
        .and_then(Value::as_str)
        .ok_or_else(|| BridgeError::engine_crash("draft export returned no text"))?;
    std::fs::write(&target, markdown.as_bytes()).map_err(|e| {
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
