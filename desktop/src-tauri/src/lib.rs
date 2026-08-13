//! Reviewer Cockpit backend: transport + process control only. The Python
//! engine remains the factual authority; Rust never re-implements caption or
//! evidence logic (spec §13).

mod commands;
mod engine;
mod errors;
mod files;
mod jobs;
mod state;

use engine::EngineLaunch;
use state::BackendState;
use tauri::Manager;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let resource_dir = app.path().resource_dir().ok();
            let app_data_dir = app.path().app_data_dir().ok();
            let launch = EngineLaunch::resolve(resource_dir.as_deref(), app_data_dir.as_deref())
                .unwrap_or(EngineLaunch::DevUv {
                    repo_root: std::env::current_dir().unwrap_or_default(),
                });
            let state = BackendState::new(launch);
            if let Some(data_dir) = app_data_dir {
                if let Ok(mut guard) = state.diagnostics_log.lock() {
                    *guard = Some(data_dir.join("logs").join("engine-worker.log"));
                }
            }
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::engine_request,
            commands::start_analysis,
            commands::start_rerun_with_anchors,
            commands::cancel_analysis,
            commands::analysis_running,
            commands::read_run_file,
            commands::list_recent_runs,
            commands::remember_recent_run,
            commands::forget_recent_run,
            commands::open_artifact_folder,
            commands::register_intake_video,
            commands::allow_video_playback,
            commands::save_ready_caption,
            commands::save_review_draft,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
