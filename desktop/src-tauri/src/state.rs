//! Shared backend state: the query worker, the job manager, and the set of
//! run directories the engine has validated this session (path allow-list).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::engine::{EngineLaunch, EngineWorker};
use crate::errors::{BridgeError, BridgeResult};
use crate::jobs::JobManager;

pub struct BackendState {
    pub launch: EngineLaunch,
    pub worker: Mutex<Option<EngineWorker>>,
    pub jobs: JobManager,
    /// Run directories validated by the engine this session; the only paths
    /// `read_run_file` will serve from (spec §76–77).
    pub allowed_run_dirs: Mutex<HashSet<PathBuf>>,
    /// The single video selected for the CURRENT new-task intake (native
    /// dialog choice registered through Rust). Playback trust is granted only
    /// to this slot or to an allowed run's recorded source video.
    pub intake_video: Mutex<Option<PathBuf>>,
    /// Source videos of engine-validated runs (from their manifests).
    pub allowed_videos: Mutex<HashSet<PathBuf>>,
    /// App-local worker stderr log (rolling); None disables capture.
    pub diagnostics_log: Mutex<Option<PathBuf>>,
}

impl BackendState {
    pub fn new(launch: EngineLaunch) -> Self {
        Self {
            launch,
            worker: Mutex::new(None),
            jobs: JobManager::default(),
            allowed_run_dirs: Mutex::new(HashSet::new()),
            intake_video: Mutex::new(None),
            allowed_videos: Mutex::new(HashSet::new()),
            diagnostics_log: Mutex::new(None),
        }
    }

    pub fn diagnostics_log_path(&self) -> Option<PathBuf> {
        self.diagnostics_log.lock().ok().and_then(|g| g.clone())
    }

    pub fn allow_run_dir(&self, run_dir: &Path) {
        if let Ok(canonical) = run_dir.canonicalize() {
            if let Ok(mut guard) = self.allowed_run_dirs.lock() {
                guard.insert(canonical);
            }
        }
    }

    pub fn is_run_dir_allowed(&self, run_dir: &Path) -> bool {
        let Ok(canonical) = run_dir.canonicalize() else {
            return false;
        };
        self.allowed_run_dirs
            .lock()
            .map(|guard| guard.contains(&canonical))
            .unwrap_or(false)
    }

    pub fn allow_video(&self, video: &Path) {
        if let Ok(canonical) = video.canonicalize() {
            if let Ok(mut guard) = self.allowed_videos.lock() {
                guard.insert(canonical);
            }
        }
    }

    pub fn is_video_allowed(&self, video: &Path) -> bool {
        let Ok(canonical) = video.canonicalize() else {
            return false;
        };
        if let Ok(guard) = self.intake_video.lock() {
            if guard.as_deref() == Some(canonical.as_path()) {
                return true;
            }
        }
        self.allowed_videos
            .lock()
            .map(|guard| guard.contains(&canonical))
            .unwrap_or(false)
    }

    pub fn check_path_allowed(&self, run_dir: &Path, file: &Path) -> BridgeResult<PathBuf> {
        let run_canonical = run_dir
            .canonicalize()
            .map_err(|_| BridgeError::new("RUN_NOT_FOUND", "Run directory not found"))?;
        let allowed = self
            .allowed_run_dirs
            .lock()
            .map_err(|_| BridgeError::engine_crash("state poisoned"))?
            .contains(&run_canonical);
        if !allowed {
            return Err(BridgeError::new(
                "INVALID_INPUT",
                "Run directory has not been opened in this session",
            ));
        }
        let file_canonical = file
            .canonicalize()
            .map_err(|_| BridgeError::new("ARTIFACT_NOT_FOUND", "File not found"))?;
        if !file_canonical.starts_with(&run_canonical) {
            return Err(BridgeError::new(
                "INVALID_INPUT",
                "Path escapes the run directory",
            ));
        }
        Ok(file_canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn state_with_temp() -> (BackendState, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("mr-state-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("run").join("caption")).unwrap();
        fs::write(dir.join("run").join("manifest.json"), "{}").unwrap();
        fs::write(dir.join("run").join("caption").join("x.md"), "hello").unwrap();
        fs::write(dir.join("secret.txt"), "secret").unwrap();
        let state = BackendState::new(EngineLaunch::DevUv {
            repo_root: dir.clone(),
        });
        (state, dir)
    }

    #[test]
    fn unregistered_run_dir_is_rejected() {
        let (state, dir) = state_with_temp();
        let run = dir.join("run");
        let err = state
            .check_path_allowed(&run, &run.join("caption").join("x.md"))
            .unwrap_err();
        assert_eq!(err.code, "INVALID_INPUT");
    }

    #[test]
    fn registered_run_dir_serves_inside_paths_only() {
        let (state, dir) = state_with_temp();
        let run = dir.join("run");
        state.allow_run_dir(&run);
        assert!(state
            .check_path_allowed(&run, &run.join("caption").join("x.md"))
            .is_ok());
        // Traversal out of the run dir is refused even when the run is allowed.
        let escape = run.join("..").join("secret.txt");
        let err = state.check_path_allowed(&run, &escape).unwrap_err();
        assert_eq!(err.code, "INVALID_INPUT");
    }

    #[test]
    fn recent_run_paths_grant_no_trust_until_engine_validated() {
        // §Phase 6.1-6: an arbitrary local folder (e.g. Documents) must never
        // become readable merely by existing in the recent-runs index.
        let (state, dir) = state_with_temp();
        let documents = dir.join("Documents");
        fs::create_dir_all(&documents).unwrap();
        fs::write(documents.join("private.png"), b"png").unwrap();
        assert!(!state.is_run_dir_allowed(&documents));
        let err = state
            .check_path_allowed(&documents, &documents.join("private.png"))
            .unwrap_err();
        assert_eq!(err.code, "INVALID_INPUT");
        // Only explicit engine validation (allow_run_dir) grants access.
        state.allow_run_dir(&documents);
        assert!(state.is_run_dir_allowed(&documents));
    }

    #[test]
    fn video_playback_trust_is_slot_or_run_scoped() {
        let (state, dir) = state_with_temp();
        let stray = dir.join("stray.mp4");
        fs::write(&stray, b"vid").unwrap();
        assert!(!state.is_video_allowed(&stray));
        // Intake slot grants it…
        *state.intake_video.lock().unwrap() = Some(stray.canonicalize().unwrap());
        assert!(state.is_video_allowed(&stray));
        // …and a different video still isn't allowed.
        let other = dir.join("other.mp4");
        fs::write(&other, b"vid").unwrap();
        assert!(!state.is_video_allowed(&other));
        // Run-source registration allows it explicitly.
        state.allow_video(&other);
        assert!(state.is_video_allowed(&other));
    }

    #[test]
    fn missing_paths_are_typed_errors() {
        let (state, dir) = state_with_temp();
        let run = dir.join("run");
        state.allow_run_dir(&run);
        let err = state
            .check_path_allowed(&run, &run.join("nope.png"))
            .unwrap_err();
        assert_eq!(err.code, "ARTIFACT_NOT_FOUND");
    }
}
