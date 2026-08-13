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
}

impl BackendState {
    pub fn new(launch: EngineLaunch) -> Self {
        Self {
            launch,
            worker: Mutex::new(None),
            jobs: JobManager::default(),
            allowed_run_dirs: Mutex::new(HashSet::new()),
        }
    }

    pub fn allow_run_dir(&self, run_dir: &Path) {
        if let Ok(canonical) = run_dir.canonicalize() {
            if let Ok(mut guard) = self.allowed_run_dirs.lock() {
                guard.insert(canonical);
            }
        }
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
