//! Engine worker process management: spawn, handshake, request/response.
//!
//! Rust owns process control (spec §11). Only two fixed invocations exist:
//! the packaged engine sidecar executable, or — in development — the repo's
//! uv-managed worker module. No user-supplied executable path is ever run.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use crate::errors::{BridgeError, BridgeResult};

pub const UI_BRIDGE_PROTOCOL_VERSION: u64 = 1;
pub const SIDECAR_BASENAME: &str = "manuscript-engine-worker";

/// How to launch the engine worker. Resolved once at startup.
#[derive(Debug, Clone)]
pub enum EngineLaunch {
    /// Packaged sidecar executable bundled with the app. Carries the
    /// packaged-runtime environment: bundled FFmpeg dir, bundled uv dir,
    /// and the writable app-local ASR worker root.
    Sidecar {
        exe: PathBuf,
        ffmpeg_dir: Option<PathBuf>,
        uv_dir: Option<PathBuf>,
        asr_workers_dir: Option<PathBuf>,
    },
    /// Development mode: `uv run --project <repo> python -m manuscript_reviewer.ui_bridge.worker`.
    DevUv { repo_root: PathBuf },
}

impl EngineLaunch {
    /// Packaged sidecar wins when present; otherwise development mode
    /// requires the repository checkout (never an arbitrary path).
    pub fn resolve(resource_dir: Option<&Path>, app_data_dir: Option<&Path>) -> BridgeResult<Self> {
        if let Some(dir) = resource_dir {
            let candidates = [
                dir.join("binaries").join(format!("{SIDECAR_BASENAME}.exe")),
                dir.join("binaries")
                    .join(SIDECAR_BASENAME)
                    .join(format!("{SIDECAR_BASENAME}.exe")),
                dir.join("binaries").join(SIDECAR_BASENAME),
            ];
            if let Some(exe) = candidates.iter().find(|p| p.is_file()) {
                let ffmpeg_dir = dir.join("ffmpeg").join("bin");
                let uv_dir = dir.join("uv");
                return Ok(Self::Sidecar {
                    exe: exe.clone(),
                    ffmpeg_dir: ffmpeg_dir.is_dir().then_some(ffmpeg_dir),
                    uv_dir: uv_dir.is_dir().then_some(uv_dir),
                    asr_workers_dir: app_data_dir.map(|d| d.join("asr_workers")),
                });
            }
        }
        if let Some(repo) = find_repo_root() {
            return Ok(Self::DevUv { repo_root: repo });
        }
        Err(BridgeError::engine_not_found(
            "No packaged engine sidecar and no development repository found",
        ))
    }

    pub fn command(&self) -> Command {
        match self {
            Self::Sidecar {
                exe,
                ffmpeg_dir,
                uv_dir,
                asr_workers_dir,
            } => {
                let mut cmd = Command::new(exe);
                if let Some(parent) = exe.parent() {
                    cmd.current_dir(parent);
                }
                // The packaged FFmpeg always wins over an arbitrary system
                // one (spec §14); the env var is the engine's documented hook.
                if let Some(dir) = ffmpeg_dir {
                    cmd.env("MANUSCRIPT_FFMPEG_DIR", dir);
                }
                // ASR worker envs live in writable app data (spec §15); the
                // bundled uv is found via a PATH prepend, not a shell.
                if let Some(dir) = asr_workers_dir {
                    let _ = std::fs::create_dir_all(dir);
                    cmd.env("MANUSCRIPT_ASR_WORKERS_DIR", dir);
                }
                if let Some(dir) = uv_dir {
                    let path_var = std::env::var_os("PATH").unwrap_or_default();
                    let mut paths: Vec<PathBuf> = vec![dir.clone()];
                    paths.extend(std::env::split_paths(&path_var));
                    if let Ok(joined) = std::env::join_paths(paths) {
                        cmd.env("PATH", joined);
                    }
                }
                cmd
            }
            Self::DevUv { repo_root } => {
                let mut cmd = Command::new("uv");
                cmd.args([
                    "run",
                    "--project",
                    &repo_root.to_string_lossy(),
                    "python",
                    "-m",
                    "manuscript_reviewer.ui_bridge.worker",
                ]);
                cmd.current_dir(repo_root);
                cmd
            }
        }
    }
}

/// Locate the engine repository in development: walk up from the current
/// directory looking for `engine/manuscript_reviewer/pyproject-marked` root.
fn find_repo_root() -> Option<PathBuf> {
    let start = std::env::current_dir().ok()?;
    for dir in start.ancestors() {
        if dir.join("pyproject.toml").exists()
            && dir.join("engine").join("manuscript_reviewer").is_dir()
        {
            return Some(dir.to_path_buf());
        }
    }
    None
}

/// A running worker process speaking the JSONL protocol.
pub struct EngineWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_request: AtomicU64,
}

impl EngineWorker {
    pub fn spawn(launch: &EngineLaunch) -> BridgeResult<Self> {
        let mut cmd = launch.command();
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| BridgeError::engine_not_found(e.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BridgeError::engine_not_found("worker stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BridgeError::engine_not_found("worker stdout unavailable"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request: AtomicU64::new(1),
        })
    }

    /// Verify the worker speaks our protocol version before any other use.
    pub fn handshake(&mut self) -> BridgeResult<Value> {
        let health = self.request("health", json!({}))?;
        let version = health
            .get("protocol_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if version != UI_BRIDGE_PROTOCOL_VERSION {
            return Err(BridgeError::with_detail(
                "PROTOCOL_VERSION_MISMATCH",
                "The engine bridge protocol is incompatible with this app version.",
                format!("app speaks {UI_BRIDGE_PROTOCOL_VERSION}, engine speaks {version}"),
            ));
        }
        Ok(health)
    }

    /// One synchronous request/response. Progress events (which only long
    /// jobs emit) are surfaced to the caller via `on_event`.
    pub fn request_with_events(
        &mut self,
        command: &str,
        payload: Value,
        mut on_event: impl FnMut(&Value),
    ) -> BridgeResult<Value> {
        let request_id = format!("rust-{}", self.next_request.fetch_add(1, Ordering::Relaxed));
        let request = json!({
            "request_id": request_id,
            "command": command,
            "payload": payload,
            "protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
        });
        let line = serde_json::to_string(&request)
            .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| BridgeError::engine_crash(format!("worker stdin closed: {e}")))?;

        loop {
            let mut buf = String::new();
            let read = self
                .stdout
                .read_line(&mut buf)
                .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
            if read == 0 {
                return Err(BridgeError::engine_crash("worker exited mid-request"));
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue, // stray non-protocol output is ignored
            };
            if value.get("event").is_some() {
                on_event(&value);
                continue;
            }
            if value.get("request_id").and_then(Value::as_str) != Some(request_id.as_str()) {
                continue; // stale response from a previous (failed) exchange
            }
            let status = value.get("status").and_then(Value::as_str).unwrap_or("");
            if status == "ok" {
                return Ok(value.get("payload").cloned().unwrap_or(Value::Null));
            }
            let error = value.get("error").cloned().unwrap_or(Value::Null);
            return Err(
                serde_json::from_value::<BridgeError>(error).unwrap_or_else(|_| {
                    BridgeError::engine_crash("worker returned an unreadable error")
                }),
            );
        }
    }

    pub fn request(&mut self, command: &str, payload: Value) -> BridgeResult<Value> {
        self.request_with_events(command, payload, |_| {})
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for EngineWorker {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn sidecar_wins_when_present() {
        let dir = std::env::temp_dir().join(format!("mr-engine-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let bin = dir.join("binaries");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join(format!("{SIDECAR_BASENAME}.exe")), b"stub").unwrap();
        let launch = EngineLaunch::resolve(Some(&dir), Some(&dir)).unwrap();
        match launch {
            EngineLaunch::Sidecar { exe, .. } => {
                assert!(exe.ends_with(format!("{SIDECAR_BASENAME}.exe")))
            }
            EngineLaunch::DevUv { .. } => panic!("expected sidecar launch"),
        }
    }

    #[test]
    fn sidecar_env_carries_packaged_runtime_dirs() {
        let dir = std::env::temp_dir().join(format!("mr-engine-env-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("binaries")).unwrap();
        fs::write(
            dir.join("binaries").join(format!("{SIDECAR_BASENAME}.exe")),
            b"stub",
        )
        .unwrap();
        fs::create_dir_all(dir.join("ffmpeg").join("bin")).unwrap();
        fs::create_dir_all(dir.join("uv")).unwrap();
        let launch = EngineLaunch::resolve(Some(&dir), Some(&dir.join("appdata"))).unwrap();
        match &launch {
            EngineLaunch::Sidecar {
                ffmpeg_dir,
                uv_dir,
                asr_workers_dir,
                ..
            } => {
                assert!(ffmpeg_dir.is_some(), "bundled ffmpeg dir must be detected");
                assert!(uv_dir.is_some(), "bundled uv dir must be detected");
                assert!(
                    asr_workers_dir.is_some(),
                    "asr workers dir must be app-local"
                );
            }
            EngineLaunch::DevUv { .. } => panic!("expected sidecar launch"),
        }
        // Building the command must not panic and must target the sidecar exe.
        let cmd = launch.command();
        assert!(cmd
            .get_program()
            .to_string_lossy()
            .contains(SIDECAR_BASENAME));
    }
}
