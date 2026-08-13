//! Analysis job lifecycle: one cancellable audit job at a time, run in a
//! dedicated worker process so cancellation is honest (kill the process
//! tree; already-written artifacts remain auditable).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Stdio};
use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

use crate::engine::{EngineLaunch, UI_BRIDGE_PROTOCOL_VERSION};
use crate::errors::{BridgeError, BridgeResult};

pub const PROGRESS_EVENT: &str = "analysis://progress";
pub const DONE_EVENT: &str = "analysis://done";
pub const ERROR_EVENT: &str = "analysis://error";

#[derive(Default)]
pub struct JobManager {
    current: Mutex<Option<AnalysisJob>>,
}

pub struct AnalysisJob {
    child: Child,
}

impl JobManager {
    /// Spawn the audit in its own worker process; stream progress as Tauri
    /// events; refuse a second concurrent analysis (spec §80).
    pub fn start_analysis(
        &self,
        app: AppHandle,
        launch: &EngineLaunch,
        request_payload: Value,
    ) -> BridgeResult<String> {
        let mut guard = self
            .current
            .lock()
            .map_err(|_| BridgeError::engine_crash("job state poisoned"))?;
        if let Some(job) = guard.as_mut() {
            match job.child.try_wait() {
                Ok(Some(_)) => *guard = None,
                Ok(None) => {
                    return Err(BridgeError::new(
                        "RUN_LOCKED",
                        "An analysis is already running. Cancel it before starting another.",
                    ))
                }
                Err(e) => return Err(BridgeError::engine_crash(e.to_string())),
            }
        }

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

        let job_id = format!("job-{}", std::process::id());
        let request = json!({
            "request_id": job_id,
            "command": "start_audit",
            "payload": request_payload,
            "protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
        });
        {
            // Write the single job request, then CLOSE stdin: when the worker
            // finishes it reads EOF and exits on its own — no idle job worker
            // lingers after completion.
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| BridgeError::engine_not_found("job stdin unavailable"))?;
            let line = serde_json::to_string(&request)
                .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
            stdin
                .write_all(line.as_bytes())
                .and_then(|_| stdin.write_all(b"\n"))
                .and_then(|_| stdin.flush())
                .map_err(|e| BridgeError::engine_crash(e.to_string()))?;
            drop(stdin);
        }
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BridgeError::engine_not_found("job stdout unavailable"))?;

        let reader_app = app;
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut finished = false;
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                    continue;
                };
                if value.get("event").and_then(Value::as_str) == Some("progress") {
                    let payload = value.get("payload").cloned().unwrap_or(Value::Null);
                    let _ = reader_app.emit(PROGRESS_EVENT, payload);
                    continue;
                }
                let status = value.get("status").and_then(Value::as_str).unwrap_or("");
                if status == "ok" {
                    let payload = value.get("payload").cloned().unwrap_or(Value::Null);
                    let _ = reader_app.emit(DONE_EVENT, payload);
                } else {
                    let error = value.get("error").cloned().unwrap_or_else(
                        || json!({"code": "ENGINE_CRASH", "message": "Analysis failed"}),
                    );
                    let _ = reader_app.emit(ERROR_EVENT, error);
                }
                finished = true;
                break;
            }
            if !finished {
                let _ = reader_app.emit(
                    ERROR_EVENT,
                    json!({
                        "code": "ENGINE_CRASH",
                        "message": "The analysis worker exited without a result.",
                    }),
                );
            }
        });

        *guard = Some(AnalysisJob { child });
        Ok(job_id)
    }

    /// Honest cancellation: kill the worker process tree (ffmpeg / uv / ASR
    /// children included on Windows via taskkill /T).
    pub fn cancel(&self) -> BridgeResult<()> {
        let mut guard = self
            .current
            .lock()
            .map_err(|_| BridgeError::engine_crash("job state poisoned"))?;
        if let Some(mut job) = guard.take() {
            kill_process_tree(&mut job.child);
        }
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        let mut guard = match self.current.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        match guard.as_mut() {
            Some(job) => match job.child.try_wait() {
                Ok(Some(_)) => {
                    *guard = None;
                    false
                }
                Ok(None) => true,
                Err(_) => false,
            },
            None => false,
        }
    }
}

fn kill_process_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

impl Drop for JobManager {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}
