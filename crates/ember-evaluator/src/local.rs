use crate::{
    Error, EvaluationRequest, EvaluationResult, Evaluator, Hello, Result, MAX_MESSAGE_BYTES,
};
use serde_json::{json, Value};
use std::{path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

#[derive(Clone)]
pub struct Launch {
    pub executable: PathBuf,
    pub args: Vec<String>,
    pub working_dir: PathBuf,
}
struct Session {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // The child owns a process group; user-code descendants never share the UI group.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        if let Some(pid) = self.child.id() {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.start_kill();
    }
}
impl Session {
    async fn rpc(&mut self, method: &str, id: &str, params: Value) -> Result<Value> {
        let mut message =
            serde_json::to_vec(&json!({"id": id, "method": method, "params": params}))?;
        if message.len() > MAX_MESSAGE_BYTES {
            return Err(Error::Invalid("Request is too large".into()));
        }
        message.push(b'\n');
        self.input.write_all(&message).await?;
        self.input.flush().await?;
        let mut line = Vec::new();
        loop {
            let buffer = self.output.fill_buf().await?;
            if buffer.is_empty() {
                return Err(Error::Invalid(
                    "Evaluator closed its protocol stream".into(),
                ));
            }
            let count = buffer
                .iter()
                .position(|&v| v == b'\n')
                .map_or(buffer.len(), |i| i + 1);
            if line.len() + count > MAX_MESSAGE_BYTES {
                return Err(Error::Invalid("Evaluator response is too large".into()));
            }
            let done = buffer[count - 1] == b'\n';
            line.extend_from_slice(&buffer[..count]);
            self.output.consume(count);
            if done {
                break;
            }
        }
        let response: Value = serde_json::from_slice(&line)?;
        if response["id"] != id {
            return Err(Error::Invalid("Evaluator request ID mismatch".into()));
        }
        if let Some(error) = response.get("error") {
            return Err(Error::Invalid(error.to_string()));
        }
        response
            .get("result")
            .cloned()
            .ok_or_else(|| Error::Invalid("Missing evaluator result".into()))
    }
}
pub struct LocalEvaluator {
    launch: Launch,
    catalog: String,
    session: Mutex<Option<Session>>,
}
impl LocalEvaluator {
    pub fn new(launch: Launch, catalog: String) -> Self {
        Self {
            launch,
            catalog,
            session: Mutex::new(None),
        }
    }
    pub async fn probe(&self) -> Result<()> {
        let mut guard = self.session.lock().await;
        if guard.is_none() {
            *guard = Some(
                tokio::time::timeout(Duration::from_secs(20), self.start())
                    .await
                    .map_err(|_| Error::Invalid("Evaluator handshake timed out".into()))??,
            );
        }
        Ok(())
    }
    pub async fn shutdown(&self) {
        self.session.lock().await.take();
    }
    async fn start(&self) -> Result<Session> {
        let mut command = Command::new(&self.launch.executable);
        command
            .args(&self.launch.args)
            .current_dir(&self.launch.working_dir)
            .env("PYTHONUNBUFFERED", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| Error::Invalid("Missing stdin".into()))?;
        let output = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| Error::Invalid("Missing stdout".into()))?,
        );
        let mut session = Session {
            child,
            input,
            output,
        };
        let hello: Hello =
            serde_json::from_value(session.rpc("hello", "hello", Value::Null).await?)?;
        hello.validate(&self.catalog)?;
        Ok(session)
    }
}
impl Evaluator for LocalEvaluator {
    fn evaluate<'a>(
        &'a self,
        request: EvaluationRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<EvaluationResult>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut guard = self.session.lock().await;
            let operation = async {
                if guard.is_none() {
                    *guard = Some(self.start().await?);
                }
                let value = guard
                    .as_mut()
                    .ok_or_else(|| Error::Invalid("Evaluator unavailable".into()))?
                    .rpc(
                        "evaluate",
                        &request.request_id,
                        serde_json::to_value(&request)?,
                    )
                    .await?;
                let result: EvaluationResult = serde_json::from_value(value)?;
                result.validate()?;
                Ok(result)
            };
            // Includes cold import/handshake; the Python worker separately enforces its per-job budget.
            let response = tokio::time::timeout(Duration::from_secs(60), operation).await;
            match response {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(error)) => {
                    guard.take();
                    Err(error)
                }
                Err(_) => {
                    guard.take();
                    Err(Error::Invalid(
                        "Evaluator exceeded 60s transport deadline".into(),
                    ))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn probe_retains_session_and_protocol_failure_allows_restart() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let venv = root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        let python = std::env::var_os("EMBER_TEST_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if venv.is_file() {
                    venv
                } else {
                    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
                }
            });
        let directory = tempfile::tempdir().unwrap();
        let fixture = directory.path().join("fixture.py");
        std::fs::write(
            &fixture,
            r#"
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['method'] == 'hello':
        result = dict(protocolVersion=1, catalogVersion='test', engineVersion='test')
    else:
        if request['params']['code'] == 'bad_protocol':
            print(json.dumps(dict(id='wrong-id',result={})),flush=True)
            continue
        result = dict(outcome='passed',passed=1,total=1,results=[os.getpid()],totalTimeMs=1)
    print(json.dumps(dict(id=request['id'],result=result)),flush=True)
"#,
        )
        .unwrap();
        let evaluator = LocalEvaluator::new(
            Launch {
                executable: python,
                args: vec![fixture.to_string_lossy().into_owned()],
                working_dir: root,
            },
            "test".into(),
        );
        evaluator.probe().await.unwrap();
        let pid = evaluator
            .session
            .lock()
            .await
            .as_ref()
            .unwrap()
            .child
            .id()
            .unwrap();
        evaluator.probe().await.unwrap();
        let request = |code: &str| EvaluationRequest {
            request_id: "job".into(),
            task_id: "relu".into(),
            code: code.into(),
            test_indices: None,
            resource_budget: Default::default(),
        };
        let worker_pid = evaluator.evaluate(request("valid")).await.unwrap().results[0].clone();
        for _ in 0..2 {
            let result = evaluator.evaluate(request("valid")).await.unwrap();
            assert_eq!(result.results[0], worker_pid);
            assert_eq!(
                evaluator.session.lock().await.as_ref().unwrap().child.id(),
                Some(pid)
            );
        }
        assert!(evaluator.evaluate(request("bad_protocol")).await.is_err());
        assert!(evaluator.session.lock().await.is_none());
        let recovered = evaluator.evaluate(request("valid")).await.unwrap();
        assert_ne!(recovered.results[0], worker_pid);
        evaluator.shutdown().await;
        assert!(evaluator.session.lock().await.is_none());
    }
}
