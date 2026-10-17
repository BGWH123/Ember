//! Open-loop application-core load driver. Deliberately separate from the serial NDJSON adapter.
use ember_core::{Application, Error};
use ember_evaluator::local::Launch;
use serde_json::{json, Value};
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("EMBER_ROOT")?);
    let data = PathBuf::from(std::env::var("EMBER_DATA")?);
    let mode = std::env::var("STRESS_MODE").unwrap_or("mixed".into());
    let burst = std::env::var("STRESS_ARRIVAL").as_deref() == Ok("burst");
    let rate: u64 = std::env::var("STRESS_RATE")
        .unwrap_or("500".into())
        .parse()?;
    let seconds: u64 = std::env::var("STRESS_SECONDS")
        .unwrap_or("30".into())
        .parse()?;
    let seed: usize = std::env::var("STRESS_HISTORY")
        .unwrap_or("10000".into())
        .parse()?;
    if rate == 0 || rate > 100_000 || seconds == 0 || seconds > 3600 || seed > 1_000_000 {
        return Err("Workload exceeds driver limits".into());
    }
    let app = Arc::new(Application::open(
        &root.join("content/catalog"),
        &data,
        None,
    )?);
    // Fixture insertion is outside the timed region and uses the production schema.
    let mut db = rusqlite::Connection::open(data.join("ember-desktop.db"))?;
    let tx = db.transaction()?;
    for i in 0..seed {
        let id = format!("fixture-{i}");
        tx.execute(
            "INSERT OR IGNORE INTO submissions VALUES(?,?,?,'submit',?,?,'completed',?,?)",
            rusqlite::params![
                id,
                if i % 100 == 0 { "relu" } else { "attention" },
                "def relu(x): return x",
                id,
                id,
                r#"{"outcome":"passed","totalTimeMs":1}"#,
                format!("{i:012}")
            ],
        )?;
    }
    tx.commit()?;
    drop(db);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;
    let workloads: Value =
        serde_json::from_slice(&std::fs::read(root.join("scripts/loadtest/workload.json"))?)?;
    if mode == "evaluate" || mode == "fault" {
        app.use_development_evaluator(Launch {
            executable: std::env::var("EMBER_DEV_PYTHON")?.into(),
            args: vec![root
                .join("modules/evaluator/runner.py")
                .to_string_lossy()
                .into()],
            working_dir: root.clone(),
        })?;
        // Cold start excluded from warm stress latency; measured separately by compare.py.
        let result = rt.block_on(app.request(
            "/api/submit",
            json!({"taskId":"relu","code":workloads[0]["code"]}),
        ))?;
        if result["outcome"] != "passed" {
            return Err(result.to_string().into());
        }
    }
    let inflight = Arc::new(AtomicUsize::new(0));
    println!("{}", json!({"ready":true}));
    std::io::stdout().flush()?;
    let (send, receive) = mpsc::sync_channel::<Value>(256);
    let writer = std::thread::spawn(move || {
        let mut output = std::io::BufWriter::new(std::io::stdout());
        for row in receive {
            serde_json::to_writer(&mut output, &row).expect("write sample");
            writeln!(output).expect("write newline");
        }
        output.flush().expect("flush samples");
    });
    let start = Instant::now();
    let mut dropped = 0;
    for i in 0..rate * seconds {
        let arrival_seconds = if burst {
            (i / rate) as f64
        } else {
            i as f64 / rate as f64
        };
        let scheduled = start + Duration::from_secs_f64(arrival_seconds);
        if let Some(wait) = scheduled.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        if inflight.load(Ordering::Relaxed) >= 256
            || scheduled.elapsed() > Duration::from_millis(100)
        {
            dropped += 1;
            continue;
        }
        inflight.fetch_add(1, Ordering::Relaxed);
        let app = app.clone();
        let inflight = inflight.clone();
        let send = send.clone();
        let is_eval = (mode == "evaluate" || mode == "fault") && i % 5 == 0;
        let (path, payload) = if is_eval {
            let w = &workloads[(i / 5 % 100 >= 40) as usize
                + (i / 5 % 100 >= 65) as usize
                + (i / 5 % 100 >= 85) as usize];
            let code = if mode == "fault" && i == 0 {
                json!("while True: pass")
            } else {
                w["code"].clone()
            };
            ("/api/submit", json!({"taskId":w["taskId"],"code":code}))
        } else if mode == "catalog" {
            ("/api/problems", Value::Null)
        } else if mode == "history" || i % 10 < 5 {
            ("/api/submissions/relu", Value::Null)
        } else if i % 10 < 7 {
            ("/api/problems", Value::Null)
        } else if i % 10 < 9 {
            ("/api/progress", Value::Null)
        } else {
            ("draft", Value::Null)
        };
        rt.spawn(async move {
            let dispatched = Instant::now();
            let result = if path == "draft" {
                let a = app.clone();
                tokio::task::spawn_blocking(move || a.storage.save_draft("relu", &format!("# revision {i}\ndef relu(x): return x"))).await.expect("draft worker").map(|_| Value::Null)
            } else { app.request(path, payload).await };
            let (outcome, bytes) = match result {
                Ok(value) => {
                    let outcome = value["outcome"].as_str().unwrap_or("ok").to_owned();
                    (outcome, serde_json::to_vec(&value).expect("response encoding").len())
                },
                Err(Error::Busy(_)) => ("busy".into(), 0),
                Err(e) => (format!("error:{e}"), 0),
            };
            let _ = send.send(json!({"i":i,"path":path,"outcome":outcome,"bytes":bytes,"ms":scheduled.elapsed().as_secs_f64()*1000.,"dispatchLagMs":dispatched.duration_since(scheduled).as_secs_f64()*1000.}));
            inflight.fetch_sub(1, Ordering::Relaxed);
        });
    }
    if let Some(wait) =
        (start + Duration::from_secs(seconds)).checked_duration_since(Instant::now())
    {
        std::thread::sleep(wait);
    }
    drop(send);
    writer.join().expect("sample writer");
    rt.block_on(app.shutdown_local());
    println!(
        "{}",
        json!({"mode":mode,"arrival":if burst {"burst"} else {"constant"},"rate":rate,"seconds":seconds,"seed":seed,"offered":rate*seconds,"generatorDropped":dropped,"elapsedSeconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}
