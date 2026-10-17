#!/usr/bin/env bash
set -euo pipefail

repo_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_path"
export PROFILE="${PROFILE:-capacity}"
export MODE="${MODE:-web}"
export SUITE="${SUITE:-all}"
export USERS="${USERS:-20}"
export RUN_ID="${RUN_ID:-ember-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
default_target=http://localhost:3000
if [[ "$MODE" == grading ]]; then default_target=http://localhost:8000; fi
export TARGET_URL="${TARGET_URL:-$default_target}"
artifact_path="${LOADTEST_OUTPUT_DIR:-$repo_path/artifacts/loadtest/$RUN_ID}"
mkdir -p "$artifact_path"
artifact_path="$(cd -- "$artifact_path" && pwd)"
generator_name="ember-loadgen-$$"
runner_kind="${K6_RUNNER:-auto}"
if [[ "$runner_kind" == auto ]]; then
  if command -v k6 >/dev/null 2>&1; then runner_kind=local; else runner_kind=docker; fi
fi
if [[ "$runner_kind" != local && "$runner_kind" != docker ]]; then
  echo 'K6_RUNNER must be auto, local or docker' >&2
  exit 2
fi
k6_image="${K6_IMAGE:-grafana/k6:1.6.0}"
if [[ "$runner_kind" == docker ]]; then
  command -v docker >/dev/null
  docker image inspect "$k6_image" >/dev/null 2>&1 || docker pull "$k6_image"
  runner=(docker run --rm --name "$generator_name" --network host --user "$(id -u):$(id -g)"
    -v "$repo_path:/work:ro" -v "$artifact_path:/results" -w /work
    -e RESULTS_DIR=/results -e K6_WEB_DASHBOARD=true -e K6_WEB_DASHBOARD_PORT=0
    -e K6_WEB_DASHBOARD_EXPORT=/results/dashboard.html -e K6_NEW_MACHINE_READABLE_SUMMARY=false)
  for variable in PROFILE MODE SUITE USERS BURST_USERS RUN_ID TARGET_URL READ_RPS READ_INTERVAL_SECONDS JOBS_PER_MINUTE SUBMIT_INTERVAL_SECONDS POLL_INTERVAL_MS JOB_TIMEOUT_SECONDS SAMPLE_RATIO READ_PRE_VUS READ_MAX_VUS JUDGE_PRE_VUS JUDGE_MAX_VUS READ_P95_MS READ_P99_MS ENQUEUE_P95_MS POLL_P95_MS JOB_P95_MS MAX_REJECTION_RATE; do
    if [[ -n "${!variable:-}" ]]; then runner+=(-e "$variable"); fi
  done
  runner+=("$k6_image")
  raw_path=/results/samples.json.gz
else
  command -v k6 >/dev/null
  export RESULTS_DIR="$artifact_path"
  export K6_WEB_DASHBOARD=true K6_WEB_DASHBOARD_PORT=0
  export K6_WEB_DASHBOARD_EXPORT="$artifact_path/dashboard.html"
  export K6_NEW_MACHINE_READABLE_SUMMARY=false
  runner=(k6)
  raw_path="$artifact_path/samples.json.gz"
  generator_name=''
fi
k6_version="$("${runner[@]}" version)"
if [[ "$k6_version" != *'k6 v1.'* ]]; then
  echo "本脚本要求 k6 1.x，当前为：$k6_version；可设置 K6_RUNNER=docker 使用固定版本。" >&2
  exit 2
fi
python3 - "$artifact_path/context.json" "$runner_kind" "$k6_image" "$k6_version" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
with open(sys.argv[1], 'w') as output:
    sources = [p for root in ('grading_service','torch_judge/tasks','web/src') for p in Path(root).rglob('*') if p.suffix in ('.py','.ts','.tsx')]
    json.dump(dict(commit=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'), platform=platform.platform(), started_at=datetime.now(timezone.utc).isoformat(), runner=sys.argv[2], image=sys.argv[3], k6_version=sys.argv[4], loadtest_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('scripts/loadtest').iterdir() if p.suffix in ('.js','.json','.sh','.py')}, application_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, configuration={k:v for k,v in os.environ.items() if k in ('TARGET_URL','PROFILE','MODE','SUITE','USERS','BURST_USERS','RUN_ID','READ_RPS','JOBS_PER_MINUTE','POLL_INTERVAL_MS')}), output, indent=2)
Path(sys.argv[1]).with_name('worktree.patch').write_text(git('diff', '--binary'))
PY
observer_pid=''
cleanup() {
  if [[ -n "$observer_pid" ]]; then
    kill -TERM "$observer_pid" 2>/dev/null || true
    wait "$observer_pid" 2>/dev/null || true
  fi
  if [[ -n "$generator_name" ]]; then docker stop -t 2 "$generator_name" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 scripts/loadtest/observe.py --output "$artifact_path/resources.jsonl" \
  --project "${OBSERVE_COMPOSE_PROJECT:-}" --generator "$generator_name" \
  --db "${OBSERVE_DB_PATH:-}" --pids "${OBSERVE_PIDS:-}" \
  --interval "${OBSERVE_INTERVAL_SECONDS:-5}" --duration 1400 \
  >"$artifact_path/observer.log" 2>&1 &
observer_pid=$!
echo "Ember 压测：$TARGET_URL ($PROFILE, $MODE/$SUITE)"
echo "产物目录：$artifact_path"
result=0
"${runner[@]}" run --address 127.0.0.1:0 --quiet --out "json=$raw_path" scripts/loadtest/load.js > >(tee "$artifact_path/k6.log") 2>&1 || result=$?
cleanup
observer_pid=''
generator_name=''
python3 - "$artifact_path/execution.json" "$result" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as output:
    json.dump(dict(exit_code=int(sys.argv[2])), output)
PY
if [[ -f "$artifact_path/summary.json" ]]; then
  python3 scripts/loadtest/report.py "$artifact_path"
fi
echo "k6 退出码：$result；0 表示预设 SLO 通过，99 表示阈值失败。"
exit "$result"
