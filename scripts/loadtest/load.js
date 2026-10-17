import http from 'k6/http';
import exec from 'k6/execution';
import { sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

function number(name, fallback, minimum = 0) {
  const value = Number(__ENV[name] === undefined ? fallback : __ENV[name]);
  if (!Number.isFinite(value) || value < minimum) throw new Error(`Invalid ${name}`);
  return value;
}

const mode = __ENV.MODE || 'web';
const suite = __ENV.SUITE || 'all';
const profile = __ENV.PROFILE || 'capacity';
if (!['web', 'grading'].includes(mode)) throw new Error('MODE must be web or grading');
if (!['all', 'read', 'judge'].includes(suite)) throw new Error('SUITE must be all, read or judge');
if (!['capacity', 'smoke'].includes(profile)) throw new Error('PROFILE must be capacity or smoke');
const base = (__ENV.TARGET_URL || (mode === 'web' ? 'http://localhost:3000' : 'http://localhost:8000')).replace(/\/$/, '');
const smoke = profile === 'smoke';
const users = Math.floor(number('USERS', 20, 1));
const burstUsers = Math.floor(number('BURST_USERS', smoke ? 0 : Math.min(users, 20)));
const readRps = number('READ_RPS', users / number('READ_INTERVAL_SECONDS', 20, 1), 0.01);
const jobsPerMinute = number('JOBS_PER_MINUTE', users * 60 / number('SUBMIT_INTERVAL_SECONDS', 120, 1), 0.01);
const pollMs = number('POLL_INTERVAL_MS', 250, 1);
const jobTimeout = number('JOB_TIMEOUT_SECONDS', 30, 1);
const sampleRatio = number('SAMPLE_RATIO', 0.7);
if (sampleRatio > 1) throw new Error('SAMPLE_RATIO must be at most 1');
const runId = __ENV.RUN_ID || `ember-${Date.now()}`;
const resultsDir = __ENV.RESULTS_DIR || '.';
const fixtures = JSON.parse(open('./workload.json'));
if (!Array.isArray(fixtures) || !fixtures.length || fixtures.some((f) => !f.taskId || !f.code || !(f.weight > 0))) {
  throw new Error('Invalid workload.json');
}
const weightTotal = fixtures.reduce((sum, item) => sum + item.weight, 0);
const effectiveReadRps = smoke ? 2 : readRps;
const effectiveJobsPerMinute = smoke ? 24 : jobsPerMinute;
const readNames = mode === 'web' ? ['problem_list', 'problem_detail', 'progress', 'history', 'page_html'] : ['health', 'progress', 'history'];
const phases = [
  { name: 'warmup', seconds: smoke ? 5 : 120, factor: 0.1 },
  { name: 'baseline', seconds: smoke ? 5 : 180, factor: 0.25 },
  { name: 'ramp', seconds: smoke ? 5 : 180, factor: 1 },
  { name: 'steady', seconds: smoke ? 10 : 360, factor: 1 },
  { name: 'spike', seconds: smoke ? 5 : 120, factor: 2 },
  { name: 'recovery', seconds: smoke ? 5 : 120, factor: 0.1 },
  { name: 'drain', seconds: smoke ? 10 : 120, factor: 0.1 },
];
const readLatency = new Trend('read_latency_ms', true);
const enqueueLatency = new Trend('enqueue_latency_ms', true);
const acceptedLatency = new Trend('enqueue_accepted_ms', true);
const pollLatency = new Trend('poll_latency_ms', true);
const e2eLatency = new Trend('job_e2e_ms', true);
const queueWait = new Trend('queue_wait_ms', true);
const workerLatency = new Trend('worker_total_ms', true);
const persistLatency = new Trend('persist_ms', true);
const testLatency = new Trend('test_execution_ms', true);
const jobTimingMetrics = { queue_wait_ms: queueWait, total_ms: workerLatency, persist_ms: persistLatency, test_execution_ms: testLatency };
for (const key of ['worker_startup_ms', 'runtime_import_ms', 'setup_ms', 'parse_compile_ms', 'user_code_ms',
  'process_launch_ms', 'process_reap_ms', 'result_transfer_ms', 'result_encode_ms', 'result_decode_ms',
  'claim_ms', 'claim_db_lock_ms', 'persist_db_lock_ms', 'persist_db_commit_ms', 'server_e2e_ms', 'child_total_ms']) {
  jobTimingMetrics[key] = new Trend(key, true);
}
const httpTimingMetrics = {};
for (const key of ['backend', 'dispatch', 'session', 'db_connect', 'db_lock', 'db_sql', 'db_commit', 'db_close',
  'admission', 'admission_cpu', 'db_connect_cpu', 'db_sql_cpu', 'db_commit_cpu', 'result_read', 'result_read_cpu',
  'progress_read', 'progress_read_cpu', 'history_read', 'history_read_cpu', 'proxy_upstream', 'proxy_overhead', 'web']) {
  httpTimingMetrics[key] = new Trend(`api_${key}_ms`, true);
}
const timedHttpRequests = new Counter('timed_http_requests');
const requestErrors = new Rate('infrastructure_error_rate');
const rejectedRate = new Rate('admission_rejected_rate');
const clientSuccess = new Rate('job_client_success_rate');
const verdictSuccess = new Rate('judge_correct_rate');
const offered = new Counter('jobs_offered');
const admitted = new Counter('jobs_admitted');
const rejected = new Counter('jobs_rejected');
const admissionErrors = new Counter('jobs_admission_errors');
const completed = new Counter('jobs_completed');
const correct = new Counter('jobs_correct');
const goodput = new Counter('jobs_within_slo');
const timedOut = new Counter('jobs_client_timeout');
const pollErrors = new Counter('jobs_poll_errors');
const queueFull = new Counter('rejected_queue_full');
const userBusy = new Counter('rejected_user_busy');
const otherRejected = new Counter('rejected_other');
const reads = new Counter('read_requests');

function rateStages(rate, selectedPhases) {
  return selectedPhases.map((p) => ({ duration: `${p.seconds}s`, target: Math.max(1, Math.round(rate * p.factor)) }));
}

const scenarios = {};
if (suite !== 'judge') {
  scenarios.read = {
    executor: 'ramping-arrival-rate', exec: 'browse', startRate: Math.max(1, Math.round(effectiveReadRps * 0.1)),
    timeUnit: '1s', stages: rateStages(effectiveReadRps, phases),
    preAllocatedVUs: Math.floor(number('READ_PRE_VUS', 5, 1)),
    maxVUs: Math.floor(number('READ_MAX_VUS', 30, 1)), gracefulStop: '5s',
  };
}
if (suite !== 'read') {
  scenarios.judge = {
    executor: 'ramping-arrival-rate', exec: 'evaluate', startRate: Math.max(1, Math.round(effectiveJobsPerMinute * 0.1)),
    timeUnit: '1m', stages: rateStages(effectiveJobsPerMinute, phases.slice(0, -1)),
    preAllocatedVUs: Math.floor(number('JUDGE_PRE_VUS', smoke ? 5 : 20, 1)),
    maxVUs: Math.floor(number('JUDGE_MAX_VUS', 40, 1)),
    gracefulStop: `${Math.max(jobTimeout + 5, phases[phases.length - 1].seconds)}s`,
  };
}
const burstAt = phases.slice(0, 4).reduce((sum, phase) => sum + phase.seconds, 0);
if (suite !== 'read' && burstUsers > 0) {
  scenarios.burst = { executor: 'shared-iterations', exec: 'evaluate', vus: burstUsers, iterations: burstUsers,
    startTime: `${burstAt}s`, maxDuration: `${jobTimeout + 10}s`, gracefulStop: '5s' };
}
if (suite === 'judge') {
  // Keep observing recovery even when all client iterations finish early.
  scenarios.probe = {
    executor: 'constant-arrival-rate', exec: 'probe', rate: 1, timeUnit: '10s',
    duration: `${phases.reduce((sum, phase) => sum + phase.seconds, 0)}s`,
    preAllocatedVUs: 1, maxVUs: 2, gracefulStop: '5s',
  };
}
const thresholds = {
  infrastructure_error_rate: ['rate<0.01'],
  dropped_iterations: ['count==0'],
};
if (suite !== 'judge') {
  thresholds.read_latency_ms = [`p(95)<${number('READ_P95_MS', 200, 1)}`, `p(99)<${number('READ_P99_MS', 500, 1)}`];
  thresholds.read_requests = ['count>0'];
  for (const name of readNames) thresholds[`read_latency_ms{name:${name}}`] = ['max>=0'];
}
if (suite !== 'read') {
  thresholds.enqueue_latency_ms = [`p(95)<${number('ENQUEUE_P95_MS', 300, 1)}`];
  thresholds.enqueue_accepted_ms = thresholds.enqueue_latency_ms;
  thresholds.poll_latency_ms = [`p(95)<${number('POLL_P95_MS', 200, 1)}`];
  thresholds.job_e2e_ms = [`p(95)<${number('JOB_P95_MS', 5000, 1)}`];
  thresholds.job_client_success_rate = ['rate>0.99'];
  thresholds.judge_correct_rate = ['rate>0.99'];
  thresholds.admission_rejected_rate = [`rate<${number('MAX_REJECTION_RATE', 0.05)}`];
  for (const metric of ['jobs_offered', 'jobs_admitted', 'jobs_completed', 'jobs_correct']) thresholds[metric] = ['count>0'];
  for (const key of Object.keys(jobTimingMetrics)) thresholds[key === 'total_ms' ? 'worker_total_ms' : key] = ['max>=0'];
  if (burstUsers > 0) {
    thresholds['jobs_offered{scenario:burst}'] = [`count==${burstUsers}`];
    thresholds['jobs_admitted{scenario:burst}'] = [`count==${burstUsers}`];
    thresholds['jobs_correct{scenario:burst}'] = [`count==${burstUsers}`];
    thresholds['job_e2e_ms{scenario:burst}'] = [`p(95)<${number('JOB_P95_MS', 5000, 1)}`];
  }
}
for (const key of Object.keys(httpTimingMetrics)) {
  thresholds[`api_${key}_ms`] = ['max>=0'];
  for (const name of ['enqueue', 'poll', 'progress', 'history']) thresholds[`api_${key}_ms{name:${name}}`] = ['max>=0'];
}
thresholds.timed_http_requests = ['count>0'];
// Submetrics preserve phase sample counts/percentiles in summary.json. Spike is
// diagnostic; steady has its own gates so warmup cannot hide capacity failures.
for (const phase of phases) {
  const tag = `{phase:${phase.name}}`;
  if (suite !== 'judge') {
    thresholds[`read_latency_ms${tag}`] = phase.name === 'steady' ? thresholds.read_latency_ms : ['max>=0'];
    thresholds[`read_requests${tag}`] = ['count>=0'];
  }
  if (suite !== 'read') {
    for (const metric of ['enqueue_latency_ms', 'enqueue_accepted_ms', 'poll_latency_ms', 'job_e2e_ms',
      ...Object.keys(jobTimingMetrics).map(key => key === 'total_ms' ? 'worker_total_ms' : key)]) {
      thresholds[`${metric}${tag}`] = ['max>=0'];
    }
    for (const metric of ['jobs_offered', 'jobs_admitted', 'jobs_rejected', 'jobs_admission_errors', 'jobs_completed', 'jobs_correct', 'jobs_within_slo', 'jobs_client_timeout', 'jobs_poll_errors']) {
      thresholds[`${metric}${tag}`] = ['count>=0'];
    }
    if (phase.name === 'steady') {
      thresholds[`enqueue_latency_ms${tag}`] = thresholds.enqueue_latency_ms;
      thresholds[`enqueue_accepted_ms${tag}`] = thresholds.enqueue_accepted_ms;
      thresholds[`job_e2e_ms${tag}`] = thresholds.job_e2e_ms;
      thresholds[`admission_rejected_rate${tag}`] = thresholds.admission_rejected_rate;
      thresholds[`job_client_success_rate${tag}`] = thresholds.job_client_success_rate;
      thresholds[`judge_correct_rate${tag}`] = thresholds.judge_correct_rate;
    }
  }
}
export const options = {
  scenarios, thresholds, discardResponseBodies: true,
  summaryTrendStats: ['count', 'avg', 'min', 'med', 'p(95)', 'p(99)', 'max'],
  systemTags: ['name', 'method', 'status', 'scenario', 'expected_response', 'error_code'],
  setupTimeout: '180s',
};

function phaseName() {
  let elapsed = (Date.now() - exec.scenario.startTime) / 1000 + (exec.scenario.name === 'burst' ? burstAt : 0);
  for (const phase of phases) {
    if (elapsed < phase.seconds) return phase.name;
    elapsed -= phase.seconds;
  }
  return 'drain';
}

function fixture(index) {
  // A deterministic permutation, independent of VU scheduling.
  let bucket = (index * 37 + 11) % weightTotal;
  for (const item of fixtures) {
    if (bucket < item.weight) return item;
    bucket -= item.weight;
  }
  return fixtures[0];
}

function params(token, tags, body = false, timeoutMs = 5000) {
  const jar = new http.CookieJar();
  if (mode === 'web') jar.set(base, 'session_token', token, { path: '/' });
  return {
    jar, headers: mode === 'grading' ? { 'X-Session-Token': token, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' },
    tags, timeout: `${Math.max(1, timeoutMs)}ms`, responseType: body ? 'text' : 'none', redirects: 0,
  };
}

function json(response) {
  try { return response.json(); } catch (_) { return null; }
}

function captureHttpTimings(response, tags) {
  const header = response.headers['Server-Timing'] || response.headers['server-timing'];
  if (!header) return;
  timedHttpRequests.add(1, tags);
  for (const span of header.split(',')) {
    const match = span.trim().match(/^([a-z_]+);dur=([0-9.]+)/);
    if (match && httpTimingMetrics[match[1]] && Number.isFinite(Number(match[2]))) {
      httpTimingMetrics[match[1]].add(Number(match[2]), tags);
    }
  }
}

export function setup() {
  const path = mode === 'web' ? '/api/problems' : '/health';
  const response = http.get(`${base}${path}`, { timeout: '10s', responseType: 'text', tags: { name: 'preflight' } });
  const data = json(response);
  if (response.status !== 200 || !data || (mode === 'web' && !Array.isArray(data.problems))) {
    throw new Error(`Preflight failed: ${base}${path} returned ${response.status}`);
  }
  if (mode === 'web') {
    const ids = new Set(data.problems.map((item) => item.id));
    for (const item of fixtures) if (!ids.has(item.taskId)) throw new Error(`Missing task ${item.taskId}`);
  }
  if (suite !== 'read') {
    // Prove the canonical inputs can pass before interpreting judge throughput.
    for (const task of fixtures) {
      const token = `${runId}-preflight`;
      const payload = { taskId: task.taskId, code: task.code, mode: 'submit', idempotencyKey: `${runId}-preflight-${task.taskId}` };
      const response = http.post(`${base}${mode === 'web' ? '/api/submit' : '/submissions'}`, JSON.stringify(payload), params(token, { name: 'preflight_enqueue' }, true));
      const submission = json(response);
      if (response.status !== 202 || !submission || !submission.submission_id) throw new Error(`Judge preflight admission failed for ${task.taskId}: HTTP ${response.status}`);
      const path = mode === 'web' ? `/api/submissions/status/${submission.submission_id}` : `/submissions/${submission.submission_id}`;
      const deadline = Date.now() + 30000;
      let passed = false;
      while (Date.now() < deadline) {
        const response = http.get(`${base}${path}`, params(token, { name: 'preflight_poll' }, true));
        const result = json(response);
        if (response.status !== 200 || !result) throw new Error(`Judge preflight polling failed for ${task.taskId}`);
        if (result.status === 'completed') {
          passed = result.allPassed === true && result.total > 0 && !result.error;
          if (!passed) throw new Error(`Judge preflight failed for ${task.taskId}: ${JSON.stringify({ error: result.error, passed: result.passed, total: result.total })}`);
          break;
        }
        sleep(0.25);
      }
      if (!passed) throw new Error(`Judge preflight timed out for ${task.taskId}`);
    }
  }
  console.log(JSON.stringify({ profile, mode, suite, users, readRps: effectiveReadRps, jobsPerMinute: effectiveJobsPerMinute, pollMs, jobTimeout, phases }));
}

export function probe() {
  const response = http.get(`${base}${mode === 'web' ? '/api/problems' : '/health'}`, params(`${runId}-probe`, { name: 'recovery_probe', phase: phaseName() }));
  requestErrors.add(response.status !== 200, { name: 'recovery_probe', phase: phaseName() });
}

export function browse() {
  const index = exec.scenario.iterationInTest;
  const task = fixture(index);
  const token = `${runId}-user-${index % users}`;
  const paths = mode === 'web'
    ? ['/api/problems', `/api/problems/${task.taskId}`, '/api/progress', `/api/submissions/${task.taskId}`, '/problems']
    : ['/health', '/progress', `/submissions/history/${task.taskId}`];
  const position = index % paths.length;
  const tags = { phase: phaseName(), name: readNames[position] };
  const response = http.get(`${base}${paths[position]}`, params(token, tags));
  captureHttpTimings(response, tags);
  readLatency.add(response.timings.duration, tags);
  reads.add(1, tags);
  requestErrors.add(response.status !== 200, tags);
}

export function evaluate() {
  const index = exec.scenario.iterationInTest;
  const task = fixture(index);
  const burst = exec.scenario.name === 'burst';
  const operation = burst ? 'submit' : ((index * 47 + 3) % 100) / 100 < sampleRatio ? 'sample' : 'submit';
  const token = burst ? `${runId}-burst-user-${index}` : `${runId}-user-${index % users}`;
  const tags = { phase: phaseName(), task: task.taskId, operation };
  const start = Date.now();
  const path = mode === 'web' ? (operation === 'sample' ? '/api/run' : '/api/submit') : '/submissions';
  const payload = { taskId: task.taskId, code: task.code, mode: operation, idempotencyKey: `${runId}-${exec.scenario.name}-${index}` };
  if (operation === 'sample') payload.testIndices = [0, 1];
  offered.add(1, tags);
  const response = http.post(`${base}${path}`, JSON.stringify(payload), params(token, { ...tags, name: 'enqueue' }, true));
  captureHttpTimings(response, { ...tags, name: 'enqueue' });
  enqueueLatency.add(response.timings.duration, tags);
  const data = json(response);
  rejectedRate.add(response.status === 429, tags);
  requestErrors.add(response.status !== 202 && response.status !== 429, { ...tags, name: 'enqueue' });
  if (response.status === 429) {
    rejected.add(1, tags);
    const code = data && data.detail && data.detail.code;
    (code === 'tenant_queue_full' ? queueFull : code === 'user_concurrency_limited' ? userBusy : otherRejected).add(1, tags);
    return;
  }
  if (response.status !== 202 || !data || !data.submission_id) {
    admissionErrors.add(1, tags);
    return;
  }
  admitted.add(1, tags);
  acceptedLatency.add(response.timings.duration, tags);
  const statusPath = mode === 'web' ? `/api/submissions/status/${data.submission_id}` : `/submissions/${data.submission_id}`;
  const deadline = start + jobTimeout * 1000;
  while (Date.now() < deadline) {
    const remaining = deadline - Date.now();
    const pollTags = { ...tags, name: 'poll', phase: phaseName() };
    const poll = http.get(`${base}${statusPath}`, params(token, pollTags, true, Math.min(5000, remaining)));
    captureHttpTimings(poll, pollTags);
    pollLatency.add(poll.timings.duration, pollTags);
    const result = json(poll);
    const badResponse = poll.status !== 200 || !result || !['queued', 'running', 'completed', 'cancelled'].includes(result.status);
    requestErrors.add(badResponse, pollTags);
    if (badResponse) {
      pollErrors.add(1, { ...tags, phase: phaseName() });
      clientSuccess.add(false, tags);
      return;
    }
    if (result.status === 'completed' || result.status === 'cancelled') {
      const completionTags = { ...tags, phase: phaseName() };
      const valid = result.status === 'completed' && result.allPassed === true && result.total > 0 && !result.error;
      completed.add(1, completionTags);
      correct.add(valid ? 1 : 0, completionTags);
      goodput.add(valid && Date.now() - start < number('JOB_P95_MS', 5000, 1) ? 1 : 0, completionTags);
      verdictSuccess.add(valid, tags);
      clientSuccess.add(valid, tags);
      e2eLatency.add(Date.now() - start, tags);
      const timings = result.timings || {};
      for (const [key, metric] of Object.entries(jobTimingMetrics)) {
        if (Number.isFinite(timings[key])) metric.add(timings[key], tags);
      }
      return;
    }
    sleep(Math.max(0, Math.min(pollMs, deadline - Date.now())) / 1000);
  }
  timedOut.add(1, { ...tags, phase: phaseName() });
  clientSuccess.add(false, tags);
}

export function handleSummary(data) {
  const config = { runId, base, profile, mode, suite, users, burstUsers, burstAt, readRps: effectiveReadRps, jobsPerMinute: effectiveJobsPerMinute,
    plannedReadRps: readRps, plannedJobsPerMinute: jobsPerMinute, readNames, pollMs, jobTimeout, sampleRatio, phases, scenarios };
  const failures = [];
  for (const [name, metric] of Object.entries(data.metrics)) {
    for (const [rule, result] of Object.entries(metric.thresholds || {})) if (!result.ok) failures.push(`${name}: ${rule}`);
  }
  return {
    [`${resultsDir}/summary.json`]: JSON.stringify(data, null, 2),
    [`${resultsDir}/config.json`]: JSON.stringify(config, null, 2),
    stdout: `\nEmber load test: ${(data.metrics.read_requests && data.metrics.read_requests.values.count > 0) || (data.metrics.jobs_offered && data.metrics.jobs_offered.values.count > 0) ? (failures.length ? 'SLO FAILED' : 'SLO PASSED') : 'NO LOAD GENERATED'}\n${failures.join('\n')}\nArtifacts: ${resultsDir}\n`,
  };
}
