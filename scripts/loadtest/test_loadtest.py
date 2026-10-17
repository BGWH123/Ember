"""Timing, read-only WAL observation and optional real-k6 failure accounting."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from grading_service import database
from grading_service.domain import JudgeResult, Outcome
from grading_service.service import SubmissionService
from grading_service.judge import execute, prewarm
from grading_service.domain import ResourceBudget
from scripts.loadtest.observe import container_program, database_snapshot, docker_snapshot
from scripts.loadtest.report import render


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'test.db'
    monkeypatch.setattr(database, 'DB_PATH', str(path))
    database.initialize()
    tenant, user = database.resolve_local_session('test-session')
    return path, tenant, user


def test_queue_wait_excludes_execution_and_persist_is_saved(store, monkeypatch):
    path, tenant, user = store
    service = SubmissionService()
    created = service.create(tenant, user, 'relu', 'def relu(x): return x', 'submit', 'timing-test')
    with database.transaction() as conn:
        conn.execute("UPDATE judge_job SET queued_at='2026-10-04T00:00:00+00:00'")
        job = conn.execute('SELECT id FROM judge_job').fetchone()['id']
    monkeypatch.setattr('grading_service.service.now', lambda: '2026-10-04T00:00:01+00:00')
    monkeypatch.setattr('grading_service.service.execute', lambda *_args, **_kwargs: JudgeResult(Outcome.PASSED, 1, 1, total_ms=9000))
    service._run(job)
    result = service.get(tenant, created['submission_id'])
    assert result['timings']['queue_wait_ms'] == 1000
    assert result['timings']['persist_ms'] > 0
    assert result['timings']['persist_db_commit_ms'] >= 0
    assert result['status'] == 'completed' and result['allPassed']
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT queue_wait_ms,persist_ms FROM judge_result').fetchone() == (1000, result['timings']['persist_ms'])


def test_database_observer_can_read_while_writer_holds_wal(store):
    path, tenant, user = store
    service = SubmissionService()
    service.create(tenant, user, 'relu', 'def relu(x): return x', 'submit', 'observer-test')
    with database.transaction() as writer:
        writer.execute("UPDATE judge_job SET status='running'")
        snapshot = database_snapshot(str(path), '2020-01-01T00:00:00+00:00')
        assert snapshot['states'] == {'queued': 1}  # committed view, without writer lock
        assert snapshot['finished_since_start'] == 0
        assert snapshot['avg_queue_wait_ms'] is None
    assert database_snapshot(str(path), '2020-01-01T00:00:00+00:00')['states'] == {'running': 1}


def test_database_observer_does_not_create_missing_db(tmp_path):
    path = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        database_snapshot(str(path), '2020-01-01T00:00:00+00:00')
    assert not path.exists()


def test_container_sampler_compiles_as_standalone_stdin_program():
    compile(container_program('2026-10-04T00:00:00+00:00'), '<container-sampler>', 'exec')


@pytest.mark.parametrize('error', ['Error response from daemon: No such object', 'stats stream closed'])
def test_sampler_ignores_own_finished_generator_and_preserves_grading_error(monkeypatch, error):
    def fake_command(args, **_kwargs):
        if args[:2] == ['docker', 'ps']:
            if args[-1] == 'id=generator-id':
                return ''
            return 'grading-id' if 'label=' in args[-1] else 'generator-id'
        raise subprocess.CalledProcessError(1, args, stderr=error)
    monkeypatch.setattr('scripts.loadtest.observe.command', fake_command)
    result = docker_snapshot('test-project', 'own-generator', '2026-10-04T00:00:00+00:00')
    assert [item['container'] for item in result['errors']] == ['grading-id']
    assert error in result['errors'][0]['error']


def test_all_catalog_tests_compile_before_service_start():
    # Includes Python 3.11, used by grading.Dockerfile.
    prewarm()


def test_native_child_abort_is_mapped_to_result_not_raised(monkeypatch):
    class Pipe:
        closed = False
        def poll(self, *_args): return True
        def recv_bytes(self, **_kwargs): raise EOFError
        def close(self): self.closed = True

    class Process:
        exitcode = -6
        pid = 0
        def start(self): pass
        def join(self, *_args, **_kwargs): pass
        def is_alive(self): return False
        def close(self): pass

    class Context:
        def Pipe(self, **_kwargs): return parent, child
        def Process(self, **_kwargs): return Process()
        def get_start_method(self): return 'test'

    parent, child = Pipe(), Pipe()
    result = execute('relu', 'def relu(x): return x', None, ResourceBudget(), context=Context())
    assert result.outcome == Outcome.CRASHED and result.error_code == 'worker_crashed'
    assert 'exit code -6' in result.error
    assert parent.closed and child.closed


def test_legacy_submission_migration_is_repeatable(tmp_path, monkeypatch):
    path = tmp_path / 'legacy.db'
    monkeypatch.setattr(database, 'DB_PATH', str(path))
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE users(id INTEGER PRIMARY KEY,session_token TEXT,created_at TEXT);
            CREATE TABLE submissions(id INTEGER PRIMARY KEY,user_id INTEGER,task_id TEXT,code TEXT,passed INTEGER,exec_time_ms INTEGER,submitted_at TEXT);
            INSERT INTO users VALUES(1,'legacy-token','2026-10-01T00:00:00+00:00');
            INSERT INTO submissions VALUES(2,1,'relu','def relu(x): return x',1,10,'2026-10-01T00:00:00+00:00');
        """)
    database.initialize()
    database.initialize()
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT id,user_id,status FROM submission').fetchall() == [('legacy-submission-2', 'legacy-user-1', 'completed')]


def test_report_marks_preflight_failure_and_absent_samples(tmp_path):
    config = dict(base='http://test', mode='web', suite='all', profile='smoke', users=1000, readRps=50,
                  jobsPerMinute=500, pollMs=250, jobTimeout=30, phases=[dict(name='steady', seconds=10)])
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(metrics={'read_latency_ms{phase:steady}': dict(type='trend', values={'count': 0, 'p(95)': 0, 'p(99)': 0})})))
    (tmp_path / 'execution.json').write_text(json.dumps(dict(exit_code=107)))
    report = render(tmp_path)
    assert '未完成（预检或执行失败）' in report
    assert '未采集/未采集' in report


@pytest.mark.skipif(os.environ.get('EMBER_TEST_K6') != '1', reason='set EMBER_TEST_K6=1 to run a 45s Docker/k6 integration test')
@pytest.mark.parametrize('mode', ['web', 'grading'])
def test_real_k6_accounts_for_rejection_errors_timeouts_and_wrong_results(tmp_path, mode):
    submissions = {}
    fixtures = json.loads((Path(__file__).parent / 'workload.json').read_text())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def token(self):
            if mode == 'grading':
                return self.headers.get('X-Session-Token')
            cookies = SimpleCookie(self.headers.get('Cookie', ''))
            return cookies['session_token'].value if 'session_token' in cookies else None

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            key = payload['idempotencyKey']
            assert self.token(), 'load generator must send the browser session cookie'
            bucket = -1 if '-preflight-' in key else int(key.rsplit('-', 1)[1]) % 7
            if bucket == 0:
                self.send(429, {'detail': {'code': 'tenant_queue_full'}})
            elif bucket == 1:
                self.send(500, {'error': 'test admission failure'})
            else:
                submissions[key] = (self.token(), bucket)
                self.send(202, {'submission_id': key, 'status': 'queued'})

        def do_GET(self):
            if self.path in ('/api/problems', '/health'):
                self.send(200, {'problems': [{'id': item['taskId']} for item in fixtures]})
            elif self.path.startswith('/api/submissions/status/') or (self.path.startswith('/submissions/') and '/history/' not in self.path):
                token, bucket = submissions[self.path.rsplit('/', 1)[1]]
                if self.token() != token:
                    self.send(401, {'error': 'wrong session'})
                elif bucket == 2:
                    self.send(503, {'error': 'test polling failure'})
                elif bucket == 4:
                    self.send(200, {'status': 'queued'})
                else:
                    self.send(200, {'status': 'completed', 'allPassed': bucket != 3, 'passed': 0 if bucket == 3 else 1,
                                    'total': 1, 'error': None, 'timings': {'queue_wait_ms': 10, 'total_ms': 20, 'persist_ms': 1}})
            else:
                self.send(200, {})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {**os.environ, 'PROFILE': 'smoke', 'MODE': mode, 'SUITE': 'all', 'K6_RUNNER': 'docker',
               'TARGET_URL': f'http://127.0.0.1:{server.server_port}', 'JOB_TIMEOUT_SECONDS': '1',
               'LOADTEST_OUTPUT_DIR': str(tmp_path / 'evidence')}
        result = subprocess.run(['bash', 'scripts/loadtest/run.sh'], env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 99, result.stdout + result.stderr
        summary = json.loads((tmp_path / 'evidence/summary.json').read_text())['metrics']
        count = lambda name: summary.get(name, {}).get('values', {}).get('count', 0)
        assert count('jobs_offered') == count('jobs_admitted') + count('jobs_rejected') + count('jobs_admission_errors')
        assert count('jobs_admitted') == count('jobs_completed') + count('jobs_client_timeout') + count('jobs_poll_errors')
        for name in ('jobs_rejected', 'jobs_admission_errors', 'jobs_client_timeout', 'jobs_poll_errors', 'jobs_correct'):
            assert count(name) > 0, name
        assert count('jobs_correct') < count('jobs_completed')
        assert count('dropped_iterations') == 0
        assert (tmp_path / 'evidence/dashboard.html').stat().st_size > 0
        config = json.loads((tmp_path / 'evidence/config.json').read_text())
        assert config['readRps'] == 2 and config['jobsPerMinute'] == 24
        assert '未通过' in (tmp_path / 'evidence/report.md').read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
