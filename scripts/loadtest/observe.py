"""Bounded, read-only sampling of explicitly selected Ember services (stdlib only)."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import os
import platform
import signal
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str], timeout: float = 4) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout.strip()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def database_snapshot(path: str, since: str) -> dict:
    """Observe WAL updates without BEGIN IMMEDIATE or creating a missing database."""
    tick = time.monotonic()
    conn = sqlite3.connect(f"file:{quote(str(Path(path).resolve()))}?mode=ro", uri=True, timeout=0.2)
    try:
        # One snapshot, read-only transaction; never retain it between samples.
        conn.execute("BEGIN")
        states = dict(conn.execute("SELECT status,count(*) FROM judge_job GROUP BY status"))
        outcomes = dict(conn.execute("SELECT outcome,count(*) FROM judge_result WHERE created_at>=? GROUP BY outcome", (since,)))
        errors = dict(conn.execute("SELECT COALESCE(error_code,'none'),count(*) FROM judge_result WHERE created_at>=? GROUP BY error_code", (since,)))
        oldest = conn.execute("SELECT MAX(0,COALESCE((julianday('now')-julianday(MIN(queued_at)))*86400,0)) FROM judge_job WHERE status='queued'").fetchone()[0]
        timings = conn.execute("SELECT AVG((julianday(started_at)-julianday(queued_at))*86400000), AVG((julianday(finished_at)-julianday(started_at))*86400000), count(*) FROM judge_job WHERE finished_at>=?", (since,)).fetchone()
        return {
            "states": states, "outcomes_since_start": outcomes, "errors_since_start": errors,
            "oldest_queued_seconds": oldest,
            "finished_since_start": timings[2], "avg_queue_wait_ms": timings[0], "avg_worker_wall_ms": timings[1],
            "db_bytes": Path(path).stat().st_size,
            "wal_bytes": Path(path + '-wal').stat().st_size if Path(path + '-wal').exists() else 0,
            "sample_ms": (time.monotonic() - tick) * 1000,
        }
    finally:
        conn.close()


def cgroup_snapshot() -> dict:
    root = '/sys/fs/cgroup/'
    values = {name: read_text(root + name) for name in ('cpu.stat', 'cpu.max', 'memory.current', 'memory.max', 'memory.events', 'pids.current')}
    # None means unavailable (e.g. cgroup v1), never zero usage.
    if values['cpu.stat']:
        values['cpu.stat'] = {line.split()[0]: int(line.split()[1]) for line in values['cpu.stat'].splitlines()}
    values['runtime'] = {'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version}
    for package in ('torch', 'numpy', 'fastapi', 'uvicorn'):
        try:
            values['runtime'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    values['pid1_status'] = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in (read_text('/proc/1/status') or '').splitlines() if line.startswith(('VmSize:', 'VmRSS:', 'Threads:'))}
    return values


def host_snapshot() -> dict:
    memory = read_text('/proc/meminfo') or ''
    wanted = {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}
    return {
        "cpu_stat": (read_text('/proc/stat') or '').splitlines()[:1],
        "memory_kib": {line.split(':')[0]: int(line.split()[1]) for line in memory.splitlines() if line.split(':')[0] in wanted},
        "pressure": {name: read_text(f'/proc/pressure/{name}') for name in ('cpu', 'io', 'memory')},
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
    }


def process_snapshot(pids: list[int]) -> list[dict]:
    # Include disposable judge children; CPU ticks of already-exited children
    # are not recoverable here. Container cpu.stat includes those CPU costs.
    table = {}
    for folder in Path('/proc').glob('[0-9]*'):
        try:
            stat = (folder / 'stat').read_text()
            rest = stat[stat.rfind(')') + 2:].split()
            table[int(folder.name)] = {"pid": int(folder.name), "ppid": int(rest[1]), "cpu_ticks": int(rest[11]) + int(rest[12]), "threads": int(rest[17]), "vms_bytes": int(rest[20]), "rss_bytes": int(rest[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    selected = set(pids)
    while True:
        children = {pid for pid, data in table.items() if data['ppid'] in selected}
        if children <= selected:
            break
        selected |= children
    return [table[pid] for pid in sorted(selected) if pid in table]


def container_program(since: str) -> str:
    # Anchor to a real line boundary: the marker also appears in this source
    # as a string literal and must not truncate its own function definition.
    source = Path(__file__).read_text().rsplit("\nif __name__ == '__main__':", 1)[0]
    return source + f"\nprint(json.dumps(dict(database=database_snapshot(os.environ.get('DB_PATH','/app/data/pyre.db'),{since!r}),cgroup=cgroup_snapshot())))\n"


def docker_snapshot(project: str, generator: str, since: str) -> dict:
    result = {"containers": [], "grading": []}
    ids = command(['docker', 'ps', '-q', '--filter', f'label=com.docker.compose.project={project}']).splitlines() if project else []
    generator_ids = []
    if generator:
        generator_ids = command(['docker', 'ps', '-q', '--filter', f'name=^/{generator}$']).splitlines()
        ids += generator_ids
    for container in ids:
        try:
            info = json.loads(command(['docker', 'inspect', container]))[0]
            stats = json.loads(command(['docker', 'stats', '--no-stream', '--format', '{{json .}}', container]))
            labels = info['Config'].get('Labels') or {}
            role = labels.get('com.docker.compose.service', 'load_generator')
            stats.update(role=role, limits=info['HostConfig'])
            stats['image_id'] = info['Image']
            # Record resource limits without retaining environment variables.
            stats['limits'] = {key: stats['limits'].get(key) for key in ('NanoCpus', 'Memory', 'CpusetCpus', 'CpuQuota', 'CpuPeriod')}
            result['containers'].append(stats)
            if role == 'grading':
                # Existing grading images do not contain scripts/. Pipe this
                # file into Python's stdin, avoiding writes in the container.
                sampled = subprocess.run(['docker', 'exec', '-i', container, 'python', '-'], input=container_program(since), capture_output=True, text=True, timeout=4)
                if sampled.returncode:
                    raise RuntimeError(sampled.stderr.strip()[-2000:] or f'container sampler exited {sampled.returncode}')
                result['grading'].append({"container": info['Name'], **json.loads(sampled.stdout)})
        except (OSError, ValueError, subprocess.SubprocessError, KeyError, RuntimeError) as exc:
            if container in generator_ids:
                disappeared = isinstance(exc, subprocess.CalledProcessError) and any(
                    message in (exc.stderr or '').lower() for message in ('no such container', 'no such object'))
                if not disappeared:
                    try:
                        disappeared = not command(['docker', 'ps', '-q', '--filter', f'id={container}'])
                    except (OSError, subprocess.SubprocessError):
                        pass
                if disappeared:
                    continue  # our --rm generator exited between discovery and stats
            reason = type(exc).__name__ + ': ' + str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                reason += ': ' + exc.stderr.strip()[-2000:]
            result.setdefault('errors', []).append({"container": container, "error": reason})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--project', default='')
    parser.add_argument('--generator', default='')
    parser.add_argument('--db', default='')
    parser.add_argument('--pids', default='')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--duration', type=float, default=1300)
    parser.add_argument('--since', default=utc_now())
    args = parser.parse_args()
    if args.interval <= 0 or args.duration <= 0:
        parser.error('interval and duration must be positive')
    pids = [int(value) for value in args.pids.split(',') if value]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        cpu_models = sorted({line.split(':', 1)[1].strip() for line in (read_text('/proc/cpuinfo') or '').splitlines() if line.startswith('model name')})
        output.write(json.dumps({"type": "context", "time": utc_now(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "cpu_models": cpu_models, "since": args.since, "project": args.project, "pids": pids, "scope": "local host and explicitly selected services"}) + '\n')
        while True:
            tick = time.monotonic()
            sample = {"type": "sample", "time": utc_now(), "host": host_snapshot()}
            if pids:
                sample['processes'] = process_snapshot(pids)
            if args.db:
                try:
                    sample['database'] = database_snapshot(args.db, args.since)
                except (OSError, sqlite3.Error) as exc:
                    sample['database_error'] = str(exc)
            if args.project or args.generator:
                try:
                    sample['docker'] = docker_snapshot(args.project, args.generator, args.since)
                except (OSError, subprocess.SubprocessError) as exc:
                    sample['docker_error'] = str(exc)
            sample['observer_sample_ms'] = (time.monotonic() - tick) * 1000
            output.write(json.dumps(sample) + '\n')
            output.flush()
            # Capture a final snapshot after SIGTERM or the sampling deadline,
            # including runs whose preflight finishes within one interval.
            if stop.is_set() or time.monotonic() - started >= args.duration:
                break
            stop.wait(max(0.1, args.interval - (time.monotonic() - tick)))


if __name__ == '__main__':
    main()
