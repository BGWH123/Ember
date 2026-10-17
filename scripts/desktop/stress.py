#!/usr/bin/env python3
"""Bounded, open-loop core stress run; raw rows and process-tree samples stay local."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', default='target/release/ember-stress')
    parser.add_argument('--mode', choices=['catalog', 'history', 'mixed', 'evaluate', 'fault'], default='mixed')
    parser.add_argument('--rate', type=int, default=500)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--history', type=int, default=10000)
    parser.add_argument('--arrival', choices=['constant', 'burst'], default='constant', help='burst delivers each second\'s arrivals together')
    parser.add_argument('--output', required=True)
    parser.add_argument('--profiler', choices=['none', 'nsys', 'cpu', 'heap'], default='none')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary).resolve()
    env = dict(os.environ, EMBER_ROOT=str(ROOT), EMBER_DEV_PYTHON=str(ROOT / '.venv/bin/python'),
               STRESS_MODE=args.mode, STRESS_ARRIVAL=args.arrival, STRESS_RATE=str(args.rate), STRESS_SECONDS=str(args.seconds), STRESS_HISTORY=str(args.history))
    command = [str(binary)]
    profile_env = {}
    libraries = ROOT / 'artifacts/desktop/stress/tools/root/usr/lib/x86_64-linux-gnu'
    if args.profiler == 'nsys':
        command = ['nsys', 'profile', '--trace=osrt', '--sample=process-tree', '--cpuctxsw=process-tree', '--backtrace=fp',
                   '--force-overwrite=true', '--output=' + str(output / 'timeline'), *command]
    elif args.profiler == 'cpu':
        profile_env.update(LD_PRELOAD=str(libraries / 'libprofiler.so.0'), CPUPROFILE=str(output / 'cpu.prof'))
    elif args.profiler == 'heap':
        profile_env.update(LD_PRELOAD=str(libraries / 'libtcmalloc.so.4'), HEAPPROFILE=str(output / 'heap'),
                   HEAP_PROFILE_ALLOCATION_INTERVAL='67108864')
    if profile_env:
        command = ['/usr/bin/env', *(f'{key}={value}' for key, value in profile_env.items()), *command]
    # GNU time follows wait/reap accounting. Forkserver grandchildren killed on
    # shutdown may not propagate usage: evaluator totals must be treated as a lower bound.
    command = ['/usr/bin/time', '-f', '%U %S', '-o', str(output / 'cpu-seconds.txt'), *command]
    samples = []
    with tempfile.TemporaryDirectory(prefix='ember-stress-') as data:
        env['EMBER_DATA'] = data
        with (output / 'rows.jsonl').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            start = time.monotonic()
            try:
                while process.poll() is None:
                    sample = {'t': time.monotonic() - start, 'rss': 0, 'pss': 0, 'cpu': 0, 'processes': 0}
                    try:
                        parent = psutil.Process(process.pid)
                        for child in [parent, *parent.children(recursive=True)]:
                            try:
                                memory = child.memory_full_info()
                                sample['rss'] += memory.rss
                                sample['pss'] += memory.pss
                                cpu = child.cpu_times()
                                sample['cpu'] += cpu.user + cpu.system
                                sample['processes'] += 1
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    samples.append(sample)
                    if sample['t'] > args.seconds + 120:
                        raise TimeoutError('Stress process did not drain within watchdog budget')
                    time.sleep(.2)
            finally:
                if process.poll() is None:
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        descendants = []
                    os.killpg(process.pid, signal.SIGTERM)
                    # Evaluators deliberately have their own process group.
                    for child in descendants:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(f'Stress exited {process.returncode}; see {output / "stderr.log"}')
    rows = [json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines() if line.startswith('{')]
    summary = rows[-1]
    rows = [row for row in rows if 'path' in row]
    counts = collections.Counter(row['outcome'] for row in rows)
    assert summary['offered'] == len(rows) + summary['generatorDropped'], 'Unaccounted arrivals'
    errors = [outcome for outcome in counts if outcome not in ('ok', 'passed', 'busy', 'timeout')]
    assert not errors, f'Unexpected results: {errors}'
    if args.mode == 'fault':
        timeouts = [row for row in rows if row['outcome'] == 'timeout']
        assert len(timeouts) == 1, f'Expected one timeout, got {len(timeouts)}'
        assert any(row['outcome'] == 'passed' and row['i']/args.rate > timeouts[0]['ms']/1000 for row in rows), 'No successful evaluation after timeout'
    groups = {}
    for path in sorted({r['path'] for r in rows}):
        values = [r['ms'] for r in rows if r['path'] == path and r['outcome'] in ('ok', 'passed')]
        groups[path] = {'successes': len(values), 'p50Ms': percentile(values, .5), 'p95Ms': percentile(values, .95), 'p99Ms': percentile(values, .99)}
    summary.update(counts=dict(counts), completed=len(rows), admitted=len(rows) - counts['busy'], rejected=counts['busy'],
                   retries=0, cancelled=0, paths=groups, binarySha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   environment={'platform': platform.platform(), 'profiler': args.profiler, 'command': command},
                   dispatchLagP99Ms=percentile([r['dispatchLagMs'] for r in rows], .99),
                   outcomes={outcome: {'count': count, 'p99Ms': percentile([r['ms'] for r in rows if r['outcome'] == outcome], .99)} for outcome, count in counts.items()},
                   peakPssMiB=max((s['pss'] for s in samples), default=0)/2**20,
                   pssWindowsMiB=[percentile([s['pss']/2**20 for s in samples if a <= s['t'] < a + 10], .5) for a in range(0, args.seconds, 10)])
    user_cpu, system_cpu = map(float, (output / 'cpu-seconds.txt').read_text().split())
    summary['cpuSecondsIncludingFixtureAndWarmup'] = user_cpu + system_cpu
    summary['cpuMsPerCompletedIncludingFixtureAndWarmup'] = (user_cpu + system_cpu) * 1000 / len(rows) if rows else None
    summary['cpuAccountingNote'] = 'GNU time reaped-process accounting; complete for core-only modes, lower bound for evaluator process trees; includes fixture/warmup/driver.'
    # CPU sums from live process trees are snapshots, not cumulative totals across departed workers.
    (output / 'resources.json').write_text(json.dumps(samples))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
