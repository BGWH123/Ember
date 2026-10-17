"""Render k6 phase summaries and scoped resource evidence without averaging P99s."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(directory: Path) -> str:
    summary = json.loads((directory / 'summary.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    metrics = summary['metrics']
    execution_path = directory / 'execution.json'
    exit_code = json.loads(execution_path.read_text())['exit_code'] if execution_path.exists() else None
    interrupted = exit_code is not None and exit_code not in (0, 99)

    def value(name: str, key: str = 'count', phase: str | None = None):
        metric = metrics.get(name + (f'{{phase:{phase}}}' if phase else ''), {})
        values = metric.get('values', {})
        if metric.get('type') == 'trend' and key != 'count' and not values.get('count'):
            return None
        return values.get(key)

    def display(item):
        return '未采集' if item is None else f'{item:.2f}'

    def count(name):
        return value(name) or 0

    failures = [f'{name}: {rule}' for name, metric in metrics.items() for rule, result in metric.get('thresholds', {}).items() if not result['ok']]
    lines = [
        '# Ember 压测报告', '',
        f"目标：{config['base']}；模式：{config['mode']}/{config['suite']}；配置：{config['profile']}；模拟会话：{config['users']}。",
        f"目标负载：{config['readRps']:.2f} 次读请求/秒，{config['jobsPerMinute']:.2f} 次评测提交/分钟；轮询间隔 {config['pollMs']}ms，客户端等待上限 {config['jobTimeout']}s。",
        '', f"预设 SLO：{'未完成（预检或执行失败）' if interrupted else '未通过' if failures else '通过'}；k6 退出码：{exit_code}。本轮只验证所列工作负载，线上容量还取决于真实题目、数据规模和硬件。", '',
        '## 吞吐与完整性', '',
        '| 提交尝试 | 准入 | 429 拒绝 | 准入异常 | 观察到终态 | 正确通过 | 客户端超时 | 轮询异常 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| ' + ' | '.join(str(int(count(name))) for name in ('jobs_offered', 'jobs_admitted', 'jobs_rejected', 'jobs_admission_errors', 'jobs_completed', 'jobs_correct', 'jobs_client_timeout', 'jobs_poll_errors')) + ' |', '',
        f"负载发生器未启动的计划迭代：{int(count('dropped_iterations'))}；准入后未记下客户端终态的迭代：{int(count('jobs_admitted') - count('jobs_completed') - count('jobs_client_timeout') - count('jobs_poll_errors'))}。",
        '客户端超时不代表服务端取消；请结合排空阶段的数据库状态。正确通过要求 allPassed=true、total>0 且无 error，防止快速报错制造高吞吐。', '',
        '## 分阶段延迟', '',
        '| 阶段 | 时长(s) | 读样本 | 读 P95/P99(ms) | 准入/拒绝 | 202 入队 P95(ms) | 排队 P95(ms) | 评测 P95(ms) | 完成数 | 正确且达标 RPS | E2E 样本 | E2E P95/P99(ms) |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for phase in config['phases']:
        name, seconds = phase['name'], phase['seconds']
        cells = [name, str(seconds), display(value('read_latency_ms', 'count', name)),
                 '/'.join(display(value('read_latency_ms', key, name)) for key in ('p(95)', 'p(99)')),
                 '/'.join(display(value(metric, 'count', name)) for metric in ('jobs_admitted', 'jobs_rejected')),
                 display(value('enqueue_accepted_ms', 'p(95)', name)), display(value('queue_wait_ms', 'p(95)', name)),
                 display(value('worker_total_ms', 'p(95)', name)), display(value('jobs_completed', 'count', name)),
                 display(value('jobs_within_slo', 'count', name) / seconds if value('jobs_within_slo', 'count', name) is not None else None),
                 display(value('job_e2e_ms', 'count', name)),
                 '/'.join(display(value('job_e2e_ms', key, name)) for key in ('p(95)', 'p(99)'))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '完成数按完成时刻分桶；E2E/排队/执行延迟按提交时刻分桶。E2E 仅覆盖观察到终态的任务，超时和丢弃必须一起看；少样本 P99 不宜外推。', '']
    api_lines = ['## 读接口延迟', '', '| 接口 | 样本数 | P95(ms) | P99(ms) |', '|---|---:|---:|---:|']
    for name in config.get('readNames', []):
        data = metrics.get(f'read_latency_ms{{name:{name}}}', {}).get('values', {})
        api_lines.append(f"| {name} | {display(data.get('count'))} | {display(data.get('p(95)') if data.get('count') else None)} | {display(data.get('p(99)') if data.get('count') else None)} |")
    if config.get('readNames'):
        lines += api_lines + ['']
    if config.get('burstUsers'):
        lines += ['## 同时提交', '',
                  f"第 {config['burstAt']} 秒额外启动 {config['burstUsers']} 个并发完整评测会话。背景开放到达率继续运行。",
                  '| 尝试 | 准入 | 正确完成 | E2E P95(ms) | E2E P99(ms) |', '|---:|---:|---:|---:|---:|',
                  '| ' + ' | '.join(display(value(name + '{scenario:burst}', key)) for name, key in (
                      ('jobs_offered', 'count'), ('jobs_admitted', 'count'), ('jobs_correct', 'count'),
                      ('job_e2e_ms', 'p(95)'), ('job_e2e_ms', 'p(99)'))) + ' |', '']
    lines += ['## 请求计时', '',
              '| 路径类型 | 样本 | 后端 P95(ms) | 分发/校验 P95 | 会话 P95 | 写锁 P95 | SQL P95 | COMMIT P95 | 业务线程 CPU P95 | Web 转发开销 P95 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('enqueue', 'poll', 'progress', 'history'):
        suffix = f'{{name:{name}}}'
        cpu_point = {'enqueue': 'admission_cpu', 'poll': 'result_read_cpu', 'progress': 'progress_read_cpu', 'history': 'history_read_cpu'}[name]
        lines.append('| ' + ' | '.join([name, display(value('api_backend_ms' + suffix, 'count'))]
                     + [display(value('api_' + point + '_ms' + suffix, 'p(95)'))
                        for point in ('backend', 'dispatch', 'session', 'db_lock', 'db_sql', 'db_commit', cpu_point, 'proxy_overhead')]) + ' |')
    lines += ['', '来自 Server-Timing；后端、会话、SQL 等 span 有包含关系，不能把这些 P95 相加。分发含 HTTP 参数校验和线程池等待；Web 转发开销不含上游 fetch 等待。', '',
              '## 评测计时', '', '| 计时点 | 样本 | P50(ms) | P95(ms) | P99(ms) |', '|---|---:|---:|---:|---:|']
    for point in ('queue_wait_ms', 'claim_ms', 'worker_startup_ms', 'runtime_import_ms', 'setup_ms', 'parse_compile_ms',
                  'user_code_ms', 'test_execution_ms', 'result_encode_ms', 'result_transfer_ms', 'result_decode_ms',
                  'process_reap_ms', 'worker_total_ms', 'persist_db_lock_ms', 'persist_db_commit_ms', 'persist_ms', 'server_e2e_ms', 'job_e2e_ms'):
        lines.append('| ' + ' | '.join([point] + [display(value(point, key)) for key in ('count', 'med', 'p(95)', 'p(99)')]) + ' |')
    lines += ['', 'worker_total_ms 统一计父进程启动到回收；persist_ms 包含主结果事务 COMMIT，排除随后保存计时元数据的第二次写入。server_e2e_ms 到服务端结果提交，job_e2e_ms 到客户端轮询看到终态。', '']
    lines += ['## 资源证据', '']
    samples = []
    observer = directory / 'resources.jsonl'
    if observer.exists():
        samples = [json.loads(line) for line in observer.read_text().splitlines() if line.strip()]
    dbs, cgroups, errors = [], {}, []
    for sample in samples:
        if sample.get('database'):
            dbs.append(sample['database'])
        for grading in sample.get('docker', {}).get('grading', []):
            dbs.append(grading['database'])
            cgroups.setdefault(grading['container'], []).append(grading['cgroup'])
        errors += sample.get('docker', {}).get('errors', [])
        for key in ('database_error', 'docker_error'):
            if sample.get(key):
                errors.append(sample[key])
    if dbs:
        last = dbs[-1]
        backlog = sum(last['states'].get(state, 0) for state in ('queued', 'leased', 'running'))
        lines += [f"采样中最大排队数：{max(d['states'].get('queued', 0) for d in dbs)}；最长队龄：{max(d['oldest_queued_seconds'] for d in dbs):.2f}s；末次采样仍在途：{backlog}。",
                  f"采样窗口服务端结果分布：{json.dumps(last['outcomes_since_start'], ensure_ascii=False)}；错误码分布：{json.dumps(last['errors_since_start'], ensure_ascii=False)}。"]
        if backlog:
            lines.append('排空窗口结束仍有任务：本轮尚未排空，容量结论需要延长观察或降低负载后复测。')
    else:
        lines.append('未采集服务端数据库；不能由客户端延迟独立确定数据库锁、真实队列长度或剩余任务。')
    for container, snapshots in cgroups.items():
        usable = [c['cpu.stat'] for c in snapshots if c.get('cpu.stat')]
        if len(usable) < 2:
            continue
        if any(b['usage_usec'] < a['usage_usec'] for a, b in zip(usable, usable[1:])):
            lines.append(f'{container} 的 CPU 计数器重置，可能重启；本轮不推算单位任务成本。')
            continue
        first, last = usable[0], usable[-1]
        cpu_seconds = (last['usage_usec'] - first['usage_usec']) / 1_000_000
        throttled = (last.get('throttled_usec', 0) - first.get('throttled_usec', 0)) / 1_000_000
        lines.append(f'{container} cgroup 累计 CPU：{cpu_seconds:.2f}s；节流累计时间：{throttled:.2f}s（多线程节流时间不等于墙钟时间）。')
        if dbs and len(cgroups) == 1:
            finished = dbs[-1]['finished_since_start'] - dbs[0]['finished_since_start']
            if finished > 0 and not backlog:
                lines.append(f'同一采样窗口 CPU 秒/服务端终态任务：{cpu_seconds / finished:.4f}；分母包含失败任务，应同时看正确通过比例。')
    if errors:
        lines.append(f'资源采样有 {len(errors)} 个错误，详见 resources.jsonl；缺失指标不表示零负载。')
    lines += ['', '## SLO 失败项', ''] + (['- ' + failure for failure in failures] or ['无。'])
    evidence = [name for name in ('summary.json', 'config.json', 'samples.json.gz', 'resources.jsonl', 'context.json', 'execution.json', 'dashboard.html', 'worktree.patch', 'backend.log') if (directory / name).exists()]
    lines += ['', '## 下一轮决策', '',
              '- 排队增长而入队稳定：先比较独立评测 worker 的并行度、任务隔离与可靠队列。',
              '- 入队/轮询/进度同时变慢且 CPU 未满：检查 SQLite 写锁、会话 last_seen 写入及事务范围。',
              '- Web 与直连 grading 的入队/轮询差异大：检查转发、连接复用、Node 事件循环。两种模式的读接口不同，不能直接比较读 QPS。',
              '- 容器 cpu.stat 节流增长：先核对 CPU 配额与 PyTorch 线程数；换语言的收益尚未证实。',
              '- dropped_iterations>0：当前结果受发生器/VU 预算影响，增加发生器资源后按相同负载复测。', '',
              '原始证据：' + '、'.join(evidence) + '。正式选型需要相同硬件、数据、负载和多轮 A/B；本报告不自动作因果判断。', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory / 'report.md'
    output.write_text(render(args.directory))
    print(f'压测报告：{output}')


if __name__ == '__main__':
    main()
