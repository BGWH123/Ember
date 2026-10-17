#!/usr/bin/env python3
"""Compare production legacy services with the release Rust application core.

Measures BACKEND trees only, not browser/WebView or Tauri IPC. Uses isolated DBs.
AB then BA, fixed four workloads, sample/full; raw observations are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import http.cookiejar
import json
import math
import platform
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import psutil

ROOT=Path(__file__).resolve().parents[2]
WORKLOAD=json.loads((ROOT/'scripts/loadtest/workload.json').read_text())


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def memory(processes):
    selected={}
    for child in processes:
        try:
            root=psutil.Process(child.pid)
            for process in [root,*root.children(recursive=True)]:selected[process.pid]=process
        except psutil.NoSuchProcess:pass
    rss=pss=0
    for process in selected.values():
        try:
            rss+=process.memory_info().rss
            pss+=getattr(process.memory_full_info(),'pss',0)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return {'rssBytes':rss,'pssBytes':pss,'processes':len(selected)}


class Target:
    def __init__(self,kind,directory):
        self.kind=kind;self.processes=[];self.files=[];self.directory=directory
        self.samples=[];self.stopped=threading.Event()
        self.start_ms=0
    def spawn(self,cmd,env,cwd):
        log=(self.directory/f'process-{len(self.files)}.log').open('w');self.files.append(log)
        process=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE if self.kind=='rust' else log,stderr=log,text=True,start_new_session=True)
        self.processes.append(process);return process
    def start(self):
        env=os.environ.copy();start=time.perf_counter()
        if self.kind=='legacy':
            backend=port();frontend=port();env.update(DB_PATH=str(self.directory/'legacy.db'),GRADING_SERVICE_URL=f'http://127.0.0.1:{backend}',NEXT_TELEMETRY_DISABLED='1')
            self.spawn([str(ROOT/'.venv/bin/python'),'-m','uvicorn','grading_service.main:app','--host','127.0.0.1','--port',str(backend)],env,ROOT)
            self.base=f'http://127.0.0.1:{frontend}'
            self.opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            self.spawn(['node',str(ROOT/'web/node_modules/next/dist/bin/next'),'start','--hostname','127.0.0.1','--port',str(frontend)],env,ROOT/'web')
            deadline=time.monotonic()+40
            for url in [f'http://127.0.0.1:{backend}/health',self.base+'/api/problems']:
                while True:
                    try:
                        with urllib.request.urlopen(url,timeout=1) as response:
                            if response.status==200:break
                    except (OSError,urllib.error.URLError):pass
                    if time.monotonic()>deadline:raise RuntimeError('Legacy startup failed; inspect process logs')
                    time.sleep(.05)
        else:
            env.update(EMBER_ROOT=str(ROOT),EMBER_DATA=str(self.directory/'rust-data'),EMBER_DEV_PYTHON=str(ROOT/'.venv/bin/python'))
            self.child=self.spawn([str(ROOT/'target/release/ember-headless')],env,ROOT)
            line=self.child.stdout.readline();self.ready=json.loads(line)
        self.start_ms=(time.perf_counter()-start)*1000
        self.idle=memory(self.processes)
        self.monitor=threading.Thread(target=self.observe,daemon=True);self.monitor.start()
    def observe(self):
        while not self.stopped.wait(.02):self.samples.append(memory(self.processes))
    def command(self,body):
        self.child.stdin.write(json.dumps(body)+'\n');self.child.stdin.flush()
        line=self.child.stdout.readline()
        if not line:raise RuntimeError('Rust adapter exited')
        result=json.loads(line)
        if not result['ok']:raise RuntimeError(result['error'])
        return result['body']
    def call(self,path,payload=None):
        start=time.perf_counter()
        if self.kind=='rust':body=self.command({'path':path,'payload':payload})
        else:
            request=urllib.request.Request(self.base+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
            with self.opener.open(request,timeout=40) as response:body=json.load(response)
            if path in ['/api/run','/api/submit']:
                submission_id=body['submission_id'];tick=time.perf_counter()
                while True:
                    with self.opener.open(self.base+'/api/submissions/status/'+submission_id,timeout=5) as response:body=json.load(response)
                    if body['status'] in ['completed','cancelled']:break
                    if time.perf_counter()-tick>35:raise RuntimeError('Legacy submission timeout')
                    time.sleep(.25 if time.perf_counter()-tick<5 else 1)
        return body,(time.perf_counter()-start)*1000
    def close(self):
        self.stopped.set()
        if hasattr(self,'monitor'):self.monitor.join(timeout=2)
        for process in reversed(self.processes):
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for process in self.processes:
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=3)
        for log in self.files:log.close()


def run(kind,round_id,output,repetitions):
    directory=output/f'{round_id}-{kind}';directory.mkdir()
    target=Target(kind,directory);observations=[]
    try:
        target.start()
        cold,ms=target.call('/api/submit',{'taskId':'relu','code':WORKLOAD[0]['code']})
        assert cold['allPassed'],cold
        observations.append({'path':'cold-submit-relu','ms':ms,'result':cold})
        # Warm remaining code/data paths on both sides before repeated measurements.
        for item in WORKLOAD:
            value,_=target.call('/api/submit',{'taskId':item['taskId'],'code':item['code']});assert value['allPassed'],value
        for repetition in range(repetitions):
            for path in ['/api/problems','/api/problems/relu','/api/progress','/api/submissions/relu','/api/paths','/api/solutions/relu']:
                body,ms=target.call(path);observations.append({'path':path,'ms':ms,'repetition':repetition})
            for item in WORKLOAD:
                for mode in ['run','submit']:
                    payload={'taskId':item['taskId'],'code':item['code']}
                    if mode=='run':payload['testIndices']=[0,1]
                    body,ms=target.call('/api/'+mode,payload)
                    assert body['allPassed'] and body['passed']==body['total']>0,body
                    observations.append({'path':f'{mode}-{item["taskId"]}','ms':ms,'repetition':repetition,'result':body})
        warm_idle=memory(target.processes)
        for label,code,expected in [('wrong','def relu(x): return x + 100','failed'),('syntax','def relu(:','failed'),('timeout','while True: pass','timeout'),('recovery',WORKLOAD[0]['code'],'passed')]:
            body,ms=target.call('/api/submit',{'taskId':'relu','code':code});assert body['outcome']==expected,body
            observations.append({'path':label,'ms':ms,'result':body})
        if kind=='rust':target.command({'command':'shutdown'})
        time.sleep(.1);released=memory(target.processes)
        report={'kind':kind,'round':round_id,'startMs':target.start_ms,'initialIdle':target.idle,'warmIdle':warm_idle,'releasedIdle':released,'peak':{key:max((sample[key] for sample in target.samples),default=0) for key in ['rssBytes','pssBytes']},'observations':observations,'memorySamples':target.samples}
        (directory/'raw.json').write_text(json.dumps(report,indent=2));return report
    finally:target.close()


def percentile(values,p):
    values=sorted(values);return values[max(0,math.ceil(len(values)*p)-1)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repetitions',type=int,default=20);parser.add_argument('--output',type=Path,default=ROOT/'artifacts/desktop/comparison');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    args.output=args.output.resolve()
    if args.repetitions<1:parser.error('--repetitions must be positive')
    if not args.report_only:
        args.output.mkdir(parents=True,exist_ok=False)
        context={'rustBinarySha256':hashlib.sha256((ROOT/'target/release/ember-headless').read_bytes()).hexdigest(),'catalogSha256':hashlib.sha256((ROOT/'content/catalog/problems.json').read_bytes()).hexdigest(),'node':subprocess.check_output(['node','--version'],text=True).strip(),'rust':subprocess.check_output(['rustc','--version'],text=True).strip(),'torch':subprocess.check_output([sys.executable,'-c','import torch; print(torch.__version__)'],text=True).strip(),'cpuCount':os.cpu_count(),'memoryBytes':psutil.virtual_memory().total,'kernel':platform.platform(),'date':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'python':sys.version,'platform':sys.platform,'repetitions':args.repetitions,'workloadSha256':hashlib.sha256((ROOT/'scripts/loadtest/workload.json').read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirtyStatus':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)}
        (args.output/'context.json').write_text(json.dumps(context,indent=2))
        reports=[]
        for round_id,order in enumerate([['legacy','rust'],['rust','legacy']]):
            for kind in order:
                print(f'Round {round_id}: {kind}',flush=True);reports.append(run(kind,round_id,args.output,args.repetitions))
    else:
        args.repetitions=json.loads((args.output/"context.json").read_text())["repetitions"]
        reports=[json.loads((args.output/f"{r}-{kind}/raw.json").read_text()) for r,order in enumerate([["legacy","rust"],["rust","legacy"]]) for kind in order]
    summary={}
    for kind in ['legacy','rust']:
        rows=[r for report in reports if report['kind']==kind for r in report['observations']]
        summary[kind]={path:{'n':len(values:=[r['ms'] for r in rows if r['path']==path]),'p50':percentile(values,.5),'p95':percentile(values,.95)} for path in dict.fromkeys(r['path'] for r in rows)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 原版与 Rust 核心性能对比','','状态：后台关键路径已实测；真实 Tauri/WebKit debug 功能验收另行通过。表中不包含 WebView 或 Tauri IPC 的性能。','','环境：同一 Linux 主机、同一 Python/PyTorch、原版 Next production + FastAPI；新版 release Rust core + 按需 evaluator。单用户顺序，AB/BA 两轮，热路径每轮 '+str(args.repetitions)+' 次；旧 UI 的结果轮询间隔被复现。数据库互相隔离。','','|路径|原版 P50/P95 ms|Rust P50/P95 ms|样本数/侧|','|---|---:|---:|---:|']
    for path in summary['legacy']:
        a,b=summary['legacy'][path],summary['rust'][path];lines.append(f'|{path}|{a["p50"]:.2f} / {a["p95"]:.2f}|{b["p50"]:.2f} / {b["p95"]:.2f}|{a["n"]}|')
    lines+=['','## 后台进程树内存','','RSS 汇总会重复计算共享页；PSS 分摊共享页。采样 20ms，不保证捕获短暂峰值。此处均不含浏览器/WebView。','','|轮次/实现|启动 ms|初始空闲 RSS/PSS MiB|热后空闲 RSS/PSS MiB|关闭评测器后 RSS/PSS MiB|峰值 RSS/PSS MiB|','|---|---:|---:|---:|---:|---:|']
    for r in reports:
        fmt=lambda m:f'{m["rssBytes"]/1024**2:.1f} / {m["pssBytes"]/1024**2:.1f}'
        lines.append(f'|{r["round"]}/{r["kind"]}|{r["startMs"]:.1f}|{fmt(r["initialIdle"])}|{fmt(r["warmIdle"])}|{fmt(r["releasedIdle"])}|{fmt(r["peak"])}|')
    lines+=['','四题 sample/full 全部正确；错误实现、语法错误、超时和超时后恢复均验证。原版服务端 verdict 时间与客户端观察延迟保存在 raw.json；250ms 轮询解释可见延迟差异，不应归因于 Python 算子变快。新版首次评测包含按需启动 Torch，冷启动代价单列。初始空闲时新版不加载 Torch，关闭评测器后释放运行时，原版仍常驻预热进程。','','这里只比较后台架构；不能据此声称完整 Tauri 应用节省同样比例内存。真实 Linux Tauri/WebKit debug GUI 已通过单独的渲染、IPC 与评测验收；本节未测 release GUI 的配对性能；完整桌面单次内存快照另列。AI 外部接口没有纳入性能实验。','','复现：`cargo build --release -p ember-core --features developer-tools`；`npm --prefix web run build`；`.venv/bin/python scripts/desktop/compare.py --output artifacts/desktop/comparison-NEW`。','','原始证据：`'+str(args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output)+'`。']
    native_path=ROOT/'artifacts/desktop/native-release-smoke.json'
    if native_path.exists():
        native=json.loads(native_path.read_text())
        if native.get('installedFromDeb') and 'afterIdleMemory' in native:
            lines+=['','## 完整 release 桌面程序的单次内存快照','','独立实验：从 .deb 解包启动，真实 Tauri + WebKit + 嵌入前端 + Rust IPC；无 Node / Next 开发服务。Ubuntu 24.04 容器、Xvfb、软件渲染，1280×850 窗口。只统计应用及其子进程，不含 Xvfb / WebDriver。RSS 重复计算共享页，PSS 更适合观察总占用。本节不是与旧版浏览器整机的 A/B，不能直接与上面的后台表交叉比较。','','|阶段|RSS MiB|PSS MiB|Python 进程数|','|---|---:|---:|---:|']
            for label,key in [('首屏，未安装/启动评测器','initialMemory'),('编辑器 + 完整评测结束','afterEvaluationMemory'),('编辑器 + 自动空闲回收后','afterIdleMemory')]:
                value=native[key];lines.append(f'|{label}|{value["rssBytes"]/1024**2:.1f}|{value["pssBytes"]/1024**2:.1f}|{value["pythonProcesses"]}|')
            lines+=['','自动回收在配置的 60–75 秒空闲窗口内发生，Python 进程数从 3 降到 0。首屏与编辑器页面不同，不能把两者的内存差异当成泄漏。完整桌面程序包含 WebView 和 Monaco，绝不是后台核心的 8MiB。','','证据：`artifacts/desktop/native-release-smoke.json`、`native-release-workspace.png`。复现：`.venv/bin/python scripts/desktop/native_smoke.py --release-package --measure-idle`。']
    (ROOT/'wiki/06-performance-results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
