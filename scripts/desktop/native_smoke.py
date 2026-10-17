#!/usr/bin/env python3
"""Linux Tauri/WebKit smoke test using tauri-driver in an isolated GUI container.
Build with check-native.sh, package a module, and install `cargo install tauri-driver --locked` first.
This validates real IPC and rendering; it is not a release GUI performance benchmark.
"""
from __future__ import annotations
import base64
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[2]
BASE='http://127.0.0.1:4444'


def http(method,path,payload=None):
    request=urllib.request.Request(BASE+path,data=json.dumps(payload).encode() if payload is not None else None,method=method,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=70) as response:
            value=json.load(response)['value']
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    if isinstance(value,dict) and value.get('error'):
        raise RuntimeError(str(value))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-package',action='store_true');parser.add_argument('--measure-idle',action='store_true');args=parser.parse_args()
    runtime=ROOT/'artifacts/desktop/native-evaluator'
    if not args.release_package and not runtime.exists():
        runtime.mkdir()
        with zipfile.ZipFile(ROOT/'artifacts/desktop/module/evaluator.zip') as archive:
            archive.extractall(runtime)
            for info in archive.infolist():
                path=runtime/info.filename
                if path.is_file() and (info.external_attr>>16)&0o111:path.chmod(0o755)
    application=ROOT/'artifacts/desktop/native-target/debug/ember-desktop'
    if args.release_package:
        archive=next((ROOT/'artifacts/desktop/native-target/release/bundle/deb').glob('*.deb'))
        installed=ROOT/f'artifacts/desktop/release-install-{os.getpid()}'
        subprocess.run(['dpkg-deb','-x',str(archive),str(installed)],check=True)
        application=next((installed/'usr/bin').iterdir())
    name=f'ember-native-smoke-{os.getpid()}'
    log=(ROOT/('artifacts/desktop/native-release-smoke.log' if args.release_package else 'artifacts/desktop/native-smoke.log')).open('w')
    session=None;driver=None
    try:
        development_env=[] if args.release_package else ['-e','EMBER_DEV_PYTHON=/workspace/artifacts/desktop/native-evaluator/runtime/bin/python3']
        command=['docker','run','--rm','--name',name,'--network','host','--user',f'{os.getuid()}:{os.getgid()}',
            '-v',f'{ROOT}:/workspace','-v',f'{Path.home()}/.cargo/bin/tauri-driver:/usr/local/bin/tauri-driver:ro',
            '-e','HOME=/tmp/ember-home','-e','XDG_RUNTIME_DIR=/tmp/ember-runtime',
            '-e','WEBKIT_DISABLE_DMABUF_RENDERER=1','-e','LIBGL_ALWAYS_SOFTWARE=1',
            *development_env,
            'ember-desktop-build','bash','-c','mkdir -p "$HOME" "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR" && dbus-run-session -- xvfb-run -a tauri-driver --native-driver /usr/bin/WebKitWebDriver']
        driver=subprocess.Popen(command,stdout=log,stderr=log)
        for _ in range(100):
            try:
                http('GET','/status');break
            except OSError:time.sleep(.1)
        value=http('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'wry','tauri:options':{'application':'/workspace/'+application.relative_to(ROOT).as_posix()}}}})
        session=value['sessionId'];prefix='/session/'+session
        http('POST',prefix+'/timeouts',{'script':60000,'implicit':1000,'pageLoad':30000})
        def script(code):return http('POST',prefix+'/execute/sync',{'script':code,'args':[]})
        def invoke(command,args):
            return http('POST',prefix+'/execute/async',{'script':"const done=arguments[arguments.length-1]; window.__TAURI_INTERNALS__.invoke(arguments[0],arguments[1]).then(done).catch(e=>done({error:String(e)}));",'args':[command,args]})
        def wait(code):
            for _ in range(200):
                result=script(code)
                if result:return result
                time.sleep(.05)
            raise RuntimeError('Native UI condition timed out: '+code)
        wait('return Boolean(window.__TAURI_INTERNALS__ && document.querySelector("main"));')
        problems=invoke('request',{'path':'/api/problems','payload':None});assert problems['total']>0
        container_pid=int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',name],text=True))
        app_process=next(p for p in psutil.Process(container_pid).children(recursive=True) if p.name()=='ember-desktop')
        def memory():
            rss=pss=0;python_processes=0;count=0
            for process in [app_process,*app_process.children(recursive=True)]:
                try:
                    rss+=process.memory_info().rss
                    pss+=getattr(process.memory_full_info(),'pss',0)
                    python_processes+=int(process.name().startswith('python'))
                    count+=1
                except psutil.NoSuchProcess:pass
            return {'rssBytes':rss,'pssBytes':pss,'processes':count,'pythonProcesses':python_processes}
        initial_memory=memory()
        status=invoke('evaluator_status',{})
        if args.release_package:
            assert status['mode']=='local' and not status['available'],status
            status=invoke('import_evaluator',{'manifest':'/workspace/artifacts/desktop/module/manifest.json','archive':'/workspace/artifacts/desktop/module/evaluator.zip'})
            assert status['available'] and status['activeVersion']=='0.1.0',status
        else:
            assert status['mode']=='development'
        code='def relu(x):\n    return x.clamp(min=0)\n'
        invoke('save_draft',{'id':'relu','code':code})
        script("localStorage.setItem('locale','en');window.location.hash='#/problems/relu';")
        wait('return Boolean(document.querySelector(".monaco-editor"));')
        wait("return document.body.innerText.includes('clamp');")
        for label in ['Run','Submit']:
            script("const button=[...document.querySelectorAll('button')].find(b=>b.innerText.trim().startsWith("+json.dumps(label)+")); if(!button)throw Error('button missing');button.click();return true;")
            wait("return [...document.querySelectorAll('button')].some(b=>b.innerText.trim().startsWith("+json.dumps(label)+")&&!b.disabled);")
        progress=invoke('request',{'path':'/api/progress','payload':None});assert progress['progress']['relu']['status']=='solved',progress
        history=invoke('request',{'path':'/api/submissions/relu','payload':None});assert len(history)==2,history
        screenshot=http('GET',prefix+'/screenshot');(ROOT/('artifacts/desktop/native-release-workspace.png' if args.release_package else 'artifacts/desktop/native-workspace.png')).write_bytes(base64.b64decode(screenshot))
        report={'nativeBuild':True,'webKitRender':True,'realTauriIPC':True,'editorLoadedOffline':True,'sampleAndFullEvaluation':True,'sqliteProgressAndHistory':True,'debugBuild':not args.release_package,'embeddedFrontend':True,'installedFromDeb':args.release_package}
        report['initialMemory']=initial_memory
        report['afterEvaluationMemory']=memory()
        if args.measure_idle:
            print('Waiting for idle evaluator reclamation',flush=True)
            deadline=time.monotonic()+90
            observations=[]
            while True:
                sample=memory();observations.append(sample)
                if sample['pythonProcesses']==0:break
                if time.monotonic()>deadline:raise RuntimeError('Idle evaluator was not reclaimed within 90 seconds')
                time.sleep(1)
            report['afterIdleMemory']=sample;report['idleMemorySamples']=observations
        (ROOT/('artifacts/desktop/native-release-smoke.json' if args.release_package else 'artifacts/desktop/native-smoke.json')).write_text(json.dumps(report,indent=2));print(json.dumps({key:value for key,value in report.items() if key!='idleMemorySamples'}))
    finally:
        if session:
            try:http('DELETE','/session/'+session)
            except Exception:pass
        if driver:
            subprocess.run(['docker','stop','--time','3',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            driver.wait(timeout=10)
        log.close()


if __name__=='__main__':main()
