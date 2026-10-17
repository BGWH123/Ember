#!/usr/bin/env python3
"""Real download/install/upgrade/rollback integration exercise.
Requires a self-contained module produced by package_evaluator.py.
"""
from __future__ import annotations
import functools
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT=Path(__file__).resolve().parents[2]

def main():
    module=ROOT/'artifacts/desktop/module'
    signing=Ed25519PrivateKey.from_private_bytes(bytes.fromhex((ROOT/'artifacts/desktop/test-signing-key').read_text()))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',8765),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(module)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    log=(ROOT/'artifacts/desktop/module-smoke.log').open('w')
    try:
        with tempfile.TemporaryDirectory(prefix='ember-module-smoke-') as directory:
            env=os.environ.copy();env.pop('EMBER_DEV_PYTHON',None)
            env.update(EMBER_ROOT=str(ROOT),EMBER_DATA=directory,EMBER_EVALUATOR_PUBLIC_KEY=(module/'public-key.txt').read_text())
            process=subprocess.Popen([str(ROOT / ('target/release/ember-headless.exe' if os.name == 'nt' else 'target/release/ember-headless'))],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=True)
            try:
                ready=json.loads(process.stdout.readline());assert not ready['status']['available']
                def command(**body):
                    process.stdin.write(json.dumps(body)+'\n');process.stdin.flush();reply=json.loads(process.stdout.readline());return reply
                assert command(path='/api/problems')['ok']
                installed=command(command='install',url='http://127.0.0.1:8765/manifest.json');assert installed['ok'],installed
                assert installed['body']['activeVersion']=='0.1.0'
                code='def relu(x): return x.clamp(min=0)'
                result=command(path='/api/submit',payload={'taskId':'relu','code':code});assert result['body']['allPassed'],result
                # Publish the same compatible engine under a second independent artifact version.
                # Engine version is informational; compatibility is protocol+catalog, not shell semver.
                signed=json.loads((module/'manifest.json').read_text());payload=json.loads(signed['payload']);payload['version']='0.2.0'
                updated=json.dumps(payload,separators=(',',':'));(module/'upgrade.json').write_text(json.dumps({'payload':updated,'signature':signing.sign(updated.encode()).hex()}))
                reply=command(command='install',url='http://127.0.0.1:8765/upgrade.json');assert reply['ok'],reply
                assert reply['body']['activeVersion']=='0.2.0'
                reply=command(command='activate',version='0.1.0');assert reply['ok'] and reply['body']['activeVersion']=='0.1.0',reply
                # A forged manifest never replaces the working active module.
                forged=json.loads((module/'upgrade.json').read_text());forged['payload']=forged['payload'].replace('0.2.0','0.3.0');(module/'forged.json').write_text(json.dumps(forged))
                assert not command(command='install',url='http://127.0.0.1:8765/forged.json')['ok']
                assert command(command='status')['body']['activeVersion']=='0.1.0'
                # Verify the active local module still works after rollback.
                result=command(path='/api/submit',payload={'taskId':'relu','code':code});assert result['body']['allPassed'],result
                assert command(path='/api/progress')['body']['progress']['relu']['status']=='solved'
                print(json.dumps({'download':True,'standaloneRuntime':True,'upgrade':True,'rollback':True,'forgeryRejected':True,'historyRetained':True}))
                command(command='shutdown')
            finally:
                process.stdin.close();process.wait(timeout=10)
    finally:
        server.shutdown();server.server_close();log.close()


if __name__=='__main__':main()
