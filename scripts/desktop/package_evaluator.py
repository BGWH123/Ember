#!/usr/bin/env python3
"""Build a self-contained evaluator ZIP and Ed25519-signed release manifest.

Run with the release venv's Python. --key is a 32-byte hex private-key file;
keys are generated explicitly with --generate-key and are never stored in Git.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--generate-key', action='store_true')
    parser.add_argument('--version', default='0.1.0')
    parser.add_argument('--url', default='http://127.0.0.1:8765/evaluator.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/desktop/module')
    args = parser.parse_args()
    if args.generate_key:
        args.key.parent.mkdir(parents=True, exist_ok=True)
        key = Ed25519PrivateKey.generate()
        with args.key.open('x') as file:
            file.write(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex())
        args.key.chmod(0o600)
    else:
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(args.key.read_text().strip()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'public-key.txt').write_text(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex())
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Build from a dedicated release venv, not the system Python')
    sys.path.insert(0,str(ROOT))
    from torch_judge.tasks import TASKS
    catalog=json.loads((ROOT/'content/catalog/problems.json').read_text())
    if {p['id'] for p in catalog['problems']} != set(TASKS):
        raise SystemExit('Task IDs differ; run export_catalog.py')
    for problem in catalog['problems']:
        task=TASKS[problem['id']]
        normalize=lambda tests:[(t['name'],ast.dump(ast.parse(t['code'].replace('{fn}',task['function_name'])))) for t in tests]
        if problem['functionName']!=task['function_name'] or normalize(problem['tests'])!=normalize(task['tests']):
            raise SystemExit('Catalog tests differ; run export_catalog.py')
    catalog_version = hashlib.sha256((ROOT / 'content/catalog/problems.json').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='ember-module-') as temporary:
        stage = Path(temporary)
        # A venv alone is not portable: include the base interpreter and standard library.
        shutil.copytree(sys.base_prefix, stage / 'runtime', symlinks=False,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'test', 'tests'))
        version = f'python{sys.version_info.major}.{sys.version_info.minor}'
        if os.name == 'nt':
            site_source = Path(sys.prefix) / 'Lib/site-packages'
            site_destination = stage / 'runtime/Lib/site-packages'
            entrypoint = 'runtime/python.exe'
        else:
            site_source = Path(sys.prefix) / 'lib' / version / 'site-packages'
            site_destination = stage / 'runtime/lib' / version / 'site-packages'
            entrypoint = 'runtime/bin/python3'
        shutil.copytree(site_source, site_destination, dirs_exist_ok=True, symlinks=False,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'modules/evaluator/ember_evaluator', site_destination / 'ember_evaluator',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT / 'torch_judge', site_destination / 'torch_judge',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'modules/evaluator/runner.py', stage / 'runner.py')
        (site_destination / 'catalog-version.txt').write_text(catalog_version)
        # Ensure engine version follows independently published module version.
        protocol = site_destination / 'ember_evaluator/protocol.py'
        protocol.write_text(protocol.read_text().replace('VERSION = "0.1.0"', f'VERSION = {args.version!r}', 1))
        archive = args.output / 'evaluator.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
            for file in sorted(stage.rglob('*')):
                if file.is_file():
                    output.write(file, file.relative_to(stage).as_posix())
    machine = {'AMD64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine(), platform.machine())
    target_os = {'Darwin': 'macos', 'Windows': 'windows', 'Linux': 'linux'}[platform.system()]
    with archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    payload = json.dumps({'version': args.version, 'target': f'{machine}-{target_os}',
        'protocolVersion': 1, 'catalogVersion': catalog_version, 'url': args.url,
        'sha256': digest, 'size': archive.stat().st_size, 'entrypoint': entrypoint,
        'args': ['-I', 'runner.py']}, separators=(',', ':'))
    manifest = {'payload': payload, 'signature': key.sign(payload.encode()).hex()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(args.output / 'manifest.json'), 'archiveBytes': archive.stat().st_size,
                      'publicKeyFile': str(args.output / 'public-key.txt')}))


if __name__ == '__main__':
    main()
