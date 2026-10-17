from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from ember_evaluator.protocol import evaluate, hello


def test_invalid_inputs_before_execution():
    for indices in ([], [0, 0], [True], [-1], [999]):
        with pytest.raises(ValueError):
            evaluate({'taskId':'relu', 'code':'def relu(x): return x', 'testIndices':indices})
    with pytest.raises(ValueError):
        evaluate({'taskId':'relu', 'code':'x', 'resourceBudget':{'maxCpuMs':999999}})


def test_stdio_real_four_workloads_and_errors():
    workloads = json.loads((ROOT / 'scripts/desktop/workload.json').read_text(encoding='utf-8'))
    requests = [{'id':'hello','method':'hello','params':None}]
    for index, item in enumerate(workloads):
        requests.append({'id':str(index),'method':'evaluate','params':{'requestId':str(index),**item}})
    requests.extend([
        {'id':'wrong','method':'evaluate','params':{'taskId':'relu','code':'def relu(x): return x + 100'}},
        {'id':'timeout','method':'evaluate','params':{'taskId':'relu','code':'while True: pass','resourceBudget':{'maxCpuMs':1000}}},
        {'id':'recovery','method':'evaluate','params':{'taskId':'relu','code':workloads[0]['code']}},
    ])
    result = subprocess.run([sys.executable, str(ROOT/'modules/evaluator/runner.py')],
        input=''.join(json.dumps(r)+'\n' for r in requests),text=True,encoding='utf-8',capture_output=True,
        # Windows spawn imports Torch for every evaluation; allow each request
        # its production 10-second budget plus process startup/teardown overhead.
        timeout=10 + 12 * (len(requests) - 1),cwd=ROOT)
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(replies)==len(requests)
    assert replies[0]['result']==hello()
    for reply in replies[1:5]:
        assert reply['result']['outcome']=='passed', reply
        assert reply['result']['passed']==reply['result']['total']>0
    assert replies[5]['result']['outcome']=='failed'
    assert replies[6]['result']['outcome']=='timeout'
    assert replies[7]['result']['outcome']=='passed'
