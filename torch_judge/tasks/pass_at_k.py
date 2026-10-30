"""Unbiased pass@k estimator task."""

TASK = {
    "title": "Unbiased Pass@k Estimator (Chen et al.)",
    "title_zh": "Pass@k 无偏估计器",
    "difficulty": "Medium",
    "category": "评测与基准",
    "function_name": "pass_at_k",
    "starter_code": '''import torch

def pass_at_k(n, c, k):
    pass
''',
    "hint": "Compute the probability that all k draws are incorrect as a short product, then subtract it from one. Handle fewer than k incorrect samples first.",
    "hint_zh": "先用乘积计算 k 次抽样全部错误的概率，再用 1 减去它。首先处理错误样本少于 k 的情况。",
    "description_en": "Implement the unbiased per-problem pass@k estimator from sample totals and correct counts. Return one float32 value for each problem.",
    "description_zh": "根据每道题的采样总数与正确数，实现无偏 pass@k 估计器。为每道题返回一个 float32 数值。",
    "theory_en": "Under sampling without replacement, failure means choosing all k candidates from the `n-c` incorrect samples. Its probability is a ratio of combinations, evaluated stably as a product of k ratios.",
    "theory_zh": "在无放回抽样下，失败意味着 k 个候选全部来自 `n-c` 个错误样本。其概率是组合数之比，可表示为 k 个比值的乘积。",
    "diagram_en": "```mermaid\nflowchart LR\nN[n total, c correct] --> F[probability all k fail]\nF --> P[pass@k = 1 - failure]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nN[n 个总样本，c 个正确] --> F[k 个全错的概率]\nF --> P[pass@k = 1 - 失败概率]\n```",
    "tests": [
        {"name": "exact reference values", "code": """import torch
n = torch.tensor([10, 20])
c = torch.tensor([2, 5])
out = {fn}(n, c, 3)
expected = torch.tensor([
    1.0 - (8/10) * (7/9) * (6/8),
    1.0 - (15/20) * (14/19) * (13/18),
], dtype=torch.float32)
assert out.shape == (2,) and out.dtype == torch.float32
assert torch.allclose(out, expected, atol=1e-7)
"""},
        {"name": "certain pass branch", "code": """import torch
out = {fn}(torch.tensor([5, 5]), torch.tensor([3, 5]), 3)
assert out.shape == (2,)
assert torch.equal(out, torch.ones(2))
"""},
        {"name": "k one", "code": """import torch
n = torch.tensor([4.0, 10.0])
c = torch.tensor([1.0, 7.0])
out = {fn}(n, c, 1)
expected = c.float() / n.float()
assert out.shape == n.shape
assert torch.allclose(out, expected, atol=0, rtol=0)
"""},
        {"name": "zero correct and device", "code": """import torch
n = torch.tensor([3, 8])
c = torch.zeros(2, dtype=torch.long)
out = {fn}(n, c, 2)
assert out.device == n.device and out.shape == (2,)
assert torch.equal(out, torch.zeros(2, dtype=torch.float32))
"""},
    ],
    "solution": '''import torch

def pass_at_k(n, c, k):
    n_float = n.to(dtype=torch.float32)
    c_float = c.to(device=n.device, dtype=torch.float32)
    result = torch.empty_like(n_float)
    for i in range(n.numel()):
        if n_float[i] - c_float[i] < k:
            result[i] = 1.0
        else:
            failure = torch.ones((), device=n.device, dtype=torch.float32)
            for j in range(k):
                failure = failure * ((n_float[i] - c_float[i] - j) / (n_float[i] - j))
            result[i] = 1.0 - failure
    return result
''',
    "demo": '''n = torch.tensor([100, 100])
c = torch.tensor([1, 10])
print(pass_at_k(n, c, 10))
''',
    "explanation": "The shortcut returns one whenever it is impossible to choose k failures. Otherwise, the product exactly represents the ratio `C(n-c,k) / C(n,k)` without explicitly computing large combinations.",
}
