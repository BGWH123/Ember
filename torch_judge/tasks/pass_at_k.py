"""Unbiased Pass@k Estimator (Chen et al.) task."""

TASK = {
    'title': 'Unbiased Pass@k Estimator (Chen et al.)',
    'title_zh': 'Pass@k 无偏估计器',
    'difficulty': 'Medium',
    'category': '评测与基准',
    'function_name': 'pass_at_k',
    'starter_code': """def pass_at_k(n, c, k):
    pass""",
    'hint': """Compute the probability that all k draws are incorrect as a short product, then subtract it from one. Handle fewer than k incorrect samples first.""",
    'hint_zh': '先用乘积计算 k 次抽样全部错误的概率，再用 1 减去它。首先处理错误样本少于 k 的情况。',
    'description_en': """Implement the unbiased per-problem pass@k estimator from sample totals and correct counts. Return one float32 value for each problem.""",
    'description_zh': '根据每道题的采样总数与正确数，实现无偏 pass@k 估计器。为每道题返回一个 float32 数值。',
    'theory_en': """Under sampling without replacement, failure means choosing all k candidates from the `n-c` incorrect samples. Its probability is a ratio of combinations, evaluated stably as a product of k ratios.""",
    'theory_zh': '在无放回抽样下，失败意味着 k 个候选全部来自 `n-c` 个错误样本。其概率是组合数之比，可表示为 k 个比值的乘积。',
    'diagram_en': """```mermaid
flowchart LR
N[n total, c correct] --> F[probability all k fail]
F --> P[pass@k = 1 - failure]
```""",
    'diagram_zh': """```mermaid
flowchart LR
N[n 个总样本，c 个正确] --> F[k 个全错的概率]
F --> P[pass@k = 1 - 失败概率]
```""",
    "tests": [
        {
            "name": 'exact reference values',
            "code": """import torch
n = torch.tensor([10, 20])
c = torch.tensor([2, 5])
out = {fn}(n, c, 3)
expected = torch.tensor([
    1.0 - (8/10) * (7/9) * (6/8),
    1.0 - (15/20) * (14/19) * (13/18),
], dtype=torch.float32)
assert out.shape == (2,) and out.dtype == torch.float32
assert torch.allclose(out, expected, atol=1e-7)""",
        },
        {
            "name": 'certain pass branch',
            "code": """import torch
out = {fn}(torch.tensor([5, 5]), torch.tensor([3, 5]), 3)
assert out.shape == (2,)
assert torch.equal(out, torch.ones(2))""",
        },
        {
            "name": 'k one',
            "code": """import torch
n = torch.tensor([4.0, 10.0])
c = torch.tensor([1.0, 7.0])
out = {fn}(n, c, 1)
expected = c.float() / n.float()
assert out.shape == n.shape
assert torch.allclose(out, expected, atol=0, rtol=0)""",
        },
        {
            "name": 'zero correct and device',
            "code": """import torch
n = torch.tensor([3, 8])
c = torch.zeros(2, dtype=torch.long)
out = {fn}(n, c, 2)
assert out.device == n.device and out.shape == (2,)
assert torch.equal(out, torch.zeros(2, dtype=torch.float32))""",
        },
    ],
    'solution': '''def pass_at_k(n, c, k):
    """Pass@k 无偏估计器 (Unbiased Pass@k Estimator (Chen et al.))。"""
    # Step 1: 将总样本数与正确样本数转为浮点数，准备逐题估计
    n_float = n.to(dtype=torch.float32)
    c_float = c.to(device=n.device, dtype=torch.float32)
    result = torch.empty_like(n_float)
    # Step 2: 计算无放回抽到 k 个错误答案的组合数比值
    for problem_index in range(n.numel()):
        if n_float[problem_index] - c_float[problem_index] < k:
            result[problem_index] = 1.0
        else:
            failure = torch.ones((), device=n.device, dtype=torch.float32)
            for draw_index in range(k):
                failure = failure * ((n_float[problem_index] - c_float[problem_index] - draw_index) / (n_float[problem_index] - draw_index))
            result[problem_index] = 1.0 - failure
    # Step 3: 用 1 减去全错概率，返回每道题的 pass@k
    return result''',
    'demo': """torch.manual_seed(42)
n = torch.tensor([10, 20, 5])
c = torch.tensor([2, 5, 3])
scores = pass_at_k(n, c, k=3)
print('n / correct:', list(zip(n.tolist(), c.tolist())))
print('pass@3:', scores)
print('Certain-pass branch:', scores[-1].item())""",
    'explanation': r"""### 核心思路
对一道题生成 $n$ 个候选且其中 $c$ 个正确时，从中无放回选 $k$ 个全部错误的概率是 $\binom{n-c}{k}/\binom nk$，所以 $\operatorname{pass@k}=1-\binom{n-c}{k}/\binom nk$。代码不直接计算巨大阶乘，而展开为 $\prod_{i=0}^{k-1}(n-c-i)/(n-i)$，得到 Chen 等人使用的无偏估计。

### 代码步骤拆解
- **Step 1（统一计数精度）**：`n_float`、`c_float` 保持输入设备并转成 `float32`，`result` 与题目批次形状一致。
- **Step 2（累乘全错概率）**：对第 `problem_index` 道题，`failure` 逐次乘入第 `draw_index` 次仍从错误集合抽样的条件概率。
- **Step 3（转换为至少一解成功率）**：若 `n-c<k`，不可能选出 $k$ 个全错候选，结果直接为 1；否则写入 `1-failure`。

### 重点与易错点分析
- **注意点 1（无放回而非独立抽样）**：不能使用 $1-(1-c/n)^k$；每次抽取后分子与分母都减少 1，组合数比值才是标准 pass@k。
- **注意点 2（确定成功分支）**：条件是错误样本数 `n-c` 小于 `k`，不是正确样本数大于 `k`；此时全错组合数为零。
- **注意点 3（乘积实现）**：乘积避免阶乘溢出，也比先构造组合数稳定；该指标是离散评测统计量，通常不作为可微训练目标。

### 复杂度分析
- **时间复杂度**：若有 $B$ 道题，每题累乘 $k$ 项，复杂度为 $O(Bk)$。
- **空间复杂度**：输出和计数副本占 $O(B)$，每道题的 `failure` 仅需 $O(1)$ 标量空间。""",
}
