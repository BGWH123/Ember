"""Contrastive Loss (InfoNCE) task."""

TASK = {
    'title': 'Contrastive Loss (InfoNCE)',
    'title_zh': '对比损失（InfoNCE）',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'contrastive_loss',
    'starter_code': """def contrastive_loss(q, k, temperature=0.07):
    pass""",
    'hint': """1. `logits = (q @ k.T) / temperature`  shape `(N, N)`
2. `log_sum_exp = logits.logsumexp(dim=-1)`
3. `pos = logits[arange(N), arange(N)]`
4. `return -(pos - log_sum_exp).mean()`""",
    'hint_zh': """1. `logits = (q @ k.T) / temperature`  形状 `(N, N)`
2. `log_sum_exp = logits.logsumexp(dim=-1)`
3. `pos = logits[arange(N), arange(N)]`
4. `return -(pos - log_sum_exp).mean()`""",
    'description_en': """Implement InfoNCE / NT-Xent contrastive loss, the objective used in CLIP and SimCLR.

Given a batch of query-key pairs, each query `q[i]` should be pulled toward its matching key `k[i]` and pushed away from all other keys.

**Signature:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`

**Parameters:**
- `q` — query embeddings, shape (N, D), assumed L2-normalized
- `k` — key embeddings, shape (N, D), assumed L2-normalized; `k[i]` is the positive for `q[i]`
- `temperature` — softmax temperature τ

**Returns:** scalar mean loss

**Formula:**
```
L = mean over i of: -log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```
This is equivalent to cross-entropy with targets `[0, 1, 2, ..., N-1]`.

**Constraints:**
- Do not use `F.*` or `nn.*` for cross-entropy or softmax
- Implement the log-sum-exp manually""",
    'description_zh': """实现 InfoNCE / NT-Xent 对比损失，即 CLIP 和 SimCLR 中使用的训练目标。

给定一批查询-键对，每个查询 `q[i]` 应被拉近其匹配键 `k[i]`，并远离所有其他键。

**签名:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`

**参数:**
- `q` — 查询嵌入，形状 (N, D)，假设已 L2 归一化
- `k` — 键嵌入，形状 (N, D)，假设已 L2 归一化；`k[i]` 是 `q[i]` 的正样本
- `temperature` — softmax 温度 τ

**返回:** 标量均值损失

**公式:**
```
L = 对 i 求均值：-log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )
```
等价于目标为 `[0, 1, 2, ..., N-1]` 的交叉熵。

**约束:**
- 不得使用 `F.*` 或 `nn.*` 计算交叉熵或 softmax
- 手动实现 log-sum-exp""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
N, D = 8, 64
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss = {fn}(q, k)
assert loss.shape == (), f'Expected scalar, got {loss.shape}'
assert loss.item() > 0""",
        },
        {
            "name": 'Perfect alignment gives near-zero loss',
            "code": """import torch
torch.manual_seed(3)
N, D = 4, 32
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
# k == q means each query perfectly matches its key
loss = {fn}(q, q.clone(), temperature=0.07)
assert loss.item() < 0.1, f'Perfect alignment should give near-zero loss, got {loss.item():.4f}'""",
        },
        {
            "name": 'Random embeddings give loss ≈ log(N)',
            "code": """import torch
import math
torch.manual_seed(99)
N, D = 64, 128
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss = {fn}(q, k, temperature=0.07)
expected = math.log(N)
assert abs(loss.item() - expected) < 1.0, f'Expected loss ≈ log({N})={expected:.3f}, got {loss.item():.3f}'""",
        },
        {
            "name": 'Lower temperature sharpens distribution',
            "code": """import torch
torch.manual_seed(5)
N, D = 8, 32
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss_high_temp = {fn}(q, k, temperature=1.0)
loss_low_temp = {fn}(q, k, temperature=0.07)
# lower temperature amplifies differences, so loss values differ
assert abs(loss_high_temp.item() - loss_low_temp.item()) > 0.1, 'Temperature should affect loss magnitude'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
N, D = 4, 16
q = torch.randn(N, D, requires_grad=True)
k = torch.randn(N, D, requires_grad=True)
loss = {fn}(q, k)
loss.backward()
assert q.grad is not None, 'No gradient for q'
assert k.grad is not None, 'No gradient for k'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
torch.manual_seed(42)
N, D = 4, 8
q = torch.randn(N, D)
k = torch.randn(N, D)
temperature = 0.07
# Pre-normalize as the function expects
q = q / q.norm(dim=-1, keepdim=True)
k = k / k.norm(dim=-1, keepdim=True)
sim = q @ k.T / temperature
log_probs = sim - torch.log(torch.exp(sim).sum(dim=-1, keepdim=True))
expected = -log_probs[torch.arange(N), torch.arange(N)].mean()
out = {fn}(q, k, temperature)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'""",
        },
    ],
    'solution': '''def contrastive_loss(q, k, temperature=0.07):
    """批内对比损失 (InfoNCE Contrastive Loss)。"""
    N = q.shape[0]

    # Step 1: 计算温度缩放的批内相似度矩阵
    logits = q @ k.T / temperature

    # Step 2: 提取对角正样本并稳定计算对数归一化项
    indices = torch.arange(N, device=q.device)
    positive_logits = logits[indices, indices]
    log_partition = logits.logsumexp(dim=-1)

    # Step 3: 对正样本负对数似然取均值
    return -(positive_logits - log_partition).mean()''',
    'demo': """torch.manual_seed(42)
N, D = 8, 16
q = torch.randn(N, D)
k = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = k / k.norm(dim=-1, keepdim=True)
loss = contrastive_loss(q, k)
print("Output shape:", loss.shape)
print("Loss:", loss.item())
print("Positive similarity:", (q[0] * k[0]).sum().item())""",
    'explanation': r"""### 核心思路
InfoNCE 把同索引 `(q_i,k_i)` 视为正样本，其余批内键为负样本，最小化 $-\log\frac{e^{q_i k_i/\tau}}{\sum_j e^{q_i k_j/\tau}}$。

### 代码步骤拆解
- **Step 1（批内 logits）**：矩阵乘法得到 `(N, N)` 相似度并除以 `temperature`。
- **Step 2（稳定归一化）**：提取对角 `positive_logits`，用 `logsumexp` 计算每行对数配分函数。
- **Step 3（均值损失）**：正样本 logit 减配分函数后取负均值。

### 重点与易错点分析
- **注意点 1（数值稳定）**：`logsumexp` 内部执行最大值平移，避免显式 `exp` 溢出。
- **注意点 2（温度作用）**：温度越小分布越尖锐，但过小会放大梯度和数值误差。
- **注意点 3（配对假设）**：本题假定正样本位于相似度矩阵对角线，且 `q/k` 已做 L2 归一化。

### 复杂度分析
- **时间复杂度**：$O(N^2D)$，主要是全批次相似度矩阵乘法。
- **空间复杂度**：$O(N^2)$，主要保存 logits。""",
}
