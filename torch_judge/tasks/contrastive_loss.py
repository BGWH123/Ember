"""Contrastive Loss (InfoNCE) task."""

TASK = {
    "title": "Contrastive Loss (InfoNCE)",
    "title_zh": "对比损失（InfoNCE）",
    "difficulty": "Medium",
    "category": "损失函数",
    "description_en": (
        "Implement InfoNCE / NT-Xent contrastive loss, the objective used in CLIP and SimCLR.\n\nGiven a batch of query-key pairs, each query `q[i]` should be pulled toward its matching key `k[i]` and pushed away from all other keys.\n\n**Signature:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`\n\n**Parameters:**\n- `q` — query embeddings, shape (N, D), assumed L2-normalized\n- `k` — key embeddings, shape (N, D), assumed L2-normalized; `k[i]` is the positive for `q[i]`\n- `temperature` — softmax temperature τ\n\n**Returns:** scalar mean loss\n\n**Formula:**\n```\nL = mean over i of: -log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )\n```\nThis is equivalent to cross-entropy with targets `[0, 1, 2, ..., N-1]`.\n\n**Constraints:**\n- Do not use `F.*` or `nn.*` for cross-entropy or softmax\n- Implement the log-sum-exp manually"
    ),
    "description_zh": (
        "实现 InfoNCE / NT-Xent 对比损失，即 CLIP 和 SimCLR 中使用的训练目标。\n\n给定一批查询-键对，每个查询 `q[i]` 应被拉近其匹配键 `k[i]`，并远离所有其他键。\n\n**签名:** `contrastive_loss(q, k, temperature=0.07) -> Tensor`\n\n**参数:**\n- `q` — 查询嵌入，形状 (N, D)，假设已 L2 归一化\n- `k` — 键嵌入，形状 (N, D)，假设已 L2 归一化；`k[i]` 是 `q[i]` 的正样本\n- `temperature` — softmax 温度 τ\n\n**返回:** 标量均值损失\n\n**公式:**\n```\nL = 对 i 求均值：-log( exp(q_i·k_i / τ) / Σ_j exp(q_i·k_j / τ) )\n```\n等价于目标为 `[0, 1, 2, ..., N-1]` 的交叉熵。\n\n**约束:**\n- 不得使用 `F.*` 或 `nn.*` 计算交叉熵或 softmax\n- 手动实现 log-sum-exp"
    ),
    "function_name": "contrastive_loss",
    "hint": (
        "1. `logits = (q @ k.T) / temperature`  shape `(N, N)`\n2. `log_sum_exp = logits.logsumexp(dim=-1)`\n3. `pos = logits[arange(N), arange(N)]`\n4. `return -(pos - log_sum_exp).mean()`"
    ),
    "hint_zh": (
        "1. `logits = (q @ k.T) / temperature`  形状 `(N, N)`\n2. `log_sum_exp = logits.logsumexp(dim=-1)`\n3. `pos = logits[arange(N), arange(N)]`\n4. `return -(pos - log_sum_exp).mean()`"
    ),
    "tests": [
        {
            "name": "Returns scalar",
            "code": """









import torch
torch.manual_seed(0)
N, D = 8, 64
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss = {fn}(q, k)
assert loss.shape == (), f'Expected scalar, got {loss.shape}'
assert loss.item() > 0

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Perfect alignment gives near-zero loss",
            "code": """









import torch
torch.manual_seed(3)
N, D = 4, 32
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
# k == q means each query perfectly matches its key
loss = {fn}(q, q.clone(), temperature=0.07)
assert loss.item() < 0.1, f'Perfect alignment should give near-zero loss, got {loss.item():.4f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Random embeddings give loss ≈ log(N)",
            "code": """









import torch
import math
torch.manual_seed(99)
N, D = 64, 128
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss = {fn}(q, k, temperature=0.07)
expected = math.log(N)
assert abs(loss.item() - expected) < 1.0, f'Expected loss ≈ log({N})={expected:.3f}, got {loss.item():.3f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Lower temperature sharpens distribution",
            "code": """









import torch
torch.manual_seed(5)
N, D = 8, 32
q = torch.randn(N, D)
q = q / q.norm(dim=-1, keepdim=True)
k = torch.randn(N, D)
k = k / k.norm(dim=-1, keepdim=True)
loss_high_temp = {fn}(q, k, temperature=1.0)
loss_low_temp = {fn}(q, k, temperature=0.07)
# lower temperature amplifies differences, so loss values differ
assert abs(loss_high_temp.item() - loss_low_temp.item()) > 0.1, 'Temperature should affect loss magnitude'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
N, D = 4, 16
q = torch.randn(N, D, requires_grad=True)
k = torch.randn(N, D, requires_grad=True)
loss = {fn}(q, k)
loss.backward()
assert q.grad is not None, 'No gradient for q'
assert k.grad is not None, 'No gradient for k'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Exact numerical value",
            "code": """









import torch
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
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def contrastive_loss(q, k, temperature=0.07):
    # q, k: (N, D), assumed L2-normalized
    logits = (q @ k.T) / temperature  # (N, N)
    N = q.shape[0]
    # manual cross-entropy: log_p = logit_ii - log(sum_j exp(logit_ij))
    log_sum_exp = logits.logsumexp(dim=-1)  # (N,)
    positive_logits = logits[torch.arange(N, device=q.device), torch.arange(N, device=q.device)]
    log_p = positive_logits - log_sum_exp
    return -log_p.mean()  # 求均值
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








torch.manual_seed(0)
N, D = 8, 16

v = torch.randn(N, D)
q = v / v.norm(dim=-1, keepdim=True)
k = q.clone()
loss_perfect = contrastive_loss(q, k)
print(f"Perfect alignment loss: {loss_perfect:.4f}  (should be near log(1/N) = {-torch.log(torch.tensor(N, dtype=torch.float)):.4f})")

q_rand = torch.randn(N, D)
q_rand = q_rand / q_rand.norm(dim=-1, keepdim=True)
k_rand = torch.randn(N, D)
k_rand = k_rand / k_rand.norm(dim=-1, keepdim=True)
loss_rand = contrastive_loss(q_rand, k_rand)
print(f"Random embeddings loss:  {loss_rand:.4f}  (should be higher)")
    
    
    
    
    
    
    
    
    ''',
}
