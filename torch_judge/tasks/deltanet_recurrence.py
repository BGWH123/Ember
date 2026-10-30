"""DeltaNet linear-attention recurrence task."""

TASK = {
    "title": "DeltaNet Linear Attention Recurrence",
    "title_zh": "DeltaNet 线性注意力递推",
    "difficulty": "Hard",
    "category": "注意力机制",
    "function_name": "deltanet_recurrence",
    "starter_code": '''def deltanet_recurrence(q, k, v, beta, eps=1e-6):
    pass
''',
    "hint": "Normalize q/k once, then at each step compute `v_old=S@k`, apply the gated outer-product delta update, and read with `S@q`.",
    "hint_zh": "先归一化 q/k；每一步计算 `v_old=S@k`，执行门控外积增量更新，再用 `S@q` 读取。",
    "description_en": "Implement the recurrent DeltaNet state update and return every readout together with the final matrix-valued state.",
    "description_zh": "实现 DeltaNet 的循环状态更新，返回所有时间步输出以及最终矩阵状态。",
    "theory_en": "The state is an associative memory mapping normalized keys to values. The delta rule first removes the old prediction at a key, then writes the residual toward the new value.",
    "theory_zh": "状态是将归一化键映射到值的联想记忆。增量规则先取得该键的旧预测，再把预测残差写入新值方向。",
    "diagram_en": "```text\nS,k -> v_old -> beta*(v-v_old) outer k -> S' -> S'@q\n```",
    "diagram_zh": "```text\nS,k -> 旧值 -> beta*(v-旧值) 外积 k -> S' -> S'@q\n```",
    "tests": [
        {
            "name": "output and state shapes",
            "code": '''import torch
q = torch.randn(2, 5, 4)
k = torch.randn(2, 5, 4)
v = torch.randn(2, 5, 4)
out, state = {fn}(q, k, v, torch.full((2, 5), 0.5))
assert out.shape == (2, 5, 4) and state.shape == (2, 4, 4)
''',
        },
        {
            "name": "numerical recurrence",
            "code": '''import torch
torch.manual_seed(1)
q, k, v = [torch.randn(1, 3, 2) for _ in range(3)]
beta = torch.tensor([[0.2, 0.7, 0.4]])
out, state = {fn}(q, k, v, beta)
qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
s = torch.zeros(1, 2, 2)
ys = []
for t in range(3):
    old = (s @ kn[:, t, :, None]).squeeze(-1)
    s = s + beta[:, t, None, None] * (v[:, t] - old)[:, :, None] * kn[:, t, None, :]
    ys.append((s @ qn[:, t, :, None]).squeeze(-1))
ref = torch.stack(ys, 1)
assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(state, s, atol=1e-6)
''',
        },
        {
            "name": "zero gate edge case",
            "code": '''import torch
q, k, v = [torch.randn(2, 4, 3) for _ in range(3)]
out, state = {fn}(q, k, v, torch.zeros(2, 4))
assert torch.count_nonzero(out) == 0 and torch.count_nonzero(state) == 0
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q = torch.randn(1, 4, 3, requires_grad=True)
k = torch.randn(1, 4, 3, requires_grad=True)
v = torch.randn(1, 4, 3, requires_grad=True)
beta = torch.full((1, 4), 0.5, requires_grad=True)
out, state = {fn}(q, k, v, beta)
(out.square().sum() + state.square().sum()).backward()
assert all(t.grad is not None for t in (q, k, v, beta))
''',
        },
    ],
    "solution": '''import torch

def deltanet_recurrence(q, k, v, beta, eps=1e-6):
    B, T, D = q.shape
    q_norm = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
    k_norm = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
    state = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(T):
        q_t, k_t, v_t = q_norm[:, t], k_norm[:, t], v[:, t]
        v_old = (state @ k_t.unsqueeze(-1)).squeeze(-1)
        state = state + beta[:, t, None, None] * (v_t - v_old).unsqueeze(-1) * k_t.unsqueeze(-2)
        outputs.append((state @ q_t.unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=1), state
''',
    "demo": '''q, k, v = [torch.randn(2, 10, 16) for _ in range(3)]
out, final_state = deltanet_recurrence(q, k, v, torch.sigmoid(torch.randn(2, 10)))
print(out.shape, final_state.shape)
''',
    "explanation": "The update is a batched outer product between the value prediction error and the normalized key. Reading after writing produces one `(B,D)` output per time step.",
}
