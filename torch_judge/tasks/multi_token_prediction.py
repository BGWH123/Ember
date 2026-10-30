"""Multi-Token Prediction Loss task."""

TASK = {
    'title': 'Multi-Token Prediction Loss',
    'title_zh': '多 Token 预测',
    'difficulty': 'Medium',
    'category': '训练技巧',
    'function_name': 'multi_token_prediction_loss',
    'starter_code': """def multi_token_prediction_loss(hidden_states, heads, targets):
    pass""",
    'hint': """For each head `i`:
1. `logits = hidden_states @ heads[i]`
2. Stable log-softmax: `shifted = logits - logits.max(-1,keepdim=True).values`
   `log_probs = shifted - log(exp(shifted).sum(-1,keepdim=True))`
3. `log_p = log_probs.gather(-1, targets[:,:,i].unsqueeze(-1)).squeeze(-1)`
4. `loss += -log_p.mean()`
Return `total_loss / N`""",
    'hint_zh': """对每个头 `i`：
1. `logits = hidden_states @ heads[i]`
2. 稳定 log-softmax：`shifted = logits - logits.max(-1,keepdim=True).values`
   `log_probs = shifted - log(exp(shifted).sum(-1,keepdim=True))`
3. `log_p = log_probs.gather(-1, targets[:,:,i].unsqueeze(-1)).squeeze(-1)`
4. `loss += -log_p.mean()`
返回 `total_loss / N`""",
    'description_en': """Implement the Multi-Token Prediction (MTP) training loss from Meta's 2024 paper.

Instead of predicting only the next token, train N independent prediction heads to simultaneously predict the next N tokens from the same hidden states. Average the cross-entropy loss across all heads.

**Signature:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`

**Parameters:**
- `hidden_states` — transformer hidden states (B, S, D)
- `heads` — list of N weight matrices, each (D, vocab_size)
- `targets` — target token IDs (B, S, N), where `targets[:, :, i]` are the labels for head i

**Returns:** scalar mean loss (average CE across all N heads)

**Constraints:**
- Implement cross-entropy manually: log-softmax then gather
- Do NOT use F.cross_entropy or F.log_softmax
- Use numerically stable log-softmax: subtract max before exp""",
    'description_zh': """实现 Meta 2024 年论文中的多 Token 预测（MTP）训练损失。

不只预测下一个 token，而是训练 N 个独立的预测头，从相同的隐藏状态同时预测接下来的 N 个 token。对所有头的交叉熵损失取平均。

**签名:** `multi_token_prediction_loss(hidden_states, heads, targets) -> Tensor`

**参数:**
- `hidden_states` — Transformer 隐藏状态 (B, S, D)
- `heads` — N 个权重矩阵的列表，每个形状为 (D, vocab_size)
- `targets` — 目标 token ID (B, S, N)，其中 `targets[:, :, i]` 是第 i 个头的标签

**返回:** 标量均值损失（所有 N 个头的 CE 均值）

**约束:**
- 手动实现交叉熵：log-softmax 后 gather
- 不能使用 F.cross_entropy 或 F.log_softmax
- 使用数值稳定的 log-softmax：exp 前先减去最大值""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
B, S, D, vocab, N = 2, 6, 16, 50, 3
hidden = torch.randn(B, S, D)
heads = [torch.randn(D, vocab) for _ in range(N)]
targets = torch.randint(0, vocab, (B, S, N))
loss = {fn}(hidden, heads, targets)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'""",
        },
        {
            "name": 'N=1 matches standard next-token CE',
            "code": """import torch
torch.manual_seed(1)
B, S, D, vocab = 1, 4, 8, 20
hidden = torch.randn(B, S, D)
W = torch.randn(D, vocab)
targets_3d = torch.randint(0, vocab, (B, S, 1))
loss_mtp = {fn}(hidden, [W], targets_3d)
# Manual CE for reference
logits = hidden @ W
lmax = logits.max(dim=-1, keepdim=True).values
shifted = logits - lmax
log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
log_p = log_probs.gather(-1, targets_3d[:, :, 0].unsqueeze(-1)).squeeze(-1)
expected = -log_p.mean()
assert torch.allclose(loss_mtp, expected, atol=1e-5), f'N=1 loss {loss_mtp.item():.4f} != expected {expected.item():.4f}'""",
        },
        {
            "name": 'Loss decreases when predictions are correct',
            "code": """import torch
torch.manual_seed(2)
B, S, D, vocab, N = 1, 3, 8, 10, 2
hidden = torch.randn(B, S, D)
targets = torch.randint(0, vocab, (B, S, N))
# Random heads -> high loss
heads_random = [torch.randn(D, vocab) for _ in range(N)]
loss_random = {fn}(hidden, heads_random, targets)
# Heads tuned to produce correct logits -> low loss
heads_good = []
for i in range(N):
    W = torch.zeros(D, vocab)
    # Make the target class have high logit
    for b in range(B):
        for s in range(S):
            t = targets[b, s, i].item()
            W[:, t] += hidden[b, s] * 10
    heads_good.append(W)
loss_good = {fn}(hidden, heads_good, targets)
assert loss_good < loss_random, f'Good predictions should have lower loss: {loss_good.item():.4f} vs {loss_random.item():.4f}'""",
        },
        {
            "name": 'Gradient flows through hidden_states and heads',
            "code": """import torch
torch.manual_seed(3)
B, S, D, vocab, N = 1, 4, 8, 15, 2
hidden = torch.randn(B, S, D, requires_grad=True)
heads = [torch.randn(D, vocab, requires_grad=True) for _ in range(N)]
targets = torch.randint(0, vocab, (B, S, N))
loss = {fn}(hidden, heads, targets)
loss.backward()
assert hidden.grad is not None, 'Missing gradient for hidden_states'
for i, h in enumerate(heads):
    assert h.grad is not None, f'Missing gradient for heads[{i}]'""",
        },
    ],
    'solution': '''def multi_token_prediction_loss(hidden_states, heads, targets):
    """多 Token 预测 (Multi-Token Prediction Loss)。"""
    # Step 1: 读取并行预测头数量，为各未来偏移的交叉熵建立累计量
    B, S, D = hidden_states.shape
    num_heads = len(heads)
    total_loss = 0.0
    # Step 2: 每个头投影词表 logits，并用减最大值方式手动计算稳定 log-softmax
    for head_index, head in enumerate(heads):
        logits = hidden_states @ head
        logits_max = logits.max(dim=-1, keepdim=True).values
        shifted = logits - logits_max
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
        # Step 3: gather 当前未来偏移的目标对数概率，累加后对所有预测头等权平均
        target_ids = targets[:, :, head_index]
        log_p = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        total_loss = total_loss + -log_p.mean()
    return total_loss / num_heads''',
    'demo': """torch.manual_seed(42)
B, S, D, V = 2, 5, 16, 10
hidden_states = torch.randn(B, S, D)
heads = [torch.randn(D, V) for _ in range(3)]
targets = torch.randint(0, V, (B, S, 3))
loss = multi_token_prediction_loss(hidden_states, heads, targets)
print('隐藏状态/目标形状:', tuple(hidden_states.shape), tuple(targets.shape))
print('并行未来预测头数:', len(heads))
print('平均多 Token 交叉熵:', f'{loss.item():.4f}')""",
    'explanation': r"""### 核心思路
MTP 让同一位置的隐藏向量 $h_{b,s}$ 通过 H 个独立矩阵 $W_j$，分别预测第 $j$ 个未来 token。第 j 头 logits 为 $z^{(j)}=hW_j\in\mathbb R^V$，总损失是 $L=\frac1H\sum_{j=1}^{H}\frac1{BS}\sum_{b,s}-\log p^{(j)}_{b,s,y^{(j)}_{b,s}}$。各头共享隐藏状态但不共享词表投影，因此多个未来监督共同训练主干。

### 代码步骤拆解
- **Step 1（对应未来偏移）**：`len(heads)=H`，并约定 `heads[j]` 对应 `targets[:,:,j]`；每个权重矩阵形状为 `(D,V)`。
- **Step 2（稳定词表归一化）**：`hidden_states @ head` 产生 `(B,S,V)`；逐位置减 `logits_max` 后计算 $\log p=z-\log\sum_v e^z$，避免大 logit 的指数溢出。
- **Step 3（抽取标签并跨头平均）**：`gather` 用 `(B,S,1)` 的标签索引取正确类别对数概率，每头先对 BS 个位置平均，最后除以 `num_heads`。

### 重点与易错点分析
- **注意点 1（最大值平移不改变概率）**：每个 `(b,s)` 必须沿词表维 V 单独减最大值；跨 batch 或序列取一个全局最大值虽数学上仍可归一化，却会造成不必要的下溢。
- **注意点 2（标签轴与头轴一致）**：`targets[:,:,head_index]` 才是该头的未来偏移标签，误把最后一维当词表维会导致 gather 形状或监督错位。
- **注意点 3（平均策略）**：实现对头等权，再对每头全部位置等权；这与把 H 个损失相加不同，后者会让整体梯度幅度随预测头数线性增长。

### 复杂度分析
- **时间复杂度**：$O(HBSDV)$，H 次隐藏状态到词表的矩阵乘法占主导，log-softmax 另需 $O(HBSV)$。
- **空间复杂度**：循环逐头执行时前向峰值为 $O(BSV)$，用于当前头的 logits、shifted 和 log_probs；参数 `heads` 本身占 $O(HDV)$。""",
}
