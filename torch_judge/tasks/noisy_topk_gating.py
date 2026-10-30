"""Noisy Top-K Gating (MoE) task."""

TASK = {
    'title': 'Noisy Top-K Gating (MoE)',
    'title_zh': '噪声 Top-K 门控（MoE）',
    'difficulty': 'Hard',
    'category': '混合专家模型',
    'function_name': 'NoisyTopKGating',
    'starter_code': """class NoisyTopKGating(nn.Module):
    pass""",
    'hint': """Generate noise using torch.randn. Use softplus for noise scaling. Top-k with softmax on selected logits only. Expose the clean-logit and noise projections as self.W_g and self.W_noise, matching the starter class.""",
    'hint_zh': """使用 torch.randn 生成噪声。用 softplus 做噪声缩放。仅对选中的 top-k logits 做 softmax。 按起始模板将干净 logits 和噪声投影分别命名为 self.W_g 与 self.W_noise。""",
    'description_en': """Implement Noisy Top-K Gating for Mixture-of-Experts. Given input x and n_experts, compute: logits = x @ W_g + noise * (x @ W_noise), where noise = StandardNormal * softplus(x @ W_noise). Select top-k experts, apply softmax to their logits. Return (weights, expert_indices) where weights sum to 1 for each token.""",
    'description_zh': """实现混合专家模型（MoE）的噪声 Top-K 门控。给定输入 x 和专家数量 n_experts，计算：logits = x @ W_g + noise * (x @ W_noise)，其中 noise = StandardNormal * softplus(x @ W_noise)。选择 top-k 专家，对它们的 logits 应用 softmax。返回 (weights, expert_indices)，其中每个 token 的权重和为1。""",
    'theory_en': """Noisy Top-K gating is the routing mechanism in Switch Transformer and other MoE models. Noise helps explore different expert assignments during training, improving load balancing.""",
    'theory_zh': '噪声 Top-K 门控是 Switch Transformer 等 MoE 模型的路由机制。噪声帮助训练时探索不同的专家分配，改善负载均衡。',
    "tests": [
        {
            "name": 'Deterministic expert ranking without noise',
            "code": """layer = {fn}(3, 4, top_k=2)
with torch.no_grad():
    layer.W_g.weight.copy_(torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [-1., 0., 0.]]))
    layer.W_g.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
x = torch.tensor([[2., 1., -1.], [-1., 2., 1.]])
logits = F.linear(x, layer.W_g.weight, layer.W_g.bias)
values, indices = logits.topk(2, dim=-1)
weights, actual_indices = layer(x, training=False)
assert torch.equal(actual_indices, indices)
assert torch.allclose(weights, values.softmax(-1), atol=1e-6)""",
        },
        {
            "name": 'Top-one and all experts',
            "code": """for k in [1, 5]:
    torch.manual_seed(123)
    layer = {fn}(3, 5, top_k=k)
    weights, indices = layer(torch.ones(4, 3), training=False)
    assert weights.shape == indices.shape == (4, k)
    assert indices.dtype == torch.long
    assert ((indices >= 0) & (indices < 5)).all()
    assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)
    for row in indices:
        assert row.unique().numel() == k""",
        },
        {
            "name": 'Training noise and differentiable weights',
            "code": """layer = {fn}(3, 4, top_k=2)
with torch.no_grad():
    layer.W_g.weight.zero_(); layer.W_g.bias.zero_()
    layer.W_noise.weight.zero_(); layer.W_noise.bias.zero_()
x = torch.ones(32, 3)
torch.manual_seed(99)
w1, i1 = layer(x, training=True)
torch.manual_seed(99)
w2, i2 = layer(x, training=True)
assert torch.equal(i1, i2) and torch.allclose(w1, w2)
assert i1.unique().numel() > 2
assert torch.allclose(w1.sum(-1), torch.ones(32), atol=1e-6)
(w1[:, 0].sum()).backward()
assert layer.W_noise.bias.grad is not None
assert layer.W_noise.bias.grad.abs().sum() > 0""",
        },
    ],
    'solution': '''class NoisyTopKGating(nn.Module):
    """噪声 Top-K 门控（MoE） (Noisy Top-K Gating (MoE))。"""
    # Step 1: 建立独立的专家得分投影与输入相关噪声尺度投影

    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.W_g = nn.Linear(input_dim, num_experts)
        self.W_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x, training=True):
        clean_logits = self.W_g(x)
        # Step 2: 训练时用 softplus 保证噪声标准差为正，并扰动专家排序以促进探索
        if training:
            noise_std = F.softplus(self.W_noise(x))
            noise = torch.randn_like(clean_logits) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        # Step 3: 选取扰动分数最高的 k 个专家，并仅在入选集合内归一化门控权重
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return (top_k_weights, top_k_indices)''',
    'demo': """torch.manual_seed(42)
B, D = 4, 8
x = torch.randn(B, D)
gating = NoisyTopKGating(D, 6, top_k=2)
weights, indices = gating(x, training=False)
print('门控权重/索引形状:', tuple(weights.shape), tuple(indices.shape))
print('首个样本选中专家:', indices[0].tolist())
print('每个样本权重和:', weights.sum(dim=-1).tolist())""",
    'explanation': r"""### 核心思路
门控器为每个 token 生成干净专家分数 $g(x)=W_gx+b_g$。训练时另一个投影给出正标准差 $\sigma(x)=\operatorname{softplus}(W_nx+b_n)$，采样 $\tilde g=g+\epsilon\odot\sigma,\ \epsilon\sim\mathcal N(0,I)$，使边界附近的专家有机会被探索。随后只保留 top-k 分数，并在这 k 个值上 softmax 得到和为 1 的稀疏组合权重。

### 代码步骤拆解
- **Step 1（双投影参数化）**：`W_g` 和 `W_noise` 都从 `input_dim` 映射到 `num_experts`，但前者决定均值，后者决定逐 token、逐专家噪声尺度。
- **Step 2（条件高斯探索）**：仅当 `training=True` 时计算 `noise_std` 并乘标准正态样本；推理分支直接使用 `clean_logits`，相同输入得到确定路由。
- **Step 3（稀疏选择与归一化）**：`topk` 返回形状 `(...,K)` 的分数和专家编号，softmax 只在最后 K 维执行，因此各 token 的返回权重严格和为 1。

### 重点与易错点分析
- **注意点 1（softplus 的职责）**：噪声投影可为任意实数，不能直接充当标准差；softplus 将其平滑映射到正数，同时保留 `W_noise` 的梯度。
- **注意点 2（随机性只属于训练分支）**：这里显式参数 `training` 控制噪声，而不是 `self.training`；调用方必须传入正确模式，测试也据此验证确定性。
- **注意点 3（top-k 离散边界）**：索引选择对分数不可微，但选中 logits 的 softmax 权重可微，噪声尺度能通过已选分支获得梯度；噪声有助探索，却不等同于负载均衡损失。

### 复杂度分析
- **时间复杂度**：对 N 个 token，两个线性投影为 $O(ND E)$，top-k 选择约为 $O(NE)$，K 维 softmax 为 $O(NK)$；投影通常占主导。
- **空间复杂度**：干净 logits、噪声尺度和扰动 logits 各为 $O(NE)$，返回权重与索引为 $O(NK)$。""",
}
