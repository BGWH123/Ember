"""VLA Action Chunking Temporal Ensemble & OpenVLA Tokenizer task."""

TASK = {
    'title': 'VLA Action Chunking Temporal Ensemble & OpenVLA Tokenizer',
    'title_zh': 'VLA 动作分块时序集成与 OpenVLA 离散化',
    'difficulty': 'Medium',
    'category': '具身智能与控制',
    'function_name': 'vla_action_ensemble_and_tokenize',
    'starter_code': """def vla_action_ensemble_and_tokenize(chunk_preds, decay_m, actions, low, high, n_bins=256):
    pass""",
    'hint': """Use index zero as age zero for exponential temporal weights. Clamp normalized actions before converting them to integer bins, then decode bin centers.""",
    'hint_zh': '令索引 0 对应年龄 0 以计算指数时序权重；将归一化动作截断后转为整数 bin，再解码 bin 中心。',
    'description_en': """Ensemble overlapping action-chunk predictions and uniformly tokenize bounded continuous actions in the OpenVLA style.""",
    'description_zh': '集成重叠动作块预测，并以 OpenVLA 风格对有界连续动作进行均匀离散化。',
    'theory_en': """Exponential weighting favors newer chunk predictions while retaining older estimates. Uniform bins map continuous controls to token IDs, and bin centers provide deterministic approximate decoding.""",
    'theory_zh': '指数加权偏重新动作块预测，同时保留较旧估计。均匀分箱将连续控制映射到 token ID，箱中心提供确定性的近似解码。',
    'diagram_en': """```text
chunks -> age decay -> weighted action
actions,bounds -> normalize -> bins -> bin centers
```""",
    'diagram_zh': """```text
动作块 -> 年龄衰减 -> 加权动作
动作、边界 -> 归一化 -> 分箱 -> 箱中心
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
chunks = torch.randn(4, 3)
actions = torch.randn(2, 3)
ens, ids, recon = {fn}(chunks, .2, actions, -torch.ones(3), torch.ones(3))
assert ens.shape == (3,) and ids.shape == recon.shape == (2, 3)""",
        },
        {
            "name": 'exact reference',
            "code": """import torch
chunks = torch.tensor([[1., 2.], [3., 4.]])
a = torch.tensor([[0., 1.]])
low, high = torch.tensor([-1., 0.]), torch.tensor([1., 2.])
ens, ids, recon = {fn}(chunks, torch.tensor(.5), a, low, high, n_bins=4)
w = torch.exp(-torch.tensor(.5) * torch.arange(2.)); w = w / w.sum()
ref_ids = torch.tensor([[2, 2]])
ref_recon = low + ((ref_ids.float() + .5) / 4) * (high - low)
assert torch.allclose(ens, (w[:, None] * chunks).sum(0)) and torch.equal(ids, ref_ids) and torch.allclose(recon, ref_recon)""",
        },
        {
            "name": 'bounds clipping edge case',
            "code": """import torch
a = torch.tensor([[-5., 5.]])
low, high = torch.tensor([0., 0.]), torch.tensor([1., 1.])
ens, ids, recon = {fn}(torch.tensor([[2., 3.]]), 1., a, low, high, n_bins=2)
assert torch.equal(ens, torch.tensor([2., 3.])) and torch.equal(ids, torch.tensor([[0, 1]])) and torch.allclose(recon, torch.tensor([[.25, .75]]))""",
        },
        {
            "name": 'ensemble gradient flow',
            "code": """import torch
chunks = torch.randn(3, 2, requires_grad=True)
decay = torch.tensor(.3, requires_grad=True)
ens, ids, recon = {fn}(chunks, decay, torch.zeros(1, 2), -torch.ones(2), torch.ones(2))
ens.sum().backward()
assert chunks.grad is not None and decay.grad is not None and ids.dtype == torch.long and recon.shape == (1, 2)""",
        },
    ],
    'solution': '''def vla_action_ensemble_and_tokenize(chunk_preds, decay_m, actions, low, high, n_bins=256):
    """VLA 动作时序集成与离散化 (Temporal Ensemble and Tokenization)。"""
    # Step 1: 按预测年龄计算并归一化指数权重
    K = chunk_preds.shape[0]
    ages = torch.arange(K, device=chunk_preds.device, dtype=chunk_preds.dtype)
    weights = torch.exp(-decay_m * ages)
    weights = weights / weights.sum()
    ensembled_actions = (weights.unsqueeze(-1) * chunk_preds).sum(dim=0)

    # Step 2: 将有界连续动作归一化并映射到离散区间
    normalized_actions = (actions - low) / (high - low + 1e-8)
    normalized_actions = normalized_actions.clamp(0.0, 1.0)
    token_ids = (normalized_actions * n_bins).long().clamp(0, n_bins - 1)

    # Step 3: 使用区间中心重建连续动作
    reconstructed_actions = low + ((token_ids.float() + 0.5) / n_bins) * (high - low)
    return ensembled_actions, token_ids, reconstructed_actions''',
    'demo': """torch.manual_seed(42)
chunk_preds = torch.randn(5, 7)
actions = torch.rand(3, 7) * 2 - 1
low, high = -torch.ones(7), torch.ones(7)
ensembled, token_ids, reconstructed = vla_action_ensemble_and_tokenize(
    chunk_preds, 0.1, actions, low, high
)
print("Output shape:", ensembled.shape)
print("Token shape:", token_ids.shape)
print("Sample values:", reconstructed[0, :3])""",
    'explanation': """### 核心思路
先用指数衰减权重对重叠动作预测加权，再将连续动作按 $u=(a-low)/(high-low)$ 映射到均匀离散区间，并用区间中心重建。

### 代码步骤拆解
- **Step 1（时序集成）**：`ages` 生成预测年龄，`weights` 归一化后沿首维聚合 `chunk_preds`。
- **Step 2（动作离散化）**：将 `actions` 广播归一化到 $[0,1]$，得到形状不变的 `token_ids`。
- **Step 3（中心解码）**：用 `(token_ids + 0.5) / n_bins` 取每个区间中心并恢复原始量纲。

### 重点与易错点分析
- **注意点 1（边界安全）**：归一化分母加入 `1e-8`，并在整数化前截断，避免上界产生越界编号。
- **注意点 2（年龄约定）**：索引 0 的年龄为 0，因此它是权重最大的最新预测。
- **注意点 3（广播规则）**：`low`、`high` 按动作维广播，`weights.unsqueeze(-1)` 按特征维广播。

### 复杂度分析
- **时间复杂度**：$O(KD+BD)$，瓶颈是动作集成和批量离散化。
- **空间复杂度**：$O(K+BD)$，主要保存权重、编号和重建动作。""",
}
