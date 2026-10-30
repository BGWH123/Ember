"""Kaiming Initialization task."""

TASK = {
    'title': 'Kaiming Initialization',
    'title_zh': 'Kaiming 初始化',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'kaiming_init',
    'starter_code': """def kaiming_init(weight):
    pass""",
    'hint': """`fan_in = weight.shape[1]` → `std = sqrt(2 / fan_in)`
`weight.normal_(0, std)` in-place → return `weight`""",
    'hint_zh': """`fan_in = weight.shape[1]` → `std = sqrt(2 / fan_in)`
`weight.normal_(0, std)` 原地操作 → 返回 `weight`""",
    'description_en': """Implement Kaiming (He) weight initialization.

Kaiming initialization sets weight variance based on fan-in to preserve signal magnitude through ReLU networks, preventing vanishing/exploding activations.

**Signature:** `kaiming_init(weight) -> Tensor`

**Parameters:**
- `weight` — tensor to initialize in-place (out_features, in_features)

**Returns:** the same tensor (in-place operation)

**Constraints:**
- `std = sqrt(2 / fan_in)` where `fan_in = weight.shape[1]`
- Fill with `normal(0, std)`
- Smaller fan_in should give larger std""",
    'description_zh': """实现 Kaiming（He）权重初始化。

Kaiming 初始化根据 fan-in 设置权重方差，以在 ReLU 网络中保持信号幅度，防止激活值消失或爆炸。

**签名:** `kaiming_init(weight) -> Tensor`

**参数:**
- `weight` — 需要原地初始化的张量 (out_features, in_features)

**返回:** 同一张量（原地操作）

**约束:**
- `std = sqrt(2 / fan_in)`，其中 `fan_in = weight.shape[1]`
- 用 `normal(0, std)` 填充
- 更小的 fan_in 应产生更大的 std""",
    "tests": [
        {
            "name": 'Mean approximately 0',
            "code": """import torch
torch.manual_seed(0)
w = torch.empty(256, 512)
{fn}(w)
assert abs(w.mean().item()) < 0.02, f'Mean too far from 0: {w.mean().item():.4f}'""",
        },
        {
            "name": 'Std matches sqrt(2/fan_in)',
            "code": """import torch, math
torch.manual_seed(0)
fan_in = 1024
w = torch.empty(256, fan_in)
{fn}(w)
expected = math.sqrt(2.0 / fan_in)
assert abs(w.std().item() - expected) < 0.005, f'Std {w.std().item():.4f} vs expected {expected:.4f}'""",
        },
        {
            "name": 'Returns same tensor (in-place)',
            "code": """import torch
w = torch.empty(64, 32)
out = {fn}(w)
assert out is w, 'Should return the same tensor'
assert out.shape == (64, 32), 'Shape should be unchanged'""",
        },
        {
            "name": 'Smaller fan_in gives larger std',
            "code": """import torch
w1 = torch.empty(64, 16)
w2 = torch.empty(64, 256)
{fn}(w1)
{fn}(w2)
assert w1.std().item() > w2.std().item(), 'Smaller fan_in should give larger std'""",
        },
    ],
    'solution': '''def kaiming_init(weight):
    """实现Kaiming 初始化（Kaiming Initialization）。"""
    # Step 1: 从输入连接数 fan_in 推导补偿 ReLU 截断的 He 标准差 sqrt(2/fan_in)
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    # Step 2: 在 no_grad 域内用 N(0, 2/fan_in) 原地填充全部权重
    with torch.no_grad():
        weight.normal_(0, std)
    # Step 3: 返回同一张量对象，使参数身份与形状均保持不变
    return weight''',
    'demo': """torch.manual_seed(42)
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2 / 512):.4f})')""",
    'explanation': r"""### 核心思路
若输入分量独立且权重零均值，线性层输出方差约为 $\operatorname{fan\_in}\operatorname{Var}(w)\operatorname{Var}(x)$。ReLU 约丢弃一半能量，因此设置 $\operatorname{Var}(w)=2/\operatorname{fan\_in}$，即正态分布标准差 $\sqrt{2/\operatorname{fan\_in}}$，以跨层维持激活尺度。

### 代码步骤拆解
- **Step 1（计算 He 尺度）**：二维线性权重 `(out_features,in_features)` 取第二维为 `fan_in`，计算 `std=math.sqrt(2.0/fan_in)`。
- **Step 2（原地正态采样）**：在 `torch.no_grad()` 中执行 `weight.normal_(0,std)`，覆盖每个参数但不记录初始化操作的梯度历史。
- **Step 3（保持参数身份）**：返回原对象 `weight`，因此调用方持有的 Parameter、存储地址和 shape 均不改变。

### 重点与易错点分析
- **注意点 1（标准差不是方差）**：`normal_` 的第二个参数是标准差，必须传 $\sqrt{2/fan\_in}$ 而非 $2/fan\_in$。
- **注意点 2（fan 模式）**：这里按前向 ReLU 保方差采用 `fan_in`；若目标是稳定反向梯度可选择 `fan_out`，卷积权重还需把输入通道与核面积相乘。
- **注意点 3（原地叶张量修改）**：可训练叶 Parameter 在启用梯度时原地赋值会报错或污染图，因此初始化必须置于 `no_grad`；采样统计只会近似理论均值和方差。

### 复杂度分析
- **时间复杂度**：设权重元素数为 $N$，生成并写入随机数需要 $O(N)$。
- **空间复杂度**：原地写入不分配同尺寸输出，除随机内核状态外额外空间为 $O(1)$。""",
}
