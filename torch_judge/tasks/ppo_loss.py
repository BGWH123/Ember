"""PPO (Proximal Policy Optimization) Clipped Loss task."""

TASK = {
    'title': 'PPO (Proximal Policy Optimization) Clipped Loss',
    'title_zh': 'PPO 损失',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'ppo_loss',
    'starter_code': """def ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2):
    pass""",
    'hint': """1. r = exp(new_logps - old_logps.detach())
2. unclipped = r * advantages.detach()
3. clipped = clamp(r, 1-ε, 1+ε) * advantages.detach()
4. loss = -min(unclipped, clipped).mean()""",
    'hint_zh': """1. r = exp(new_logps - old_logps.detach())
2. unclipped = r * advantages.detach()
3. clipped = clamp(r, 1-ε, 1+ε) * advantages.detach()
4. loss = -min(unclipped, clipped).mean()""",
    'description_en': """Implement the PPO clipped surrogate loss.

PPO constrains policy updates by clipping the importance sampling ratio, preventing destructively large updates during reinforcement learning.

**Signature:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`

**Parameters:**
- `new_logps` — current policy log-probs (B,)
- `old_logps` — old policy log-probs (B,), treated as constant
- `advantages` — advantage estimates (B,), treated as constant
- `clip_ratio` — clipping range epsilon

**Returns:** scalar loss

**Constraints:**
- Ratio: `r = exp(new_logps - old_logps.detach())`
- Loss: `-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`
- Gradients flow only through new_logps""",
    'description_zh': """实现 PPO 裁剪代理损失。

PPO 通过裁剪重要性采样比率来约束策略更新，防止强化学习中的破坏性大幅更新。

**签名:** `ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2) -> Tensor`

**参数:**
- `new_logps` — 当前策略对数概率 (B,)
- `old_logps` — 旧策略对数概率 (B,)，视为常量
- `advantages` — 优势估计 (B,)，视为常量
- `clip_ratio` — 裁剪范围 epsilon

**返回:** 标量损失

**约束:**
- 比率：`r = exp(new_logps - old_logps.detach())`
- 损失：`-min(r * A, clamp(r, 1-eps, 1+eps) * A).mean()`
- 梯度仅通过 new_logps 流动""",
    "tests": [
        {
            "name": 'Basic shape & type',
            "code": """import torch
from torch import Tensor
new_logps = torch.randn(16, requires_grad=True)
old_logps = torch.randn(16)
advantages = torch.randn(16)
loss = {fn}(new_logps, old_logps, advantages)
assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'""",
        },
        {
            "name": 'Numeric check vs fixed value',
            "code": """import torch
new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
loss = {fn}(new_logps, old_logps, advantages, clip_ratio=0.2)
expected = torch.tensor(-0.0488)
assert torch.allclose(loss, expected, atol=1e-4, rtol=0), 'Loss should match the expected numeric value on the fixed example'""",
        },
        {
            "name": 'Gradient flows to new_logps only',
            "code": """import torch
new_logps = torch.randn(8, requires_grad=True)
old_logps = torch.randn(8, requires_grad=True)
advantages = torch.randn(8, requires_grad=True)
loss = {fn}(new_logps, old_logps, advantages)
loss.backward()
assert new_logps.grad is not None, 'Gradients should flow through new_logps'
assert old_logps.grad is None, 'Gradients should not flow through old_logps (treat as constant baseline)'
assert advantages.grad is None, 'Gradients should not flow through advantages (treat as constant advantages)'""",
        },
    ],
    'solution': '''def ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2):
    """PPO 损失 (PPO (Proximal Policy Optimization) Clipped Loss)。"""
    # Step 1: 冻结行为策略对数概率与优势，构造新旧策略重要性比率
    old_logps_detached = old_logps.detach()
    adv_detached = advantages.detach()
    ratios = torch.exp(new_logps - old_logps_detached)
    # Step 2: 分别计算原始代理目标与限制在 [1-eps,1+eps] 的代理目标
    unclipped = ratios * adv_detached
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached
    # Step 3: 逐样本取更保守的下界并取负均值，得到可最小化损失
    return -torch.min(unclipped, clipped).mean()''',
    'demo': """torch.manual_seed(42)
old_logps = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25]))
new_logps = torch.log(torch.tensor([0.40, 0.20, 0.30, 0.10])).requires_grad_()
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
loss = ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2)
loss.backward()
print('Policy ratios:', torch.exp(new_logps.detach() - old_logps))
print('Clipped PPO loss:', loss.item())
print('Gradient on new log-probs:', new_logps.grad)""",
    'explanation': r"""### 核心思路
PPO 用重要性比率 $r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\mathrm{old}}(a_t|s_t))$ 复用旧策略轨迹。裁剪目标 $L^{CLIP}=\mathbb E[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$ 对会过度改善目标的策略变化设置平台；函数返回其负值以供梯度下降。

### 代码步骤拆解
- **Step 1（构造策略比率）**：先对 `old_logps` 与 `advantages` 调用 `detach`，再由 `new_logps-old_logps_detached` 的指数得到逐样本 `ratios`。
- **Step 2（建立裁剪下界）**：`unclipped=ratios*adv_detached` 保留原代理目标，`clipped` 将比率限制到 `[1-clip_ratio,1+clip_ratio]` 后乘同一优势。
- **Step 3（保守聚合）**：逐元素 `min` 选更小的收益项，均值后取负，得到标量 loss；梯度只回到 `new_logps`。

### 重点与易错点分析
- **注意点 1（优势符号决定裁剪方向）**：$A>0$ 时抑制 $r>1+\epsilon$ 的继续增益；$A<0$ 时抑制 $r<1-\epsilon$ 的继续增益。不能简单先裁剪比率后只使用裁剪项，否则另一侧也会错误失去梯度。
- **注意点 2（旧策略必须冻结）**：轨迹由 $\pi_{\mathrm{old}}$ 采样，它在本轮优化中是参考分布；不 `detach` 会让优化器同时移动分母，破坏重要性采样语义。
- **注意点 3（log-space 比率）**：用对数概率差再 `exp` 比直接相除概率更合适，但极端 log-ratio 仍可能溢出；实际训练通常监控 KL、限制更新轮数并使用合理精度。

### 复杂度分析
- **时间复杂度**：对 $B$ 个样本执行常数次逐元素运算和一次归约，为 $O(B)$。
- **空间复杂度**：`ratios`、两个代理目标及其反向激活均为长度 $B$，空间为 $O(B)$。""",
}
