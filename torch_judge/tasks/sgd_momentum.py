"""SGD with Momentum task."""

TASK = {
    'title': 'SGD with Momentum',
    'title_zh': '带动量的 SGD',
    'difficulty': 'Easy',
    'category': '优化器与学习率',
    'function_name': 'sgd_momentum',
    'starter_code': """def sgd_momentum(param, grad, velocity, lr, momentum):
    pass""",
    'hint': 'Update velocity first, then update parameter using the new velocity. Momentum is typically 0.9.',
    'hint_zh': '先更新速度，再用新速度更新参数。动量通常取 0.9。',
    'description_en': """Implement SGD with momentum update step. Given parameter p, gradient g, and velocity v: v_new = momentum * v + g, p_new = p - lr * v_new. This accumulates velocity in directions of consistent gradient, accelerating convergence in relevant directions and dampening oscillations.""",
    'description_zh': """实现带动量的 SGD 更新步。给定参数 p、梯度 g 和速度 v：v_new = momentum * v + g，p_new = p - lr * v_new。这会在梯度一致的方向上积累速度，加速收敛并抑制震荡。""",
    'theory_en': """Momentum simulates the inertia of a heavy ball rolling down the loss landscape. It accelerates in directions of consistent gradient and reduces oscillations in directions with high curvature.""",
    'theory_zh': '动量模拟重球在损失 landscape 上滚动的惯性。在梯度一致的方向上加速，在高曲率方向上减少震荡。',
    "tests": [
        {
            "name": 'Nonzero velocity update',
            "code": """p = torch.tensor([[1., -2.], [3., 4.]])
g = torch.tensor([[0.5, -1.], [2., -0.5]])
v = torch.tensor([[-0.5, 2.], [1., 0.]])
pn, vn = {fn}(p, g, v, 0.2, 0.8)
expected_v = 0.8 * v + g
assert torch.allclose(vn, expected_v)
assert torch.allclose(pn, p - 0.2 * expected_v)""",
        },
        {
            "name": 'Two steps and zero momentum',
            "code": """p = torch.tensor([2., -3.]); v = torch.zeros(2)
for g in [torch.tensor([1., 2.]), torch.tensor([-0.5, 1.])]:
    old_p, old_v = p.clone(), v.clone()
    p, v = {fn}(p, g, v, 0.1, 0.9)
    assert torch.allclose(v, 0.9 * old_v + g)
    assert torch.allclose(p, old_p - 0.1 * v)
pn, vn = {fn}(p, torch.ones(2), v, 0.1, 0.)
assert torch.equal(vn, torch.ones(2))
assert torch.allclose(pn, p - 0.1)""",
        },
        {
            "name": 'Zero learning rate and input preservation',
            "code": """p = torch.tensor([2., 4.]); g = torch.tensor([1., -1.]); v = torch.tensor([0.2, 0.3])
snapshots = [x.clone() for x in [p, g, v]]
pn, vn = {fn}(p, g, v, 0., 0.5)
assert torch.equal(pn, p)
assert torch.allclose(vn, 0.5 * v + g)
for x, before in zip([p, g, v], snapshots):
    assert torch.equal(x, before)""",
        },
    ],
    'solution': '''def sgd_momentum(param, grad, velocity, lr, momentum):
    """实现带动量的 SGD（SGD with Momentum）。"""
    # Step 1: 对历史速度施加动量衰减以保留梯度方向惯性
    decayed_velocity = momentum * velocity
    # Step 2: 累加当前梯度形成这一时刻的新速度
    velocity_new = decayed_velocity + grad
    # Step 3: 使用新速度而非旧速度更新参数并返回优化器状态
    param_new = param - lr * velocity_new
    return (param_new, velocity_new)''',
    'demo': """torch.manual_seed(42)
param = torch.randn(3)
grad = torch.randn(3)
velocity = torch.zeros_like(param)
updated_param, updated_velocity = sgd_momentum(param, grad, velocity, 0.1, 0.9)
print("Gradient:", grad)
print("New velocity:", updated_velocity)
print("Parameter delta:", updated_param - param)""",
    'explanation': r"""### 核心思路
经典 Polyak 动量维护速度 $v_t=\mu v_{t-1}+g_t$，再用 $\theta_t=\theta_{t-1}-\eta v_t$ 更新参数。展开可得 $v_t=\sum_{j=0}^{t}\mu^jg_{t-j}$，因此同向梯度会持续累积，而交替变号的高曲率方向会相互抵消。

### 代码步骤拆解
- **Step 1（衰减历史速度）**：`decayed_velocity = momentum * velocity` 将上一步状态按 $\mu$ 衰减，控制历史梯度的有效记忆长度。
- **Step 2（注入当前梯度）**：`velocity_new = decayed_velocity + grad` 得到与 `param` 同形状的新状态，当前梯度权重为 1。
- **Step 3（沿速度下降）**：计算 `param_new = param - lr * velocity_new`，并返回 `velocity_new` 供下一次调用递推。

### 重点与易错点分析
- **注意点 1（更新顺序）**：参数必须使用 $v_t$ 而不是旧的 $v_{t-1}$；先改参数再算速度会产生一拍延迟，与题目公式不符。
- **注意点 2（状态连续性）**：调用方需要把返回的 `velocity_new` 传入下一步；每轮重置为零会退化成普通 SGD，`momentum=0` 也会精确得到这一退化形式。
- **注意点 3（与 Nesterov 区别）**：这里直接沿累积速度更新，不计算前瞻点梯度，因此不是 Nesterov momentum；实现采用非原地表达式，输入 `param` 和 `velocity` 保持不变。

### 复杂度分析
- **时间复杂度**：$O(P)$，对 $P$ 个参数各执行常数次乘加。
- **空间复杂度**：$O(P)$，新参数和新速度各需与参数同规模的存储；优化器持久状态本身为一个 $O(P)$ 速度张量。""",
}
