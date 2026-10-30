"""Flow Matching Heun (2nd-Order RK2) ODE Step task."""

TASK = {
    'title': 'Flow Matching Heun (2nd-Order RK2) ODE Step',
    'title_zh': '流匹配 Heun 二阶 ODE 采样步',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'flow_heun_step',
    'starter_code': """def flow_heun_step(x, velocity_fn, t_curr, t_next):
    pass""",
    'hint': """Evaluate velocity at the current state, make an Euler predictor, evaluate again at the endpoint, then average the two slopes.""",
    'hint_zh': '先在当前状态计算速度并执行 Euler 预测，再在终点计算速度，最后平均两次斜率。',
    'description_en': """Implement one explicit trapezoidal (Heun/RK2) ODE step for a flow-matching velocity field. x is (B,D), velocity_fn maps (state,time) to (B,D), and times are floats. Return (x_heun,x_euler).""",
    'description_zh': """为流匹配速度场实现一个显式梯形（Heun/RK2）ODE 步。x 为 (B,D)，velocity_fn 将 (状态,时间) 映射到 (B,D)，时间为浮点数。返回 (x_heun,x_euler)。""",
    'theory_en': """Euler uses the initial slope only. Heun predicts an endpoint with Euler and averages the initial and endpoint slopes, giving second-order local accuracy for smooth fields.""",
    'theory_zh': 'Euler 仅使用初始斜率。Heun 先用 Euler 预测终点，再平均初始与终点斜率，对光滑速度场达到二阶精度。',
    'diagram_en': r"""```text
x --v(t_curr)--> Euler predictor --v(t_next)-->
 \---------- average both slopes ----------> Heun state
```""",
    'diagram_zh': r"""```text
x --v(t_curr)--> Euler 预测 --v(t_next)-->
 \----------- 平均两次斜率 -----------> Heun 状态
```""",
    "tests": [
        {
            "name": 'shape and constant field',
            "code": """import torch
x = torch.randn(2,4); velocity = torch.full_like(x, 3.)
h, e = {fn}(x, lambda state, t: velocity, .2, .7)
assert h.shape == x.shape and e.shape == x.shape
assert torch.allclose(h, x+1.5) and torch.equal(h, e)""",
        },
        {
            "name": 'exact linear field reference',
            "code": """import torch
x = torch.tensor([[1.,2.],[-1.,3.]])
def vel(state, t): return 2*state+t
h, e = {fn}(x, vel, .25, .5)
dt=.25; v1=2*x+.25; ref_e=x+dt*v1; v2=2*ref_e+.5
assert torch.equal(e, ref_e) and torch.equal(h, x+.5*dt*(v1+v2))""",
        },
        {
            "name": 'zero step edge case',
            "code": """import torch
x = torch.randn(3,2); calls=[]
def vel(state, t):
    calls.append(t); return state.square()
h, e = {fn}(x, vel, 1.0, 1.0)
assert torch.equal(h, x) and torch.equal(e, x)
assert calls == [1.0, 1.0]""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2,3, requires_grad=True)
w = torch.randn(3,3, requires_grad=True)
def vel(state, t): return state @ w * t
h, e = {fn}(x, vel, .1, .6)
(h.sum()+e.sum()).backward()
assert x.grad is not None and w.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()""",
        },
    ],
    'solution': '''def flow_heun_step(x, velocity_fn, t_curr, t_next):
    """执行流匹配的 Heun 二阶 ODE 步 (Heun RK2 Step)。"""
    dt = t_next - t_curr

    # Step 1: 在当前状态计算斜率并执行 Euler 预测
    velocity_start = velocity_fn(x, t_curr)
    x_euler = x + dt * velocity_start

    # Step 2: 在预测终点重新评估速度场
    velocity_end = velocity_fn(x_euler, t_next)

    # Step 3: 平均起点与终点斜率得到二阶更新
    x_heun = x + 0.5 * dt * (velocity_start + velocity_end)
    return x_heun, x_euler''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 3)
x_heun, x_euler = flow_heun_step(x, lambda state, t: -state, 0.0, 0.1)
print("Heun output shape:", x_heun.shape)
print("Euler sample:", x_euler[0])
print("Heun sample:", x_heun[0])""",
    'explanation': r"""### 核心思路
Heun 法先用起点速度做 Euler 预测，再在预测终点计算第二个速度，并以 $x_{n+1}=x_n+\frac{\Delta t}{2}(v_n+v_{pred})$ 更新。

### 代码步骤拆解
- **Step 1（Euler 预测）**：在 `(x, t_curr)` 调用 `velocity_fn`，生成与 `x` 同形状的 `x_euler`。
- **Step 2（终点校正）**：以 `(x_euler, t_next)` 再调用一次速度场，获得 `velocity_end`。
- **Step 3（梯形积分）**：平均两次斜率并乘 `dt`，返回二阶 `x_heun` 与一阶预测作对照。

### 重点与易错点分析
- **注意点 1（第二次输入）**：终点速度必须在 `x_euler` 上计算，若仍使用原状态就失去二阶校正。
- **注意点 2（时间方向）**：`dt` 可正可负，应直接使用 `t_next - t_curr`，这样同一实现支持正向和反向积分。
- **注意点 3（零步长）**：即使 `dt=0`，按题意仍调用速度场两次，但两个返回状态都严格等于输入。

### 复杂度分析
- **时间复杂度**：$O(2C_v+N)$，其中 $C_v$ 是一次速度网络前向成本，主要瓶颈是两次模型评估。
- **空间复杂度**：$O(N+A_v)$，保存 Euler 状态、两组速度及速度网络激活。""",
}
