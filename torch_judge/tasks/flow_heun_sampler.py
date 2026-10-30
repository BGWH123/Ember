"""Flow-matching Heun ODE step task."""

TASK = {
    "title": "Flow Matching Heun (2nd-Order RK2) ODE Step",
    "title_zh": "流匹配 Heun 二阶 ODE 采样步",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "flow_heun_step",
    "starter_code": '''def flow_heun_step(x, velocity_fn, t_curr, t_next):
    pass
''',
    "hint": "Evaluate velocity at the current state, make an Euler predictor, evaluate again at the endpoint, then average the two slopes.",
    "hint_zh": "先在当前状态计算速度并执行 Euler 预测，再在终点计算速度，最后平均两次斜率。",
    "description_en": "Implement one explicit trapezoidal (Heun/RK2) ODE step for a flow-matching velocity field. x is (B,D), velocity_fn maps (state,time) to (B,D), and times are floats. Return (x_heun,x_euler).",
    "description_zh": "为流匹配速度场实现一个显式梯形（Heun/RK2）ODE 步。x 为 (B,D)，velocity_fn 将 (状态,时间) 映射到 (B,D)，时间为浮点数。返回 (x_heun,x_euler)。",
    "theory_en": "Euler uses the initial slope only. Heun predicts an endpoint with Euler and averages the initial and endpoint slopes, giving second-order local accuracy for smooth fields.",
    "theory_zh": "Euler 仅使用初始斜率。Heun 先用 Euler 预测终点，再平均初始与终点斜率，对光滑速度场达到二阶精度。",
    "diagram_en": "```text\nx --v(t_curr)--> Euler predictor --v(t_next)-->\n \\---------- average both slopes ----------> Heun state\n```",
    "diagram_zh": "```text\nx --v(t_curr)--> Euler 预测 --v(t_next)-->\n \\----------- 平均两次斜率 -----------> Heun 状态\n```",
    "tests": [
        {
            "name": "shape and constant field",
            "code": '''import torch
x = torch.randn(2,4); velocity = torch.full_like(x, 3.)
h, e = {fn}(x, lambda state, t: velocity, .2, .7)
assert h.shape == x.shape and e.shape == x.shape
assert torch.allclose(h, x+1.5) and torch.equal(h, e)
''',
        },
        {
            "name": "exact linear field reference",
            "code": '''import torch
x = torch.tensor([[1.,2.],[-1.,3.]])
def vel(state, t): return 2*state+t
h, e = {fn}(x, vel, .25, .5)
dt=.25; v1=2*x+.25; ref_e=x+dt*v1; v2=2*ref_e+.5
assert torch.equal(e, ref_e) and torch.equal(h, x+.5*dt*(v1+v2))
''',
        },
        {
            "name": "zero step edge case",
            "code": '''import torch
x = torch.randn(3,2); calls=[]
def vel(state, t):
    calls.append(t); return state.square()
h, e = {fn}(x, vel, 1.0, 1.0)
assert torch.equal(h, x) and torch.equal(e, x)
assert calls == [1.0, 1.0]
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2,3, requires_grad=True)
w = torch.randn(3,3, requires_grad=True)
def vel(state, t): return state @ w * t
h, e = {fn}(x, vel, .1, .6)
(h.sum()+e.sum()).backward()
assert x.grad is not None and w.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
''',
        },
    ],
    "solution": '''def flow_heun_step(x, velocity_fn, t_curr, t_next):
    dt = t_next - t_curr
    v1 = velocity_fn(x, t_curr)
    x_euler = x + dt * v1
    v2 = velocity_fn(x_euler, t_next)
    x_heun = x + 0.5 * dt * (v1 + v2)
    return x_heun, x_euler
''',
    "demo": '''x = torch.randn(2, 3)
heun, euler = flow_heun_step(x, lambda state, t: -state, 0.0, 0.1)
print(heun, euler)
''',
    "explanation": "The returned Euler state is both the endpoint predictor used for the second velocity evaluation and a useful first-order comparison.",
}
