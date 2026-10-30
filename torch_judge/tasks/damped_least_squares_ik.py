"""Robotics Damped Least Squares IK & Joint PD Control task."""

TASK = {
    'title': 'Robotics Damped Least Squares IK & Joint PD Control',
    'title_zh': '阻尼最小二乘逆运动学（DLS IK）与关节 PD 控制',
    'difficulty': 'Medium',
    'category': '具身智能与控制',
    'function_name': 'dls_ik_and_pd',
    'starter_code': """def dls_ik_and_pd(J, error_x, lam, q, qd, q_des, qd_des, kp, kd):
    pass""",
    'hint': """Solve the damped task-space M by M system before left-multiplying by J.T; compute proportional and derivative torque terms elementwise.""",
    'hint_zh': '先求解带阻尼的任务空间 M×M 线性系统，再左乘 J.T；逐元素计算比例与微分力矩项。',
    'description_en': 'Compute one damped least-squares inverse-kinematics joint step and a joint-space PD control torque.',
    'description_zh': '计算一步阻尼最小二乘逆运动学关节增量及关节空间 PD 控制力矩。',
    'theory_en': """Damping regularizes Jacobian inversion near singularities. PD control combines position-error restoring torque with velocity-error damping.""",
    'theory_zh': '阻尼在雅可比矩阵接近奇异时正则化其求逆；PD 控制将位置误差恢复力矩与速度误差阻尼结合。',
    'diagram_en': """```text
J,error -> solve(JJ^T + lambda^2 I) -> J^T -> delta_q
joint targets/current -> P + D -> torque
```""",
    'diagram_zh': """```text
J、误差 -> 求解(JJ^T + lambda^2 I) -> J^T -> delta_q
关节目标/当前状态 -> P + D -> 力矩
```""",
    "tests": [
        {
            "name": 'joint output shapes',
            "code": """import torch
J = torch.randn(3, 5)
dq, tau = {fn}(J, torch.randn(3), .1, *[torch.randn(5) for _ in range(4)], 2., .5)
assert dq.shape == (5,) and tau.shape == (5,)""",
        },
        {
            "name": 'exact reference',
            "code": """import torch
J = torch.tensor([[1., 2.], [0., 1.]])
e = torch.tensor([.5, -1.])
q, qd = torch.tensor([1., 2.]), torch.tensor([.1, -.2])
qdes, qddes = torch.tensor([2., 0.]), torch.tensor([0., .3])
dq, tau = {fn}(J, e, .2, q, qd, qdes, qddes, torch.tensor([3., 4.]), .5)
ref_dq = J.T @ torch.linalg.solve(J @ J.T + .2 ** 2 * torch.eye(2), e)
ref_tau = torch.tensor([3., 4.]) * (qdes - q) + .5 * (qddes - qd)
assert torch.allclose(dq, ref_dq) and torch.allclose(tau, ref_tau)""",
        },
        {
            "name": 'zero Jacobian edge case',
            "code": """import torch
z = torch.zeros(2)
dq, tau = {fn}(torch.zeros(2, 3), torch.ones(2), 1., torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), 2., 3.)
assert torch.equal(dq, torch.zeros(3)) and torch.equal(tau, torch.zeros(3))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
J = torch.randn(2, 3, requires_grad=True)
e = torch.randn(2, requires_grad=True)
q = torch.randn(3, requires_grad=True)
qd = torch.randn(3, requires_grad=True)
dq, tau = {fn}(J, e, .3, q, qd, torch.zeros(3), torch.zeros(3), 2., .5)
(dq.sum() + tau.sum()).backward()
assert J.grad is not None and e.grad is not None and q.grad is not None and qd.grad is not None""",
        },
    ],
    'solution': '''def dls_ik_and_pd(J, error_x, lam, q, qd, q_des, qd_des, kp, kd):
    """计算阻尼最小二乘逆运动学与关节 PD 控制 (DLS IK and PD Control)。"""
    # Step 1: 在任务空间构造带阻尼的正规方程
    task_dim = J.shape[0]
    identity = torch.eye(task_dim, device=J.device, dtype=J.dtype)
    regularized_system = J @ J.T + lam ** 2 * identity

    # Step 2: 求解任务空间增量并映射回关节空间
    delta_q = J.T @ torch.linalg.solve(regularized_system, error_x)

    # Step 3: 逐关节组合比例项与微分项
    torque = kp * (q_des - q) + kd * (qd_des - qd)
    return delta_q, torque''',
    'demo': """torch.manual_seed(42)
J = torch.randn(6, 7)
delta_q, torque = dls_ik_and_pd(J, torch.randn(6), 0.05, torch.zeros(7), torch.zeros(7), torch.ones(7), torch.zeros(7), 100.0, 10.0)
print("Joint update shape:", delta_q.shape)
print("Torque sample:", torque[:3])""",
    'explanation': r"""### 核心思路
阻尼最小二乘使用 $\Delta q=J^\top(JJ^\top+\lambda^2I)^{-1}e$ 稳定求解逆运动学，PD 控制则计算 $\tau=k_p(q_d-q)+k_d(\dot q_d-\dot q)$。

### 代码步骤拆解
- **Step 1（阻尼系统）**：由 `(M, N)` 的 `J` 构造 `(M, M)` 的 `regularized_system`。
- **Step 2（关节增量）**：先求解任务空间线性系统，再左乘 `J.T` 得到 `(N,)` 的 `delta_q`。
- **Step 3（PD 力矩）**：对位置误差与速度误差逐元素加权，得到 `(N,)` 的 `torque`。

### 重点与易错点分析
- **注意点 1（数值稳定性）**：$\lambda^2I$ 可缓解奇异位形；应使用 `torch.linalg.solve`，避免显式求逆放大误差。
- **注意点 2（设备与精度）**：单位阵必须与 `J` 的 `device`、`dtype` 一致。
- **注意点 3（计算维度）**：任务空间形式求解 $M\times M$ 系统，当 $M<N$ 时通常比关节空间形式更省计算。

### 复杂度分析
- **时间复杂度**：$O(M^2N+M^3)$，主要瓶颈是构造 $JJ^\top$ 与线性求解。
- **空间复杂度**：$O(M^2+N)$，主要保存正则化任务空间矩阵和输出向量。""",
}
