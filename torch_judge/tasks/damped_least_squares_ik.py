"""Damped least-squares inverse kinematics and PD control task."""

TASK = {
    "title": "Robotics Damped Least Squares IK & Joint PD Control",
    "title_zh": "阻尼最小二乘逆运动学（DLS IK）与关节 PD 控制",
    "difficulty": "Medium",
    "category": "具身智能与控制",
    "function_name": "dls_ik_and_pd",
    "starter_code": '''def dls_ik_and_pd(J, error_x, lam, q, qd, q_des, qd_des, kp, kd):
    pass
''',
    "hint": "Solve the damped task-space M by M system before left-multiplying by J.T; compute proportional and derivative torque terms elementwise.",
    "hint_zh": "先求解带阻尼的任务空间 M×M 线性系统，再左乘 J.T；逐元素计算比例与微分力矩项。",
    "description_en": "Compute one damped least-squares inverse-kinematics joint step and a joint-space PD control torque.",
    "description_zh": "计算一步阻尼最小二乘逆运动学关节增量及关节空间 PD 控制力矩。",
    "theory_en": "Damping regularizes Jacobian inversion near singularities. PD control combines position-error restoring torque with velocity-error damping.",
    "theory_zh": "阻尼在雅可比矩阵接近奇异时正则化其求逆；PD 控制将位置误差恢复力矩与速度误差阻尼结合。",
    "diagram_en": "```text\nJ,error -> solve(JJ^T + lambda^2 I) -> J^T -> delta_q\njoint targets/current -> P + D -> torque\n```",
    "diagram_zh": "```text\nJ、误差 -> 求解(JJ^T + lambda^2 I) -> J^T -> delta_q\n关节目标/当前状态 -> P + D -> 力矩\n```",
    "tests": [
        {"name": "joint output shapes", "code": '''import torch
J = torch.randn(3, 5)
dq, tau = {fn}(J, torch.randn(3), .1, *[torch.randn(5) for _ in range(4)], 2., .5)
assert dq.shape == (5,) and tau.shape == (5,)
'''},
        {"name": "exact reference", "code": '''import torch
J = torch.tensor([[1., 2.], [0., 1.]])
e = torch.tensor([.5, -1.])
q, qd = torch.tensor([1., 2.]), torch.tensor([.1, -.2])
qdes, qddes = torch.tensor([2., 0.]), torch.tensor([0., .3])
dq, tau = {fn}(J, e, .2, q, qd, qdes, qddes, torch.tensor([3., 4.]), .5)
ref_dq = J.T @ torch.linalg.solve(J @ J.T + .2 ** 2 * torch.eye(2), e)
ref_tau = torch.tensor([3., 4.]) * (qdes - q) + .5 * (qddes - qd)
assert torch.allclose(dq, ref_dq) and torch.allclose(tau, ref_tau)
'''},
        {"name": "zero Jacobian edge case", "code": '''import torch
z = torch.zeros(2)
dq, tau = {fn}(torch.zeros(2, 3), torch.ones(2), 1., torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), 2., 3.)
assert torch.equal(dq, torch.zeros(3)) and torch.equal(tau, torch.zeros(3))
'''},
        {"name": "gradient flow", "code": '''import torch
J = torch.randn(2, 3, requires_grad=True)
e = torch.randn(2, requires_grad=True)
q = torch.randn(3, requires_grad=True)
qd = torch.randn(3, requires_grad=True)
dq, tau = {fn}(J, e, .3, q, qd, torch.zeros(3), torch.zeros(3), 2., .5)
(dq.sum() + tau.sum()).backward()
assert J.grad is not None and e.grad is not None and q.grad is not None and qd.grad is not None
'''},
    ],
    "solution": '''import torch

def dls_ik_and_pd(J, error_x, lam, q, qd, q_des, qd_des, kp, kd):
    task_eye = torch.eye(J.shape[0], device=J.device, dtype=J.dtype)
    delta_q = J.T @ torch.linalg.solve(J @ J.T + (lam ** 2) * task_eye, error_x)
    torque = kp * (q_des - q) + kd * (qd_des - qd)
    return delta_q, torque
''',
    "demo": '''J = torch.randn(6, 7)
delta_q, torque = dls_ik_and_pd(J, torch.randn(6), 0.05, torch.zeros(7), torch.zeros(7), torch.ones(7), torch.zeros(7), 100., 10.)
print(delta_q, torque)
''',
    "explanation": "The task-space form solves an M-dimensional system and then maps back to N joints. Squared damping keeps the system invertible for nonzero lambda.",
}
