"""Linear Kalman Filter Predict & Update Step task."""

TASK = {
    'title': 'Linear Kalman Filter Predict & Update Step',
    'title_zh': '线性卡尔曼滤波预测与更新步',
    'difficulty': 'Hard',
    'category': '具身智能与控制',
    'function_name': 'kalman_filter_step',
    'starter_code': """def kalman_filter_step(x, P, u, z, A, B, H, Q, R):
    pass""",
    'hint': """First propagate state and covariance. Build innovation covariance from the prediction, compute the Kalman gain, then correct the state and covariance.""",
    'hint_zh': '先传播状态和协方差；由预测构造新息协方差并计算卡尔曼增益，最后修正状态与协方差。',
    'description_en': """Implement one linear Kalman-filter prediction and measurement-update step, returning the posterior state, covariance, and Kalman gain.""",
    'description_zh': '实现一次线性卡尔曼滤波预测与测量更新，返回后验状态、协方差和卡尔曼增益。',
    'theory_en': """The prediction propagates uncertainty through linear dynamics and adds process noise. The gain balances predicted uncertainty against measurement noise when applying the innovation.""",
    'theory_zh': '预测步骤通过线性动力学传播不确定性并加入过程噪声；更新时增益在预测不确定性与测量噪声之间权衡并应用新息。',
    'diagram_en': """```text
x,P,u -> predict x-,P- -> innovation S -> gain K -> update x+,P+
```""",
    'diagram_zh': """```text
x、P、u -> 预测 x-、P- -> 新息 S -> 增益 K -> 更新 x+、P+
```""",
    "tests": [
        {
            "name": 'return shapes',
            "code": """import torch
n, m, k = 3, 2, 1
x, P, K = {fn}(torch.zeros(n), torch.eye(n), torch.zeros(m), torch.zeros(k), torch.eye(n), torch.randn(n, m), torch.randn(k, n), torch.eye(n), torch.eye(k))
assert x.shape == (n,) and P.shape == (n, n) and K.shape == (n, k)""",
        },
        {
            "name": 'exact scalar reference',
            "code": """import torch
x0 = torch.tensor([2.])
P0 = torch.tensor([[4.]])
args = (torch.tensor([1.]), torch.tensor([5.]), torch.tensor([[1.]]), torch.tensor([[2.]]), torch.tensor([[1.]]), torch.tensor([[1.]]), torch.tensor([[3.]]))
x, P, K = {fn}(x0, P0, *args)
xpred = torch.tensor([4.])
Ppred = torch.tensor([[5.]])
Kref = torch.tensor([[5/8]])
assert torch.allclose(K, Kref) and torch.allclose(x, xpred + Kref @ torch.tensor([1.])) and torch.allclose(P, torch.tensor([[15/8]]))""",
        },
        {
            "name": 'uninformative measurement edge case',
            "code": """import torch
x0 = torch.tensor([1., -2.])
P0 = torch.eye(2)
x, P, K = {fn}(x0, P0, torch.tensor([.5]), torch.tensor([99.]), torch.eye(2), torch.tensor([[1.], [2.]]), torch.zeros(1, 2), torch.zeros(2, 2), torch.eye(1))
assert torch.equal(K, torch.zeros(2, 1)) and torch.allclose(x, torch.tensor([1.5, -1.])) and torch.equal(P, P0)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x0 = torch.randn(2, requires_grad=True)
P0 = torch.eye(2, requires_grad=True)
A = torch.tensor([[1., .1], [0., 1.]], requires_grad=True)
x, P, K = {fn}(x0, P0, torch.ones(1), torch.randn(1), A, torch.ones(2, 1), torch.tensor([[1., 0.]]), torch.eye(2) * .1, torch.eye(1))
(x.sum() + P.sum() + K.sum()).backward()
assert x0.grad is not None and P0.grad is not None and A.grad is not None and torch.isfinite(A.grad).all()""",
        },
    ],
    'solution': '''def kalman_filter_step(x, P, u, z, A, B, H, Q, R):
    """执行线性卡尔曼滤波预测与更新 (Kalman Filter Step)。"""
    # Step 1: 传播状态均值与协方差
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Q

    # Step 2: 计算新息协方差和卡尔曼增益
    innovation_cov = H @ P_pred @ H.T + R
    kalman_gain = P_pred @ H.T @ torch.linalg.inv(innovation_cov)

    # Step 3: 使用观测新息修正后验状态
    innovation = z - H @ x_pred
    x_new = x_pred + kalman_gain @ innovation
    identity = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    P_new = (identity - kalman_gain @ H) @ P_pred
    return x_new, P_new, kalman_gain''',
    'demo': """torch.manual_seed(42)
x = torch.zeros(2)
P = torch.eye(2)
A = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
B = torch.zeros(2, 1)
H = torch.tensor([[1.0, 0.0]])
x_new, P_new, gain = kalman_filter_step(x, P, torch.zeros(1), torch.tensor([1.0]), A, B, H, torch.eye(2) * 0.01, torch.eye(1) * 0.1)
print("State shape:", x_new.shape)
print("Updated state:", x_new)
print("Kalman gain:", gain)""",
    'explanation': """### 核心思路
先按线性动力学预测 $(x^-,P^-)$，再以 $K=P^-H^T(HP^-H^T+R)^{-1}$ 平衡预测与观测，完成后验修正。

### 代码步骤拆解
- **Step 1（预测）**：`x_pred` 为状态向量，`P_pred` 传播旧协方差并加入过程噪声。
- **Step 2（增益）**：`innovation_cov` 位于测量空间，`kalman_gain` 形状为 `(state_dim, measurement_dim)`。
- **Step 3（更新）**：观测残差修正均值，$(I-KH)P^-$ 修正协方差。

### 重点与易错点分析
- **注意点 1（矩阵顺序）**：矩阵乘法不可交换，尤其是增益中的 `P_pred @ H.T`。
- **注意点 2（数值求解）**：显式 `inv` 符合参考公式；工程中通常用线性求解提高稳定性。
- **注意点 3（设备一致性）**：单位阵必须继承 `P` 的设备和 dtype。

### 复杂度分析
- **时间复杂度**：$O(n^3+m^3)$，协方差传播和新息矩阵求逆占主导。
- **空间复杂度**：$O(n^2+m^2)$，保存预测协方差与新息协方差。""",
}
