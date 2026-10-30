"""Linear Regression task."""

TASK = {
    'title': 'Linear Regression',
    'title_zh': '线性回归',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'LinearRegression',
    'starter_code': """class LinearRegression:
    pass""",
    'hint': """closed_form: augment X with ones col → `lstsq(X_aug, y)` → split `w, b`
gradient_descent: `grad_w = (2/N) * X.T @ (X@w+b - y)`, `w -= lr*grad_w`
nn_linear: `nn.Linear(D,1)` + `MSELoss` + `optimizer.step()` loop""",
    'hint_zh': """closed_form：X 增加全 1 列 → `lstsq(X_aug, y)` → 拆分 `w, b`
gradient_descent：`grad_w = (2/N) * X.T @ (X@w+b - y)`，`w -= lr*grad_w`
nn_linear：`nn.Linear(D,1)` + `MSELoss` + `optimizer.step()` 循环""",
    'description_en': """Implement linear regression three ways: closed-form, gradient descent, and nn.Linear.

This task covers the fundamental approaches to fitting a linear model, from the normal equation to autograd-based training.

**Signature:** `LinearRegression()` (class)

**Methods:**
- `closed_form(X, y) -> (w, b)` — solve via normal equation
- `gradient_descent(X, y, lr, steps) -> (w, b)` — manual gradient updates
- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd training loop

**Constraints:**
- All three methods should converge to similar weights
- Closed-form should not use autograd
- X is (N, D), y is (N,), returns w (D,) and b (scalar)""",
    'description_zh': """用三种方式实现线性回归：闭式解、梯度下降和 nn.Linear。

本题涵盖拟合线性模型的基本方法，从正规方程到基于 autograd 的训练。

**签名:** `LinearRegression()`（类）

**方法:**
- `closed_form(X, y) -> (w, b)` — 通过正规方程求解
- `gradient_descent(X, y, lr, steps) -> (w, b)` — 手动梯度更新
- `nn_linear(X, y, lr, steps) -> (w, b)` — PyTorch autograd 训练循环

**约束:**
- 三种方法应收敛到相近的权重
- 闭式解不应使用 autograd
- X 为 (N, D)，y 为 (N,)，返回 w (D,) 和 b（标量）""",
    "tests": [
        {
            "name": 'Closed-form returns correct shapes',
            "code": """import torch
torch.manual_seed(42)
X = torch.randn(50, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0 + torch.randn(50) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'""",
        },
        {
            "name": 'Closed-form finds correct weights',
            "code": """import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert torch.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert torch.allclose(b, torch.tensor(true_b), atol=1e-4), f'b: {b.item():.4f} vs true: {true_b}'""",
        },
        {
            "name": 'Gradient descent converges',
            "code": """import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'GD b: {b.item():.4f} vs true: {true_b}'""",
        },
        {
            "name": 'nn.Linear approach works',
            "code": """import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'nn b: {b.item():.4f} vs true: {true_b}'""",
        },
        {
            "name": 'All three methods agree',
            "code": """import torch
torch.manual_seed(0)
X = torch.randn(200, 2)
true_w = torch.tensor([1.5, -2.0])
y = X @ true_w + 1.0 + torch.randn(200) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert torch.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {(w_cf - w_gd).abs().max():.4f}'
assert torch.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {(w_cf - w_nn).abs().max():.4f}'
assert abs(b_cf.item() - b_gd.item()) < 0.15, f'Bias CF vs GD: {b_cf.item():.4f} vs {b_gd.item():.4f}'
assert abs(b_cf.item() - b_nn.item()) < 0.15, f'Bias CF vs NN: {b_cf.item():.4f} vs {b_nn.item():.4f}'""",
        },
        {
            "name": 'Closed-form uses no autograd',
            "code": """import torch
X = torch.randn(30, 2)
y = X @ torch.tensor([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert not w.requires_grad, 'Closed-form w should not require grad'""",
        },
    ],
    'solution': '''class LinearRegression:
    """提供三种线性回归求解方式 (Linear Regression)。"""

    def closed_form(self, X, y):
        # Step 1: 增广设计矩阵并用最小二乘求闭式解
        N, D = X.shape
        ones = torch.ones(N, 1, dtype=X.dtype, device=X.device)
        X_augmented = torch.cat((X, ones), dim=1)
        parameters = torch.linalg.lstsq(X_augmented, y).solution
        return parameters[:D].detach(), parameters[D].detach()

    def gradient_descent(self, X, y, lr=0.01, steps=1000):
        # Step 2: 按均方误差解析梯度迭代参数
        N, D = X.shape
        w = torch.zeros(D, dtype=X.dtype, device=X.device)
        b = torch.zeros((), dtype=X.dtype, device=X.device)
        for _ in range(steps):
            error = X @ w + b - y
            w = w - lr * (2.0 / N) * (X.T @ error)
            b = b - lr * (2.0 / N) * error.sum()
        return w, b

    def nn_linear(self, X, y, lr=0.01, steps=1000):
        # Step 3: 使用自动求导训练等价的 nn.Linear
        _, D = X.shape
        layer = nn.Linear(D, 1).to(dtype=X.dtype, device=X.device)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            predictions = layer(X).squeeze(-1)
            loss = F.mse_loss(predictions, y)
            loss.backward()
            optimizer.step()
        return layer.weight.detach().squeeze(0), layer.bias.detach().squeeze(0)''',
    'demo': """torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0
model = LinearRegression()
w, b = model.closed_form(X, y)
print("Weight shape:", w.shape)
print("Estimated weights:", w)
print("Estimated bias:", b.item())""",
    'explanation': r"""### 核心思路
线性回归拟合 $y=Xw+b$：闭式法对增广矩阵求最小二乘，手动法按 MSE 梯度更新，自动求导法训练 `nn.Linear`。三种接口都返回 `(D,)` 的 `w` 和标量 `b`。

### 代码步骤拆解
- **Step 1（闭式求解）**：给 `X` 拼接全一列并调用 `torch.linalg.lstsq`。
- **Step 2（手动迭代）**：由预测误差计算 `grad_w`、`grad_b` 并更新。
- **Step 3（自动求导）**：用 `F.mse_loss` 反传并由 SGD 更新线性层。

### 重点与易错点分析
- **注意点 1（数值稳定）**：`lstsq` 比显式求 $(X^TX)^{-1}$ 更稳定，也能处理秩亏。
- **注意点 2（设备类型）**：初始化张量和模块应跟随 `X` 的 dtype 与 device。
- **注意点 3（梯度隔离）**：闭式结果和训练后参数均 `.detach()`，避免暴露无意义的历史计算图。

### 复杂度分析
- **时间复杂度**：闭式法约 $O(ND^2+D^3)$；迭代法为 $O(steps\,ND)$。
- **空间复杂度**：$O(ND+D^2)$，迭代法额外状态为 $O(D)$。""",
}
