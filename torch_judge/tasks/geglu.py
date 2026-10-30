"""GeGLU Feed-Forward Network task."""

TASK = {
    'title': 'GeGLU Feed-Forward Network',
    'title_zh': 'GeGLU 前馈网络',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'geglu',
    'starter_code': """def geglu(x, W_gate, W_up, W_down):
    pass""",
    'hint': """Apply GELU to the gate projection, multiply it elementwise by the up projection, then project back down.""",
    'hint_zh': '对门投影应用 GELU，与上投影逐元素相乘，再投影回模型维度。',
    'description_en': 'Implement a bias-free GeGLU feed-forward network using gate, up, and down weight matrices.',
    'description_zh': '使用门投影、上投影和下投影权重矩阵实现无偏置 GeGLU 前馈网络。',
    'theory_en': """GeGLU is a gated linear unit whose nonlinear branch uses GELU. Multiplicative gating gives the feed-forward block richer feature interactions than a single activation.""",
    'theory_zh': 'GeGLU 是非线性分支使用 GELU 的门控线性单元。乘法门控比单一激活提供更丰富的特征交互。',
    'diagram_en': """```text
x -> W_gate -> GELU --*---> W_down -> output
x -> W_up ------------/
```""",
    'diagram_zh': """```text
x -> W_gate -> GELU --*---> W_down -> 输出
x -> W_up ------------/
```""",
    "tests": [
        {
            "name": 'arbitrary leading dimensions',
            "code": """import torch
x = torch.randn(2, 3, 5)
out = {fn}(x, torch.randn(5, 9), torch.randn(5, 9), torch.randn(9, 5))
assert out.shape == (2, 3, 5)""",
        },
        {
            "name": 'exact numerical formula',
            "code": """import torch
import torch.nn.functional as F
torch.manual_seed(0)
x = torch.randn(4, 6)
wg, wu, wd = torch.randn(6, 10), torch.randn(6, 10), torch.randn(10, 6)
out = {fn}(x, wg, wu, wd)
ref = (F.gelu(x @ wg) * (x @ wu)) @ wd
assert torch.equal(out, ref)""",
        },
        {
            "name": 'zero up projection edge',
            "code": """import torch
x = torch.randn(3, 4)
out = {fn}(x, torch.randn(4, 7), torch.zeros(4, 7), torch.randn(7, 4))
assert torch.count_nonzero(out) == 0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2, 4, requires_grad=True)
wg = torch.randn(4, 8, requires_grad=True)
wu = torch.randn(4, 8, requires_grad=True)
wd = torch.randn(8, 4, requires_grad=True)
{fn}(x, wg, wu, wd).square().sum().backward()
assert all(t.grad is not None for t in (x, wg, wu, wd))""",
        },
    ],
    'solution': '''def geglu(x, W_gate, W_up, W_down):
    """计算门控 GELU 前馈网络 (GeGLU)。"""
    # Step 1: 计算门分支和上投影分支
    gate = F.gelu(x @ W_gate)
    up = x @ W_up

    # Step 2: 逐元素门控隐藏特征
    hidden = gate * up

    # Step 3: 下投影恢复模型维度
    return hidden @ W_down''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 4, 8
d_ff = 12
x = torch.randn(B, S, D)
W_gate = torch.randn(D, d_ff)
W_up = torch.randn(D, d_ff)
W_down = torch.randn(d_ff, D)
out = geglu(x, W_gate, W_up, W_down)
print("Output shape:", out.shape)
print("Output mean:", out.mean().item())
print("Sample values:", out[0, 0, :4])""",
    'explanation': r"""### 核心思路
门分支计算 $\operatorname{GELU}(xW_g)$，与上投影 $xW_u$ 逐元素相乘后，经 $W_d$ 映射回模型维度。

### 代码步骤拆解
- **Step 1（双分支投影）**：`gate` 和 `up` 将最后一维从 `D` 变为 `d_ff`。
- **Step 2（门控交互）**：`gate * up` 形成逐元素乘性特征交互。
- **Step 3（维度恢复）**：`hidden @ W_down` 将最后一维恢复为 `D`。

### 重点与易错点分析
- **注意点 1（权重方向）**：权重布局为 `(输入维, 输出维)`，无需像 `nn.Linear.weight` 那样转置。
- **注意点 2（激活位置）**：GELU 只作用于门分支，对两支都激活会改变算法定义。
- **注意点 3（前导维兼容）**：矩阵乘法只作用于最后一维，因此可保留任意前导批次维。

### 复杂度分析
- **时间复杂度**：$O(MDd_{ff})$，其中 $M$ 为前导维元素数，矩阵乘法占主导。
- **空间复杂度**：$O(Md_{ff})$，保存两个投影和门控激活。""",
}
