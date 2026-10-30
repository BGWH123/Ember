"""GeGLU feed-forward task."""

TASK = {
    "title": "GeGLU Feed-Forward Network",
    "title_zh": "GeGLU 前馈网络",
    "difficulty": "Easy",
    "category": "激活函数",
    "function_name": "geglu",
    "starter_code": '''def geglu(x, W_gate, W_up, W_down):
    pass
''',
    "hint": "Apply GELU to the gate projection, multiply it elementwise by the up projection, then project back down.",
    "hint_zh": "对门投影应用 GELU，与上投影逐元素相乘，再投影回模型维度。",
    "description_en": "Implement a bias-free GeGLU feed-forward network using gate, up, and down weight matrices.",
    "description_zh": "使用门投影、上投影和下投影权重矩阵实现无偏置 GeGLU 前馈网络。",
    "theory_en": "GeGLU is a gated linear unit whose nonlinear branch uses GELU. Multiplicative gating gives the feed-forward block richer feature interactions than a single activation.",
    "theory_zh": "GeGLU 是非线性分支使用 GELU 的门控线性单元。乘法门控比单一激活提供更丰富的特征交互。",
    "diagram_en": "```text\nx -> W_gate -> GELU --*---> W_down -> output\nx -> W_up ------------/\n```",
    "diagram_zh": "```text\nx -> W_gate -> GELU --*---> W_down -> 输出\nx -> W_up ------------/\n```",
    "tests": [
        {
            "name": "arbitrary leading dimensions",
            "code": '''import torch
x = torch.randn(2, 3, 5)
out = {fn}(x, torch.randn(5, 9), torch.randn(5, 9), torch.randn(9, 5))
assert out.shape == (2, 3, 5)
''',
        },
        {
            "name": "exact numerical formula",
            "code": '''import torch
import torch.nn.functional as F
torch.manual_seed(0)
x = torch.randn(4, 6)
wg, wu, wd = torch.randn(6, 10), torch.randn(6, 10), torch.randn(10, 6)
out = {fn}(x, wg, wu, wd)
ref = (F.gelu(x @ wg) * (x @ wu)) @ wd
assert torch.equal(out, ref)
''',
        },
        {
            "name": "zero up projection edge",
            "code": '''import torch
x = torch.randn(3, 4)
out = {fn}(x, torch.randn(4, 7), torch.zeros(4, 7), torch.randn(7, 4))
assert torch.count_nonzero(out) == 0
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2, 4, requires_grad=True)
wg = torch.randn(4, 8, requires_grad=True)
wu = torch.randn(4, 8, requires_grad=True)
wd = torch.randn(8, 4, requires_grad=True)
{fn}(x, wg, wu, wd).square().sum().backward()
assert all(t.grad is not None for t in (x, wg, wu, wd))
''',
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def geglu(x, W_gate, W_up, W_down):
    return (F.gelu(x @ W_gate) * (x @ W_up)) @ W_down
''',
    "demo": '''x = torch.randn(2, 128, 512)
out = geglu(x, torch.randn(512, 1365), torch.randn(512, 1365), torch.randn(1365, 512))
print(out.shape)
''',
    "explanation": "Both input projections preserve all leading dimensions and replace only the final feature size. Their gated product has `d_ff` features before the down projection restores `d_model`.",
}
