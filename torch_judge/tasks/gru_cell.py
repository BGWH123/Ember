"""GRU Cell task."""

TASK = {
    'title': 'GRU Cell',
    'title_zh': 'GRU 单元',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'gru_cell',
    'starter_code': """def gru_cell(x_t, h_prev, input_size, hidden_size):
    pass""",
    'hint': """Concatenate h and x for gates. For candidate, multiply h_prev by reset gate r before concatenating with x. Create the gate projections with nn.Linear; tests supply fixed weights to verify the recurrence independently of random initialization.""",
    'hint_zh': '门控计算时拼接 h 和 x。候选值计算时，先将 h_prev 乘以重置门 r，再与 x 拼接。 使用 nn.Linear 创建门控投影；测试会提供固定权重，避免随机初始化影响递推公式的验证。',
    'description_en': """Implement a single GRU cell step. Given input x_t and previous hidden state h_{t-1}, compute: update gate z = σ(W_z·[h,x] + b_z), reset gate r = σ(W_r·[h,x] + b_r), candidate h~ = tanh(W_h·[r⊙h, x] + b_h). Then h_t = (1-z)⊙h_{t-1} + z⊙h~.""",
    'description_zh': """实现单个 GRU 单元的前向步。给定输入 x_t 和前一时刻隐状态 h_{t-1}，计算：更新门 z = σ(W_z·[h,x] + b_z)，重置门 r = σ(W_r·[h,x] + b_r)，候选值 h~ = tanh(W_h·[r⊙h, x] + b_h)。然后 h_t = (1-z)⊙h_{t-1} + z⊙h~。""",
    'theory_en': """GRU simplifies LSTM by merging cell state and hidden state into one vector, using only two gates (update and reset). It has fewer parameters and trains faster while maintaining similar performance.""",
    'theory_zh': 'GRU 将 LSTM 的细胞状态和隐状态合并为一个向量，仅使用两个门（更新门和重置门）。参数量更少，训练更快，性能相似。',
    "tests": [
        {
            "name": 'Gate equations B2 I3 H4',
            "code": """original_linear = nn.Linear
def fixed_linear(*args, **kwargs):
    layer = original_linear(*args, **kwargs)
    with torch.no_grad():
        layer.weight.fill_(0.2)
        if layer.bias is not None:
            layer.bias.fill_(0.1)
    return layer

x = torch.linspace(-0.8, 0.7, 6).reshape(2, 3).requires_grad_()
h = torch.linspace(-0.5, 0.9, 8).reshape(2, 4).requires_grad_()
gate = torch.sigmoid(0.2 * (h.sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1)
candidate = torch.tanh(0.2 * ((gate * h).sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1)
expected = (1 - gate) * h + gate * candidate
nn.Linear = fixed_linear
try:
    out = {fn}(x, h, 3, 4)
finally:
    nn.Linear = original_linear
assert out.shape == h.shape
assert torch.allclose(out, expected, atol=1e-6)
gx, gh = torch.autograd.grad(out.sum(), (x, h), retain_graph=True)
rx, rh = torch.autograd.grad(expected.sum(), (x, h))
assert torch.allclose(gx, rx, atol=1e-6)
assert torch.allclose(gh, rh, atol=1e-6)""",
        },
        {
            "name": 'Gate equations B1 I2 H2',
            "code": """original_linear = nn.Linear
def fixed_linear(*args, **kwargs):
    layer = original_linear(*args, **kwargs)
    with torch.no_grad():
        layer.weight.fill_(0.2)
        if layer.bias is not None:
            layer.bias.fill_(0.1)
    return layer

x = torch.linspace(-0.8, 0.7, 2).reshape(1, 2).requires_grad_()
h = torch.linspace(-0.5, 0.9, 2).reshape(1, 2).requires_grad_()
gate = torch.sigmoid(0.2 * (h.sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1)
candidate = torch.tanh(0.2 * ((gate * h).sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1)
expected = (1 - gate) * h + gate * candidate
nn.Linear = fixed_linear
try:
    out = {fn}(x, h, 2, 2)
finally:
    nn.Linear = original_linear
assert out.shape == h.shape
assert torch.allclose(out, expected, atol=1e-6)
gx, gh = torch.autograd.grad(out.sum(), (x, h), retain_graph=True)
rx, rh = torch.autograd.grad(expected.sum(), (x, h))
assert torch.allclose(gx, rx, atol=1e-6)
assert torch.allclose(gh, rh, atol=1e-6)""",
        },
    ],
    'solution': '''def gru_cell(x_t, h_prev, input_size, hidden_size):
    """执行单步门控循环单元计算 (GRU Cell)。"""
    # Step 1: 拼接输入与旧隐状态并计算两个门
    combined = torch.cat((h_prev, x_t), dim=-1)
    update_layer = nn.Linear(hidden_size + input_size, hidden_size)
    reset_layer = nn.Linear(hidden_size + input_size, hidden_size)
    update_gate = torch.sigmoid(update_layer(combined))
    reset_gate = torch.sigmoid(reset_layer(combined))

    # Step 2: 使用重置门构造候选隐状态
    candidate_input = torch.cat((reset_gate * h_prev, x_t), dim=-1)
    candidate_layer = nn.Linear(hidden_size + input_size, hidden_size)
    candidate = torch.tanh(candidate_layer(candidate_input))

    # Step 3: 用更新门插值旧状态与候选状态
    return (1.0 - update_gate) * h_prev + update_gate * candidate''',
    'demo': """torch.manual_seed(42)
B, input_size, hidden_size = 2, 3, 4
x_t = torch.randn(B, input_size)
h_prev = torch.randn(B, hidden_size)
h_t = gru_cell(x_t, h_prev, input_size, hidden_size)
print("Output shape:", h_t.shape)
print("Hidden norm:", h_t.norm().item())
print("Sample values:", h_t[0])""",
    'explanation': r"""### 核心思路
GRU 用更新门 $z$ 和重置门 $r$ 控制状态，先计算候选 $\tilde h=\tanh(W_h[r\odot h,x])$，再按 $(1-z)h+z\tilde h$ 插值。

### 代码步骤拆解
- **Step 1（门控计算）**：`combined` 为 `(B, hidden_size+input_size)`，两个 sigmoid 门均为 `(B, hidden_size)`。
- **Step 2（候选状态）**：重置后的旧状态与输入拼接，再通过 tanh 投影。
- **Step 3（状态更新）**：逐元素插值得到与 `h_prev` 同形的 `h_t`。

### 重点与易错点分析
- **注意点 1（重置门位置）**：`reset_gate` 只作用于候选分支中的旧隐状态。
- **注意点 2（更新约定）**：题目采用 $(1-z)h+z\tilde h$，与部分库中的门定义方向可能相反。
- **注意点 3（层创建语义）**：本题要求函数内创建三个 `nn.Linear`，测试会替换构造器验证公式。

### 复杂度分析
- **时间复杂度**：$O(B(H+I)H)$，三次线性投影占主导。
- **空间复杂度**：$O(BH+(H+I)H)$，包括门激活与临时层参数。""",
}
