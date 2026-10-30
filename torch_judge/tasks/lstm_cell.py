"""LSTM Cell task."""

TASK = {
    'title': 'LSTM Cell',
    'title_zh': 'LSTM 单元',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'lstm_cell',
    'starter_code': """def lstm_cell(x_t, h_prev, c_prev, input_size, hidden_size):
    pass""",
    'hint': """Concatenate h and x along the last dimension. Use four separate linear projections, then apply sigmoid/tanh to the respective gates. Create the gate projections with nn.Linear; tests supply fixed weights to verify the recurrence independently of random initialization.""",
    'hint_zh': """沿最后一维拼接 h 和 x。使用四个独立的线性投影，然后对 respective 门应用 sigmoid/tanh。 使用 nn.Linear 创建门控投影；测试会提供固定权重，避免随机初始化影响递推公式的验证。""",
    'description_en': """Implement a single LSTM cell step. Given input x_t, previous hidden state h_{t-1}, and previous cell state c_{t-1}, compute: forget gate f = σ(W_f·[h,x] + b_f), input gate i = σ(W_i·[h,x] + b_i), candidate g = tanh(W_g·[h,x] + b_g), output gate o = σ(W_o·[h,x] + b_o). Then c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t).""",
    'description_zh': """实现单个 LSTM 单元的前向步。给定输入 x_t、前一时刻隐状态 h_{t-1} 和细胞状态 c_{t-1}，计算：遗忘门 f = σ(W_f·[h,x] + b_f)，输入门 i = σ(W_i·[h,x] + b_i)，候选值 g = tanh(W_g·[h,x] + b_g)，输出门 o = σ(W_o·[h,x] + b_o)。然后 c_t = f*c_{t-1} + i*g，h_t = o*tanh(c_t)。""",
    'theory_en': """LSTM uses gating mechanisms to control information flow, solving the vanishing gradient problem in vanilla RNNs. The cell state acts as a conveyor belt for long-term memory.""",
    'theory_zh': 'LSTM 使用门控机制控制信息流，解决了普通 RNN 的梯度消失问题。细胞状态充当长期记忆传送带。',
    "tests": [
        {
            "name": 'Cell and output equations B2 I3 H4',
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
c = torch.linspace(0.2, 1.2, 8).reshape(2, 4).requires_grad_()
affine = 0.2 * (h.sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1
gate = torch.sigmoid(affine)
expected_c = gate * c + gate * torch.tanh(affine)
expected_h = gate * torch.tanh(expected_c)
nn.Linear = fixed_linear
try:
    out_h, out_c = {fn}(x, h, c, 3, 4)
finally:
    nn.Linear = original_linear
assert out_h.shape == out_c.shape == h.shape
assert torch.allclose(out_h, expected_h, atol=1e-6)
assert torch.allclose(out_c, expected_c, atol=1e-6)
actual_grads = torch.autograd.grad((out_h + out_c).sum(), (x, h, c), retain_graph=True)
expected_grads = torch.autograd.grad((expected_h + expected_c).sum(), (x, h, c))
for actual, expected in zip(actual_grads, expected_grads):
    assert torch.allclose(actual, expected, atol=1e-6)""",
        },
        {
            "name": 'Cell and output equations B1 I2 H2',
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
c = torch.linspace(0.2, 1.2, 2).reshape(1, 2).requires_grad_()
affine = 0.2 * (h.sum(-1, keepdim=True) + x.sum(-1, keepdim=True)) + 0.1
gate = torch.sigmoid(affine)
expected_c = gate * c + gate * torch.tanh(affine)
expected_h = gate * torch.tanh(expected_c)
nn.Linear = fixed_linear
try:
    out_h, out_c = {fn}(x, h, c, 2, 2)
finally:
    nn.Linear = original_linear
assert out_h.shape == out_c.shape == h.shape
assert torch.allclose(out_h, expected_h, atol=1e-6)
assert torch.allclose(out_c, expected_c, atol=1e-6)
actual_grads = torch.autograd.grad((out_h + out_c).sum(), (x, h, c), retain_graph=True)
expected_grads = torch.autograd.grad((expected_h + expected_c).sum(), (x, h, c))
for actual, expected in zip(actual_grads, expected_grads):
    assert torch.allclose(actual, expected, atol=1e-6)""",
        },
    ],
    'solution': '''def lstm_cell(x_t, h_prev, c_prev, input_size, hidden_size):
    """执行单步长短期记忆递推 (LSTM Cell)。"""
    # Step 1: 拼接当前输入与上一时刻隐状态并创建四组门投影
    combined = torch.cat((h_prev, x_t), dim=-1)
    gate_input_size = hidden_size + input_size
    forget_proj = nn.Linear(gate_input_size, hidden_size)
    input_proj = nn.Linear(gate_input_size, hidden_size)
    candidate_proj = nn.Linear(gate_input_size, hidden_size)
    output_proj = nn.Linear(gate_input_size, hidden_size)

    # Step 2: 计算遗忘门、输入门、候选记忆和输出门
    forget_gate = torch.sigmoid(forget_proj(combined))
    input_gate = torch.sigmoid(input_proj(combined))
    candidate = torch.tanh(candidate_proj(combined))
    output_gate = torch.sigmoid(output_proj(combined))

    # Step 3: 更新细胞状态与隐状态
    c_t = forget_gate * c_prev + input_gate * candidate
    h_t = output_gate * torch.tanh(c_t)
    return h_t, c_t''',
    'demo': """torch.manual_seed(42)
B, input_size, hidden_size = 2, 3, 4
x_t = torch.randn(B, input_size)
h_prev = torch.randn(B, hidden_size)
c_prev = torch.randn(B, hidden_size)
h_t, c_t = lstm_cell(x_t, h_prev, c_prev, input_size, hidden_size)
print("Hidden shape:", h_t.shape)
print("Cell shape:", c_t.shape)
print("Sample hidden:", h_t[0])""",
    'explanation': r"""### 核心思路
LSTM 将 `h_prev` 与 `x_t` 拼接后计算四个门，按 $c_t=f_t\odot c_{t-1}+i_t\odot g_t$ 更新长期记忆，再由 $h_t=o_t\odot\tanh(c_t)$ 产生输出。

### 代码步骤拆解
- **Step 1（拼接投影）**：得到 `(B, hidden_size + input_size)` 的 `combined` 并建立四个线性层。
- **Step 2（门控激活）**：三个门用 sigmoid，候选记忆用 tanh。
- **Step 3（状态递推）**：逐元素组合旧记忆和新候选，返回 `h_t`、`c_t`。

### 重点与易错点分析
- **注意点 1（测试注入）**：本题测试会替换 `nn.Linear` 为固定层，因此四个投影必须在函数内独立创建。
- **注意点 2（门顺序）**：候选分支不能用 sigmoid，输出隐状态必须再对 `c_t` 做 tanh。
- **注意点 3（工程实现）**：真实模型通常把四个投影合并为一次矩阵乘法并将层参数持久化，本题接口则只验证单步公式。

### 复杂度分析
- **时间复杂度**：$O(B(I+H)H)$，四个线性投影占主导。
- **空间复杂度**：$O(BH)$，保存四个门及两个状态。""",
}
