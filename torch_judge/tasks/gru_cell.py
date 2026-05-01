"""GRU Cell task."""

TASK = {
    "title": "GRU Cell",
    "title_zh": "GRU 单元",
    "difficulty": "Medium",
    "category": "基础网络组件",
    "description_en": "Implement a single GRU cell step. Given input x_t and previous hidden state h_{t-1}, compute: update gate z = σ(W_z·[h,x] + b_z), reset gate r = σ(W_r·[h,x] + b_r), candidate h~ = tanh(W_h·[r⊙h, x] + b_h). Then h_t = (1-z)⊙h_{t-1} + z⊙h~.",
    "description_zh": "实现单个 GRU 单元的前向步。给定输入 x_t 和前一时刻隐状态 h_{t-1}，计算：更新门 z = σ(W_z·[h,x] + b_z)，重置门 r = σ(W_r·[h,x] + b_r)，候选值 h~ = tanh(W_h·[r⊙h, x] + b_h)。然后 h_t = (1-z)⊙h_{t-1} + z⊙h~。",
    "function_name": "gru_cell",
    "hint": "Concatenate h and x for gates. For candidate, multiply h_prev by reset gate r before concatenating with x.",
    "hint_zh": "门控计算时拼接 h 和 x。候选值计算时，先将 h_prev 乘以重置门 r，再与 x 拼接。",
    "theory_en": "GRU simplifies LSTM by merging cell state and hidden state into one vector, using only two gates (update and reset). It has fewer parameters and trains faster while maintaining similar performance.",
    "theory_zh": "GRU 将 LSTM 的细胞状态和隐状态合并为一个向量，仅使用两个门（更新门和重置门）。参数量更少，训练更快，性能相似。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch
import torch.nn as nn

def gru_cell(x_t: torch.Tensor, h_prev: torch.Tensor,
             input_size: int, hidden_size: int) -> torch.Tensor:
    """
    Single GRU step.
    x_t: (B, input_size)
    h_prev: (B, hidden_size)
    Returns: h_t (B, hidden_size)
    """
    # 拼接隐状态和输入，用于计算门控
    # Concatenate hidden state and input for gate computation
    hx = torch.cat([h_prev, x_t], dim=-1)  # (B, hidden_size + input_size)

    # 更新门：决定保留多少旧状态
    # Update gate: how much old state to keep
    W_z = nn.Linear(hidden_size + input_size, hidden_size)
    z_t = torch.sigmoid(W_z(hx))

    # 重置门：决定遗忘多少旧状态
    # Reset gate: how much old state to forget
    W_r = nn.Linear(hidden_size + input_size, hidden_size)
    r_t = torch.sigmoid(W_r(hx))

    # 候选隐状态：重置后的旧状态 + 新输入
    # Candidate hidden state: reset old state + new input
    hx_reset = torch.cat([r_t * h_prev, x_t], dim=-1)
    W_h = nn.Linear(hidden_size + input_size, hidden_size)
    h_tilde = torch.tanh(W_h(hx_reset))

    # 更新隐状态：旧状态的(1-z)部分 + 新候选的z部分
    # Update hidden state: (1-z) of old + z of new candidate
    h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

    return h_t

    
    ''',
    "demo": "",
}
