"""LSTM Cell task."""

TASK = {
    "title": "LSTM Cell",
    "title_zh": "LSTM 单元",
    "difficulty": "Medium",
    "category": "基础网络组件",
    "description_en": "Implement a single LSTM cell step. Given input x_t, previous hidden state h_{t-1}, and previous cell state c_{t-1}, compute: forget gate f = σ(W_f·[h,x] + b_f), input gate i = σ(W_i·[h,x] + b_i), candidate g = tanh(W_g·[h,x] + b_g), output gate o = σ(W_o·[h,x] + b_o). Then c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t).",
    "description_zh": "实现单个 LSTM 单元的前向步。给定输入 x_t、前一时刻隐状态 h_{t-1} 和细胞状态 c_{t-1}，计算：遗忘门 f = σ(W_f·[h,x] + b_f)，输入门 i = σ(W_i·[h,x] + b_i)，候选值 g = tanh(W_g·[h,x] + b_g)，输出门 o = σ(W_o·[h,x] + b_o)。然后 c_t = f*c_{t-1} + i*g，h_t = o*tanh(c_t)。",
    "function_name": "lstm_cell",
    "hint": "Concatenate h and x along the last dimension. Use four separate linear projections, then apply sigmoid/tanh to the respective gates.",
    "hint_zh": "沿最后一维拼接 h 和 x。使用四个独立的线性投影，然后对 respective 门应用 sigmoid/tanh。",
    "theory_en": "LSTM uses gating mechanisms to control information flow, solving the vanishing gradient problem in vanilla RNNs. The cell state acts as a conveyor belt for long-term memory.",
    "theory_zh": "LSTM 使用门控机制控制信息流，解决了普通 RNN 的梯度消失问题。细胞状态充当长期记忆传送带。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch
import torch.nn as nn

def lstm_cell(x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
              input_size: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single LSTM step.
    x_t: (B, input_size)
    h_prev, c_prev: (B, hidden_size)
    Returns: (h_t, c_t) both (B, hidden_size)
    """
    batch_size = x_t.shape[0]

    # 拼接前一时刻隐状态和当前输入
    # Concatenate previous hidden state and current input
    hx = torch.cat([h_prev, x_t], dim=-1)  # (B, hidden_size + input_size)

    # 四个门控的线性投影（实际中通常合并为一个大的矩阵乘法）
    # Linear projections for four gates (often merged into one big matmul)
    W_f = nn.Linear(hidden_size + input_size, hidden_size)
    W_i = nn.Linear(hidden_size + input_size, hidden_size)
    W_g = nn.Linear(hidden_size + input_size, hidden_size)
    W_o = nn.Linear(hidden_size + input_size, hidden_size)

    # 遗忘门：决定保留多少旧记忆
    # Forget gate: how much old memory to keep
    f_t = torch.sigmoid(W_f(hx))

    # 输入门：决定写入多少新信息
    # Input gate: how much new info to write
    i_t = torch.sigmoid(W_i(hx))

    # 候选记忆：新候选值
    # Candidate memory: new candidate values
    g_t = torch.tanh(W_g(hx))

    # 输出门：决定读出多少信息
    # Output gate: how much to read out
    o_t = torch.sigmoid(W_o(hx))

    # 更新细胞状态：遗忘旧记忆 + 写入新记忆
    # Update cell state: forget old + write new
    c_t = f_t * c_prev + i_t * g_t

    # 新的隐状态：输出门过滤后的细胞状态
    # New hidden state: cell state filtered by output gate
    h_t = o_t * torch.tanh(c_t)

    return h_t, c_t

    
    ''',
    "demo": "",
}
