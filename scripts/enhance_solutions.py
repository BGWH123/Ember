"""
Enhance all task solutions with detailed line-by-line comments
and rewrite explanations to strictly match the solution code.
"""
import json, re, ast
from pathlib import Path

TASKS_DIR = Path(r"d:\BGWH_Code\pyre-code-main\torch_judge\tasks")
WEB_LIB = Path(r"d:\BGWH_Code\pyre-code-main\web\src\lib")

# Mock torch for safe exec
class _MockTensor: pass
class _MockModule: pass
class _MockNn:
    Module = _MockModule
    Parameter = _MockTensor
    Linear = _MockModule
    LayerNorm = _MockModule
    Dropout = _MockModule
    Sequential = list
    Embedding = _MockModule
class _MockTorch:
    Tensor = _MockTensor
    nn = _MockNn()

# ------------------------------------------------------------------
# Per-task solution rewrites with detailed Chinese + English comments
# ------------------------------------------------------------------

def build_mha_solution():
    return '''class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) implementation.
    将输入投影到多个注意力头，分别计算缩放点积注意力，最后拼接并输出投影。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()                               # 调用父类 nn.Module 的初始化，注册所有子模块
        self.num_heads = num_heads                       # 注意力头数 H
        self.d_k = d_model // num_heads                  # 每个头的维度 d_k = d_model / H
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 四个线性投影层: W_q, W_k, W_v 分别投影 Q/K/V, W_o 合并多头输出
        self.W_q = nn.Linear(d_model, d_model)           # Q 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_k = nn.Linear(d_model, d_model)           # K 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_v = nn.Linear(d_model, d_model)           # V 投影: (B, S, d_model) -> (B, S, d_model)
        self.W_o = nn.Linear(d_model, d_model)           # 输出投影: 拼接后的 (B, S, d_model) -> (B, S, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape                              # B: batch size, S_q: query 序列长度
        S_k = K.shape[1]                                 # S_k: key/value 序列长度 (支持 cross-attention)

        # Step 1: 线性投影 + 多头拆分
        # view(B, S, H, d_k) 将最后一维拆分为 H 个头，每个头 d_k 维
        # transpose(1, 2) 交换维度得到 (B, H, S, d_k)，便于并行做矩阵乘法
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_q, d_k)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_k, d_k)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)   # (B, H, S_k, d_k)

        # Step 2: 计算缩放点积注意力分数
        # scores[b,h,i,j] = q[b,h,i] · k[b,h,j] / sqrt(d_k)
        # 除以 sqrt(d_k) 防止大 d_k 时 softmax 饱和（梯度消失）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)      # (B, H, S_q, S_k)

        # Step 3: Softmax 归一化得到注意力权重
        # dim=-1 表示沿 key 维度归一化，每行和为 1
        weights = torch.softmax(scores, dim=-1)          # (B, H, S_q, S_k)

        # Step 4: 加权求和: 注意力权重 @ V 得到上下文向量
        attn = torch.matmul(weights, v)                  # (B, H, S_q, d_k)

        # Step 5: 合并多头 + 输出投影
        # transpose(1,2): (B, H, S_q, d_k) -> (B, S_q, H, d_k)
        # contiguous(): 保证内存连续，因为 transpose 改变了 strides
        # view(B, S_q, -1): 将 H * d_k = d_model 合并回最后一维
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)                  # (B, S_q, d_model)
        return self.W_o(out)                             # 最终线性投影: (B, S_q, d_model) -> (B, S_q, d_model)
'''

def build_cross_attention_solution():
    return '''class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention implementation.
    解码器通过交叉注意力关注编码器输出。Q 来自解码器，K/V 来自编码器。
    与自注意力不同，交叉注意力不使用因果掩码（编码器输出完全可见）。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()                               # 调用父类 nn.Module 初始化
        self.num_heads = num_heads                       # 注意力头数
        self.d_k = d_model // num_heads                  # 每个头的维度

        # 独立的线性投影层
        self.W_q = nn.Linear(d_model, d_model)           # Q 投影 (来自解码器)
        self.W_k = nn.Linear(d_model, d_model)           # K 投影 (来自编码器)
        self.W_v = nn.Linear(d_model, d_model)           # V 投影 (来自编码器)
        self.W_o = nn.Linear(d_model, d_model)           # 输出投影

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape                            # B: batch, S_q: decoder 序列长度
        S_kv = x_kv.shape[1]                             # S_kv: encoder 序列长度 (可与 S_q 不同)

        # Q 来自解码器输入 x_q, K/V 来自编码器输入 x_kv
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)    # (B, H, S_q, d_k)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S_kv, d_k)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S_kv, d_k)

        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)         # (B, H, S_q, S_kv)

        # Softmax 归一化 (无因果掩码，编码器输出完全可见)
        weights = torch.softmax(scores, dim=-1)          # (B, H, S_q, S_kv)

        # 加权求和
        attn = torch.matmul(weights, v)                  # (B, H, S_q, d_k)

        # 合并多头并输出投影
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))         # (B, S_q, d_model)
'''

def build_adam_solution():
    return '''class MyAdam:
    """
    Adam Optimizer implementation.
    结合动量（一阶矩估计）和 RMSProp（二阶矩估计），通过偏差校正实现自适应学习率。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)                       # 待优化的参数列表
        self.lr = lr                                     # 学习率 η
        self.beta1, self.beta2 = betas                   # 一阶矩衰减 β1=0.9, 二阶矩衰减 β2=0.999
        self.eps = eps                                   # 数值稳定常数 ε=1e-8，防止除以零
        self.t = 0                                       # 时间步计数器，用于偏差校正

        # 初始化一阶矩 m 和二阶矩 v 为零张量
        # m_t: 梯度的一阶矩（动量），v_t: 梯度的二阶矩（RMSProp）
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """
        执行一步参数更新。
        更新公式:
            m_t = β1·m_{t-1} + (1-β1)·g_t
            v_t = β2·v_{t-1} + (1-β2)·g_t²
            m̂_t = m_t / (1-β1^t)   ← 偏差校正，早期步骤 m_t 偏向零
            v̂_t = v_t / (1-β2^t)   ← 偏差校正
            θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
        """
        self.t += 1                                      # 更新时间步
        with torch.no_grad():                            # 参数更新不需要梯度追踪
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue                             # 跳过无梯度的参数

                g = p.grad                               # 当前梯度 g_t

                # 更新一阶矩（动量估计）: m_t = β1·m_{t-1} + (1-β1)·g_t
                # 相当于指数移动平均，β1 控制历史梯度的衰减速度
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                # 更新二阶矩（梯度平方的移动平均）: v_t = β2·v_{t-1} + (1-β2)·g_t²
                # 用于自适应调整每个参数的学习率
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                # 偏差校正: m̂_t = m_t / (1 - β1^t)
                # 早期 t 较小时，(1-β1^t) 接近 0，m̂_t 会被放大，补偿初始化偏差
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # 偏差校正: v̂_t = v_t / (1 - β2^t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 参数更新: θ -= η · m̂ / (√v̂ + ε)
                # m̂ 提供更新方向（类似动量），√v̂ 提供自适应缩放（大梯度参数学习率更小）
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """将所有参数的梯度清零。在每次 backward() 前调用，防止梯度累积。"""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()                           # 原地置零，避免内存分配
'''

def build_layernorm_solution():
    return '''def my_layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization implementation.
    对每个样本沿特征维度独立归一化，再应用可学习的缩放 γ 和偏移 β。
    与 BatchNorm 不同，LayerNorm 不依赖 batch size，适合变长序列（Transformer）。
    """
    # 沿最后一个维度（特征维度）计算均值
    # x: (B, S, D) 或 (N, D)，mean 结果: (B, S, 1) 或 (N, 1)
    mean = x.mean(dim=-1, keepdim=True)                # μ = (1/D) Σ_i x_i

    # 计算方差（无偏估计，unbiased=False 对应 PyTorch 默认）
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # σ² = (1/D) Σ_i (x_i - μ)²

    # 标准化: (x - μ) / √(σ² + ε)
    # ε 防止除零，保证数值稳定
    x_norm = (x - mean) / torch.sqrt(var + eps)        # (B, S, D)，均值为0，方差为1

    # 可学习的仿射变换: γ * x_norm + β
    # γ 控制输出缩放，β 控制输出偏移。允许网络学习到"不需要归一化"（γ=1, β=0 即恒等）
    return gamma * x_norm + beta                       # (B, S, D)
'''

def build_rmsnorm_solution():
    return '''def rms_norm(x, weight, eps=1e-6):
    """
    RMS Normalization implementation (used in LLaMA, Mistral).
    相比 LayerNorm 省去了均值计算，仅使用均方根进行缩放，计算量减少约30%。
    """
    # 计算均方根 RMS(x) = √( (1/D) Σ_i x_i² )
    # 沿特征维度求均值后开方，结果形状与 x 的最后一个维度 broadcast 兼容
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)   # (B, S, 1)

    # 归一化: x / RMS(x)，再乘以可学习的缩放参数 weight
    # 没有偏移项（无 β），因为下一层（如注意力）对输入平移不敏感
    return (x / rms) * weight                            # (B, S, D)
'''

def build_softmax_solution():
    return '''def my_softmax(x):
    """
    Numerically stable Softmax implementation.
    Softmax 将任意实数向量映射为概率分布，所有元素非负且和为1。
    """
    # 数值稳定性: 先减去最大值，防止指数溢出
    # x_max 沿最后一个维度（类别维度）取最大值，keepdim=True 保证广播兼容
    x_max = x.max(dim=-1, keepdim=True).values           # max(x) 沿类别维

    # 指数化: exp(x - x_max)，所有值 ≤ 1，避免溢出
    exp_x = torch.exp(x - x_max)                         # e^(x_i - max(x))

    # 归一化: 每个元素除以该维度上的指数和
    # dim=-1 保证每个样本/位置独立归一化
    return exp_x / exp_x.sum(dim=-1, keepdim=True)       # (B, S, C)，每行和为1
'''

def build_cross_entropy_solution():
    return '''def cross_entropy_loss(logits, targets):
    """
    Cross-Entropy Loss implementation.
    衡量预测概率分布与真实标签之间的差异。对于 one-hot 标签，等价于负对数似然。
    """
    # logits: (N, C) 未归一化的分数，targets: (N,) 类别索引
    N = logits.shape[0]                                  # 样本数

    # Step 1: Log-Sum-Exp 技巧计算 log(Σ_j exp(logit_j))
    # 先减去最大值防止溢出: log(Σ exp(x_j)) = max(x) + log(Σ exp(x_j - max(x)))
    log_sum_exp = logits.logsumexp(dim=-1)               # (N,)，每个样本的归一化常数

    # Step 2: 提取正样本（真实标签）对应的 logits
    # logits[range(N), targets] 取出每个样本真实类别的未归一化分数
    positive_logits = logits[torch.arange(N), targets]   # (N,)

    # Step 3: 计算交叉熵: -log(p_i) = -(logit_i - log_sum_exp) = log_sum_exp - logit_i
    # 平均到每个样本，得到最终损失
    return (log_sum_exp - positive_logits).mean()        # 标量，平均交叉熵损失
'''

def build_gelu_solution():
    return '''def my_gelu(x):
    """
    GELU (Gaussian Error Linear Unit) implementation.
    使用 tanh 近似公式，比精确实现（erf）更快，被 PyTorch 默认采用。
    公式: GELU(x) ≈ 0.5x * (1 + tanh[√(2/π) * (x + 0.044715x³)])
    """
    # 系数: √(2/π) ≈ 0.7978845608
    sqrt_2_over_pi = (2.0 / 3.141592653589793) ** 0.5

    # 内部多项式: x + 0.044715 * x³
    # 三次项提供非线性，使函数在负值区域平滑趋近于0
    inner = x + 0.044715 * (x ** 3)

    # tanh(√(2/π) * inner)
    tanh_val = torch.tanh(sqrt_2_over_pi * inner)

    # GELU(x) = 0.5x * (1 + tanh(...))
    # 当 x → +∞ 时 tanh → 1，GELU(x) → x；当 x → -∞ 时 tanh → -1，GELU(x) → 0
    return 0.5 * x * (1.0 + tanh_val)
'''

def build_dropout_solution():
    return '''def MyDropout(x, p=0.5, training=True):
    """
    Dropout implementation (Inverted Dropout).
    训练时随机将 p 比例的元素置零，并将保留的元素缩放 1/(1-p)。
    推理时直接返回输入（无需调整，因为训练时已缩放）。
    """
    if not training:
        return x                                         # 推理模式: 恒等映射

    # 生成与 x 同形状的随机掩码，元素以概率 (1-p) 为 1，概率 p 为 0
    # torch.rand_like(x) 生成 [0,1) 均匀分布，> p 的位置保留
    mask = (torch.rand_like(x) > p).float()              # Bernoulli(1-p) 掩码

    # 应用掩码并缩放: 保留的元素乘以 1/(1-p)
    # 缩放保证期望值不变: E[mask] = 1-p, 所以 E[x * mask / (1-p)] = x
    return x * mask / (1 - p)                            # (B, S, D)
'''

def build_gpt2_block_solution():
    return '''class GPT2Block(nn.Module):
    """
    GPT-2 Transformer Block (Pre-Norm architecture).
    先对输入做 LayerNorm，再经过 Attention/MLP，最后残差连接。
    Pre-Norm 比 Post-Norm 更稳定，允许训练更深的网络。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()                               # nn.Module 初始化
        self.ln1 = nn.LayerNorm(d_model)                 # 注意力前的 LayerNorm
        self.ln2 = nn.LayerNorm(d_model)                 # MLP 前的 LayerNorm
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # 多头自注意力
        self.mlp = nn.Sequential(                        # MLP: 扩张 -> 激活 -> 收缩
            nn.Linear(d_model, 4 * d_model),             # 扩张到 4*d_model (GPT-2 惯例)
            nn.GELU(),                                    # GELU 激活
            nn.Linear(4 * d_model, d_model),             # 收缩回 d_model
        )
        self.dropout = nn.Dropout(dropout)               # Dropout 正则化

    def forward(self, x):
        # Pre-Norm Attention Block:
        # 1. LayerNorm -> Attention -> Dropout -> 残差连接
        # 残差连接缓解梯度消失，允许深层网络训练
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dropout(attn_out)                   # 残差: x + Dropout(Attn(LN(x)))

        # Pre-Norm MLP Block:
        # 2. LayerNorm -> MLP -> Dropout -> 残差连接
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)                    # 残差: x + Dropout(MLP(LN(x)))
        return x                                         # (B, S, d_model)
'''

def build_attention_solution():
    return '''def scaled_dot_product_attention(Q, K, V):
    """
    Scaled Dot-Product Attention (single-head).
    计算 Q 和 K 的相似度，用 softmax 归一化后对 V 加权求和。
    这是 Transformer 的核心算子，也是 MHA 的基础组件。
    """
    d_k = K.size(-1)                                     # 键的维度，用于缩放因子

    # Step 1: 计算相似度分数 Q @ K^T
    # torch.bmm: 批量矩阵乘法，Q(B,S_q,D) @ K^T(B,D,S_k) = scores(B, S_q, S_k)
    scores = torch.bmm(Q, K.transpose(1, 2))             # (B, S_q, S_k)

    # Step 2: 缩放: 除以 sqrt(d_k)
    # 防止大 d_k 时分数过大导致 softmax 梯度极小（饱和问题）
    scores = scores / (d_k ** 0.5)                       # (B, S_q, S_k)

    # Step 3: Softmax 归一化得到注意力权重
    # dim=-1 沿 key 维度归一化，每行和为 1
    weights = torch.softmax(scores, dim=-1)              # (B, S_q, S_k)

    # Step 4: 加权求和: 注意力权重 @ V
    # weights(B,S_q,S_k) @ V(B,S_k,D_v) = out(B, S_q, D_v)
    return torch.bmm(weights, V)                         # (B, S_q, D_v)
'''

def build_causal_attention_solution():
    return '''def causal_attention(Q, K, V):
    """
    Causal (Masked) Self-Attention.
    在标准注意力基础上添加因果掩码，确保每个位置只能关注当前及之前的位置。
    这是 GPT 等自回归模型的核心机制。
    """
    d_k = K.size(-1)                                     # 键维度

    # Step 1: 计算相似度分数
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)   # (B, S, S)

    S = scores.size(-1)                                  # 序列长度

    # Step 2: 生成因果掩码 (上三角为 True，需遮蔽)
    # torch.triu(..., diagonal=1) 生成主对角线以上的上三角矩阵
    # 例如 S=4: [[F,T,T,T],[F,F,T,T],[F,F,F,T],[F,F,F,F]]
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)

    # Step 3: 应用掩码: 将未来位置设为负无穷
    # masked_fill(mask, value) 将 mask=True 的位置替换为 value
    # -inf 经过 softmax 后概率为 0，实现"看不见未来"
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))   # (B, S, S)

    # Step 4: Softmax 归一化 (被遮蔽的位置概率为 0)
    weights = torch.softmax(scores, dim=-1)              # (B, S, S)

    # Step 5: 加权求和得到输出
    return torch.bmm(weights, V)                         # (B, S, D_v)
'''

def build_transformer_encoder_solution():
    return '''class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder (stack of N identical layers).
    每个 Encoder Layer = Multi-Head Self-Attention + Feed-Forward Network，
    均采用 Pre-Norm + 残差连接。
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 词嵌入: 将 token ID 映射为 d_model 维向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 正弦位置编码: 为每个位置提供唯一的位置信息
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)     # 偶数维: sin
        pe[:, 1::2] = torch.cos(position * div_term)     # 奇数维: cos
        self.register_buffer('pe', pe.unsqueeze(0))      # (1, max_len, d_model)，不参与训练

        # 堆叠 N 个 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)                # 最终 LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1: 词嵌入 + 位置编码
        # 乘以 √(d_model) 缩放嵌入，防止位置编码相对过大
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]                # 广播相加位置编码
        x = self.dropout(x)

        # Step 2: 逐层通过 Encoder
        for layer in self.layers:
            x = layer(x)                                 # (B, S, d_model)

        # Step 3: 最终归一化
        return self.norm(x)                              # (B, S, d_model)


class TransformerEncoderLayer(nn.Module):
    """单个 Transformer Encoder Layer (Pre-Norm + 残差)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)          # FFN 第一层: d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)          # FFN 第二层: d_ff -> d_model
        self.norm1 = nn.LayerNorm(d_model)               # Attention 前的 LayerNorm
        self.norm2 = nn.LayerNorm(d_model)               # FFN 前的 LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()                      # FFN 激活函数

    def forward(self, x):
        # Pre-Norm Self-Attention + 残差
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)

        # Pre-Norm FFN + 残差
        # FFN(x) = Linear2(GELU(Linear1(x)))
        ffn_out = self.linear2(self.activation(self.linear2(self.norm2(x))))
        # 修正: 应该是 linear2(activation(linear1(norm2(x))))
        # 上面这行有 bug，修正如下:
        # ffn_out = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + self.dropout(ffn_out)
        return x
'''

# 上面的 transformer_encoder_solution 有bug，让我修正。

def build_transformer_encoder_solution_fixed():
    return '''class TransformerEncoderLayer(nn.Module):
    """
    单个 Transformer Encoder Layer (Pre-Norm + Residual).
    结构: LN -> Self-Attention -> Residual -> LN -> FFN -> Residual
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)          # FFN 第一层: d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)          # FFN 第二层: d_ff -> d_model
        self.norm1 = nn.LayerNorm(d_model)               # Attention 前的 LayerNorm
        self.norm2 = nn.LayerNorm(d_model)               # FFN 前的 LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()                      # FFN 激活函数

    def forward(self, x):
        # Pre-Norm Self-Attention + 残差
        # self_attn(Q,K,V) 输入均为 self.norm1(x)，因为是自注意力
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)                   # 残差连接

        # Pre-Norm FFN + 残差
        # FFN(x) = Linear2(Activation(Linear1(Norm2(x))))
        ffn_out = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + self.dropout(ffn_out)                    # 残差连接
        return x


class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder (堆叠 N 层)。
    包含词嵌入、正弦位置编码、N 个 Encoder Layer、最终 LayerNorm。
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层: 将整数 token ID 映射为 d_model 维连续向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 正弦位置编码: 为每个序列位置提供唯一的位置表示
        # 使用不同频率的正弦/余弦函数，使模型能学习相对位置关系
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       # (max_len, 1)
        # div_term: 频率衰减项，使不同维度的波长呈指数增长
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)     # 偶数维: sin(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)     # 奇数维: cos(pos / 10000^(2i/d_model))
        self.register_buffer('pe', pe.unsqueeze(0))      # (1, max_len, d_model)，不参与梯度更新

        # 堆叠 N 个相同的 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)                # 最终输出归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Step 1: 词嵌入 + 位置编码
        # 乘以 √(d_model) 缩放嵌入向量，使位置编码与嵌入向量量级相当
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]                # 广播相加: (B,S,d_model) + (1,S,d_model)
        x = self.dropout(x)                              # 随机丢弃防止过拟合

        # Step 2: 逐层前向传播
        for layer in self.layers:
            x = layer(x)                                 # 每层保持 (B, S, d_model)

        # Step 3: 最终 LayerNorm
        return self.norm(x)                              # (B, S, d_model)
'''

def build_transformer_decoder_solution():
    return '''class TransformerDecoderLayer(nn.Module):
    """
    单个 Transformer Decoder Layer (Pre-Norm + 3 sub-layers).
    1. Masked Self-Attention (causal, 只能看已生成的 token)
    2. Cross-Attention (Q from decoder, K/V from encoder)
    3. Feed-Forward Network
    每层都有残差连接和 Dropout。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)               # Self-Attention 前
        self.norm2 = nn.LayerNorm(d_model)               # Cross-Attention 前
        self.norm3 = nn.LayerNorm(d_model)               # FFN 前
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, memory):
        # Sub-layer 1: Masked Self-Attention (causal)
        # attn_mask 保证每个位置只能关注自己和之前的位置
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        self_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                      attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(self_out)                   # 残差连接

        # Sub-layer 2: Cross-Attention (decoder attends to encoder output)
        # Q from decoder, K/V from encoder memory
        cross_out, _ = self.cross_attn(self.norm2(x), memory, memory, need_weights=False)
        x = x + self.dropout(cross_out)                  # 残差连接

        # Sub-layer 3: Feed-Forward Network
        ffn_out = self.linear2(self.activation(self.linear1(self.norm3(x))))
        x = x + self.dropout(ffn_out)                    # 残差连接
        return x


class TransformerDecoder(nn.Module):
    """
    完整的 Transformer Decoder (用于 Seq2Seq 或自回归生成)。
    包含词嵌入、位置编码、N 个 Decoder Layer、最终投影到词表。
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # 正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)     # 投影到词表维度，用于预测下一个 token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # 词嵌入 + 位置编码 + Dropout
        x = self.embedding(x) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)

        # 逐层通过 Decoder
        for layer in self.layers:
            x = layer(x, memory)                         # (B, S, d_model)

        # 最终归一化 + 投影到词表
        x = self.norm(x)
        return self.fc_out(x)                            # (B, S, vocab_size)
'''

def build_clip_solution():
    return '''class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) simplified implementation.
    核心思想: 通过对比学习让匹配的图像-文本对在嵌入空间中距离近，不匹配的距离远。
    """
    def __init__(self, image_encoder, text_encoder, embed_dim, text_dim, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder               # 图像编码器 (e.g. ResNet/ViT)
        self.text_encoder = text_encoder                 # 文本编码器 (e.g. Transformer)

        # 文本投影层: 将 text_encoder 的输出维度映射到统一的 embed_dim
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 可学习的温度参数: logit_scale = log(1 / temperature)
        # 温度控制相似度分布的"尖锐程度": 温度越低，模型对正负样本区分越严格
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))

    def forward(self, images, text_features):
        # Step 1: 编码图像和文本
        image_embed = self.image_encoder(images)         # (B, embed_dim)
        text_embed = self.text_proj(self.text_encoder(text_features))  # (B, embed_dim)

        # Step 2: L2 归一化
        # 归一化后，点积等价于余弦相似度: a·b = |a||b|cosθ = cosθ (当 |a|=|b|=1)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)   # (B, embed_dim)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)      # (B, embed_dim)

        # Step 3: 计算相似度矩阵
        # logits[i,j] = cosine_similarity(image_i, text_j) * scale
        # scale = exp(logit_scale) = 1 / temperature (可学习)
        scale = torch.exp(self.logit_scale)              # 标量，可学习
        logits = image_embed @ text_embed.T * scale      # (B, B)，对角线为匹配对

        # Step 4: 对称交叉熵损失
        # 图像→文本方向: 对每个图像，找最匹配的文本
        # 文本→图像方向: 对每个文本，找最匹配的图像
        targets = torch.arange(logits.shape[0], device=logits.device)        # [0, 1, ..., B-1]
        loss_i2t = torch.nn.functional.cross_entropy(logits, targets)        # 图像→文本
        loss_t2i = torch.nn.functional.cross_entropy(logits.T, targets)      # 文本→图像
        return (loss_i2t + loss_t2i) / 2.0               # 对称平均损失
'''

def build_einsum_solution():
    return '''def einsum_operations(A, B):
    """
    Einsum operations demonstration.
    实现 batch matrix multiplication、trace、outer product 和 diagonal extraction。
    """
    results = {}

    # 1. Batch Matrix Multiplication: 'bmk,bkn->bmn'
    # A: (B, M, K), B: (B, K, N) -> out: (B, M, N)
    # 相当于对每个 batch 做独立的矩阵乘法
    results['batch_matmul'] = torch.einsum('bmk,bkn->bmn', A, B)

    # 2. Matrix Trace (对角线元素之和): 'bii->b'
    # A: (B, I, I) -> out: (B,)
    # trace = Σ_i A[:, i, i]，即每个 batch 矩阵对角线元素之和
    results['trace'] = torch.einsum('bii->b', A)

    # 3. Outer Product: 'i,j->ij'
    # a: (I,), b: (J,) -> out: (I, J)
    # outer[i,j] = a[i] * b[j]，所有元素两两相乘
    a = A[0, :, 0]                                     # 取第一个 batch 的第一列: (M,)
    b = B[0, 0, :]                                     # 取第一个 batch 的第一行: (N,)
    results['outer'] = torch.einsum('i,j->ij', a, b)   # (M, N)

    # 4. Diagonal Extraction: 'ii->i'
    # A: (I, I) -> out: (I,)
    # 提取方阵的对角线元素
    mat = A[0] if A.ndim >= 2 else A                   # 取第一个 batch 的 (M, M) 子矩阵
    min_dim = min(mat.shape[-2], mat.shape[-1])
    results['diagonal'] = torch.einsum('ii->i', mat[:min_dim, :min_dim])   # (min_dim,)

    return results
'''

def build_broadcasting_solution():
    return '''def broadcast_and_add(A, B):
    """
    手动实现 PyTorch 风格的广播机制（Broadcasting）。
    广播规则: 从后往前比较维度，要么相等，要么其中一个为 1，要么缺失（视为 1）。
    """
    # 获取形状并在左侧补 1，使两个张量维度相同
    shape_a = list(A.shape)                            # 例如 [3, 1]
    shape_b = list(B.shape)                            # 例如 [1, 4]
    max_ndim = max(len(shape_a), len(shape_b))         # 最大维度数
    shape_a = [1] * (max_ndim - len(shape_a)) + shape_a   # 左侧补 1: [3, 1] -> [3, 1] (假设 2D)
    shape_b = [1] * (max_ndim - len(shape_b)) + shape_b   # 左侧补 1: [1, 4] -> [1, 4]

    # 计算广播后的输出形状
    out_shape = []
    for da, db in zip(shape_a, shape_b):
        if da == db:
            out_shape.append(da)                       # 维度相同，直接保留
        elif da == 1:
            out_shape.append(db)                       # A 维度为 1，广播为 B 的维度
        elif db == 1:
            out_shape.append(da)                       # B 维度为 1，广播为 A 的维度
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {A.shape} and {B.shape}")

    # 辅助函数: 将张量扩展到目标形状
    def expand_tensor(t, from_shape, to_shape):
        # 在左侧补充维度 (unsqueeze)
        for _ in range(len(to_shape) - t.ndim):
            t = t.unsqueeze(0)                         # 在最左侧添加新维度
        # 对大小为 1 的维度进行扩展
        for i, (fs, ts) in enumerate(zip(from_shape, to_shape)):
            if fs == 1 and ts != 1:
                t = t.expand(*to_shape[:i+1], *t.shape[i+1:])   # expand 到目标形状
        return t

    # 扩展两个张量到广播后的形状
    A_exp = expand_tensor(A, shape_a, out_shape)       # (out_shape)
    B_exp = expand_tensor(B, shape_b, out_shape)       # (out_shape)

    # 逐元素相加
    return A_exp + B_exp                               # (out_shape)
'''

# --------------- 智能注释增强器（通用版本） ---------------

def enhance_solution_comments(task_id: str, solution: str) -> str:
    """Add detailed line-by-line comments to solution code."""
    # Skip if already enhanced (contains our specific comment patterns)
    if solution and ("调用父类 nn.Module" in solution or "Step 1:" in solution
                     or "详解" in solution or "importance" in solution
                     or "核心思想" in solution or "面试考点" in solution):
        return solution

    # If solution is one of the core tasks we manually rewrote, use that instead
    manual_solutions = {
        'mha': build_mha_solution,
        'cross_attention': build_cross_attention_solution,
        'adam': build_adam_solution,
        'layernorm': build_layernorm_solution,
        'rmsnorm': build_rmsnorm_solution,
        'softmax': build_softmax_solution,
        'cross_entropy': build_cross_entropy_solution,
        'gelu': build_gelu_solution,
        'dropout': build_dropout_solution,
        'gpt2_block': build_gpt2_block_solution,
        'attention': build_attention_solution,
        'causal_attention': build_causal_attention_solution,
        'transformer_encoder': build_transformer_encoder_solution_fixed,
        'transformer_decoder': build_transformer_decoder_solution,
        'clip_model': build_clip_solution,
        'einsum_ops': build_einsum_solution,
        'broadcasting': build_broadcasting_solution,
    }
    if task_id in manual_solutions:
        return manual_solutions[task_id]()

    # For other tasks, apply rule-based comment enhancement
    lines = solution.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            continue

        comment = None
        # Detect common patterns and add comments
        if re.search(r'super\(\).__init__\(\)', stripped):
            comment = '调用父类 nn.Module 初始化，注册所有子模块'
        elif re.search(r'nn\.Linear\((.+?),\s*(.+?)\)', stripped):
            m = re.search(r'nn\.Linear\((.+?),\s*(.+?)\)', stripped)
            comment = f'线性投影: {m.group(1)}维 -> {m.group(2)}维'
        elif re.search(r'nn\.LayerNorm\(', stripped):
            comment = '层归一化: 稳定训练，防止梯度爆炸/消失'
        elif re.search(r'nn\.Dropout\(', stripped):
            comment = 'Dropout正则化: 防止过拟合'
        elif re.search(r'nn\.Embedding\(', stripped):
            comment = '词嵌入: 将 token ID 映射为连续向量'
        elif re.search(r'nn\.MultiheadAttention\(', stripped):
            comment = '多头自注意力'
        elif re.search(r'nn\.Sequential\(', stripped):
            comment = '顺序容器: 按顺序堆叠多个层'
        elif re.search(r'nn\.GELU\(\)', stripped):
            comment = 'GELU 激活函数'
        elif re.search(r'torch\.bmm\(', stripped):
            comment = '批量矩阵乘法'
        elif re.search(r'torch\.matmul\(', stripped):
            comment = '矩阵乘法'
        elif re.search(r'torch\.softmax\(', stripped):
            comment = 'Softmax归一化: 转换为概率分布'
        elif re.search(r'torch\.triu\(', stripped):
            comment = '生成上三角掩码: 用于因果注意力'
        elif re.search(r'masked_fill\(', stripped):
            comment = '应用掩码: 将指定位置替换为指定值'
        elif re.search(r'\.view\(.+?\)\.transpose\(1,\s*2\)', stripped):
            comment = '多头拆分: (B,S,D) -> (B,H,S,d_k)'
        elif re.search(r'\.transpose\(1,\s*2\)\.contiguous\(\)\.view\(', stripped):
            comment = '合并多头: (B,H,S,d_k) -> (B,S,D)'
        elif re.search(r'torch\.exp\(', stripped):
            comment = '指数运算'
        elif re.search(r'torch\.log\(', stripped):
            comment = '对数运算'
        elif re.search(r'torch\.sqrt\(', stripped):
            comment = '开方运算'
        elif re.search(r'\.mean\(', stripped):
            comment = '求均值'
        elif re.search(r'\.var\(', stripped):
            comment = '求方差'
        elif re.search(r'\.max\(', stripped):
            comment = '求最大值'
        elif re.search(r'\.sum\(', stripped):
            comment = '求和'
        elif re.search(r'register_buffer\(', stripped):
            comment = '注册缓冲区: 不参与梯度更新'
        elif re.search(r'nn\.ModuleList\(', stripped):
            comment = '模块列表: PyTorch 自动追踪其中的子模块'
        elif stripped.startswith('class ') and 'nn.Module' in stripped:
            comment = '继承 nn.Module，注册为可训练模块'
        elif stripped.startswith('def forward(') or stripped.startswith('def __call__('):
            comment = '前向传播: 定义数据流'
        elif stripped.startswith('def __init__('):
            comment = '初始化: 定义模型结构和参数'

        if comment and '#' not in stripped:
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            new_lines.append(line + '  # ' + comment)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


# --------------- Enhanced explanations ---------------

def build_enhanced_explanation(task_id: str) -> str:
    """Build explanation that strictly matches the solution code."""
    explanations = {
        'mha': (
            "**Multi-Head Attention 详解**\n\n"
            "本实现严格按照论文《Attention Is All You Need》的原始设计。\n\n"
            "**Step 1 — 线性投影 + 多头拆分:**\n"
            "- `self.W_q(Q)` 将 Q 从 d_model 投影到 d_model（实际是 H 个头的拼接）\n"
            "- `.view(B, S, H, d_k)` 将最后一维拆分为 H × d_k\n"
            "- `.transpose(1, 2)` 得到 (B, H, S, d_k)，使每个头可以独立做矩阵乘法\n\n"
            "**Step 2 — 缩放点积:**\n"
            "- `scores = Q @ K^T / sqrt(d_k)`，除以 sqrt(d_k) 防止大 d_k 时 softmax 饱和\n"
            "- 这是从理论上保证方差稳定在 1 的关键（当 Q,K 独立同分布时 Var(scores)=1）\n\n"
            "**Step 3 — Softmax + 加权求和:**\n"
            "- `softmax(scores)` 将相似度转换为概率分布\n"
            "- `weights @ V` 得到每个查询位置的上下文向量\n\n"
            "**Step 4 — 合并 + 输出投影:**\n"
            "- `.transpose(1, 2).contiguous().view(...)` 将多头输出拼接回 d_model 维\n"
            "- `self.W_o` 做最终线性投影，让模型学习到如何混合不同头的信息\n\n"
            "**面试考点:**\n"
            "- 为什么要除以 sqrt(d_k)？→ 防止点积值过大导致 softmax 梯度消失\n"
            "- 为什么要 transpose？→ 为了让 batch × head 维度在最前面，便于并行计算\n"
            "- W_o 的作用？→ 不同头学到不同的特征子空间，W_o 负责融合这些子空间"
        ),
        'cross_attention': (
            "**Cross-Attention 详解**\n\n"
            "与 Self-Attention 的核心区别：Q 和 K/V 来自不同序列。\n\n"
            "**数据流:**\n"
            "- `x_q` (decoder) → `W_q` → Q heads\n"
            "- `x_kv` (encoder) → `W_k/W_v` → K/V heads\n"
            "- 计算注意力时，decoder 的每个位置可以看到 encoder 的所有位置\n\n"
            "**为什么没有因果掩码？**\n"
            "- 编码器输出在推理时已经全部生成，decoder 应该能访问全部信息\n"
            "- 因果掩码只用于 decoder 的自注意力（防止偷看未来 token）\n\n"
            "**面试考点:**\n"
            "- Cross-Attention 在 T2I 模型（如 Stable Diffusion）中的作用？\n"
            "  → 图像 token 关注文本 CLIP 嵌入，实现文本引导图像生成\n"
            "- 在 LLaVA 等多模态模型中？→ 视觉特征作为 K/V，文本查询作为 Q"
        ),
        'adam': (
            "**Adam Optimizer 详解**\n\n"
            "Adam = Adaptive Moment Estimation，结合了 Momentum 和 RMSProp。\n\n"
            "**一阶矩 m（Momentum）:**\n"
            "- `m_t = β1·m_{t-1} + (1-β1)·g_t`\n"
            "- 指数移动平均，β1=0.9 表示历史梯度权重 90%，当前梯度 10%\n"
            "- m_t 提供更新方向（类似物理中的速度）\n\n"
            "**二阶矩 v（RMSProp）:**\n"
            "- `v_t = β2·v_{t-1} + (1-β2)·g_t²`\n"
            "- 记录梯度平方的移动平均，用于自适应调整学习率\n"
            "- 梯度大的参数 → v 大 → 更新步长小（更谨慎）\n"
            "- 梯度小的参数 → v 小 → 更新步长大（更大胆）\n\n"
            "**偏差校正:**\n"
            "- 初始化 m=0, v=0，早期步骤会偏向零\n"
            "- `m̂ = m / (1-β1^t)` 补偿这个偏差，t 很小时分母很小，m̂ 被放大\n"
            "- 没有偏差校正，前几十步更新会非常小，收敛变慢\n\n"
            "**面试考点:**\n"
            "- Adam 和 SGD 的区别？→ Adam 自适应学习率，SGD 固定/手动调度\n"
            "- 为什么需要 eps？→ 防止 v̂ 接近 0 时除以极小数导致数值爆炸\n"
            "- AdamW 和 Adam 的区别？→ AdamW 将权重衰减直接作用于参数（decoupled），而非加入梯度"
        ),
        'layernorm': (
            "**Layer Normalization 详解**\n\n"
            "与 BatchNorm 的关键区别：LN 沿特征维度归一化，BN 沿 batch 维度归一化。\n\n"
            "**计算流程:**\n"
            "1. `mean = x.mean(dim=-1)` — 每个样本每个位置的均值\n"
            "2. `var = x.var(dim=-1, unbiased=False)` — 每个样本每个位置的方差\n"
            "3. `x_norm = (x - mean) / sqrt(var + eps)` — 标准化为 N(0,1)\n"
            "4. `gamma * x_norm + beta` — 可学习的仿射变换\n\n"
            "**为什么 Transformer 用 LN 不用 BN？**\n"
            "- 序列长度变化大，batch 统计量不稳定\n"
            "- 推理时 batch size 可能为 1，BN 无法计算 batch 统计量\n"
            "- LN 对每个样本独立归一化，不依赖 batch size\n\n"
            "**Pre-Norm vs Post-Norm:**\n"
            "- Post-Norm: x = LN(x + Attention(x)) — 残差路径上的梯度经过 LN，深层网络梯度消失\n"
            "- Pre-Norm: x = x + Attention(LN(x)) — 残差路径直通，梯度稳定，可训练更深网络\n"
            "- GPT-2/3, LLaMA 等均采用 Pre-Norm"
        ),
        'rmsnorm': (
            "**RMS Normalization 详解**\n\n"
            "RMSNorm = Root Mean Square Normalization，LLaMA 和 Mistral 采用的归一化方法。\n\n"
            "**与 LayerNorm 的对比:**\n\n"
            "| 特性 | LayerNorm | RMSNorm |\n"
            "|------|-----------|---------|\n"
            "| 均值计算 | 有 (减均值) | 无 |\n"
            "| 方差计算 | 有 | 有 (用 RMS 代替) |\n"
            "| FLOPs | 多 | 少约 30% |\n"
            "| 偏移项 β | 有 | 无 |\n\n"
            "**为什么去掉均值减法？**\n"
            "- 如果下一层（如 attention）对输入平移不敏感，则减均值是冗余的\n"
            "- Attention 的 query-key 点积: (x-μ)·(y-μ) = x·y - μ·y - x·μ + μ²，平移会影响结果\n"
            "- 但实验表明去掉均值后效果几乎不变，说明 Transformer 内部对这种平移不敏感\n\n"
            "**公式推导:**\n"
            "`RMS(x) = sqrt(mean(x²))`，`RMSNorm(x) = x / RMS(x) * γ`\n"
            "没有 β 偏移项，因为归一化后的均值为 0，加上 γ 缩放即可"
        ),
        'softmax': (
            "**Softmax 详解**\n\n"
            "Softmax 将任意实数向量映射为概率分布：$p_i = e^{x_i} / Σ_j e^{x_j}$\n\n"
            "**数值稳定性技巧:**\n"
            "- 直接计算 `exp(x)` 可能溢出（x 很大时 e^x → ∞）\n"
            "- 减去最大值: `exp(x - x_max)`，此时最大指数为 0，不会溢出\n"
            "- 数学等价性: $e^{x_i}/Σe^{x_j} = e^{x_i-x_{max}}/Σe^{x_j-x_{max}}$\n\n"
            "**温度缩放:**\n"
            "- `softmax_τ(x) = exp(x/τ) / Σexp(x/τ)`\n"
            "- τ→0: 分布尖锐，接近 argmax\n"
            "- τ→∞: 分布均匀，接近均匀分布\n"
            "- CLIP 中的 temperature 就是可学习的 τ"
        ),
        'cross_entropy': (
            "**Cross-Entropy Loss 详解**\n\n"
            "CE Loss = -Σ y_i · log(p_i)，对于 one-hot y，简化为 -log(p_{true})\n\n"
            "**实现要点:**\n\n"
            "1. **Log-Sum-Exp 技巧:**\n"
            "   - `log(Σ exp(x_j)) = max(x) + log(Σ exp(x_j - max(x)))`\n"
            "   - 防止 exp 溢出，是数值稳定的 softmax + log 计算\n\n"
            "2. **提取正样本 logit:**\n"
            "   - `logits[range(N), targets]` 用高级索引取出每个样本真实类别的分数\n"
            "   - 这比 one-hot 乘法更高效\n\n"
            "3. **最终损失:**\n"
            "   - `log_sum_exp - positive_logits` = -log(p_true)\n"
            "   - `.mean()` 平均到每个样本\n\n"
            "**面试考点:**\n"
            "- 为什么 CE 比 MSE 更适合分类？→ CE 的梯度在错误预测时更大，收敛更快\n"
            "- Label Smoothing 的作用？→ 防止模型对正确标签过度自信，提升泛化"
        ),
        'gelu': (
            "**GELU 详解**\n\n"
            "GELU(x) = x · Φ(x)，其中 Φ 是标准正态分布的 CDF。\n\n"
            "**精确公式:** $GELU(x) = x · (1/2)[1 + erf(x/√2)]$\n\n"
            "**为什么用近似而不是精确实现？**\n"
            "- erf 函数计算开销大，近似公式用 tanh 只需基本运算\n"
            "- PyTorch 内部也使用这个近似\n\n"
            "**近似公式推导:**\n"
            "`GELU(x) ≈ 0.5x · (1 + tanh[√(2/π)·(x + 0.044715x³)])`\n\n"
            "- 0.044715 是拟合系数，使近似误差最小\n"
            "- 当 x→+∞: tanh→1, GELU→x（恒等映射）\n"
            "- 当 x→-∞: tanh→-1, GELU→0（关闭）\n"
            "- 在 x=0 处光滑（不同于 ReLU 的折点）\n\n"
            "**面试考点:**\n"
            "- GELU vs ReLU？→ GELU 处处光滑，负值区域有微小梯度（非零），缓解 dying ReLU\n"
            "- 为什么 Transformer 用 GELU？→ 经验上比 ReLU 在 NLP 任务上效果更好"
        ),
        'dropout': (
            "**Dropout 详解**\n\n"
            "训练时随机丢弃 p 比例的神经元，防止 co-adaptation（神经元过度依赖彼此）。\n\n"
            "**Inverted Dropout:**\n"
            "- 训练时: `x * mask / (1-p)`，保留元素放大 1/(1-p)\n"
            "- 推理时: 直接返回 x（无需缩放）\n"
            "- 好处: 推理代码简单，只需判断 training 标志\n\n"
            "**为什么缩放 1/(1-p)？**\n"
            "- E[mask] = 1-p，所以 E[x*mask/(1-p)] = x·(1-p)/(1-p) = x\n"
            "- 保证训练时和推理时的期望值一致\n\n"
            "**面试考点:**\n"
            "- Dropout 和 BN 能一起用吗？→ 可以，但要注意训练/推理行为一致\n"
            "- 为什么 Dropout 在推理时关闭？→ 需要完整的模型表达能力\n"
            "- Dropout 的等价解释？→ 集成学习: 每次训练用不同的子网络，推理时近似平均"
        ),
        'gpt2_block': (
            "**GPT-2 Block 详解**\n\n"
            "GPT-2 使用 Pre-Norm 架构，核心是两个子层 + 残差连接。\n\n"
            "**Pre-Norm 公式:**\n"
            "- `x = x + Dropout(Attn(LN₁(x)))`\n"
            "- `x = x + Dropout(MLP(LN₂(x)))`\n\n"
            "**为什么 Pre-Norm 比 Post-Norm 好？**\n"
            "- Post-Norm: `x = LN(x + Attn(x))`，残差路径经过 LN，梯度衰减快\n"
            "- Pre-Norm: 残差路径 `x → x + ...` 是直通的，梯度可以无损传播\n"
            "- 实验表明 Pre-Norm 可以稳定训练 100+ 层的 Transformer\n\n"
            "**MLP 结构:**\n"
            "- `Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)`\n"
            "- 4× 扩张是 GPT-2 的惯例，FFN 参数量占总参数的 2/3\n\n"
            "**面试考点:**\n"
            "- FFN 的作用？→ 提供非线性变换和特征维度转换，是模型的记忆部分\n"
            "- 为什么用 GELU 而不用 ReLU？→ GELU 更平滑，在 NLP 上效果更好"
        ),
        'attention': (
            "**Scaled Dot-Product Attention 详解**\n\n"
            "这是 Transformer 最核心的算子，MHA 的基础组件。\n\n"
            "**计算步骤:**\n"
            "1. `scores = Q @ K^T` — 点积衡量查询与键的相似度\n"
            "2. `scores /= sqrt(d_k)` — 缩放防止大 d_k 时 softmax 饱和\n"
            "3. `weights = softmax(scores)` — 归一化为概率分布\n"
            "4. `out = weights @ V` — 加权求和得到上下文向量\n\n"
            "**为什么用点积而不是欧氏距离？**\n"
            "- 点积计算快（可用矩阵乘法优化）\n"
            "- 对于归一化向量，点积 = cosθ，直接衡量方向相似度\n"
            "- 点积可解释为「查询在键方向上的投影长度」\n\n"
            "**Cross-Attention 支持:**\n"
            "- Q 和 K/V 可以有不同的序列长度 (S_q ≠ S_k)\n"
            "- 输出形状 = (B, S_q, D_v)，由 Q 的序列长度决定"
        ),
        'causal_attention': (
            "**Causal (Masked) Attention 详解**\n\n"
            "自回归模型的核心机制，确保每个位置只能关注自己和之前的位置。\n\n"
            "**因果掩码生成:**\n"
            "- `torch.triu(ones(S,S), diagonal=1)` 生成上三角矩阵（主对角线以上全 1）\n"
            "- 例如 S=4: `[[F,T,T,T], [F,F,T,T], [F,F,F,T], [F,F,F,F]]`\n"
            "- True 表示需要被遮蔽的位置（未来位置）\n\n"
            "**掩码应用:**\n"
            "- `masked_fill(mask, -inf)` 将未来位置设为 -inf\n"
            "- softmax 后这些位置概率为 0，实现「看不见未来」\n\n"
            "**面试考点:**\n"
            "- 为什么不用下三角掩码？→ triu 生成上三角，mask=True 的是未来位置，正好对应 j>i\n"
            "- 推理时 KV Cache 如何与因果掩码配合？→ 每次只生成一个新 token，只需要看过去的位置"
        ),
        'transformer_encoder': (
            "**Transformer Encoder 详解**\n\n"
            "完整的 Encoder = 词嵌入 + 位置编码 + N × Encoder Layer + 最终 LayerNorm。\n\n"
            "**正弦位置编码:**\n"
            "- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`\n"
            "- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`\n"
            "- 10000 是基数，d_model 是维度\n"
            "- 不同维度使用不同频率，使模型能学习相对位置（PE(pos+k) 可由 PE(pos) 线性表示）\n\n"
            "**Encoder Layer 结构:**\n"
            "- Self-Attention (多头自注意力，全可见)\n"
            "- FFN (两层 Linear + GELU，4× 扩张)\n"
            "- 均采用 Pre-Norm + 残差连接\n\n"
            "**面试考点:**\n"
            "- 位置编码为什么用正弦而不是可学习？→ 正弦可外推到更长序列，可学习的位置编码超出训练长度后失效\n"
            "- Encoder 和 Decoder 的区别？→ Encoder 用全可见自注意力，Decoder 用因果自注意力 + 交叉注意力"
        ),
        'transformer_decoder': (
            "**Transformer Decoder 详解**\n\n"
            "Decoder 有 3 个子层（比 Encoder 多一个 Cross-Attention）。\n\n"
            "**3 个子层:**\n"
            "1. **Masked Self-Attention:** Q/K/V 均来自 decoder 输入，带因果掩码\n"
            "2. **Cross-Attention:** Q 来自 decoder，K/V 来自 encoder memory\n"
            "3. **FFN:** 同 Encoder\n\n"
            "**因果掩码生成:**\n"
            "- `torch.triu(ones(S,S), diagonal=1).bool()`\n"
            "- 传入 MultiheadAttention 的 attn_mask 参数\n"
            "- 确保位置 i 只能看到位置 0..i\n\n"
            "**面试考点:**\n"
            "- Decoder 为什么要用 causal mask？→ 自回归生成，每个 token 只能基于已生成的内容\n"
            "- Cross-Attention 中 Q/K/V 的来源？→ Q=decoder, K/V=encoder\n"
            "- 训练时 Decoder 的输入是什么？→ teacher forcing: 使用真实标签右移一位（start token + 前 i-1 个真实 token）"
        ),
        'clip_model': (
            "**CLIP 详解**\n\n"
            "CLIP 的核心是对称对比学习：让匹配的图像-文本对相似度高，不匹配的低。\n\n"
            "**对比学习损失:**\n"
            "- 图像→文本: `cross_entropy(logits, targets)`，对角线是正样本\n"
            "- 文本→图像: `cross_entropy(logits.T, targets)`\n"
            "- 取平均保证对称性\n\n"
            "**可学习温度:**\n"
            "- `scale = exp(logit_scale) = 1/temperature`\n"
            "- 温度低 → 分布尖锐，模型区分正负样本更严格\n"
            "- 温度高 → 分布平滑，训练更稳定\n"
            "- 让模型自己学习最佳温度，而不是固定值\n\n"
            "**L2 归一化:**\n"
            "- 点积 = |a||b|cosθ，归一化后 |a|=|b|=1，点积 = cosθ\n"
            "- 这样相似度矩阵直接表示余弦相似度，更直观\n\n"
            "**面试考点:**\n"
            "- CLIP 的 zero-shot 能力来源？→ 在大量图像-文本对上预训练，学会了通用的视觉-语义对齐\n"
            "- 为什么用对称损失？→ 图像和文本模态等价，不应该有偏向\n"
            "- 温度参数的作用？→ 控制分布尖锐程度，类似对比学习中的 temperature scaling"
        ),
        'einsum_ops': (
            "**Einsum 详解**\n\n"
            "Einsum 是爱因斯坦求和约定，用字符串描述任意张量运算。\n\n"
            "**Batch MatMul ('bmk,bkn->bmn'):**\n"
            "- b: batch, m: rows, k: inner, n: cols\n"
            "- 对每个 batch 做 (m×k) @ (k×n) → (m×n)\n"
            "- 等价于 `torch.bmm(A, B)`\n\n"
            "**Trace ('bii->b'):**\n"
            "- 对最后两个维度求迹: Σ_i A[:, i, i]\n"
            "- 每个 batch 矩阵的对角线元素之和\n\n"
            "**Outer Product ('i,j->ij'):**\n"
            "- 两个向量所有元素两两相乘\n"
            "- outer[i,j] = a[i] * b[j]\n\n"
            "**面试考点:**\n"
            "- Einsum 比 bmm 灵活在哪里？→ 可以描述任意索引模式的运算，无需手动 reshape/transpose\n"
            "- 性能如何？→ PyTorch 内部会优化为对应的 BLAS 调用，性能与手写等价"
        ),
        'broadcasting': (
            "**Broadcasting 详解**\n\n"
            "PyTorch 的广播规则允许不同形状的张量进行逐元素运算。\n\n"
            "**规则:**\n"
            "1. 从后往前比较维度\n"
            "2. 要么相等，要么其中一个为 1，要么缺失（视为 1）\n"
            "3. 否则报错\n\n"
            "**示例:**\n"
            "- A: (3, 1), B: (1, 4) → 输出: (3, 4)\n"
            "- A 的第 2 维为 1，广播为 4\n"
            "- B 的第 1 维为 1，广播为 3\n\n"
            "**实现要点:**\n"
            "- 左侧补 1 使维度对齐\n"
            "- `unsqueeze(0)` 添加新维度\n"
            "- `expand()` 将大小为 1 的维度扩展到目标大小（不复制内存，仅改变 view）\n"
            "- 最后逐元素相加\n\n"
            "**面试考点:**\n"
            "- expand 和 repeat 的区别？→ expand 不分配新内存（共享数据），repeat 会复制数据\n"
            "- 广播的内存开销？→ 广播是虚拟的，不会实际复制数据"
        ),
    }
    return explanations.get(task_id, "")


# --------------- Main: process all tasks ---------------

def serialize_task(task: dict) -> str:
    """Serialize TASK dict to Python code."""
    lines = ['TASK = {']
    key_order = [
        "title", "title_zh", "difficulty", "category",
        "description_en", "description_zh",
        "function_name", "hint", "hint_zh",
        "theory_en", "theory_zh", "diagram_en", "diagram_zh",
        "explanation", "tests", "solution", "demo"
    ]
    for key in key_order:
        if key not in task:
            continue
        val = task[key]
        if key == "tests":
            lines.append(f'    "{key}": [')
            for test in val:
                lines.append('        {')
                lines.append(f'            "name": {json.dumps(test["name"], ensure_ascii=False)},')
                code = test.get("code", "")
                if '\n' in code:
                    lines.append(f'            "code": """')
                    for cline in code.split('\n'):
                        lines.append(cline)
                    lines.append('            """,')
                else:
                    lines.append(f'            "code": {json.dumps(code, ensure_ascii=False)},')
                lines.append('        },')
            lines.append('    ],')
        elif key in ("solution", "demo") and '\n' in str(val):
            lines.append(f'    "{key}": \'\'\'')
            for sline in str(val).split('\n'):
                lines.append(sline)
            lines.append("    \'\'\',")
        elif isinstance(val, str) and '\n' in val:
            lines.append(f'    "{key}": (')
            lines.append(f'        {json.dumps(val, ensure_ascii=False)}')
            lines.append('    ),')
        else:
            lines.append(f'    "{key}": {json.dumps(val, ensure_ascii=False)},')
    lines.append('}')
    return '\n'.join(lines)


def main():
    print("Enhancing all task solutions with detailed comments...")

    for fpath in sorted(TASKS_DIR.glob("*.py")):
        task_id = fpath.stem
        if task_id.startswith("_"):
            continue

        # Read and exec TASK
        source = fpath.read_text(encoding="utf-8")
        namespace = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "torch": _MockTorch(),
            "nn": _MockNn(),
        }
        try:
            exec(compile(source, str(fpath), "exec"), namespace)
        except Exception as e:
            print(f"  SKIP {task_id}: exec failed ({e})")
            continue

        task = namespace.get("TASK")
        if task is None:
            print(f"  SKIP {task_id}: no TASK")
            continue

        # 1. Enhance solution with detailed comments
        old_solution = task.get("solution", "")
        if old_solution:
            new_solution = enhance_solution_comments(task_id, old_solution)
            task["solution"] = new_solution
            print(f"  {task_id}: solution enhanced")

        # 2. Add enhanced explanation if available
        enhanced_exp = build_enhanced_explanation(task_id)
        if enhanced_exp:
            task["explanation"] = enhanced_exp
            print(f"  {task_id}: explanation enhanced")

        # 3. Write back
        docstring = f'"""{task.get("title", "")} task."""\n\n'
        task_code = docstring + serialize_task(task) + "\n"
        fpath.write_text(task_code, encoding="utf-8")

    print("Done enhancing solutions!")
    print("Now run: python scripts/rebuild_all.py")


if __name__ == "__main__":
    main()
