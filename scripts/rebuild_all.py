"""
Mass-rebuild all 76 task files with:
1. Correct categories
2. Case-insensitive test fixes
3. Detailed comments in solutions
4. Theory explanations with LaTeX formulas
5. Diagrams (Mermaid / ASCII flow)
6. Sync problems.json + solutions.json + starters.json
"""

import json
import textwrap
from pathlib import Path

TASKS_DIR = Path(r"d:\BGWH_Code\pyre-code-main\torch_judge\tasks")
WEB_LIB = Path(r"d:\BGWH_Code\pyre-code-main\web\src\lib")

# Precise category mapping by task_id
CATEGORY_MAP = {
    # 激活函数
    "relu": "激活函数",
    "gelu": "激活函数",
    "swiglu": "激活函数",
    "softmax": "激活函数",
    # 归一化
    "batchnorm": "归一化",
    "layernorm": "归一化",
    "rmsnorm": "归一化",
    "adaln_zero": "归一化",
    # 注意力机制
    "attention": "注意力机制",
    "mha": "注意力机制",
    "gqa": "注意力机制",
    "mla": "注意力机制",
    "cross_attention": "注意力机制",
    "causal_attention": "注意力机制",
    "flash_attention": "注意力机制",
    "ring_attention": "注意力机制",
    "linear_attention": "注意力机制",
    "diff_attention": "注意力机制",
    "sliding_window": "注意力机制",
    "alibi": "注意力机制",
    "rope": "注意力机制",
    "ntk_rope": "位置编码与嵌入",
    # 位置编码与嵌入
    "embedding": "位置编码与嵌入",
    "sinusoidal_pe": "位置编码与嵌入",
    # 优化器与学习率
    "adam": "优化器与学习率",
    "gradient_clipping": "优化器与学习率",
    "gradient_accumulation": "优化器与学习率",
    "cosine_lr": "优化器与学习率",
    # 正则化
    "dropout": "正则化",
    "label_smoothing": "正则化",
    # 损失函数
    "cross_entropy": "损失函数",
    "focal_loss": "损失函数",
    "contrastive_loss": "损失函数",
    "dpo_loss": "损失函数",
    "ppo_loss": "损失函数",
    "grpo_loss": "损失函数",
    "gae": "损失函数",
    "flow_matching": "损失函数",
    # 扩散与流模型
    "ddim_step": "扩散与流模型",
    "noise_schedule": "扩散与流模型",
    # 量化
    "int8_quantization": "量化",
    "qlora": "量化",
    # 高效训练
    "fsdp_step": "高效训练",
    "mixed_precision": "高效训练",
    "tensor_parallel": "高效训练",
    "activation_checkpointing": "高效训练",
    # 图神经网络
    "gat_layer": "图神经网络",
    "gcn_layer": "图神经网络",
    "gin_layer": "图神经网络",
    "graphsage_layer": "图神经网络",
    "mpnn_layer": "图神经网络",
    "link_prediction": "图神经网络",
    "graph_readout": "图神经网络",
    # 采样与解码
    "topk_sampling": "采样与解码",
    "beam_search": "采样与解码",
    "mcts_search": "采样与解码",
    "speculative_decoding": "采样与解码",
    # Transformer组件
    "gpt2_block": "Transformer组件",
    "vit_block": "Transformer组件",
    "vit_patch": "Transformer组件",
    "mlp": "Transformer组件",
    "transformer_encoder": "Transformer组件",
    "transformer_decoder": "Transformer组件",
    # 推理优化
    "kv_cache": "推理优化",
    "paged_attention": "推理优化",
    # 基础网络组件
    "conv2d": "基础网络组件",
    "depthwise_conv": "基础网络组件",
    "max_pool2d": "基础网络组件",
    "linear": "基础网络组件",
    "linear_regression": "基础网络组件",
    "weight_init": "基础网络组件",
    # 参数高效微调
    "lora": "参数高效微调",
    # 分词
    "bpe": "分词",
    # 状态空间模型
    "mamba_ssm": "状态空间模型",
    # 混合专家模型
    "moe": "混合专家模型",
    "moe_load_balance": "混合专家模型",
    # 训练技巧
    "multi_token_prediction": "训练技巧",
    # 强化学习
    "reward_model": "强化学习",
    # 多模态
    "clip_model": "多模态",
}


def get_theory(task_id: str, task: dict) -> tuple[str, str]:
    """Generate theory explanation (en, zh) with LaTeX formulas."""
    title = task.get("title", "")
    desc = task.get("description_en", "")
    
    theories = {
        "mha": (
            "Multi-Head Attention splits the input into multiple heads, allowing the model to jointly attend to information from different representation subspaces.\n\n"
            "**Mathematical Formulation:**\n"
            "For each head $h$:\n"
            "$$\\text{head}_h = \\text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \\text{softmax}\\left(\\frac{QW_q^h (KW_k^h)^T}{\\sqrt{d_k}}\\right) VW_v^h$$\n"
            "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_H) W_o$$\n\n"
            "**Key Insight:**\n"
            "- $d_k = d_{model} / H$ ensures total computation stays constant\n"
            "- Each head can learn different attention patterns (syntax, semantics, long-range dependencies)\n"
            "- The scaling factor $1/\\sqrt{d_k}$ prevents softmax saturation for large $d_k$",
            
            "多头注意力将输入拆分为多个头，使模型能够联合关注来自不同表示子空间的信息。\n\n"
            "**数学公式：**\n"
            "对于每个头 $h$：\n"
            "$$\\text{head}_h = \\text{Attention}(QW_q^h, KW_k^h, VW_v^h) = \\text{softmax}\\left(\\frac{QW_q^h (KW_k^h)^T}{\\sqrt{d_k}}\\right) VW_v^h$$\n"
            "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_H) W_o$$\n\n"
            "**核心要点：**\n"
            "- $d_k = d_{model} / H$ 保证总计算量不变\n"
            "- 每个头可以学习不同的注意力模式（语法、语义、长距离依赖）\n"
            "- 缩放因子 $1/\\sqrt{d_k}$ 防止大 $d_k$ 时 softmax 饱和"
        ),
        "gqa": (
            "Grouped Query Attention reduces KV cache memory by sharing KV heads across groups of query heads.\n\n"
            "**Motivation:**\n"
            "In standard MHA with $H$ heads, KV cache stores $2 \\times B \\times S \\times H \\times d_k$ values.\n"
            "GQA with $K$ KV heads ($K < H$) stores only $2 \\times B \\times S \\times K \\times d_k$.\n\n"
            "**Memory Reduction:**\n"
            "$$\\text{Compression ratio} = \\frac{K}{H}$$\n\n"
            "**Trade-off:**\n"
            "- $K=1$: Maximum Memory Multi-Query Attention (MQA) — fastest, slightly lower quality\n"
            "- $1 < K < H$: GQA — balanced\n"
            "- $K=H$: Standard MHA — no compression\n\n"
            "KV heads are repeated $\\frac{H}{K}$ times via `repeat_interleave` before attention computation.",
            
            "分组查询注意力通过在查询头组之间共享 KV 头来减少 KV 缓存内存。\n\n"
            "**动机：**\n"
            "标准 MHA 有 $H$ 个头时，KV 缓存存储 $2 \\times B \\times S \\times H \\times d_k$ 个值。\n"
            "GQA 使用 $K$ 个 KV 头（$K < H$）只存储 $2 \\times B \\times S \\times K \\times d_k$。\n\n"
            "**内存压缩比：**\n"
            "$$\\text{压缩比} = \\frac{K}{H}$$\n\n"
            "**权衡：**\n"
            "- $K=1$：最大内存节省的多查询注意力（MQA）——最快，质量略降\n"
            "- $1 < K < H$：GQA —— 平衡\n"
            "- $K=H$：标准 MHA —— 无压缩\n\n"
            "KV 头通过 `repeat_interleave` 重复 $\\frac{H}{K}$ 次后参与注意力计算。"
        ),
        "cross_attention": (
            "Cross-Attention allows a decoder sequence to attend to an encoder sequence. Q comes from the decoder, while K and V come from the encoder.\n\n"
            "**Formula:**\n"
            "$$\\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \\text{softmax}\\left(\\frac{Q_{dec} K_{enc}^T}{\\sqrt{d_k}}\\right) V_{enc}$$\n\n"
            "**Applications:**\n"
            "- Machine Translation: decoder attends to source sentence\n"
            "- T2I models (Stable Diffusion): image tokens attend to text CLIP embeddings\n"
            "- Multimodal LLMs: vision tokens attend to language queries\n\n"
            "Unlike self-attention, cross-attention has no causal mask since encoder tokens are fully visible.",
            
            "交叉注意力允许解码器序列关注编码器序列。Q 来自解码器，K 和 V 来自编码器。\n\n"
            "**公式：**\n"
            "$$\\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \\text{softmax}\\left(\\frac{Q_{dec} K_{enc}^T}{\\sqrt{d_k}}\\right) V_{enc}$$\n\n"
            "**应用：**\n"
            "- 机器翻译：解码器关注源句子\n"
            "- 文生图模型（Stable Diffusion）：图像 token 关注文本 CLIP 嵌入\n"
            "- 多模态 LLM：视觉 token 关注语言查询\n\n"
            "与自注意力不同，交叉注意力不使用因果掩码，因为编码器 token 是完全可见的。"
        ),
        "adam": (
            "Adam combines momentum (first moment) and RMSProp (second moment) with bias correction.\n\n"
            "**Update Rules:**\n"
            "$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$\n"
            "$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n"
            "$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n"
            "$$\\theta_t = \\theta_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\n"
            "**Hyperparameters:**\n"
            "- $\\beta_1 = 0.9$ (momentum decay)\n"
            "- $\\beta_2 = 0.999$ (second moment decay)\n"
            "- $\\epsilon = 10^{-8}$ (numerical stability)\n\n"
            "Bias correction is critical in early steps when $m_t$ and $v_t$ are biased toward zero.",
            
            "Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），并进行偏差校正。\n\n"
            "**更新规则：**\n"
            "$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$\n"
            "$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n"
            "$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n"
            "$$\\theta_t = \\theta_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\n"
            "**超参数：**\n"
            "- $\\beta_1 = 0.9$（动量衰减）\n"
            "- $\\beta_2 = 0.999$（二阶矩衰减）\n"
            "- $\\epsilon = 10^{-8}$（数值稳定性）\n\n"
            "偏差校正在早期步骤至关重要，因为此时 $m_t$ 和 $v_t$ 偏向零。"
        ),
        "layernorm": (
            "Layer Normalization normalizes each sample independently across the feature dimension, unlike BatchNorm which normalizes across the batch.\n\n"
            "**Formula:**\n"
            "$$\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$\n"
            "where $\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i$ and $\\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2$\n\n"
            "**Why LayerNorm over BatchNorm in Transformers?**\n"
            "- Sequence lengths vary; batch statistics become unreliable\n"
            "- Independent per-sample normalization avoids batch-size dependencies\n"
            "- Pre-norm (LN before attention/FFN) is more stable than post-norm for deep networks",
            
            "层归一化对每个样本沿特征维度独立归一化，与 BatchNorm 沿批次维度归一化不同。\n\n"
            "**公式：**\n"
            "$$\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$\n"
            "其中 $\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i$，$\\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2$\n\n"
            "**为什么 Transformer 中使用 LayerNorm 而非 BatchNorm？**\n"
            "- 序列长度变化；批次统计量不可靠\n"
            "- 逐样本独立归一化避免依赖批大小\n"
            "- Pre-norm（在注意力/FFN 之前做 LN）比 post-norm 对深层网络更稳定"
        ),
        "rmsnorm": (
            "RMSNorm simplifies LayerNorm by removing mean centering, using only root-mean-square scaling.\n\n"
            "**Formula:**\n"
            "$$\\text{RMS}(x) = \\sqrt{\\frac{1}{D} \\sum_{i=1}^D x_i^2}$$\n"
            "$$\\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x) + \\epsilon} \\odot \\gamma$$\n\n"
            "**Comparison with LayerNorm:**\n"
            "- ~30% fewer FLOPs (no mean computation)\n"
            "- Empirically works as well as LayerNorm for most tasks\n"
            "- Used in LLaMA, Mistral, and other modern LLMs\n\n"
            "The key insight: mean subtraction may be unnecessary if the next layer (e.g., attention) is shift-invariant.",
            
            "RMSNorm 简化了 LayerNorm，去除了均值中心化，仅使用均方根缩放。\n\n"
            "**公式：**\n"
            "$$\\text{RMS}(x) = \\sqrt{\\frac{1}{D} \\sum_{i=1}^D x_i^2}$$\n"
            "$$\\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x) + \\epsilon} \\odot \\gamma$$\n\n"
            "**与 LayerNorm 对比：**\n"
            "- 计算量减少约 30%（无需计算均值）\n"
            "-  empirically 在大多数任务上与 LayerNorm 表现相当\n"
            "- 用于 LLaMA、Mistral 等现代 LLM\n\n"
            "核心洞见：如果下一层（如注意力）对平移不变，则均值减法可能是不必要的。"
        ),
        "softmax": (
            "Softmax converts a vector of real values into a probability distribution.\n\n"
            "**Formula:**\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^C e^{x_j}}$$\n\n"
            "**Numerical Stability:**\n"
            "Direct computation can overflow for large $x_i$. The stable version subtracts $\\max(x)$ before exponentiation:\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i - \\max(x)}}{\\sum_{j=1}^C e^{x_j - \\max(x)}}$$\n\n"
            "**Temperature Scaling:**\n"
            "$$\\text{softmax}_\\tau(x_i) = \\frac{e^{x_i / \\tau}}{\\sum_j e^{x_j / \\tau}}$$\n"
            "- $\\tau \\to 0$: sharp (argmax-like)\n"
            "- $\\tau \\to \\infty$: uniform distribution",
            
            "Softmax 将实数值向量转换为概率分布。\n\n"
            "**公式：**\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^C e^{x_j}}$$\n\n"
            "**数值稳定性：**\n"
            "大值 $x_i$ 的直接计算可能溢出。稳定版本在指数化前减去 $\\max(x)$：\n"
            "$$\\text{softmax}(x_i) = \\frac{e^{x_i - \\max(x)}}{\\sum_{j=1}^C e^{x_j - \\max(x)}}$$\n\n"
            "**温度缩放：**\n"
            "$$\\text{softmax}_\\tau(x_i) = \\frac{e^{x_i / \\tau}}{\\sum_j e^{x_j / \\tau}}$$\n"
            "- $\\tau \\to 0$：尖锐（接近 argmax）\n"
            "- $\\tau \\to \\infty$：均匀分布"
        ),
        "cross_entropy": (
            "Cross-Entropy Loss measures the difference between predicted probability distribution and true labels.\n\n"
            "**Formula:**\n"
            "$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^C y_{i,c} \\log(\\hat{y}_{i,c})$$\n\n"
            "For hard labels (one-hot $y$), this simplifies to:\n"
            "$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\log(\\hat{y}_{i, t_i})$$\n"
            "where $t_i$ is the true class index.\n\n"
            "**Log-Sum-Exp Trick:**\n"
            "To compute $\\log \\sum_j e^{x_j}$ stably:\n"
            "$$\\text{lse}(x) = \\max(x) + \\log \\sum_j e^{x_j - \\max(x)}$$\n"
            "This prevents overflow when logits are large.",
            
            "交叉熵损失衡量预测概率分布与真实标签之间的差异。\n\n"
            "**公式：**\n"
            "$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^C y_{i,c} \\log(\\hat{y}_{i,c})$$\n\n"
            "对于硬标签（one-hot $y$），简化为：\n"
            "$$\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{i=1}^N \\log(\\hat{y}_{i, t_i})$$\n"
            "其中 $t_i$ 是真实类别索引。\n\n"
            "**Log-Sum-Exp 技巧：**\n"
            "为稳定计算 $\\log \\sum_j e^{x_j}$：\n"
            "$$\\text{lse}(x) = \\max(x) + \\log \\sum_j e^{x_j - \\max(x)}$$\n"
            "这防止了大 logits 时的溢出问题。"
        ),
        "relu": (
            "ReLU (Rectified Linear Unit) is the most widely used activation function.\n\n"
            "**Formula:**\n"
            "$$\\text{ReLU}(x) = \\max(0, x)$$\n\n"
            "**Properties:**\n"
            "- Non-linear but piecewise linear\n"
            "- Computationally cheap (no exponentials or divisions)\n"
            "- Sparse activation: ~50% of neurons are inactive on average\n\n"
            "**The Dying ReLU Problem:**\n"
            "If a neuron's weights are updated such that it always outputs 0, gradients are zero and it never recovers."
            " Solutions: LeakyReLU, PReLU, ELU.",
            
            "ReLU（修正线性单元）是最广泛使用的激活函数。\n\n"
            "**公式：**\n"
            "$$\\text{ReLU}(x) = \\max(0, x)$$\n\n"
            "**性质：**\n"
            "- 非线性但分段线性\n"
            "- 计算廉价（无指数或除法）\n"
            "- 稀疏激活：平均约 50% 神经元不活跃\n\n"
            "**死亡 ReLU 问题：**\n"
            "如果神经元的权重更新导致其始终输出 0，则梯度为 0 且永远无法恢复。"
            "解决方案：LeakyReLU、PReLU、ELU。"
        ),
        "gelu": (
            "GELU (Gaussian Error Linear Unit) smoothly gates inputs by their probability under a Gaussian distribution.\n\n"
            "**Exact Formula:**\n"
            "$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2} \\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]$$\n\n"
            "**Approximation (faster):**\n"
            "$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left[\\sqrt{\\frac{2}{\\pi}} \\left(x + 0.044715x^3\\right)\\right]\\right)$$\n\n"
            "**Properties:**\n"
            "- Smooth everywhere (unlike ReLU's kink at 0)\n"
            "- Biased toward 0 for negative inputs (stochastic regularization effect)\n"
            "- Used in BERT, GPT, ViT, and most modern Transformers",
            
            "GELU（高斯误差线性单元）根据输入在高斯分布下的概率平滑地进行门控。\n\n"
            "**精确公式：**\n"
            "$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2} \\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]$$\n\n"
            "**近似公式（更快）：**\n"
            "$$\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left[\\sqrt{\\frac{2}{\\pi}} \\left(x + 0.044715x^3\\right)\\right]\\right)$$\n\n"
            "**性质：**\n"
            "- 处处光滑（不同于 ReLU 在 0 处的折点）\n"
            "- 负输入偏向 0（随机正则化效果）\n"
            "- 用于 BERT、GPT、ViT 和大多数现代 Transformer"
        ),
        "dropout": (
            "Dropout randomly sets a fraction of input elements to zero during training, preventing co-adaptation of neurons.\n\n"
            "**Training:**\n"
            "$$y = m \\odot x \\cdot \\frac{1}{1-p}$$\n"
            "where $m \\sim \\text{Bernoulli}(1-p)$\n\n"
            "**Inference:**\n"
            "$$y = x$$ (identity)\n\n"
            "**Inverted Dropout:**\n"
            "The scaling $1/(1-p)$ is applied during training, so no adjustment is needed at test time. This is the modern standard.\n\n"
            "**Why it works:**\n"
            "- Each training iteration uses a different 'thinned' network\n"
            "- At test time, averaging predictions over all $2^N$ possible networks approximates the geometric mean",
            
            "Dropout 在训练时随机将一部分输入元素置零，防止神经元共适应。\n\n"
            "**训练：**\n"
            "$$y = m \\odot x \\cdot \\frac{1}{1-p}$$\n"
            "其中 $m \\sim \\text{Bernoulli}(1-p)$\n\n"
            "**推理：**\n"
            "$$y = x$$（恒等映射）\n\n"
            "**Inverted Dropout：**\n"
            "缩放 $1/(1-p)$ 在训练时应用，因此测试时无需调整。这是现代标准做法。\n\n"
            "**工作原理：**\n"
            "- 每次训练迭代使用不同的'稀疏'网络\n"
            "- 测试时，对所有 $2^N$ 可能网络的预测取平均近似几何平均"
        ),
        "gpt2_block": (
            "GPT-2 uses a **pre-norm** transformer block: LayerNorm is applied *before* attention and MLP, with residual connections around both.\n\n"
            "**Architecture:**\n"
            "$$x = x + \\text{Attention}(\\text{LN}_1(x))$$\n"
            "$$x = x + \\text{MLP}(\\text{LN}_2(x))$$\n\n"
            "**Pre-norm vs Post-norm:**\n"
            "- Post-norm (original Transformer): $x = \\text{LN}(x + \\text{Attention}(x))$ — harder to train deep\n"
            "- Pre-norm: more stable gradients, allows training deeper networks (GPT-2, GPT-3, LLaMA)\n\n"
            "**MLP Structure:**\n"
            "$$\\text{MLP}(x) = W_2 \\cdot \\text{GELU}(W_1 x)$$\n"
            "where hidden dim = $4 \\times d_{model}$ (GPT-2 convention)",
            
            "GPT-2 使用 **pre-norm** Transformer 块：LayerNorm 在注意力和 MLP *之前* 应用，两者都有残差连接。\n\n"
            "**架构：**\n"
            "$$x = x + \\text{Attention}(\\text{LN}_1(x))$$\n"
            "$$x = x + \\text{MLP}(\\text{LN}_2(x))$$\n\n"
            "**Pre-norm vs Post-norm：**\n"
            "- Post-norm（原始 Transformer）：$x = \\text{LN}(x + \\text{Attention}(x))$ —— 深层训练困难\n"
            "- Pre-norm：梯度更稳定，允许训练更深网络（GPT-2、GPT-3、LLaMA）\n\n"
            "**MLP 结构：**\n"
            "$$\\text{MLP}(x) = W_2 \\cdot \\text{GELU}(W_1 x)$$\n"
            "其中隐藏维度 = $4 \\times d_{model}$（GPT-2 惯例）"
        ),
    }
    
    return theories.get(task_id, ("", ""))


def get_diagram(task_id: str) -> tuple[str, str]:
    """Generate Mermaid/ASCII diagram (en, zh)."""
    diagrams = {
        "mha": (
            "```mermaid\n"
            "flowchart LR\n"
            "    Q[Q] -->|W_q| QH[Q heads]\n"
            "    K[K] -->|W_k| KH[K heads]\n"
            "    V[V] -->|W_v| VH[V heads]\n"
            "    QH -->|scaled dot-product| ATTN[Attention weights]\n"
            "    KH --> ATTN\n"
            "    ATTN -->|@ V| OUT[Head outputs]\n"
            "    VH --> OUT\n"
            "    OUT -->|concat| CAT[Concatenated]\n"
            "    CAT -->|W_o| FINAL[Output]\n"
            "```",
            
            "```mermaid\n"
            "flowchart LR\n"
            "    Q[Q] -->|W_q| QH[Q 多头]\n"
            "    K[K] -->|W_k| KH[K 多头]\n"
            "    V[V] -->|W_v| VH[V 多头]\n"
            "    QH -->|缩放点积| ATTN[注意力权重]\n"
            "    KH --> ATTN\n"
            "    ATTN -->|@ V| OUT[头输出]\n"
            "    VH --> OUT\n"
            "    OUT -->|拼接| CAT[拼接结果]\n"
            "    CAT -->|W_o| FINAL[输出]\n"
            "```"
        ),
        "gqa": (
            "```mermaid\n"
            "flowchart LR\n"
            "    Q[Q] -->|W_q| QH[Query heads<br/>H heads]\n"
            "    K[K] -->|W_k| KH[KV heads<br/>K heads]\n"
            "    V[V] -->|W_v| VH[KV heads<br/>K heads]\n"
            "    KH -->|repeat_interleave<br/>H/K times| KHE[Expanded K heads]\n"
            "    VH -->|repeat_interleave<br/>H/K times| VHE[Expanded V heads]\n"
            "    QH -->|attention| OUT[Output]\n"
            "    KHE --> OUT\n"
            "    VHE --> OUT\n"
            "```",
            
            "```mermaid\n"
            "flowchart LR\n"
            "    Q[Q] -->|W_q| QH[查询头<br/>H 个头]\n"
            "    K[K] -->|W_k| KH[KV 头<br/>K 个头]\n"
            "    V[V] -->|W_v| VH[KV 头<br/>K 个头]\n"
            "    KH -->|repeat_interleave<br/>H/K 次| KHE[扩展 K 头]\n"
            "    VH -->|repeat_interleave<br/>H/K 次| VHE[扩展 V 头]\n"
            "    QH -->|注意力| OUT[输出]\n"
            "    KHE --> OUT\n"
            "    VHE --> OUT\n"
            "```"
        ),
        "cross_attention": (
            "```mermaid\n"
            "flowchart LR\n"
            "    subgraph Encoder\n"
            "        E[Encoder Output]\n"
            "    end\n"
            "    subgraph Decoder\n"
            "        D[Decoder Input] -->|W_q| Q[Q heads]\n"
            "    end\n"
            "    E -->|W_k| K[K heads]\n"
            "    E -->|W_v| V[V heads]\n"
            "    Q -->|scaled dot-product| A[Attention]\n"
            "    K --> A\n"
            "    A -->|@ V| O[Output]\n"
            "    V --> O\n"
            "```",
            
            "```mermaid\n"
            "flowchart LR\n"
            "    subgraph 编码器\n"
            "        E[编码器输出]\n"
            "    end\n"
            "    subgraph 解码器\n"
            "        D[解码器输入] -->|W_q| Q[Q 多头]\n"
            "    end\n"
            "    E -->|W_k| K[K 多头]\n"
            "    E -->|W_v| V[V 多头]\n"
            "    Q -->|缩放点积| A[注意力]\n"
            "    K --> A\n"
            "    A -->|@ V| O[输出]\n"
            "    V --> O\n"
            "```"
        ),
        "adam": (
            "```mermaid\n"
            "flowchart TD\n"
            "    G[Gradient g_t] --> M[First moment<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]\n"
            "    G --> V[Second moment<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]\n"
            "    M --> BC1[Bias correct<br/>m_hat = m_t / (1-beta1^t)]\n"
            "    V --> BC2[Bias correct<br/>v_hat = v_t / (1-beta2^t)]\n"
            "    BC1 --> U[Update<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]\n"
            "    BC2 --> U\n"
            "```",
            
            "```mermaid\n"
            "flowchart TD\n"
            "    G[梯度 g_t] --> M[一阶矩<br/>m_t = beta1*m_t-1 + (1-beta1)*g_t]\n"
            "    G --> V[二阶矩<br/>v_t = beta2*v_t-1 + (1-beta2)*g_t^2]\n"
            "    M --> BC1[偏差校正<br/>m_hat = m_t / (1-beta1^t)]\n"
            "    V --> BC2[偏差校正<br/>v_hat = v_t / (1-beta2^t)]\n"
            "    BC1 --> U[参数更新<br/>theta -= lr * m_hat / (sqrt(v_hat) + eps)]\n"
            "    BC2 --> U\n"
            "```"
        ),
        "layernorm": (
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> MEAN[Compute mean<br/>mu = mean(x, dim=-1)]\n"
            "    X --> VAR[Compute variance<br/>sigma^2 = var(x, dim=-1)]\n"
            "    MEAN --> NORM[Normalize<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]\n"
            "    VAR --> NORM\n"
            "    GAMMA[gamma] --> SCALE[Scale & Shift<br/>gamma * x_norm + beta]\n"
            "    BETA[beta] --> SCALE\n"
            "    NORM --> SCALE\n"
            "```",
            
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> MEAN[计算均值<br/>mu = mean(x, dim=-1)]\n"
            "    X --> VAR[计算方差<br/>sigma^2 = var(x, dim=-1)]\n"
            "    MEAN --> NORM[归一化<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]\n"
            "    VAR --> NORM\n"
            "    GAMMA[gamma] --> SCALE[缩放与偏移<br/>gamma * x_norm + beta]\n"
            "    BETA[beta] --> SCALE\n"
            "    NORM --> SCALE\n"
            "```"
        ),
        "rmsnorm": (
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> RMS[Compute RMS<br/>rms = sqrt(mean(x^2))]\n"
            "    RMS --> NORM[Normalize<br/>x / rms]\n"
            "    W[weight] --> SCALE[Scale<br/>x / rms * weight]\n"
            "    NORM --> SCALE\n"
            "    style RMS fill:#e1f5e1\n"
            "    note right of RMS\n"
            "        No mean subtraction!<br/>vs LayerNorm\n"
            "    end\n"
            "```",
            
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> RMS[计算 RMS<br/>rms = sqrt(mean(x^2))]\n"
            "    RMS --> NORM[归一化<br/>x / rms]\n"
            "    W[weight] --> SCALE[缩放<br/>x / rms * weight]\n"
            "    NORM --> SCALE\n"
            "    style RMS fill:#e1f5e1\n"
            "    note right of RMS\n"
            "        不减均值！<br/>与 LayerNorm 的区别\n"
            "    end\n"
            "```"
        ),
        "softmax": (
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> MAX[x_max = max(x)]\n"
            "    MAX --> SUB[Subtract: x - x_max]\n"
            "    SUB --> EXP[Exp: e^(x - x_max)]\n"
            "    EXP --> SUM[Sum: sum(e^(x - x_max))]\n"
            "    EXP --> DIV[Divide: exp / sum]\n"
            "    SUM --> DIV\n"
            "    DIV --> OUT[Softmax output]\n"
            "```",
            
            "```mermaid\n"
            "flowchart TD\n"
            "    X[x] --> MAX[x_max = max(x)]\n"
            "    MAX --> SUB[减去: x - x_max]\n"
            "    SUB --> EXP[指数: e^(x - x_max)]\n"
            "    EXP --> SUM[求和: sum(e^(x - x_max))]\n"
            "    EXP --> DIV[相除: exp / sum]\n"
            "    SUM --> DIV\n"
            "    DIV --> OUT[Softmax 输出]\n"
            "```"
        ),
        "gpt2_block": (
            "```mermaid\n"
            "flowchart TD\n"
            "    IN[Input x] --> LN1[LayerNorm]\n"
            "    LN1 --> ATT[Masked Self-Attention]\n"
            "    ATT --> ADD1[x + Attention]\n"
            "    IN --> ADD1\n"
            "    ADD1 --> LN2[LayerNorm]\n"
            "    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]\n"
            "    MLP --> ADD2[x + MLP]\n"
            "    ADD1 --> ADD2\n"
            "    ADD2 --> OUT[Output]\n"
            "```",
            
            "```mermaid\n"
            "flowchart TD\n"
            "    IN[输入 x] --> LN1[LayerNorm]\n"
            "    LN1 --> ATT[掩码自注意力]\n"
            "    ATT --> ADD1[x + Attention]\n"
            "    IN --> ADD1\n"
            "    ADD1 --> LN2[LayerNorm]\n"
            "    LN2 --> MLP[MLP<br/>Linear -> GELU -> Linear]\n"
            "    MLP --> ADD2[x + MLP]\n"
            "    ADD1 --> ADD2\n"
            "    ADD2 --> OUT[输出]\n"
            "```"
        ),
    }
    return diagrams.get(task_id, ("", ""))


def add_comments_to_solution(task_id: str, solution: str, task: dict) -> str:
    """Add structured comments to solution code."""
    if not solution:
        return solution
    
    fn_name = task.get("function_name", "")
    title = task.get("title", "")
    
    # Check if already has comments
    if '"""' in solution or "'''" in solution or solution.strip().startswith('#'):
        return solution
    
    # Build comment block
    comment_lines = [
        f'    """',
        f'    {title} implementation.',
        f'    """',
    ]
    
    # For class-based solutions, add comments to __init__ and forward
    lines = solution.split('\n')
    new_lines = []
    in_init = False
    in_forward = False
    indent_level = 4
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('class '):
            new_lines.append(line)
            new_lines.append(f'    """')
            new_lines.append(f'    {title} module.')
            new_lines.append(f'    """')
            continue
        
        if stripped.startswith('def __init__'):
            in_init = True
            in_forward = False
            new_lines.append(line)
            new_lines.append(f'        # Initialize layers and parameters')
            continue
        
        if stripped.startswith('def forward') or stripped.startswith('def __call__') or stripped.startswith('def step'):
            in_init = False
            in_forward = True
            new_lines.append(line)
            new_lines.append(f'        # Compute forward pass')
            continue
        
        # Add inline comments for key operations
        if 'nn.Linear' in stripped and '#' not in stripped:
            line = line + '  # Linear projection'
        elif 'softmax' in stripped and '#' not in stripped:
            line = line + '  # Apply softmax to get attention weights'
        elif 'matmul' in stripped and '#' not in stripped:
            line = line + '  # Matrix multiplication'
        elif 'LayerNorm' in stripped and '#' not in stripped:
            line = line + '  # Layer normalization'
        elif 'Dropout' in stripped and '#' not in stripped:
            line = line + '  # Dropout for regularization'
        elif 'relu' in stripped.lower() and '#' not in stripped:
            line = line + '  # ReLU activation'
        elif 'gelu' in stripped.lower() and '#' not in stripped:
            line = line + '  # GELU activation'
        elif 'contiguous().view' in stripped and '#' not in stripped:
            line = line + '  # Reshape: merge heads back'
        elif '.view(' in stripped and 'transpose' in stripped and '#' not in stripped:
            line = line + '  # Reshape to (B, heads, S, d_k)'
        elif '.transpose(1, 2)' in stripped and '#' not in stripped:
            line = line + '  # Transpose to (B, heads, S, d_k)'
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def fix_case_sensitive_tests(task_id: str, tests: list) -> list:
    """Fix tests that hardcode attribute names like W_q/w_q."""
    if task_id not in ("mha", "gqa", "cross_attention"):
        return tests
    
    fixed_tests = []
    for test in tests:
        code = test.get("code", "")
        name = test.get("name", "")
        
        # For attribute checks, add fallback to lowercase
        if "W_q" in code or "W_k" in code or "W_v" in code or "W_o" in code:
            # Add helper to check both cases
            helper = """
# Helper: check attribute with case fallback
def _get_attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, name.lower()):
        return getattr(obj, name.lower())
    raise AttributeError(f"Need {name} or {name.lower()}")
"""
            # Actually, simpler: replace getattr(mha, 'W_q') with case-insensitive version
            # But this is risky. Better approach: just ensure the error message is clear.
            pass
        
        fixed_tests.append({"name": name, "code": code})
    
    return fixed_tests


def serialize_task(task: dict) -> str:
    """Serialize TASK dict to Python code."""
    # We use json.dumps for the dict content but preserve Python docstring style
    # Actually, let's build it manually for better formatting
    lines = ['TASK = {']
    
    key_order = [
        "title", "title_zh", "difficulty", "category",
        "description_en", "description_zh",
        "function_name", "hint", "hint_zh",
        "theory_en", "theory_zh", "diagram_en", "diagram_zh",
        "tests", "solution", "demo"
    ]
    
    for key in key_order:
        if key not in task:
            continue
        val = task[key]
        
        if key == "tests":
            lines.append(f'    "{key}": [')
            for test in val:
                lines.append('        {')
                lines.append(f'            "name": {json.dumps(test.get("name", "basic"), ensure_ascii=False)},')
                code = test.get("code", "")
                # Use triple quotes for multiline code
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
    print("Starting mass rebuild of all tasks...")
    
    all_problems = []
    all_solutions = {}
    all_starters = {}
    
    for fpath in sorted(TASKS_DIR.glob("*.py")):
        task_id = fpath.stem
        if task_id.startswith("_"):
            continue
        
        print(f"Processing {task_id}...")
        
        # Read TASK
        source = fpath.read_text(encoding="utf-8")
        
        # Mock torch for safe exec
        class _MockTensor:
            pass
        class _MockModule:
            pass
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
        
        namespace = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "torch": _MockTorch(),
            "nn": _MockNn(),
        }
        try:
            exec(compile(source, str(fpath), "exec"), namespace)
        except Exception as e:
            print(f"  WARN: exec failed for {task_id}: {e}")
            continue
        
        task = namespace.get("TASK")
        if task is None:
            print(f"  WARN: no TASK in {task_id}")
            continue
        
        # 1. Add category
        category = CATEGORY_MAP.get(task_id)
        if category:
            task["category"] = category
        
        # 2. Add theory and diagram
        theory_en, theory_zh = get_theory(task_id, task)
        if theory_en:
            task["theory_en"] = theory_en
            task["theory_zh"] = theory_zh
        
        diagram_en, diagram_zh = get_diagram(task_id)
        if diagram_en:
            task["diagram_en"] = diagram_en
            task["diagram_zh"] = diagram_zh
        
        # 3. Fix case-sensitive tests
        if task_id in ("mha", "gqa", "cross_attention"):
            tests = task.get("tests", [])
            for test in tests:
                code = test.get("code", "")
                # Replace hardcoded attr access with case-insensitive helper
                # Instead of getattr(mha, 'W_q'), use a pattern that checks both cases
                # We'll use a simple approach: add a case-insensitive getattr wrapper
                if "getattr(" in code and "W_" in code:
                    # Already uses getattr, make it case-insensitive
                    code = code.replace(
                        "getattr(mha, name)",
                        "(getattr(mha, name) if hasattr(mha, name) else getattr(mha, name.lower()))"
                    )
                    code = code.replace(
                        "getattr(gqa, name)",
                        "(getattr(gqa, name) if hasattr(gqa, name) else getattr(gqa, name.lower()))"
                    )
                    code = code.replace(
                        "getattr(attn, name)",
                        "(getattr(attn, name) if hasattr(attn, name) else getattr(attn, name.lower()))"
                    )
                    test["code"] = code
            
            # Also fix direct attribute access like mha.W_q
            for test in tests:
                code = test.get("code", "")
                # Replace direct access with hasattr fallback
                for attr in ["W_q", "W_k", "W_v", "W_o"]:
                    for obj in ["mha", "gqa", "attn"]:
                        old = f"{obj}.{attr}"
                        new = f"({obj}.{attr} if hasattr({obj}, '{attr}') else {obj}.{attr.lower()})"
                        code = code.replace(old, new)
                test["code"] = code
        
        # 4. Add comments to solution
        sol = task.get("solution", "")
        if sol:
            task["solution"] = add_comments_to_solution(task_id, sol, task)
        
        # 5. Write back task file
        docstring = f'"""{task.get("title", "")} task."""\n\n'
        task_code = docstring + serialize_task(task) + "\n"
        fpath.write_text(task_code, encoding="utf-8")
        
        # 6. Build problems.json entry
        prob = {
            "id": task_id,
            "title": task.get("title", ""),
            "titleZh": task.get("title_zh", ""),
            "difficulty": task.get("difficulty", ""),
            "functionName": task.get("function_name", ""),
            "hint": task.get("hint", ""),
            "hintZh": task.get("hint_zh", ""),
            "descriptionEn": task.get("description_en", ""),
            "descriptionZh": task.get("description_zh", ""),
            "category": task.get("category", ""),
            "theoryEn": task.get("theory_en", ""),
            "theoryZh": task.get("theory_zh", ""),
            "diagramEn": task.get("diagram_en", ""),
            "diagramZh": task.get("diagram_zh", ""),
            "tests": task.get("tests", []),
        }
        all_problems.append(prob)
        
        # 7. Build solutions.json entry
        cells = []
        if sol:
            cells.append({"type": "code", "source": sol.strip(), "role": "solution"})
        
        demo = task.get("demo", "")
        if demo:
            cells.append({"type": "code", "source": demo.strip(), "role": "demo"})
        
        # Use enhanced explanation if available, fallback to theory
        explanation = task.get("explanation", "") or theory_en
        if explanation:
            cells.append({"type": "markdown", "source": explanation.strip(), "role": "explanation"})
        
        if cells:
            all_solutions[task_id] = {"cells": cells}
        
        # 8. Build starters entry
        fn = task.get("function_name", "")
        starter = f"def {fn}(...):\n    pass"
        all_starters[task_id] = starter
    
    # Write problems.json
    problems_json = {"problems": all_problems}
    (WEB_LIB / "problems.json").write_text(
        json.dumps(problems_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote problems.json with {len(all_problems)} problems")
    
    # Write solutions.json
    (WEB_LIB / "solutions.json").write_text(
        json.dumps(all_solutions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote solutions.json with {len(all_solutions)} entries")
    
    # Write starters.json
    (WEB_LIB / "starters.json").write_text(
        json.dumps(all_starters, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote starters.json with {len(all_starters)} entries")
    
    print("Done!")


if __name__ == "__main__":
    main()
