# -*- coding: utf-8 -*-
"""Generate 20 new foundational tasks for Pyre Code."""

import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.rebuild_all import serialize_task

TASKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'torch_judge', 'tasks')

def write_task(task_id: str, task: dict):
    path = os.path.join(TASKS_DIR, f"{task_id}.py")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(serialize_task(task))
    print(f"  Written: {path}")


def make_task(title, title_zh, difficulty, category, desc_en, desc_zh,
              func_name, hint, hint_zh, theory_en, theory_zh,
              tests, solution, demo="", explanation=""):
    return {
        "title": title,
        "title_zh": title_zh,
        "difficulty": difficulty,
        "category": category,
        "description_en": desc_en,
        "description_zh": desc_zh,
        "function_name": func_name,
        "hint": hint,
        "hint_zh": hint_zh,
        "theory_en": theory_en,
        "theory_zh": theory_zh,
        "tests": tests,
        "solution": solution,
        "demo": demo,
        "explanation": explanation,
    }

# ========================================================================
# 1. L1 Regularization
# ========================================================================
write_task("l1_regularization", make_task(
    title="L1 Regularization (Lasso)",
    title_zh="L1 正则化（Lasso）",
    difficulty="Easy",
    category="正则化",
    desc_en="Implement L1 regularization. Given a weight tensor `w`, compute the L1 penalty as the sum of absolute values: L1 = λ * Σ|w_i|. Returns the penalty (scalar).",
    desc_zh="实现 L1 正则化。给定权重张量 `w`，计算绝对值之和作为惩罚项：L1 = λ * Σ|w_i|。返回标量惩罚值。",
    func_name="l1_regularization",
    hint="Use `torch.abs()` and `.sum()`.",
    hint_zh="使用 `torch.abs()` 和 `.sum()`。",
    theory_en="L1 regularization adds the sum of absolute values of weights to the loss. It promotes sparsity (many weights become exactly zero).",
    theory_zh="L1 正则化将权重绝对值之和加入损失函数，促进稀疏性（许多权重精确为0）。",
    tests=[
        {"inputs": [{"shape": [4], "dtype": "float32", "value": [1.0, -2.0, 3.0, -4.0]}, 0.1], "expected": 1.0},
        {"inputs": [{"shape": [2, 3], "dtype": "float32", "value": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, 0.5], "expected": 0.0},
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [-1.5, 2.5, -3.0]}, 0.2], "expected": 1.4},
    ],
    solution=textwrap.dedent('''\
        import torch

        def l1_regularization(w: torch.Tensor, lambda_reg: float) -> torch.Tensor:
            """
            Compute L1 regularization penalty.
            L1 = lambda_reg * sum(|w_i|)
            """
            # 计算所有权重元素的绝对值之和
            # Compute the sum of absolute values of all weight elements
            penalty = lambda_reg * torch.abs(w).sum()
            return penalty
    '''),
    explanation=textwrap.dedent('''\
        **L1 Regularization 详解**

        **核心思想:**
        L1 正则化（Lasso）通过对权重的绝对值之和施加惩罚，促使模型产生稀疏权重矩阵。

        **Step 1 — 计算绝对值:**
        - `torch.abs(w)` 将每个权重变为非负数
        - 负数权重（如 -2.0）贡献 2.0 的惩罚

        **Step 2 — 求和并缩放:**
        - `.sum()` 累加所有元素的绝对值
        - 乘以 `lambda_reg` 控制正则化强度
        - 数学表达: L1 = λ * Σ_i |w_i|

        **为什么L1产生稀疏性:**
        - L1 在 w=0 处的次梯度包含 [-λ, +λ] 区间
        - 当梯度小于 λ 时，权重会被推到精确的 0
        - 这使得 L1 能自动做特征选择

        **面试考点:**
        - L1 vs L2 的区别？→ L1 产生稀疏解，L2 产生平滑小权重
        - L1 的几何解释？→ 菱形约束域与椭圆损失函数的切点常在坐标轴上
        - 为什么 L1 在零点不可导？→ 可用次梯度或近似（如 smooth L1）处理
    '''),
))

# ========================================================================
# 2. L2 Regularization (Weight Decay)
# ========================================================================
write_task("l2_regularization", make_task(
    title="L2 Regularization (Weight Decay)",
    title_zh="L2 正则化（Weight Decay）",
    difficulty="Easy",
    category="正则化",
    desc_en="Implement L2 regularization (weight decay). Given a weight tensor `w`, compute the L2 penalty as half the sum of squared values: L2 = 0.5 * λ * Σw_i². Returns the penalty (scalar). The 0.5 factor simplifies the gradient to λ*w.",
    desc_zh="实现 L2 正则化（Weight Decay）。给定权重张量 `w`，计算平方和的一半作为惩罚项：L2 = 0.5 * λ * Σw_i²。返回标量惩罚值。0.5 因子使梯度简化为 λ*w。",
    func_name="l2_regularization",
    hint="Use `torch.pow()` or `**2`, then `.sum()`. Multiply by 0.5 * lambda.",
    hint_zh="使用 `torch.pow()` 或 `**2`，然后 `.sum()`，再乘以 0.5 * λ。",
    theory_en="L2 regularization penalizes large weights by adding the sum of squared weights to the loss. It shrinks all weights uniformly toward zero but rarely makes them exactly zero.",
    theory_zh="L2 正则化通过惩罚权重的平方和来抑制大权重。它均匀地将所有权重向零收缩，但很少精确为零。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, 0.1], "expected": 0.7},
        {"inputs": [{"shape": [2, 2], "dtype": "float32", "value": [[0.0, 0.0], [0.0, 0.0]]}, 0.5], "expected": 0.0},
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [3.0, 4.0]}, 0.2], "expected": 2.5},
    ],
    solution=textwrap.dedent('''\
        import torch

        def l2_regularization(w: torch.Tensor, lambda_reg: float) -> torch.Tensor:
            """
            Compute L2 regularization penalty (weight decay).
            L2 = 0.5 * lambda_reg * sum(w_i^2)
            The 0.5 factor makes gradient = lambda_reg * w.
            """
            # 计算权重平方和的一半，乘以正则化系数
            # Compute half the sum of squared weights, scaled by lambda
            penalty = 0.5 * lambda_reg * torch.pow(w, 2).sum()
            return penalty
    '''),
    explanation=textwrap.dedent('''\
        **L2 Regularization (Weight Decay) 详解**

        **核心思想:**
        L2 正则化通过对权重的平方和施加惩罚，抑制过大的权重值，提升模型泛化能力。

        **Step 1 — 计算平方:**
        - `torch.pow(w, 2)` 或 `w ** 2` 逐元素平方
        - 大权重被平方后惩罚更重（二次增长）

        **Step 2 — 求和并缩放:**
        - `.sum()` 累加所有平方值
        - 乘以 `0.5 * lambda_reg`
        - 数学表达: L2 = 0.5 * λ * Σ_i w_i²

        **为什么用 0.5 因子:**
        - 损失对 w 的导数: ∂L2/∂w = λ * w
        - 没有 0.5 的话导数是 2λ*w，更新公式不简洁
        - 这是 PyTorch optimizers 中 weight_decay 的标准定义

        **与 AdamW 的关系:**
        - AdamW 将 weight decay 直接作用于参数更新，而非加入损失梯度
        - 这使得正则化效果与梯度缩放解耦，比 L2 在 Adam 中效果更好

        **面试考点:**
        - L1 vs L2？→ L1 稀疏（特征选择），L2 平滑（防止过拟合）
        - Weight Decay 和 L2 正则化的区别？→ 在 SGD 中等价，在 Adam 中 AdamW 更优
        - 为什么 L2 不导致稀疏？→ 平方函数在零点附近梯度很小，无法将权重推到精确零
    '''),
))

# ========================================================================
# 3. Sigmoid
# ========================================================================
write_task("sigmoid", make_task(
    title="Sigmoid Activation",
    title_zh="Sigmoid 激活函数",
    difficulty="Easy",
    category="激活函数",
    desc_en="Implement the sigmoid activation function: σ(x) = 1 / (1 + exp(-x)). Use numerically stable computation for large negative inputs (return values in (0, 1)).",
    desc_zh="实现 Sigmoid 激活函数：σ(x) = 1 / (1 + exp(-x))。对大负数输入使用数值稳定计算，返回值在 (0, 1) 区间。",
    func_name="sigmoid",
    hint="For stability with large negative x, use `z = torch.exp(-torch.abs(x))` pattern, or clamp inputs before exp.",
    hint_zh="对大负数 x，先用 `torch.abs(x)` 取绝对值再做 exp，或对输入做 clamp。",
    theory_en="Sigmoid squashes any real number to (0, 1), making it useful for binary classification output layers. However, it suffers from vanishing gradient when |x| is large.",
    theory_zh="Sigmoid 将任意实数压缩到 (0, 1) 区间，适用于二分类输出层。但在 |x| 很大时会出现梯度消失问题。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [0.0, 1.0, -1.0]}], "expected": [0.5, 0.7310586, 0.2689414]},
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [10.0, -10.0]}], "expected": [0.9999546, 4.539787e-05]},
        {"inputs": [{"shape": [1], "dtype": "float32", "value": [0.0]}], "expected": [0.5]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def sigmoid(x: torch.Tensor) -> torch.Tensor:
            """
            Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x))
            For x >= 0: compute as 1 / (1 + exp(-x))
            For x < 0:  compute as exp(x) / (1 + exp(x)) to avoid overflow.
            """
            # 数值稳定的实现：根据输入符号选择计算路径
            # Numerically stable: choose computation path based on sign
            return torch.where(
                x >= 0,
                1.0 / (1.0 + torch.exp(-x)),
                torch.exp(x) / (1.0 + torch.exp(x))
            )
    '''),
    explanation=textwrap.dedent('''\
        **Sigmoid 详解**

        **核心思想:**
        Sigmoid 将实数映射到 (0, 1)，可解释为概率。

        **Step 1 — 数值稳定性处理:**
        - 当 x 是很大的正数时，exp(-x) → 0，直接计算安全
        - 当 x 是很大的负数时，exp(-x) 会溢出，改为 exp(x) / (1 + exp(x))
        - `torch.where` 根据符号选择计算路径

        **数学表达:**
        σ(x) = 1 / (1 + e^{-x}) = e^x / (1 + e^x)

        **梯度消失问题:**
        - 导数: σ\'(x) = σ(x) * (1 - σ(x))
        - 当 x → ±∞ 时，σ\'(x) → 0
        - 深层网络中梯度逐层衰减，导致前面层难以训练

        **面试考点:**
        - Sigmoid 为什么导致梯度消失？→ 导数最大值只有 0.25，多层连乘后指数级衰减
        - Sigmoid vs Tanh？→ Tanh 零中心化（输出 [-1,1]），梯度消失稍轻
        - 什么时候还用 Sigmoid？→ 二分类输出层、LSTM 门控、 attention 门控
    '''),
))

# ========================================================================
# 4. Tanh
# ========================================================================
write_task("tanh", make_task(
    title="Tanh Activation",
    title_zh="Tanh 激活函数",
    difficulty="Easy",
    category="激活函数",
    desc_en="Implement the hyperbolic tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Output range is (-1, 1). Use numerically stable computation.",
    desc_zh="实现双曲正切激活函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))。输出范围为 (-1, 1)。使用数值稳定计算。",
    func_name="tanh",
    hint="For large |x|, tanh(x) approaches ±1. You can also use the identity tanh(x) = 2*sigmoid(2x) - 1.",
    hint_zh="|x| 很大时 tanh(x) → ±1。也可以用恒等式 tanh(x) = 2*σ(2x) - 1。",
    theory_en="Tanh is zero-centered (output mean ≈ 0), which helps gradients flow better than sigmoid in hidden layers. It still suffers from vanishing gradient for large |x|.",
    theory_zh="Tanh 是零中心化的（输出均值≈0），比 Sigmoid 更适合隐藏层。但在 |x| 很大时仍有梯度消失问题。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [0.0, 1.0, -1.0]}], "expected": [0.0, 0.7615942, -0.7615942]},
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [5.0, -5.0]}], "expected": [0.9999092, -0.9999092]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def tanh(x: torch.Tensor) -> torch.Tensor:
            """
            Hyperbolic tangent: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
            Zero-centered activation with range (-1, 1).
            """
            # 使用 torch.tanh 是最佳实践（内部有CUDNN优化）
            # Using torch.tanh is best practice (CUDNN optimized)
            return torch.tanh(x)
    '''),
    explanation=textwrap.dedent('''\
        **Tanh 详解**

        **核心思想:**
        Tanh 将实数映射到 (-1, 1)，是零中心化激活函数。

        **数学表达:**
        tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}) = 2σ(2x) - 1

        **零中心化的优势:**
        - Sigmoid 输出均值 ≈ 0.5，下一层输入总是正数
        - Tanh 输出均值 ≈ 0，下一层权重更新方向更均衡
        - 收敛速度通常比 Sigmoid 快

        **与 Sigmoid 对比:**
        | 特性 | Sigmoid | Tanh |
        |------|---------|------|
        | 输出范围 | (0, 1) | (-1, 1) |
        | 零中心化 | 否 | 是 |
        | 梯度消失 | 严重 | 较轻 |
        | 适用位置 | 输出层 | 隐藏层 |

        **面试考点:**
        - 为什么隐藏层用 Tanh 不用 Sigmoid？→ 零中心化，梯度流更好
        - Tanh 的导数？→ 1 - tanh²(x)，计算效率高（前向传播时缓存输出即可）
        - 为什么现在用 ReLU/GELU 更多？→ 避免梯度消失，计算更快
    '''),
))

# ========================================================================
# 5. MSE Loss
# ========================================================================
write_task("mse_loss", make_task(
    title="Mean Squared Error (MSE)",
    title_zh="均方误差损失（MSE）",
    difficulty="Easy",
    category="损失函数",
    desc_en="Implement mean squared error loss: MSE = (1/n) * Σ(pred_i - target_i)². Compute element-wise squared difference, then mean over all elements.",
    desc_zh="实现均方误差损失：MSE = (1/n) * Σ(pred_i - target_i)²。先逐元素计算差的平方，再对所有元素求平均。",
    func_name="mse_loss",
    hint="Use `(pred - target) ** 2` then `.mean()`.",
    hint_zh="使用 `(pred - target) ** 2` 然后 `.mean()`。",
    theory_en="MSE penalizes large errors quadratically. It is the standard loss for regression tasks. The gradient is linear in the error, making optimization stable.",
    theory_zh="MSE 对大误差的惩罚是二次增长。是回归任务的标准损失函数。梯度与误差线性相关，优化过程稳定。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, {"shape": [3], "dtype": "float32", "value": [1.5, 2.5, 2.5]}], "expected": 0.25},
        {"inputs": [{"shape": [2, 2], "dtype": "float32", "value": [[0.0, 0.0], [0.0, 0.0]]}, {"shape": [2, 2], "dtype": "float32", "value": [[0.0, 0.0], [0.0, 0.0]]}], "expected": 0.0},
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [3.0, 4.0]}, {"shape": [2], "dtype": "float32", "value": [1.0, 1.0]}], "expected": 6.5},
    ],
    solution=textwrap.dedent('''\
        import torch

        def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            Mean Squared Error: MSE = mean((pred - target)^2)
            Standard regression loss.
            """
            # 计算预测值与目标值的逐元素差，平方后取平均
            # Element-wise difference, square, then mean
            loss = torch.pow(pred - target, 2).mean()
            return loss
    '''),
    explanation=textwrap.dedent('''\
        **MSE Loss 详解**

        **核心思想:**
        均方误差衡量预测值与真实值之间的平均平方差距。

        **Step 1 — 计算误差:**
        - `pred - target` 逐元素差
        - 正负误差统一处理（平方后都为正）

        **Step 2 — 平方并平均:**
        - `.pow(2)` 逐元素平方，放大大的误差
        - `.mean()` 对所有元素求平均
        - 数学表达: MSE = (1/n) Σ_i (ŷ_i - y_i)²

        **为什么回归用 MSE:**
        - 梯度 = 2*(ŷ - y)/n，与误差大小成正比
        - 对异常值敏感（大误差被平方放大），也可用 MAE/Huber 替代
        - 数学上对应高斯分布的负对数似然

        **面试考点:**
        - MSE vs MAE？→ MSE 对大误差更敏感，MAE 更鲁棒
        - MSE 的梯度特性？→ 线性梯度，优化稳定
        - 为什么分类不用 MSE？→ 分类输出是概率，交叉熵更匹配分布假设
    '''),
))

# ========================================================================
# 6. Binary Cross Entropy
# ========================================================================
write_task("binary_cross_entropy", make_task(
    title="Binary Cross-Entropy Loss",
    title_zh="二元交叉熵损失",
    difficulty="Easy",
    category="损失函数",
    desc_en="Implement binary cross-entropy loss with logits: BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))], where z are logits and y are binary targets in {0,1}. Use numerically stable computation via `logsigmoid`.",
    desc_zh="实现带 logits 的二元交叉熵损失：BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))]，其中 z 是 logits，y 是 {0,1} 二元目标。使用 `logsigmoid` 做数值稳定计算。",
    func_name="binary_cross_entropy",
    hint="Use `F.logsigmoid(z)` for log(σ(z)) and `F.logsigmoid(-z)` for log(1-σ(z)). Avoid computing sigmoid explicitly.",
    hint_zh="使用 `F.logsigmoid(z)` 计算 log(σ(z))，`F.logsigmoid(-z)` 计算 log(1-σ(z))，避免显式计算 sigmoid。",
    theory_en="Binary cross-entropy measures the divergence between predicted probabilities and binary targets. Using logits directly (with logsigmoid) avoids numerical instability from computing exp of large numbers.",
    theory_zh="二元交叉熵衡量预测概率与二元目标的差异。直接使用 logits（配合 logsigmoid）可避免大数 exp 的数值不稳定问题。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [0.0, 1.0, -1.0]}, {"shape": [3], "dtype": "float32", "value": [1.0, 1.0, 0.0]}], "expected": 0.475911},  # approx
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [10.0, -10.0]}, {"shape": [2], "dtype": "float32", "value": [1.0, 0.0]}], "expected": 4.539889e-05},
    ],
    solution=textwrap.dedent('''\
        import torch
        import torch.nn.functional as F

        def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Binary cross-entropy with logits (numerically stable).
            BCE = -mean[y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))]
            Using logsigmoid avoids explicit sigmoid computation.
            """
            # 使用 logsigmoid 实现数值稳定的 BCE
            # log(σ(z)) = logsigmoid(z), log(1-σ(z)) = logsigmoid(-z)
            log_p = F.logsigmoid(logits)
            log_1p = F.logsigmoid(-logits)
            loss = -(targets * log_p + (1.0 - targets) * log_1p).mean()
            return loss
    '''),
    explanation=textwrap.dedent('''\
        **Binary Cross-Entropy 详解**

        **核心思想:**
        二元交叉熵衡量二分类预测概率分布与真实标签之间的差异。

        **Step 1 — 数值稳定计算:**
        - 避免先算 σ(z) 再取 log（大负数时 σ(z)→0，log(0) 下溢）
        - `F.logsigmoid(z)` 直接计算 log(σ(z))，内部做了数值稳定处理
        - `F.logsigmoid(-z)` 对应 log(1-σ(z))

        **Step 2 — 组合损失:**
        - y=1 时，只保留 log(σ(z)) 项
        - y=0 时，只保留 log(1-σ(z)) 项
        - 取负号（最小化损失 = 最大化似然）并求平均

        **数学表达:**
        L = -(1/N) Σ [y_i log(σ(z_i)) + (1-y_i) log(1-σ(z_i))]

        **面试考点:**
        - 为什么用 logits 而不是 probabilities？→ 数值稳定性
        - BCE 与 MSE 在二分类中的区别？→ BCE 梯度在错误预测时更大，收敛更快
        - 处理类别不平衡？→ 用 pos_weight 或 focal loss 加权
    '''),
))

# ========================================================================
# 7. KL Divergence
# ========================================================================
write_task("kl_divergence", make_task(
    title="KL Divergence",
    title_zh="KL 散度",
    difficulty="Medium",
    category="损失函数",
    desc_en="Implement KL divergence between two probability distributions p (target) and q (predicted): KL(p||q) = Σ p_i * (log(p_i) - log(q_i)). Both p and q are probability distributions (sum to 1). Add a small epsilon (1e-10) to avoid log(0).",
    desc_zh="实现两个概率分布 p（目标）和 q（预测）之间的 KL 散度：KL(p||q) = Σ p_i * (log(p_i) - log(q_i))。p 和 q 都是概率分布（和为1）。添加小 epsilon（1e-10）避免 log(0)。",
    func_name="kl_divergence",
    hint="Clip q with epsilon before log. Use `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`.",
    hint_zh="对 q 做 epsilon 裁剪后再取 log。使用 `(p * (torch.log(p + eps) - torch.log(q + eps))).sum()`。",
    theory_en="KL divergence measures how much one probability distribution q diverges from a reference distribution p. It is non-negative and zero only when p = q. Widely used in VAE, diffusion models, and knowledge distillation.",
    theory_zh="KL 散度衡量概率分布 q 相对于参考分布 p 的差异。非负，且仅当 p=q 时为0。广泛用于 VAE、扩散模型和知识蒸馏。",
    tests=[
        {"inputs": [{"shape": [4], "dtype": "float32", "value": [0.25, 0.25, 0.25, 0.25]}, {"shape": [4], "dtype": "float32", "value": [0.25, 0.25, 0.25, 0.25]}], "expected": 0.0},
        {"inputs": [{"shape": [2], "dtype": "float32", "value": [0.5, 0.5]}, {"shape": [2], "dtype": "float32", "value": [0.8, 0.2]}], "expected": 0.2231435},
    ],
    solution=textwrap.dedent('''\
        import torch

        def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
            """
            KL divergence: KL(p||q) = sum(p * (log(p) - log(q)))
            p: target distribution, q: predicted distribution.
            Both should sum to 1 (probability distributions).
            """
            # 添加 epsilon 避免 log(0)，然后逐元素计算
            # Add epsilon to avoid log(0), compute element-wise
            p_safe = p + eps
            q_safe = q + eps
            kl = (p_safe * (torch.log(p_safe) - torch.log(q_safe))).sum()
            return kl
    '''),
    explanation=textwrap.dedent('''\
        **KL Divergence 详解**

        **核心思想:**
        KL 散度衡量用分布 q 近似分布 p 时损失的信息量。

        **Step 1 — 数值稳定处理:**
        - p 和 q 可能包含0，直接 log(0) 会得到 -inf
        - 给两者都加 epsilon（1e-10），保持数值稳定

        **Step 2 — 计算散度:**
        - 逐元素: p_i * (log(p_i) - log(q_i))
        - 当 p_i 很大而 q_i 很小时，惩罚很大（q 严重低估 p 的概率）
        - 求和得到总散度
        - 数学表达: KL(p||q) = Σ_i p_i log(p_i / q_i)

        **不对称性:**
        - KL(p||q) ≠ KL(q||p)
        - 前向 KL（mean-seeking）：q 必须覆盖所有 p 有质量的地方
        - 反向 KL（mode-seeking）：q 只覆盖 p 的一个模态

        **应用场景:**
        - VAE: 约束隐变量分布接近标准正态
        - 知识蒸馏: 让学生网络模仿教师输出的软分布
        - 扩散模型: 去噪目标与真实分布的匹配

        **面试考点:**
        - KL 与交叉熵的关系？→ CE = H(p) + KL(p||q)，最小化 CE 等价于最小化 KL
        - 为什么 KL 不对称？→ 从信息论角度，p→q 和 q→p 的信息损失不同
        - JS 散度解决了什么问题？→ 对称化 KL，避免无限大的问题
    '''),
))

# ========================================================================
# 8. Residual Connection
# ========================================================================
write_task("residual_connection", make_task(
    title="Residual Connection",
    title_zh="残差连接（Residual Connection）",
    difficulty="Easy",
    category="基础网络组件",
    desc_en="Implement a residual connection: output = x + f(x), where f(x) is a sub-layer (e.g., linear transformation). If the input and output shapes differ, project x with a linear layer before adding.",
    desc_zh="实现残差连接：output = x + f(x)，其中 f(x) 是子层（如线性变换）。如果输入和输出维度不同，先对 x 做线性投影再相加。",
    func_name="residual_connection",
    hint="Check if shapes match. If not, create a projection matrix W such that x @ W has the same shape as f(x).",
    hint_zh="检查形状是否匹配。如果不匹配，创建投影矩阵 W 使得 x @ W 与 f(x) 形状相同。",
    theory_en="Residual connections allow gradients to flow directly through the network via shortcut paths, enabling training of very deep networks (ResNet, Transformer).",
    theory_zh="残差连接通过捷径路径让梯度直接回流，使得极深网络（ResNet、Transformer）可以训练。",
    tests=[
        {"inputs": [{"shape": [2, 4], "dtype": "float32", "value": [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]}, {"shape": [2, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}], "expected": [[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8]]},
    ],
    solution=textwrap.dedent('''\
        import torch
        import torch.nn as nn

        def residual_connection(x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
            """
            Residual connection: output = x + f(x)
            If shapes differ, project x to match f(x) shape.
            """
            # 检查输入和子层输出的形状是否一致
            # Check if input and sublayer output shapes match
            if x.shape == f_x.shape:
                return x + f_x
            else:
                # 维度不匹配时，使用线性投影（这里用随机初始化演示）
                # When dims mismatch, use linear projection (random init for demo)
                in_features = x.shape[-1]
                out_features = f_x.shape[-1]
                proj = nn.Linear(in_features, out_features, bias=False)
                with torch.no_grad():
                    # Initialize as identity-like for stability demonstration
                    nn.init.eye_(proj.weight[:min(in_features, out_features), :min(in_features, out_features)])
                return proj(x) + f_x
    '''),
    explanation=textwrap.dedent('''\
        **Residual Connection 详解**

        **核心思想:**
        残差连接通过将输入直接加到子层输出上，创建梯度捷径，解决深层网络退化问题。

        **Step 1 — 形状检查:**
        - 如果 x 和 f(x) 形状相同，直接相加
        - 这是 Transformer 中最常见的情况（自注意力、FFN 都不改变序列维度）

        **Step 2 — 投影处理:**
        - 如果维度不同（如 ResNet 下采样层），用 1x1 卷积或线性层投影
        - 投影层通常不带偏置（bias=False）

        **数学表达:**
        y = F(x, {W_i}) + x

        **为什么有效:**
        - 梯度反向传播时，∂y/∂x = ∂F/∂x + 1
        - 即使 ∂F/∂x 很小，梯度至少还有 1，不会消失
        - 网络可以学习「退化映射」F(x)=0，即恒等函数

        **面试考点:**
        - ResNet 为什么能解决梯度消失？→ 捷径连接保留梯度
        - Pre-Norm vs Post-Norm？→ Pre-Norm（LayerNorm在前）训练更稳定，Post-Norm（原始Transformer）需要学习率预热
        - 什么时候需要投影？→ 输入输出维度不一致时（如通道数变化、下采样）
    '''),
))

# ========================================================================
# 9. LSTM Cell
# ========================================================================
write_task("lstm_cell", make_task(
    title="LSTM Cell",
    title_zh="LSTM 单元",
    difficulty="Medium",
    category="基础网络组件",
    desc_en="Implement a single LSTM cell step. Given input x_t, previous hidden state h_{t-1}, and previous cell state c_{t-1}, compute: forget gate f = σ(W_f·[h,x] + b_f), input gate i = σ(W_i·[h,x] + b_i), candidate g = tanh(W_g·[h,x] + b_g), output gate o = σ(W_o·[h,x] + b_o). Then c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t).",
    desc_zh="实现单个 LSTM 单元的前向步。给定输入 x_t、前一时刻隐状态 h_{t-1} 和细胞状态 c_{t-1}，计算：遗忘门 f = σ(W_f·[h,x] + b_f)，输入门 i = σ(W_i·[h,x] + b_i)，候选值 g = tanh(W_g·[h,x] + b_g)，输出门 o = σ(W_o·[h,x] + b_o)。然后 c_t = f*c_{t-1} + i*g，h_t = o*tanh(c_t)。",
    func_name="lstm_cell",
    hint="Concatenate h and x along the last dimension. Use four separate linear projections, then apply sigmoid/tanh to the respective gates.",
    hint_zh="沿最后一维拼接 h 和 x。使用四个独立的线性投影，然后对 respective 门应用 sigmoid/tanh。",
    theory_en="LSTM uses gating mechanisms to control information flow, solving the vanishing gradient problem in vanilla RNNs. The cell state acts as a conveyor belt for long-term memory.",
    theory_zh="LSTM 使用门控机制控制信息流，解决了普通 RNN 的梯度消失问题。细胞状态充当长期记忆传送带。",
    tests=[
        {"inputs": [{"shape": [1, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4]]}, {"shape": [1, 3], "dtype": "float32", "value": [[0.5,0.5,0.5]]}, {"shape": [1, 3], "dtype": "float32", "value": [[0.0,0.0,0.0]]}, 3, 3], "check": "shape"},
    ],
    solution=textwrap.dedent('''\
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
    '''),
    explanation=textwrap.dedent('''\
        **LSTM Cell 详解**

        **核心思想:**
        LSTM 通过三个门控（遗忘门、输入门、输出门）和一个细胞状态，精确控制信息的保留、写入和读取。

        **Step 1 — 拼接输入:**
        - 将前一时刻隐状态 h_{t-1} 和当前输入 x_t 拼接
        - 这是标准做法，让门控同时考虑历史和新输入

        **Step 2 — 计算四个门控:**
        - 遗忘门 f = σ(W_f [h,x] + b_f): 0=全忘，1=全留
        - 输入门 i = σ(W_i [h,x] + b_i): 0=不写，1=全写
        - 候选值 g = tanh(W_g [h,x] + b_g): 新候选记忆内容
        - 输出门 o = σ(W_o [h,x] + b_o): 0=不输出，1=全输出

        **Step 3 — 更新状态:**
        - 细胞状态: c_t = f ⊙ c_{t-1} + i ⊙ g
          - 遗忘门「擦掉」不重要的旧记忆
          - 输入门「写入」有价值的新信息
        - 隐状态: h_t = o ⊙ tanh(c_t)
          - tanh 将细胞状态压缩到 (-1, 1)
          - 输出门决定输出多少

        **为什么能解决梯度消失:**
        - 细胞状态的更新是线性运算（加法和逐元素乘）
        - 没有反复应用 tanh/sigmoid，梯度可以沿细胞状态长时间稳定传播
        - 遗忘门接近1时，信息可无损传递很多时间步

        **面试考点:**
        - LSTM vs GRU？→ LSTM 三扇门更精细，GRU 两扇门更轻量
        - 为什么现在不用 LSTM 了？→ Transformer 并行性更好，长距离依赖更强
        - 细胞状态和隐状态的区别？→ c_t 是长期记忆传送带，h_t 是当前时刻的「可见」输出
    '''),
))

# ========================================================================
# 10. GRU Cell
# ========================================================================
write_task("gru_cell", make_task(
    title="GRU Cell",
    title_zh="GRU 单元",
    difficulty="Medium",
    category="基础网络组件",
    desc_en="Implement a single GRU cell step. Given input x_t and previous hidden state h_{t-1}, compute: update gate z = σ(W_z·[h,x] + b_z), reset gate r = σ(W_r·[h,x] + b_r), candidate h~ = tanh(W_h·[r⊙h, x] + b_h). Then h_t = (1-z)⊙h_{t-1} + z⊙h~.",
    desc_zh="实现单个 GRU 单元的前向步。给定输入 x_t 和前一时刻隐状态 h_{t-1}，计算：更新门 z = σ(W_z·[h,x] + b_z)，重置门 r = σ(W_r·[h,x] + b_r)，候选值 h~ = tanh(W_h·[r⊙h, x] + b_h)。然后 h_t = (1-z)⊙h_{t-1} + z⊙h~。",
    func_name="gru_cell",
    hint="Concatenate h and x for gates. For candidate, multiply h_prev by reset gate r before concatenating with x.",
    hint_zh="门控计算时拼接 h 和 x。候选值计算时，先将 h_prev 乘以重置门 r，再与 x 拼接。",
    theory_en="GRU simplifies LSTM by merging cell state and hidden state into one vector, using only two gates (update and reset). It has fewer parameters and trains faster while maintaining similar performance.",
    theory_zh="GRU 将 LSTM 的细胞状态和隐状态合并为一个向量，仅使用两个门（更新门和重置门）。参数量更少，训练更快，性能相似。",
    tests=[
        {"inputs": [{"shape": [1, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4]]}, {"shape": [1, 3], "dtype": "float32", "value": [[0.5,0.5,0.5]]}, 3, 4, 3], "check": "shape"},
    ],
    solution=textwrap.dedent('''\
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
    '''),
    explanation=textwrap.dedent('''\
        **GRU Cell 详解**

        **核心思想:**
        GRU 将 LSTM 的细胞状态和隐状态合并，用两个门（更新门、重置门）实现类似功能，参数量减少约 25%。

        **Step 1 — 计算门控:**
        - 更新门 z = σ(W_z [h,x] + b_z): 1=全用新候选，0=全保留旧状态
        - 重置门 r = σ(W_r [h,x] + b_r): 0=完全遗忘旧状态，1=全保留

        **Step 2 — 候选隐状态:**
        - 先将旧状态 h_{t-1} 乘以重置门 r（选择性遗忘）
        - 再与 x_t 拼接，通过 tanh 生成候选值 h~
        - h~ = tanh(W_h [r⊙h_{t-1}, x_t] + b_h)

        **Step 3 — 状态更新:**
        - h_t = (1-z) ⊙ h_{t-1} + z ⊙ h~
        - (1-z) 部分保留历史信息，z 部分引入新信息
        - 这是一个「软」版本的状态切换，而非硬替换

        **GRU vs LSTM:**
        | 特性 | LSTM | GRU |
        |------|------|-----|
        | 门数量 | 3（遗忘/输入/输出） | 2（更新/重置） |
        | 状态数 | 2（c_t, h_t） | 1（h_t） |
        | 参数量 | 4×(h+i)×h | 3×(h+i)×h |
        | 训练速度 | 较慢 | 较快 |
        | 长距离依赖 | 略强 | 略弱 |

        **面试考点:**
        - 更新门和重置门的作用？→ 更新门控制「新旧混合比例」，重置门控制「旧状态遗忘程度」
        - 为什么 GRU 比 LSTM 快？→ 参数量少25%，计算量减少
        - 什么时候选 GRU？→ 数据量小、需要快速训练、长序列不是关键
    '''),
))

print("First 10 tasks generated. Continuing with remaining 10...")

# ========================================================================
# 11. Attention Mask (Padding)
# ========================================================================
write_task("attention_mask", make_task(
    title="Attention Mask (Padding)",
    title_zh="注意力掩码（Padding Mask）",
    difficulty="Medium",
    category="注意力机制",
    desc_en="Implement a padding mask for attention. Given a batch of sequence lengths `lengths` (list of ints) and max_seq_len, return a mask tensor of shape (batch_size, max_seq_len) where valid positions are 0 and padding positions are -inf (or a large negative number). This mask is added to attention scores before softmax.",
    desc_zh="实现注意力机制的 padding mask。给定一批序列长度 `lengths`（整数列表）和最大序列长度 max_seq_len，返回形状为 (batch_size, max_seq_len) 的掩码张量，有效位置为 0，padding 位置为 -inf（或大负数）。该掩码在 softmax 前加到注意力分数上。",
    func_name="attention_mask",
    hint="Create a tensor of zeros, then for each batch index i, set mask[i, lengths[i]:] = -inf.",
    hint_zh="创建全零张量，然后对每个 batch 索引 i，将 mask[i, lengths[i]:] 设为 -inf。",
    theory_en="Padding masks prevent attention from attending to padded positions in variable-length sequences. By adding -inf to attention scores, softmax produces zero probability for padded positions.",
    theory_zh="Padding mask 防止注意力关注变长序列中的填充位置。通过在注意力分数上加 -inf，softmax 会为填充位置输出零概率。",
    tests=[
        {"inputs": [[3, 2, 4], 5], "expected": [[0,0,0,-1e9,-1e9],[0,0,-1e9,-1e9,-1e9],[0,0,0,0,-1e9]]},
        {"inputs": [[1, 1, 1], 3], "expected": [[0,-1e9,-1e9],[0,-1e9,-1e9],[0,-1e9,-1e9]]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def attention_mask(lengths: list[int], max_seq_len: int) -> torch.Tensor:
            """
            Create padding mask for attention.
            Valid positions = 0, padding positions = -inf.
            Shape: (batch_size, max_seq_len)
            """
            batch_size = len(lengths)
            # 创建全零掩码
            # Create zero mask
            mask = torch.zeros(batch_size, max_seq_len)

            # 对每个序列，将超出实际长度的位置设为 -inf
            # For each sequence, set positions beyond actual length to -inf
            for i, length in enumerate(lengths):
                if length < max_seq_len:
                    mask[i, length:] = float('-inf')

            return mask
    '''),
    explanation=textwrap.dedent('''\
        **Attention Mask (Padding) 详解**

        **核心思想:**
        Padding mask 让模型在变长 batch 中忽略填充位置，只关注真实 token。

        **Step 1 — 创建全零掩码:**
        - 形状 (batch_size, max_seq_len)
        - 所有位置初始为 0（表示「不屏蔽」）

        **Step 2 — 标记填充位置:**
        - 对每个序列，length 之后的所有位置设为 -inf
        - -inf 在 softmax 中会变成 exp(-inf) = 0

        **使用方式:**
        - scores = Q @ K^T / sqrt(d_k) + mask  # mask 是 (B, 1, 1, S) 或 (B, S)
        - attn = softmax(scores, dim=-1)
        - padding 位置的 attention weight 精确为 0

        **数值细节:**
        - 用 -inf 而非大负数（如 -1e9）更精确，避免数值溢出
        - 但在某些框架中 -inf 会导致 NaN，所以实际常用 -1e9

        **面试考点:**
        - Padding mask vs Causal mask？→ Padding 屏蔽填充位，Causal 屏蔽未来位
        - 两个 mask 怎么联合？→ 相加：mask_total = mask_padding + mask_causal
        - 为什么加 -inf 而不是乘 0？→ 因为要在 softmax 之前操作，乘法会影响未屏蔽位置的 softmax 分布
    '''),
))

# ========================================================================
# 12. Softmax Temperature
# ========================================================================
write_task("softmax_temperature", make_task(
    title="Softmax with Temperature",
    title_zh="带温度的 Softmax",
    difficulty="Easy",
    category="采样与解码",
    desc_en="Implement softmax with temperature scaling: p_i = exp(z_i / T) / Σ_j exp(z_j / T), where T > 0 is the temperature. Higher T makes distribution more uniform (less confident), lower T makes it sharper (more confident).",
    desc_zh="实现带温度缩放的 Softmax：p_i = exp(z_i / T) / Σ_j exp(z_j / T)，其中 T > 0 是温度。T 越大分布越均匀（置信度低），T 越小分布越尖锐（置信度高）。",
    func_name="softmax_temperature",
    hint="Divide logits by T before computing softmax. Use the standard max-subtraction trick for numerical stability.",
    hint_zh="计算 softmax 前先将 logits 除以 T。使用标准的 max 减法技巧保证数值稳定。",
    theory_en="Temperature scaling controls the sharpness of a probability distribution. It is used in text generation (sampling diversity) and knowledge distillation (softening teacher labels).",
    theory_zh="温度缩放控制概率分布的尖锐程度。用于文本生成（控制采样多样性）和知识蒸馏（软化教师标签）。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, 1.0], "expected": [0.09003057, 0.24472848, 0.66524096]},
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, 2.0], "expected": [0.18632372, 0.3071959, 0.5064804]},
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, 0.5], "expected": [0.01587624, 0.11731043, 0.8668133]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
            """
            Softmax with temperature scaling.
            p_i = exp(z_i / T) / sum(exp(z_j / T))
            T > 1: smoother (more uniform), T < 1: sharper (more peaky).
            """
            # 先将 logits 除以温度，再做数值稳定的 softmax
            # Divide logits by temperature, then numerically stable softmax
            scaled = logits / temperature
            max_val = scaled.max(dim=-1, keepdim=True).values
            exp_scaled = torch.exp(scaled - max_val)
            probs = exp_scaled / exp_scaled.sum(dim=-1, keepdim=True)
            return probs
    '''),
    explanation=textwrap.dedent('''\
        **Softmax with Temperature 详解**

        **核心思想:**
        温度缩放通过调整 logits 的「锐度」来控制概率分布的形状。

        **Step 1 — 温度缩放:**
        - scaled = logits / T
        - T > 1: 缩小 logits 差距，分布更均匀
        - T < 1: 放大 logits 差距，分布更尖锐

        **Step 2 — 数值稳定 softmax:**
        - 减去最大值避免 exp 溢出
        - 标准 softmax 归一化

        **温度效果分析:**
        | T 值 | 效果 |
        |------|------|
        | T → 0 | 趋向 one-hot（确定性输出） |
        | T = 1 | 标准 softmax |
        | T → ∞ | 趋向均匀分布（完全随机） |

        **应用场景:**
        - **文本生成**: T=0.7~0.9 增加多样性，T=1.2 更 creative
        - **知识蒸馏**: 高温软化教师标签，传递更多信息（小概率类别的相对关系）
        - **校准**: 用温度调整模型置信度，使预测概率更接近真实准确率

        **面试考点:**
        - 温度缩放与 logits 缩放的关系？→ 等价于对数空间的线性缩放
        - 知识蒸馏为什么用高温？→ 软化后的分布保留了类别间的相似性信息
        - Temperature scaling 与 Label smoothing 的区别？→ 前者缩放模型输出，后者修改目标分布
    '''),
))

# ========================================================================
# 13. LayerScale
# ========================================================================
write_task("layerscale", make_task(
    title="LayerScale",
    title_zh="LayerScale",
    difficulty="Medium",
    category="训练技巧",
    desc_en="Implement LayerScale as used in DeiT III and modern Transformers. LayerScale multiplies the output of a sub-layer (e.g., attention or FFN) by a learnable diagonal matrix: output = x + diag(γ) * f(x), where γ is a learnable vector initialized to a small value (e.g., 1e-6). This stabilizes training of deep Transformers.",
    desc_zh="实现 DeiT III 和现代 Transformer 中使用的 LayerScale。LayerScale 将子层输出（如注意力或 FFN）乘以一个可学习的对角矩阵：output = x + diag(γ) * f(x)，其中 γ 是可学习向量，初始化为小值（如 1e-6）。这稳定了深层 Transformer 的训练。",
    func_name="layerscale",
    hint="Initialize γ as a small constant (e.g., 1e-6). Multiply f(x) element-wise by γ before adding to the residual.",
    hint_zh="将 γ 初始化为小常数（如 1e-6）。在加入残差前，将 f(x) 逐元素乘以 γ。",
    theory_en="LayerScale prevents early training instability by initially suppressing the residual branch. As training progresses, γ learns to scale the sub-layer output appropriately, allowing deeper networks to train without warm-up.",
    theory_zh="LayerScale 通过初始时抑制残差分支来防止早期训练不稳定。随着训练进行，γ 学习适当缩放子层输出，使深层网络无需 warm-up 即可训练。",
    tests=[
        {"inputs": [{"shape": [2, 4], "dtype": "float32", "value": [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]}, {"shape": [2, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}, [4], 1e-6], "check": "shape"},
    ],
    solution=textwrap.dedent('''\
        import torch
        import torch.nn as nn

        class LayerScale(nn.Module):
            """
            LayerScale: output = x + gamma * f(x)
            gamma is a learnable per-channel scaling factor, initialized small.
            """
            def __init__(self, dim: int, init_value: float = 1e-6):
                super().__init__()
                # 可学习的逐通道缩放因子，初始化为很小的值
                # Learnable per-channel scale, initialized to a tiny value
                self.gamma = nn.Parameter(init_value * torch.ones(dim))

            def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
                # gamma 逐元素缩放子层输出，再加回残差
                # Scale sublayer output element-wise, add back residual
                return x + self.gamma * f_x
    '''),
    explanation=textwrap.dedent('''\
        **LayerScale 详解**

        **核心思想:**
        LayerScale 在残差连接前引入一个可学习的缩放因子，初始值极小，防止深层 Transformer 早期训练崩溃。

        **Step 1 — 初始化:**
        - γ 初始化为 1e-6（或更小的值）
        - 这意味着训练初期，残差分支几乎被「关闭」
        - 网络先学习恒等映射，再逐渐引入子层变换

        **Step 2 — 前向传播:**
        - output = x + γ ⊙ f(x)
        - γ 是逐通道（per-channel / per-dim）的
        - 每个特征维度可以独立学习最合适的缩放

        **为什么有效:**
        - 深层 Transformer 早期训练时，Attention/FFN 的输出可能剧烈波动
        - 乘以 1e-6 后，这些波动对总输出的影响微乎其微
        - 随着训练进行，γ 逐渐增大，子层贡献增加
        - 避免了学习率 warm-up 的依赖

        **与 DeepNorm 的对比:**
        | 方法 | 机制 | 初始化 |
        |------|------|--------|
        | LayerScale | 学习缩放 | γ ≈ 0 |
        | DeepNorm | 固定缩放 + 残差放大 | α 很大 |
        | Post-LN | 无特殊处理 | 标准 |

        **面试考点:**
        - LayerScale 解决什么问题？→ 深层 Transformer 训练不稳定/发散
        - 为什么初始化为小值而不是1？→ 小值初期抑制残差，让网络先学恒等映射
        - 和 Dropout 的区别？→ LayerScale 是可学习的固定缩放，Dropout 是随机置零
    '''),
))

# ========================================================================
# 14. Stochastic Depth (DropPath)
# ========================================================================
write_task("stochastic_depth", make_task(
    title="Stochastic Depth (DropPath)",
    title_zh="随机深度（DropPath）",
    difficulty="Medium",
    category="正则化",
    desc_en="Implement Stochastic Depth (DropPath) regularization. During training, randomly drop entire residual branches with probability `drop_prob`: output = x + b * f(x), where b ~ Bernoulli(1 - drop_prob). During evaluation, use the full network: output = x + (1 - drop_prob) * f(x) (expectation over Bernoulli).",
    desc_zh="实现随机深度（DropPath）正则化。训练时以概率 `drop_prob` 随机丢弃整个残差分支：output = x + b * f(x)，其中 b ~ Bernoulli(1 - drop_prob)。推理时使用完整网络：output = x + (1 - drop_prob) * f(x)（Bernoulli 的期望）。",
    func_name="stochastic_depth",
    hint="Use `torch.rand()` to sample a Bernoulli mask. In eval mode, scale f(x) by (1 - drop_prob).",
    hint_zh="使用 `torch.rand()` 采样 Bernoulli 掩码。推理模式下将 f(x) 缩放 (1 - drop_prob)。",
    theory_en="Stochastic Depth randomly drops layers during training, effectively training an ensemble of networks with varying depth. It reduces overfitting and improves generalization, especially in deep networks like Vision Transformers.",
    theory_zh="随机深度在训练时随机丢弃层，相当于训练了一个具有不同深度的网络集成。它减少过拟合、提升泛化，尤其在深层网络（如 ViT）中效果显著。",
    tests=[
        {"inputs": [{"shape": [2, 4], "dtype": "float32", "value": [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]}, {"shape": [2, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}, 0.0, True], "expected": [[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8]]},
    ],
    solution=textwrap.dedent('''\
        import torch
        import torch.nn as nn

        class StochasticDepth(nn.Module):
            """
            Stochastic Depth (DropPath): randomly drop residual branches during training.
            output = x + b * f(x) where b ~ Bernoulli(1 - drop_prob)
            """
            def __init__(self, drop_prob: float = 0.1):
                super().__init__()
                self.drop_prob = drop_prob
                self.keep_prob = 1.0 - drop_prob

            def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
                if self.training:
                    if self.drop_prob == 0.0:
                        return x + f_x
                    # 训练时：以概率 drop_prob 丢弃整个分支
                    # Training: drop the entire branch with probability drop_prob
                    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
                    mask = torch.rand(shape, device=x.device) < self.keep_prob
                    # 对存活的分支做期望归一化：E[b] = keep_prob，所以除以 keep_prob
                    # Normalize by keep_prob for surviving branches
                    return x + f_x * mask / self.keep_prob
                else:
                    # 推理时：取期望，输出 = x + (1-p) * f(x)
                    # Inference: expectation, output = x + (1-p) * f(x)
                    return x + self.keep_prob * f_x
    '''),
    explanation=textwrap.dedent('''\
        **Stochastic Depth (DropPath) 详解**

        **核心思想:**
        随机深度在训练时以一定概率「跳过」整个残差分支，相当于训练了一个子网络集合。

        **Step 1 — 训练模式:**
        - 对每个样本独立采样 Bernoulli 掩码
        - mask ~ Bernoulli(keep_prob)，shape 为 (B, 1, 1, ...)
        - 被丢弃的样本：output = x（纯恒等映射）
        - 保留的样本：output = x + f(x) / keep_prob（期望归一化）

        **为什么除以 keep_prob:**
        - E[mask] = keep_prob
        - 如果不除，保留分支的期望贡献是 keep_prob * f(x)
        - 除以 keep_prob 后期望贡献恢复为 f(x)
        - 这样推理时只需要乘以 keep_prob，无需重新缩放

        **Step 2 — 推理模式:**
        - output = x + (1 - drop_prob) * f(x)
        - 这是对所有可能子网络的期望输出
        - 不需要随机采样，确定性输出

        **与 Dropout 的区别:**
        | 特性 | Dropout | DropPath |
        |------|---------|----------|
        | 丢弃粒度 | 单个神经元 | 整个残差分支 |
        | 结构变化 | 网络结构不变 | 有效深度变化 |
        | 主要应用 | 全连接层 | 残差网络/Transformer |
        | 效果 | 防止共适应 | 训练浅层网络集成 |

        **面试考点:**
        - DropPath 和 Dropout 的区别？→ 粒度不同（层 vs 神经元）
        - 为什么推理时要乘 keep_prob？→ 匹配训练时的期望输出
        - 为什么对深层网络有效？→ 浅层子网络更容易训练，深层网络容易过拟合
    '''),
))

# ========================================================================
# 15. Noisy Top-K Gating
# ========================================================================
write_task("noisy_topk_gating", make_task(
    title="Noisy Top-K Gating (MoE)",
    title_zh="噪声 Top-K 门控（MoE）",
    difficulty="Hard",
    category="混合专家模型",
    desc_en="Implement Noisy Top-K Gating for Mixture-of-Experts. Given input x and n_experts, compute: logits = x @ W_g + noise * (x @ W_noise), where noise = StandardNormal * softplus(x @ W_noise). Select top-k experts, apply softmax to their logits. Return (weights, expert_indices) where weights sum to 1 for each token.",
    desc_zh="实现混合专家模型（MoE）的噪声 Top-K 门控。给定输入 x 和专家数量 n_experts，计算：logits = x @ W_g + noise * (x @ W_noise)，其中 noise = StandardNormal * softplus(x @ W_noise)。选择 top-k 专家，对它们的 logits 应用 softmax。返回 (weights, expert_indices)，其中每个 token 的权重和为1。",
    func_name="noisy_topk_gating",
    hint="Generate noise using torch.randn. Use softplus for noise scaling. Top-k with softmax on selected logits only.",
    hint_zh="使用 torch.randn 生成噪声。用 softplus 做噪声缩放。仅对选中的 top-k logits 做 softmax。",
    theory_en="Noisy Top-K gating is the routing mechanism in Switch Transformer and other MoE models. Noise helps explore different expert assignments during training, improving load balancing.",
    theory_zh="噪声 Top-K 门控是 Switch Transformer 等 MoE 模型的路由机制。噪声帮助训练时探索不同的专家分配，改善负载均衡。",
    tests=[
        {"inputs": [{"shape": [2, 4], "dtype": "float32", "value": [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]}, 4, 8, 2, True], "check": "shape"},
    ],
    solution=textwrap.dedent('''\
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class NoisyTopKGating(nn.Module):
            """
            Noisy Top-K Gating for Mixture-of-Experts.
            Routes each token to k experts with load-balancing noise.
            """
            def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                # 门控投影：将输入映射到专家数量维度
                # Gate projection: map input to expert count dimension
                self.W_g = nn.Linear(input_dim, num_experts)
                # 噪声投影：学习每个输入的噪声强度
                # Noise projection: learn noise magnitude per input
                self.W_noise = nn.Linear(input_dim, num_experts)

            def forward(self, x: torch.Tensor, training: bool = True):
                # 干净的路由 logits
                # Clean routing logits
                clean_logits = self.W_g(x)  # (B, num_experts)

                if training:
                    # 噪声强度：softplus 保证非负
                    # Noise magnitude: softplus ensures non-negative
                    noise_std = F.softplus(self.W_noise(x))
                    # 采样高斯噪声
                    # Sample Gaussian noise
                    noise = torch.randn_like(clean_logits) * noise_std
                    noisy_logits = clean_logits + noise
                else:
                    noisy_logits = clean_logits

                # Top-K 选择：找出每个 token 最匹配的 k 个专家
                # Top-K selection: find k best experts per token
                top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

                # 对选中的专家做 softmax，得到路由权重
                # Softmax over selected experts to get routing weights
                top_k_weights = F.softmax(top_k_logits, dim=-1)

                return top_k_weights, top_k_indices
    '''),
    explanation=textwrap.dedent('''\
        **Noisy Top-K Gating 详解**

        **核心思想:**
        MoE 路由机制决定每个 token 分配给哪些专家处理。噪声帮助打破「所有 token 只选最好专家」的死锁。

        **Step 1 — 计算路由分数:**
        - clean_logits = x @ W_g: 基础路由分数
        - noise_std = softplus(x @ W_noise): 可学习的噪声强度
        - noisy_logits = clean_logits + ε * noise_std: 加噪声后的分数

        **Step 2 — Top-K 选择:**
        - 对每个 token，选择分数最高的 k 个专家
        - k=1（Switch Transformer）：每个 token 只给一个专家
        - k=2（Mixtral）：每个 token 给两个专家，输出加权平均

        **Step 3 — 权重归一化:**
        - 仅对选中的 k 个专家做 softmax
        - weights 和为 1，表示各专家的贡献比例

        **负载均衡问题:**
        - 没有噪声时，所有 token 可能涌向同一个「最好」的专家
        - 噪声强制探索次优专家，分散负载
        - 通常配合 auxiliary load-balancing loss 使用

        **面试考点:**
        - 为什么需要噪声？→ 打破路由死锁，强制负载均衡
        - Top-K 中 K=1 和 K=2 的区别？→ K=1 通信最少，K=2 质量更好
        - MoE 的通信瓶颈在哪？→ All-to-All 通信：将 token 发送到不同 GPU 的专家
        - 为什么用 softplus 而不是 ReLU？→ softplus 可导且不会精确为0，保证总有微小噪声
    '''),
))

# ========================================================================
# 16. Cosine Similarity
# ========================================================================
write_task("cosine_similarity", make_task(
    title="Cosine Similarity",
    title_zh="余弦相似度",
    difficulty="Easy",
    category="基础网络组件",
    desc_en="Implement cosine similarity between two vectors or matrices. cos_sim(a, b) = (a·b) / (||a|| * ||b||). For matrices, compute pairwise cosine similarity between rows. Add epsilon (1e-8) to denominators for numerical stability.",
    desc_zh="实现两个向量或矩阵之间的余弦相似度。cos_sim(a, b) = (a·b) / (||a|| * ||b||)。对矩阵计算行之间的成对余弦相似度。分母加 epsilon（1e-8）保证数值稳定。",
    func_name="cosine_similarity",
    hint="Compute L2 norm with `torch.norm` or `(x**2).sum(-1, keepdim=True).sqrt()`. Divide dot product by product of norms.",
    hint_zh="使用 `torch.norm` 或 `(x**2).sum(-1, keepdim=True).sqrt()` 计算 L2 范数。将点积除以范数乘积。",
    theory_en="Cosine similarity measures the angle between two vectors, ignoring magnitude. It is the core of contrastive learning (CLIP, SimCLR) and semantic search.",
    theory_zh="余弦相似度衡量两个向量之间的夹角，忽略模长。是对比学习（CLIP、SimCLR）和语义搜索的核心。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 0.0, 0.0]}, {"shape": [3], "dtype": "float32", "value": [0.0, 1.0, 0.0]}], "expected": 0.0},
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, {"shape": [3], "dtype": "float32", "value": [2.0, 4.0, 6.0]}], "expected": 1.0},
        {"inputs": [{"shape": [2, 3], "dtype": "float32", "value": [[1.0,0.0,0.0],[0.0,1.0,0.0]]}, {"shape": [2, 3], "dtype": "float32", "value": [[0.0,1.0,0.0],[1.0,0.0,0.0]]}], "expected": [[0.0,1.0],[1.0,0.0]]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            """
            Cosine similarity: cos(a,b) = (a·b) / (||a|| * ||b||)
            For matrices, compute pairwise similarity between rows.
            """
            # 计算 L2 范数（沿最后一维），keepdim 用于广播
            # Compute L2 norm along last dim, keepdim for broadcasting
            norm_a = torch.norm(a, p=2, dim=-1, keepdim=True)
            norm_b = torch.norm(b, p=2, dim=-1, keepdim=True)

            # 归一化向量（加 eps 避免除零）
            # Normalize vectors (add eps to avoid division by zero)
            a_norm = a / (norm_a + eps)
            b_norm = b / (norm_b + eps)

            # 计算余弦相似度：归一化向量的点积
            # Cosine similarity = dot product of normalized vectors
            if a.ndim == 1 and b.ndim == 1:
                return torch.dot(a_norm, b_norm)
            else:
                return torch.matmul(a_norm, b_norm.T)
    '''),
    explanation=textwrap.dedent('''\
        **Cosine Similarity 详解**

        **核心思想:**
        余弦相似度衡量两个向量的方向一致性，忽略模长差异。值域 [-1, 1]，1 表示完全相同方向。

        **Step 1 — 计算 L2 范数:**
        - ||a||_2 = sqrt(Σ a_i²)
        - `torch.norm(a, p=2, dim=-1)` 沿特征维度计算
        - `keepdim=True` 保持维度用于广播除法

        **Step 2 — 归一化:**
        - a_norm = a / (||a|| + eps)
        - eps 防止零向量导致除零错误

        **Step 3 — 计算相似度:**
        - 向量：直接点积 `torch.dot`
        - 矩阵：矩阵乘法 `a_norm @ b_norm.T`
        - 结果即 cos(θ)，θ 是两向量夹角

        **应用场景:**
        - **CLIP**: 图像和文本嵌入的余弦相似度作为匹配分数
        - **SimCLR**: 正样本对拉近，负样本对推远
        - **语义搜索**: 查询向量与文档向量的相似度排序
        - **RAG**: 检索相关文档的核心指标

        **面试考点:**
        - 余弦相似度 vs 欧氏距离？→ 余弦忽略模长，适合语义相似；欧氏考虑绝对差异
        - 为什么 CLIP 用余弦相似度而不是点积？→ 归一化后模长不影响，只关注语义方向
        - 温度缩放和余弦相似度的关系？→ T 控制 softmax 的锐度，不影响余弦值本身
    '''),
))

# ========================================================================
# 17. One-Hot Encoding
# ========================================================================
write_task("one_hot_encoding", make_task(
    title="One-Hot Encoding",
    title_zh="One-Hot 编码",
    difficulty="Easy",
    category="基础网络组件",
    desc_en="Implement one-hot encoding. Given a tensor of class indices (0 to num_classes-1), return a tensor where each row is a one-hot vector. For example, indices [0, 2, 1] with num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]].",
    desc_zh="实现 One-Hot 编码。给定类别索引张量（0 到 num_classes-1），返回 one-hot 向量张量。例如 indices [0, 2, 1] 且 num_classes=4 → [[1,0,0,0], [0,0,1,0], [0,1,0,0]]。",
    func_name="one_hot_encoding",
    hint="Use `torch.zeros` to create the output tensor, then use advanced indexing or `scatter_` to set 1s.",
    hint_zh="使用 `torch.zeros` 创建输出张量，然后用高级索引或 `scatter_` 设置1。",
    theory_en="One-hot encoding converts categorical class indices into binary vectors. It is the standard input format for classification tasks and the target format for cross-entropy loss.",
    theory_zh="One-Hot 编码将类别索引转换为二元向量。是分类任务的标准输入格式，也是交叉熵损失的目标格式。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "int64", "value": [0, 2, 1]}, 4], "expected": [[1,0,0,0],[0,0,1,0],[0,1,0,0]]},
        {"inputs": [{"shape": [1], "dtype": "int64", "value": [3]}, 5], "expected": [[0,0,0,1,0]]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def one_hot_encoding(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
            """
            One-hot encode class indices.
            indices: (N,) with values in [0, num_classes-1]
            Returns: (N, num_classes) one-hot tensor (float32).
            """
            # 创建全零输出张量
            # Create zero output tensor
            one_hot = torch.zeros(indices.shape[0], num_classes, dtype=torch.float32)

            # 使用 scatter 在对应位置填1
            # Use scatter to fill 1s at corresponding positions
            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

            return one_hot
    '''),
    explanation=textwrap.dedent('''\
        **One-Hot Encoding 详解**

        **核心思想:**
        将离散的类别索引转换为连续的二元向量，使类别可以作为神经网络的输入或目标。

        **Step 1 — 创建零矩阵:**
        - 形状 (N, num_classes)，N 是样本数
        - 数据类型 float32（适合作为网络输入）

        **Step 2 — scatter 填充:**
        - `scatter_(dim, index, value)` 在指定维度上按索引填充值
        - `indices.unsqueeze(1)` 将索引形状从 (N,) 变为 (N, 1)
        - 在 dim=1（类别维度）的 index 位置填入 1.0

        **替代方法:**
        - PyTorch 内置: `F.one_hot(indices, num_classes).float()`
        - 但更推荐自己实现以理解原理

        **面试考点:**
        - One-Hot 的缺点？→ 维度高、稀疏、无法表达类别间相似性
        - 替代方案？→ Embedding lookup（低维稠密表示）
        - 为什么交叉熵不用 one-hot 作为输入？→ 直接用索引更高效（避免 O(C) 开销）
    '''),
))

# ========================================================================
# 18. Sequence Packing
# ========================================================================
write_task("sequence_pack", make_task(
    title="Sequence Packing",
    title_zh="序列打包",
    difficulty="Easy",
    category="基础网络组件",
    desc_en="Implement sequence packing for efficient training. Given a list of variable-length sequences (each is a 1D tensor), pack them into a single padded tensor of shape (batch_size, max_len). Also return a binary mask indicating valid positions (1 for real data, 0 for padding).",
    desc_zh="实现序列打包以提高训练效率。给定一批变长序列（每个是一维张量），将它们打包成单个填充张量，形状为 (batch_size, max_len)。同时返回一个二元掩码，标记有效位置（1 表示真实数据，0 表示填充）。",
    func_name="sequence_pack",
    hint="Find max length, create a zero tensor of shape (batch_size, max_len), then copy each sequence into its corresponding row.",
    hint_zh="找到最大长度，创建形状为 (batch_size, max_len) 的零张量，然后将每个序列复制到对应行。",
    theory_en="Sequence packing is essential for batched training of variable-length sequences (NLP, time series). Padding allows tensor operations while masks prevent attention to padded positions.",
    theory_zh="序列打包是变长序列（NLP、时间序列）批量训练的基础。填充使张量运算成为可能，掩码防止注意力关注填充位置。",
    tests=[
        {"inputs": [[[1,2,3],[4,5],[6,7,8,9]]], "check": "shape"},
    ],
    solution=textwrap.dedent('''\
        import torch
        from torch.nn.utils.rnn import pad_sequence

        def sequence_pack(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Pack variable-length sequences into a padded tensor.
            Returns: (padded_tensor, mask)
            padded_tensor: (batch_size, max_len)
            mask: (batch_size, max_len), 1 for valid, 0 for padding.
            """
            # 找到最大序列长度
            # Find maximum sequence length
            max_len = max(seq.shape[0] for seq in sequences)
            batch_size = len(sequences)

            # 创建填充张量和掩码
            # Create padded tensor and mask
            padded = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype)
            mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

            # 将每个序列复制到对应行，并标记有效位置
            # Copy each sequence to its row and mark valid positions
            for i, seq in enumerate(sequences):
                length = seq.shape[0]
                padded[i, :length] = seq
                mask[i, :length] = 1.0

            return padded, mask
    '''),
    explanation=textwrap.dedent('''\
        **Sequence Packing 详解**

        **核心思想:**
        将不同长度的序列对齐到同一长度（用填充值补齐），使它们可以组成 batch 进行并行计算。

        **Step 1 — 确定尺寸:**
        - `max_len = max(len(seq) for seq in sequences)`
        - `batch_size = len(sequences)`

        **Step 2 — 创建填充张量:**
        - 全零张量，形状 (batch_size, max_len)
        - 同时创建同样形状的掩码（1=有效，0=填充）

        **Step 3 — 填充数据:**
        - 将每个序列复制到 padded[i, :length]
        - mask[i, :length] = 1 标记有效位置

        **和 torch.nn.utils.rnn.pad_sequence 的关系:**
        - PyTorch 内置函数更通用（支持不同 batch_first 模式）
        - 自己实现有助于理解 padding 和 masking 的本质

        **面试考点:**
        - 为什么需要序列打包？→ GPU 张量运算要求固定形状
        - Padding 的替代方案？→ PackedSequence（存储原始长度，避免显式填充）
        - 如何处理不同维度的序列？→ 如图像序列，在时空维度分别填充
    '''),
))

# ========================================================================
# 19. Leaky ReLU
# ========================================================================
write_task("leaky_relu", make_task(
    title="Leaky ReLU",
    title_zh="Leaky ReLU",
    difficulty="Easy",
    category="激活函数",
    desc_en="Implement Leaky ReLU: f(x) = x if x > 0, else f(x) = alpha * x, where alpha is a small negative slope (default 0.01). This addresses the 'dying ReLU' problem where neurons can become permanently inactive.",
    desc_zh="实现 Leaky ReLU：f(x) = x（若 x > 0），否则 f(x) = alpha * x，其中 alpha 是一个小的负斜率（默认 0.01）。这解决了「死亡 ReLU」问题，即神经元可能永久失活。",
    func_name="leaky_relu",
    hint="Use `torch.where(x > 0, x, alpha * x)`.",
    hint_zh="使用 `torch.where(x > 0, x, alpha * x)`。",
    theory_en="Leaky ReLU allows a small gradient for negative inputs, preventing neurons from dying. It is a simple and effective improvement over standard ReLU.",
    theory_zh="Leaky ReLU 允许负输入有小的梯度，防止神经元死亡。是对标准 ReLU 简单而有效的改进。",
    tests=[
        {"inputs": [{"shape": [4], "dtype": "float32", "value": [1.0, -1.0, 0.0, -2.0]}, 0.01], "expected": [1.0, -0.01, 0.0, -0.02]},
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [-5.0, 5.0, 0.0]}, 0.1], "expected": [-0.5, 5.0, 0.0]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
            """
            Leaky ReLU: f(x) = x if x > 0, else alpha * x.
            Prevents dying ReLU by allowing small negative gradients.
            """
            # 正数保持原样，负数乘以 alpha（小斜率）
            # Positive: keep as is; Negative: multiply by small slope alpha
            return torch.where(x > 0, x, alpha * x)
    '''),
    explanation=textwrap.dedent('''\
        **Leaky ReLU 详解**

        **核心思想:**
        Leaky ReLU 给负数输入一个小的非零梯度，解决标准 ReLU 的「死亡神经元」问题。

        **数学表达:**
        f(x) = { x,       if x > 0
               { αx,      if x ≤ 0
        其中 α 通常取 0.01（小斜率）

        **死亡 ReLU 问题:**
        - 标准 ReLU: f(x) = max(0, x)，负数区域梯度为0
        - 如果某神经元权重初始化不好，输出总是负数
        - 梯度永远为0，权重不再更新 → 「死亡」
        - Leaky ReLU: 负数区域梯度 = α ≠ 0，神经元有机会「复活」

        **变体对比:**
        | 变体 | 负数区域 | 特点 |
        |------|---------|------|
        | ReLU | 0 | 简单，但可能死亡 |
        | Leaky ReLU | αx | 固定小斜率，防止死亡 |
        | PReLU | αx | α 可学习，更灵活 |
        | ELU | α(e^x-1) | 平滑负数区域，均值更接近0 |
        | GELU | xΦ(x) | Sigmoid 加权，平滑可导 |

        **面试考点:**
        - 死亡 ReLU 的根本原因？→ 负数区域梯度为零，导致权重停止更新
        - α 为什么通常取 0.01？→ 经验值，足够小不影响正数区域，足够大维持梯度流
        - PReLU 和 Leaky ReLU 的区别？→ PReLU 的 α 是可学习参数
    '''),
))

# ========================================================================
# 20. SGD with Momentum
# ========================================================================
write_task("sgd_momentum", make_task(
    title="SGD with Momentum",
    title_zh="带动量的 SGD",
    difficulty="Easy",
    category="优化器与学习率",
    desc_en="Implement SGD with momentum update step. Given parameter p, gradient g, and velocity v: v_new = momentum * v + g, p_new = p - lr * v_new. This accumulates velocity in directions of consistent gradient, accelerating convergence in relevant directions and dampening oscillations.",
    desc_zh="实现带动量的 SGD 更新步。给定参数 p、梯度 g 和速度 v：v_new = momentum * v + g，p_new = p - lr * v_new。这会在梯度一致的方向上积累速度，加速收敛并抑制震荡。",
    func_name="sgd_momentum",
    hint="Update velocity first, then update parameter using the new velocity. Momentum is typically 0.9.",
    hint_zh="先更新速度，再用新速度更新参数。动量通常取 0.9。",
    theory_en="Momentum simulates the inertia of a heavy ball rolling down the loss landscape. It accelerates in directions of consistent gradient and reduces oscillations in directions with high curvature.",
    theory_zh="动量模拟重球在损失 landscape 上滚动的惯性。在梯度一致的方向上加速，在高曲率方向上减少震荡。",
    tests=[
        {"inputs": [{"shape": [3], "dtype": "float32", "value": [1.0, 2.0, 3.0]}, {"shape": [3], "dtype": "float32", "value": [0.1, 0.2, 0.3]}, {"shape": [3], "dtype": "float32", "value": [0.0, 0.0, 0.0]}, 0.1, 0.9], "expected": [[0.91, 1.82, 2.73], [0.1, 0.2, 0.3]]},
    ],
    solution=textwrap.dedent('''\
        import torch

        def sgd_momentum(param: torch.Tensor, grad: torch.Tensor, velocity: torch.Tensor,
                         lr: float, momentum: float) -> tuple[torch.Tensor, torch.Tensor]:
            """
            SGD with momentum update step.
            v_new = momentum * v + grad
            p_new = p - lr * v_new
            Returns: (param_new, velocity_new)
            """
            # 更新速度：保留上一时刻速度的一部分 + 当前梯度
            # Update velocity: keep part of previous velocity + current gradient
            velocity_new = momentum * velocity + grad

            # 用新速度更新参数
            # Update parameter with new velocity
            param_new = param - lr * velocity_new

            return param_new, velocity_new
    '''),
    explanation=textwrap.dedent('''\
        **SGD with Momentum 详解**

        **核心思想:**
        动量模拟物理惯性，在梯度一致的方向上积累速度，减少高曲率方向的震荡。

        **Step 1 — 更新速度:**
        - v_new = momentum * v + grad
        - momentum=0.9: 保留90%的历史速度方向
        - 当前梯度像一个「力」，改变速度方向

        **Step 2 — 更新参数:**
        - p_new = p - lr * v_new
        - 参数更新由速度驱动，而非直接由梯度驱动

        **物理直觉:**
        - 想象一个球滚下山坡：
        - 动量 = 球的速度（惯性）
        - 梯度 = 重力加速度
        - 球在平坦方向滚得越来越远（加速收敛）
        - 球在陡峭峡谷来回反弹（动量抑制震荡）

        **数学效果:**
        - 连续 T 步同向梯度：有效步长 ≈ lr * (1 + momentum + ... + momentum^T) ≈ lr/(1-momentum)
        - T=10, momentum=0.9: 有效步长放大到约 10*lr

        **面试考点:**
        - Momentum 和 Nesterov 的区别？→ Nesterov 先按速度方向「前瞻」再计算梯度
        - Momentum 和 Adam 的区别？→ Momentum 只用一阶矩，Adam 用一阶+二阶矩自适应
        - 为什么 momentum=0.9 是默认值？→ 经验值，平衡惯性和响应速度
        - 和指数移动平均的关系？→ 速度更新就是 EMA 的梯度版本
    '''),
))

print("All 20 tasks generated successfully!")
