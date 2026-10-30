"""SigLIP Pairwise Sigmoid Contrastive Loss task."""

TASK = {
    'title': 'SigLIP Pairwise Sigmoid Contrastive Loss',
    'title_zh': 'SigLIP 成对 Sigmoid 对比损失',
    'difficulty': 'Medium',
    'category': '多模态',
    'function_name': 'siglip_loss',
    'starter_code': """def siglip_loss(image_embeds, text_embeds, temp_log, bias, eps=1e-08):
    pass""",
    'hint': """Normalize both embedding sets, form every image-text logit, label diagonal pairs +1 and off-diagonal pairs -1, then average the summed log-sigmoid loss by N.""",
    'hint_zh': '归一化两组嵌入，构造所有图文 logits，将对角配对标为 +1、其余标为 -1，再将 log-sigmoid 损失总和除以 N。',
    'description_en': 'Implement the SigLIP pairwise sigmoid contrastive objective for matching image and text embeddings.',
    'description_zh': '实现用于匹配图像与文本嵌入的 SigLIP 成对 sigmoid 对比目标。',
    'theory_en': """Unlike softmax contrastive learning, SigLIP treats every entry of the N by N similarity matrix as an independent binary pair. The learned log-temperature controls scale and the bias controls the positive prior.""",
    'theory_zh': '不同于 softmax 对比学习，SigLIP 将 N×N 相似度矩阵的每个元素视为独立二元配对；可学习的对数温度控制尺度，偏置控制正样本先验。',
    'diagram_en': """```text
image,text -> L2 normalize -> pairwise logits -> signed log-sigmoid -> sum / N
```""",
    'diagram_zh': """```text
图像、文本 -> L2 归一化 -> 成对 logits -> 带符号 log-sigmoid -> 求和 / N
```""",
    "tests": [
        {
            "name": 'scalar shape',
            "code": """import torch
x = torch.randn(4, 6)
y = torch.randn(4, 6)
out = {fn}(x, y, torch.tensor(0.2), torch.tensor(-1.0))
assert out.shape == torch.Size([]) and torch.isfinite(out)""",
        },
        {
            "name": 'exact reference',
            "code": """import torch
import torch.nn.functional as F
x = torch.tensor([[1., 0.], [1., 1.]])
y = torch.tensor([[0., 1.], [1., 0.]])
t, b = torch.tensor(0.3), torch.tensor(-0.2)
out = {fn}(x, y, t, b)
xn = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
yn = y / (y.norm(dim=-1, keepdim=True) + 1e-8)
s = t.exp() * (xn @ yn.T) + b
labels = 2 * torch.eye(2) - 1
ref = -F.logsigmoid(labels * s).sum() / 2
assert torch.allclose(out, ref, atol=1e-7)""",
        },
        {
            "name": 'single pair edge case',
            "code": """import torch
import torch.nn.functional as F
x = torch.zeros(1, 3)
y = torch.zeros(1, 3)
b = torch.tensor(0.7)
out = {fn}(x, y, torch.tensor(0.), b)
assert torch.allclose(out, -F.logsigmoid(b)) and out.ndim == 0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(3, 5, requires_grad=True)
y = torch.randn(3, 5, requires_grad=True)
t = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(-0.5, requires_grad=True)
{fn}(x, y, t, b).backward()
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (x, y, t, b))""",
        },
    ],
    'solution': '''def siglip_loss(image_embeds, text_embeds, temp_log, bias, eps=1e-08):
    """实现SigLIP 成对 Sigmoid 对比损失（SigLIP Pairwise Sigmoid Contrastive Loss）。"""
    # Step 1: 将图像和文本嵌入归一化到单位球面
    img = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + eps)
    # Step 2: 生成经可学习温度与偏置校准的全配对余弦 logits
    scores = torch.exp(temp_log) * (img @ txt.T) + bias
    batch_size = scores.shape[0]
    labels = 2.0 * torch.eye(batch_size, device=scores.device, dtype=scores.dtype) - 1.0
    # Step 3: 对正负图文对分别施加带符号 log-sigmoid 并按样本数归一化
    return -F.logsigmoid(labels * scores).sum() / batch_size''',
    'demo': """torch.manual_seed(42)
image_embeds = torch.randn(8, 64)
text_embeds = torch.randn(8, 64)
loss = siglip_loss(image_embeds, text_embeds, torch.tensor(0.0), torch.tensor(-1.0))
print("Pairwise SigLIP loss:", loss.item())
print("Embedding batch/width:", image_embeds.shape)
print("Loss is finite:", torch.isfinite(loss).item())""",
    'explanation': r"""### 核心思路
SigLIP 不在一行相似度上做互斥 softmax，而把每个图文组合 $(i,j)$ 当作独立二分类样本。单位化后令 $s_{ij}=\exp(t)\,\hat x_i^\top\hat y_j+b$，标签 $z_{ij}=+1$ 当 $i=j$、否则为 $-1$，目标为 $\mathcal L=-\frac1N\sum_{i,j}\log\sigma(z_{ij}s_{ij})$。

### 代码步骤拆解
- **Step 1（单位化双模态嵌入）**：`img`、`txt` 都保持 $(N,D)$，沿 $D$ 维除以 L2 范数，使矩阵乘积表示余弦相似度。
- **Step 2（构造全配对 logits）**：`img @ txt.T` 产生 $(N,N)$ 相似度矩阵，`exp(temp_log)` 保证尺度始终为正，标量 `bias` 调整匹配先验。
- **Step 3（带符号二元逻辑损失）**：`labels` 对角为 $+1$、非对角为 $-1$；`F.logsigmoid(labels * scores)` 稳定计算所有正负 pair，求和后除以 `batch_size`。

### 重点与易错点分析
- **注意点 1（归一化约定）**：本题按 $N$ 而非 $N^2$ 归一化，所以每个图像贡献一个正 pair 和 $N-1$ 个负 pair；擅自取全矩阵均值会把梯度缩小 $N$ 倍。
- **注意点 2（零向量与温度参数）**：`eps` 让零嵌入保持有限；温度传入的是对数尺度 `temp_log`，必须先 `exp`，不能直接用作乘数或除数。
- **注意点 3（与 CLIP 对比）**：CLIP 的行列 softmax 会让候选在批内竞争，SigLIP 的每个 pair 独立，因此无需跨设备聚合 softmax 分母，但显式实现仍需存储 $N^2$ logits 与标签。

### 复杂度分析
- **时间复杂度**：$O(N^2D)$，图像与文本嵌入的全配对矩阵乘法占主导。
- **空间复杂度**：$O(N^2+ND)$，`scores`、`labels` 及其损失矩阵为平方级，归一化嵌入为线性级。""",
}
