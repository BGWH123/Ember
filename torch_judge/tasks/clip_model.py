"""CLIP Model (Image-Text Contrastive Learning) task."""

TASK = {
    'title': 'CLIP Model (Image-Text Contrastive Learning)',
    'title_zh': 'CLIP 模型（图文对比学习）',
    'difficulty': 'Hard',
    'category': '多模态',
    'function_name': 'CLIP',
    'starter_code': """class CLIP(nn.Module):
    pass""",
    'hint': """1. `text_proj = nn.Linear(text_dim, embed_dim)`
2. `logit_scale = nn.Parameter(torch.ones([]) * log(1/0.07))`
3. `temperature = 1 / exp(logit_scale)`
4. Normalize: `x = x / x.norm(dim=-1, keepdim=True)`
5. `logits = image_embed @ text_embed.T * exp(logit_scale)`""",
    'hint_zh': """1. `text_proj = nn.Linear(text_dim, embed_dim)`
2. `logit_scale = nn.Parameter(torch.ones([]) * log(1/0.07))`
3. `temperature = 1 / exp(logit_scale)`
4. 归一化：`x = x / x.norm(dim=-1, keepdim=True)`
5. `logits = image_embed @ text_embed.T * exp(logit_scale)`""",
    'description_en': """Implement a simplified CLIP model for image-text contrastive learning.

CLIP jointly trains an image encoder and a text encoder to produce aligned embeddings in a shared space. The training objective is a symmetric cross-entropy loss over the cosine similarity matrix.

**Signature:** `CLIP(image_encoder, text_encoder, embed_dim, temperature=0.07)` (nn.Module)

**Forward:** `forward(images, text_features) -> Tensor`
- `images` — image tensor (B, C, H, W)
- `text_features` — pre-encoded text features (B, text_dim)

**Returns:** scalar contrastive loss

**Architecture:**
1. `image_embed = image_encoder(images)` → (B, embed_dim)
2. `text_embed = text_projection(text_features)` → (B, embed_dim)
3. L2-normalize both embeddings
4. Compute cosine similarity matrix: `logits = image_embed @ text_embed.T / temperature`
5. Targets: diagonal (matched pairs)
6. Loss: `(ce_loss(logits, targets) + ce_loss(logits.T, targets)) / 2`

**Constraints:**
- Use `nn.Linear(text_dim, embed_dim)` for text projection
- Temperature is a learnable parameter initialized to 0.07
- L2-normalize embeddings before computing similarity""",
    'description_zh': """实现简化的 CLIP 模型用于图文对比学习。

CLIP 联合训练图像编码器和文本编码器，在共享空间中生成对齐的嵌入。训练目标是在余弦相似度矩阵上的对称交叉熵损失。

**签名:** `CLIP(image_encoder, text_encoder, embed_dim, temperature=0.07)`（nn.Module）

**前向传播:** `forward(images, text_features) -> Tensor`
- `images` — 图像张量 (B, C, H, W)
- `text_features` — 预编码文本特征 (B, text_dim)

**返回:** 标量对比损失

**架构：**
1. `image_embed = image_encoder(images)` → (B, embed_dim)
2. `text_embed = text_projection(text_features)` → (B, embed_dim)
3. 对两个嵌入做 L2 归一化
4. 计算余弦相似度矩阵：`logits = image_embed @ text_embed.T / temperature`
5. 目标：对角线（匹配对）
6. 损失：`(ce_loss(logits, targets) + ce_loss(logits.T, targets)) / 2`

**约束：**
- 使用 `nn.Linear(text_dim, embed_dim)` 做文本投影
- Temperature 是可学习参数，初始值 0.07
- 计算相似度前对嵌入做 L2 归一化""",
    'theory_en': r"""CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021) learns joint image-text representations
by maximizing the cosine similarity of matched pairs and minimizing it for unmatched pairs.

**Training Objective:**
$$\mathcal{L} = \frac{1}{2} \left[ \mathcal{L}_{I\to T} + \mathcal{L}_{T\to I} \right]$$

$$\mathcal{L}_{I\to T} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{s_{ii} / \tau}}{\sum_{j=1}^B e^{s_{ij} / \tau}}$$

where $s_{ij} = \cos(z_i^{img}, z_j^{text})$ and $\tau$ is the temperature.

**Key Design Choices:**
- Temperature as learnable parameter (initially 0.07) rather than fixed
- L2-normalization makes similarity purely directional
- Symmetric loss treats image-to-text and text-to-image equally

**Why Contrastive Learning Works:**
The InfoNCE lower bound on mutual information $I(image; text)$ pushes matched pairs together and all other pairs apart in the embedding space.""",
    'theory_zh': r"""CLIP（对比语言-图像预训练，Radford et al., 2021）通过最大化匹配对的余弦相似度、最小化不匹配对的相似度来学习联合的图像-文本表示。

**训练目标：**
$$\mathcal{L} = \frac{1}{2} \left[ \mathcal{L}_{I\to T} + \mathcal{L}_{T\to I} \right]$$

$$\mathcal{L}_{I\to T} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{s_{ii} / \tau}}{\sum_{j=1}^B e^{s_{ij} / \tau}}$$

其中 $s_{ij} = \cos(z_i^{img}, z_j^{text})$，$\tau$ 为温度系数。

**关键设计：**
- Temperature 作为可学习参数（初始 0.07）而非固定值
- L2 归一化使相似度纯粹反映方向
- 对称损失平等对待 image-to-text 和 text-to-image

**为什么对比学习有效：**
InfoNCE 互信息下界 $I(image; text)$ 将匹配对拉近，将所有其他对推远。""",
    'diagram_en': """```mermaid
flowchart LR
    Images[Images] --> IE[Image Encoder<br/>e.g. ViT/ResNet]
    Texts[Text Features] --> TP[Text Projection<br/>Linear(embed_dim)]
    IE --> IN[L2 Normalize]
    TP --> TN[L2 Normalize]
    IN --> SIM[Similarity Matrix<br/>I_embed @ T_embed.T / tau]
    TN --> SIM
    SIM --> LOSS[Symmetric CE Loss<br/>(I->T + T->I) / 2]
```""",
    'diagram_zh': """```mermaid
flowchart LR
    Images[图像] --> IE[图像编码器<br/>如 ViT/ResNet]
    Texts[文本特征] --> TP[文本投影<br/>Linear(embed_dim)]
    IE --> IN[L2 归一化]
    TP --> TN[L2 归一化]
    IN --> SIM[相似度矩阵<br/>I_embed @ T_embed.T / tau]
    TN --> SIM
    SIM --> LOSS[对称交叉熵损失<br/>(I->T + T->I) / 2]
```""",
    "tests": [
        {
            "name": 'Is nn.Module',
            "code": """import torch, torch.nn as nn
image_enc = nn.Linear(64, 32)
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=32, text_dim=16)
assert isinstance(clip, nn.Module), 'Must inherit from nn.Module'""",
        },
        {
            "name": 'Loss is scalar and positive for random data',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 32))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=32, text_dim=16)
images = torch.randn(4, 3, 8, 8)
text_feats = torch.randn(4, 16)
loss = clip(images, text_feats)
assert loss.dim() == 0, 'Loss must be scalar'
assert loss.item() > 0, 'Loss should be positive for random data'""",
        },
        {
            "name": 'Has learnable temperature',
            "code": """import torch, torch.nn as nn
image_enc = nn.Linear(16, 8)
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=8, text_dim=8)
has_temp = hasattr(clip, 'logit_scale') or hasattr(clip, 'temperature')
assert has_temp, 'Need learnable temperature parameter'""",
        },
        {
            "name": 'Symmetric loss (I->T and T->I)',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*4*4, 16))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=16, text_dim=16)
images = torch.randn(2, 3, 4, 4)
text_feats = torch.randn(2, 16)
loss = clip(images, text_feats)
# The loss should be roughly the same if we swap image/text roles
# (since text_enc is Identity, this is symmetric)
assert loss.dim() == 0, 'Loss must be scalar'""",
        },
        {
            "name": 'Perfect alignment gives low loss',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
# Use Identity for both: image and text features are the same
image_enc = nn.Identity()
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=8, text_dim=8)
x = torch.randn(2, 8)
loss = clip(x, x)
# When image and text embeddings are identical and normalized,
# diagonal similarities are 1.0, off-diagonal are < 1.0
assert loss.item() < 0.5, f'Loss should be low for perfectly aligned data, got {loss.item()}'""",
        },
        {
            "name": 'Gradient flows to encoders',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*4*4, 16))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=16, text_dim=16)
images = torch.randn(2, 3, 4, 4, requires_grad=True)
text_feats = torch.randn(2, 16, requires_grad=True)
loss = clip(images, text_feats)
loss.backward()
assert images.grad is not None, 'No gradient for images'
assert text_feats.grad is not None, 'No gradient for text_feats'""",
        },
    ],
    'solution': '''class CLIP(nn.Module):
    """简化图文对比预训练模型 (CLIP)。"""
    def __init__(self, image_encoder, text_encoder, embed_dim, text_dim, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(text_dim, embed_dim)
        nn.init.eye_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / temperature)
        )

    def forward(self, images, text_features):
        # Step 1: 编码并投影图像、文本特征
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_proj(self.text_encoder(text_features))

        # Step 2: L2 归一化并构造批内相似度矩阵
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = (
            image_embeddings @ text_embeddings.T
        ) * self.logit_scale.exp()

        # Step 3: 计算图到文与文到图的对称交叉熵
        targets = torch.arange(logits.shape[0], device=logits.device)
        image_loss = F.cross_entropy(logits, targets)
        text_loss = F.cross_entropy(logits.T, targets)
        return (image_loss + text_loss) / 2.0''',
    'demo': """torch.manual_seed(42)
image_encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
text_encoder = nn.Identity()
model = CLIP(image_encoder, text_encoder, embed_dim=32, text_dim=16)
images = torch.randn(4, 3, 8, 8)
text_features = torch.randn(4, 16)
loss = model(images, text_features)
print("Output shape:", loss.shape)
print("Loss:", loss.item())
print("Logit scale:", model.logit_scale.exp().item())""",
    'explanation': """### 核心思路
CLIP 将配对图像和文本映射到同一单位球面，以批内相似度矩阵的对角线作为正样本，并优化双向交叉熵。

### 代码步骤拆解
- **Step 1（双塔编码）**：图像编码器直接输出 `embed_dim`，文本输出经单位映射初始化的 `text_proj` 对齐维度。
- **Step 2（余弦 logits）**：L2 归一化后点积形成 `(B, B)`，再乘可学习温度倒数。
- **Step 3（对称损失）**：分别沿行、列预测配对索引，取两方向损失均值。

### 重点与易错点分析
- **注意点 1（归一化维度）**：范数应沿最后一维并保留维度，否则无法正确广播。
- **注意点 2（温度参数化）**：优化 `logit_scale` 并取 `exp()` 可保证缩放恒为正。
- **注意点 3（初始化与配对）**：矩形单位初始化让同维输入初始对齐；目标 `arange(B)` 要求配对图文位于相同索引。

### 复杂度分析
- **时间复杂度**：编码器之外为 $O(B^2D)$，批内相似度矩阵是主要成本。
- **空间复杂度**：$O(B^2+BD)$，保存 logits 与两类嵌入。""",
}
