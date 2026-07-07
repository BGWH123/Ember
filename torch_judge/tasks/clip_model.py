"""CLIP Model (Image-Text Contrastive Learning) task."""

TASK = {
    "title": "CLIP Model (Image-Text Contrastive Learning)",
    "title_zh": "CLIP 模型（图文对比学习）",
    "difficulty": "Hard",
    "category": "多模态",
    "description_en": (
        "Implement a simplified CLIP model for image-text contrastive learning.\n\nCLIP jointly trains an image encoder and a text encoder to produce aligned embeddings in a shared space. The training objective is a symmetric cross-entropy loss over the cosine similarity matrix.\n\n**Signature:** `CLIP(image_encoder, text_encoder, embed_dim, temperature=0.07)` (nn.Module)\n\n**Forward:** `forward(images, text_features) -> Tensor`\n- `images` — image tensor (B, C, H, W)\n- `text_features` — pre-encoded text features (B, text_dim)\n\n**Returns:** scalar contrastive loss\n\n**Architecture:**\n1. `image_embed = image_encoder(images)` → (B, embed_dim)\n2. `text_embed = text_projection(text_features)` → (B, embed_dim)\n3. L2-normalize both embeddings\n4. Compute cosine similarity matrix: `logits = image_embed @ text_embed.T / temperature`\n5. Targets: diagonal (matched pairs)\n6. Loss: `(ce_loss(logits, targets) + ce_loss(logits.T, targets)) / 2`\n\n**Constraints:**\n- Use `nn.Linear(text_dim, embed_dim)` for text projection\n- Temperature is a learnable parameter initialized to 0.07\n- L2-normalize embeddings before computing similarity"
    ),
    "description_zh": (
        "实现简化的 CLIP 模型用于图文对比学习。\n\nCLIP 联合训练图像编码器和文本编码器，在共享空间中生成对齐的嵌入。训练目标是在余弦相似度矩阵上的对称交叉熵损失。\n\n**签名:** `CLIP(image_encoder, text_encoder, embed_dim, temperature=0.07)`（nn.Module）\n\n**前向传播:** `forward(images, text_features) -> Tensor`\n- `images` — 图像张量 (B, C, H, W)\n- `text_features` — 预编码文本特征 (B, text_dim)\n\n**返回:** 标量对比损失\n\n**架构：**\n1. `image_embed = image_encoder(images)` → (B, embed_dim)\n2. `text_embed = text_projection(text_features)` → (B, embed_dim)\n3. 对两个嵌入做 L2 归一化\n4. 计算余弦相似度矩阵：`logits = image_embed @ text_embed.T / temperature`\n5. 目标：对角线（匹配对）\n6. 损失：`(ce_loss(logits, targets) + ce_loss(logits.T, targets)) / 2`\n\n**约束：**\n- 使用 `nn.Linear(text_dim, embed_dim)` 做文本投影\n- Temperature 是可学习参数，初始值 0.07\n- 计算相似度前对嵌入做 L2 归一化"
    ),
    "function_name": "CLIP",
    "hint": (
        "1. `text_proj = nn.Linear(text_dim, embed_dim)`\n2. `logit_scale = nn.Parameter(torch.ones([]) * log(1/0.07))`\n3. `temperature = 1 / exp(logit_scale)`\n4. Normalize: `x = x / x.norm(dim=-1, keepdim=True)`\n5. `logits = image_embed @ text_embed.T * exp(logit_scale)`"
    ),
    "hint_zh": (
        "1. `text_proj = nn.Linear(text_dim, embed_dim)`\n2. `logit_scale = nn.Parameter(torch.ones([]) * log(1/0.07))`\n3. `temperature = 1 / exp(logit_scale)`\n4. 归一化：`x = x / x.norm(dim=-1, keepdim=True)`\n5. `logits = image_embed @ text_embed.T * exp(logit_scale)`"
    ),
    "theory_en": (
        "CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021) learns joint image-text representations\nby maximizing the cosine similarity of matched pairs and minimizing it for unmatched pairs.\n\n**Training Objective:**\n$$\\mathcal{L} = \\frac{1}{2} \\left[ \\mathcal{L}_{I\\to T} + \\mathcal{L}_{T\\to I} \\right]$$\n\n$$\\mathcal{L}_{I\\to T} = -\\frac{1}{B} \\sum_{i=1}^B \\log \\frac{e^{s_{ii} / \\tau}}{\\sum_{j=1}^B e^{s_{ij} / \\tau}}$$\n\nwhere $s_{ij} = \\cos(z_i^{img}, z_j^{text})$ and $\\tau$ is the temperature.\n\n**Key Design Choices:**\n- Temperature as learnable parameter (initially 0.07) rather than fixed\n- L2-normalization makes similarity purely directional\n- Symmetric loss treats image-to-text and text-to-image equally\n\n**Why Contrastive Learning Works:**\nThe InfoNCE lower bound on mutual information $I(image; text)$ pushes matched pairs together and all other pairs apart in the embedding space."
    ),
    "theory_zh": (
        "CLIP（对比语言-图像预训练，Radford et al., 2021）通过最大化匹配对的余弦相似度、最小化不匹配对的相似度来学习联合的图像-文本表示。\n\n**训练目标：**\n$$\\mathcal{L} = \\frac{1}{2} \\left[ \\mathcal{L}_{I\\to T} + \\mathcal{L}_{T\\to I} \\right]$$\n\n$$\\mathcal{L}_{I\\to T} = -\\frac{1}{B} \\sum_{i=1}^B \\log \\frac{e^{s_{ii} / \\tau}}{\\sum_{j=1}^B e^{s_{ij} / \\tau}}$$\n\n其中 $s_{ij} = \\cos(z_i^{img}, z_j^{text})$，$\\tau$ 为温度系数。\n\n**关键设计：**\n- Temperature 作为可学习参数（初始 0.07）而非固定值\n- L2 归一化使相似度纯粹反映方向\n- 对称损失平等对待 image-to-text 和 text-to-image\n\n**为什么对比学习有效：**\nInfoNCE 互信息下界 $I(image; text)$ 将匹配对拉近，将所有其他对推远。"
    ),
    "diagram_en": (
        "```mermaid\nflowchart LR\n    Images[Images] --> IE[Image Encoder<br/>e.g. ViT/ResNet]\n    Texts[Text Features] --> TP[Text Projection<br/>Linear(embed_dim)]\n    IE --> IN[L2 Normalize]\n    TP --> TN[L2 Normalize]\n    IN --> SIM[Similarity Matrix<br/>I_embed @ T_embed.T / tau]\n    TN --> SIM\n    SIM --> LOSS[Symmetric CE Loss<br/>(I->T + T->I) / 2]\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart LR\n    Images[图像] --> IE[图像编码器<br/>如 ViT/ResNet]\n    Texts[文本特征] --> TP[文本投影<br/>Linear(embed_dim)]\n    IE --> IN[L2 归一化]\n    TP --> TN[L2 归一化]\n    IN --> SIM[相似度矩阵<br/>I_embed @ T_embed.T / tau]\n    TN --> SIM\n    SIM --> LOSS[对称交叉熵损失<br/>(I->T + T->I) / 2]\n```"
    ),
    "tests": [
        {
            "name": "Is nn.Module",
            "code": """








import torch, torch.nn as nn
image_enc = nn.Linear(64, 32)
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=32, text_dim=16)
assert isinstance(clip, nn.Module), 'Must inherit from nn.Module'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Loss is scalar and positive for random data",
            "code": """








import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 32))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=32, text_dim=16)
images = torch.randn(4, 3, 8, 8)
text_feats = torch.randn(4, 16)
loss = clip(images, text_feats)
assert loss.dim() == 0, 'Loss must be scalar'
assert loss.item() > 0, 'Loss should be positive for random data'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Has learnable temperature",
            "code": """








import torch, torch.nn as nn
image_enc = nn.Linear(16, 8)
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=8, text_dim=8)
has_temp = hasattr(clip, 'logit_scale') or hasattr(clip, 'temperature')
assert has_temp, 'Need learnable temperature parameter'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Symmetric loss (I->T and T->I)",
            "code": """








import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*4*4, 16))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=16, text_dim=16)
images = torch.randn(2, 3, 4, 4)
text_feats = torch.randn(2, 16)
loss = clip(images, text_feats)
# The loss should be roughly the same if we swap image/text roles
# (since text_enc is Identity, this is symmetric)
assert loss.dim() == 0, 'Loss must be scalar'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Perfect alignment gives low loss",
            "code": """








import torch, torch.nn as nn
torch.manual_seed(0)
# Use Identity for both: image and text features are the same
image_enc = nn.Identity()
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=8, text_dim=8)
x = torch.randn(2, 8)
loss = clip(x, x)
# When image and text embeddings are identical and normalized,
# diagonal similarities are 1.0, off-diagonal are < 1.0
assert loss.item() < 0.5, f'Loss should be low for perfectly aligned data, got {loss.item()}'

            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flows to encoders",
            "code": """








import torch, torch.nn as nn
torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*4*4, 16))
text_enc = nn.Identity()
clip = {fn}(image_enc, text_enc, embed_dim=16, text_dim=16)
images = torch.randn(2, 3, 4, 4, requires_grad=True)
text_feats = torch.randn(2, 16, requires_grad=True)
loss = clip(images, text_feats)
loss.backward()
assert images.grad is not None, 'No gradient for images'
assert text_feats.grad is not None, 'No gradient for text_feats'

            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''





class CLIP(nn.Module):
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

    
    
    
    
    
    ''',
    "demo": '''







torch.manual_seed(0)
image_enc = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 32))
text_enc = nn.Identity()
clip = CLIP(image_enc, text_enc, embed_dim=32, text_dim=16)
images = torch.randn(4, 3, 8, 8)
text_feats = torch.randn(4, 16)
loss = clip(images, text_feats)
print("Loss:", loss.item())
print("Logit scale:", clip.logit_scale.exp().item())
    
    
    
    
    
    
    
    ''',
}
