"""SigLIP pairwise sigmoid contrastive loss task."""

TASK = {
    "title": "SigLIP Pairwise Sigmoid Contrastive Loss",
    "title_zh": "SigLIP 成对 Sigmoid 对比损失",
    "difficulty": "Medium",
    "category": "多模态",
    "function_name": "siglip_loss",
    "starter_code": '''def siglip_loss(image_embeds, text_embeds, temp_log, bias, eps=1e-8):
    pass
''',
    "hint": "Normalize both embedding sets, form every image-text logit, label diagonal pairs +1 and off-diagonal pairs -1, then average the summed log-sigmoid loss by N.",
    "hint_zh": "归一化两组嵌入，构造所有图文 logits，将对角配对标为 +1、其余标为 -1，再将 log-sigmoid 损失总和除以 N。",
    "description_en": "Implement the SigLIP pairwise sigmoid contrastive objective for matching image and text embeddings.",
    "description_zh": "实现用于匹配图像与文本嵌入的 SigLIP 成对 sigmoid 对比目标。",
    "theory_en": "Unlike softmax contrastive learning, SigLIP treats every entry of the N by N similarity matrix as an independent binary pair. The learned log-temperature controls scale and the bias controls the positive prior.",
    "theory_zh": "不同于 softmax 对比学习，SigLIP 将 N×N 相似度矩阵的每个元素视为独立二元配对；可学习的对数温度控制尺度，偏置控制正样本先验。",
    "diagram_en": "```text\nimage,text -> L2 normalize -> pairwise logits -> signed log-sigmoid -> sum / N\n```",
    "diagram_zh": "```text\n图像、文本 -> L2 归一化 -> 成对 logits -> 带符号 log-sigmoid -> 求和 / N\n```",
    "tests": [
        {"name": "scalar shape", "code": '''import torch
x = torch.randn(4, 6)
y = torch.randn(4, 6)
out = {fn}(x, y, torch.tensor(0.2), torch.tensor(-1.0))
assert out.shape == torch.Size([]) and torch.isfinite(out)
'''},
        {"name": "exact reference", "code": '''import torch
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
assert torch.allclose(out, ref, atol=1e-7)
'''},
        {"name": "single pair edge case", "code": '''import torch
import torch.nn.functional as F
x = torch.zeros(1, 3)
y = torch.zeros(1, 3)
b = torch.tensor(0.7)
out = {fn}(x, y, torch.tensor(0.), b)
assert torch.allclose(out, -F.logsigmoid(b)) and out.ndim == 0
'''},
        {"name": "gradient flow", "code": '''import torch
x = torch.randn(3, 5, requires_grad=True)
y = torch.randn(3, 5, requires_grad=True)
t = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(-0.5, requires_grad=True)
{fn}(x, y, t, b).backward()
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (x, y, t, b))
'''},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def siglip_loss(image_embeds, text_embeds, temp_log, bias, eps=1e-8):
    img = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + eps)
    S = torch.exp(temp_log) * (img @ txt.T) + bias
    N = S.shape[0]
    Y = 2.0 * torch.eye(N, device=S.device, dtype=S.dtype) - 1.0
    return -F.logsigmoid(Y * S).sum() / N
''',
    "demo": '''image = torch.randn(8, 64)
text = torch.randn(8, 64)
print(siglip_loss(image, text, torch.tensor(0.0), torch.tensor(-1.0)))
''',
    "explanation": "The diagonal is positive and every off-diagonal pair is negative. Dividing by N follows SigLIP's batch-level normalization rather than averaging all N squared pairs.",
}
