"""MMDiT joint image-text attention task."""

TASK = {
    "title": "MMDiT Joint Image-Text Attention",
    "title_zh": "MMDiT 图文联合注意力机制",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "mmdit_joint_attention",
    "starter_code": '''def mmdit_joint_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img):
    pass
''',
    "hint": "Concatenate each modality's Q, K, and V on sequence axis 2, apply scaled dot-product attention once, then split at the text sequence length.",
    "hint_zh": "沿序列轴 2 分别拼接两种模态的 Q、K、V，执行一次缩放点积注意力，再在文本序列长度处拆分。",
    "description_en": "Implement MMDiT joint attention over text tensors (B,H,L_txt,D) and image tensors (B,H,L_img,D). Concatenate tokens, attend jointly, and return modality-specific outputs.",
    "description_zh": "对文本张量 (B,H,L_txt,D) 与图像张量 (B,H,L_img,D) 实现 MMDiT 联合注意力。拼接 token、执行联合注意力并返回各模态输出。",
    "theory_en": "Joint attention lets every text and image query attend to keys and values from both modalities. Splitting only after attention preserves modality streams while enabling full cross-modal interaction.",
    "theory_zh": "联合注意力让每个文本与图像查询都能关注两种模态的键和值。注意力后再拆分既保留模态流，又实现完整跨模态交互。",
    "diagram_en": "```text\ntext Q/K/V --\\\n               concatenate -> joint attention -> split -> text out, image out\nimage Q/K/V --/\n```",
    "diagram_zh": "```text\n文本 Q/K/V --\\\n                拼接 -> 联合注意力 -> 拆分 -> 文本输出、图像输出\n图像 Q/K/V --/\n```",
    "tests": [
        {
            "name": "split output shapes",
            "code": '''import torch
args=[torch.randn(2,3,4,5) for _ in range(3)]+[torch.randn(2,3,7,5) for _ in range(3)]
t,i={fn}(*args)
assert t.shape==(2,3,4,5) and i.shape==(2,3,7,5)
''',
        },
        {
            "name": "exact joint reference",
            "code": '''import torch
torch.manual_seed(5)
qt,kt,vt=[torch.randn(1,2,3,4) for _ in range(3)]
qi,ki,vi=[torch.randn(1,2,2,4) for _ in range(3)]
ot,oi={fn}(qt,kt,vt,qi,ki,vi)
Q=torch.cat((qt,qi),2); K=torch.cat((kt,ki),2); V=torch.cat((vt,vi),2)
ref=torch.softmax((Q@K.transpose(-2,-1))/(4**.5),-1)@V
assert torch.equal(ot,ref[:,:,:3]) and torch.equal(oi,ref[:,:,3:])
''',
        },
        {
            "name": "zero-query uniform edge case",
            "code": '''import torch
qt=torch.zeros(1,1,1,2); qi=torch.zeros(1,1,2,2)
kt=torch.randn_like(qt); ki=torch.randn_like(qi)
vt=torch.tensor([[[[1.,2.]]]]); vi=torch.tensor([[[[3.,4.],[5.,6.]]]])
ot,oi={fn}(qt,kt,vt,qi,ki,vi)
mean=torch.tensor([3.,4.])
assert torch.allclose(ot,mean.reshape(1,1,1,2)) and torch.allclose(oi,mean.reshape(1,1,1,2).expand_as(oi))
''',
        },
        {
            "name": "gradient flow through both streams",
            "code": '''import torch
xs=[torch.randn(1,2,3,4,requires_grad=True) for _ in range(3)]
ys=[torch.randn(1,2,2,4,requires_grad=True) for _ in range(3)]
a,b={fn}(*(xs+ys)); (a.square().sum()+b.square().sum()).backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in xs+ys)
''',
        },
    ],
    "solution": '''import torch

def mmdit_joint_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img):
    text_length = q_txt.size(2)
    Q = torch.cat((q_txt, q_img), dim=2)
    K = torch.cat((k_txt, k_img), dim=2)
    V = torch.cat((v_txt, v_img), dim=2)
    scale = Q.size(-1) ** 0.5
    weights = torch.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1)
    out = weights @ V
    return out[:, :, :text_length], out[:, :, text_length:]
''',
    "demo": '''text = [torch.randn(2, 4, 8, 16) for _ in range(3)]
image = [torch.randn(2, 4, 32, 16) for _ in range(3)]
print([x.shape for x in mmdit_joint_attention(*(text + image))])
''',
    "explanation": "The softmax spans the combined key sequence, so each query distributes probability mass jointly across text and image tokens.",
}
