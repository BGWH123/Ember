"""MMDiT Joint Image-Text Attention task."""

TASK = {
    'title': 'MMDiT Joint Image-Text Attention',
    'title_zh': 'MMDiT 图文联合注意力机制',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'mmdit_joint_attention',
    'starter_code': """def mmdit_joint_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img):
    pass""",
    'hint': """Concatenate each modality's Q, K, and V on sequence axis 2, apply scaled dot-product attention once, then split at the text sequence length.""",
    'hint_zh': '沿序列轴 2 分别拼接两种模态的 Q、K、V，执行一次缩放点积注意力，再在文本序列长度处拆分。',
    'description_en': """Implement MMDiT joint attention over text tensors (B,H,L_txt,D) and image tensors (B,H,L_img,D). Concatenate tokens, attend jointly, and return modality-specific outputs.""",
    'description_zh': '对文本张量 (B,H,L_txt,D) 与图像张量 (B,H,L_img,D) 实现 MMDiT 联合注意力。拼接 token、执行联合注意力并返回各模态输出。',
    'theory_en': """Joint attention lets every text and image query attend to keys and values from both modalities. Splitting only after attention preserves modality streams while enabling full cross-modal interaction.""",
    'theory_zh': '联合注意力让每个文本与图像查询都能关注两种模态的键和值。注意力后再拆分既保留模态流，又实现完整跨模态交互。',
    'diagram_en': r"""```text
text Q/K/V --\
               concatenate -> joint attention -> split -> text out, image out
image Q/K/V --/
```""",
    'diagram_zh': r"""```text
文本 Q/K/V --\
                拼接 -> 联合注意力 -> 拆分 -> 文本输出、图像输出
图像 Q/K/V --/
```""",
    "tests": [
        {
            "name": 'split output shapes',
            "code": """import torch
args=[torch.randn(2,3,4,5) for _ in range(3)]+[torch.randn(2,3,7,5) for _ in range(3)]
t,i={fn}(*args)
assert t.shape==(2,3,4,5) and i.shape==(2,3,7,5)""",
        },
        {
            "name": 'exact joint reference',
            "code": """import torch
torch.manual_seed(5)
qt,kt,vt=[torch.randn(1,2,3,4) for _ in range(3)]
qi,ki,vi=[torch.randn(1,2,2,4) for _ in range(3)]
ot,oi={fn}(qt,kt,vt,qi,ki,vi)
Q=torch.cat((qt,qi),2); K=torch.cat((kt,ki),2); V=torch.cat((vt,vi),2)
ref=torch.softmax((Q@K.transpose(-2,-1))/(4**.5),-1)@V
assert torch.equal(ot,ref[:,:,:3]) and torch.equal(oi,ref[:,:,3:])""",
        },
        {
            "name": 'zero-query uniform edge case',
            "code": """import torch
qt=torch.zeros(1,1,1,2); qi=torch.zeros(1,1,2,2)
kt=torch.randn_like(qt); ki=torch.randn_like(qi)
vt=torch.tensor([[[[1.,2.]]]]); vi=torch.tensor([[[[3.,4.],[5.,6.]]]])
ot,oi={fn}(qt,kt,vt,qi,ki,vi)
mean=torch.tensor([3.,4.])
assert torch.allclose(ot,mean.reshape(1,1,1,2)) and torch.allclose(oi,mean.reshape(1,1,1,2).expand_as(oi))""",
        },
        {
            "name": 'gradient flow through both streams',
            "code": """import torch
xs=[torch.randn(1,2,3,4,requires_grad=True) for _ in range(3)]
ys=[torch.randn(1,2,2,4,requires_grad=True) for _ in range(3)]
a,b={fn}(*(xs+ys)); (a.square().sum()+b.square().sum()).backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in xs+ys)""",
        },
    ],
    'solution': '''def mmdit_joint_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img):
    """计算图文联合注意力 (MMDiT Joint Attention)。"""
    S_txt = q_txt.shape[2]

    # Step 1: 沿序列维拼接文本与图像 Q、K、V
    q = torch.cat((q_txt, q_img), dim=2)
    k = torch.cat((k_txt, k_img), dim=2)
    v = torch.cat((v_txt, v_img), dim=2)

    # Step 2: 在联合 token 序列上执行一次缩放点积注意力
    d_h = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)
    attn_weights = torch.softmax(scores, dim=-1)
    joint_out = attn_weights @ v

    # Step 3: 按原文本长度拆回两个模态输出
    text_out = joint_out[:, :, :S_txt]
    image_out = joint_out[:, :, S_txt:]
    return text_out, image_out''',
    'demo': """torch.manual_seed(42)
text = [torch.randn(2, 4, 3, 8) for _ in range(3)]
image = [torch.randn(2, 4, 6, 8) for _ in range(3)]
text_out, image_out = mmdit_joint_attention(*(text + image))
print("Text output shape:", text_out.shape)
print("Image output shape:", image_out.shape)
print("Sample values:", text_out[0, 0, 0, :4])""",
    'explanation': r"""### 核心思路
MMDiT 把文本与图像的 Q/K/V 沿 token 轴拼接，在总长度 $S_t+S_i$ 上执行同一次注意力，使每个查询都能访问两种模态，最后按文本长度拆回独立流。

### 代码步骤拆解
- **Step 1（模态拼接）**：构造联合 `(B, H, S_t + S_i, D)` 张量。
- **Step 2（联合注意）**：按 $\sqrt D$ 缩放分数并在全部键上 softmax。
- **Step 3（输出拆分）**：使用 `S_txt` 切分 query 对应的两个输出区间。

### 重点与易错点分析
- **注意点 1（拼接轴）**：输入布局是 `(B, H, S, D)`，序列轴固定为 `dim=2`。
- **注意点 2（拆分时机）**：必须在注意力完成后拆分，提前分开会失去跨模态交互。
- **注意点 3（概率竞争）**：softmax 覆盖联合键序列，文本与图像 token 共享同一概率质量。

### 复杂度分析
- **时间复杂度**：$O(BH(S_t+S_i)^2D)$，联合注意力矩阵乘法占主导。
- **空间复杂度**：$O(BH(S_t+S_i)^2)$，主要为联合注意力权重。""",
}
