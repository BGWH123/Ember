"""DeepSeek-V3 Sequential Multi-Token Prediction (MTP) Module task."""

TASK = {
    'title': 'DeepSeek-V3 Sequential Multi-Token Prediction (MTP) Module',
    'title_zh': 'DeepSeek-V3 链式多 Token 预测（MTP）模块',
    'difficulty': 'Hard',
    'category': '训练技巧',
    'function_name': 'deepseek_mtp_step',
    'starter_code': """def deepseek_mtp_step(h_prev, next_token_embeds, rms_w_h, rms_w_emb, W_proj, W_head, targets, eps=1e-6):
    pass""",
    'hint': """RMS-normalize the two inputs independently, concatenate them, project back to the model dimension, and apply a stable mean cross-entropy over flattened positions.""",
    'hint_zh': '分别对两个输入进行 RMS 归一化，拼接后投影回模型维度，再对展平的位置计算数值稳定的平均交叉熵。',
    'description_en': """Implement one chained DeepSeek-V3 MTP depth: fuse the previous-depth representation with the shifted next-token embedding and predict the target at this depth.""",
    'description_zh': '实现一层链式 DeepSeek-V3 MTP：融合前一深度表示与移位后的下一 token 嵌入，并预测该深度的目标。',
    'theory_en': """Sequential MTP reuses the representation from depth k-1. Separate RMSNorms balance the two streams before a learned fusion projection, while the shared unembedding head aligns every depth with the base vocabulary space.""",
    'theory_zh': '链式 MTP 复用深度 k-1 的表示。独立 RMSNorm 在融合前平衡两路信号，共享反嵌入头使各深度都对齐到基础词表空间。',
    'diagram_en': r"""```text
h_(k-1) -> RMSNorm --\
                         concat -> W_proj -> h_k -> shared W_head -> logits -> CE
shifted embedding -> RMSNorm --/
```""",
    'diagram_zh': r"""```text
h_(k-1) -> RMSNorm --\
                         拼接 -> W_proj -> h_k -> 共享 W_head -> logits -> CE
移位 token 嵌入 -> RMSNorm --/
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
B,S,D,V = 2,3,4,7
h = torch.randn(B,S,D); e = torch.randn(B,S,D)
hp, logits, loss = {fn}(h,e,torch.ones(D),torch.ones(D),torch.randn(2*D,D),torch.randn(D,V),torch.randint(V,(B,S)))
assert hp.shape == (B,S,D) and logits.shape == (B,S,V)
assert loss.shape == torch.Size([])""",
        },
        {
            "name": 'exact RMS fusion and stable loss',
            "code": """import torch
import torch.nn.functional as F
torch.manual_seed(3)
B,S,D,V = 1,2,3,5
h = torch.randn(B,S,D,dtype=torch.double); e = torch.randn(B,S,D,dtype=torch.double)
wh = torch.randn(D,dtype=torch.double); we = torch.randn(D,dtype=torch.double)
wp = torch.randn(2*D,D,dtype=torch.double); wo = torch.randn(D,V,dtype=torch.double)
t = torch.tensor([[1,4]])
hc, logits, loss = {fn}(h,e,wh,we,wp,wo,t,eps=1e-7)
hn = h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+1e-7)*wh
en = e*torch.rsqrt(e.pow(2).mean(-1,keepdim=True)+1e-7)*we
rh = torch.cat((hn,en),-1)@wp; rl = rh@wo
rr = F.cross_entropy(rl.reshape(-1,V),t.reshape(-1))
assert torch.allclose(hc,rh,atol=1e-12,rtol=1e-12)
assert torch.allclose(logits,rl,atol=1e-12,rtol=1e-12) and torch.allclose(loss,rr,atol=1e-12,rtol=1e-12)""",
        },
        {
            "name": 'zero-input normalization edge',
            "code": """import torch
B,S,D,V = 1,1,2,3
hc, logits, loss = {fn}(torch.zeros(B,S,D),torch.zeros(B,S,D),torch.ones(D),torch.ones(D),torch.randn(2*D,D),torch.randn(D,V),torch.tensor([[2]]))
assert torch.equal(hc,torch.zeros_like(hc)) and torch.equal(logits,torch.zeros_like(logits))
assert torch.allclose(loss,torch.log(torch.tensor(float(V))))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
B,S,D,V = 2,2,3,4
h = torch.randn(B,S,D,requires_grad=True); e = torch.randn(B,S,D,requires_grad=True)
wh = torch.randn(D,requires_grad=True); we = torch.randn(D,requires_grad=True)
wp = torch.randn(2*D,D,requires_grad=True); wo = torch.randn(D,V,requires_grad=True)
hc, logits, loss = {fn}(h,e,wh,we,wp,wo,torch.randint(V,(B,S)))
loss.backward()
assert hc.requires_grad and logits.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (h,e,wh,we,wp,wo))""",
        },
    ],
    'solution': '''def deepseek_mtp_step(h_prev, next_token_embeds, rms_w_h, rms_w_emb, W_proj, W_head, targets, eps=1e-6):
    """执行链式多 Token 预测的一层融合 (DeepSeek MTP Step)。"""
    # Step 1: 分别对前层表示与移位 token 嵌入执行 RMSNorm
    hidden_scale = torch.rsqrt(h_prev.square().mean(dim=-1, keepdim=True) + eps)
    embed_scale = torch.rsqrt(next_token_embeds.square().mean(dim=-1, keepdim=True) + eps)
    hidden_norm = h_prev * hidden_scale * rms_w_h
    embed_norm = next_token_embeds * embed_scale * rms_w_emb

    # Step 2: 拼接双路表示并投影回模型维度
    hidden_current = torch.cat((hidden_norm, embed_norm), dim=-1) @ W_proj
    logits = hidden_current @ W_head

    # Step 3: 展平批次与序列位置计算平均交叉熵
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return hidden_current, logits, loss''',
    'demo': """torch.manual_seed(42)
B, S, D, V = 2, 8, 16, 100
h_prev = torch.randn(B, S, D)
embeds = torch.randn(B, S, D)
hidden, logits, loss = deepseek_mtp_step(h_prev, embeds, torch.ones(D), torch.ones(D), torch.randn(2 * D, D), torch.randn(D, V), torch.randint(V, (B, S)))
print("Hidden shape:", hidden.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())""",
    'explanation': """### 核心思路
MTP 深度将前一层表示与移位后的下一 token 嵌入分别 RMS 归一化，再拼接投影为新表示，并通过共享词表头计算交叉熵。

### 代码步骤拆解
- **Step 1（双路 RMSNorm）**：对两个 `(B, S, D)` 张量沿 `D` 维计算均方根缩放，并乘各自学习权重。
- **Step 2（融合与解码）**：拼接得到 `(B, S, 2D)`，经 `W_proj` 回到 `(B, S, D)`，再经 `W_head` 得到 `(B, S, V)`。
- **Step 3（监督损失）**：将前两维展平为 `B*S` 个位置，计算平均词表交叉熵。

### 重点与易错点分析
- **注意点 1（归一化稳定性）**：`eps` 加在均方值内部后使用 `rsqrt`，零输入也能得到有限且为零的归一化结果。
- **注意点 2（参数矩阵方向）**：本题矩阵约定为右乘，`W_proj` 是 `(2D, D)`，`W_head` 是 `(D, V)`。
- **注意点 3（展平对齐）**：`logits` 和 `targets` 必须按相同的批次—序列顺序展平，避免标签错位。

### 复杂度分析
- **时间复杂度**：$O(BS(D^2+DV))$，融合投影与词表投影占主要计算。
- **空间复杂度**：$O(BS(D+V))$，主要保存融合表示和词表 logits。""",
}
