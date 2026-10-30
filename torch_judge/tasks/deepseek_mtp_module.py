"""DeepSeek-V3 sequential multi-token prediction task."""

TASK = {
    "title": "DeepSeek-V3 Sequential Multi-Token Prediction (MTP) Module",
    "title_zh": "DeepSeek-V3 链式多 Token 预测（MTP）模块",
    "difficulty": "Hard",
    "category": "训练技巧",
    "function_name": "deepseek_mtp_step",
    "starter_code": '''import torch

def deepseek_mtp_step(h_prev, next_token_embeds, rms_w_h, rms_w_emb, W_proj, W_head, targets, eps=1e-6):
    pass
''',
    "hint": "RMS-normalize the two inputs independently, concatenate them, project back to the model dimension, and apply a stable mean cross-entropy over flattened positions.",
    "hint_zh": "分别对两个输入进行 RMS 归一化，拼接后投影回模型维度，再对展平的位置计算数值稳定的平均交叉熵。",
    "description_en": "Implement one chained DeepSeek-V3 MTP depth: fuse the previous-depth representation with the shifted next-token embedding and predict the target at this depth.",
    "description_zh": "实现一层链式 DeepSeek-V3 MTP：融合前一深度表示与移位后的下一 token 嵌入，并预测该深度的目标。",
    "theory_en": "Sequential MTP reuses the representation from depth k-1. Separate RMSNorms balance the two streams before a learned fusion projection, while the shared unembedding head aligns every depth with the base vocabulary space.",
    "theory_zh": "链式 MTP 复用深度 k-1 的表示。独立 RMSNorm 在融合前平衡两路信号，共享反嵌入头使各深度都对齐到基础词表空间。",
    "diagram_en": "```text\nh_(k-1) -> RMSNorm --\\\n                         concat -> W_proj -> h_k -> shared W_head -> logits -> CE\nshifted embedding -> RMSNorm --/\n```",
    "diagram_zh": "```text\nh_(k-1) -> RMSNorm --\\\n                         拼接 -> W_proj -> h_k -> 共享 W_head -> logits -> CE\n移位 token 嵌入 -> RMSNorm --/\n```",
    "tests": [
        {
            "name": "output shapes",
            "code": '''import torch
B,S,D,V = 2,3,4,7
h = torch.randn(B,S,D); e = torch.randn(B,S,D)
hp, logits, loss = {fn}(h,e,torch.ones(D),torch.ones(D),torch.randn(2*D,D),torch.randn(D,V),torch.randint(V,(B,S)))
assert hp.shape == (B,S,D) and logits.shape == (B,S,V)
assert loss.shape == torch.Size([])
''',
        },
        {
            "name": "exact RMS fusion and stable loss",
            "code": '''import torch
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
assert torch.allclose(logits,rl,atol=1e-12,rtol=1e-12) and torch.allclose(loss,rr,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "zero-input normalization edge",
            "code": '''import torch
B,S,D,V = 1,1,2,3
hc, logits, loss = {fn}(torch.zeros(B,S,D),torch.zeros(B,S,D),torch.ones(D),torch.ones(D),torch.randn(2*D,D),torch.randn(D,V),torch.tensor([[2]]))
assert torch.equal(hc,torch.zeros_like(hc)) and torch.equal(logits,torch.zeros_like(logits))
assert torch.allclose(loss,torch.log(torch.tensor(float(V))))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
B,S,D,V = 2,2,3,4
h = torch.randn(B,S,D,requires_grad=True); e = torch.randn(B,S,D,requires_grad=True)
wh = torch.randn(D,requires_grad=True); we = torch.randn(D,requires_grad=True)
wp = torch.randn(2*D,D,requires_grad=True); wo = torch.randn(D,V,requires_grad=True)
hc, logits, loss = {fn}(h,e,wh,we,wp,wo,torch.randint(V,(B,S)))
loss.backward()
assert hc.requires_grad and logits.requires_grad
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (h,e,wh,we,wp,wo))
''',
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def deepseek_mtp_step(h_prev, next_token_embeds, rms_w_h, rms_w_emb, W_proj, W_head, targets, eps=1e-6):
    h_norm = h_prev * torch.rsqrt(h_prev.pow(2).mean(dim=-1, keepdim=True) + eps) * rms_w_h
    emb_norm = next_token_embeds * torch.rsqrt(next_token_embeds.pow(2).mean(dim=-1, keepdim=True) + eps) * rms_w_emb
    h_curr = torch.cat((h_norm, emb_norm), dim=-1) @ W_proj
    logits = h_curr @ W_head
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return h_curr, logits, loss
''',
    "demo": '''B, S, D, V = 2, 8, 16, 100
h, e = torch.randn(B,S,D), torch.randn(B,S,D)
out = deepseek_mtp_step(h,e,torch.ones(D),torch.ones(D),torch.randn(2*D,D),torch.randn(D,V),torch.randint(V,(B,S)))
print(out[0].shape, out[1].shape, out[2].item())
''',
    "explanation": "Each stream receives its own learned RMS scale. Their fused representation is both passed to the next MTP depth and decoded by the shared vocabulary head; flattened cross-entropy averages all B*S targets.",
}
