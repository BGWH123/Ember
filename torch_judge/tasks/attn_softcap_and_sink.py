"""Attention soft-capping and StreamingLLM sink-mask task."""

TASK = {
    "title": "Attention Logit Soft-Capping & Streaming Sink Mask",
    "title_zh": "注意力 Logit 软截断与 StreamingLLM 汇聚掩码",
    "difficulty": "Medium",
    "category": "注意力机制",
    "function_name": "softcap_sink_attention",
    "starter_code": '''import torch
import math

def softcap_sink_attention(Q, K, V, softcap=50.0, sink_size=2, window_size=4):
    pass
''',
    "hint": "Construct query and key position grids. Combine causality with the union of the sink prefix and the strict lower bound `j > i-window_size`.",
    "hint_zh": "构造查询与键的位置网格，将因果条件与汇聚前缀、严格下界 `j > i-window_size` 的并集组合。",
    "description_en": "Apply smooth attention-logit capping and a causal streaming mask that preserves initial sink tokens plus a rolling local window.",
    "description_zh": "实现平滑注意力 logit 截断，以及保留起始汇聚 token 和滚动局部窗口的因果流式掩码。",
    "theory_en": "The tanh cap bounds extreme logits without a hard discontinuity. Attention sinks stabilize streaming models, while the local window bounds the amount of recent context retained.",
    "theory_zh": "tanh 软截断可连续地限制极端 logit。注意力汇聚 token 稳定流式模型，局部窗口则限制保留的近期上下文。",
    "diagram_en": "```text\nQK^T/sqrt(D) -> softcap*tanh(.) -> causal AND (sink OR recent) mask -> softmax -> V\n```",
    "diagram_zh": "```text\nQK^T/sqrt(D) -> softcap*tanh(.) -> 因果 AND（汇聚 OR 近期）掩码 -> softmax -> V\n```",
    "tests": [
        {
            "name": "output and mask shapes",
            "code": '''import torch
Q,K,V = [torch.randn(2,6,4) for _ in range(3)]
out, mask = {fn}(Q,K,V)
assert out.shape == (2,6,4) and mask.shape == (6,6)
assert mask.dtype == torch.bool
''',
        },
        {
            "name": "exact softcap and mask reference",
            "code": '''import torch
import math
torch.manual_seed(9)
Q,K,V = [torch.randn(1,5,3,dtype=torch.double) for _ in range(3)]
out, mask = {fn}(Q,K,V,softcap=2.5,sink_size=1,window_size=2)
i = torch.arange(5)[:,None]; j = torch.arange(5)[None,:]
rm = (j<=i)&((j<1)|(j>i-2))
s = (Q@K.transpose(-1,-2))/math.sqrt(3)
s = 2.5*torch.tanh(s/2.5)
ref = torch.softmax(s.masked_fill(~rm,float('-inf')),-1)@V
assert torch.equal(mask,rm)
assert torch.allclose(out,ref,atol=1e-12,rtol=1e-12)
''',
        },
        {
            "name": "sink-only history edge",
            "code": '''import torch
Q = torch.zeros(1,4,2); K = torch.randn(1,4,2)
V = torch.tensor([[[2.,0.],[0.,2.],[4.,4.],[8.,8.]]])
out, mask = {fn}(Q,K,V,sink_size=1,window_size=1)
assert torch.equal(mask[-1],torch.tensor([True,False,False,True]))
assert torch.allclose(out[0,-1],torch.tensor([5.,4.]))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
Q,K,V = [torch.randn(2,5,3,requires_grad=True) for _ in range(3)]
out, mask = {fn}(Q,K,V,softcap=4.0,sink_size=2,window_size=3)
out.square().sum().backward()
assert mask[0,0] and not mask[0,1]
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (Q,K,V))
''',
        },
    ],
    "solution": '''import torch
import math

def softcap_sink_attention(Q, K, V, softcap=50.0, sink_size=2, window_size=4):
    S = Q.shape[1]
    D = Q.shape[-1]
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(D)
    scores = softcap * torch.tanh(scores / softcap)
    query_pos = torch.arange(S, device=Q.device).unsqueeze(1)
    key_pos = torch.arange(S, device=Q.device).unsqueeze(0)
    mask = (key_pos <= query_pos) & ((key_pos < sink_size) | (key_pos > query_pos - window_size))
    attn = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
    out = attn @ V
    return out, mask
''',
    "demo": '''Q,K,V = [torch.randn(2,16,32) for _ in range(3)]
out, mask = softcap_sink_attention(Q,K,V,softcap=30.0,sink_size=2,window_size=8)
print(out.shape, mask[-1].sum().item())
''',
    "explanation": "The mask always includes a valid current-token key, so softmax has no all-masked row. Earlier non-sink keys fall out once they leave the rolling window.",
}
