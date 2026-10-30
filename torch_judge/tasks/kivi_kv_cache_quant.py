"""KIVI asymmetric KV-cache quantization task."""

TASK = {
    "title": "KIVI Asymmetric KV Cache Quantization (Per-Channel K, Per-Token V)",
    "title_zh": "KIVI 非对称 KV Cache 量化（按通道 K / 按 Token V）",
    "difficulty": "Medium",
    "category": "推理优化",
    "function_name": "kivi_asymmetric_kv_quant",
    "starter_code": '''import torch

def kivi_asymmetric_kv_quant(K, V, n_bits=2, eps=1e-8):
    pass
''',
    "hint": "Reduce K over sequence dimension 1 and V over channel dimension 2. Use each min as the affine offset and quantize into levels 0 through 2^n_bits-1.",
    "hint_zh": "K 沿序列维 1 归约，V 沿通道维 2 归约；以各自最小值为仿射偏移，量化到 0 至 2^n_bits-1。",
    "description_en": "Implement KIVI's asymmetric low-bit KV-cache quantization with per-channel key parameters and per-token value parameters.",
    "description_zh": "实现 KIVI 非对称低比特 KV 缓存量化：键按通道设置参数，值按 token 设置参数。",
    "theory_en": "Key outliers are structured by feature channel across positions, whereas value outliers are better localized per token. Using different quantization axes reduces error at very low bit width.",
    "theory_zh": "键异常值通常沿位置共享通道结构，而值异常值更适合按 token 局部处理。采用不同量化轴可降低极低比特下的误差。",
    "diagram_en": "```text\nK(B,L,D) --min/max over L--> scale(B,1,D) -> quantize/dequantize\nV(B,L,D) --min/max over D--> scale(B,L,1) -> quantize/dequantize\n```",
    "diagram_zh": "```text\nK(B,L,D) --沿 L 求 min/max--> scale(B,1,D) -> 量化/反量化\nV(B,L,D) --沿 D 求 min/max--> scale(B,L,1) -> 量化/反量化\n```",
    "tests": [
        {
            "name": "cache and scale shapes",
            "code": '''import torch
K = torch.randn(2,5,4); V = torch.randn(2,5,4)
kd, vd, ks, vs = {fn}(K,V,n_bits=2)
assert kd.shape == vd.shape == (2,5,4)
assert ks.shape == (2,4) and vs.shape == (2,5)
''',
        },
        {
            "name": "exact asymmetric reference",
            "code": '''import torch
K = torch.tensor([[[-2.,1.],[0.,4.],[3.,2.]]],dtype=torch.double)
V = torch.tensor([[[-1.,2.],[3.,7.],[-4.,5.]]],dtype=torch.double)
kd,vd,ks,vs = {fn}(K,V,n_bits=3,eps=1e-12)
kmn,kmx = K.amin(1,True),K.amax(1,True); rks=(kmx-kmn).clamp(min=1e-12)/7
vk,vx = V.amin(2,True),V.amax(2,True); rvs=(vx-vk).clamp(min=1e-12)/7
rkd=torch.clamp(torch.round((K-kmn)/rks),0.,7.)*rks+kmn
rvd=torch.clamp(torch.round((V-vk)/rvs),0.,7.)*rvs+vk
assert torch.equal(kd,rkd) and torch.equal(vd,rvd)
assert torch.equal(ks,rks.squeeze(1)) and torch.equal(vs,rvs.squeeze(2))
''',
        },
        {
            "name": "constant cache edge case",
            "code": '''import torch
K = torch.full((2,3,4),2.5); V = torch.full((2,3,4),-1.5)
kd,vd,ks,vs = {fn}(K,V,n_bits=2,eps=1e-6)
assert torch.equal(kd,K) and torch.equal(vd,V)
assert torch.allclose(ks,torch.full((2,4),1e-6/3))
assert torch.allclose(vs,torch.full((2,3),1e-6/3))
''',
        },
        {
            "name": "gradient through dequantization",
            "code": '''import torch
K = torch.randn(2,5,3,requires_grad=True); V = torch.randn(2,5,3,requires_grad=True)
kd,vd,ks,vs = {fn}(K,V,n_bits=4)
(kd.square().sum()+vd.square().sum()+ks.sum()+vs.sum()).backward()
assert kd.requires_grad and vd.requires_grad
assert K.grad is not None and V.grad is not None
assert torch.isfinite(K.grad).all() and torch.isfinite(V.grad).all()
''',
        },
    ],
    "solution": '''import torch

def kivi_asymmetric_kv_quant(K, V, n_bits=2, eps=1e-8):
    q_max = (2 ** n_bits) - 1
    k_min = K.amin(dim=1, keepdim=True)
    k_max = K.amax(dim=1, keepdim=True)
    k_scale = (k_max - k_min).clamp(min=eps) / q_max
    K_q = torch.clamp(torch.round((K - k_min) / k_scale), 0.0, float(q_max))
    K_deq = K_q * k_scale + k_min
    v_min = V.amin(dim=2, keepdim=True)
    v_max = V.amax(dim=2, keepdim=True)
    v_scale = (v_max - v_min).clamp(min=eps) / q_max
    V_q = torch.clamp(torch.round((V - v_min) / v_scale), 0.0, float(q_max))
    V_deq = V_q * v_scale + v_min
    return K_deq, V_deq, k_scale.squeeze(1), v_scale.squeeze(2)
''',
    "demo": '''K,V = torch.randn(2,1024,64),torch.randn(2,1024,64)
K_deq,V_deq,k_scale,v_scale = kivi_asymmetric_kv_quant(K,V,n_bits=2)
print(K_deq.shape,V_deq.shape,k_scale.shape,v_scale.shape)
''',
    "explanation": "The affine ranges follow KIVI's distinct axes. Returned scales omit only the singleton reduction dimension, making their per-channel and per-token layouts explicit.",
}
