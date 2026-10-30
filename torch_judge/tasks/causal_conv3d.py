"""Temporally causal 3D convolution task."""

TASK = {
    "title": "Video VAE Temporal Causal 3D Convolution",
    "title_zh": "视频 VAE 时间因果三维卷积（Causal Conv3D）",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "causal_conv3d",
    "starter_code": '''def causal_conv3d(x, weight, bias=None, pad_mode='replicate'):
    pass
''',
    "hint": "Prepend k_t-1 frames (copies of the first frame or zeros), then apply symmetric zero padding only in height and width before conv3d with no internal padding.",
    "hint_zh": "在时间轴前置 k_t-1 帧（首帧副本或零），再仅对高宽做对称零填充，最后调用无内部填充的 conv3d。",
    "description_en": "Apply a stride-one 3D convolution to video x while padding only the past side of time. Replicate or zero temporal padding is selectable; spatial padding is symmetric and zero-valued.",
    "description_zh": "对视频 x 应用步长为 1 的三维卷积，并仅填充时间轴过去一侧。时间填充可选择复制或零值，空间填充为对称零填充。",
    "theory_en": "Left-only temporal padding aligns each kernel window with the current and earlier frames, preventing future leakage. Replicating the first frame provides a steady boundary condition often used by video VAEs.",
    "theory_zh": "仅在时间轴左侧填充，使每个卷积窗口只对齐当前及更早帧，从而防止未来信息泄漏。复制首帧提供视频 VAE 常用的稳定边界条件。",
    "diagram_en": "```text\n[first ... first | frame0 frame1 ...] -> causal temporal window\n                 spatial zero pad -> conv3d -> T preserved\n```",
    "diagram_zh": "```text\n[首帧 ... 首帧 | 帧0 帧1 ...] -> 因果时间窗口\n                 空间零填充 -> conv3d -> 保持 T\n```",
    "tests": [
        {
            "name": "output shape",
            "code": '''import torch
x=torch.randn(2,3,4,7,9); w=torch.randn(5,3,3,3,3); b=torch.randn(5)
y={fn}(x,w,b)
assert y.shape==(2,5,4,7,9) and torch.isfinite(y).all()
''',
        },
        {
            "name": "exact padding reference",
            "code": '''import torch
import torch.nn.functional as F
x=torch.arange(1.,1+1*1*3*2*2).reshape(1,1,3,2,2); w=torch.randn(1,1,2,3,3); b=torch.tensor([.2])
y={fn}(x,w,b,'replicate')
xp=torch.cat((x[:,:,:1],x),2); xp=F.pad(xp,(1,1,1,1,0,0))
ref=F.conv3d(xp,w,b,stride=1,padding=0)
assert torch.equal(y,ref)
''',
        },
        {
            "name": "zero padding and temporal causality",
            "code": '''import torch
x=torch.randn(1,1,5,3,3); changed=x.clone(); changed[:,:,3:]+=100
w=torch.ones(1,1,3,1,1)
y={fn}(x,w,pad_mode='zero'); yc={fn}(changed,w,pad_mode='zero')
assert torch.equal(y[:,:,:3],yc[:,:,:3])
assert torch.allclose(y[:,:,0],x[:,:,0])
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x=torch.randn(2,2,4,5,5,requires_grad=True); w=torch.randn(3,2,3,3,3,requires_grad=True)
b=torch.randn(3,requires_grad=True); y={fn}(x,w,b,'replicate'); y.square().mean().backward()
assert x.grad is not None and w.grad is not None and b.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
''',
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F

def causal_conv3d(x, weight, bias=None, pad_mode='replicate'):
    k_t, k_h, k_w = weight.shape[-3:]
    temporal_pad = k_t - 1
    if pad_mode == 'replicate':
        prefix = x[:, :, :1].expand(-1, -1, temporal_pad, -1, -1)
        x_padded = torch.cat((prefix, x), dim=2)
        x_padded = F.pad(
            x_padded, (k_w // 2, k_w // 2, k_h // 2, k_h // 2, 0, 0)
        )
    elif pad_mode == 'zero':
        x_padded = F.pad(
            x,
            (k_w // 2, k_w // 2, k_h // 2, k_h // 2, temporal_pad, 0),
        )
    else:
        raise ValueError("pad_mode must be 'replicate' or 'zero'")
    return F.conv3d(x_padded, weight, bias, stride=1, padding=0)
''',
    "demo": '''video = torch.randn(2, 3, 8, 32, 32)
kernel = torch.randn(16, 3, 3, 3, 3)
output = causal_conv3d(video, kernel)
print(output.shape)
''',
    "explanation": "Temporal and spatial padding are performed separately so replicate mode affects only time; height and width always receive zeros. No future frame enters any output window.",
}
