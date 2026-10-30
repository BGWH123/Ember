"""Expected calibration error and Brier score task."""

TASK = {
    "title": "Expected Calibration Error (ECE) & Brier Score",
    "title_zh": "期望校准误差（ECE）与 Brier 分数",
    "difficulty": "Medium",
    "category": "评测与基准",
    "function_name": "calibration_metrics",
    "starter_code": '''import torch
import torch.nn.functional as F

def calibration_metrics(probs, targets, n_bins=10):
    pass
''',
    "hint": "Use top-1 confidence for ECE and one-hot targets for Brier score. Skip empty bins so their means are never evaluated.",
    "hint_zh": "ECE 使用 top-1 置信度，Brier 分数使用 one-hot 标签。跳过空桶，避免对其求均值。",
    "description_en": "Compute top-1 expected calibration error using equal-width confidence bins and the multiclass Brier score. Return both as scalar tensors.",
    "description_zh": "使用等宽置信度分桶计算 top-1 期望校准误差，并计算多分类 Brier 分数。返回两个标量张量。",
    "theory_en": "ECE compares average confidence with empirical accuracy inside each bin and weights the gap by bin frequency. Brier score measures the squared distance from the full probability vector to the one-hot outcome.",
    "theory_zh": "ECE 比较每个桶内的平均置信度与经验准确率，并按桶频率加权差距。Brier 分数衡量完整概率向量与 one-hot 结果之间的平方距离。",
    "diagram_en": "```mermaid\nflowchart LR\nP[probabilities] --> T[top-1 confidence]\nT --> B[confidence bins]\nB --> E[weighted calibration gaps]\nP --> R[squared one-hot error]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nP[概率] --> T[top-1 置信度]\nT --> B[置信度分桶]\nB --> E[加权校准差]\nP --> R[one-hot 平方误差]\n```",
    "tests": [
        {"name": "exact two-bin reference", "code": """import torch
probs = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
targets = torch.tensor([0, 1, 1, 0])
ece, brier = {fn}(probs, targets, n_bins=2)
expected_ece = 0.5 * abs(0.5 - 0.65) + 0.5 * abs(0.5 - 0.85)
one_hot = torch.nn.functional.one_hot(targets, 2).float()
expected_brier = ((probs - one_hot) ** 2).sum(1).mean()
assert ece.shape == () and brier.shape == ()
assert torch.allclose(ece, torch.tensor(expected_ece), atol=1e-7)
assert torch.allclose(brier, expected_brier)
"""},
        {"name": "perfect predictions", "code": """import torch
probs = torch.eye(3)
targets = torch.arange(3)
ece, brier = {fn}(probs, targets, n_bins=5)
assert ece.shape == () and brier.shape == ()
assert torch.equal(ece, torch.tensor(0.0))
assert torch.equal(brier, torch.tensor(0.0))
"""},
        {"name": "lower boundary belongs to first bin", "code": """import torch
probs = torch.zeros(2, 2)
targets = torch.tensor([0, 1])
ece, brier = {fn}(probs, targets, n_bins=2)
assert ece.shape == () and brier.shape == ()
assert torch.allclose(ece, torch.tensor(0.5))
assert torch.allclose(brier, torch.tensor(1.0))
"""},
        {"name": "gradient flow", "code": """import torch
raw = torch.randn(5, 3, requires_grad=True)
probs = raw.softmax(-1)
targets = torch.tensor([0, 1, 2, 1, 0])
ece, brier = {fn}(probs, targets, n_bins=4)
(ece + brier).backward()
assert ece.shape == () and brier.shape == ()
assert raw.grad is not None and raw.grad.shape == raw.shape
assert torch.isfinite(raw.grad).all()
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def calibration_metrics(probs, targets, n_bins=10):
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)
    ece = probs.new_zeros(())
    for m in range(n_bins):
        low = boundaries[m]
        high = boundaries[m + 1]
        if m == 0:
            mask = (conf >= 0.0) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        if mask.any():
            gap = torch.abs(acc[mask].mean() - conf[mask].mean())
            ece = ece + mask.float().mean() * gap
    one_hot = F.one_hot(targets, num_classes=probs.shape[-1]).to(dtype=probs.dtype)
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return ece, brier
''',
    "demo": '''probs = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
targets = torch.tensor([0, 0])
print(calibration_metrics(probs, targets))
''',
    "explanation": "The first bin explicitly includes zero and every other bin is open on the left and closed on the right, so each valid confidence belongs to exactly one bin. Empty bins contribute zero.",
}
