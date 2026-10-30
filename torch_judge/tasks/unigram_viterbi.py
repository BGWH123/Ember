"""Unigram Viterbi segmentation task."""

TASK = {
    "title": "Unigram LM Viterbi Subword Segmentation",
    "title_zh": "Unigram 维特比最优子词切分",
    "difficulty": "Hard",
    "category": "分词",
    "function_name": "unigram_viterbi",
    "starter_code": '''def unigram_viterbi(text, token_log_probs, unk_log_prob=-10.0):
    pass
''',
    "hint": "Let each character boundary be a DP state. Store the best predecessor for every reachable end position, then reverse the backtracked pieces.",
    "hint_zh": "将每个字符边界作为 DP 状态。为每个可达终点保存最佳前驱，最后回溯并反转子词。",
    "description_en": "Use Viterbi dynamic programming to find the maximum-log-probability unigram segmentation. Permit an unknown single-character fallback only when no vocabulary transition reaches that end position.",
    "description_zh": "使用维特比动态规划寻找对数概率最大的 Unigram 切分。仅当没有词表转移到达当前终点时，才允许未知单字符回退。",
    "theory_en": "Log probabilities turn a product of independent piece probabilities into an additive path score. The DP keeps only the best path to each boundary, giving an exact optimum over all segmentations.",
    "theory_zh": "对数概率将独立子词概率的乘积变成可加路径分数。DP 只保留到每个边界的最佳路径，从而精确得到所有切分中的最优解。",
    "diagram_en": "```mermaid\nflowchart LR\nZ[boundary 0] -->|piece score| I[boundary i]\nI -->|piece score| N[boundary N]\nN --> B[backtrack best path]\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nZ[边界 0] -->|子词分数| I[边界 i]\nI -->|子词分数| N[边界 N]\nN --> B[回溯最佳路径]\n```",
    "tests": [
        {"name": "finds global optimum", "code": """import torch
pieces, score = {fn}('abcd', {'a': -1.0, 'ab': -0.2, 'bc': -0.1, 'cd': -0.2, 'd': -1.0})
assert pieces == ['ab', 'cd']
assert abs(score - (-0.4)) < 1e-12
assert torch.tensor([score]).shape == (1,)
"""},
        {"name": "unknown fallback", "code": """import torch
pieces, score = {fn}('axb', {'a': -0.1, 'b': -0.2}, unk_log_prob=-3.0)
assert pieces == ['a', 'x', 'b']
assert abs(score + 3.3) < 1e-12
assert torch.isfinite(torch.tensor(score))
"""},
        {"name": "empty text", "code": """import torch
pieces, score = {fn}('', {'x': -1.0})
assert pieces == []
assert score == 0.0
assert torch.tensor(pieces).shape == (0,)
"""},
        {"name": "strict tie keeps first start", "code": """import torch
pieces, score = {fn}('aa', {'a': -0.5, 'aa': -1.0})
assert pieces == ['aa']
assert score == -1.0
assert torch.tensor(score).shape == ()
"""},
    ],
    "solution": '''def unigram_viterbi(text, token_log_probs, unk_log_prob=-10.0):
    n = len(text)
    best_score = [float('-inf')] * (n + 1)
    backptr = [-1] * (n + 1)
    best_score[0] = 0.0
    for end in range(1, n + 1):
        for start in range(end):
            piece = text[start:end]
            if piece in token_log_probs:
                score = best_score[start] + token_log_probs[piece]
                if score > best_score[end]:
                    best_score[end] = score
                    backptr[end] = start
        char = text[end - 1:end]
        if char not in token_log_probs and best_score[end] == float('-inf'):
            best_score[end] = best_score[end - 1] + unk_log_prob
            backptr[end] = end - 1
    pieces = []
    end = n
    while end > 0:
        start = backptr[end]
        pieces.append(text[start:end])
        end = start
    pieces.reverse()
    return pieces, float(best_score[n])
''',
    "demo": '''vocab = {'un': -0.2, 'i': -1.0, 'gram': -0.3, 'unigram': -1.0}
print(unigram_viterbi('unigram', vocab))
''',
    "explanation": "The strict `>` comparison implements deterministic tie handling. Backpointers identify the selected substring boundaries, while unknown characters remain literal pieces in the recovered list.",
}
