import json
import re
import os

# ─── Tech Terms Dictionary (for auto-bold) ──────────────────────────────

TECH_TERMS = [
    # Architecture
    "Transformer", "Self-Attention", "Multi-Head Attention", "MHA",
    "Feed-Forward Network", "FFN", "MLP", "Residual Connection",
    "Layer Normalization", "LayerNorm", "RMSNorm", "Batch Normalization",
    "Encoder", "Decoder", "Encoder-Decoder", "Decoder-only",
    "Causal Decoder", "Prefix Decoder", "Masked Language Model",
    "Autoregressive", "Bidirectional",
    # Models
    "GPT", "GPT-3", "GPT-3.5", "GPT-4", "GPT-4o", "ChatGPT",
    "LLaMA", "LLaMA 2", "LLaMA 3", "Alpaca", "Vicuna",
    "BERT", "RoBERTa", "T5", "BART", "ChatGLM", "GLM",
    "PaLM", "PaLM 2", "Gemini", "Claude", "Qwen", "Baichuan",
    "Mistral", "Mixtral", "MOE", "Mixture of Experts",
    # Attention variants
    "FlashAttention", "FlashAttention-2", "FlashAttention-3",
    "Multi-Query Attention", "MQA", "Grouped-Query Attention", "GQA",
    "Sliding Window Attention", "Sparse Attention",
    "KV Cache", "Key-Value Cache",
    # Positional Encoding
    "Positional Encoding", "RoPE", "Rotary Position Embedding",
    "ALiBi", "Relative Position Embedding", "Learned Position Embedding",
    # Training
    "Pre-training", "Fine-tuning", "Supervised Fine-Tuning", "SFT",
    "Instruction Fine-Tuning", "IFT", "Continual Pre-training",
    "Full Fine-tuning", "Parameter-Efficient Fine-Tuning", "PEFT",
    "LoRA", "QLoRA", "Adapter", "Prefix Tuning", "Prompt Tuning",
    "P-Tuning", "IA3", "DoRA",
    # Optimization
    "Adam", "AdamW", "SGD", "Momentum", "Nesterov",
    "Learning Rate Scheduling", "Warmup", "Cosine Decay",
    "Gradient Clipping", "Gradient Accumulation",
    "Mixed Precision Training", "FP16", "BF16", "FP8",
    # Alignment
    "RLHF", "Reinforcement Learning from Human Feedback",
    "PPO", "Proximal Policy Optimization",
    "DPO", "Direct Preference Optimization",
    "KTO", "IPO", "ORPO",
    "Reward Model", "RM",
    # Data & Evaluation
    "Prompt Engineering", "Chain-of-Thought", "CoT",
    "Few-Shot Learning", "Zero-Shot Learning", "In-Context Learning",
    "RAG", "Retrieval-Augmented Generation",
    "Agent", "Tool Use", "Function Calling",
    "Hallucination", "Alignment", "Safety",
    "Perplexity", "BLEU", "ROUGE", "BERTScore",
    "MMLU", "GSM8K", "HumanEval", "MBPP",
    # Efficiency
    "Quantization", "INT8", "INT4", "AWQ", "GPTQ", "GGUF",
    "Pruning", "Knowledge Distillation", "Distillation",
    "Speculative Decoding", "Lookahead Decoding",
    "Tensor Parallelism", "Pipeline Parallelism", "Data Parallelism",
    "ZeRO", "FSDP", "DeepSpeed", "Megatron-LM",
    # Inference
    "Temperature", "Top-p", "Top-k", "Nucleus Sampling",
    "Beam Search", "Greedy Decoding",
    "Context Window", "Long Context",
    # Chinese terms
    "大语言模型", "大模型", "预训练", "微调", "对齐",
    "注意力机制", "多头注意力", "自注意力",
    "梯度消失", "梯度爆炸", "过拟合", "欠拟合",
    "涌现能力", "上下文学习", "思维链",
    "检索增强生成", "幻觉", "对齐税",
    "量化", "剪枝", "知识蒸馏", "推理优化",
]

# Sort by length descending to avoid partial matches
TECH_TERMS.sort(key=len, reverse=True)

# ─── Outdated info corrections ──────────────────────────────────────────

OUTDATED_CORRECTIONS = [
    (r"GPT-3\.5.*?1750?\s*亿", "GPT-3.5（约 20B 参数，不同版本有差异）"),
    (r"ChatGPT.*?1750?\s*亿", "ChatGPT 基于 GPT-3.5（约 20B 参数）"),
    (r"175B.*?ChatGPT", "GPT-3 为 175B，ChatGPT/GPT-3.5 约 20B"),
    (r"GPT-3\.5.*?175B", "GPT-3.5 约 20B 参数（GPT-3 为 175B）"),
    (r"训练.*?GPT-3.*?30万.*?美元", "大模型训练成本极高（GPT-3 约数百万美元级别）"),
    (r"284吨.*?二氧化碳", "大模型训练碳排放较高，但具体数字因估算方法而异"),
]


# ─── Helper Functions ───────────────────────────────────────────────────

def clean_whitespace(text: str) -> str:
    """Clean excessive whitespace while preserving paragraph structure."""
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Replace multiple spaces with single space (but not newlines)
    lines = text.split('\n')
    lines = [re.sub(r' +', ' ', line).strip() for line in lines]
    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    # Merge consecutive empty lines to max 1
    result = []
    prev_empty = False
    for line in lines:
        is_empty = not line
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty
    return '\n'.join(result)


def normalize_numbered_list(text: str) -> str:
    """Normalize numbered lists to consistent format."""
    # Pattern: standalone number followed by space or Chinese punctuation
    # But be careful not to match things like "GPT-3" or "1亿"
    lines = text.split('\n')
    result = []
    list_counter = 0
    in_list = False

    for line in lines:
        # Match patterns like "1. xxx", "1、xxx", "(1) xxx", "① xxx"
        m = re.match(r'^(\d+)[\.、\)\]\s]+(.+)$', line)
        if m and len(m.group(1)) <= 2:  # Only small numbers (not 2024)
            num = int(m.group(1))
            content = m.group(2).strip()
            if num == 1 or (in_list and num == list_counter + 1):
                if not in_list:
                    in_list = True
                    list_counter = 0
                list_counter = num
                result.append(f"{num}. {content}")
                continue
            elif num == 1 and in_list:
                # Reset list
                list_counter = 1
                result.append(f"{num}. {content}")
                continue

        # Check if this line looks like a list item continuation
        if in_list and line.strip() and not line.strip()[0].isdigit():
            # It might be continuation of previous item
            pass
        elif line.strip() and not re.match(r'^\d+[\.、]', line.strip()):
            in_list = False
            list_counter = 0

        result.append(line)

    return '\n'.join(result)


def auto_bold_terms(text: str) -> str:
    """Bold technical terms in text."""
    # Don't bold terms inside existing bold markers or code
    # Process line by line to be safer
    result = text
    for term in TECH_TERMS:
        # Skip if already bolded or too short
        if len(term) < 2:
            continue
        # Match whole word/phrase, case-sensitive for acronyms, but allow lowercase for common terms
        pattern = re.compile(
            r'(?<![\*\w])(' + re.escape(term) + r')(?![\w\*])',
            re.IGNORECASE if term.islower() or term.isupper() and len(term) <= 4 else 0
        )
        # Apply replacement, but avoid double-bolding
        def repl(m):
            # Check if surrounded by ** already
            start = max(0, m.start() - 2)
            end = min(len(result), m.end() + 2)
            surrounding = result[start:end]
            if '**' in surrounding:
                return m.group(1)
            return f"**{m.group(1)}**"
        result = pattern.sub(repl, result)
    return result


def fix_truncated_title_in_content(content: str, title: str) -> str:
    """Remove title continuation that leaked into content."""
    # Common pattern: title was truncated, rest is in content
    # e.g., title="2 prefix Decoder...和 Encoder-", content="Decoder 区别是什么？\n..."
    if not title or not content:
        return content

    # Check if content starts with the likely continuation of title
    # Try matching last few words of title with start of content
    title_words = title.split()
    for n in range(min(5, len(title_words)), 0, -1):
        suffix = ' '.join(title_words[-n:])
        # Remove trailing punctuation for matching
        suffix_clean = re.sub(r'[\s\-]+$', '', suffix)
        if len(suffix_clean) < 3:
            continue
        # Check if content starts with this suffix or its completion
        content_start = content[:len(suffix_clean) + 20]
        if suffix_clean.lower() in content_start.lower():
            # Find where the title continuation ends (usually a question mark or first newline)
            lines = content.split('\n', 1)
            first_line = lines[0]
            # If first line looks like a title/question completion
            if '？' in first_line or '?' in first_line or '。' in first_line[:30]:
                # Remove the first line if it completes the title
                if len(lines) > 1:
                    return lines[1].lstrip('\n')
                else:
                    return ''
    return content


def add_interview_structure(content: str, title: str) -> str:
    """Add interview-style structure to answers."""
    if not content or len(content) < 50:
        return content

    lines = content.split('\n')
    result_lines = []

    # Check if already well-structured (has clear list markers)
    has_structure = any(re.match(r'^\d+\.', line) for line in lines)

    if not has_structure and len(lines) > 1:
        # Try to add structure by splitting on obvious breaks
        pass  # Don't force structure if it's a flowing paragraph

    # Add opening statement for longer answers without one
    first_line = lines[0] if lines else ''
    if len(content) > 200 and not has_structure:
        # If it's a long paragraph without structure, keep it as-is
        # but ensure it's well-formatted
        pass

    return content


def apply_outdated_corrections(text: str) -> str:
    """Fix known outdated information."""
    for pattern, replacement in OUTDATED_CORRECTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def enhance_content(content: str, title: str) -> str:
    """Main enhancement pipeline for a single answer."""
    if not content:
        return content

    # Step 1: Fix title leaking into content
    content = fix_truncated_title_in_content(content, title)

    # Step 2: Clean whitespace
    content = clean_whitespace(content)

    # Step 3: Apply outdated corrections
    content = apply_outdated_corrections(content)

    # Step 4: Normalize numbered lists
    content = normalize_numbered_list(content)

    # Step 5: Auto-bold technical terms
    content = auto_bold_terms(content)

    # Step 6: Add interview structure
    content = add_interview_structure(content, title)

    # Step 7: Final cleanup
    content = clean_whitespace(content)

    return content


def enhance_title(title: str) -> str:
    """Clean up question titles."""
    if not title:
        return title
    # Remove leading numbers like "1 ", "2 "
    title = re.sub(r'^\d+\s+', '', title)
    # Remove extra spaces
    title = re.sub(r' +', ' ', title).strip()
    # Fix common truncation patterns
    if title.endswith(('和', '与', '及', '、', '-', '—')):
        # Title was cut off, try to complete from context or leave as-is
        pass
    return title


# ─── Main Processing ────────────────────────────────────────────────────

def process_data(data):
    """Process all chapters and sections."""
    total_sections = 0
    total_subsections = 0

    for chapter in data:
        for section in chapter.get('sections', []):
            total_sections += 1

            # Enhance title
            section['title'] = enhance_title(section['title'])

            # Enhance content
            section['content'] = enhance_content(section.get('content', ''), section['title'])

            # Process subsections
            for sub in section.get('subsections', []):
                total_subsections += 1
                sub['title'] = enhance_title(sub.get('title', ''))
                sub['content'] = enhance_content(sub.get('content', ''), sub['title'])

                # Process sub-subsections
                for ss in sub.get('subsubsections', []):
                    ss['title'] = enhance_title(ss.get('title', ''))
                    ss['content'] = enhance_content(ss.get('content', ''), ss['title'])

    return data, total_sections, total_subsections


if __name__ == '__main__':
    input_path = r'd:\BGWH_Code\pyre-code-main\web\public\bagu-data.json'
    output_path = r'd:\BGWH_Code\pyre-code-main\web\public\bagu-data.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} chapters")

    data, total_sec, total_sub = process_data(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Enhanced {total_sec} sections, {total_sub} subsections")
    print(f"Saved to {output_path}")
