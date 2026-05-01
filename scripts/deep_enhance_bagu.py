import json
import re

# ─── Chapter-specific rewrite templates ────────────────────────────────

# Common high-quality answer templates for frequent question patterns
TEMPLATES = {
    # Model architecture comparisons
    "decoder": lambda text: _rewrite_decoder_comparison(text),
    "attention": lambda text: _rewrite_attention(text),
    "training": lambda text: _rewrite_training(text),
    "emergent": lambda text: _rewrite_emergent(text),
    "why_decoder": lambda text: _rewrite_why_decoder(text),
    # RLHF / PPO
    "rlhf": lambda text: _rewrite_rlhf(text),
    "ppo": lambda text: _rewrite_ppo(text),
    "reward_model": lambda text: _rewrite_reward_model(text),
    # Fine-tuning
    "lora": lambda text: _rewrite_lora(text),
    "sft": lambda text: _rewrite_sft(text),
    # Inference
    "kv_cache": lambda text: _rewrite_kv_cache(text),
    "quantization": lambda text: _rewrite_quantization(text),
}

def _rewrite_decoder_comparison(text):
    """Rewrite decoder architecture comparison answers."""
    return (
        "三种主流架构的核心差异在于 **Attention Mask** 的设计方式，这直接决定了模型的训练范式和应用场景。\n\n"
        "**1. Encoder-Decoder（如 T5、BART）**\n"
        "• 输入端采用**双向注意力**，输出端采用**因果注意力**\n"
        "• 优势：对输入文本的理解更充分，在翻译、摘要等**理解+生成**任务上表现稳定\n"
        "• 劣势：参数量大（编码器+解码器），长文本生成效率低，**涌现能力**较弱\n\n"
        "**2. Causal Decoder（如 GPT、LLaMA）**\n"
        "• 纯**自回归**结构，严格遵循从左到右的因果注意力\n"
        "• 优势：**预训练与下游任务完全一致**，零样本能力强，训练效率高，**涌现能力**显著\n"
        "• 代表：GPT 系列、LLaMA、Mistral、Qwen\n\n"
        "**3. Prefix Decoder（如 ChatGLM、U-PaLM）**\n"
        "• **Prefix 部分双向注意力**，生成部分因果注意力\n"
        "• 定位：介于前两者之间的折中方案\n"
        "• 劣势：训练效率低于 Causal Decoder，目前主流已转向 Decoder-only\n\n"
        "**面试要点**：当前大模型（2024-2025）几乎全部采用 **Causal Decoder-only** 架构，"
        "因为其在同等参数量下训练效率最高、零样本能力最强。"
    )

def _rewrite_attention(text):
    return text  # Placeholder - keep original if no specific rewrite needed

def _rewrite_training(text):
    return text

def _rewrite_emergent(text):
    return (
        "**涌现能力（Emergent Capabilities）**指模型在规模达到某个阈值后，突然展现出小模型不具备的复杂能力，"
        "如数学推理、代码生成、多步逻辑推导等。\n\n"
        "目前学界对涌现现象的成因主要有两种解释：\n\n"
        "**1. 任务评估指标的「陡峭性」**\n"
        "• 某些任务采用非线性评估指标（如精确匹配），导致能力曲线在宏观上呈现「阶跃」\n"
        "• 实际上模型在子任务上的进步可能是平滑的，只是最终指标放大了差异\n\n"
        "**2. 复杂任务的组合效应**\n"
        "• 假设任务 T 由 5 个子任务组成，每个子任务准确率从 40% → 60%\n"
        "• 宏观指标可能仅从 1% → 7%，呈现「涌现」假象\n"
        "• 最新研究（2024）倾向于认为：涌现并非真正的「相变」，而是评估方式的产物\n\n"
        "**面试要点**：涌现能力是大模型最迷人的特性之一，但也存在争议。"
        "回答时应展示你对这一话题的辩证思考。"
    )

def _rewrite_why_decoder(text):
    return (
        "当前主流大模型采用 **Decoder-only** 架构，核心原因可从**理论、实践、效率**三个维度理解：\n\n"
        "**1. 训练效率最优**\n"
        "• Causal Decoder 的预训练目标（Next Token Prediction）与下游应用完全一致\n"
        "• 无需像 Encoder-Decoder 那样在微调阶段适配不同任务格式\n"
        "• **所有 token 都参与损失计算**，训练信号更充分\n\n"
        "**2. 零样本与涌现能力**\n"
        "• Decoder-only 模型在没有任何标注数据的情况下，**zero-shot 表现最好**\n"
        "• 大规模自监督预训练能激发模型的**涌现能力**和**上下文学习**能力\n\n"
        "**3. 理论视角**\n"
        "• Encoder 的双向注意力可能导致**低秩问题**，削弱表达能力\n"
        "• 对于生成任务，引入双向注意力并无实质好处\n"
        "• Encoder-Decoder 的优势主要来自**多一倍的参数量**\n\n"
        "**结论**：在同等参数量、同等推理成本下，Decoder-only 是目前的最优解。"
    )

def _rewrite_rlhf(text):
    return text

def _rewrite_ppo(text):
    return text

def _rewrite_reward_model(text):
    return text

def _rewrite_lora(text):
    return text

def _rewrite_sft(text):
    return text

def _rewrite_kv_cache(text):
    return text

def _rewrite_quantization(text):
    return text

# ─── Pattern matching for auto-rewrite ─────────────────────────────────

def detect_pattern(title: str, content: str) -> str:
    """Detect question pattern for template matching."""
    title_lower = title.lower()
    content_lower = content.lower()

    patterns = [
        ("decoder", ["decoder", "encoder-decoder", "prefix decoder", "causal decoder",
                     "encoder-decoder区别", "三种架构", "模型体系"]),
        ("emergent", ["涌现", "emergent", "emergence", "为什么突然"]),
        ("why_decoder", ["为什么", "decoder only", "decoder-only", "为什么大部分",
                         "为何现在", "为什么大模型"]),
        ("rlhf", ["rlhf", "human feedback", "人类反馈"]),
        ("ppo", ["ppo", "proximal policy", "策略优化"]),
        ("reward_model", ["reward model", "奖励模型", "rm模型"]),
        ("lora", ["lora", "低秩适配"]),
        ("sft", ["sft", "supervised fine", "有监督微调"]),
        ("kv_cache", ["kv cache", "kv缓存", "key value cache"]),
        ("quantization", ["量化", "quantization", "int8", "int4"]),
        ("attention", ["attention", "注意力", "self-attention"]),
        ("training", ["训练目标", "training objective", "预训练目标", "怎么训练"]),
    ]

    for pattern_name, keywords in patterns:
        for kw in keywords:
            if kw in title_lower or kw in content_lower[:200]:
                return pattern_name
    return None


# ─── Content restructuring helpers ─────────────────────────────────────

def restructure_list_content(text: str) -> str:
    """Convert flat numbered lists into well-structured markdown."""
    if not text:
        return text

    lines = text.split('\n')
    result = []
    current_item = []
    item_number = 0
    in_list = False

    for line in lines:
        stripped = line.strip()
        # Match list item start: "1. xxx", "1、xxx", "• xxx", "- xxx"
        m = re.match(r'^(?:\d+[\.、\)\]]\s*|[-•*]\s+)(.+)$', stripped)

        if m and len(stripped) < 200:  # Likely a list item
            content = m.group(1)
            # Save previous item
            if current_item:
                result.append('\n'.join(current_item))
                result.append('')
            # Start new item
            item_number += 1
            current_item = [f"**{item_number}.** {content}"]
            in_list = True
        elif in_list and stripped and not re.match(r'^\d+[\.、]', stripped):
            # Continuation of current item
            current_item.append(stripped)
        else:
            # End of list
            if current_item:
                result.append('\n'.join(current_item))
                result.append('')
                current_item = []
            in_list = False
            if stripped:
                result.append(stripped)

    if current_item:
        result.append('\n'.join(current_item))

    return '\n'.join(result).strip()


def add_interview_prelude(content: str, title: str) -> str:
    """Add a brief interview-style opening for well-structured answers."""
    if len(content) < 150:
        return content

    # Don't add if already has a strong opening
    first_line = content.split('\n')[0] if content else ''
    if any(w in first_line for w in ['核心', '首先', '总的来说', '简而言之', '简单来说']):
        return content

    # Detect content type and add appropriate prelude
    if '区别' in title or '对比' in title or '差异' in title:
        prelude = "这个问题可以从**核心差异**和**应用场景**两个层面来回答：\n\n"
    elif '为什么' in title or '原因' in title or '为何' in title:
        prelude = "主要原因有以下几点：\n\n"
    elif '如何' in title or '怎么' in title or '方法' in title:
        prelude = "可以从以下几个维度来理解和实现：\n\n"
    elif '是什么' in title or '介绍' in title:
        prelude = "简单来说，可以从以下几个角度来理解：\n\n"
    else:
        return content

    return prelude + content


def fix_title_continuation(title: str, content: str) -> tuple:
    """Fix title that was truncated into content. Returns (title, content)."""
    if not title or not content:
        return title, content

    # Check common truncation patterns
    # Pattern 1: title ends with "和", "与", "及", "、", "-"
    truncated_endings = ['和', '与', '及', '、', '-', '—', '或']
    for ending in truncated_endings:
        if title.endswith(ending):
            # Try to find the completion in the first line of content
            lines = content.split('\n', 1)
            first_line = lines[0].strip()
            # Check if first line completes a natural phrase
            if '？' in first_line or '?' in first_line or '。' in first_line[:20]:
                # This first line is likely the title continuation
                completion = first_line.split('？')[0].split('?')[0].split('。')[0]
                new_title = title + completion
                new_content = '\n'.join(lines[1:]).lstrip('\n') if len(lines) > 1 else ''
                return new_title, new_content
            break

    # Pattern 2: title ends mid-word (e.g., "Encoder-")
    if title.endswith('-'):
        lines = content.split('\n', 1)
        first_line = lines[0].strip()
        # Check if first word could complete the title
        words = first_line.split()
        if words and len(words[0]) < 20:
            new_title = title.rstrip('-') + words[0]
            remaining = ' '.join(words[1:])
            new_content = remaining
            if len(lines) > 1:
                new_content += '\n' + '\n'.join(lines[1:])
            return new_title, new_content

    # Pattern 3: content starts with what looks like a question
    first_line = content.split('\n')[0] if content else ''
    if first_line.strip().endswith(('？', '?')) and len(first_line) < 50:
        # This is likely the rest of the title
        new_title = title + first_line.strip()
        lines = content.split('\n')
        new_content = '\n'.join(lines[1:]).lstrip('\n')
        return new_title, new_content

    return title, content


def enhance_paragraph(content: str) -> str:
    """Enhance paragraph-style content with better structure."""
    if not content:
        return content

    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        return content

    # If paragraphs are very short, they might be a list in disguise
    if all(len(p) < 100 for p in paragraphs):
        return '\n\n'.join(f"• {p}" for p in paragraphs)

    return content


# ─── Main Processing ───────────────────────────────────────────────────

def process_content(content: str, title: str) -> str:
    """Apply all enhancements to content."""
    if not content:
        return content

    # Try template rewrite first
    pattern = detect_pattern(title, content)
    if pattern and pattern in TEMPLATES:
        rewritten = TEMPLATES[pattern](content)
        if rewritten and rewritten != content:
            return rewritten

    # Otherwise, apply structural enhancements
    content = restructure_list_content(content)
    content = enhance_paragraph(content)
    content = add_interview_prelude(content, title)

    return content


def process_data(data):
    rewritten_count = 0
    for chapter in data:
        for section in chapter.get('sections', []):
            # Fix title
            original_title = section.get('title', '')
            original_content = section.get('content', '')
            fixed_title, fixed_content = fix_title_continuation(original_title, original_content)
            if fixed_title != original_title or fixed_content != original_content:
                section['title'] = fixed_title
                section['content'] = fixed_content

            # Enhance content
            new_content = process_content(section.get('content', ''), section['title'])
            if new_content != section.get('content', ''):
                section['content'] = new_content
                rewritten_count += 1

            # Process subsections
            for sub in section.get('subsections', []):
                sub_title = sub.get('title', '')
                sub_content = sub.get('content', '')
                ft, fc = fix_title_continuation(sub_title, sub_content)
                if ft != sub_title:
                    sub['title'] = ft
                if fc != sub_content:
                    sub['content'] = fc

                new_sub = process_content(sub.get('content', ''), sub['title'])
                if new_sub != sub.get('content', ''):
                    sub['content'] = new_sub

                for ss in sub.get('subsubsections', []):
                    ss_title = ss.get('title', '')
                    ss_content = ss.get('content', '')
                    ft2, fc2 = fix_title_continuation(ss_title, ss_content)
                    if ft2 != ss_title:
                        ss['title'] = ft2
                    if fc2 != ss_content:
                        ss['content'] = fc2

                    new_ss = process_content(ss.get('content', ''), ss['title'])
                    if new_ss != ss.get('content', ''):
                        ss['content'] = new_ss

    return data, rewritten_count


if __name__ == '__main__':
    input_path = r'd:\BGWH_Code\pyre-code-main\web\public\bagu-data.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data, count = process_data(data)

    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Deep enhanced {count} sections with template rewrites")
    print(f"Saved to {input_path}")
