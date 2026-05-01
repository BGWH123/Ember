"""Audit all task files for case-sensitivity issues, categories, and quality."""

import ast
import re
import json
from pathlib import Path

TASKS_DIR = Path(r"d:\BGWH_Code\pyre-code-main\torch_judge\tasks")
PROBLEMS_JSON = Path(r"d:\BGWH_Code\pyre-code-main\web\src\lib\problems.json")

# Extract hardcoded attribute references from test code
def extract_attr_refs(code: str) -> list[str]:
    """Find patterns like obj.W_q, obj.w_q, self.W_q, etc."""
    # Match: word.word_pattern where second part looks like a weight attr
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.(W_[qkvok]|w_[qkvok]|weight|bias|[a-zA-Z_][a-zA-Z0-9_]*)\b'
    return list(set(m[0] for m in re.finditer(pattern, code)))

# Categorization mapping based on task content
def guess_category(task: dict, task_id: str) -> str:
    title = task.get("title", "").lower()
    desc = task.get("description_en", "").lower()
    fn = task.get("function_name", "").lower()
    
    # Attention mechanisms
    if any(k in title or k in desc for k in ["attention", "mha", "gqa", "mla", "alibi", "rope", "kv cache", "flash", "ring", "paged", "sliding", "speculative"]):
        if "kv cache" in title.lower() or "kv_cache" in fn:
            return "推理优化"
        if "flash" in title.lower() or "ring" in title.lower() or "paged" in title.lower():
            return "推理优化"
        if "speculative" in title.lower():
            return "推理优化"
        return "注意力机制"
    
    # Optimizers & LR
    if any(k in title for k in ["adam", "optimizer", "gradient clip", "gradient accum", "cosine lr", "lr scheduler"]):
        return "优化器与学习率"
    
    # Normalization
    if any(k in title for k in ["batchnorm", "layernorm", "rmsnorm", "normalization", "adaln"]):
        return "归一化"
    
    # Activation
    if any(k in title for k in ["relu", "gelu", "swiglu", "softmax", "activation"]):
        return "激活函数"
    
    # Embedding / Positional Encoding
    if any(k in title for k in ["embedding", "positional", "rope", "ntk"]):
        return "位置编码与嵌入"
    
    # Loss functions
    if any(k in title for k in ["loss", "cross entropy", "focal", "contrastive", "dpo", "ppo", "grpo", "gae"]):
        return "损失函数"
    
    # Regularization
    if any(k in title for k in ["dropout", "label smoothing"]):
        return "正则化"
    
    # Diffusion / Flow
    if any(k in title for k in ["ddim", "flow matching", "noise schedule", "diffusion"]):
        return "扩散与流模型"
    
    # Quantization
    if any(k in title for k in ["quantiz", "int8", "qlora"]):
        return "量化"
    
    # Efficient training
    if any(k in title for k in ["fsdp", "mixed precision", "tensor parallel", "activation checkpoint", "gradient accumulation"]):
        return "高效训练"
    
    # GNN
    if any(k in title for k in ["gat", "gcn", "gin", "graph", "gnn", "mpnn", "link prediction", "readout"]):
        return "图神经网络"
    
    # Sampling / Decoding
    if any(k in title for k in ["sampling", "beam search", "top-k", "mcts", "decoding"]):
        return "采样与解码"
    
    # Transformer blocks
    if any(k in title for k in ["gpt2", "vit", "transformer block", "mlp", "block"]):
        return "Transformer组件"
    
    # LoRA / PEFT
    if "lora" in title.lower():
        return "参数高效微调"
    
    # Tokenization
    if "bpe" in title.lower():
        return "分词"
    
    # Linear / Basic ops
    if any(k in title for k in ["linear regression", "conv2d", "depthwise", "max pool", "weight init", "linear layer"]):
        return "基础网络组件"
    
    # MoE
    if "moe" in title.lower():
        return "混合专家模型"
    
    # Mamba
    if "mamba" in title.lower():
        return "状态空间模型"
    
    # Multi-token prediction
    if "multi token" in title.lower():
        return "训练技巧"
    
    # Reward model
    if "reward" in title.lower():
        return "强化学习"
    
    return "其他"


def main():
    results = []
    attr_issues = []
    
    for fpath in sorted(TASKS_DIR.glob("*.py")):
        task_id = fpath.stem
        if task_id.startswith("_"):
            continue
        
        # Read and execute safely to get TASK dict
        source = fpath.read_text(encoding="utf-8")
        try:
            mod = ast.parse(source)
            # Find TASK assignment
            task = None
            for node in mod.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "TASK":
                            # Can't eval safely, use regex extraction
                            break
            
            # Use exec in restricted way
            namespace = {"__builtins__": {}}
            exec(compile(mod, str(fpath), "exec"), namespace)
            task = namespace.get("TASK")
            if task is None:
                continue
        except Exception as e:
            print(f"  ERROR parsing {task_id}: {e}")
            continue
        
        tests = task.get("tests", [])
        fn_name = task.get("function_name", "")
        
        # Check for hardcoded attribute references in tests
        all_refs = []
        for test in tests:
            code = test.get("code", "")
            refs = extract_attr_refs(code)
            all_refs.extend(refs)
        
        # Check for W_q/w_q style attributes
        capital_attrs = [r for r in all_refs if ".W_" in r]
        lower_attrs = [r for r in all_refs if ".w_" in r]
        
        category = guess_category(task, task_id)
        
        has_solution_comment = False
        sol = task.get("solution", "")
        if sol:
            has_solution_comment = "#" in sol or '"""' in sol or "'''" in sol
        
        results.append({
            "id": task_id,
            "title": task.get("title", ""),
            "fn_name": fn_name,
            "difficulty": task.get("difficulty", ""),
            "category": category,
            "num_tests": len(tests),
            "capital_attr_refs": list(set(capital_attrs)),
            "lower_attr_refs": list(set(lower_attrs)),
            "has_solution_comment": has_solution_comment,
        })
        
        if capital_attrs:
            attr_issues.append({
                "id": task_id,
                "attrs": list(set(capital_attrs)),
            })
    
    # Print summary
    print(f"Total tasks: {len(results)}")
    print(f"Tasks with capital attr refs in tests: {len(attr_issues)}")
    print()
    
    # Category distribution
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    print("=== Category Distribution ===")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print()
    
    # Attr issues detail
    print("=== Case-Sensitive Attribute References ===")
    for issue in attr_issues:
        print(f"  {issue['id']}: {issue['attrs']}")
    print()
    
    # Solutions without comments
    no_comment = [r["id"] for r in results if not r["has_solution_comment"]]
    print(f"=== Solutions without comments: {len(no_comment)} ===")
    print(f"  {', '.join(no_comment[:20])}{'...' if len(no_comment) > 20 else ''}")
    
    # Save audit report
    report_path = Path(r"d:\BGWH_Code\pyre-code-main\scripts\audit_report.json")
    report_path.write_text(json.dumps({
        "tasks": results,
        "attr_issues": attr_issues,
        "no_comment": no_comment,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
