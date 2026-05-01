"""Fix unescaped ASCII double quotes inside string literals."""
import re

path = r"d:\BGWH_Code\pyre-code-main\scripts\enhance_solutions.py"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_count = 0
new_lines = []
for i, line in enumerate(lines):
    stripped = line.lstrip()
    # Only check lines that are string literals in the dict: start with spaces + quote
    if stripped.startswith('"') and stripped.rstrip('\n').endswith('"'):
        # Count quotes at the beginning and end
        leading_spaces = line[:len(line) - len(line.lstrip())]
        content = stripped[1:-1]  # Remove surrounding quotes
        # Check if there are unescaped quotes in the middle
        if content.count('"') > 0:
            # Replace "..." pairs that look like Chinese quotation marks
            # Pattern: "word" where quotes are ASCII but content is Chinese
            new_content = re.sub(r'"([^"\n]{1,20})"', r'「\1」', content)
            if new_content != content:
                line = leading_spaces + '"' + new_content + '"\n'
                fixed_count += 1
                print(f"Fixed line {i+1}")
    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Fixed {fixed_count} lines")
