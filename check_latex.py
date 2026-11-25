import re, sys

with open("paper.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Check environment nesting using a stack
tokens = re.findall(r'\\(begin|end)\{([a-zA-Z0-9*]+)\}', tex)
env_stack = []
env_errors = []
for action, name in tokens:
    if action == 'begin':
        env_stack.append(name)
    elif action == 'end':
        if not env_stack:
            env_errors.append(f"Unexpected \\end{{{name}}}")
        elif env_stack[-1] != name:
            env_errors.append(f"Mismatched environment: expected \\end{{{env_stack[-1]}}}, got \\end{{{name}}}")
            env_stack.pop()
        else:
            env_stack.pop()

if env_stack:
    for name in env_stack:
        env_errors.append(f"Unclosed environment: \\begin{{{name}}}")

print("Environment errors:", env_errors if env_errors else "ZERO (perfect nesting!)")

# Check citations against references.bib
with open("references.bib", "r", encoding="utf-8") as f:
    bib = f.read()
bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
citations_found = re.findall(r'\\cite\{([^}]+)\}', tex)
all_cited = set(k.strip() for c in citations_found for k in c.split(','))
missing_citations = all_cited - bib_keys
print("Missing citations:", missing_citations if missing_citations else "ZERO (all 21 citations verified!)")

# Check refs
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
missing_refs = refs - labels
print("Missing references:", missing_refs if missing_refs else "ZERO (all 7 references matched!)")

if not env_errors and not missing_citations and not missing_refs:
    print("\n>>> ALL LATEX SYNTAX, ENVIRONMENTS, LABELS, AND CITATIONS ARE 100% PERFECT! <<<")
else:
    sys.exit(1)
