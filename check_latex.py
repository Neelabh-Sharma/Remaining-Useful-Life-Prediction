import re
import os

def check_latex_document(tex_file="paper.tex", bib_file="references.bib"):
    print(f"Validating {tex_file} and {bib_file}...")
    errors = []
    warnings = []

    if not os.path.exists(tex_file):
        errors.append(f"Missing {tex_file}")
        return errors, warnings
    if not os.path.exists(bib_file):
        warnings.append(f"Missing {bib_file}")

    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip comments properly (unescaped % to end of line)
    clean_code = re.sub(r'(?<!\\)%.*', '', content)
    
    # 1. Check brace balance (ignore \{ and \})
    code_no_escaped_braces = re.sub(r'\\\{|\\\}', '', clean_code)
    open_braces = code_no_escaped_braces.count('{')
    close_braces = code_no_escaped_braces.count('}')
    if open_braces != close_braces:
        errors.append(f"Mismatched braces: {open_braces} open '{{' vs {close_braces} close '}}'")
    else:
        print(f"Brace balance check passed: {open_braces} pairs.")

    # 2. Check environment balance
    begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', clean_code)
    ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', clean_code)
    if begins != ends:
        from collections import Counter
        b_cnt = Counter(begins)
        e_cnt = Counter(ends)
        for env, cnt in b_cnt.items():
            if cnt != e_cnt.get(env, 0):
                errors.append(f"Environment mismatch for '{env}': {cnt} begins vs {e_cnt.get(env, 0)} ends")
        for env, cnt in e_cnt.items():
            if env not in b_cnt:
                errors.append(f"Extra end for environment '{env}': {cnt} ends with 0 begins")
    else:
        print(f"Environment balance check passed: {len(begins)} environments properly closed.")

    # 3. Check figure files
    graphics = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', clean_code)
    for g in graphics:
        if not os.path.exists(g):
            errors.append(f"Image file not found: {g}")
        else:
            print(f"Verified image asset: {g}")

    # 4. Check citations
    if os.path.exists(bib_file):
        with open(bib_file, "r", encoding="utf-8") as bf:
            bib_content = bf.read()
        bib_keys = set(re.findall(r'@\w+\{\s*([^,]+),', bib_content))
        tex_cites = set(re.findall(r'\\cite\{([^}]+)\}', clean_code))
        actual_cites = set()
        for c in tex_cites:
            for item in c.split(','):
                actual_cites.add(item.strip())
        missing_cites = actual_cites - bib_keys
        if missing_cites:
            warnings.append(f"Citation keys not found in {bib_file}: {missing_cites}")
        else:
            print(f"Verified all citations in bib: {actual_cites}")

    # 5. Check placeholder citations [CITE: ...]
    placeholder_cites = re.findall(r'\[CITE:[^\]]+\]', clean_code)
    if placeholder_cites:
        print(f"Found {len(placeholder_cites)} [CITE: ...] placeholders as requested: {placeholder_cites}")

    # 6. Check math delimiters (ignore \$)
    single_dollars = len(re.findall(r'(?<!\\)\$', clean_code))
    if single_dollars % 2 != 0:
        errors.append(f"Odd number of single dollar math delimiters: {single_dollars}")
    else:
        print(f"Math delimiter check passed: {single_dollars // 2} inline math expressions.")

    return errors, warnings

if __name__ == "__main__":
    errs, warns = check_latex_document()
    if errs:
        print("\nERRORS FOUND:")
        for e in errs:
            print(f" - {e}")
    else:
        print("\nALL CRITICAL CHECKS PASSED WITH 0 ERRORS!")
    if warns:
        print("\nWARNINGS:")
        for w in warns:
            print(f" - {w}")
