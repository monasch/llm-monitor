import argparse
import re
import sys
from pathlib import Path


def extract_boxed(text):
    """Extract the content of the last \\boxed{} in a string."""
    matches = []
    for m in re.finditer(r'\\boxed\{', text):
        start, depth, i = m.end(), 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == '{':   depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start:i-1])
    return matches[-1].strip() if matches else None


def normalise(expr: str) -> str:
    """Normalise a LaTeX expression for loose equality comparison."""
    if expr is None:
        return expr
    expr = re.sub(r'\s+', '', expr)           # collapse / remove all whitespace
    expr = expr.replace(r'\dfrac', r'\frac')  # treat \dfrac identical to \frac
    return expr


def add_boxed_match(filepath: str) -> dict:
    """
    Reads an existing critique file, computes BOXED_MATCH, and rewrites
    the file with the field appended/updated. Returns a summary dict.
    """
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")

    # --- Extract reference solution block ---
    ref_match = re.search(
        r'REFERENCE SOLUTION:\s*(.*?)(?=\nGENERATOR:|\nDATASET:|\Z)',
        content, re.DOTALL
    )
    reference_block = ref_match.group(1).strip() if ref_match else ""

    # --- Extract generated reasoning from Step blocks ---
    steps = re.findall(
        r'--- Step \d+ .+? ---\s*(.*?)(?=\n--- Step|\nRAW_CRITIC|\Z)',
        content, re.DOTALL
    )
    generated_block = "\n".join(steps)

    ref_answer  = extract_boxed(reference_block)
    gen_answer  = extract_boxed(generated_block)
    boxed_match = 0 if (ref_answer and gen_answer and normalise(ref_answer) == normalise(gen_answer)) else 1

    # --- Remove any existing BOXED_MATCH line, then append updated one ---
    content = re.sub(r'\nBOXED_MATCH:.*', '', content)
    content = content.rstrip() + f"\n\nBOXED_MATCH: {boxed_match}\n"

    path.write_text(content, encoding="utf-8")

    return {
        "file":             str(path),
        "reference_answer": ref_answer,
        "generated_answer": gen_answer,
        "boxed_match":      boxed_match,
    }


def process_directory(root_dir: str) -> None:
    """Recursively find all .txt critique files and add BOXED_MATCH."""
    files = list(Path(root_dir).rglob("*.txt"))
    print(f"Found {len(files)} file(s) under {root_dir}\n")

    for fp in files:
        result = add_boxed_match(str(fp))
        print(f"  {result['file']}")
        print(f"    ref={result['reference_answer']!r}  "
              f"gen={result['generated_answer']!r}  "
              f"BOXED_MATCH={result['boxed_match']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add BOXED_MATCH labels to critique files in a directory."
    )
    ap.add_argument("--root", default="output/factuality/critiques",
                     help="Directory to recursively search for critique .txt files (default: %(default)s)")
    args = ap.parse_args()
    process_directory(args.root)