"""Regression check: every externalised prompt must still match the original code.

The prompt YAMLs under ``reranker_prompts/`` were extracted from string literals
that used to live inside the reranker implementations. A single changed word
silently degrades ranking quality, so this script re-derives the literals from a
git revision (default: the commit that still had them inlined) and compares.

Comparison ignores placeholder *names* (``{x.text}`` may have become ``{text}``)
but requires the literal text around them, and the number of placeholders, to be
identical.

    python -m rankify.prompts.verify_prompts             # against BASELINE_REV
    python -m rankify.prompts.verify_prompts --rev <sha>
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).parent / "reranker_prompts"

# Last commit whose reranker sources still carried the prompts as inline literals.
BASELINE_REV = "9492acf"

# yaml stem -> source files that originally held its prompts
SOURCES = {
    "monot5": ["rankify/models/monot5.py"],
    "rankt5": ["rankify/models/rankt5.py"],
    "rankllama": ["rankify/models/rankllama_reranker.py"],
    "rank1-7b": ["rankify/models/rank1_reranker.py"],
    "rankr1-7b": ["rankify/models/rankr1_reranker.py"],
    "setwise-sft-7b": ["rankify/models/rankr1_reranker.py"],
    "setr": ["rankify/models/setr_reranker.py"],
    "rubric4setwise": ["rankify/models/rubric4setwise_reranker.py"],
    "rank4gen": ["rankify/models/rank4gen_reranker.py"],
    "rearank-7b": ["rankify/models/rearank_reranker.py"],
    "reasonrank-7b": ["rankify/models/reasonrank_reranker.py"],
    "rankvicuna": [
        "rankify/models/vicuna_reranker.py",
        "rankify/utils/models/rank_llm/rerank/listwise/rank_listwise_os_llm.py",
    ],
    "rankzephyr": [
        "rankify/models/zephyr_reranker.py",
        "rankify/utils/models/rank_llm/rerank/listwise/rank_listwise_os_llm.py",
    ],
    "rankgpt": ["rankify/models/rankgpt.py"],
}

# Keys that are not prompt text (metadata / non-template config).
SKIP_KEYS = {"method", "description", "notes", "default_prompt_mode", "reference"}

PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")


def segments(text: str):
    """Literal chunks around ``{placeholder}`` holes."""
    return PLACEHOLDER_RE.split(text)


def n_holes(text: str) -> int:
    return len(PLACEHOLDER_RE.findall(text))


def fstring_template(node: ast.JoinedStr) -> str:
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value).replace("{", "{{").replace("}", "}}"))
        elif isinstance(v, ast.FormattedValue):
            parts.append("{" + ast.unparse(v.value) + "}")
    return "".join(parts)


def collect_literals(source: str):
    """All string literals and f-string templates in a python source."""
    out = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append(fstring_template(node))
    return out


def git_show(rev: str, rel_path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{rev}:{rel_path}"], cwd=REPO_ROOT, text=True
    )


def walk_strings(obj, path=""):
    """Yield (dotted_key, value) for every string leaf, skipping metadata keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            yield from walk_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default=BASELINE_REV,
                    help="git revision holding the inlined prompts")
    ap.add_argument("--min-len", type=int, default=6,
                    help="ignore very short values (separators like '\\n')")
    args = ap.parse_args()

    failures = []
    checked = 0

    for stem, rel_paths in sorted(SOURCES.items()):
        yaml_path = PROMPTS_DIR / f"{stem}.yaml"
        if not yaml_path.is_file():
            print(f"[SKIP] {stem}.yaml not present")
            continue

        literals = []
        for rel in rel_paths:
            try:
                literals.extend(collect_literals(git_show(args.rev, rel)))
            except subprocess.CalledProcessError:
                failures.append(f"{stem}: cannot read {rel} at {args.rev}")

        # index by (segments, hole count) for placeholder-insensitive matching
        index = {}
        for lit in literals:
            index.setdefault((tuple(segments(lit)), n_holes(lit)), lit)
            stripped = lit.strip()
            index.setdefault((tuple(segments(stripped)), n_holes(stripped)), lit)

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for key, value in walk_strings(data):
            if len(value) < args.min_len:
                continue
            checked += 1
            k_exact = (tuple(segments(value)), n_holes(value))
            k_strip = (tuple(segments(value.strip())), n_holes(value.strip()))
            if k_exact in index:
                continue
            if k_strip in index:
                print(f"[WARN] {stem}.yaml:{key} matches only after strip()")
                continue
            preview = value[:90].replace("\n", "\\n")
            failures.append(f"{stem}.yaml:{key} has no counterpart in {rel_paths}\n"
                            f"         value starts: {preview!r}")

    print(f"\nChecked {checked} prompt strings against rev {args.rev}")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("All externalised prompts match the original code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
