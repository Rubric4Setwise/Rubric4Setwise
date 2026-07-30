"""Externalised prompt templates.

Every reranker that talks to an LLM in natural language keeps its prompts in
``rankify/prompts/reranker_prompts/<ranker>.yaml`` instead of hard-coding them
in Python. One file per ranker, named exactly like the preset id used by
``run_pipeline.py`` (e.g. ``setr.yaml``, ``rank4gen.yaml``).

Editing a prompt therefore never requires touching the model code, and the
prompt actually used by a run can be diffed / version-controlled on its own.

Usage inside a reranker implementation::

    from rankify.prompts import load_reranker_prompts

    _P = load_reranker_prompts("setr")
    SYSTEM_PROMPT = _P["system_message"]
    USER_PROMPT = _P["user_message"]

Set ``RANKIFY_PROMPTS_DIR`` to point the loader at a different directory if you
want to try alternative prompts without editing the shipped files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = [
    "PROMPTS_DIR",
    "RERANKER_PROMPTS_DIR",
    "load_reranker_prompts",
    "load_prompt_file",
    "available_reranker_prompts",
]

PROMPTS_DIR = Path(os.environ.get("RANKIFY_PROMPTS_DIR", Path(__file__).parent))
RERANKER_PROMPTS_DIR = PROMPTS_DIR / "reranker_prompts"

_CACHE: Dict[Path, Dict[str, Any]] = {}


def load_prompt_file(path) -> Dict[str, Any]:
    """Load and cache a single prompt YAML file."""
    path = Path(path)
    cached = _CACHE.get(path)
    if cached is not None:
        return cached

    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must contain a YAML mapping: {path}")

    _CACHE[path] = data
    return data


def load_reranker_prompts(ranker_name: str) -> Dict[str, Any]:
    """Return the prompt mapping for ``ranker_name``.

    Args:
        ranker_name: file stem under ``reranker_prompts/``, i.e. the preset id
            (``"rank4gen"``, ``"setwise-sft-7b"``, ...).

    Raises:
        FileNotFoundError: if no YAML exists for that ranker. The message lists
            the available ones, since a silent fallback to some default prompt
            would quietly change ranking quality.
    """
    path = RERANKER_PROMPTS_DIR / f"{ranker_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No prompt file for reranker '{ranker_name}' (expected {path}). "
            f"Available: {', '.join(available_reranker_prompts()) or 'none'}"
        )
    return load_prompt_file(path)


def available_reranker_prompts() -> list:
    """Sorted list of ranker names that ship a prompt YAML."""
    if not RERANKER_PROMPTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in RERANKER_PROMPTS_DIR.glob("*.yaml"))
