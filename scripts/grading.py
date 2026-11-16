"""Answer extraction, answer comparison and the verbatim-loop score (no MLX dependency)."""

from __future__ import annotations

import re


def last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j].strip()
    return None


def same_answer(pred: str | None, ref: str) -> bool:
    """Compare answers numerically when possible, otherwise as normalised strings."""
    if pred is None:
        return False
    # "\$" is stripped too; rows graded before that fix were regraded with scripts/regrade.py.
    norm = lambda s: re.sub(r"[\s$,]|\\!|\\\$|\\text\{[^}]*\}", "", s)
    p, r = norm(pred), norm(ref)
    try:
        return abs(float(p) - float(r)) < 1e-6
    except ValueError:
        return p == r


def repetition(ids: list[int], window: int = 1024, n: int = 16) -> float:
    """Share of n-grams in the last `window` tokens that already occurred earlier in that window.

    Near 0 for productive text, close to 1 for a verbatim loop. Works on cut samples too,
    so it separates a loop from a long but progressing chain of thought.
    """
    tail = ids[-window:]
    seen, repeats, total = set(), 0, 0
    for i in range(len(tail) - n + 1):
        gram = tuple(tail[i : i + n])
        repeats += gram in seen
        seen.add(gram)
        total += 1
    return repeats / total if total else 0.0
