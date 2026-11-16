"""Regrade stored loop_eval rows with the current grader (scripts/grading.py), in place.

Recomputes `correct` and `strict_correct` of every row in results/loop_eval_*.jsonl from its stored
`pred` and `answer`, then updates `accuracy` and `strict_accuracy` in the matching summary
(results/loop_eval_<tag>.json) and in results/final_<run>.json. Prints every row whose grade changes.

Usage: uv run python scripts/regrade.py
"""

from __future__ import annotations

import glob
import json
import os

from grading import same_answer


def main() -> None:
    summaries: dict[str, dict] = {}
    for path in sorted(glob.glob("results/loop_eval_*.jsonl")):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if not rows or rows[0].get("answer") is None:
            continue
        changed = 0
        for r in rows:
            ok = same_answer(r["pred"], r["answer"])
            if ok != bool(r["correct"]):
                changed += 1
                print(f"{os.path.basename(path)} {r['id']}: pred {r['pred']!r}, answer {r['answer']!r}, "
                      f"correct {r['correct']} -> {ok}")
            r["correct"] = ok
            if "strict_correct" in r:
                r["strict_correct"] = bool(r["closed_thinking"] and ok)
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)

        summary_path = path[: -len(".jsonl")] + ".json"
        if os.path.exists(summary_path):
            s = json.load(open(summary_path))
            s["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
            if s.get("strict_accuracy") is not None:
                s["strict_accuracy"] = sum(bool(r["strict_correct"]) for r in rows) / len(rows)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
            summaries[s["tag"]] = s

    for path in sorted(glob.glob("results/final_*.json")):
        report = json.load(open(path))
        for name in ("base", "edited"):
            if name in report and report[name]["tag"] in summaries:
                report[name] = summaries[report[name]["tag"]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
