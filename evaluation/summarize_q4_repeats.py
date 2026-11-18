
"""Summarize repeated Q4 generations."""

import json
from collections import defaultdict
from pathlib import Path

INPUT = Path(
    "evaluation/results/q4_repeated_generations.jsonl"
)

OUTPUT = Path(
    "evaluation/results/q4_repeat_summary.json"
)


def main():
    rows = [
        json.loads(line)
        for line in INPUT.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    groups = defaultdict(list)

    for row in rows:
        groups[row["condition"]].append(row)

    summary = {}

    for condition, runs in groups.items():
        answers = [r["answer"] for r in runs]

        summary[condition] = {
            "num_runs": len(runs),
            "num_unique_answers": len(set(answers)),
            "all_finished_normally": all(
                r["finish_reason"] == "stop"
                for r in runs
            ),
            "generation_tokens": [
                r["generation_tokens"]
                for r in runs
            ],
            "evidence_ids": runs[0]["evidence_ids"],
            "identical_answers": (
                len(set(answers)) == 1
            ),
        }

    OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
