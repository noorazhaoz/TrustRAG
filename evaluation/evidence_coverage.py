
import json
from pathlib import Path


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    audit = {
        row["question_id"]: row
        for row in load_jsonl(
            "evaluation/results/retrieval_audit.jsonl"
        )
    }

    labels = load_jsonl(
        "evaluation/reviews/answer_evidence_labels.jsonl"
    )

    methods = ("dense_raw", "keep", "hybrid")
    totals = {method: 0 for method in methods}

    print("\nQuestion | Dense Raw | Keep | Hybrid")
    print("-" * 45)

    for label in labels:
        qid = label["question_id"]
        required = set(label["answer_evidence_ids"])

        if not required:
            raise ValueError(f"{qid}: empty evidence labels")

        if qid not in audit:
            raise ValueError(f"{qid}: audit not found")

        row = audit[qid]
        results = {}

        for method in methods:
            retrieved = set(row[method])
            covered = required.issubset(retrieved)

            results[method] = covered
            totals[method] += int(covered)

        print(
            f"{qid} | "
            f"{int(results['dense_raw'])} | "
            f"{int(results['keep'])} | "
            f"{int(results['hybrid'])}"
        )

    n = len(labels)

    print("\nCoverage summary:")
    for method in methods:
        print(f"{method}: {totals[method]}/{n}")


if __name__ == "__main__":
    main()
