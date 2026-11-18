"""Validate bundled frozen policy, review hashes and consolidated results (stdlib only)."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evaluation"

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def unique(path):
    rows = read_jsonl(path)
    result = {r["question_id"]: r for r in rows}
    assert len(rows) == len(result) == 10, path
    return result

def main():
    policy = ROOT / "data/qasper/step66_new_300/development/retrieval_step71a_frozen"
    m = json.loads((policy / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((policy / m["model_filename"]).read_bytes()).hexdigest() == m["model_sha256"]
    files = {"dense_raw":"benchmark_v1_dense_raw_enriched.jsonl", "keep":"benchmark_v1_keep.jsonl", "hybrid":"benchmark_v1_hybrid.jsonl", "adaptive":"benchmark_v1_adaptive.jsonl"}
    runs = {k:unique(E / "results" / v) for k,v in files.items()}
    annotations = {
        "dense_raw":["benchmark_v1_dense_raw_q1_q4_reviews.jsonl", "benchmark_v1_dense_raw_q6_q10_reviews.jsonl"],
        "keep":["benchmark_v1_keep_q2_review.jsonl"],
        "hybrid":["benchmark_v1_hybrid_manual_reviews.jsonl"],
        "adaptive":["benchmark_v1_correctness_reviews.jsonl", "benchmark_v1_correctness_reviews_q1_q5.jsonl"],
    }
    direct = {}
    for method,names in annotations.items():
        direct[method] = {}
        for name in names:
            for review in read_jsonl(E / "reviews" / name):
                qid = review["question_id"]
                assert qid not in direct[method]
                assert review["condition"] == method
                assert review["run_id"] == f"benchmark_v1_{method}:{qid}"
                digest = hashlib.sha256(runs[method][qid]["answer"].encode()).hexdigest()
                assert digest == review["answer_sha256"]
                direct[method][qid] = review
    assert len(direct["adaptive"]) == 10
    report = json.loads((E / "results/benchmark_v1_four_strategy_report.json").read_text(encoding="utf-8"))
    entries = {(r["method"],r["question_id"]):r for r in report["questions"]}
    assert len(entries) == report["total_answers"] == 40
    for method, method_runs in runs.items():
        assert set(method_runs) == set(runs["adaptive"])
        counts = Counter()
        reused = 0
        tokens = 0
        for qid,run in method_runs.items():
            row = entries[(method,qid)]
            digest = hashlib.sha256(run["answer"].encode()).hexdigest()
            assert row["answer_sha256"] == digest
            if qid in direct[method]:
                label = direct[method][qid]["correctness"]
                assert row["label_source"] == "direct_review"
            else:
                assert run["answer"] == runs["adaptive"][qid]["answer"]
                label = direct["adaptive"][qid]["correctness"]
                assert row["label_source"] == "exact_answer_reuse"
                reused += 1
            assert row["correctness"] == label
            assert row["generation_tokens"] == run["generation_tokens"]
            assert row["finish_reason"] == run["finish_reason"]
            counts[label] += 1
            tokens += run["generation_tokens"]
        summary = report["summary"][method]
        assert summary["correctness"] == dict(counts)
        assert summary["total_generation_tokens"] == tokens
        assert summary["direct_reviews"] == len(direct[method])
        assert summary["reused_reviews"] == reused
        print(method, dict(counts), "tokens", tokens, "reused", reused)
    print("PASS: frozen policy hash and all 40 saved answer/report labels verified")

if __name__ == "__main__":
    main()
