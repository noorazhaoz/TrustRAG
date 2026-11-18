# Evaluation protocol and limitations

## Fixed pilot and sources
`evaluation/datasets/optimization_benchmark_v1.jsonl` contains ten questions tied to one research PDF. The original `data/optimization.pdf` and persistent document index are **not included**, so these saved runs are inspectable and verifiable but not fully reproducible from this archive alone.

The saved retrieval rankings are in `evaluation/results/retrieval_audit_v1.jsonl`. Generated answers are in `benchmark_v1_{dense_raw_enriched,keep,hybrid,adaptive}.jsonl`; reviewed correctness records are under `evaluation/reviews/` with answer SHA-256 and condition IDs. `benchmark_v1_four_strategy_report.json` is the consolidated result; `scripts/verify_portfolio.py` validates this report using saved raw answers and review records, without running the LLM.

## Results

| Retrieval | Correctness pass | fail | uncertain | Generated tokens |
|---|---:|---:|---:|---:|
| Dense Raw | 7 | 2 | 1 | 1912 |
| Keep | 7 | 2 | 1 | 1672 |
| Hybrid | 6 | 3 | 1 | 1752 |
| Frozen Ridge Adaptive | 6 | 3 | 1 | 1755 |

- The 10 questions are hand-selected and drawn from a single PDF; **no statistical inference** is warranted.
- The manual label is `correctness`, not a claim-level citation faithfulness score.
- Labels were reused only when the answer text matched an Adaptive answer for the same question exactly. Review files preserve direct annotations and answer hashes.
- The original reported citation-ID validation was 10/10 for Adaptive and Dense Raw; **valid citations do not guarantee factual support**. Some answers contain no citations at all.
- The dense raw Q2 response ended because of `max_tokens=512` (`finish_reason=length`).
- Dense Raw Q6 included 6/10 evidence units and Q7 included 8/10 due to context packing. Dense Raw Q1–Q5 included-evidence metadata was reconstructed after the run.
- Reported latencies differ by scope: fixed strategies record **generation-only**, whereas Adaptive records **pipeline**. No end-to-end latency or cost win is claimed.
- Q5 concerns exact GPU energy; answerability is provisionally uncertain and abstention is not scored as correctness pass.
- Ten Adaptive answers exactly matched the corresponding fixed policy on the questions the Router chose: 8 Keep + 2 Hybrid. This is a run-specific consistency check, not evidence of superiority.

## Failure analysis
- **Q4:** correct two-pre-trained-model combination uses `f_fair`; re-ranked responses incorrectly use `f_simp`. The audit found the decisive chunk dropped from the top-10 after reranking, while Dense Raw retained it.
- **Q2:** some answers include unsupported comparative claims about FRAPPÉ; the Dense Raw output was truncated at the token cap.
- **Q10:** every strategy writes `1 - alpha(x)(f_simp(x) - f_perf(x))` rather than `(1-alpha(x))(f_simp(x) - f_perf(x))`; this is mathematically different. Root cause has not been isolated.

## Replaying generation
`python -m evaluation.run_fixed_baselines --method keep --max-tokens 512` requires a locally indexed `paper-demo-001`, whose chunk IDs match the saved audit, and refuses overwriting prior output; it is **not** runnable end-to-end from the ZIP alone. To rerun in another environment, bring the authorized source document, index it using the original document ID and pipeline, and verify the evidence IDs first. Do not substitute a different PDF while describing it as an identical reproduction.
