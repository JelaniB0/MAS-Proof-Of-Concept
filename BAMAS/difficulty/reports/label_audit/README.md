# GSM8K label audit

These retained records document the numeric-answer regrading used by the current
all-RouterBench predictor. They are historical audit evidence, not a new accuracy
evaluation or a checkpoint recommendation.

The original supplied GSM8K scores were mostly graded values rather than binary
answer correctness. The current pipeline regrades saved responses against local
GSM8K numeric references with `gsm-final-numeric-v2`. Unclear extractions remain
unknown. The reviewed cohort retained 70,061 known individual outcomes out of
81,950 saved responses. The main study masks missing individual targets; a pool
target requires known outcomes for every member of that pool.

- `regrade_protocol.json` records the source/reference hashes and original splits.
- `regrade_summary.json` records extraction coverage and the earlier complete-case
  cohort. Its pending-review field describes generation time; the subsequent
  review decisions are recorded separately below.
- `review_sample.jsonl` and `manual_review.json` preserve the reviewed sample and
  its decision.
- `edge_review_sample.jsonl` and `edge_review.json` preserve additional edge cases.
- `label_diagnostics.json` records the historical target distributions.

For current corpus coverage, masking and measured results, see the
[all-RouterBench report](../all_routerbench_2026-10-24/REPORT.md).

Earlier correctness-only training comparisons and development reports were moved
to the ignored local `_archive/cleanup_2026-10-25/` folder during repository cleanup.
