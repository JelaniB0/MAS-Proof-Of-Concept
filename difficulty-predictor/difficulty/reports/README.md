# Evaluation evidence

For the architecture and training procedure, start with the
**[technical README](../../TECHNICAL_README.md)**.

| Evidence | Purpose |
| --- | --- |
| [Training report](all_routerbench_2026-10-24/REPORT.md) | Main/RAG study, model selection, pooled results, and limitations. |
| [Generalization report](generalization_2026-10-25/REPORT.md) | Fixed-checkpoint external evaluation and known failures. |
| [Generalization details](generalization_2026-10-25/DETAILS.md) | Detailed tables supporting that report. |
| [Label audit](label_audit/README.md) | Reviewed GSM8K regrading decisions and their provenance. |
| [Release verification](../../RELEASE_VERIFICATION.json) | Current software and packaging checks. |

Canonical machine-readable training evidence is kept once under
[`runs/all_routerbench_v1/`](../runs/all_routerbench_v1):

- Study-wide `coverage.json`, `source_audit.json`, `training_label_audit.json`,
  `validation_baselines.json`, `probe_overlap.json`, and `release_evaluation.json`.
- Per-cohort `main/` and `rag/` protocols, comparisons, selection, and validation
  results, alongside their selected checkpoints.
- `prepared/preparation_protocol.json`, plus manifests and train/validation/test
  splits under `prepared/main/` and `prepared/rag/`.

Reports preserve historical results. Software checks do not establish accuracy
on new questions or current cloud deployments. Dataset and checkpoint files keep
their original hashes. Redundant report copies and old cleanup logs were removed;
they remain available in Git history.
