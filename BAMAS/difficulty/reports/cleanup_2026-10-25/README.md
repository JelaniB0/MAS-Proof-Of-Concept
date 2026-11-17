# Repository cleanup and verification

This cleanup retains the selected all-RouterBench pipeline, main/RAG results,
external evaluation and reviewed GSM8K label evidence. It does not retrain or
reselect the production predictor.

Superseded `broad.py`, `transfer.py` and `correctness_study.py` wrappers and their
CLI commands were retired. Shared development probes now live in `probes.py`.
Tests specific to the retired wrappers were removed; the current training,
selection, artifact-integrity, label-masking and external-evaluation checks remain.
New CLI checks cover current main/RAG selection, missing recommendations and
recommendation path containment.

The interactive runner requires the current study recommendation or an explicit
checkpoint, so it cannot silently use an older math-only model. Documentation
describes the selected frozen encoder and eleven-output ensemble; superseded
checkpoint advice was removed. The pinned five-shot export is now available via
the existing downloader so the documented current preparation can be reproduced.

Earlier development/correctness-only reports, early smoke/TF-IDF result summaries,
one-off report-generation code, redundant transcripts and seven superseded local
run directories were moved to `_archive/cleanup_2026-10-25/`. This archive is
ignored by Git and contains `moved_files.json` for recovery. Label-review records
were copied to `reports/label_audit/` before the old reports were archived.

`data/` and `runs/` remain ignored. The current checkpoint and question-level
predictions remain at their original local paths. The reviewed GSM audit and
historical GSM comparison checkpoint are retained as preparation/comparison inputs.
Shared baseline implementations remain because current modules use their API,
metrics, tokenization and loss helpers.

`verification.json` records the fresh checks. Historical accuracy reports are
unchanged evidence from the earlier model runs; passing software checks is not a
new measurement of predictive accuracy.
