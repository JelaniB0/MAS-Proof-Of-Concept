# Run, read and compare experiments

Use the [paper pipeline](PAPER_PIPELINE.md) for preparation, initialization,
training, evaluation and recovery. New experiments with changed data/model
settings/prices need new output directories.

## Where results go

Use `results/runs/<experiment>/` for new raw artifacts. Existing explicit output
paths still work. Research training/evaluation print a readable report under
`<output-directory>/summaries/` when they return. Export into the Git-friendly
collection with:

```powershell
.\.venv-research\Scripts\python.exe -m agentbalance report `
  --source results/paper-small-v2 --out reports/paper-small-v2
```

Single-report examples are `math-s42-p1.training.json` and
`math-eval-slow-01.json` in that directory. Export reads files only, without
weights, training or API requests.

| File | Purpose |
| --- | --- |
| `README.md`, `runs.csv`, `index.json` | Run index/status |
| `training/<run>/summary.json` | Questions, updates, settings, attempts, training accuracy |
| `training/<run>/progress.csv` | Rollout scores, rewards, losses, usage, models/agents |
| `training/<run>/policy-history.json` | Decisions/probabilities over time, failures and probes |
| `training/<run>/probe-history.csv` | Same probes before training/after updates |
| `training/<run>/current-policy.json` | Latest probes, or last observed rollout for older runs |
| `evaluation/<run>/questions.csv` | Answers, correctness and usage |
| `evaluation/<run>/policy.json` | Models, gates, edges, pools and probabilities |
| `evaluation/<run>/summary.json` | Totals, prices, live/partial validity and conditions |

The neural policy is query-conditioned; a single global routing rule cannot
describe it. Exports describe recorded behavior. The `.pt` contains the parameters.
Policy-gradient loss is not prediction error. A falling loss can reflect increasingly
certain decisions; it does not by itself demonstrate higher accuracy or efficiency.

## Logging

`trace_level` defaults to `compact`: original question once, final answer,
graph decisions/probabilities, model identity/prices, agent usage, finish reasons,
tool status and errors. Repeated prompts, role instructions, intermediate agent
text and duplicated training key lists are omitted. Execution/grading still use
the complete information in memory.

For debugging, add `--trace-level full` to `research train`, `research evaluate`,
`run`, `chat` or `compare`, or set it in the config. Full traces remain local.
Reporting verbosity is excluded from checkpoint identity; it requires no retraining.
Historical raw logs are preserved.

New training takes an initial deterministic snapshot and one after each update
on the first three training questions. It uses no hosted calls/evaluation labels,
restores PyTorch RNG state and model mode, and adds local controller work outside
graph latency. Rows record exact before/after update indices. Older runs have no
fixed snapshots/exact row update boundaries; those fields remain empty.

Resumed exports follow explicit `resumed_from` provenance and include available
attempt usage/failures. A warm start without optimizer recovery is not automatically
merged. Keep original recovery files in place until export is complete.

## Next controlled experiment

`data/math-pilot-20-10/` is prepared locally: 20 training and 10 evaluation
questions. The first three eval questions overlap the smoke test; reserve a
separate untouched holdout for final claims. Start with one seed, two epochs and
a fresh controller. Comparing learning rates `0.1` (paper setting) and `0.001`
(local alternative) tests stability; neither guarantees success with this roster.

Choose efficiency before training: actual rates, labeled hypothetical prices,
or explicit token/request penalties. Direct token/request penalties require a
separate code change; this cleanup does not change rewards or optimizer settings.
With zero rates, monetary penalties vanish.

Compare learned, cheap and strong on identical evaluation questions, roles,
output limits and pacing. Report accuracy with tokens, requests, configured cost,
workers, connections and completion/failures. Up to 480 requests cover profiling
(60), 20-question/two-epoch training (240), and three-policy/ten-question evaluation
(180), assuming six calls per graph. At 65-second spacing that's roughly nine
hours before overhead; quotas may stretch it across days.

Use the compact real reports as initial demo evidence, then add matched baselines
as they complete. Distinguish mechanism coverage from measured efficiency gains
and label incomplete experiments.
