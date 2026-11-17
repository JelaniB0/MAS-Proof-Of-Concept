# BAMAS review for difficulty-aware routing

The separation between text filtering, topology selection, role allocation and
execution is useful for this experiment. Cache generation also retains actions,
budgets, correctness and actual cost, which are useful starting points for offline
comparisons. The issues below should be addressed before interpreting policy
metrics as evidence of model selection or reliable budget control. This was a
targeted static review; no paid LLM experiments were run and existing BAMAS runtime
code was left unchanged.

## Evaluation and learning

**High: checkpoint selection is not held-out policy evaluation.**
`main_train_offline.py:450` assigns `eval_dataset = dataset`. Training calls at
approximately line 512 do not pass an independent evaluation set to `train_epoch`.
At lines 346-375, reward is summarized only on sampled actions that match logged
actions; lines 518-519 choose checkpoints using `agreed_mean_reward`. That changing
subset is not a fixed test set or an unbiased estimate of policy value, and logged
actions are not an oracle merely because they appear in a cache. Split by source
question before generating task/budget/topology variants. Select checkpoints on
validation policy value or an appropriate offline estimate, then evaluate fresh
tasks with the actual executor.

**High: the offline objective requires assumptions about behavior coverage.**
`main_train_offline.py:197-214` loads `behavior_log_probs` but uses
`-mean(log pi(logged_action|state) * logged_reward)` without behavior correction.
In general this is a reward-weighted logged-action objective, not an unbiased
on-policy REINFORCE estimator. The generators can enumerate all topology actions
under uniform behavior, which may permit direct full-action reward optimization
for those cached states; that is a special case to implement and state explicitly.
If behavior changes, use a justified offline method with coverage/propensity
handling. Match-only reward and action agreement are insufficient substitutes.

## Allocation and cost

**High: High/Low labels currently do not establish different LLM capabilities.**
`configs/0_agent_library.yml:2-47` assigns both High and Low variants of planner
and math executor to `openai/gpt-oss-20b`, with the same role-specific token cap,
while assigning different performance and cost coefficients. The rest of this
GSM8K library also uses that model. These are configured assumptions, not measured
quality tiers. Define actual model snapshots, token configurations and measured
cost/quality before comparing weak/intermediate/strong routing. Check the other
dataset libraries separately rather than assuming they share this configuration.

**High: group-average costs do not guarantee the cost of the executed choice.**
`planning/low_level_instantiator.py:186-200` and `306-320` constrain estimated group
average costs. Lines 373-385 then expose agent-class choices for each selected
slot. A more expensive actual configuration, longer history, repeated planner
calls or replanning can exceed the estimate. Use candidate-specific variables and
consistent token/cost units, plus runtime remaining-budget enforcement. Treat the
current quantity as a configured energy proxy, not measured joules or API dollars.

**Medium: unconditional tier priority can defeat task-conditional allocation.**
`planning/low_level_instantiator.py:36-69` defines two tiers and lexicographic
weights. The objectives at lines 179-184 and 299-304 use static performance
coefficients multiplied by these tier preferences. Supplying a difficulty number
alone will not make the solver learn token utility or three-tier selection. Add
question/role/model/token conditional utility coefficients and re-evaluate the
priority objective. Group-average quality also does not identify the best model
inside a pool.

## Active merge script

**Medium: a bare output filename fails.**
`scripts/merge_batch_datasets.py:40` calls `os.makedirs(os.path.dirname(output))`.
For `--output merged.jsonl`, the directory is empty and directory creation fails.
Use `Path(output).parent.mkdir(parents=True, exist_ok=True)`.

**Medium: duplicate verification silently discards records.**
At lines 33-35, `--verify` skips later rows with the same work ID, but the duplicate
count is never reported. It can silently discard a successful retry after an
earlier failure. Work IDs encode task index/budget index/pattern index rather than
dataset/configuration/seed, so combining caches from different runs can collide.
Detect conflicting duplicates and require an explicit retention policy; include
dataset and run configuration fingerprints in cache identities.

**Low: empty input and broad globs are fragile.**
Lines 51-52 divide by the merged count; empty batch files cause division by zero.
Missing patterns return silently; blank/malformed lines lack file/line diagnostics.
A glob can include the previous output and duplicate it on reruns. Stream to a
temporary output file, exclude that output from inputs, and report skipped or
invalid rows. Preserve the current source files while adding these checks.
