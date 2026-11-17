# GSM8K, MATH, HumanEval and new-question evaluation

The fixed all-family predictor was evaluated on **7,659 questions**. It shows
measurable within-benchmark signal and modest transfer to MATH's human difficulty
levels. Short arithmetic remains unreliable. HumanEval questions received
predictions, but their pool-difficulty accuracy cannot yet be measured because
the required model-outcome labels are absent.

The production checkpoint, model pools and support threshold were unchanged.
This evaluates the frozen-MPNet/MLP ensemble at
`runs/all_routerbench_v1/main/selected_model`, rather than the earlier partially fine-tuned MPNet baseline.

## What was tested

| Set | Questions evaluated | What we can measure |
| --- | ---: | --- |
| RouterBench GSM8K holdout | 1,095 | Error against recorded reference-regraded pool outcomes |
| RouterBench MBPP holdout | 61 | Error against published pool outcomes |
| Official GSM8K test split | 1,319 | New-question predictions and support; pool outcomes unavailable |
| MATH test split | 5,000 | Predictions, support and association with human difficulty levels |
| HumanEval | 164 | Predictions and support; pool outcomes unavailable |
| Authored/debugging prompts | 20 | Qualitative plausibility and wording sensitivity |

MATH and HumanEval are absent from the 86-eval RouterBench training export. MBPP
is its coding benchmark; `mtbench-math` and MMLU mathematics subjects are separate
tasks from the MATH dataset.

Every external question was checked against normalized actual question text in
all predictor training, validation and inspected test splits. There were **zero
exact matches to exclude**, and no exact duplicates within these external sets.
This does not rule out semantic near-duplicates or exposure during MPNet
pretraining. The authored set deliberately repeats the previously inspected
`what's 1 + 1` debugging prompt; these probes are exploratory, not a fresh
confirmatory evaluation.

## Measured prediction accuracy

Lower MSE is better. The family-mean baseline knows the benchmark family and
predicts its training-set average for each pool; it uses no test outcomes.

| Dataset | Family-mean baseline MSE | Predictor MSE | Relative error reduction |
| --- | ---: | ---: | ---: |
| RouterBench GSM8K | 0.077436 | **0.073355** | **5.27%** |
| RouterBench MBPP | 0.158490 | **0.150679** | **4.93%** |

The paired 95% interval for GSM8K's MSE reduction is **[0.000898, 0.007090]**,
favoring the predictor. For MBPP it is **[-0.022163, 0.036446]**, so its point
improvement does not establish a clear advantage on this small test set.
These intervals condition on the trained model and observed labels; they do not
include all training/label uncertainty. This is a repeated evaluation of an
already inspected test split, not new independent evidence from a new split.

GSM8K has missing labels where archived responses could not be confidently
regraded. Of 1,095 questions, 920 have at least one complete pool target; 175
have none. Unknowns are excluded independently per pool and never counted as
failures. All 61 MBPP questions have three targets.

| Pool | Known GSM8K targets | Known MBPP targets | Combined MAE | Combined RMSE | Combined rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weak | 683 | 61 | 0.2656 | 0.3206 | 0.301 |
| Intermediate | 717 | 61 | 0.2373 | 0.2906 | 0.255 |
| Strong | 827 | 61 | 0.1600 | 0.2232 | 0.248 |

The absolute errors are substantial. Beating a baseline does not make the
individual difficulty estimates precise. The labeled results contain five
fixed-seed sampled questions per dataset with actual targets and predictions.

## MATH: evidence of limited transfer

The [MATH dataset](https://github.com/hendrycks/math) includes human difficulty
levels 1–5. Those levels let us test a ranking proxy without pretending they are
the observed failure fractions of our eleven RouterBench models.

| Human level | Questions | Mean predicted weak | Mean predicted intermediate | Mean predicted strong |
| --- | ---: | ---: | ---: | ---: |
| 1 | 437 | 0.4821 | 0.4164 | 0.2699 |
| 2 | 894 | 0.5221 | 0.4527 | 0.2986 |
| 3 | 1,131 | 0.5558 | 0.4779 | 0.3178 |
| 4 | 1,214 | 0.5981 | 0.5197 | 0.3633 |
| 5 | 1,324 | 0.6310 | 0.5461 | 0.3903 |

All three mean scores rise at every level. Spearman correlations with human
levels are **0.272 weak / 0.239 intermediate / 0.230 strong**. This is a modest
positive association, with substantial overlap among levels, rather than precise
problem-by-problem ranking.

A supplemental exploratory breakdown shows the effect also varies by subject:
Algebra correlations are about **0.32–0.35**, while Counting & Probability is
about **0.006–0.016**, effectively no useful ranking association in that subset.
The per-subject results are saved in `math_subject_proxy.json`.

This suggests some transferable math difficulty signal. It does **not** verify
calibration against actual MATH outcomes for the three model pools. That would
require recorded model answers and grading under the intended prompt conditions.

## External prediction distributions

These are average predicted quality deficits, not measured failure rates.

| Dataset | Mean weak | Mean intermediate | Mean strong | Within input-similarity threshold |
| --- | ---: | ---: | ---: | ---: |
| Official GSM8K test | 0.4507 | 0.3239 | 0.1114 | 99.7% |
| MATH | 0.5735 | 0.4962 | 0.3404 | 53.2% |
| HumanEval | 0.6584 | 0.5075 | 0.3853 | 78.7% |
| Authored/debugging prompts | 0.4770 | 0.4214 | 0.2766 | 45.0% |

HumanEval is an external code-generation domain. Its canonical solutions and
unit tests were not supplied to the predictor or executed. No LLM completions
were generated, so this study measures neither HumanEval pass@1 nor the accuracy
of predicted pool failure fractions there. The same missing-outcome limitation
applies to the official GSM8K test split, despite it providing gold answers.

The support check is a text-similarity heuristic. Its acceptance percentages
describe input resemblance; they are not accuracy or confidence percentages.
Raw scores are retained for both supported and unsupported inputs in this audit.

## New prompts: useful behavior and clear failures

| Prompt | Weak | Intermediate | Strong |
| --- | ---: | ---: | ---: |
| What's 1 + 1 | 0.6708 | 0.7427 | 0.5947 |
| What is 1 + 1? | 0.5289 | 0.6331 | 0.3949 |
| Maya has one pear and receives another pear. How many pears does Maya have? | 0.3162 | 0.2040 | 0.0819 |
| Compute 1 + 1 and briefly explain your answer. | 0.8407 | 0.8227 | 0.6478 |
| What is the capital of France? | 0.0957 | 0.1043 | 0.0235 |
| Write a Python function that returns the square of an integer. | 0.3773 | 0.1277 | 0.1129 |
| Implement edit distance in Python using dynamic programming and explain its complexity. | 0.8858 | 0.8110 | 0.7234 |

Some predictions have plausible ordering, such as lower difficulty for a simple
coding function than a longer algorithm request. The arithmetic examples expose
a large reliability problem: trivial addition receives excessive scores and
large wording-dependent changes. Across the six addition phrasings, the score
ranges span **0.525 weak / 0.619 intermediate / 0.566 strong**. Different response
instructions can affect real outcomes, but this is a poor sanity result for such
a simple underlying task. These examples have no measured pool-outcome labels;
they are qualitative diagnostics, not accuracy estimates.

## Process and verification

1. Selected the existing validation-chosen checkpoint and checked its pool
   contract against the prepared target definitions.
2. Downloaded the seven MATH test Parquet files from the
   [EleutherAI mirror](https://huggingface.co/datasets/EleutherAI/hendrycks_math),
   pinned revision `21a5633873b6a120296cce3e2df9d5550074f4a3`, and verified their
   SHA256 hashes against the platform metadata.
3. Downloaded [OpenAI HumanEval](https://github.com/openai/human-eval) at revision
   `b9453e8f1865df0a0057a1383767caa400acfbcf`, verified the Git blob and recorded
   the SHA256. Data was parsed as Parquet/JSON; no dataset code was executed.
4. Used actual task text as neural input, excluding reference solutions, tests,
   dataset IDs and labels. Checked exact question overlap before external inference.
5. Ran CUDA inference through the public saved-model API. Compared labeled
   predictions with baselines using the existing masked metrics and paired
   bootstrap. Checked MATH human levels only as a separate proxy.
6. Reproduced prior saved held-out predictions within **5.96e-7**; the tiny
   difference is below the 1e-6 numerical tolerance. Checked that checkpoint
   metadata was unchanged after inference.
7. Ran the complete offline software suite: **47 tests passed**, including three
   new checks for external overlap, reference/test-code isolation, asset tampering
   and rejection of mismatched model pools. Software tests validate the pipeline;
   they do not establish model accuracy.

The benchmark function took **174.6 seconds** after imports, including loading
inputs/model and evaluation. No production training, checkpoint selection, pool
reranking, support-threshold fitting, controller changes or paid LLM calls occurred.

## Run it yourself

From the repository root, repeat the fixed-model test with:

```powershell
.\.venv-research\Scripts\python.exe -m BAMAS.difficulty benchmark --device cuda
```

The data is already present. If it needs to be restored, the pinned public
downloader is:

```powershell
.\.venv-research\Scripts\python.exe -m BAMAS.difficulty download-evaluation-data
```

For your own terminal questions:

```powershell
.\.venv-research\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py --show-unsupported
```

The benchmark writes detailed tables and sampled labeled questions (`../../runs/generalization_test/RESULTS.md`, broader-workspace reference)
and per-question JSONL files under `runs/generalization_test`. The machine-readable
results, protocol and source checksums are also copied alongside this report.
Historical execution transcripts were moved to the ignored local archive during
cleanup; current software checks are recorded under `reports/cleanup_2026-10-25/`. The production model remains in its existing
study directory; rerunning this test does not train it.

## What to do with these results

Use the predictor as an experimental benchmark-performance feature with explicit
support/target metadata. The MATH level trend is encouraging evidence of limited
transfer, while the arithmetic wording failures show why arbitrary-query
reliability is not established. To measure MATH/HumanEval pool-difficulty accuracy
or prepare an RL/ILP controller, collect graded per-question outcomes from the
intended BAMAS model/role/token configurations, including short easy prompts,
and retain an independent held-out evaluation set.
