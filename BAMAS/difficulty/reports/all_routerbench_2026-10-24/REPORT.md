# All-RouterBench difficulty predictor

This study expands the corrected GSM8K experiment to every released main
RouterBench benchmark family and its archived RAG benchmark. It produces three
pool-relative quality deficits for a question. The running BAMAS RL controller,
ILP planner, agent prompts and token allocations have not been changed.

The main model achieves held-out pooled MSE **0.086935**, versus **0.101255**
for a training-family-mean baseline, a **14.1% error reduction**. Its paired 95%
interval favors the NN. It beats that baseline on six of eight reporting groups;
ARC and Winogrande remain worse on point estimates. The RAG model improves its
point estimate too, but its article-cluster interval does not establish an
advantage over the family-mean baseline.

**Broader training did not fix short arithmetic.** For `what's 1 + 1`, raw scores
are **0.6708 / 0.7427 / 0.5947**, worse than the earlier math-only predictor's
0.5559 / 0.5415 / 0.3248. The terminal withholds that unsupported estimate.
Even the supported `What is 12 multiplied by 8?` receives excessive scores, so
the similarity guard cannot be treated as confidence. On the same 514 GSM8K test
questions, the broad model's MSE is 0.067117 versus 0.064663 for the math specialist;
that difference's interval narrowly includes zero. Broader coverage is useful,
but this study does not establish improved general math or trivial-query accuracy.

The next substantive data step is to measure outcomes on short/easy queries
using the actual intended BAMAS model pool, with verified answers and an
independent held-out set. More RouterBench coverage alone was insufficient.

## Coverage and splits

| Main family | All questions | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 7,450 | 5,257 | 1,098 | 1,095 |
| MBPP | 427 | 305 | 61 | 61 |
| MMLU | 14,042 | 9,854 | 2,088 | 2,100 |
| ARC-Challenge | 1,470 | 1,027 | 234 | 209 |
| HellaSwag | 10,042 | 7,117 | 1,466 | 1,459 |
| Winogrande | 1,267 | 893 | 182 | 192 |
| MT-Bench | 80 | 60 | 11 | 9 |
| Additional released evals | 1,719 | 1,197 | 266 | 256 |
| **Main total** | **36,497** | **25,710** | **5,406** | **5,381** |

All 86 original main-export eval names remain in `source_eval_name`. MMLU subjects
and MT-Bench variants are grouped for family-level reporting; other released
evals are retained as `additional-evals`, including tiny auxiliary evals such as
`test-match`. The coverage file lists each original eval separately. MATH is not
one of these released RouterBench benchmark families.

| Separate RAG family | All questions | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| News | 349 | 262 | 26 | 61 |
| Wikipedia | 308 | 183 | 55 | 70 |
| **RAG total** | **657** | **445** | **81** | **131** |

Overall coverage is **37,154 question records**, not 37,154 training examples.
Validation selects checkpoints/configurations; test questions remain excluded
from training. Splits group identical normalized task text and sample IDs.
Historical GSM8K/MBPP folds are preserved. RAG also groups source articles: 74
training, 14 validation and 23 test articles, with zero cross-split article overlap.
Both cohorts have zero exact normalized-task cross-split overlap. This does not
rule out semantic near-duplicates or pretraining contamination.

## Sources and audits

The [RouterBench paper](https://arxiv.org/html/2403.12031v1) describes seven main
benchmark families and a separate RAG experiment. The released
[official exports](https://huggingface.co/datasets/withmartian/routerbench/tree/main)
contain additional eval names as well.

The RAG file was deleted from the current dataset revision. I recovered its
[previous public revision](https://huggingface.co/datasets/withmartian/routerbench/tree/784e39ca67a6c6291685b4862a05dd95fb0b40da),
pinned its byte count and SHA256, and converted it using the inert pickle reader.
The reader inspects supported data structures without executing pickle reducers.
It now supports noncontiguous numeric block placements and bounded text inspection.

I also downloaded and inspected the 1.20 GB raw export. Its 401,467 rows are a
long-form view of model outcomes, with columns `index`, `sample_id`, `prompt`,
`eval_name`, `performance`, `model_response`, `model_name`, `cost`. It has no extra
gold-reference column and is not an independent set of additional questions.
Large response strings were omitted during bounded inspection; the raw export
was not treated as another training corpus.

For MMLU, ARC, HellaSwag and Winogrande I selected the five-shot export's
outcomes. Other tasks use the zero-shot-named export; its GSM8K prompts already
contain worked examples. The two export names therefore do not imply two
independent observations for every question. Fourteen ARC IDs occupy two rows
each in the five-shot file, with different demonstrations and complementary model
columns. These are merged only after confirming identical actual question text
and answer options. They do not add 14 new questions.

An internal multiple-choice consistency audit covered 26,821 questions and
295,031 model observations. Of those, 275,796 responses were strict single-letter
answers. It found zero cases of conflicting positively scored answer letters for
one question, and zero cases where a negatively scored strict letter matched a
positively scored letter. This is useful consistency evidence, **not independent
gold regrading of every multiple-choice answer**.

## Targets and model groups

The target remains `D_k(x) = 1 - mean(q_m(x) for m in G_k)`. Higher values mean
greater recorded quality deficit. Targets are relative to historical model
snapshots, prompts and evaluation procedures, rather than inherent universal
question difficulty.

GSM8K uses the existing reviewed `gsm-final-numeric-v2` regrader and local numeric
references. This retains 70,061 known individual outcomes out of 81,950 saved
responses. Ambiguous or truncated answers remain unknown. Other tasks retain
their published binary or normalized scores. MBPP scores are not newly measured
execution-based pass rates; RAG and graded tasks are not universal binary failure
probabilities.

A pool target is present only when all members of its fixed roster have known
outcomes. Across the main corpus, 1,132 questions have no complete pool target;
only seven have no known individual outcome at all. Within training, the direct
pool head has 72,270 observed pool targets; the per-model alternative has 274,513
individual targets. Neither head fabricates missing labels. This distinction
matters when interpreting "training on all datasets."

The main pools retain their earlier membership:

| Pool | Exact RouterBench model IDs |
| --- | --- |
| Weak | `mistralai/mistral-7b-chat`, `meta/llama-2-70b-chat`, `WizardLM/WizardLM-13B-V1.2`, `zero-one-ai/Yi-34B-Chat` |
| Intermediate | `meta/code-llama-instruct-34b-chat`, `mistralai/mixtral-8x7b-chat`, `claude-instant-v1`, `claude-v1` |
| Strong | `gpt-3.5-turbo-1106`, `claude-v2`, `gpt-4-1106-preview` |

RAG contains a different five-model roster, so it has a **separate predictor and
pool contract**. Training-only, equal-news/wiki mean scores rank its groups:

| RAG pool | Exact archived model IDs |
| --- | --- |
| Weak | `openai/gpt-3.5-turbo`, `openai/openai/gpt-4-1106-preview` |
| Intermediate | `youcom/rag`, `perplexity/sonar-small-online` |
| Strong | `perplexity/sonar-medium-online` |

The doubled `openai/openai` prefix is retained exactly as exported. These RAG
groups describe observed retrieval-task performance; GPT-4 being in its weak pool
does not establish that it is generally less sophisticated. These group names
must never be silently substituted for the main eleven-model roster.

## Training method

Inputs use the versioned `routerbench-task-v1` transformation. It removes
recognized worked demonstrations and benchmark answer-format footers while
retaining the actual question, options and task content. The same transformation
runs during inference. Gold answers, saved model responses, dataset names and
correctness labels are not neural input features.

The pretrained local `all-mpnet-base-v2` encoder is **frozen**. It produces
normalized 768-dimensional embeddings, with a 384-token limit and left
truncation. A head uses `768 -> 128 -> outputs`, GELU, dropout 0.2 and sigmoid
outputs. CUDA handles embedding extraction on this machine's RTX 4050 Laptop
GPU; the small heads train on CPU. This is neural head training, **not full
transformer fine-tuning**. Earlier partially fine-tuned checkpoints are retained in the ignored local archive.

The main study compares three predeclared alternatives:

1. Direct three-pool head, natural question sampling, masked MSE.
2. Direct three-pool head, square-root inverse-family sampling, masked MSE.
3. Eleven-output per-model head, square-root inverse-family sampling, masked
   individual-outcome MSE; average predicted difficulties into the fixed pools.

RAG compares the two direct-pool alternatives. Each configuration uses seeds
42/43/44, up to 100 epochs, early stopping after 12 nonimproving validation epochs,
AdamW with learning rate 0.001 and weight decay 0.001, batch size 256. Each seed
restores its best checkpoint. Configuration selection uses the three-seed
ensemble's **equal-family validation MSE**, so large MMLU/HellaSwag sets do not
alone choose the winner. The selected checkpoint contains three heads, averaged
at inference. Fifteen heads are fitted in total; only the selected three per
cohort are needed for inference.

The loss masks unknown targets. Evaluation averages observed MSE within each
pool, then equally across pools. The bootstrap was corrected to match that
estimand even when different pools have different missing labels. Comparisons
use paired, family-stratified bootstrap intervals; duplicate-task/source-article
clusters stay together. Intervals condition on these trained checkpoints and do
not account for all possible training or labeling uncertainty.

## Results

Lower MSE is better. Pooled MSE gives each pool equal weight over its
observed questions; family macro MSE also gives each benchmark family equal
weight. Missing targets never count as zeros.

### MAIN checkpoint

Selected configuration: **`task_model_sqrt_mse`**, three-seed ensemble.

| Candidate | Validation family macro MSE | Validation pooled MSE |
| --- | ---: | ---: |
| task_natural_mse | 0.085969 | 0.085219 |
| task_sqrt_balanced_mse | 0.085920 | 0.086631 |
| task_model_sqrt_mse | 0.085641 | 0.085842 |

| Held-out predictor | Pooled MSE | Family macro MSE |
| --- | ---: | ---: |
| global_mean | 0.110930 | 0.124122 |
| dataset_mean | 0.101255 | 0.106804 |
| selected_frozen_ensemble | 0.086935 | 0.091953 |

The family-mean baseline uses training outcomes and the known benchmark
family. It is a diagnostic baseline, not the deployed text predictor.

| Family | Global-mean MSE | Family-mean MSE | Selected NN MSE |
| --- | ---: | ---: | ---: |
| additional-evals | 0.205382 | 0.136061 | 0.045035 |
| arc-challenge | 0.103745 | 0.084251 | 0.091890 |
| grade-school-math | 0.092619 | 0.077436 | 0.073355 |
| hellaswag | 0.089245 | 0.085249 | 0.081175 |
| mbpp | 0.187209 | 0.158490 | 0.150679 |
| mmlu | 0.123367 | 0.120164 | 0.100333 |
| mtbench | 0.122877 | 0.125818 | 0.122806 |
| winogrande | 0.068532 | 0.066963 | 0.070348 |

Paired held-out MSE reductions (positive favors the NN):

- Versus `global_mean`: pooled reduction 0.023996, 95% interval [0.021987, 0.026075]; equal-family reduction 0.032169, interval [0.025198, 0.039557].
- Versus `dataset_mean`: pooled reduction 0.014320, 95% interval [0.012447, 0.016189]; equal-family reduction 0.014851, interval [0.008657, 0.021024].

| Pool | Observed test targets | MSE | RMSE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| weak | 4969 | 0.094545 | 0.3075 | 0.433 |
| intermediate | 5003 | 0.091269 | 0.3021 | 0.381 |
| strong | 5113 | 0.074990 | 0.2738 | 0.524 |

The strong-versus-weak gain-event AUC is 0.644. This evaluates recorded pool quality gain above 0.1, not actual BAMAS routing success.
Reloaded inference matches saved test predictions within 0.
The similarity guard accepts 95.0% of held-out inputs. Acceptance does not guarantee accuracy.

Among 518 held-out questions whose recorded pool targets are all zero, mean predicted difficulties are weak=0.276, intermediate=0.279, strong=0.184.
This is a descriptive known-outcome subgroup, not an independent set of trivial arithmetic queries.

### RAG checkpoint

Selected configuration: **`task_sqrt_balanced_mse`**, three-seed ensemble.

| Candidate | Validation family macro MSE | Validation pooled MSE |
| --- | ---: | ---: |
| task_natural_mse | 0.172185 | 0.166579 |
| task_sqrt_balanced_mse | 0.171787 | 0.165472 |

| Held-out predictor | Pooled MSE | Family macro MSE |
| --- | ---: | ---: |
| global_mean | 0.204792 | 0.205877 |
| dataset_mean | 0.197985 | 0.198433 |
| selected_frozen_ensemble | 0.186379 | 0.187144 |

The family-mean baseline uses training outcomes and the known benchmark
family. It is a diagnostic baseline, not the deployed text predictor.

| Family | Global-mean MSE | Family-mean MSE | Selected NN MSE |
| --- | ---: | ---: | ---: |
| rag_news | 0.221675 | 0.204946 | 0.198281 |
| rag_wiki | 0.190080 | 0.191919 | 0.176006 |

Paired held-out MSE reductions (positive favors the NN):

- Versus `global_mean`: pooled reduction 0.018413, 95% interval [0.006574, 0.032468]; equal-family reduction 0.018733, interval [0.006543, 0.033487].
- Versus `dataset_mean`: pooled reduction 0.011607, 95% interval [-0.004991, 0.028466]; equal-family reduction 0.011289, interval [-0.005455, 0.027557].

| Pool | Observed test targets | MSE | RMSE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| weak | 131 | 0.156853 | 0.3960 | 0.390 |
| intermediate | 131 | 0.152854 | 0.3910 | 0.227 |
| strong | 131 | 0.249428 | 0.4994 | 0.126 |

The strong-versus-weak gain-event AUC is 0.626. This evaluates recorded pool quality gain above 0.1, not actual BAMAS routing success.
Reloaded inference matches saved test predictions within 0.
The similarity guard accepts 96.2% of held-out inputs. Acceptance does not guarantee accuracy.

Among 28 held-out questions whose recorded pool targets are all zero, mean predicted difficulties are weak=0.449, intermediate=0.370, strong=0.333.
This is a descriptive known-outcome subgroup, not an independent set of trivial arithmetic queries.

### Matched historical GSM8K comparison

On the same 514 complete-case GSM8K test questions and reference labels:

- Earlier math-only predictor MSE: **0.064663**.
- New all-family predictor MSE: **0.067117**.
- Paired MSE reduction: -0.002453; 95% interval [-0.004973169848148926, 4.40861681903848e-05].

This comparison keeps questions, splits and targets fixed. Broadening the training
corpus also changes the training cohort, input transformation and candidate
selection, so it does not isolate one causal reason for any difference.

### Short prompts and broader development probes

These are raw NN outputs, regardless of whether the terminal withholds them.

| Question | Weak | Intermediate | Strong | Within similarity threshold? |
| --- | ---: | ---: | ---: | --- |
| what's 1 + 1 | 0.6708 | 0.7427 | 0.5947 | No |
| 1 + 1 | 0.8305 | 0.8258 | 0.7627 | No |
| What is one plus one? | 0.5530 | 0.6867 | 0.4534 | No |
| What is 12 multiplied by 8? | 0.4378 | 0.4001 | 0.2946 | Yes |
| Alice has one apple and receives one more apple. How many apples does she have now? | 0.3249 | 0.2473 | 0.0811 | Yes |
| Twice Angie's age, plus 4, is 20. How old is Angie? (**corpus overlap: train**) | 0.3833 | 0.2920 | 0.0577 | Yes |
| Prove that every finite integral domain is a field. | 0.1069 | 0.2756 | 0.2564 | No |
| Write a Python function to find the common elements in two lists. | 0.4912 | 0.3670 | 0.3438 | Yes |
| Which planet is known as the Red Planet? A) Earth B) Mars C) Venus D) Jupiter | 0.1279 | 0.1170 | 0.0883 | Yes |
| Write a friendly two-sentence birthday message. | 0.2706 | 0.3259 | 0.2778 | No |
| A person drops a glass onto a hard floor. What is most likely to happen next? | 0.3181 | 0.2626 | 0.2434 | Yes |

The Angie question already occurs naturally in the training corpus; its output
is not evidence of unseen-question generalization. The overlap audit identifies
this explicitly. No hand-coded arithmetic override, synthetic easy labels,
probe-driven augmentation or post-test configuration selection was used.
A support rejection is abstention,
not a corrected prediction. Broader benchmark coverage alone cannot establish
reliable short-prompt difficulty prediction.

### Training and verification evidence

- main: 9 completed head fits; total recorded fitting time 744.7 seconds (excludes embedding extraction, import/preparation and release verification).
- rag: 6 completed head fits; total recorded fitting time 44.5 seconds (excludes embedding extraction, import/preparation and release verification).
- Device: NVIDIA GeForce RTX 4050 Laptop GPU; torch 2.14.0+cu130, transformers 5.17.0.
- All 44 offline automated tests passed. They cover missing labels, model-output order, real tiny-encoder train/export/reload, duplicate tasks/articles, safe extraction and legacy interfaces.
- Both selected checkpoints were reloaded and evaluated through their public inference path. Main and RAG interactive sessions were also exercised.

## What this means for BAMAS

Broader training gives the predictor evidence from more task families. It does
not make it a solver or guarantee correct calibration on `1 + 1`, arbitrary chat
prompts, unseen model versions or new agent roles. Short arithmetic probes have
no newly measured historical model outcomes in this study; their expected ease
does not substitute for a labeled external evaluation set.

The terminal's support guard fits a nearest-training-embedding distance threshold
from validation text only. It is an input-similarity heuristic, not a correctness
probability. Withholding unsupported scores does not repair the underlying
prediction; `--show-unsupported` exposes raw values explicitly.

The controller API still returns exactly three pool scores and signed estimated
quality gains. A per-model winner additionally exposes model-specific scores.
Every feature includes model/pool identities, and RAG has a distinct cohort.
These are suitable experimental features for later RL/ILP comparisons. They do
not establish individual-model routing quality: selection evaluates pool averages,
and any per-model outputs need separate calibration and routing tests. They do
not yet establish how extra tokens, collaboration or a specific BAMAS role
changes success. That requires outcomes from BAMAS's actual model/role/token
configurations and an evaluation of routing quality versus cost.

The present tests are exploratory: earlier versions of the benchmark and these
development probes have been inspected. MT-Bench has only nine test questions;
RAG has only 23 held-out source articles. Historical benchmark improvements
should not be presented as a fresh, universal generalization result.

## Run and inspect

From the repository root:

```powershell
# Train/resume the fixed all-family study.
.\.venv-research\Scripts\python.exe -m BAMAS.difficulty all-routerbench --output BAMAS/difficulty/runs/all_routerbench_v1 --device cuda
# Fit support checks; verify saved predictions; write default recommendations.
.\.venv-research\Scripts\python.exe -m BAMAS.difficulty finalize-all-routerbench --output BAMAS/difficulty/runs/all_routerbench_v1 --device cuda
# Main eleven-model pools, loading the checkpoint once.
.\.venv-research\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py
# Inspect unsupported raw scores explicitly.
.\.venv-research\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py --show-unsupported
# Separate five-model RAG pools.
.\.venv-research\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py --cohort rag
```

Add `--prepare-only` to the first command to regenerate coverage without training.
Verified asset downloading is available via `download-routerbench --assets rag raw`;
the raw export is for the audit, not required during model training. Preparation
also uses the existing zero/five-shot exports, original split manifest, reviewed
GSM audit and local MPNet weights. Reusing a directory with a different experiment
protocol is rejected; use a new output directory to change the experiment.

Evidence is under [the study directory](../../runs/all_routerbench_v1):
`coverage.json`, `training_label_audit.json`, `source_audit.json`,
`prepared/preparation_protocol.json`, per-cohort `protocol.json`,
`validation_results.json`, `selection.json`, `comparison.json`, `RESULTS.md`,
`release_evaluation.json`, saved test predictions and candidate training histories.
Candidate training histories record actual epochs. The report folder contains
copied coverage/audit evidence and model verification output. Redundant execution
transcripts were moved to the ignored local archive during cleanup.

Code changes cover asset verification, inert data extraction, task normalization,
all-family preparation, per-model heads and aggregation, masked bootstrap
evaluation, release checks, and main/RAG terminal selection. The automated checks
cover those invariants plus the earlier training/prediction interfaces. No paid
LLM inference, BAMAS controller retraining, deployment or Git commit was performed.
