# Difficulty prediction and LinUCB: how the pieces fit together

> Standalone branch scope: this checkout contains neural predictors and their
> prepared data/checkpoints. Azure-neighbor routing and LinUCB implementation
> references describe the broader workspace and are not shipped in this branch.

This guide explains what our difficulty scores mean, where their training data
comes from, how embeddings improve the available evidence, and how we can use
those scores to guide a bandit that chooses one answering model.

**The predictor supplies an estimate. LinUCB learns which choice actually works.**
Thresholds can provide starting preferences, while observed answer quality and
cost provide the feedback that changes future decisions.

Jump to [score meaning](#2-what-does-a-difficulty-score-mean),
[data](#3-what-data-did-we-use), [training](#4-how-was-the-neural-predictor-trained),
[embedding blend](#6-how-are-the-two-difficulty-estimates-combined),
[LinUCB](#7-what-would-linucb-learn),
[your proposed thresholds](#8-how-can-our-proposed-thresholds-guide-the-starting-pool),
[feedback and retries](#9-what-happens-when-an-answer-is-unsatisfactory), or
[running it](#12-how-do-i-run-and-inspect-it).

## 1. What is implemented today?

| Component | Current state |
| --- | --- |
| Trained difficulty predictor | Available: frozen MPNet encoder and three trained MLP heads. |
| Similar-question retrieval | Available: Azure `text-embedding-3-small`, with 25,701 unique training references. |
| Combining predictor and neighbor evidence | Available: a weighted blend produces three final difficulty scores. |
| Interactive previews | Available: prints difficulty, similar questions, similarity scores, and timing. |
| Model selection in that preview | A deterministic quality-target/cost rule; selection metadata is saved in JSONL. |
| Answer generation in that preview | Disabled. It does not run the selected answering model. |
| LinUCB choosing among the three model pools using this hybrid signal | Proposed integration, described below. |
| Retrying an unsuccessful answer in a stronger pool | Proposed extension; not part of the current preview. |

There is also a separate BAMAS_adjusted experiment (`../../BAMAS_adjusted/README.md`, broader-workspace reference)
that already accepts the neural predictor's features and uses LinUCB to select
among **four collaboration topologies**. Its arms are not our three model pools,
and it does not currently consume the Azure-neighbor blend. Having that adapter
does not mean the single-agent preview has a trained bandit policy.

For the next single-agent experiment, one decision selects one answering model
from the weak, intermediate, or strong pool. Multi-agent topology selection is a
separate decision to study later.

## 2. What does a difficulty score mean?

Each question gets three scores:

```text
weak difficulty          = how difficult this looks for the historical weak pool
intermediate difficulty  = how difficult this looks for the historical intermediate pool
strong difficulty        = how difficult this looks for the historical strong pool
```

Scores range from **0 to 1**. Lower means a smaller predicted quality deficit;
higher means a larger predicted quality deficit.

The historical target is:

```text
pool difficulty = 1 - average recorded quality of that pool's models
```

For example, if a weak pool's four models had binary outcomes `[0, 0, 0, 1]`,
their average quality would be `0.25` and the observed difficulty would be `0.75`.
Here the label describes three failures out of four recorded model outcomes.
Some datasets use graded quality scores, so the general interpretation is a
quality deficit rather than a binary failure probability.

For a new question, we predict that target from its text. We do not know its
actual difficulty or answer correctness yet.

An illustrative prediction is:

| Pool | Predicted difficulty | Reading |
| --- | ---: | --- |
| Weak | 0.80 | Historically similar capability levels appear likely to struggle. |
| Intermediate | 0.50 | The expected quality deficit is smaller. |
| Strong | 0.30 | This is the most promising pool according to these estimates. |

These are **not calibrated failure probabilities for our current Azure models**.
For example, `strong = 0.30` does not establish a 70% chance that today's strong
deployment answers correctly. The labels describe older models, prompts, and
evaluation procedures; transferring them to current deployments needs measured
feedback.

The three predictions are independent. A stronger-named pool is not forced to
have a lower score on every question. The ordering is evidence to examine, not
something the code silently repairs.

## 3. What data did we use?

### Main corpus: RouterBench

The selected predictor uses prepared RouterBench question records and archived
outcomes from eleven historical models. The main corpus contains **36,497 rows**:

| Split | Question records | Purpose |
| --- | ---: | --- |
| Training | 25,710 | Fit the neural prediction heads. |
| Validation | 5,406 | Select checkpoints/configurations and calibrate input-support checks. |
| Test | 5,381 | Evaluate predictions after selection; excluded from fitting. |

The eight reporting groups are GSM8K, MBPP, MMLU, ARC-Challenge, HellaSwag,
Winogrande, MT-Bench, and additional released evaluations. They cover arithmetic
and reasoning, coding, subject knowledge, commonsense, and graded conversational
tasks. MATH is an external evaluation set, not part of this main training corpus.

Question text and sample IDs are grouped when forming splits to keep matching
questions together. The preparation preserves the original GSM8K/MBPP folds
where applicable. This checks exact normalized overlap; it cannot establish that
two questions are semantically unrelated or absent from a model's pretraining.

### Which models define the historical pools?

| Pool | Historical members |
| --- | --- |
| Weak: four models | Mistral 7B, Llama 2 70B, WizardLM 13B, Yi 34B. |
| Intermediate: four models | Code Llama 34B, Mixtral 8x7B, Claude Instant v1, Claude v1. |
| Strong: three models | GPT-3.5 Turbo 1106, Claude v2, GPT-4 1106 Preview. |

Exact IDs are preserved in the selected checkpoint's `manifest.json` and the
[training report](../BAMAS/difficulty/reports/all_routerbench_2026-10-24/REPORT.md#targets-and-model-groups).
These groups are historical definitions, not a current model-quality ranking.

Our cloud representatives are configured separately in
`configs/azure.local.json` (`../../configs/azure.local.json`, broader-workspace reference), using deployment
variables from `.env`. A historical pool average does not directly distinguish
the best individual current model within a pool.

### How were the labels checked?

GSM8K archived final numeric answers were regraded against references using the
reviewed `gsm-final-numeric-v2` extraction procedure. Other tasks retain their
published binary or normalized quality metrics. This does not mean all archived
answers were independently regraded or all code responses were executed again.

Unknown outcomes are masked during training; they are not counted as failures.
A recorded pool target is present only when all of that fixed pool's members
have known outcomes. The per-model training head can still use available
individual outcomes when some other models have missing labels.

The main training split supplies **274,513 observed individual-model targets**.
That number counts known outcomes, not additional independent questions.
See the [label audit](../BAMAS/difficulty/reports/label_audit/README.md) for provenance.

There is also a separate archived RAG corpus with 657 rows and a different
five-model roster. It has its own predictor and pool contract. It is not silently
combined with the main eleven-model predictor used by our current preview.

## 4. How was the neural predictor trained?

The selected configuration is `task_model_sqrt_mse`:

```text
Question text
    -> normalize the actual task
    -> frozen pretrained MPNet encoder
    -> normalized 768-number representation
    -> three independently trained MLP heads
    -> average their eleven individual-model predictions
    -> average the predictions within the fixed 4 / 4 / 3 pools
    -> three neural difficulty scores
```

The three heads are an ensemble trained with different random seeds. They are
**not one head per pool**: every head predicts all eleven historical models.

### Input preparation

The same `routerbench-task-v1` transformation runs for training and prediction.
It removes recognized worked demonstrations and answer-format footers while
retaining the actual task and answer options. Reference answers, saved model
responses, correctness labels, and dataset names are not predictor inputs.

The encoder processes at most 384 tokens. Longer inputs are left-truncated,
which keeps the ending and can discard important earlier context.

### Architecture and training settings

| Setting | Selected study |
| --- | --- |
| Encoder | `sentence-transformers/all-mpnet-base-v2`, frozen. |
| Prediction head | `768 -> 128 -> 11`, GELU, dropout 0.2, sigmoid outputs. |
| Training target | `1 - recorded quality` for each observed individual model. |
| Loss | Masked mean squared error; missing outcomes do not contribute. |
| Optimizer | AdamW. |
| Learning rate / weight decay | 0.001 / 0.001. |
| Batch size | 256. |
| Training limit | Up to 100 epochs, with early stopping after 12 non-improving epochs. |
| Random seeds | 42, 43, 44. |
| Sampling | Replacement sampling weighted by `1 / sqrt(training family size)`. |
| Selection | Validation MSE averaged equally across reporting families. |

The sampling gives smaller families more opportunity to contribute without
fully equalizing their total sampling mass. Each seed restores its best
validation checkpoint. The final configuration is selected using validation
results, not by choosing the lowest test error.

MPNet weights were not fine-tuned in this selected study. Embedding extraction
could use CUDA; the small prediction heads trained on CPU. Running a new question
does not update these weights or add another training example automatically.

The selected checkpoint is:

```text
BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model
```

Encoder, tokenizer, head, and pool fingerprints bind saved predictions to the
correct model and label definition. Training/reproduction details are retained
in the [technical README](../TECHNICAL_README.md#training-pipeline) and
[saved protocol](../BAMAS/difficulty/runs/all_routerbench_v1/main/protocol.json).

## 5. Why do we also use Azure embeddings?

The neural predictor and Azure embedding model do different jobs:

| Component | What it contributes |
| --- | --- |
| MPNet plus trained prediction heads | A learned estimate of difficulty from the question text. |
| Azure `text-embedding-3-small` | A representation used to locate similar historical training questions. |
| Historical labels on retrieved questions | Evidence about how the reference pools performed on those examples. |

We do not ask the Azure embedding model to assign a difficulty score. It finds
nearby examples; their recorded difficulty labels provide the second estimate.
Creating these embeddings does not fine-tune either embedding model.

Our expanded searchable collection has **25,701 unique training questions**.
The 25,710 training rows include repeated normalized questions. For this index,
repeated questions become one reference with the mean of available pool labels,
so repeated observations do not receive extra neighbor votes.

We also embedded **5,402 unique validation questions** to calibrate retrieval
support. Their outcomes do not become reference labels. Test questions can be
embedded as evaluation queries, but their labels stay outside the reference
collection. The 100 demonstration queries are cached queries, not newly labeled
training references.

Vectors are stored in our local index and embedding cache. ChromaDB is optional;
it would change vector storage/search rather than the meaning of difficulty.
New uncached questions use the Azure embedding API and can incur charges.
Retrieving already stored vectors locally makes no embedding API request.

## 6. How are the two difficulty estimates combined?

For each new question:

1. Run the neural predictor to obtain three scores.
2. Embed the question using the same Azure deployment as the reference index.
3. Retrieve the eight nearest training questions by cosine similarity, excluding
   exact normalized self-matches.
4. Keep neighbors that pass the retrieval support threshold. For each pool,
   ignore neighbors whose label for that pool is unknown.
5. Give closer neighbors more weight and calculate a neighbor difficulty estimate.
6. Blend that estimate with the neural prediction, separately for each pool.

The neighbor weights are `softmax(cosine_similarity / 0.1)`. Before blending,
the weighted neighbor estimate is pulled toward the training-pool mean with
prior strength 2. This reduces the influence of a single reference example.

The final blend is:

```text
D = (1 - w) * neural_difficulty + w * neighbor_difficulty

w = 0.5 * min(1, effective_neighbor_count / 8)
```

The effective neighbor count is `1 / sum(normalized_neighbor_weight ** 2)`.
It is high when several neighbors contribute similarly, and lower when one
neighbor dominates. It is an evidence-weighting measure, not calibrated confidence.

For example, suppose the neural estimate is `0.60`, the already-adjusted neighbor
estimate is `0.20`, and the effective count is 4:

```text
w = 0.5 * (4 / 8) = 0.25
D = 0.75 * 0.60 + 0.25 * 0.20 = 0.50
```

No usable neighbor labels means `w = 0`: the neural estimate is retained.
Eight equally weighted usable neighbors reach a 50/50 blend. The blend is a
hand-designed rule today; it is not a learned LinUCB decision.

### Three different kinds of threshold

| Threshold | What it controls | Current value or proposed example |
| --- | --- | --- |
| Neural input support | Whether MPNet sees a sufficiently nearby training representation. Lower cosine distance is closer. | Distance at most about 0.4764, calibrated from validation text. |
| Azure retrieval support | Whether a reference is close enough to contribute neighbor labels. Higher cosine similarity is closer. | Similarity at least about 0.5103 in the expanded index. |
| Routing preference bands | Which pool we would initially prefer given predicted difficulty. | Proposed 0.40 / 0.60 bands below; not implemented in the preview. |

These values are in different spaces and are not interchangeable. The current
preview requires both input-support checks for its normal eligibility rule.
Passing either check is not an accuracy guarantee.

## 7. What would LinUCB learn?

For single-agent model selection, the three arms would be the weak, intermediate,
and strong representatives. LinUCB would learn the expected **observed reward**
for each arm as a function of the question's context.

```text
Question
    -> predicted difficulties + evidence/support + budget
    -> LinUCB chooses one feasible model arm
    -> that model answers
    -> grade the answer and measure actual cost/latency
    -> update the selected arm
    -> use the updated policy on later questions
```

Initially, we can keep the neural predictor frozen. The bandit learns how to
use its scores; a bandit update does not retrain MPNet or its MLP heads.

A useful starting context includes the three difficulties, score gaps, overall
spread, difficulty-band flags, remaining budget, and prompt length. We can then
add the separate neural and neighbor scores, their disagreement, support flags,
and effective neighbor counts as an ablation. Preserve presence flags when
neighbor labels are missing; missing evidence must not become a zero-difficulty
observation.

Using separate signals lets the policy learn when neighbor evidence helps.
Using only the final blend leaves the fixed blending rule in place, although
LinUCB can still learn which arm works best given that blend.

Standard disjoint LinUCB uses one reward model per arm:

```text
arm score = predicted reward + exploration bonus
          = theta_arm dot context
            + alpha * sqrt(context.T @ inverse(A_arm) @ context)
```

After a measured reward `r` for the selected arm:

```text
A_arm = A_arm + outer(context, context)
b_arm = b_arm + r * context
theta_arm = solve(A_arm, b_arm)
```

Other arms are not updated with guessed outcomes. The exploration bonus concerns
uncertainty in the bandit's reward model; it does not turn a similarity score
into a confidence probability. See the [original LinUCB paper](https://arxiv.org/abs/1003.0146).

## 8. How can our proposed thresholds guide the starting pool?

The following is a **proposed starting policy**, not the current preview rule.
All numbers are tunable development settings, not validated optimal cutoffs.

First make "the scores are close" explicit:

```text
spread = max(weak, intermediate, strong) - min(weak, intermediate, strong)
scores_are_close = spread <= 0.10
overall_level = mean(weak, intermediate, strong)
```

When the scores are close, use their mean to choose the starting band:

| Similar-score situation | Starting preference |
| --- | --- |
| Mean at or below 0.40 | Weak, assuming it is the least expensive feasible representative. |
| Mean above 0.40 and below 0.60 | Intermediate. |
| Mean at or above 0.60 | Strong, if feasible under the remaining budget. |

This assigns exact `0.40` to the weak band and exact `0.60` to the strong band.
Using the mean avoids conflicting bands when individual close scores straddle
a boundary. A `0.10` spread is a chosen tolerance, not a statistical interval.

For clearly separated scores, consider the quality advantage and price together:

| Weak | Intermediate | Strong | Proposed initial preference |
| ---: | ---: | ---: | --- |
| 0.32 | 0.30 | 0.29 | Weak: close scores, low mean, little estimated upgrade benefit. |
| 0.53 | 0.50 | 0.47 | Intermediate: close scores and a mean of 0.50. |
| 0.72 | 0.69 | 0.66 | Strong: the proposed conservative starting preference for high scores. |
| 0.80 | 0.50 | 0.30 | Strong: a large predicted advantage, subject to budget and cost. |
| 0.20 | 0.45 | 0.35 | Weak can be preferable: do not replace the actual score ordering with pool names. |

An initial large-gap setting could be `spread >= 0.20`. Between the close and
large-gap cases, let learned reward and cost evidence drive the decision rather
than introducing another arbitrary hard rule. For a large gap, the lowest-score
pool is a promising candidate; a large price increase can still outweigh its
estimated advantage. The advantage may also be wrong on an unsupported query.

There are two useful qualifications:

- **All scores high and close does not prove strong is better.** Starting strong
  there expresses a conservative prior. The scores may instead indicate that all
  pools struggle. Feedback may favor a cheaper choice, a different tool, or an
  explicit inability to satisfy the request under budget.
- **A difficulty of 0.40 is not necessarily easy enough.** Its historical target
  corresponds to a quality deficit of 0.40, not a guaranteed acceptable answer.
  Desired answer quality must be validated separately.

Also check actual configured prices. "Weak" describes a capability group, not a
guarantee that its current representative is cheapest for every input/output mix.

### Preferences should guide learning, rather than permanently override it

One option is to put the band flags and gaps into the context and let LinUCB
learn their weights. An untrained bandit will not automatically know what those
flags should mean; it needs real warm-start outcomes or an explicit cold-start
policy with exploration.

Another option is a temporary preference bonus:

```text
hybrid arm score = LinUCB score + beta * starting_preference_bonus
```

The preferred starting arm gets a small bonus, and `beta` decreases as useful
feedback accumulates. Preserve exploration among feasible alternatives and log
the prior separately from learned reward and exploration. This is a proposed
hybrid policy, not vanilla LinUCB or an already implemented feature; the original
algorithm's guarantees do not automatically extend to the extra bonus.

Availability, context limits, and a hard remaining budget can exclude an arm.
Difficulty bands should generally be soft preferences. Permanently suppressing
an arm prevents the bandit from learning whether it was a better choice.

### How is this different from today's preview?

Today's preview has a default estimated-quality target of **0.80**. For supported
inputs, a pool is eligible when `1 - difficulty >= 0.80`, approximately
`difficulty <= 0.20`. It then ranks eligible models using configured token prices
at a fixed reference workload; without prices it uses tier order. Otherwise,
it falls back to strong. That fallback does not guarantee the target is met.

The proposed 0.40/0.60 starting bands are a different policy and do not redefine
the historical labels. They would need their own validation before adoption.

## 9. What happens when an answer is unsatisfactory?

There are two different timescales.

**Learning across questions:** if weak performs well on a class of low-score
questions at low cost, its reward model improves for similar contexts. If it
fails repeatedly, its estimated reward falls and intermediate or strong becomes
more attractive. Successful intermediate answers on cases previously sent to
strong can support cheaper choices on later questions. This is the first
single-call bandit experiment to establish.

**Escalation within one question:** a later optional cascade could start at the
preferred pool, check the answer, and try a stronger pool if the check fails:

```text
weak answer -> independent check fails -> intermediate answer -> check passes
```

Define an answer check before enabling this. Depending on the task, it might use
unit tests, reference-answer grading, task-specific constraints, human feedback,
or an independently evaluated judge. Fluency or the answering model's own
confidence is not sufficient evidence of correctness. A live task may have no
reliable automatic grader; record missing feedback rather than inventing a reward.

Every retry adds cost and latency. Apply an attempt limit, keep a remaining-budget
ledger, and specify whether a retry receives the earlier answer or starts fresh.
Those are different actions with different context and cost.

**Keep credit assignment clear.** If weak fails and strong succeeds, do not give
weak the successful strong answer's quality as its own reward. One design records
and updates each attempted model separately using its own observed outcome.
Another learns the value of an initial choice followed by a fixed fallback chain;
then the reward must include the entire chain's cost and the arms represent
complete strategies. These designs should not be mixed in one learning log.

Retrying changes what the policy observes: it knows that an earlier attempt
failed and has less budget left. That is a sequential extension, beyond the
initial one-call contextual-bandit setup. Begin with one selected model per
question so we can first isolate whether the difficulty features help.

Dropping down a pool is usually most useful on later, similar questions, after
evidence supports the cheaper choice. Once a strong model has already produced
an accepted answer, asking a weaker model to answer the same completed question
does not recover the money already spent.

## 10. What should the reward be?

The reward should describe a **measured result**, not the predicted difficulty.
A candidate objective is:

```text
reward = observed_quality
         - cost_weight * normalized_actual_cost
         - latency_weight * normalized_actual_latency
```

Choose positive normalization budgets/scales and weights on development data.
Record grading overhead and, for a cascade, every answering attempt and check.
If the implementation requires rewards in `[-1, 1]`, clip or scale consistently
and preserve that definition with the controller checkpoint.

The existing adjusted topology experiment uses a different, already implemented
reward: quality minus `0.1 * actual_cost / budget`, minus a budget-overflow penalty,
clipped to `[-1, 1]`; infeasible plans receive `-1`. It currently has no latency
term. See `BAMAS_adjusted/reward.py` (`../../BAMAS_adjusted/reward.py`, broader-workspace reference).

For example, a correct inexpensive weak answer can earn more utility than an
equally correct expensive strong answer. A low predicted difficulty followed by
a failed answer should receive the failed answer's measured result, not a good
reward because the predictor expected success.

Delayed feedback can be applied later using the original context, arm, and a
unique observation ID. Replaying the same observation repeatedly is not new
evidence. Save model/deployment IDs, predictor and index IDs, prompt settings,
and reward definition so changed runtimes are not silently combined.

## 11. How do we tell whether this improves BAMAS?

Use the same independently graded, held-out questions and budgets to compare:

| Comparison | What it tests |
| --- | --- |
| Fixed weak / intermediate / strong | Whether routing adds value over a static choice. |
| Current deterministic router | Whether learned decisions beat our current rule. |
| LinUCB with basic task/budget context or embeddings | Whether difficulty adds information beyond general task features. |
| LinUCB with neural difficulty only | What the trained predictor contributes. |
| LinUCB with neighbor evidence only | What historical retrieval contributes. |
| LinUCB with both signals and support features | Whether their combination helps. |
| Threshold prior enabled / disabled | Whether the proposed initial preferences improve learning. |

Measure answer quality at matched budgets, actual dollars and tokens, total
latency including routing/grading, budget violations, and learning progress over
time. Include exploration costs, multiple random seeds, and success on questions
from domains not used for tuning. A future topology experiment needs real team
outcomes; single-model pool labels do not measure collaboration effects.

Keep final evaluation questions out of predictor fitting, retrieval references,
and bandit training/tuning. When controller-training questions overlap predictor
training, use out-of-fold predictor estimates or a disjoint controller dataset.
Do not feed known difficulty targets into a context that is supposed to contain
predictions available before answering. Frozen offline evaluation also needs
appropriate action coverage or a justified bandit/off-policy evaluator; an
unobserved arm's outcome cannot be guessed from the selected arm's log.

The selected neural predictor's historical pooled test MSE was **0.086935**, versus
**0.101255** for a training-family-mean baseline, a 14.1% reduction. This is not an
answer-accuracy percentage or a live routing improvement. The test split has
been inspected during development, so the result is exploratory. Short arithmetic
and wording sensitivity remain documented weaknesses, including supported cases.

Likewise, our 100-question preview runs establish that scoring and retrieval work.
Their intended difficulty levels are author-assigned coverage tags. With no
generated and independently graded answers, they do not establish bandit learning
or live quality/cost gains. New evidence should determine whether the thresholds,
blend, and learned policy deserve to stay.

## 12. How do I run and inspect it?

### Ask questions interactively

From the repository root in PowerShell:

```powershell
.\.venv-research\Scripts\python.exe -m BAMAS.single_agent interactive --config configs/azure.local.json
```

Wait for `Question>`, enter a question, and press Enter. The predictor and expanded
reference index load once per session. Type `/quit` or `/exit` to finish.
First startup loads the local ML libraries, encoder, prediction heads, and index;
it can take longer than subsequent question processing. Startup does not require
an answering-model request.

The terminal prints the final blended scores, top similar questions and cosine
scores, support checks, cache hits, and timings. Selected pool/model names are
hidden in the display but retained in the full JSONL record. It generates no
answer text, answer tokens, or answer cost. New uncached questions can still make
paid Azure embedding requests.

Refresh the readable exports after adding questions:

```powershell
.\.venv-research\Scripts\python.exe -m BAMAS.single_agent.report
```

| Artifact | Contents |
| --- | --- |
| `BAMAS/single_agent/outputs/routing_previews.jsonl` | Full-precision predictions, all neighbors, selection metadata, timings, and identities. |
| `BAMAS/single_agent/outputs/routing_previews.json` | Readable display fields and the top three neighbors. |
| `BAMAS/single_agent/outputs/routing_previews.txt` | Terminal-style output with run headings. |

Useful fields in the full JSONL:

| Field | Meaning |
| --- | --- |
| `neural_difficulty` | The three trained-predictor estimates. |
| `retrieval.difficulty` | The three neighbor estimates after shrinkage toward training means. |
| `retrieval_weight` | The actual blend weight for each pool. |
| `difficulty` | The final blended estimates shown in the terminal. |
| `neural_input_support` | MPNet-based input-support assessment. |
| `retrieval.neighbors` | Retrieved prompts, similarities, support flags, and reference labels. |
| `within_support` | Whether both support checks passed. |

### Use the neural predictor from Python

```python
from BAMAS.difficulty.model import DifficultyPredictor
from BAMAS.difficulty.inputs import controller_features

predictor = DifficultyPredictor.load(
    "BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model",
    device="auto",
)
features = controller_features(predictor, ["Your question"])[0]
print(features["difficulty"])
```

This API supplies **neural** features. It does not invoke the Azure-neighbor
blend. A future hybrid LinUCB adapter should explicitly consume the single-agent
pipeline's combined record and preserve its predictor/index identity. See the
[controller contract](CONTROLLER_INTEGRATION.md) before constructing features.

## Further reading and implementation map

| File or resource | Use |
| --- | --- |
| [Technical predictor README](../TECHNICAL_README.md) | Training provenance, reproduction commands, and evaluation links. |
| [Training report](../BAMAS/difficulty/reports/all_routerbench_2026-10-24/REPORT.md) | Full source coverage, audits, targets, and results. |
| [Controller integration](CONTROLLER_INTEGRATION.md) | Versioned neural feature API and integration constraints. |
| Single-agent pipeline (`../single_agent/pipeline.py`, broader-workspace reference) | Current blending and deterministic model-selection rule. |
| Neighbor index (`../single_agent/index.py`, broader-workspace reference) | Training-only retrieval, support calibration, and neighbor weighting. |
| Single-agent README (`../single_agent/README.md`, broader-workspace reference) | Embedding setup, full-index expansion, and runnable previews. |
| Adjusted topology controller (`../../BAMAS_adjusted/README.md`, broader-workspace reference) | Existing separate difficulty-aware LinUCB topology experiment. |
| [AgentBalance, Appendix F](https://arxiv.org/html/2512.11426v1#A6) | Supervised difficulty-estimator inspiration; it already uses MPNet and an MLP. |
| [RouterBench](https://arxiv.org/html/2403.12031v1) | Historical model-outcome dataset used for supervision. |
| [LinUCB](https://arxiv.org/abs/1003.0146) | Contextual-bandit learning and exploration. |

The current experiment separates three questions: **How difficult does this task
look? What model should we try? Did that choice actually work?** The neural and
retrieval components address the first. A trained bandit can address the second.
Independent outcomes and measured resource use provide the evidence for the third.
