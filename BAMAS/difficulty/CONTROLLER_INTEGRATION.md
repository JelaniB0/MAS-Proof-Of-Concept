# Contract for BAMAS integration

> Standalone branch scope: this checkout contains neural predictors and their
> prepared data/checkpoints. Azure-neighbor routing and LinUCB implementation
> references describe the broader workspace and are not shipped in this branch.

The difficulty predictor does not change the original BAMAS RL policy, ILP
objective, agents or executor. A separate `BAMAS_adjusted` adapter already uses
its neural features for LinUCB topology selection. The single-agent Azure-neighbor
blend is not connected to a model-pool LinUCB policy yet. For the plain-language
design, including proposed starting thresholds and feedback, see
[Difficulty prediction and LinUCB](DIFFICULTY_AND_LINUCB_README.md).

## Current checkpoint and API

The main predictor is a frozen MPNet encoder and an ensemble of three eleven-output
MLP heads. Each head predicts individual historical-model difficulties; pool
scores are averages over the fixed 4/4/3 roster. The verified default is
`runs/all_routerbench_v1/main/selected_model`. The RAG checkpoint is a separate
five-model contract at `runs/all_routerbench_v1/rag/selected_model`.

```python
from BAMAS.difficulty.model import DifficultyPredictor
from BAMAS.difficulty.inputs import controller_features

predictor = DifficultyPredictor.load(
    "BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model",
    device="auto",
)
scores = predictor.predict(["Your question"])[0]
features = controller_features(predictor, ["Your question"])[0]
```

`features` contains:

| Field | Meaning |
| --- | --- |
| `difficulty` | Predicted weak/intermediate/strong quality deficits |
| `difficulty_vector` | Those three scores in explicit `group_order` |
| `estimated_quality` | `1 - predicted difficulty` per pool |
| `estimated_quality_gains` | Signed differences for upgrades between pools |
| `model_difficulty`, `estimated_model_quality` | Per-model estimates for the selected eleven-output head |
| `pool_contract_id`, `predictor_id`, `cohort` | Target/pool, checkpoint and cohort identity |
| `input_mode`, `target_definition` | Input processing and target semantics |
| `input_support` | Text-similarity assessment; not confidence |

The same `routerbench-task-v1` processing runs during training and inference.
It removes recognized demonstrations and formatting footers while retaining task
content and answer options. Only task text enters the predictor. Never include
reference answers, hidden tests, observed outcomes or dataset labels as features.

A consumer can pass `expected_contract=...` to `controller_features` to reject a
pool mismatch. Preserve the predictor identity with cached features and rollouts.
The contract detects roster/target differences; it does not establish that live
BAMAS models perform like the historical RouterBench roster.

## Interpretation and support

The training target is `D_G(x) = 1 - mean(q_m(x), m in G)`. GSM8K uses reviewed
numeric-answer correctness; other tasks retain published binary/graded metrics.
A pool target is available only when every member has a known outcome. Scores
on graded tasks describe quality deficits rather than failure probabilities.
Individual-model predictions need their own calibration before model selection.

The input-support threshold is the validation 95th percentile of nearest-training
cosine distance. Define a fallback for unsupported inputs. Passing the threshold
does not establish accuracy; raw short-arithmetic scores remain unreliable.
No score-order clamp, arithmetic shortcut or automatic pool recommendation is
applied. Quality gains remain signed.

## RL and ILP experiments

A candidate RL state can concatenate the task embedding, all three predicted
difficulties, remaining budget and execution/history features. Compare the
three-coordinate vector with scalar difficulty, constant/family-mean features,
flat per-model prediction and a controller without difficulty features.

Keep final BAMAS evaluation questions out of predictor fitting/calibration and
controller fitting/tuning. Audit actual task text across both components; separate
split names or dataset IDs are insufficient. For controller training questions
that also trained the predictor, use out-of-fold predictions or disjoint data.
Never provide observed correctness-derived difficulty targets as deployment state.
Already running inference on a question does not update the predictor, but tuning
against its evaluation results makes that set development evidence.

For an ILP, predicted utilities may be fixed coefficients on binary
`z[role, model, token_bin]` choices under measured cost and role/topology
constraints. Pool averages alone do not distinguish the best model, role or token
allocation. Measure those conditional utilities before claiming team-success or
marginal-token estimates. See [CODE_REVIEW.md](CODE_REVIEW.md) for existing planner
preferences that an added feature alone will not change.

Collect graded outcomes for the intended BAMAS models, prompts, roles and token
caps, with repeated trials as needed and actual tokens, latency and costs. Compare
end-to-end success versus cost, budget violations and predictor overhead at equal
budgets. Single-model historical outcomes do not measure multi-agent interactions.

For current training methods and results, see the [README](README.md),
[all-family report](reports/all_routerbench_2026-10-24/REPORT.md) and
[external evaluation](reports/generalization_2026-10-25/REPORT.md).
