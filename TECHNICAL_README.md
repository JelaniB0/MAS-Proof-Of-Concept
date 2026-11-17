# Difficulty predictor: technical README

> Standalone branch scope: this checkout contains neural predictors and their
> prepared data/checkpoints. Azure-neighbor routing and LinUCB implementation
> references describe the broader workspace and are not shipped in this branch.

For a plain-language walkthrough of the data, training, embedding blend, and
proposed threshold-guided LinUCB model selection, start with
[Difficulty prediction and LinUCB](docs/DIFFICULTY_AND_LINUCB.md). It includes
worked score examples, escalation/de-escalation, and the distinction between
implemented behavior and the proposed policy.

This package estimates how difficult a question is for three fixed historical
model pools using task text alone. It includes local prediction, data preparation,
training, and evaluation. The design guide above covers integration with
retrieval and LinUCB in the broader BAMAS workspace.

Jump to [architecture](#current-model), [training](#training-pipeline),
[scores and pools](#scores-and-pools), [running predictions](#run-predictions),
[results](#results-and-limitations), [reproduction](#reproduce-the-current-study),
or [source layout](#source-layout).

## Current model

The main predictor is **frozen `sentence-transformers/all-mpnet-base-v2` plus a
three-seed ensemble of MLP heads**, selected as `task_model_sqrt_mse`.

```text
Question -> task-text normalization -> frozen MPNet -> normalized 768-D embedding
         -> three MLP heads (768 -> 128 -> 11), averaged
         -> individual-model difficulty estimates -> three fixed pool averages
```

MPNet itself was not fine-tuned in the selected study. Each MLP uses GELU,
dropout 0.2 and sigmoid outputs. The input limit is 384 tokens with left
truncation. The selected checkpoint is:

```text
BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model
```

The local encoder source was pinned to revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Checkpoints record encoder, tokenizer,
head and pool-contract fingerprints. The separate RAG checkpoint has a different
five-model roster and cannot substitute for the main eleven-model predictor.

## Training pipeline

1. Download pinned RouterBench exports and verify their checksums. Convert the
   zero-shot-named export with the inert pickle reader; no pickle code executes.
2. Normalize actual task text, removing recognized demonstrations and formatting
   footers. Answers, saved responses, dataset names and outcome labels are not
   neural inputs. Export filenames do not guarantee the actual shot condition:
   GSM8K prompts already contain worked examples.
3. Preserve the original GSM8K/MBPP question folds and fixed main model groups.
   Group matching canonical questions and sample IDs into one split. Use five-shot
   outcomes for MMLU, ARC, HellaSwag and Winogrande; zero-shot-named outcomes for
   other main tasks. Merge complementary ARC records only when task text agrees.
4. Regrade archived GSM8K final numeric answers against references using the
   reviewed `gsm-final-numeric-v2` extractor. Other tasks retain published metrics.
   Mask unknown outcomes; never turn them into failures. Preserve the
   [label-review evidence](BAMAS/difficulty/reports/label_audit/README.md).
5. Extract and cache normalized embeddings with frozen MPNet. Embedding extraction
   uses CUDA when requested; the small prediction heads train on CPU.
6. Compare natural-sampling and square-root inverse-family-sampling three-pool
   heads with an eleven-output per-model head. The winning per-model head trains
   on `1 - recorded model quality`, using masked MSE and sampling weights
   proportional to `1 / sqrt(training family size)`.
7. Train each configuration with seeds 42/43/44, AdamW, learning rate 0.001,
   weight decay 0.001, batch size 256, up to 100 epochs, and patience 12. Restore
   each seed's best validation checkpoint. Select the ensemble using validation
   MSE averaged equally across benchmark families.
8. Freeze the selection before test inference. Compare against global and
   family-mean baselines, with paired bootstrap intervals. Fit the input-support
   threshold using training/validation text only; verify saved-model reload parity.
9. Run fixed-model external evaluation separately. No external outcomes are
   invented, and those predictions do not update the model.

| Corpus | Training | Validation | Testing | Total |
| --- | ---: | ---: | ---: | ---: |
| Main RouterBench | 25,710 | 5,406 | 5,381 | 36,497 |
| Separate archived RAG | 445 | 81 | 131 | 657 |

The main collection covers GSM8K, MBPP, MMLU, ARC-Challenge, HellaSwag,
Winogrande, MT-Bench and additional released evaluations. These are question
records, not necessarily complete labels for every model or pool. The main
per-model head has 274,513 observed individual training targets. A pool target
is evaluated only when all of its fixed members have known outcomes.

## Scores and pools

For each pool `G`, `D_G(x) = 1 - mean(recorded quality of models in G)`.
Higher scores mean greater difficulty/quality deficit. With binary correctness,
the target is the observed fraction of models failing. Graded metrics describe
quality deficits rather than binary failure probabilities. Predicted scores are
estimates of these targets, not measured outcomes for a new question.

| Pool | Historical model snapshots |
| --- | --- |
| Weak | Mistral 7B, Llama 2 70B, WizardLM 13B, Yi 34B |
| Intermediate | Code Llama 34B, Mixtral 8x7B, Claude Instant v1, Claude v1 |
| Strong | GPT-3.5 Turbo 1106, Claude v2, GPT-4 1106 Preview |

Exact model IDs are in the checkpoint manifest and the
[full training report](BAMAS/difficulty/reports/all_routerbench_2026-10-24/REPORT.md).
Pool order is not forced to be monotonic on every question. These historical
pools do not automatically describe current BAMAS models, role prompts, token
budgets or collaborating teams.

## Run predictions

From the repository root, after the [quick-start installation](README.md#install-dependencies):

```powershell
.\.venv\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py
.\.venv\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py --show-unsupported
.\.venv\Scripts\python.exe .\BAMAS\scripts\try_difficulty.py --cohort rag
```

The default requires a finalized all-family recommendation. To select an explicit
checkpoint, use `--model PATH`. Device selection defaults to automatic CUDA/CPU.

```powershell
python -m BAMAS.difficulty predict --model BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model --text "Your question" --details
```

Batch inference accepts `--input tasks.jsonl --output predictions.jsonl` with a
`prompt` or `task_description` field. Python callers can use
`DifficultyPredictor.load(...)` and `predict(...)`; see the
[controller interface](docs/CONTROLLER_INTEGRATION.md).

## Results and limitations

The main held-out pooled MSE is **0.086935**, versus **0.101255** for the
training-family-mean baseline, a **14.1% reduction**. This is an exploratory
historical benchmark result; the split has been inspected during development.
Short arithmetic and wording sensitivity remain known failures. The support
check measures input similarity, not confidence or guaranteed accuracy.

| Results | Location |
| --- | --- |
| Full training, splits, baselines and main/RAG results | [All-RouterBench report](BAMAS/difficulty/reports/all_routerbench_2026-10-24/REPORT.md) |
| External transfer and GSM8K/MBPP holdouts | [Generalization report](BAMAS/difficulty/reports/generalization_2026-10-25/REPORT.md) |
| GSM8K regrading provenance and review | [Label audit](BAMAS/difficulty/reports/label_audit/README.md) |
| Main question-level test predictions | `BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model/test_predictions.jsonl` |
| External question-level predictions (generated when rerun) | `BAMAS/difficulty/runs/generalization_test/*_predictions.jsonl` |

External evaluation includes all 1,319 official GSM8K test questions, 5,000 MATH
test problems and 164 HumanEval prompts. There were zero exact normalized-question
matches against the predictor development splits. This does not exclude semantic
near-duplicates or pretrained-model exposure. These external sets lack outcomes
for the fixed model pools; only the historical GSM8K/MBPP holdouts support
pool-outcome error measurements. MATH human levels are a separate ranking proxy.
The external audit does not establish disjointness of BAMAS's MBPP test split.

```powershell
python -m BAMAS.difficulty benchmark --device auto
```

This repeats inference and writes `runs/generalization_test/RESULTS.md` and JSONL
records. It does not retrain the predictor or generate new paid model responses.

## Reproduce the current study

Install the standalone dependencies with
`python -m pip install -r BAMAS/difficulty/requirements.txt` (Python 3.11+).
This standalone branch includes the selected checkpoints and prepared main/RAG
splits through Git LFS. Raw export archives and intermediate caches are excluded.
The GSM8K reference files needed for regrading are included under `BAMAS/data/gsm8k/`.

Download and prepare the inputs from the repository root:

```powershell
python -m BAMAS.difficulty download
python -m BAMAS.difficulty download-routerbench --assets five rag
python -m BAMAS.difficulty convert --input BAMAS/difficulty/data/routerbench_0shot.pkl --output BAMAS/difficulty/data/routerbench_0shot.jsonl
python -m BAMAS.difficulty prepare --input BAMAS/difficulty/data/routerbench_0shot.jsonl --output BAMAS/difficulty/data/prepared_overlap --datasets grade-school-math mbpp
# GSM8K train/test references are already included in this branch.
python -m BAMAS.difficulty regrade-gsm --output BAMAS/difficulty/runs/gsm_correctness_audit_v2
python -m BAMAS.difficulty download-encoder --revision e8c3b32edf5434bc2275fc9bab85f82640a19130
```

The regrader creates `review_sample.jsonl`. Before all-family preparation it needs
an accompanying `manual_review.json` that records the extraction review. The
retained review in `reports/label_audit/` can be reused only when the regenerated
sample SHA256 matches its `sample_sha256` and the source/reference/split hashes
match `regrade_protocol.json`; otherwise review the regenerated sample. Do not
reuse a review decision for changed inputs.

With the reviewed audit in place:

```powershell
python -m BAMAS.difficulty all-routerbench --output BAMAS/difficulty/runs/all_routerbench_v1 --device auto
python -m BAMAS.difficulty finalize-all-routerbench --output BAMAS/difficulty/runs/all_routerbench_v1 --device auto
python -m BAMAS.difficulty download-evaluation-data
python -m BAMAS.difficulty benchmark --device auto
```

Existing compatible heads and fingerprinted embedding caches are reused. Changed
protocols require a new output directory. `all-routerbench --prepare-only` checks
coverage without training. The optional 1.20 GB raw RouterBench audit export is
available with `download-routerbench --assets raw`; it is not needed for training.

## Source layout

The Python modules, `reports/`, and `runs/` below are under `BAMAS/difficulty/`.
The `docs/` directory is at the repository root.

- `all_routerbench.py`, `research.py`, `frozen.py`: current preparation, head
  training, validation selection and frozen-encoder inference.
- `data.py`, `safe_pickle.py`, `gsm_correctness.py`, `audit.py`: data extraction,
  splits, targets and label audits.
- `model.py`, `transformer.py`: shared API, metrics/tokenization and retained
  TF-IDF/partial-fine-tuning baselines; the active pipeline still uses these modules.
- `inputs.py`, `support.py`, `probes.py`: input/feature contract, support checks
  and explicitly previously inspected development probes.
- `benchmark.py`, `evaluation.py`, asset modules: evaluation and verified downloads.
- `reports/`: study summaries, external evaluation, and label-review evidence;
  [report index](BAMAS/difficulty/reports/README.md).
- `runs/all_routerbench_v1/`: canonical training protocols, audit JSON, prepared
  data, recommendations, selected checkpoints, and saved test predictions.
- `docs/`: the readable difficulty/LinUCB guide and controller API contract.

The release keeps training, inference, and evaluation together. Shared baseline
modules remain dependencies of the active predictor. Duplicate report copies,
historical cleanup logs, the unrelated BAMAS review, and the redundant
question-enriched prediction export were removed. Checkpoints and prepared data
are unchanged; the preparation protocol lives at
`BAMAS/difficulty/runs/all_routerbench_v1/prepared/preparation_protocol.json`.

Run the offline software checks with:

```powershell
python -m unittest discover -s BAMAS/difficulty -t . -p "test*.py" -v
```

These tests validate pipeline behavior, not predictive accuracy. Current release
verification is recorded in [RELEASE_VERIFICATION.json](RELEASE_VERIFICATION.json).
