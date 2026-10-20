# Paper mode: laptop pipeline and reproduction limits

This is the current guide for `controller_method: "paper"`. Start with [readable results](../reports/README.md), [experiment reporting](EXPERIMENTS.md), and [the method comparison](METHOD_COMPARISON.md). The core method is independently implemented; not every experiment or the published numerical results are reproduced.

Reference: [paper v1](https://arxiv.org/html/2512.11426v1), [source at e5aeac4](https://github.com/usail-hkust/AgentBalance_/tree/e5aeac4366fe6296d4a1381ef537508e3af784e9). The pinned source is in `.reference/AgentBalance_-e5aeac4366fe6296d4a1381ef537508e3af784e9`. Prompts are read as static literals; the author project is not executed.

## Laptop and imported datasets

Hardware inspection reports **RTX 4050 Laptop GPU, 6,141 MiB VRAM**, rather than the estimated 3060. You reported 16 GB RAM. PyTorch currently uses CPU only. CPU pretraining works but the full run can take substantial time. For optional GPU training, select Windows/Pip/Python/CUDA in the [official PyTorch installer](https://pytorch.org/get-started/locally/), install its recommended build in `.venv-research`, then verify `torch.cuda.is_available()`. Start with batch size 1 or 2. `--device auto` falls back to CPU.

The laptop runs MPNet, the controller, data processing and Docker grading; large language models use hosted APIs. This machine cannot host the original full backbone roster at the original precision and serving configuration. CUDA does not speed up hosted API calls. Docker is installed, but its Linux engine was stopped during verification.

| Dataset | Train | Eval | Imported source |
| --- | ---: | ---: | --- |
| MMLU | 40 | 153 | Native dev/validation CSVs, reference interleaving and NumPy seed 888 |
| MATH | 40 | 160 | Selected from the complete public 5,000-question test export, seven categories |
| HumanEval | 20 | 144 | Official ordered 164-task file |
| RouterBench | 14,974 usable prompts before difficulty split | Separate holdout | Official zero-shot export converted using 11 correctness columns |

These files and hash manifests already exist in `data/paper`. RouterBench is `data/paper/routerbench-difficulty.jsonl`; the approved original pickle is retained. Conversion excluded 21,512 rows with MMLU/MATH/HumanEval evaluation names. Filtering cannot establish the absence of semantic overlap or model pretraining contamination. Running the commands below does not deserialize pickle again.

**The authors' MATH `sampled_test` artifact and pretrained difficulty checkpoint were not supplied in the source.** Public MATH is an explicit substitute. `prepare --math-source PATH` accepts a native sample if obtained; matching the original ordering still requires its provenance. All public data needed by this pipeline is imported; all exact author artifacts are not.

## Implementations and differences

| Component | New paper-mode implementation | Difference from source or remaining limit |
| --- | --- | --- |
| Difficulty | Separate MPNet encoder; last four layers trainable; 512-unit GELU/dropout ease head; equal BCE/MSE loss; separate learning rates and weight decay | Retrained from public RouterBench, not author weights. Domain grouping is inferred from evaluation names; weighted sampling approximates 1:1:4 rather than reconstructing the unknown author subset. Integration length/subset choices change the experiment. |
| Pools | Pareto filtering, k-medoids on quality/log-cost/log-latency, farthest-member downsampling and overlapping supplementation; measured calibration cost | Deterministic initialization and normalization change clustering. Pipeline pool size is capped at three by default; `pool_max_size` can change it. `explicit_pools` can reproduce known table memberships. Calibration uses training examples rather than original official-performance profiles and 2–4-query calibration. |
| Selection | Ease converted to demand, difficulty offset, smooth buckets and upper pool bound | Three Groq candidates normally produce two pools, possibly fewer after profiling. Four operating points can share pool caps. |
| Matching | Cost/type fusion, performance fusion, role/query/pool context and dot-product compatibility | Independent parameterization; author checkpoints cannot be loaded directly. |
| Representation | Query/role attention over profiles and residual context | Uses equations rather than every source-specific width and normalization. |
| Gating | Binary Concrete sampling with straight-through gradients; minimum two workers, as in the paper | Logistic noise differs from the pinned source's one-sided perturbation. The minimum-worker fallback affects effective actions. Pipeline evaluation samples configurations by default; `--no-sample-configuration` uses deterministic decoding. |
| Topology | Ordered forward worker DAG, shared edge projection, hop simplex and iterative weakest critical-path edge removal | Corrects source repeated-scalar hop logits and one-removal behavior. The original also uses an ordered DAG. |
| Decision | Separate node collects retained workers; configurable decision model, default strong | The source uses a positional pool item in some paths. |
| Optimization | Raw accuracy minus USD/latency penalties, joint discrete likelihood, length penalty, Table 6 settings and separate policy/ease-head rates | Discrete likelihood follows the equations instead of source relaxed-density bookkeeping. Gradient clipping, batching and hard runtime admission are local choices. MPNet itself is frozen during policy training. |
| Prompts/execution | Five source workers plus decision role; source MATH few-shot content; ready-task scheduling | Wrappers/placement are not byte-identical. Contribution/output caps truncate content. Request pacing adds quota waiting to latency. |
| Tools/grading | Docker MATH execution with `answer`; HumanEval public-example checks and parent reuse; hidden-test grader; paper-style MMLU/MATH parsers | Docker/Python environment differs. Hidden tests never reach workers. MATH normalization is not general symbolic equivalence. |
| Metrics | Total USD, mean graph latency, tokens/calls, per-seed concave frontier, interpolation, shared base endpoint, AUC and mean/stdev | Config time is recorded separately. No extrapolation. Free-price runs have undefined cost AUC; strongest fixed star graph is our base endpoint. |
| Comparisons/studies | Cheap/medium/strong/balanced fixed star baselines; ablation switches; explicit transfer command | **AgentPrune, AgentDropout, G-Designer and MasRouter are not implemented.** Their tables and plug-in studies require those projects. Inductive/sensitivity/case-study suites are not automatically orchestrated. They require further experiment code/configuration/model access, not necessarily larger local hardware. |

The core learning/evaluation procedure is feasible locally with hosted inference. Exact published scores, historical latency/prices/service behavior, missing samples and missing pretrained weights cannot be recovered from available code alone. The Groq roster is a substitution for the original Qwen reasoning/non-reasoning candidates and transfer models.

`provider: "openai-compatible"` allows additional hosted endpoints. Supply actual endpoint, model ID, key environment name, supported request fields, prices, type and performance profile. Reasoning modes need distinct profiles and supported provider parameters. `explicit_pools` accepts ordered model-ID lists. Original-roster access is account/configuration work, not a GPU upgrade.

## Short integration run

Run PowerShell at the repository root. The existing `.env` provides Groq model IDs/key. Check model access in your account without printing the key. Only `run` makes live model requests.

```powershell
.\.venv-research\Scripts\Activate.ps1
python -m pip install -e ".[research,routerbench]"
python -m unittest discover -s tests -q
python -m unittest discover -s tests/research -q
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Start Docker Desktop's Linux engine first.
docker info
docker pull python:3.13-slim

# Verifies/reuses the already imported full benchmark splits.
python -m agentbalance.pipeline prepare --out data/paper

python -m agentbalance.pipeline configure --config configs/groq.local.json --out results/paper-laptop-configs --reference-dir .reference/AgentBalance_-e5aeac4366fe6296d4a1381ef537508e3af784e9 --code-grader-image python:3.13-slim --min-request-interval 2

python -m agentbalance.pipeline pretrain --config results/paper-laptop-configs/mmlu.json --difficulty-data data/paper/routerbench-difficulty.jsonl --out checkpoints/paper-laptop-smoke --limit 120 --epochs 1 --batch-size 2 --max-length 128 --device cpu

# data/paper-integration is already prepared: 4 train and 3 eval per benchmark.
python -m agentbalance.pipeline run --configs results/paper-laptop-configs --data data/paper-integration --difficulty-checkpoint checkpoints/paper-laptop-smoke/difficulty.pt --out results/paper-laptop-smoke --benchmarks mmlu math humaneval --seeds 42 --points 2 --epochs 1 --batch-size 2 --code-grader-image python:3.13-slim --allow-live --max-calls 1000 --total-budget 0 --budget 0.05 --latency-budget 600

python -m agentbalance.pipeline summarize --index results/paper-laptop-smoke/index.json --out results/paper-laptop-smoke/summary
Get-Content results/paper-laptop-smoke/summary/runs.csv
```

For a fresh checkout, create the integration splits once after preparation:

```powershell
python -m agentbalance.pipeline subset --data data/paper --out data/paper-integration --train-limit 4 --eval-limit 3
```

This checks data → pretraining → profiling → fresh policy → training → three graders → learned/fixed evaluation → summary. Subset results are not paper results. The existing `checkpoints/paper-pretrain-validation` used only 12 prompts and 64-token inputs; it is a verification artifact, not a full pretrained estimator.

Commands use the zero rates in the current Groq config. **Zero is a configured assumption, not verification of your provider bill or quota.** Cost penalties vanish, and USD AUC is undefined. Accuracy, latency, tokens and calls remain measurable. For USD efficiency, enter actual deployment rates, regenerate configs in a fresh directory, and choose a positive overall allowance. These token-based caps are not provider-side billing limits.

## Full imported data, four points, three seeds

After integration succeeds, omit the difficulty subset limit and use 384-token inputs. `--device auto` can replace CPU after optional CUDA installation.

```powershell
python -m agentbalance.pipeline pretrain --config results/paper-laptop-configs/mmlu.json --difficulty-data data/paper/routerbench-difficulty.jsonl --out checkpoints/paper-laptop-full --epochs 5 --batch-size 2 --max-length 384 --device cpu

python -m agentbalance.pipeline run --configs results/paper-laptop-configs --data data/paper --difficulty-checkpoint checkpoints/paper-laptop-full/difficulty.pt --out results/paper-laptop-full --benchmarks mmlu math humaneval --seeds 42 43 44 --points 1 2 3 4 --epochs 5 --batch-size 4 --code-grader-image python:3.13-slim --allow-live --max-calls 110000 --total-budget 0 --budget 0.05 --latency-budget 600

python -m agentbalance.pipeline summarize --index results/paper-laptop-full/index.json --out results/paper-laptop-full/summary
Get-Content results/paper-laptop-full/summary/runs.csv
Get-Content results/paper-laptop-full/summary/summary.json
```

Five epochs is a local run choice, not an exact reconstruction of the original duration. With six agents per graph, this sweep can reach about **102,108 calls** including profiling/baselines. At two-second spacing that upper-bound workload alone takes about 57 hours, before other delays. Gating/public-test reuse reduce calls; free quotas can stop it much earlier. Build up smaller seed/point/epoch runs in fresh directories. Inspect incomplete outputs before increasing `max_output_tokens` or `context_chars` in a new source config; those increase usage and require fresh training.

Artifacts: global `index.json` ledger, measured profiles, per-point configs, initial/trained checkpoints, `.optimizer.pt` recovery state, rollout/evaluation JSON, `runs.csv` and `summary.json`. Training/evaluation now print compact reports under `summaries/`; `python -m agentbalance report --source RUN_DIRECTORY --out reports/EXPERIMENT` exports an indexed collection with separate training/evaluation folders and policy history. New training records deterministic probes after every update. Default compact tracing omits repeated prompts and intermediate answers; use `--trace-level full` for debugging. P@ values are fractions. `null` means an uncovered budget or zero cost extent. Partial/simulated runs are excluded from research frontiers. Inspect graph status and finish reasons alongside accuracy.

## Recovery and other experiments

### Empty answers after reasoning reaches the completion limit

The first live small MMLU run stopped when GPT-OSS 20B returned an empty answer with `finish_reason: length` and 2,048 completion tokens. Its trained policy was saved; the partial evaluation is not a valid benchmark result. Internal reasoning likely consumed the cap. GPT-OSS supports `reasoning_effort: low`; this controls effort rather than guaranteeing a fixed token count. See [Groq reasoning](https://console.groq.com/docs/reasoning).

A separate local configuration is now prepared in `configs/groq.small.local.json`: it raises the cap to 4,096 and sets low reasoning effort for the GPT-OSS cheap/strong candidates. Generated benchmark configurations are in `results/paper-small-configs-v2`. Existing profiles and checkpoints in `results/paper-small` are retained. Because inference settings changed, profile and train a fresh policy; **reuse the already trained MPNet difficulty checkpoint**. The new settings need a live validation run and can still reach token/quota limits.

```powershell
# Optional: regenerate/reuse the already prepared configurations.
python -m agentbalance.pipeline configure --config configs/groq.small.local.json --out results/paper-small-configs-v2 --reference-dir .reference/AgentBalance_-e5aeac4366fe6296d4a1381ef537508e3af784e9 --code-grader-image python:3.13-slim --min-request-interval 2

# 41 calls were used by the first attempt; 959 preserves the original 1,000-call allowance.
python -m agentbalance.pipeline run --configs results/paper-small-configs-v2 --data data/paper-integration --difficulty-checkpoint checkpoints/paper-small/difficulty.pt --out results/paper-small-v2 --benchmarks mmlu math humaneval --seeds 42 --points 1 2 3 4 --epochs 1 --batch-size 2 --code-grader-image python:3.13-slim --allow-live --max-calls 959 --total-budget 0 --budget 0.05 --latency-budget 600

python -m agentbalance.pipeline summarize --index results/paper-small-v2/index.json --out results/paper-small-v2/summary
```

These reasoning settings change the experiment. Do not combine old/new measurements into one frontier. In the first four-question MMLU profile, cheap and strong both scored 4/4, medium scored 2/4, and cheap was faster. With zero prices this retained only cheap in the worker pool; strong still supplied the decision agent. More profiling examples are needed before interpreting backbone rankings or pool-selection performance.

An unchanged pipeline command reuses completed stages. Failed evaluations can be rerun with the same command: the pipeline archives the partial report and any temporary snapshot as `.failed-001.json` and `.failed-001.json.tmp` (incrementing the number for later attempts), then restarts that evaluation. Completed training/profiling/evaluations are skipped. Previously attempted calls remain in the ledger; only new attempt usage is added, including usage in a newer temporary snapshot if saving failed. A rejected retry that makes no requests adds no calls.

For an incomplete profile, rerun the same pipeline command with `--resume-profiles` appended. Successful model/task measurements are reused; failed, empty or truncated outputs are retried, preserving every original trace row. Only new requests/costs are added to the ledger, and the original global limits still apply. The recovery flag does not change experiment settings; config/data hashes and other settings must still match the saved index. The standalone `research profile --resume-profile` option also reuses measurements, but requires you to keep config, data and scoring settings unchanged; its allowances cover only new requests.

For failed pipeline training, wait for the provider limit to reset, then rerun the same pipeline command with `--resume-training` appended (keep `--resume-profiles` if needed). The pipeline preserves the checkpoint, optimizer, report and rollout files as `.failed-001` artifacts, then restores optimizer, RNG and the saved cursor into a new attempt at the original output path. Later failures use incrementing archive numbers. The ledger retains previous usage and adds only the new attempt; completed stages are still skipped. Both recovery flags are excluded from experiment settings, so they can be appended without changing the run identity.

Standalone training recovery uses `research train --resume-training --checkpoint OLD.pt --out NEW.pt`; keep dataset, seed, batch, reward, budget and learning-rate settings identical. `--epochs` is the total target. Recovery restores optimizer, RNG and last committed cursor; uncommitted batches may repeat already charged requests. Exact controlled recovery and evaluation retries through the research CLI are tested; API output/timing remain nondeterministic.

If Windows reports `WinError 5` when replacing `.json.tmp` with `.json`, the failure happened while saving an artifact. Atomic JSON/checkpoint replacements now retry brief Windows access/sharing locks without repeating API requests. Persistent errors still stop the stage and keep the prior artifact and temporary snapshot. A reader or sync client can temporarily block rename; the application responsible is not established by the error alone. If this keeps happening in the OneDrive workspace, pause sync and close result-file previews while running. Completed stages remain reusable; do not delete the run directory to fix a temporary save failure.

For ablations, set `ablation` in generated benchmark configs to `random-pool`, `random-matching`, `role-only`, `no-gating`, or `no-topology`; run in a fresh directory with aligned seeds/data. For transfer, use `research transfer --source-config ... --config ... --checkpoint ... --out ...` with identical ordered roles/method, then evaluate on held-out data. This rebuilds pools without retraining and does not itself reproduce the paper's inductive study.

Keep checkpoint `.encoder` directories: checkpoints currently reference their absolute local path, so copying only `.pt` to another machine is insufficient.

Locally verified: baseline/research tests, gradients/hop pruning, scheduler behavior/budgets, checkpoint round trips, exact controlled optimizer recovery, parsers/frontiers, and real CPU MPNet fine-tuning on RouterBench. Tensor comparisons confirmed that only transformer layers 8–11 changed. The live MMLU smoke evaluations and the slow MATH smoke evaluation are now complete; MATH scored 3/3 using the Docker code tool. The full multi-benchmark/multi-seed sweep and matching MATH baselines remain incomplete. See the [current reports](../reports/README.md) for measured evidence.
