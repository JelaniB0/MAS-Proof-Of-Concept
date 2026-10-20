# AgentBalance research prototype

A local controller assigns hosted language models to agent roles and builds a
communication graph for each question. Training updates the controller; hosted
models remain inference services. This independently implements the main
AgentBalance v1 method with documented experimental differences.

Start here:

- [Readable live results](reports/README.md): training, evaluation and policy history.
- [Paper versus this implementation](docs/METHOD_COMPARISON.md): the four requested comparisons.
- [Experiments and logging](docs/EXPERIMENTS.md): setup, reports and next steps.
- [Paper pipeline](docs/PAPER_PIPELINE.md): dataset provenance, pretraining and recovery.
- [Implementation reference](docs/IMPLEMENTATION.md): module inventory and details.
- [Repository layout](docs/REPOSITORY.md): where files belong and what to commit.

## Current measured result

The MATH smoke evaluation answered **3/3 held-out questions correctly**, with
**18 requests** and **26,740 input/output tokens**. Recorded query time was about
**18.5 minutes**, including 65-second request spacing. All five workers and the
decision agent used `strong`. Training used four questions, one epoch and three
parameter updates. This is integration evidence; no matched MATH baseline has
established accuracy or efficiency gains.

Open the [evaluation](reports/paper-small-v2/evaluation/math-eval-slow-01/README.md)
and [training](reports/paper-small-v2/training/math-s42-p1/README.md) reports.
Zero configured prices cannot demonstrate dollar savings. Partial evaluations
are labeled in the index.

## Setup

Python 3.11+ is required. Fixed baselines and report export use the standard library;
training requires the optional research dependencies. For a fresh setup:

```powershell
python -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install -e ".[research,routerbench]"
Copy-Item .env.example .env
```

Reuse existing environments and local settings when present. Fill provider fields
in `.env`; review a config copied from `configs/groq.example.json` or
`configs/azure.example.json`. Docker Desktop's Linux engine is required for
paper-mode MATH code tools and HumanEval grading.

## Read results without model calls

```powershell
.\.venv-research\Scripts\python.exe -m agentbalance report `
  --source results/paper-small-v2 --out reports/paper-small-v2
```

A single training/evaluation JSON also works as `--source`. Open the generated
`README.md` first. Curated reports are included in Git; raw datasets, checkpoints
and execution logs remain local artifacts.

## Interactive use

```powershell
.\.venv-research\Scripts\python.exe -m agentbalance --provider groq --policy balanced --topology star
```

Learned mode requires its compatible trained checkpoint and benchmark configuration:

```powershell
.\.venv-research\Scripts\python.exe -m agentbalance `
  --config results/paper-small-v2/math-s42-p1.slow-eval.json `
  --policy learned --topology adaptive `
  --checkpoint results/paper-small-v2/math-s42-p1.pt `
  --budget 0.05 --latency-budget 600
```

Interactive queries make live requests. Use `research train` and
`research evaluate` for controlled experiments with overall request/USD allowances.

## Validation

```powershell
.\.venv-research\Scripts\python.exe -m unittest discover -s tests -q
.\.venv-research\Scripts\python.exe -m unittest discover -s tests/research -q
```

Tests use fixtures and controlled providers; they verify implementation behavior,
not live benchmark performance.
