# Repository layout

| Location | Purpose | Commit? |
| --- | --- | --- |
| `agentbalance/`, `tests/` | Runtime, controllers, pipeline, reporting and tests | Yes |
| `configs/*.example.json`, `.env.example` | Portable configuration templates | Yes |
| `docs/`, `examples/` | Current guides and runnable examples | Yes |
| `reports/` | Curated experiment summaries and policy history | Yes, after review |
| `results/` | Raw runs, generated configs, traces and recovery state | No |
| `checkpoints/`, `data/` | Local weights, encoders, datasets and manifests | No |
| `.env`, `*.local.json` | Credentials and local settings | No |
| `.reference/` | Pinned upstream source used to load prompt templates | No |
| `.venv-research/`, `*.egg-info/` | Installed development environment | No |

Start with [readable results](../reports/README.md), the
[paper pipeline](PAPER_PIPELINE.md), and [experiment reporting](EXPERIMENTS.md).
Use `results/runs/<experiment>/` for new raw runs. Existing experiments retain
their paths because checkpoints embed encoder paths and ledgers reference
recovery artifacts. Keep optimizer sidecars and failed attempts for recovery
and provenance.

Superseded guides, the previous README copy, the historical cleanup manifest,
the generated `source-tree.json` snapshot and Python bytecode caches were removed.
Runtime code, tests, reference prompts, environments and experiment artifacts
are retained. Historical raw outputs remain ignored under `results/`.

Git ignores credentials, local configs, raw artifacts and caches. Curated reports
omit repeated prompts and agent reasoning. Review custom data before publishing.
