# Difficulty predictor

The predictor is contained in **[`difficulty-predictor/`](difficulty-predictor/)**.

| Documentation | Purpose |
| --- | --- |
| **[Technical README](difficulty-predictor/TECHNICAL_README.md)** | Architecture, data, training, evaluation, and reproduction. |
| [Quick start](difficulty-predictor/README.md) | Install and ask questions interactively. |
| [Difficulty and LinUCB guide](difficulty-predictor/docs/DIFFICULTY_AND_LINUCB.md) | Score meaning, examples, and proposed model-pool selection. |
| [Controller API](difficulty-predictor/docs/CONTROLLER_INTEGRATION.md) | Python features and compatibility checks. |
| [Integration into another branch](difficulty-predictor/docs/INTEGRATION.md) | Import the component without replacing the receiving branch. |
| [Datasets](difficulty-predictor/DATASETS.md) | Prepared splits, provenance, and upstream terms. |

The folder name uses a hyphen; its Python package is `difficulty`. Run CLI and
tests from inside `difficulty-predictor/`. Selected checkpoints and datasets use
Git LFS. The component contains local prediction, training, evaluation, and tests.
Retrieval and LinUCB integration are described as design work.
