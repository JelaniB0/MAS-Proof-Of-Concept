# BAMAS: Budget-Aware Multi-Agent Systems

This repository contains the implementation for the AAAI-26 paper **"BAMAS: Structuring Budget-Aware Multi-Agent Systems"**.
Yang, L., Luo, J., Liu, X., Lou, Y., & Chen, Z. (2026). 
BAMAS: Structuring Budget-Aware Multi-Agent Systems. 
Proceedings of the AAAI Conference on Artificial Intelligence, 40(35), 29802-29810. https://doi.org/10.1609/aaai.v40i35.40226

## Overview

BAMAS is a framework for constructing budget-aware multi-agent systems that balance performance and cost. The system selects both LLM configurations and collaboration topologies under a user-specified budget constraint.

## Project Structure

```text
BAMAS/
|-- eapae_agent_sys/          # Core system implementation
|   |-- agents/               # Agent implementations and configurations
|   |-- planning/             # Resource planning and topology selection
|   |-- execution/            # Task execution engine
|   |-- utils/                # Utility functions and API wrappers
|   `-- data_processing/      # Dataset loaders and preprocessing
|-- configs/                  # Configuration files
|-- scripts/                  # Dataset download and cache generation scripts
|-- experiments/              # Experiment entry points
|-- main_train_offline.py     # Offline RL training entry point
`-- data/                     # Generated benchmark data and caches (not tracked)
```

## Dataset Preparation

The repository does **not** include benchmark data or generated offline RL caches. Those files are expected to be created locally.

Raw benchmark datasets can be downloaded into the repository layout with:

```bash
pip install -r requirements.txt
python scripts/download_real_datasets.py --datasets gsm8k mbpp math
```

The script downloads from the following public sources:

| Dataset | Public source | Local files created |
| --- | --- | --- |
| GSM8K | Hugging Face `openai/gsm8k` | `data/gsm8k/train.jsonl`, `data/gsm8k/test.jsonl` |
| MBPP | Hugging Face `google-research-datasets/mbpp` | `data/mbpp/mbpp.jsonl` and processed split files under `data/processed/mbpp/` |
| MATH | Hugging Face `EleutherAI/hendrycks_math` | `data/math/train/`, `data/math/test/`, and sampled subsets under `data/processed/math/` |

Detailed paths, cache-generation commands, and batch-generation notes are documented in [docs/datasets.md](docs/datasets.md).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create `configs/secrets.yml` before generating offline RL data. Raw benchmark download does not require API keys, but cache generation does.

```yaml
api_keys:
  deepseek: "YOUR_DEEPSEEK_API_KEY"
  openai: "YOUR_OPENAI_COMPATIBLE_API_KEY"
api_bases:
  openai: "https://your-openai-compatible-base/v1"
```

### 3. Download the benchmark datasets

```bash
python scripts/download_real_datasets.py --datasets gsm8k mbpp math
```

### 4. Generate the offline RL dataset

```bash
# GSM8K
python scripts/generate_training_cache.py --num_samples -1 --num_budget_steps 5 --num_workers 16

# MATH
python scripts/generate_math_training_cache.py --num_samples -1 --num_budget_steps 3 --num_workers 24

# MBPP
python scripts/generate_mbpp_training_cache.py --num_samples -1 --num_budget_steps 5 --num_workers 16
```

This creates the dataset files expected by offline training:

- `data/processed/offline_rl_dataset.jsonl`
- `data/processed/math/offline_rl_dataset.jsonl`
- `data/processed/mbpp/offline_rl_dataset.jsonl`

### 5. Train the topology selection policy

```bash
python main_train_offline.py --dataset_name gsm8k
python main_train_offline.py --dataset_name math
python main_train_offline.py --dataset_name mbpp
```

### 6. Run evaluation

```bash
python experiments/main_experiment.py
python experiments/main_experiment_math.py
python experiments/main_experiment_mbpp.py
```

## Supported Datasets

- **GSM8K**: Grade school math word problems
- **MATH**: Competition-style mathematical reasoning problems
- **MBPP**: Python code generation tasks

## Key Features

- **Budget-aware optimization**: Automatically balances performance and cost
- **Flexible topologies**: Supports linear, star, feedback, and planner-driven collaboration patterns
- **Modular design**: Easy to extend with new agents and topologies
- **Multi-dataset support**: Works across different task domains

## Citation

If you use this code in your research, please cite our paper:

To be released

## License

This project is licensed under the terms specified in the LICENSE file.
