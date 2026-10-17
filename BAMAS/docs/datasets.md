# Dataset And Offline Cache Preparation

This document explains where the BAMAS training data comes from and how to generate the `offline_rl_dataset.jsonl` files used by offline training.

## What Is Tracked In Git

The repository does not store:

- Raw benchmark datasets under `data/`
- Generated offline RL datasets under `data/processed/`
- Training outputs under `outputs/` or experiment summaries under `experiments/results/`

These files are generated locally because they are large and, for offline RL caches, depend on the configured model providers and budget settings.

## Public Dataset Sources

| Dataset | Public source | Download command | Main local outputs |
| --- | --- | --- | --- |
| GSM8K | Hugging Face `openai/gsm8k` | `python scripts/download_real_datasets.py --datasets gsm8k` | `data/gsm8k/train.jsonl`, `data/gsm8k/test.jsonl` |
| MBPP | Hugging Face `google-research-datasets/mbpp` | `python scripts/download_real_datasets.py --datasets mbpp` | `data/mbpp/mbpp.jsonl`, processed splits under `data/processed/mbpp/` |
| MATH | Hugging Face `EleutherAI/hendrycks_math` | `python scripts/download_real_datasets.py --datasets math` | `data/math/train/`, `data/math/test/`, sampled subsets under `data/processed/math/` |

To download all supported datasets in one step:

```bash
python scripts/download_real_datasets.py --datasets gsm8k mbpp math
```

## Secrets File

Raw dataset download does not require API keys.

Offline RL cache generation does require a `configs/secrets.yml` file because each sample is executed through the configured LLM providers. A minimal example is:

```yaml
api_keys:
  deepseek: "YOUR_DEEPSEEK_API_KEY"
  openai: "YOUR_OPENAI_COMPATIBLE_API_KEY"
api_bases:
  openai: "https://your-openai-compatible-base/v1"
```

## Offline RL Dataset Generation

After downloading the raw benchmarks and configuring `configs/secrets.yml`, generate the offline RL datasets with the dataset-specific cache scripts:

```bash
# GSM8K
python scripts/generate_training_cache.py --num_samples -1 --num_budget_steps 5 --num_workers 16

# MATH
python scripts/generate_math_training_cache.py --num_samples -1 --num_budget_steps 3 --num_workers 24

# MBPP
python scripts/generate_mbpp_training_cache.py --num_samples -1 --num_budget_steps 5 --num_workers 16
```

The canonical output files used by `main_train_offline.py` are:

- `data/processed/offline_rl_dataset.jsonl`
- `data/processed/math/offline_rl_dataset.jsonl`
- `data/processed/mbpp/offline_rl_dataset.jsonl`

## Batch Generation

If full cache generation is too expensive to run in one shot, the cache scripts can generate batches with `--begin_with` and `--num_samples`. In that case the output filenames include a batch suffix such as:

- `data/processed/offline_rl_dataset_batch_0_499.jsonl`
- `data/processed/math/offline_rl_dataset_batch_0_999.jsonl`
- `data/processed/mbpp/offline_rl_dataset_batch_0_499.jsonl`

Batch files can be merged with:

```bash
python scripts/merge_batch_datasets.py --input_pattern "data/processed/offline_rl_dataset_batch_*.jsonl" --output "data/processed/offline_rl_dataset.jsonl" --verify
python scripts/merge_batch_datasets.py --input_pattern "data/processed/math/offline_rl_dataset_batch_*.jsonl" --output "data/processed/math/offline_rl_dataset.jsonl" --verify
python scripts/merge_batch_datasets.py --input_pattern "data/processed/mbpp/offline_rl_dataset_batch_*.jsonl" --output "data/processed/mbpp/offline_rl_dataset.jsonl" --verify
```

## Training Commands

Once the canonical offline datasets exist, offline training can be launched with:

```bash
python main_train_offline.py --dataset_name gsm8k
python main_train_offline.py --dataset_name math
python main_train_offline.py --dataset_name mbpp
```

## Notes

- `scripts/download_real_datasets.py` also creates processed helper files used by the current loaders, such as the GSM8K curriculum file and the sampled MATH subsets.
- The provided MATH preparation step writes the full raw dataset to `data/math/` and also builds deterministic `1000`-sample train and test subsets under `data/processed/math/`, which are used by the default MATH cache-generation script.
