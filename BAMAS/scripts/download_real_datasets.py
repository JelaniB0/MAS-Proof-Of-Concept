import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eapae_agent_sys.data_processing.prepare_gsm8k_dataset import main as prepare_gsm8k_curriculum
from eapae_agent_sys.data_processing.sample_math_subset import create_math_subset


DATASET_SOURCES = {
    "gsm8k": "Hugging Face dataset openai/gsm8k",
    "mbpp": "Hugging Face dataset google-research-datasets/mbpp",
    "math": "Hugging Face dataset EleutherAI/hendrycks_math",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_gsm8k() -> None:
    print(f"Downloading GSM8K from {DATASET_SOURCES['gsm8k']}")
    dataset = load_dataset("openai/gsm8k", "main")
    train_rows = [{"question": row["question"], "answer": row["answer"]} for row in dataset["train"]]
    test_rows = [{"question": row["question"], "answer": row["answer"]} for row in dataset["test"]]

    train_path = ROOT / "data" / "gsm8k" / "train.jsonl"
    test_path = ROOT / "data" / "gsm8k" / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    # Optional processed curriculum file used by some utilities.
    prepare_gsm8k_curriculum()


def download_mbpp() -> None:
    print(f"Downloading MBPP from {DATASET_SOURCES['mbpp']}")
    dataset = load_dataset("google-research-datasets/mbpp")
    split_to_filename = {
        "train": "train.jsonl",
        "test": "test.jsonl",
        "validation": "val.jsonl",
        "prompt": "few_shot.jsonl",
    }

    raw_rows = []
    for split_name in ("prompt", "test", "validation", "train"):
        split_rows = []
        for row in dataset[split_name]:
            clean_row = {
                "task_id": row["task_id"],
                "text": row["text"],
                "code": row["code"],
                "test_list": row["test_list"],
                "test_setup_code": row["test_setup_code"],
                "challenge_test_list": row["challenge_test_list"],
            }
            raw_rows.append(clean_row)
            split_rows.append(clean_row)
        write_jsonl(ROOT / "data" / "processed" / "mbpp" / split_to_filename[split_name], split_rows)

    raw_rows.sort(key=lambda item: item["task_id"])
    write_jsonl(ROOT / "data" / "mbpp" / "mbpp.jsonl", raw_rows)


def download_math() -> None:
    print(f"Downloading MATH from {DATASET_SOURCES['math']}")
    train_root = ROOT / "data" / "math" / "train"
    test_root = ROOT / "data" / "math" / "test"
    reset_dir(train_root)
    reset_dir(test_root)

    for config_name in get_dataset_config_names("EleutherAI/hendrycks_math"):
        for split_name in ("train", "test"):
            split_dir = (train_root if split_name == "train" else test_root) / config_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_dataset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split_name)
            for idx, row in enumerate(split_dataset):
                row_to_save = dict(row)
                if not row_to_save.get("type"):
                    row_to_save["type"] = config_name
                output_path = split_dir / f"{idx:05d}.json"
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(row_to_save, handle, ensure_ascii=False)

    create_math_subset()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BAMAS benchmark datasets into the repository layout.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "gsm8k", "mbpp", "math"],
        help="Which datasets to download.",
    )
    args = parser.parse_args()

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"gsm8k", "mbpp", "math"}

    os.chdir(ROOT)
    if "gsm8k" in selected:
        download_gsm8k()
    if "mbpp" in selected:
        download_mbpp()
    if "math" in selected:
        download_math()

    print("Dataset download and local preprocessing complete.")


if __name__ == "__main__":
    main()
