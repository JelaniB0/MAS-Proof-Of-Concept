"""Normalize RouterBench tables and create leakage-aware, pool-relative targets."""

from collections import defaultdict
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

GROUPS = ("weak", "intermediate", "strong")


def prompt_text(value):
    if isinstance(value, str):
        # RouterBench exports may contain JSON-encoded chat messages.
        if value.lstrip().startswith("["):
            try:
                return prompt_text(json.loads(value))
            except (ValueError, TypeError):
                try:
                    return prompt_text(ast.literal_eval(value))
                except (ValueError, TypeError, SyntaxError):
                    pass
        return value.strip()
    if isinstance(value, (list, tuple, np.ndarray)):
        return "\n".join(prompt_text(item) for item in value).strip()
    if isinstance(value, dict) and "content" in value:
        return prompt_text(value["content"])
    raise ValueError("Prompt must be text or a sequence of chat messages")


def score(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        value = value.lower() == "true"
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Quality must be finite and in [0, 1], got {value!r}")
    return number


def read_table(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Use JSONL, CSV, or Parquet. Convert RouterBench pickle with safe_pickle.py first.")


def normalize(frame):
    """Accept wide RouterBench scores or long (sample_id, model_name, performance) rows.

    Repeated long-form observations are averaged per model first, so models with
    more trials do not receive more weight in the group target.
    """
    required = {"sample_id", "eval_name", "prompt"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    long = "model_name" in frame.columns
    quality_column = next((c for c in ("performance", "is_correct") if c in frame.columns), None)
    if long and quality_column is None:
        raise ValueError("Long format needs performance or is_correct")
    metadata = required | {"model_name", "performance", "is_correct", "response", "cost",
                           "total_cost", "embedding", "oracle", "oracle_model_to_route_to",
                           "no_model_correct", "split", "source_split"}
    # Only bare model columns are targets; responses, costs and gold answers are excluded.
    wide_models = [c for c in frame.columns if c not in metadata and "|" not in c
                   and c not in {"gold_response", "true_label", "model_response"}]
    samples = {}
    for row in frame.to_dict("records"):
        if pd.isna(row["sample_id"]) or pd.isna(row["eval_name"]):
            raise ValueError("sample_id and eval_name cannot be missing")
        key = (str(row["eval_name"]), str(row["sample_id"]))
        text = prompt_text(row["prompt"])
        if not text:
            raise ValueError(f"Empty prompt for {key}")
        if key not in samples:
            samples[key] = {"sample_id": key[1], "eval_name": key[0], "prompt": text,
                            "observations": defaultdict(list)}
        sample = samples[key]
        if text != sample["prompt"]:
            raise ValueError(f"Conflicting prompts for {key}; do not mix shot settings")
        if long:
            pairs = [(str(row["model_name"]), row[quality_column])]
        else:
            if sample["observations"]:
                raise ValueError(f"Duplicate wide sample: {key}")
            pairs = [(model, row[model]) for model in wide_models]
        for model, value in pairs:
            quality = score(value)
            if quality is not None:
                sample["observations"][model].append(quality)
    result = []
    for sample in samples.values():
        observations = sample.pop("observations")
        sample["scores"] = {m: sum(values) / len(values) for m, values in observations.items()}
        sample["trials"] = {m: len(values) for m, values in observations.items()}
        if sample["scores"]:
            result.append(sample)
    if not result:
        raise ValueError("No observed model scores")
    return sorted(result, key=lambda r: (r["eval_name"], r["sample_id"]))


def assign_splits(rows, seed=42, validation_fraction=0.15, test_fraction=0.15):
    """Hash connected question identities so duplicate IDs OR prompts stay together.

    Dataset names are deliberately omitted from ID links: exports may rename evals.
    Fractions are approximate, deterministic and independent of model outcomes.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    identities = {}
    for i, row in enumerate(rows):
        normalized = re.sub(r"\s+", " ", row["prompt"]).strip()
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        row["prompt_hash"] = fingerprint
        for identity in ("id:" + row["sample_id"], "prompt:" + fingerprint):
            if identity in identities:
                a, b = root(i), root(identities[identity])
                parent[max(a, b)] = min(a, b)
            identities[identity] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row["prompt_hash"])
    for i, row in enumerate(rows):
        key = min(components[root(i)])
        value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
        row["split"] = "test" if value < test_fraction else (
            "validation" if value < test_fraction + validation_fraction else "train")
    if set(r["split"] for r in rows) != {"train", "validation", "test"}:
        raise ValueError("Not enough independent questions for all three splits")
    return rows


def rank_groups(rows):
    """Macro-average quality across datasets, using training questions only."""
    by_model = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] == "train":
            for model, value in row["scores"].items():
                by_model[model][row["eval_name"]].append(value)
    datasets = {r["eval_name"] for r in rows if r["split"] == "train"}
    if len(by_model) < 3:
        raise ValueError("Need at least three models to build three pools")
    if any(set(by_eval) != datasets for by_eval in by_model.values()):
        raise ValueError("Automatic ranking needs each model on every training dataset; use explicit groups")
    ranking = {model: float(np.mean([np.mean(v) for v in by_eval.values()]))
               for model, by_eval in by_model.items()}
    ordered = sorted(ranking, key=lambda model: (ranking[model], model))
    groups = {name: list(chunk) for name, chunk in zip(GROUPS, np.array_split(ordered, 3))}
    return groups, ranking


def validate_groups(groups, models):
    if set(groups) != set(GROUPS):
        raise ValueError(f"Group names must be {GROUPS}")
    members = [m for name in GROUPS for m in groups[name]]
    if any(not isinstance(groups[k], list) or not groups[k] for k in GROUPS):
        raise ValueError("Each group must be a nonempty list")
    if len(members) != len(set(members)):
        raise ValueError("Model groups must be disjoint")
    if set(members) != set(models):
        raise ValueError(f"Groups must partition observed models; missing={set(models)-set(members)}, unknown={set(members)-set(models)}")


def add_targets(rows, groups, min_coverage=1.0):
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    for row in rows:
        row["difficulty"] = {}
        row["coverage"] = {}
        for group in GROUPS:
            values = [row["scores"][model] for model in groups[group] if model in row["scores"]]
            row["coverage"][group] = len(values)
            row["difficulty"][group] = (
                1 - sum(values) / len(values)
                if len(values) / len(groups[group]) >= min_coverage else None
            )
    return rows


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(source, destination, groups_path=None, seed=42, min_coverage=1.0, datasets=None):
    frame = read_table(source)
    if datasets:
        unknown = set(datasets) - set(frame["eval_name"])
        if unknown:
            raise ValueError(f"Datasets not found: {sorted(unknown)}")
        frame = frame[frame["eval_name"].isin(datasets)]
    rows = assign_splits(normalize(frame), seed=seed)
    models = sorted({model for row in rows for model in row["scores"]})
    if groups_path:
        groups = json.loads(Path(groups_path).read_text(encoding="utf-8"))
        ranking = None
    else:
        groups, ranking = rank_groups(rows)
    validate_groups(groups, models)
    add_targets(rows, groups, min_coverage)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "seed": seed, "group_order": list(GROUPS), "groups": groups,
                "training_macro_quality": ranking, "min_coverage": min_coverage,
                "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
                "datasets": sorted({r["eval_name"] for r in rows}),
                "split_method": "hash of connected sample IDs / normalized full prompts; 70/15/15"}
    write_json(destination / "manifest.json", manifest)
    return manifest
