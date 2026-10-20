"""Convert RouterBench's processed wide outcomes into local difficulty targets."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

from .data import task_key


def prompt_text(value):
    if isinstance(value, str):
        # Some exports serialize chat messages as a Python literal, not JSON.
        if value.lstrip().startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                value = parsed
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(value, list) and value:
        if all(isinstance(v, str) for v in value):
            text = "\n".join(value)
        elif all(isinstance(v, dict) and isinstance(v.get("content"), str) for v in value):
            text = "\n\n".join(f"{v.get('role', 'user')}: {v['content']}" for v in value)
        else:
            raise ValueError("Unsupported RouterBench prompt message format")
        if text.strip():
            return text.strip()
    raise ValueError("RouterBench prompt must contain nonempty text")


def difficulty_rows(records, score_columns, exclude_eval=()):
    grouped = {}
    excluded = 0
    for row in records:
        dataset = str(row.get("eval_name", "")).casefold()
        if exclude_eval and not dataset:
            raise ValueError("Cannot exclude evaluation datasets: eval_name is missing")
        if any(part.casefold() in dataset for part in exclude_eval):
            excluded += 1
            continue
        task = prompt_text(row["prompt"])
        entry = grouped.setdefault(task_key(task), {"task": task, "scores": {}, "eval_names": set()})
        entry["eval_names"].add(dataset)
        for model in score_columns:
            raw = row[model]
            if raw is None or raw == "":
                continue
            score = float(raw)
            if math.isnan(score):
                continue
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"RouterBench correctness column {model} must contain scores in [0, 1]")
            total, count = entry["scores"].get(model, (0., 0))
            entry["scores"][model] = total + score, count + 1
    result = []
    for entry in grouped.values():
        means = {model: total / count for model, (total, count) in entry["scores"].items()}
        if not means:
            raise ValueError("RouterBench prompt has no valid model outcomes")
        result.append({"task": entry["task"], "difficulty": 1 - sum(means.values()) / len(means),
                       "model_scores": means, "eval_names": sorted(entry["eval_names"])})
    if not result:
        raise ValueError("No RouterBench prompts remain after filtering")
    return result, excluded


def convert_routerbench(source, output, trust_pickle=False, exclude_eval=()):
    source, output = Path(source), Path(output)
    manifest = output.with_suffix(".source.json")
    if output.exists() or manifest.exists():
        raise ValueError("Output or source manifest exists; choose a new path")
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"} and not trust_pickle:
        raise ValueError("Pickle can execute code. Use only a trusted official download and explicitly pass --trust-pickle, or supply CSV/Parquet")
    if suffix == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            records = list(reader)
    elif suffix in {".pkl", ".pickle", ".parquet"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Native RouterBench conversion needs pandas: python -m pip install -e '.[routerbench]'") from exc
        frame = pd.read_parquet(source) if suffix == ".parquet" else pd.read_pickle(source)
        columns = list(frame.columns)
        # Discard response text and precomputed embeddings; MPNet is used locally.
        names = [c.removesuffix("|model_response") for c in columns
                 if isinstance(c, str) and c.endswith("|model_response")]
        keep = [c for c in ("prompt", "eval_name", *names) if c in columns]
        records = frame[keep].to_dict("records")
    else:
        raise ValueError("Use a processed RouterBench .pkl, .csv, or .parquet file, not the raw long export")
    score_columns = [c.removesuffix("|model_response") for c in columns
                     if isinstance(c, str) and c.endswith("|model_response")]
    if "prompt" not in columns or not score_columns or any(c not in columns for c in score_columns):
        raise ValueError("Expected processed RouterBench: prompt, model correctness columns, and matching MODEL|model_response columns")
    rows, excluded = difficulty_rows(records, score_columns, exclude_eval)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {"source": str(source), "sha256": digest.hexdigest(), "input_rows": len(records),
                  "output_prompts": len(rows), "excluded_rows": excluded, "excluded_eval_names": list(exclude_eval),
                  "score_columns": score_columns,
                  "difficulty": "1 - mean(valid per-model correctness); duplicate prompts grouped before splitting"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance
