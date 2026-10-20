"""Explicit JSONL schemas, safe CSV imports, and reproducible prompt-level splits."""

import csv
import hashlib
import json
import math
from pathlib import Path
import random


def task_key(task):
    return hashlib.sha256(" ".join(task.casefold().split()).encode()).hexdigest()


def load_records(path, kind="tasks"):
    rows, seen = [], set()
    for index, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"{path}:{index}: task must be a nonempty string")
        key = task_key(task)
        if key in seen:
            raise ValueError(f"{path}:{index}: duplicate task; split by prompt to avoid leakage")
        seen.add(key)
        if kind == "difficulty":
            value = row.get("difficulty")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path}:{index}: difficulty must be in [0, 1], with 1 meaning hardest")
        elif row.get("scorer") == "humaneval":
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("task_id", "code_prompt", "test", "entry_point")):
                raise ValueError(f"{path}:{index}: HumanEval requires task_id, code_prompt, test, entry_point")
        elif not isinstance(row.get("answers"), list) or not row["answers"] or any(
                not isinstance(a, str) or not a.strip() for a in row["answers"]):
            raise ValueError(f"{path}:{index}: answers must be a nonempty list of accepted strings")
        if row.get("scorer", "exact") not in {"exact", "mmlu", "math", "humaneval"}:
            raise ValueError(f"{path}:{index}: unsupported scorer")
        rows.append({**row, "key": key})
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def split_records(rows, validation=None, seed=42):
    if validation is None:
        if len(rows) < 4:
            raise ValueError("At least four distinct prompts are needed for a train/validation split")
        ordered = list(rows)
        random.Random(seed).shuffle(ordered)
        count = max(1, len(rows) // 5)
        train, validation = ordered[count:], ordered[:count]
    else:
        train = rows
    if {r["key"] for r in train} & {r["key"] for r in validation}:
        raise ValueError("Training and validation prompts overlap")
    return train, validation


def exact_match(answer, accepted):
    normalize = lambda s: " ".join((s or "").strip().casefold().split())
    return float(normalize(answer) in {normalize(a) for a in accepted})


def import_scores(source, output, task_column, score_columns, model_column=None):
    """CSV export only: explicit score columns avoid treating prices as correctness.

    Wide: one task with one score column per model.
    Long: one task/model/score per row (one score column and model_column).
    No pickle, eval, or execution of dataset contents.
    """
    if model_column and len(score_columns) != 1:
        raise ValueError("Long CSV imports require exactly one score column")
    grouped = {}
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {task_column, *score_columns} | ({model_column} if model_column else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CSV does not contain all requested columns")
        for index, row in enumerate(reader, 2):
            task = row[task_column]
            if not task.strip():
                raise ValueError(f"CSV line {index}: empty task")
            entry = grouped.setdefault(task_key(task), {"task": task, "scores": {}})
            for column in score_columns:
                raw = row[column]
                if raw is None or not raw.strip():
                    continue
                score = float(raw)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"CSV line {index}: scores must be in [0, 1]")
                model = row[model_column] if model_column else column
                if not model or model in entry["scores"]:
                    raise ValueError(f"CSV line {index}: empty/duplicate model outcome for this prompt")
                entry["scores"][model] = score
    result = []
    for entry in grouped.values():
        scores = entry["scores"]
        if not scores:
            raise ValueError("A prompt has no valid scores")
        result.append({"task": entry["task"], "difficulty": 1 - sum(scores.values()) / len(scores),
                       "model_scores": scores, "source": str(source)})
    if not result:
        raise ValueError("CSV has no outcomes")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf-8")
    return len(result)
