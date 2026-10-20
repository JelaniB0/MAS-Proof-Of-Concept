"""Normalize public benchmark exports without sending labels/tests to agents."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import urlencode
from urllib.request import urlopen

from .data import task_key
from .scoring import boxed


def fetch_dataset(dataset, subset, split, output, limit, offset=0):
    """Fetch an explicit public HF slice; no credentials or remote dataset scripts."""
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    output = Path(output)
    manifest = output.with_suffix(".download.json")
    if output.exists() or manifest.exists():
        raise ValueError("Dataset export or manifest already exists; choose a new output")
    rows, total = [], None
    while len(rows) < limit:
        query = urlencode({"dataset": dataset, "config": subset, "split": split,
                           "offset": offset + len(rows), "length": min(100, limit - len(rows))})
        with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=60) as response:
            data = json.load(response)
        batch = data["rows"]
        total = data.get("num_rows_total", total)
        if not batch:
            break
        for item in batch:
            if item.get("truncated_cells"):
                raise ValueError("Dataset viewer truncated a cell; download a complete native export instead")
            rows.append(item["row"])
        if total is not None and offset + len(rows) >= total:
            break
    if len(rows) != limit:
        raise ValueError(f"Requested {limit} records at offset {offset}, but only {len(rows)} are available")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    manifest.write_text(json.dumps({"dataset": dataset, "subset": subset, "split": split,
                                   "offset": offset, "count": len(rows), "num_rows_total": total,
                                   "sha256": hashlib.sha256(content.encode()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def source_rows(source, benchmark):
    path = Path(source)
    if path.is_dir():
        files = sorted(path.rglob("*.csv" if benchmark == "mmlu" else "*.json"))
        for file in files:
            yield from source_rows(file, benchmark)
        return
    if path.suffix == ".csv" and benchmark == "mmlu":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 6 or row[-1].strip() not in "ABCD" or len(row[-1].strip()) != 1:
                    raise ValueError("Native MMLU CSV requires question, four choices, and one A-D answer (no header)")
                yield {"question": row[0], "choices": row[1:5], "answer": row[5].strip(), "subject": path.stem}
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        if path.suffix == ".json":
            data = json.load(handle)
            yield from data if isinstance(data, list) else [data]
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def normalize(row, benchmark, origin):
    metadata = {"benchmark": benchmark, "source": origin}
    if benchmark == "mmlu":
        question, choices, answer = row["question"], row["choices"], row["answer"]
        if len(choices) != 4 or any(not isinstance(v, str) for v in choices):
            raise ValueError("MMLU requires four string choices")
        if isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer <= 3:
            answer = "ABCD"[answer]
        if not isinstance(answer, str) or answer not in {"A", "B", "C", "D"}:
            raise ValueError("MMLU answer must be A-D or an integer index 0..3")
        task = question.strip() + "\n" + "\n".join(f"{label}. {choice}" for label, choice in zip("ABCD", choices))
        task += "\nReturn only the letter of the correct option."
        return {**metadata, "task": task, "answers": [answer], "scorer": "mmlu",
                "subject": row.get("subject", "unknown"), "source_id": row.get("id", task_key(question))}
    if benchmark == "math":
        answer = row.get("answer") or boxed(row.get("solution", ""))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("MATH row needs an answer or a solution with a boxed final answer")
        problem = row["problem"]
        return {**metadata, "task": problem.strip() + "\nGive your final answer in \\boxed{...}.",
                "answers": [answer], "scorer": "math", "level": row.get("level"),
                "subject": row.get("type"), "source_id": row.get("id", task_key(problem))}
    if benchmark == "humaneval":
        for key in ("task_id", "prompt", "test", "entry_point"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"HumanEval row requires {key}")
        return {**metadata, "task": "Complete this Python function. Return only the completion code that follows the prompt, "
                "preserving its indentation. Do not include tests or an explanation.\n\n" + row["prompt"],
                "scorer": "humaneval", "source_id": row["task_id"], "task_id": row["task_id"],
                "code_prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]}
    raise ValueError("Unknown benchmark")


def prepare(source, benchmark, output, limit=None, offset=0, seed=None):
    if limit is not None and limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    rows = [normalize(row, benchmark, str(source)) for row in source_rows(source, benchmark)]
    keys = [task_key(row["task"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prompts in benchmark source; supply a single, deduplicated split")
    if seed is not None:
        random.Random(seed).shuffle(rows)
    rows = rows[offset:offset + limit if limit is not None else None]
    if not rows:
        raise ValueError("No benchmark records selected")
    output = Path(output)
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise ValueError("Prepared dataset or manifest already exists; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output.write_text(content, encoding="utf-8")
    manifest.write_text(json.dumps({"benchmark": benchmark, "source": str(source), "count": len(rows),
                                   "offset": offset, "limit": limit, "shuffle_seed": seed,
                                   "sha256": hashlib.sha256(content.encode()).hexdigest(),
                                   "note": "Local selection; not an assertion of the authors' exact split or ordering."}, indent=2) + "\n", encoding="utf-8")
    return len(rows)
