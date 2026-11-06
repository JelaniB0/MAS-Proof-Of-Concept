"""Inspect source labels, prompts and representation-level split overlap."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from .data import GROUPS, write_json
from .inputs import question_text, GSM_DEMO_HEADER


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_prepared(directory):
    directory = Path(directory)
    rows = [row for split in ("train", "validation", "test") for row in read_rows(directory / f"{split}.jsonl")]
    report = {"questions": len(rows), "splits": dict(Counter(r["split"] for r in rows)), "by_dataset": {}}
    identities = defaultdict(set)
    for row in rows:
        normalized = re.sub(r"\s+", " ", question_text(row["prompt"])).strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        identities[digest].add(row["split"])
    report["question_only_cross_split_overlaps"] = sum(len(splits) > 1 for splits in identities.values())
    for dataset in sorted({r["eval_name"] for r in rows}):
        selected = [r for r in rows if r["eval_name"] == dataset]
        scores = [score for r in selected for score in r["scores"].values()]
        truth = np.array([[r["difficulty"][g] for g in GROUPS] for r in selected], dtype=float)
        report["by_dataset"][dataset] = {
            "questions": len(selected), "split_counts": dict(Counter(r["split"] for r in selected)),
            "model_score_observations": len(scores),
            "nonbinary_score_fraction": sum(s not in (0, 1) for s in scores) / len(scores),
            "score_histogram": {str(k): v for k, v in sorted(Counter(scores).items())},
            "prompts_with_worked_examples": sum(GSM_DEMO_HEADER in r["prompt"] for r in selected),
            "target_mean": dict(zip(GROUPS, map(float, np.nanmean(truth, axis=0)))),
            "target_std": dict(zip(GROUPS, map(float, np.nanstd(truth, axis=0)))),
            "mean_full_characters": float(np.mean([len(r["prompt"]) for r in selected])),
            "mean_question_characters": float(np.mean([len(question_text(r["prompt"])) for r in selected])),
            "strong_pool_worse_than_weak_fraction": float(np.mean(truth[:, 2] > truth[:, 0]))}
    return report


def compare_exports(zero_jsonl, five_pickle, selected_datasets):
    from .safe_pickle import read_routerbench
    zero = [r for r in read_rows(zero_jsonl) if r["eval_name"] in selected_datasets]
    five_frame = read_routerbench(five_pickle)
    five = {(str(r["eval_name"]), str(r["sample_id"])): r for r in
            five_frame[five_frame.eval_name.isin(selected_datasets)].to_dict("records")}
    models = [c for c in five_frame.columns if c not in {"sample_id", "eval_name", "prompt", "oracle_model_to_route_to"} and "|" not in c]
    equal_scores, equal_prompts, compared = 0, 0, 0
    for row in zero:
        other = five.get((row["eval_name"], row["sample_id"]))
        if other is None:
            continue
        compared += 1
        equal_scores += all((row[m] is None and np.isnan(other[m])) or row[m] == other[m] for m in models)
        equal_prompts += row["prompt"] == other["prompt"]
    return {"compared_questions": compared, "identical_all_model_scores": equal_scores,
            "identical_prompts": equal_prompts,
            "five_export_sha256": hashlib.sha256(Path(five_pickle).read_bytes()).hexdigest(),
            "interpretation": "Do not count matching exports as independent extra inference trials."}


def run_audit(prepared, output, zero_jsonl=None, five_pickle=None):
    report = audit_prepared(prepared)
    if zero_jsonl and five_pickle:
        report["export_comparison"] = compare_exports(zero_jsonl, five_pickle, set(report["by_dataset"]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def external_overlap(prepared, external_file, question_field="question"):
    """Compare local benchmark task text with predictor splits without reading answers."""
    def canonical(text):
        value = unicodedata.normalize("NFKC", question_text(text))
        value = value.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
        return " ".join(value.split())
    external_file = Path(external_file)
    if external_file.suffix == ".json":
        external = json.loads(external_file.read_text(encoding="utf-8"))
    else:
        external = read_rows(external_file)
    if not isinstance(external, list) or any(question_field not in row for row in external):
        raise ValueError("External data needs a list of records with the specified question field")
    questions = {canonical(row[question_field]) for row in external}
    by_split, all_seen = {}, set()
    for split in ("train", "validation", "test"):
        rows = read_rows(Path(prepared) / f"{split}.jsonl")
        matched = [row for row in rows if canonical(row["prompt"]) in questions]
        by_split[split] = {"matched_predictor_records": len(matched),
                           "matched_external_questions": len({canonical(row["prompt"]) for row in matched})}
        all_seen.update(canonical(row["prompt"]) for row in matched)
    return {"external_file": str(external_file), "question_field": question_field,
            "external_records": len(external), "unique_external_questions": len(questions),
            "matched_external_questions_any_predictor_split": len(all_seen), "by_predictor_split": by_split,
            "method": "NFKC, quote normalization and whitespace-normalized task text; not a paraphrase or semantic-overlap guarantee."}
