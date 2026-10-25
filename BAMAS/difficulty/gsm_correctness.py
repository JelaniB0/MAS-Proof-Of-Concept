"""Conservative, reference-based regrading of saved GSM8K model responses.

Unclear answers are quarantined, never silently counted as incorrect. Parsing
never uses the reference value to choose a number from a model response.
"""

import ast
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from .data import GROUPS, add_targets, write_json
from .inputs import question_text

GRADER_VERSION = "gsm-final-numeric-v2"
TARGET = "one minus mean reference-checked numeric-answer correctness per fixed pool"
NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
NUMBERS = re.compile(r"(?<![\w.])" + NUMBER + r"(?![\w.])")


def canonical_question(value):
    text = unicodedata.normalize("NFKC", question_text(value))
    for original, replacement in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')]:
        text = text.replace(original, replacement)
    return " ".join(text.split())


def numeric_value(text):
    text = text.strip().replace(",", "").replace("\u2212", "-")
    if not re.fullmatch(NUMBER, text):
        raise ValueError("Not a single numeric answer")
    parts = text.split("/")
    result = Fraction(parts[0].strip())
    return result / Fraction(parts[1].strip()) if len(parts) == 2 else result


def reference_value(answer):
    if "####" not in answer:
        raise ValueError("Reference lacks GSM8K final-answer marker")
    return numeric_value(answer.rsplit("####", 1)[1])


def decode_response(value):
    if not isinstance(value, str) or not value.strip():
        return None, "missing_response"
    if len(value) > 500_000:
        return None, "oversized_response"
    if value.lstrip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None, "invalid_serialization"
        if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], str):
            return None, "multiple_or_invalid_responses"
        value = parsed[0]
    return value.replace("\\n", "\n").replace("\\t", "\t").replace("\u2212", "-"), None


def extract_answer(response):
    text, error = decode_response(response)
    if error:
        return {"status": "unknown", "method": error, "value": None}
    if not text.strip():
        return {"status": "unknown", "method": "empty_response", "value": None}
    text = text.replace("**", "").replace("\\$", "$")
    # Treat a LaTeX fraction as one number. No expression evaluation is used.
    text = re.sub(r"\\(?:d|t)?frac\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}", r"\1/\2", text)
    if re.match(r"\s*(?:error\b|traceback\b|exception\b)", text, re.I):
        return {"status": "unknown", "method": "error_response", "value": None}
    candidates, method = [], None
    marked = re.findall(r"####\s*\$?(" + NUMBER + r")", text)
    boxed = re.findall(r"\\boxed\{\s*\$?(" + NUMBER + r")\s*\}", text)
    if marked or boxed:
        candidates, method = marked + boxed, "explicit_marker"
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        last = lines[-1]
        if last.endswith((":", "?")) or re.match(r"^(?:[*-]\s*)?(?:First\b|Let's\b|To (?:find|solve|calculate|determine)\b|We need to\b)", last, re.I):
            return {"status": "unknown", "method": "unfinished_or_introductory_line", "value": None}
        if last.startswith("```") or last.startswith(("print(", "return ")):
            return {"status": "unknown", "method": "code_or_fence", "value": None}
        # A concluding equation may contain several operands; use its final RHS.
        if "=" in last:
            last = last.rsplit("=", 1)[1]
            method = "final_equation_rhs"
        else:
            method = "final_line"
        candidates = NUMBERS.findall(last.rstrip(".!?"))
    try:
        values = {numeric_value(value) for value in candidates}
    except (ValueError, ZeroDivisionError):
        values = set()
    if len(values) != 1:
        return {"status": "unknown", "method": "no_unique_final_number", "value": None}
    return {"status": "parsed", "method": method, "value": str(next(iter(values)))}


def regrade_rows(frame, references, original_rows, groups):
    gold = {}
    for record in references:
        key, value = canonical_question(record["question"]), reference_value(record["answer"])
        if key in gold and gold[key] != value:
            raise ValueError("Conflicting reference answers for identical questions")
        gold[key] = value
    by_id = {str(record["sample_id"]): record for record in frame.to_dict("records")
             if record["eval_name"] == "grade-school-math"}
    models = [name for group in GROUPS for name in groups[group]]
    complete, observations, quarantine = [], [], []
    for original in original_rows:
        if original["eval_name"] != "grade-school-math":
            continue
        key = canonical_question(original["prompt"])
        if key not in gold:
            raise ValueError(f"Reference not found for {original['sample_id']}")
        source = by_id[original["sample_id"]]
        if canonical_question(source["prompt"]) != key:
            raise ValueError("Response source question does not match prepared question")
        scores = {}
        for model in models:
            raw = source.get(model + "|model_response")
            parsed = extract_answer(raw)
            correct = int(Fraction(parsed["value"]) == gold[key]) if parsed["status"] == "parsed" else None
            if correct is not None:
                scores[model] = correct
            observation = {"sample_id": original["sample_id"], "split": original["split"], "question": question_text(original["prompt"]),
                           "model": model, "reference": str(gold[key]), "supplied_quality": original["scores"][model],
                           "response": raw, **parsed, "correct": correct}
            observations.append(observation)
        if len(scores) != len(models):
            quarantine.append({"sample_id": original["sample_id"], "split": original["split"],
                               "unresolved_models": [m for m in models if m not in scores]})
            continue
        row = copy.deepcopy(original)
        row["source_scores"] = row["scores"]
        row["scores"] = scores
        row["trials"] = {model: 1 for model in models}
        complete.append(row)
    add_targets(complete, groups)
    return complete, observations, quarantine


def prepare_correctness(source, references, original_prepared, output):
    from .model import load_splits
    from .safe_pickle import read_routerbench
    from .audit import audit_prepared
    source, references, original_prepared, output = map(Path, (source, references, original_prepared, output))
    manifest = json.loads((original_prepared / "manifest.json").read_text(encoding="utf-8"))
    provenance = {"grader_version": GRADER_VERSION, "response_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "reference_sha256": hashlib.sha256(references.read_bytes()).hexdigest(),
                  "original_split_sha256": {s: hashlib.sha256((original_prepared / f"{s}.jsonl").read_bytes()).hexdigest()
                                            for s in ("train", "validation", "test")}, "groups": manifest["groups"]}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "regrade_protocol.json").is_file():
        previous = json.loads((output / "regrade_protocol.json").read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError("Different regrading protocol; choose a new output directory")
    original = load_splits(original_prepared)
    # splitlines() would also split Unicode paragraph separators inside JSON strings.
    with references.open(encoding="utf-8") as stream:
        reference_rows = [json.loads(line) for line in stream if line.strip()]
    rows, observations, quarantine = regrade_rows(read_routerbench(source), reference_rows,
                                                [r for split in original.values() for r in split], manifest["groups"])
    write_json(output / "regrade_protocol.json", provenance)
    for filename, records in (("observations.jsonl", observations), ("quarantine.jsonl", quarantine)):
        with (output / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    for label in ("correctness", "matched_supplied_scores"):
        destination = output / label
        destination.mkdir(exist_ok=True)
        selected = copy.deepcopy(rows)
        if label == "matched_supplied_scores":
            for row in selected:
                row["scores"] = row["source_scores"]
            add_targets(selected, manifest["groups"])
        for split in ("train", "validation", "test"):
            with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in selected:
                    if row["split"] == split:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
        semantics = TARGET if label == "correctness" else "one minus mean supplied RouterBench quality per fixed pool"
        updated = {**manifest, "source": str(source), "source_sha256": provenance["response_source_sha256"],
                   "datasets": ["grade-school-math"], "target_definition": semantics, "label_provenance": provenance,
                   "split_counts": dict(Counter(r["split"] for r in selected)),
                   "pool_definition": "Original pools fixed; no reranking after regrading.",
                   "training_macro_quality": None, "complete_case_policy": "Only questions with parsed single responses for all 11 models.",
                   "evaluation_status": "Exploratory regrading study; original split identities preserved, previously inspected."}
        write_json(destination / "manifest.json", updated)
        audit = audit_prepared(destination)
        if audit["question_only_cross_split_overlaps"]:
            raise ValueError("Cross-split question overlap after regrading")
        write_json(destination / "audit.json", audit)
    categories = defaultdict(list)
    for record in observations:
        if record["split"] == "train":
            categories["unknown" if record["correct"] is None else str(record["correct"])].append(record)
    rng = random.Random(20261024)
    review = [r for category in sorted(categories) for r in rng.sample(categories[category], min(20, len(categories[category])))]
    with (output / "review_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in review:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = {"grader_version": GRADER_VERSION, "observations": len(observations),
             "parsed": sum(r["correct"] is not None for r in observations),
             "methods": dict(Counter(r["method"] for r in observations)),
             "questions_complete": len(rows), "questions_quarantined": len(quarantine),
             "complete_by_split": dict(Counter(r["split"] for r in rows)),
             "quarantined_by_split": dict(Counter(r["split"] for r in quarantine)),
             "review_sample_records": len(review), "review_sample_status": "Pending manual inspection before training.",
             "limitations": "Numeric extraction proxies answer correctness, not formal reasoning correctness; complete-case selection may bias results."}
    write_json(output / "regrade_summary.json", stats)
    return stats
