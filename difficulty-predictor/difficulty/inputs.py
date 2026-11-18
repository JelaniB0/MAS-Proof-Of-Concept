"""Versioned, label-free transformations of question text."""

import hashlib
import json
import math
import re

from .data import GROUPS, prompt_text

INPUT_VERSION = "question-input-v1"
TASK_INPUT_VERSION = "routerbench-task-v1"
GSM_DEMO_HEADER = "The following are examples of grade school math problems and answers:"


def question_text(value):
    """Remove only the recognized GSM8K worked-example preamble.

    Ordinary questions, code tasks and other datasets pass through unchanged.
    The same transformation applies to training and inference; no answers, scores,
    dataset IDs, or correctness information are used.
    """
    text = prompt_text(value)
    if GSM_DEMO_HEADER in text:
        answers = list(re.finditer(r"(?m)^Answer:[^\n]*\n+", text))
        if answers:
            candidate = text[answers[-1].end():].strip()
            if candidate:
                return candidate
    return text


def input_text(value, mode="question"):
    if mode == "question":
        return question_text(value)
    if mode == "full":
        return prompt_text(value)
    if mode == "task":
        return task_text(value)
    raise ValueError("Input mode must be question, full or task")


def input_version(mode):
    if mode not in {"full", "question", "task"}:
        raise ValueError("Unsupported input mode")
    return TASK_INPUT_VERSION if mode == "task" else INPUT_VERSION


def task_text(value):
    """Strip recognized benchmark examples/formatting while retaining task options.

    A completed MCQ example requires the exact known export footer and a
    single-letter answer on its answer line. The target's unanswered footer
    never qualifies. No gold answer or dataset identifier enters the transform.
    """
    text = question_text(value)
    footer = r"(?m)^Print only a single choice[^\n]*?without explanation\.\s*Answer:\s*"
    completed = list(re.finditer(footer + r"[A-E][ \t]*\n+", text))
    if completed:
        text = text[completed[-1].end():].strip()
    text = text.removeprefix("Please answer with the letter of the correct answer.").strip()
    text = re.sub(r"\nPrint only a single choice[^\n]*?without explanation\.\s*Answer:\s*$", "", text).strip()
    return text


def contract_id(manifest):
    """Fingerprint the target pool definition, not a particular question or split."""
    payload = {"group_order": list(GROUPS), "groups": manifest["groups"],
               "source_sha256": manifest.get("source_sha256"),
               "target": "one minus mean observed normalized quality per pool"}
    # Preserve legacy identities while distinguishing new correctness targets.
    if "target_definition" in manifest:
        payload["target"] = manifest["target_definition"]
        payload["label_provenance"] = manifest.get("label_provenance")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def controller_features(predictor, texts, expected_contract=None):
    """Stable features for later RL/ILP consumers; does not make allocation choices."""
    pool_contract = contract_id(predictor.manifest)
    if expected_contract is not None and expected_contract != pool_contract:
        raise ValueError("Predictor pool definition does not match the controller's expected contract")
    metadata = getattr(predictor, "metadata", {})
    if metadata.get("input_support") or metadata.get("prediction_level") == "model":
        assessed = predictor.predict_with_support(texts)
        predictions = [r["prediction"] for r in assessed]
        support = [r["input_support"] for r in assessed]
    else:
        predictions, support = predictor.predict(texts), None
    outputs = []
    for scores in predictions:
        if set(scores) != set(GROUPS) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Predictor must return exactly three finite difficulty scores in [0, 1]")
        vector = [scores[group] for group in GROUPS]
        outputs.append({"schema_version": "bamas-difficulty-v1", "pool_contract_id": pool_contract,
                        "target_definition": predictor.manifest.get("target_definition", "one minus mean observed normalized quality per pool"),
                        "predictor_id": metadata.get("predictor_id"), "input_mode": metadata.get("input_mode", "full"),
                        "group_order": list(GROUPS), "difficulty": scores, "difficulty_vector": vector,
                        "estimated_quality": {group: 1 - scores[group] for group in GROUPS},
                        "estimated_quality_gains": {
                            "weak_to_intermediate": scores["weak"] - scores["intermediate"],
                            "intermediate_to_strong": scores["intermediate"] - scores["strong"],
                            "weak_to_strong": scores["weak"] - scores["strong"]},
                        "interpretation": "Pool-relative quality deficit; not calibrated agent success or token utility."})
        if support is not None:
            outputs[-1]["input_support"] = support[len(outputs)-1]
        if "cohort" in predictor.manifest:
            outputs[-1]["cohort"] = predictor.manifest["cohort"]
        if metadata.get("prediction_level") == "model":
            model_scores=assessed[len(outputs)-1]["model_difficulty"]
            outputs[-1]["model_difficulty"]=model_scores
            outputs[-1]["estimated_model_quality"]={model:1-value for model,value in model_scores.items()}
    return outputs
