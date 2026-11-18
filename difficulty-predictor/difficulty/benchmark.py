"""Fixed-checkpoint accuracy tests and explicitly unlabeled generalization probes."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import GROUPS, write_json
from .evaluation import baseline_predictions, summarize, gain_diagnostics, paired_interval, safe_spearman
from .inputs import contract_id, task_text
from .model import DifficultyPredictor, load_splits, targets
from .support import file_sha256


NEW_QUESTIONS = [
    ("addition", "what's 1 + 1"),
    ("addition", "What is 1 + 1?"),
    ("addition", "Add one and one."),
    ("addition", "Maya has one pear and receives another pear. How many pears does Maya have?"),
    ("addition", "Compute 1 + 1. Return only the number."),
    ("addition", "Compute 1 + 1 and briefly explain your answer."),
    ("arithmetic", "What is 17 + 26?"),
    ("arithmetic", "Calculate 89 minus 34."),
    ("arithmetic", "What is three quarters plus two thirds?"),
    ("algebra", "Solve 7x - 5 = 44 for x."),
    ("algebra", "Find all real roots of x^4 - 5x^2 + 4 = 0."),
    ("proof", "Prove that there are infinitely many primes congruent to 3 modulo 4."),
    ("code", "Write a Python function that returns the square of an integer."),
    ("code", "Write a Python function that checks whether parentheses, brackets, and braces are balanced."),
    ("code", "Implement edit distance in Python using dynamic programming and explain its complexity."),
    ("code", "Implement red-black tree insertion in Python, including rotations and rebalancing."),
    ("knowledge", "What is the capital of France?"),
    ("comparison", "Which integer is larger, 9 or 4?"),
    ("explanation", "Explain why a photon can carry momentum despite having zero rest mass."),
    ("writing", "Write a warm two-sentence thank-you note to a helpful neighbor."),
]


def identity(text):
    return hashlib.sha256(" ".join(task_text(text).split()).encode()).hexdigest()


def reject_overlap(rows, development):
    """External results exclude exact matches in any predictor development split."""
    kept, excluded, seen = [], [], set()
    for row in rows:
        key = identity(row["prompt"])
        if key in development or key in seen:
            excluded.append({"sample_id":row["sample_id"], "development_splits":sorted(development.get(key, [])),
                             "duplicate_within_external_set":key in seen})
        else:
            kept.append(row)
        seen.add(key)
    return kept, excluded


def external_rows(assets, gsm_test):
    assets = Path(assets)
    manifest = json.loads((assets / "source_manifest.json").read_text(encoding="utf-8"))
    math_rows = []
    for entry in manifest["math"]["files"]:
        path = assets / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("MATH test asset changed since download")
        for i,row in enumerate(pd.read_parquet(path).to_dict("records")):
            level = int(str(row["level"]).removeprefix("Level "))
            if level not in range(1,6):
                raise ValueError("Invalid MATH difficulty level")
            # Solutions are deliberately not passed to the predictor.
            math_rows.append({"sample_id":f"MATH/{path.parent.name}/{i}","prompt":str(row["problem"]),
                              "eval_name":"MATH", "level":level,"subject":str(row["type"])})
    he_path = assets / manifest["humaneval"]["path"]
    if file_sha256(he_path) != manifest["humaneval"]["sha256"]:
        raise ValueError("HumanEval test asset changed since download")
    with gzip.open(he_path,"rb") as stream:
        data = stream.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Unexpectedly large HumanEval data")
    # Canonical solutions and test code remain data: never fed to inference or executed.
    humaneval = [{"sample_id":r["task_id"],"eval_name":"HumanEval","prompt":r["prompt"]}
                 for r in map(json.loads,data.decode("utf-8").splitlines()) if r]
    gsm = [{"sample_id":f"GSM8K-official-test/{i}","eval_name":"GSM8K-official-test","prompt":r["question"]}
           for i,r in enumerate(map(json.loads,Path(gsm_test).read_text(encoding="utf-8").splitlines())) if r]
    if len(math_rows) != 5000 or len(humaneval) != 164 or len(gsm) != 1319:
        raise ValueError(f"Unexpected benchmark counts: MATH={len(math_rows)}, HumanEval={len(humaneval)}, GSM={len(gsm)}")
    authored = [{"sample_id":f"authored/{i}","eval_name":"new-questions","prompt":q,"probe_kind":kind}
                for i,(kind,q) in enumerate(NEW_QUESTIONS)]
    return {"GSM8K-official-test":gsm,"MATH":math_rows,"HumanEval":humaneval,"new-questions":authored}, manifest


def infer(predictor, rows, label):
    results = []
    for start in range(0,len(rows),256):
        texts = [r["prompt"] for r in rows[start:start+256]]
        results.extend(predictor.predict_with_support(texts))
        if start == 0 or (start+256) % 512 == 0 or start+256 >= len(rows):
            print(f"{label}: {min(start+256,len(rows))}/{len(rows)} questions evaluated",flush=True)
    return results


def distribution(results):
    values = np.array([[r["prediction"][g] for g in GROUPS] for r in results],dtype=float)
    support = [r["input_support"]["within_support"] for r in results if r.get("input_support")]
    return {"questions":len(results),"supported_fraction":float(np.mean(support)) if support else None,
            "mean_difficulty":dict(zip(GROUPS,map(float,values.mean(axis=0)))),
            "std_difficulty":dict(zip(GROUPS,map(float,values.std(axis=0))))}


def save_predictions(path, rows, results, labeled=False):
    with Path(path).open("w",encoding="utf-8") as stream:
        for row,result in zip(rows,results):
            record = {"sample_id":row["sample_id"],"eval_name":row["eval_name"],
                      "question":task_text(row["prompt"]),**result,
                      "pool_outcome_labels_available":labeled}
            for field in ("difficulty","level","subject","probe_kind"):
                if field in row:
                    record["target" if field=="difficulty" else field] = row[field]
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")


def run_benchmark(study, assets, gsm_test, output, device="auto"):
    started = time.perf_counter()
    study, output = Path(study), Path(output)
    recommendation = json.loads((study/"recommendation.json").read_text(encoding="utf-8"))
    model = (study/recommendation["model_relative_path"]).resolve()
    if not model.is_relative_to(study.resolve()) or output.resolve().is_relative_to(model):
        raise ValueError("Invalid model or evaluation output directory")
    prepared = study/"prepared/main"
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    splits = load_splits(prepared)
    development = defaultdict(set)
    for split,rows in splits.items():
        for row in rows:
            development[identity(row["prompt"])].add(split)
    if any(len(v)>1 for v in development.values()):
        raise ValueError("Canonical question overlap across development splits")
    external, source_manifest = external_rows(assets,gsm_test)
    predictor = DifficultyPredictor.load(model,device=device)
    if contract_id(predictor.manifest) != contract_id(manifest):
        raise ValueError("Prepared labels do not match the checkpoint's pool/target contract")
    if predictor.metadata["predictor_id"] != recommendation["predictor_id"]:
        raise ValueError("Study recommendation does not identify the loaded model")
    predictor.encoder.batch_size = 32
    output.mkdir(parents=True,exist_ok=True)
    model_hash = file_sha256(model/"model.json")
    protocol = {"started_utc":datetime.now(timezone.utc).isoformat(),"model":str(model),
                "predictor_id":predictor.metadata["predictor_id"],"pool_contract_id":contract_id(manifest),
                "model_metadata_sha256":model_hash,"source_manifest":source_manifest,
                "prepared_sha256":{s:file_sha256(prepared/f"{s}.jsonl") for s in splits},
                "gsm_official_test_sha256":file_sha256(gsm_test),"new_questions":NEW_QUESTIONS,
                "selection":"Checkpoint fixed before evaluation. No training, selection, threshold fitting or pool reranking.",
                "test_status":"Repeated evaluation of an already inspected held-out split; not new independent validation.",
                "external_scope":"Generalization in the supervised predictor only; encoder pretraining overlap cannot be ruled out.",
                "unlabeled_interpretation":"MATH, HumanEval, official GSM8K test and authored prompts lack fixed-pool outcome labels. Predictions alone cannot establish difficulty accuracy."}
    write_json(output/"protocol.json",protocol)
    rows = [r for r in splits["test"] if r["eval_name"] in {"grade-school-math","mbpp"}]
    print(f"Testing fixed checkpoint on {len(rows)} labeled GSM8K/MBPP holdout questions",flush=True)
    results = infer(predictor,rows,"Labeled holdout")
    prediction = np.array([[r["prediction"][g] for g in GROUPS] for r in results],dtype=np.float32)
    truth = targets(rows)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):
        raise ValueError("Invalid three-pool model predictions")
    saved = {(r["eval_name"],r["sample_id"]):r for r in
             map(json.loads,(model/"test_predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    original = [saved[(r["eval_name"],r["sample_id"])] for r in rows]
    original_truth = np.array([[r["target"][g] for g in GROUPS] for r in original],dtype=float)
    original_prediction = np.array([[r["prediction"][g] for g in GROUPS] for r in original],dtype=np.float32)
    parity = float(np.max(np.abs(prediction-original_prediction)))
    if not np.allclose(original_truth,truth,equal_nan=True) or parity>1e-6:
        raise ValueError("Fresh evaluation must reproduce the fixed checkpoint's saved labels and predictions")
    baselines = baseline_predictions(splits["train"],rows)
    accuracy = {"questions":len(rows),"questions_with_any_pool_label":int(np.isfinite(truth).any(axis=1).sum()),
                "saved_prediction_max_abs_difference":parity,
                "models":{name:summarize(p,truth,rows) for name,p in {**baselines,"predictor":prediction}.items()},
                "paired_comparisons":{name:paired_interval(p,prediction,truth,rows) for name,p in baselines.items()},
                "gain_diagnostics":gain_diagnostics(prediction,truth,rows),"input_distribution":distribution(results)}
    rng = np.random.default_rng(42)
    examples = []
    for family in sorted({r["eval_name"] for r in rows}):
        indices = [i for i,r in enumerate(rows) if r["eval_name"]==family and np.isfinite(truth[i]).any()]
        for i in rng.choice(indices,min(5,len(indices)),replace=False):
            examples.append({"sample_id":rows[i]["sample_id"],"family":family,"question":task_text(rows[i]["prompt"]),
                             "target":rows[i]["difficulty"],**results[i]})
    accuracy["fixed_seed_examples"] = examples
    accuracy["by_family_comparisons"] = {}
    for family in sorted({r["eval_name"] for r in rows}):
        mask = np.array([r["eval_name"]==family for r in rows])
        family_rows = [r for r,m in zip(rows,mask) if m]
        accuracy["by_family_comparisons"][family] = {
            name:paired_interval(p[mask],prediction[mask],truth[mask],family_rows) for name,p in baselines.items()}
    save_predictions(output/"labeled_predictions.jsonl",rows,results,labeled=True)
    write_json(output/"labeled_results.json",accuracy)
    report = {"protocol":protocol,"labeled":accuracy,"external":{}}
    for name,candidates in external.items():
        selected,excluded = reject_overlap(candidates,development)
        if not selected:
            raise ValueError(f"No nonoverlapping external questions: {name}")
        results = infer(predictor,selected,name)
        value = {**distribution(results),"source_questions":len(candidates),"excluded":excluded,
                 "pool_difficulty_accuracy_available":False,"interpretation":protocol["unlabeled_interpretation"]}
        if name=="MATH":
            levels = np.array([r["level"] for r in selected])
            value["level_proxy_spearman"] = {g:safe_spearman([r["prediction"][g] for r in results],levels) for g in GROUPS}
            value["proxy_interpretation"] = "Association with MATH's human difficulty levels 1-5, not measured model-pool difficulty accuracy."
            value["by_level"] = {str(level):distribution([p for row,p in zip(selected,results) if row["level"]==level]) for level in range(1,6)}
            value["by_subject"] = {}
            for subject in sorted({r["subject"] for r in selected}):
                pairs=[(r,p) for r,p in zip(selected,results) if r["subject"]==subject]
                value["by_subject"][subject] = {
                    "questions":len(pairs),"human_level_spearman":{
                        g:safe_spearman([p["prediction"][g] for r,p in pairs],[r["level"] for r,p in pairs]) for g in GROUPS}}
            write_json(output/"math_subject_proxy.json",{
                "predictor_id":predictor.metadata["predictor_id"],
                "interpretation":"Exploratory within-subject association with human levels; not fixed-pool difficulty accuracy.",
                "subjects":value["by_subject"]})
        if name=="new-questions":
            value["predictions"] = [{"question":r["prompt"],"probe_kind":r["probe_kind"],**p} for r,p in zip(selected,results)]
            addition = np.array([[p["prediction"][g] for g in GROUPS] for r,p in zip(selected,results) if r["probe_kind"]=="addition"])
            value["addition_paraphrase_range"] = dict(zip(GROUPS,map(float,np.ptp(addition,axis=0))))
        save_predictions(output/f"{name}_predictions.jsonl",selected,results)
        report["external"][name] = value
        write_json(output/"results.json",report)
    if file_sha256(model/"model.json") != model_hash:
        raise ValueError("Checkpoint metadata changed during evaluation")
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(output/"results.json",report)
    write_report(output,report)
    return report


def write_report(output, report):
    lines = ["# Fixed-checkpoint difficulty evaluation", "", report["protocol"]["test_status"], "",
             "No model or threshold training took place. Lower MSE/MAE is better; higher Spearman means better difficulty ranking.", "",
             "## Recorded pool-outcome accuracy", "", "Unknown targets are excluded separately for each pool.", "",
             "| Dataset | Questions | Global-mean MSE | Family-mean MSE | Predictor MSE |", "| --- | ---: | ---: | ---: | ---: |"]
    labeled = report["labeled"]
    for family,value in labeled["models"]["predictor"]["by_dataset"].items():
        count = sum(1 for line in (Path(output)/"labeled_predictions.jsonl").open(encoding="utf-8") if json.loads(line)["eval_name"]==family)
        scores = [labeled["models"][name]["by_dataset"][family]["macro_mse"] for name in ("global_mean","dataset_mean","predictor")]
        lines.append(f"| {family} | {count} | "+" | ".join(f"{s:.6f}" for s in scores)+" |")
    lines += ["", "| Pool | Observed targets | MAE | RMSE | Spearman |", "| --- | ---: | ---: | ---: | ---: |"]
    for group in GROUPS:
        value = labeled["models"]["predictor"][group]
        lines.append(f"| {group} | {value['n']} | {value['mae']:.4f} | {value['rmse']:.4f} | {value['spearman']:.3f} |")
    lines += ["", "Paired MSE reduction intervals (positive favors predictor):", ""]
    for name,interval in labeled["paired_comparisons"].items():
        lines.append(f"- {name}: {interval['mean_mse_reduction']:.6f}, 95% interval {interval['ci95']}.")
    lines += ["", "Per-dataset comparisons against the family-mean baseline:", ""]
    for family,comparisons in labeled["by_family_comparisons"].items():
        interval = comparisons["dataset_mean"]
        lines.append(f"- {family}: MSE reduction {interval['mean_mse_reduction']:.6f}; 95% interval {interval['ci95']}.")
    lines += ["", "## External questions: predictions without pool-outcome labels", "",
              report["protocol"]["unlabeled_interpretation"], "",
              "All exact normalized-question matches to training, validation or the inspected test split are excluded.", "",
              "| Dataset | Evaluated | Excluded | Within similarity threshold | Mean weak | Mean intermediate | Mean strong |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,value in report["external"].items():
        means=value["mean_difficulty"]
        lines.append(f"| {name} | {value['questions']} | {len(value['excluded'])} | {100*value['supported_fraction']:.1f}% | "+
                     " | ".join(f"{means[g]:.4f}" for g in GROUPS)+" |")
    math = report["external"]["MATH"]
    lines += ["", "MATH difficulty-level association (proxy only):", "", json.dumps(math["level_proxy_spearman"]), "",
              "| MATH human level | Questions | Mean weak | Mean intermediate | Mean strong |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for level,value in math["by_level"].items():
        lines.append(f"| {level} | {value['questions']} | "+" | ".join(f"{value['mean_difficulty'][g]:.4f}" for g in GROUPS)+" |")
    lines += ["",
              "A dataset mean is a prediction distribution, not proof of accuracy. Similarity acceptance is not confidence.", "",
              "## New-question probes", "", "These are qualitative sanity checks, not labeled accuracy measurements.", "",
              "The set includes the previously inspected `what's 1 + 1` debugging prompt; it is not a fresh confirmatory test.", "",
              "| Prompt | Weak | Intermediate | Strong | Supported |", "| --- | ---: | ---: | ---: | --- |"]
    for row in report["external"]["new-questions"]["predictions"]:
        lines.append("| "+row["question"].replace("|","\\|")+" | "+" | ".join(f"{row['prediction'][g]:.4f}" for g in GROUPS)+
                     " | "+str(row["input_support"]["within_support"])+" |")
    lines += ["", "## Sample labeled questions", "", "Five examples per family, sampled with seed 42 before inspecting their errors.", ""]
    for row in labeled["fixed_seed_examples"]:
        lines += [f"### {row['sample_id']}", "", row["question"], "",
                  "Recorded target: `"+json.dumps(row["target"])+"`", "Predicted: `"+json.dumps(row["prediction"])+"`", ""]
    lines += ["", "MATH source: https://huggingface.co/datasets/EleutherAI/hendrycks_math",
              "HumanEval source: https://github.com/openai/human-eval", "",
              "Exact revisions and checksums are in protocol.json. Source solutions/tests were never used as predictor inputs or executed.",
              "Every evaluated prediction is saved in the JSONL files. No paid LLM calls were made.", ""]
    (Path(output)/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
