"""All released RouterBench task families, with explicit metric/model cohorts."""

from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import GROUPS, add_targets, normalize, rank_groups, score, write_json
from .evaluation import summarize, gain_diagnostics, paired_interval, read_saved_predictions
from .inputs import task_text, contract_id, controller_features
from .model import DifficultyPredictor, load_splits, targets
from .research import run_suite
from .safe_pickle import read_routerbench
from .support import file_sha256, attach_support
from .probes import PROBES

VERSION = "all-routerbench-v1"
MCQ_FAMILIES = {"arc-challenge", "hellaswag", "winogrande"}
CONFIGURATIONS = [
    {"name":"task_natural_mse", "input_mode":"task", "balanced":False, "loss":"mse"},
    {"name":"task_sqrt_balanced_mse", "input_mode":"task", "balanced":True, "balance_power":.5, "loss":"mse"},
]


def family_name(name):
    if name.startswith("mmlu-"):
        return "mmlu"
    if name.startswith("mtbench"):
        return "mtbench"
    if name in MCQ_FAMILIES | {"grade-school-math", "mbpp"}:
        return name
    return "additional-evals"


def _fingerprint(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def partition_rows(rows, by_article=False):
    """Group duplicate tasks; RAG also groups all questions from one article.

    The hash anchor is the old zero-export prompt when available, preserving
    historical GSM8K/MBPP folds while grouping newly stripped task duplicates.
    """
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    identities = {}
    for i, row in enumerate(rows):
        row["prompt_hash"] = _fingerprint(task_text(row["prompt"]))
        keys = ["task:"+row["prompt_hash"], "id:"+row["sample_id"]]
        if by_article:
            keys.append("article:"+row["article_source"])
        for key in keys:
            if key in identities:
                a,b = root(i),root(identities[key])
                parent[max(a,b)] = min(a,b)
            identities[key] = i
    members = defaultdict(list)
    for i,row in enumerate(rows):
        members[root(i)].append(i)
    for indices in members.values():
        key = min(_fingerprint(rows[i].get("split_anchor",rows[i]["prompt"])) for i in indices)
        uniform = int(hashlib.sha256(f"42:{key}".encode()).hexdigest()[:16],16) / 2**64
        split = "test" if uniform < .15 else "validation" if uniform < .30 else "train"
        for i in indices:
            rows[i]["split"] = split
            rows[i]["bootstrap_cluster"] = key
            rows[i].pop("split_anchor",None)
    if set(r["split"] for r in rows) != {"train","validation","test"}:
        raise ValueError("Not enough independent task/article groups for all splits")
    return rows


def assemble_main(zero, five, groups, gsm_scores):
    """Select paper-matched MCQ conditions; keep every other available task."""
    zero_rows = normalize(zero)
    five_rows = normalize_condition_export(five,groups)
    by_zero = {r["sample_id"]:r for r in zero_rows}
    by_five = {r["sample_id"]:r for r in five_rows}
    rows, semantic = [], {}
    for sample_id in sorted(set(by_zero) | set(by_five)):
        available = by_zero.get(sample_id,by_five.get(sample_id))
        source_name = available["eval_name"]
        mcq = source_name.startswith("mmlu-") or source_name in MCQ_FAMILIES
        selected = by_five.get(sample_id,available) if mcq else by_zero.get(sample_id,available)
        row = {**selected, "scores":dict(selected["scores"]), "source_eval_name":source_name,
               "eval_name":family_name(source_name), "source_condition":"5shot-export" if mcq and sample_id in by_five else "0shot-export",
               "split_anchor":by_zero.get(sample_id,selected)["prompt"]}
        observed = list(row["scores"].values())
        kind = "published_binary_outcome" if all(x in (0,1) for x in observed) else "published_normalized_grade"
        if source_name == "grade-school-math":
            if sample_id not in gsm_scores:
                raise ValueError("GSM8K question missing from reference-based audit")
            row["scores"] = {m:v for m,v in gsm_scores[sample_id].items() if v is not None}
            kind = "reference_checked_numeric_correctness"
        row["metric_kind"] = kind
        semantic.setdefault(source_name,set()).add(kind)
        rows.append(row)
    partition_rows(rows)
    add_targets(rows,groups,min_coverage=1)
    # A rating endpoint (0 or 1) remains a rating if its source uses graded scores.
    source_kinds={name:("reference_checked_numeric_correctness" if name=="grade-school-math" else
                       "published_normalized_grade" if "published_normalized_grade" in kinds else "published_binary_outcome")
                  for name,kinds in semantic.items()}
    for row in rows:
        row["metric_kind"]=source_kinds[row["source_eval_name"]]
    return rows, {name:[kind] for name,kind in source_kinds.items()}


def normalize_condition_export(frame, groups):
    """Merge model-specific few-shot prompt variants only for identical tasks.

    Fourteen ARC IDs in the official five-shot export span two wide rows with
    different worked examples and complementary model columns. Their actual
    question/options must agree before those observations can be pooled.
    """
    models=[m for g in GROUPS for m in groups[g]]
    merged={}
    for record in frame.to_dict("records"):
        key=(str(record['eval_name']),str(record['sample_id']))
        prompt=task_text(record['prompt'])
        if key not in merged:
            merged[key]={"sample_id":key[1],"eval_name":key[0],"prompt":prompt,"observations":defaultdict(list),"prompt_variants":set()}
        current=merged[key]
        if _fingerprint(current['prompt']) != _fingerprint(prompt):
            raise ValueError(f"Source ID refers to conflicting actual tasks: {key}")
        current['prompt_variants'].add(_fingerprint(str(record['prompt'])))
        for model in models:
            quality=score(record.get(model))
            if quality is not None:
                current['observations'][model].append(quality)
    result=[]
    for row in merged.values():
        observations=row.pop('observations')
        variants=row.pop('prompt_variants')
        row['scores']={model:sum(values)/len(values) for model,values in observations.items()}
        row['trials']={model:len(values) for model,values in observations.items()}
        row['benchmark_prompt_variants']=len(variants)
        result.append(row)
    return sorted(result,key=lambda row:(row['eval_name'],row['sample_id']))


def prepare_all(directory, original_prepared, gsm_audit, destination):
    directory, original_prepared, gsm_audit, destination = map(Path,(directory,original_prepared,gsm_audit,destination))
    zero_file,five_file,rag_file = [directory/name for name in ("routerbench_0shot.jsonl","routerbench_5shot.pkl","routerbench_rag.pkl")]
    old = json.loads((original_prepared/"manifest.json").read_text(encoding="utf-8"))
    audit_protocol = json.loads((gsm_audit/"regrade_protocol.json").read_text())
    if audit_protocol["grader_version"] != "gsm-final-numeric-v2":
        raise ValueError("This study requires the reviewed v2 GSM8K grader")
    review = json.loads((gsm_audit/"manual_review.json").read_text())
    if review["status"] != "passed_sample_review" or review["accepted_extraction_errors_found"]:
        raise ValueError("GSM8K extraction review has not passed")
    provenance = {"version":VERSION,"zero_sha256":file_sha256(zero_file),"five_sha256":file_sha256(five_file),
                  "rag_sha256":file_sha256(rag_file),"gsm_observations_sha256":file_sha256(gsm_audit/"observations.jsonl"),
                  "gsm_regrade_protocol":audit_protocol,"main_groups":old["groups"],
                  "condition_policy":"5shot-export scores/prompts for MMLU, ARC, HellaSwag, Winogrande; 0shot-export otherwise. GSM0 already contains worked examples.",
                  "input_policy":"routerbench-task-v1; remove recognized worked examples and answer-format footers, retain options.",
                  "missing_policy":"Unknown GSM responses omitted; pool target absent unless every fixed-pool member is observed."}
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/"preparation_protocol.json").exists() and json.loads((destination/"preparation_protocol.json").read_text()) != provenance:
        raise ValueError("Different all-dataset preparation protocol; use a new directory")
    gsm_scores = defaultdict(dict)
    with (gsm_audit/"observations.jsonl").open(encoding="utf-8") as stream:
        for record in map(json.loads,stream):
            gsm_scores[record["sample_id"]][record["model"]] = record["correct"]
    five = read_routerbench(five_file)
    models = [m for g in GROUPS for m in old["groups"][g]]
    five = five[["sample_id","eval_name","prompt",*models]]
    zero = pd.read_json(zero_file,lines=True)
    rows,semantics = assemble_main(zero,five,old["groups"],gsm_scores)
    previous = load_splits(original_prepared)
    original_folds = {r["sample_id"]:split for split,rs in previous.items() for r in rs}
    changes = [r["sample_id"] for r in rows if r["sample_id"] in original_folds and r["split"] != original_folds[r["sample_id"]]]
    if changes:
        raise ValueError(f"Canonical-task grouping changes original GSM8K/MBPP splits: {changes[:3]}")
    main_manifest = {**old,"source":str(directory),"source_sha256":hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest(),
                     "groups":old["groups"],"training_macro_quality":None,
                     "target_definition":"one minus mean task-specific normalized outcome per fixed pool; GSM8K uses reference numeric correctness, other tasks use published binary/graded metrics",
                     "label_provenance":provenance,"source_metrics":semantics,"cohort":"main-11-models",
                     "split_method":"Connected canonical task texts/sample IDs; 70/15/15 hash folds anchored to original 0shot-export prompts where available.",
                     "generation_condition_policy":provenance["condition_policy"],"pool_definition":"Original eleven-model pools, fixed before this study."}
    _save_prepared(destination/"main",rows,main_manifest)
    del zero,five,rows
    gc.collect()
    rag = read_routerbench(rag_file)
    rag_models = [name for name in rag.columns if "|model_response" in name]
    rag_models = [name.split("|",1)[0] for name in rag_models]
    rag_rows = normalize(rag[["sample_id","eval_name","prompt",*rag_models]])
    articles = dict(zip(rag.sample_id.astype(str),rag.source.astype(str)))
    for row in rag_rows:
        row["article_source"] = articles[row["sample_id"]]
        row["source_eval_name"] = row["eval_name"]
        row["metric_kind"] = "published_RAG_judged_outcome"
    partition_rows(rag_rows,by_article=True)
    rag_groups,ranking = rank_groups(rag_rows)
    add_targets(rag_rows,rag_groups,min_coverage=1)
    rag_manifest = {"source":str(rag_file),"source_sha256":file_sha256(rag_file),"groups":rag_groups,"group_order":list(GROUPS),
                    "seed":42,"min_coverage":1,"training_macro_quality":ranking,"cohort":"rag-5-models",
                    "target_definition":"one minus mean published RAG evaluation score per RAG-specific pool",
                    "label_provenance":{"rag_revision":"784e39ca67a6c6291685b4862a05dd95fb0b40da","sha256":file_sha256(rag_file)},
                    "pool_definition":"RAG-specific five-model roster, ranked using training source articles only; not the main eleven-model pools.",
                    "split_method":"Source-article and duplicate-question components; article-cluster bootstrap."}
    _save_prepared(destination/"rag",rag_rows,rag_manifest)
    write_json(destination/"preparation_protocol.json",provenance)
    return {cohort:json.loads((destination/cohort/"coverage.json").read_text()) for cohort in ("main","rag")}


def _save_prepared(destination, rows, manifest):
    destination.mkdir(parents=True,exist_ok=True)
    identities = defaultdict(set)
    for row in rows:
        identities[row["prompt_hash"]].add(row["split"])
    if any(len(splits)>1 for splits in identities.values()):
        raise ValueError("Canonical task text leaks across splits")
    coverage = {"questions":len(rows),"split_counts":dict(Counter(r["split"] for r in rows)),
                "families":dict(Counter(r["eval_name"] for r in rows)),"source_evals":dict(Counter(r["source_eval_name"] for r in rows)),
                "source_conditions":dict(Counter(r.get("source_condition","archived-RAG") for r in rows)),
                "questions_without_observed_pool_target":sum(not any(v is not None for v in r["difficulty"].values()) for r in rows),
                "observed_pool_targets":{g:sum(r["difficulty"][g] is not None for r in rows) for g in GROUPS},
                "task_cross_split_overlaps":0,"label_kinds":dict(Counter(r["metric_kind"] for r in rows))}
    for split in ("train","validation","test"):
        with (destination/f"{split}.jsonl").open("w",encoding="utf-8") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    manifest = {**manifest,"datasets":sorted(coverage["families"]),"split_counts":coverage["split_counts"],
                "evaluation_status":"Exploratory all-family study; historical benchmark and development probes previously inspected."}
    write_json(destination/"manifest.json",manifest)
    write_json(destination/"coverage.json",coverage)


def train_all(data, original_prepared, gsm_audit, encoder, output, device="auto"):
    output = Path(output)
    coverage = prepare_all(data,original_prepared,gsm_audit,output/"prepared")
    write_json(output/"coverage.json",coverage)
    reports = {}
    for cohort in ("main","rag"):
        print(f"Training {cohort}: {coverage[cohort]['questions']} questions",flush=True)
        configurations=list(CONFIGURATIONS)
        if cohort=="main":
            manifest=json.loads((output/"prepared"/cohort/"manifest.json").read_text())
            configurations.append({"name":"task_model_sqrt_mse","input_mode":"task","balanced":True,"balance_power":.5,
                                   "loss":"mse","prediction_level":"model","groups":manifest["groups"],
                                   "model_order":[m for g in GROUPS for m in manifest["groups"][g]]})
        reports[cohort] = run_suite(output/"prepared"/cohort,encoder,output/cohort,seeds=(42,43,44),epochs=100,patience=12,
                                     configurations=configurations,compare_legacy=False,encoder_batch_size=32,device=device)
        gc.collect()
        torch.cuda.empty_cache()
    return reports


def probe_overlaps(prepared, questions):
    """Label any natural corpus overlap; a development query is not a holdout."""
    overlaps = {question: [] for question in questions}
    keys = {_fingerprint(task_text(question)): question for question in questions}
    for split in ("train", "validation", "test"):
        with (Path(prepared) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                key = _fingerprint(task_text(row["prompt"]))
                if key in keys:
                    overlaps[keys[key]].append({"split":split,"sample_id":row["sample_id"],
                                                "source_eval_name":row["source_eval_name"]})
    return overlaps


def finalize_all(output, device="auto"):
    """Fit validation-only support checks and verify the selected saved models.

    The main and RAG checkpoints keep separate rosters and contracts. Probes and
    old-model comparisons are descriptive and cannot change either selection.
    """
    output = Path(output)
    import sys
    import transformers
    write_json(output / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "transformers": transformers.__version__,
        "requested_device": device, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "encoder_training": "Frozen; CUDA embedding extraction when selected, CPU MLP head fitting.",
    })
    evaluations = {}
    for cohort in ("main", "rag"):
        prepared = output / "prepared" / cohort
        directory = output / cohort / "selected_model"
        if cohort == "main":
            # Correct the descriptive legacy split field; actual task-component
            # folds and the predictor's pool/label contract remain unchanged.
            for path in (prepared / "manifest.json", directory / "manifest.json"):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["split_method"] = "Connected canonical task texts/sample IDs; 70/15/15 hash folds anchored to original 0shot-export prompts where available."
                write_json(path, manifest)
        support = attach_support(directory, prepared, device=device)
        predictor = DifficultyPredictor.load(directory, device=device)
        predictor.encoder.batch_size = 32
        rows = load_splits(prepared)["test"]
        truth = targets(rows)
        results = predictor.predict_with_support([r["prompt"] for r in rows])
        prediction = np.array([[r["prediction"][g] for g in GROUPS] for r in results], dtype=np.float32)
        saved = read_saved_predictions(directory, rows)
        parity = float(np.max(np.abs(prediction - saved)))
        if parity > 1e-6:
            raise ValueError(f"{cohort} saved checkpoint does not reproduce evaluation predictions: {parity}")
        supported = np.array([r["input_support"]["within_support"] for r in results])
        selection = json.loads((output / cohort / "selection.json").read_text())
        recommendation = {
            "model_relative_path": f"{cohort}/selected_model", "cohort": predictor.manifest["cohort"],
            "scope": "All released non-RAG RouterBench families" if cohort == "main" else "Archived news/wiki RAG questions and its five-model roster",
            "selection": selection["criterion"], "validation": selection["validation"],
            "predictor_id": predictor.metadata["predictor_id"], "pool_contract_id": contract_id(predictor.manifest),
            "groups": predictor.manifest["groups"], "target_definition": predictor.manifest["target_definition"],
            "input_support": support,
        }
        report = {"recommendation": recommendation, "heldout_questions": len(rows),
                  "prediction": summarize(prediction, truth, rows),
                  "strong_vs_weak_gain": gain_diagnostics(prediction, truth, rows),
                  "saved_prediction_max_abs_difference": parity, "input_support_fraction": float(supported.mean()),
                  "support_interpretation": "Similarity heuristic only. Abstention does not repair raw scores or guarantee correctness."}
        report["observed_endpoint_subgroups"] = {}
        for label, mask in (("all_pool_targets_zero", np.isfinite(truth).all(axis=1) & (truth == 0).all(axis=1)),
                            ("all_pool_targets_one", np.isfinite(truth).all(axis=1) & (truth == 1).all(axis=1))):
            if mask.any():
                report["observed_endpoint_subgroups"][label] = {
                    "questions": int(mask.sum()), "mean_predicted_difficulty": dict(zip(GROUPS,map(float,prediction[mask].mean(axis=0)))),
                    "prediction": summarize(prediction[mask],truth[mask],[r for r,m in zip(rows,mask) if m]),
                    "interpretation": "Descriptive subset of recorded held-out outcomes; not external short-prompt validation.",
                }
        for name, mask in (("supported", supported), ("unsupported", ~supported)):
            if mask.any():
                report[name] = {"questions": int(mask.sum()), "prediction": summarize(prediction[mask], truth[mask],
                                                                                       [r for r,m in zip(rows,mask) if m])}
        if cohort == "main":
            probes = PROBES + ["Which planet is known as the Red Planet?\nA) Earth\nB) Mars\nC) Venus\nD) Jupiter",
                               "Write a friendly two-sentence birthday message.",
                               "A person drops a glass onto a hard floor. What is most likely to happen next?"]
            report["development_probes"] = [{"question": q, **r} for q,r in
                                             zip(probes,predictor.predict_with_support(probes))]
            overlaps = probe_overlaps(prepared, probes)
            for probe in report["development_probes"]:
                probe["corpus_overlap"] = overlaps[probe["question"]]
            write_json(output / "probe_overlap.json", overlaps)
            old = Path(__file__).resolve().parent / "runs/gsm_correctness_study_v2/correctness/selected_model"
            if (old / "test_predictions.jsonl").is_file():
                old_records = [json.loads(line) for line in (old / "test_predictions.jsonl").read_text().splitlines() if line]
                lookup = {r["sample_id"]: i for i,r in enumerate(rows)}
                if any(r["sample_id"] not in lookup for r in old_records):
                    raise ValueError("Historical complete-case GSM test questions are missing from the broad test set")
                indices = [lookup[r["sample_id"]] for r in old_records]
                matched_rows = [rows[i] for i in indices]
                matched_truth = truth[indices]
                old_truth = np.array([[r["target"][g] for g in GROUPS] for r in old_records],dtype=float)
                if not np.allclose(old_truth,matched_truth,equal_nan=True):
                    raise ValueError("Historical GSM comparison must have identical targets")
                old_prediction = np.array([[r["prediction"][g] for g in GROUPS] for r in old_records],dtype=np.float32)
                report["matched_historical_gsm"] = {
                    "questions": len(indices), "older_math_only": summarize(old_prediction,matched_truth,matched_rows),
                    "new_all_family": summarize(prediction[indices],matched_truth,matched_rows),
                    "paired_new_vs_old": paired_interval(old_prediction,prediction[indices],matched_truth,matched_rows),
                    "interpretation": "Identical historical complete-case GSM test questions and reference labels; exploratory comparison."}
        write_json(output / cohort / "release_evaluation.json", report)
        # Only advertise a default after reload parity and the release audit pass.
        filename = "recommendation.json" if cohort == "main" else "rag_recommendation.json"
        write_json(output / filename, recommendation)
        evaluations[cohort] = report
        del predictor, results
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Verified {cohort}: {len(rows)} held-out questions, maximum reload difference {parity:.2g}",flush=True)
    write_json(output / "release_evaluation.json", evaluations)
    return evaluations
