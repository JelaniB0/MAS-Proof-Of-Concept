"""Validation-selected, reproducible frozen-MPNet ablations and final comparison."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .audit import audit_prepared
from .data import GROUPS, write_json
from .evaluation import (summarize, gain_diagnostics, utility_diagnostics, baseline_predictions,
                         read_saved_predictions, paired_interval)
from .frozen import FrozenMPNet, PoolHead, export_ensemble, aggregate_model_difficulty
from .inputs import INPUT_VERSION, input_version, contract_id, input_text
from .model import load_splits, targets
from .transformer import combined_loss

CONFIGURATIONS = [
    {"name": "full_mse", "input_mode": "full", "balanced": False, "loss": "mse"},
    {"name": "question_mse", "input_mode": "question", "balanced": False, "loss": "mse"},
    {"name": "question_balanced_mse", "input_mode": "question", "balanced": True, "loss": "mse"},
    {"name": "question_balanced_combined", "input_mode": "question", "balanced": True, "loss": "combined"},
]


def fit_head(train_x, validation_x, rows, config, directory, seed, epochs=100, patience=12):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    per_model=config.get("prediction_level") == "model"
    model_order=config.get("model_order")
    head = PoolHead(train_x.shape[1],output_dim=len(model_order) if per_model else None)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    train_y=targets(rows["train"])
    if per_model:
        train_y=np.array([[1-row['scores'][m] if m in row['scores'] else np.nan for m in model_order]
                          for row in rows['train']],dtype=np.float32)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    val_x, val_y = torch.from_numpy(validation_x), targets(rows["validation"])
    generator = torch.Generator().manual_seed(seed)
    counts = Counter(r["eval_name"] for r in rows["train"])
    balance_power = config.get("balance_power", 1.0)
    if not 0 <= balance_power <= 1:
        raise ValueError("balance_power must be in [0, 1]")
    weights = torch.tensor([1 / counts[r["eval_name"]] ** balance_power for r in rows["train"]], dtype=torch.double)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        indices = (torch.multinomial(weights, len(x), replacement=True, generator=generator) if config["balanced"]
                   else torch.randperm(len(x), generator=generator))
        head.train()
        for batch in indices.split(256):
            logits = head(x[batch])
            if config["loss"] == "combined":
                loss = combined_loss(logits, y[batch])
            else:
                mask = torch.isfinite(y[batch])
                loss = ((logits.sigmoid() - torch.nan_to_num(y[batch])).square() * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            prediction = head(val_x).sigmoid().numpy()
        if per_model:
            prediction=aggregate_model_difficulty(prediction,model_order,config["groups"])
        evaluation = summarize(prediction, val_y, rows["validation"])
        score = evaluation["dataset_macro_mse"]
        history.append({"epoch": epoch, "validation_dataset_macro_mse": score,
                        "validation_question_weighted_mse": evaluation["macro_mse"]})
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f"{config['name']} seed={seed} epoch={epoch} validation_dataset_mse={score:.6f}", flush=True)
        if stale >= patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        prediction = head(val_x).sigmoid().numpy()
    if per_model:
        prediction=aggregate_model_difficulty(prediction,model_order,config["groups"])
    torch.save(head.state_dict(), directory / "head.pt")
    np.save(directory / "validation_predictions.npy", prediction, allow_pickle=False)
    report = {"seed": seed, "config": config, "best_epoch": best_epoch,
              "validation": summarize(prediction, val_y, rows["validation"]), "history": history,
              "seconds": time.perf_counter() - started,
              "observed_training_targets":int(np.isfinite(train_y).sum())}
    write_json(directory / "training.json", report)
    return report, prediction


def run_suite(prepared, encoder, output, seeds=(42, 43, 44), epochs=100, patience=12, device="auto",
              configurations=None, compare_legacy=True, encoder_batch_size=16):
    if min(epochs, patience) < 1 or not seeds:
        raise ValueError("Positive epochs, patience and at least one seed are required")
    configurations = CONFIGURATIONS if configurations is None else configurations
    if not configurations or len({c["name"] for c in configurations}) != len(configurations):
        raise ValueError("Configurations must be nonempty and uniquely named")
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_splits(prepared)
    audit = audit_prepared(prepared)
    if audit["question_only_cross_split_overlaps"]:
        raise ValueError("Question-focused representation exposes cross-split duplicates; reprepare data first")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    extractor = FrozenMPNet(encoder, device=device, batch_size=encoder_batch_size)
    protocol = {"version": "frozen-study-v1", "input_version": INPUT_VERSION, "pool_contract_id": contract_id(manifest),
                "prepared_sha256": {split: hashlib.sha256((prepared / f"{split}.jsonl").read_bytes()).hexdigest()
                                    for split in rows},
                "encoder_weights_sha256": extractor.weights_sha256, "seeds": list(seeds),
                "encoder_configuration_sha256": extractor.configuration_sha256, "tokenizer_sha256": extractor.tokenizer_sha256,
                "configurations": configurations, "epochs": epochs, "patience": patience,
                "head": {"hidden_dim": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 256},
                "selection": "Lowest ensemble validation MSE, averaging datasets equally; test does not select configuration.",
                "test_status": "Existing test split has been inspected previously; this is an exploratory comparison, not a fresh confirmatory test."}
    if any(config["input_mode"] == "task" for config in configurations):
        protocol["input_versions"] = {config["input_mode"]:input_version(config["input_mode"]) for config in configurations}
    if (output / "protocol.json").is_file():
        previous = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if protocol != previous:
            raise ValueError("Output already contains a different experiment protocol; choose a new directory")
    write_json(output / "protocol.json", protocol)
    write_json(output / "data_audit.json", audit)
    cache = prepared.parent / "frozen_embedding_cache"
    embeddings, truncation = {}, {}
    for mode in dict.fromkeys(config["input_mode"] for config in configurations):
        embeddings[mode], truncation[mode] = {}, {}
        for split in ("train", "validation"):
            print(f"Encoding {mode}/{split}: {len(rows[split])} questions", flush=True)
            embeddings[mode][split], truncation[mode][split] = extractor.cached_encode(rows[split], cache, mode)
    results = []
    for config in configurations:
        seed_results, validation_predictions, files = [], [], []
        for seed in seeds:
            directory = output / "candidates" / config["name"] / f"seed_{seed}"
            if (directory / "training.json").is_file():
                report = json.loads((directory / "training.json").read_text(encoding="utf-8"))
                prediction = np.load(directory / "validation_predictions.npy", allow_pickle=False)
            else:
                report, prediction = fit_head(embeddings[config["input_mode"]]["train"],
                                              embeddings[config["input_mode"]]["validation"], rows, config,
                                              directory, seed, epochs, patience)
            seed_results.append(report)
            validation_predictions.append(prediction)
            files.append(str(directory / "head.pt"))
        mean_prediction = np.mean(validation_predictions, axis=0)
        result = {"config": config, "seeds": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                                               "validation_dataset_macro_mse": r["validation"]["dataset_macro_mse"]}
                                              for r in seed_results],
                  "ensemble_validation": summarize(mean_prediction, targets(rows["validation"]), rows["validation"]),
                  "head_files": files}
        results.append(result)
        write_json(output / "validation_results.json", results)
        print(config["name"], json.dumps({key: result["ensemble_validation"][key]
                                         for key in ("macro_mse", "dataset_macro_mse")}), flush=True)
    winner = min(results, key=lambda value: value["ensemble_validation"]["dataset_macro_mse"])
    write_json(output / "selection.json", {"configuration": winner["config"], "criterion": protocol["selection"],
                                           "validation": winner["ensemble_validation"], "seeds": list(seeds)})
    destination = output / "selected_model"
    metadata = {"input_mode": winner["config"]["input_mode"], "input_dim": extractor.encoder.config.hidden_size,
                "hidden_dim": 128, "dropout": 0.2, "max_length": 384,
                "encoder_weights_sha256": extractor.weights_sha256, "pool_contract_id": contract_id(manifest),
                "encoder_configuration_sha256": extractor.configuration_sha256, "tokenizer_sha256": extractor.tokenizer_sha256,
                "selection": protocol["selection"], "seeds": list(seeds),
                "quality_definition": manifest.get("target_definition", "One minus mean observed normalized RouterBench score per pool")}
    if winner["config"].get("prediction_level") == "model":
        metadata.update({"prediction_level":"model","model_order":winner["config"]["model_order"]})
    export_ensemble(encoder, winner["head_files"], destination, metadata, manifest)
    # The selected configuration is frozen before any new test predictions are made.
    mode = winner["config"]["input_mode"]
    test_x, test_truncation = extractor.cached_encode(rows["test"], cache, mode)
    heads = []
    for filename in winner["head_files"]:
        head = PoolHead(test_x.shape[1],output_dim=len(metadata["model_order"]) if metadata.get("prediction_level")=="model" else None)
        head.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
        heads.append(head.eval())
    with torch.inference_mode():
        predictions = np.mean([head(torch.from_numpy(test_x)).sigmoid().numpy() for head in heads], axis=0)
    per_model_predictions=predictions if metadata.get("prediction_level")=="model" else None
    if per_model_predictions is not None:
        predictions=aggregate_model_difficulty(predictions,metadata["model_order"],manifest["groups"])
    test_y = targets(rows["test"])
    comparisons = baseline_predictions(rows["train"], rows["test"])
    comparisons["selected_frozen_ensemble"] = predictions
    legacy = prepared.parents[1] / "runs"
    for label, folder in (("legacy_tfidf_mlp", "routerbench_overlap_tfidf"), ("legacy_finetuned_mpnet", "agentbalance_mpnet")):
        if compare_legacy and (legacy / folder / "test_predictions.jsonl").is_file():
            comparisons[label] = read_saved_predictions(legacy / folder, rows["test"])
    evaluation = {name: {"prediction": summarize(p, test_y, rows["test"]),
                         "strong_vs_weak_gain": gain_diagnostics(p, test_y, rows["test"]),
                         "illustrative_tier_utility": utility_diagnostics(p, test_y)} for name, p in comparisons.items()}
    intervals = {name: paired_interval(p, predictions, test_y, rows["test"])
                 for name, p in comparisons.items() if name != "selected_frozen_ensemble"}
    report = {"protocol": protocol, "selected_configuration": winner["config"], "selected_validation": winner["ensemble_validation"],
              "test": evaluation, "paired_comparisons": intervals, "truncation": {"development": truncation, "test": test_truncation}}
    write_json(output / "comparison.json", report)
    write_json(destination / "metrics.json", evaluation["selected_frozen_ensemble"])
    with (destination / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index,(row, values) in enumerate(zip(rows["test"], predictions)):
            record={"sample_id": row["sample_id"], "eval_name": row["eval_name"],
                    "target": row["difficulty"], "prediction": dict(zip(GROUPS, map(float, values)))}
            if per_model_predictions is not None:
                record["model_difficulty"]=dict(zip(metadata["model_order"],map(float,per_model_predictions[index])))
            stream.write(json.dumps(record) + "\n")
    write_summary(output, report, results)
    return report


def write_summary(output, report, validation_results):
    lines = ["# Frozen MPNet experiment results", "", report["protocol"]["test_status"], "",
             "Selection uses equal-weight dataset validation MSE. Test results do not select a model.", "",
             "| Configuration | Validation dataset macro MSE | Validation question-weighted MSE |", "| --- | ---: | ---: |"]
    for result in validation_results:
        score = result["ensemble_validation"]
        lines.append(f"| {result['config']['name']} | {score['dataset_macro_mse']:.6f} | {score['macro_mse']:.6f} |")
    datasets=sorted({dataset for value in report["test"].values() for dataset in value["prediction"]["by_dataset"]})
    lines += ["", f"Selected: **{report['selected_configuration']['name']}** (ensemble of {len(report['protocol']['seeds'])} seeds).", "",
              "Pool MSEs receive equal weight; missing pool targets are excluded independently.", "",
              "| Model | Test pooled MSE | Test dataset macro MSE | "+" | ".join(dataset+" MSE" for dataset in datasets)+" |",
              "| --- | ---: | ---: | "+" | ".join("---:" for _ in datasets)+" |"]
    for name, value in report["test"].items():
        score = value["prediction"]
        by_dataset = score["by_dataset"]
        values=[by_dataset.get(dataset,{}).get('macro_mse') for dataset in datasets]
        cells=[f"{value:.6f}" if value is not None else "N/A" for value in values]
        lines.append(f"| {name} | {score['macro_mse']:.6f} | {score['dataset_macro_mse']:.6f} | "+" | ".join(cells)+" |")
    lines += ["", "Normalized tier-utility scenarios use invented [1,2,3] cost units. They are diagnostics, not API cost savings or BAMAS results.",
              "", "A positive paired MSE reduction favors the selected model; intervals containing zero do not establish a clear advantage.", ""]
    for name, value in report["paired_comparisons"].items():
        lines.append(f"- Versus {name}: question-weighted MSE reduction {value['mean_mse_reduction']:.6f}; paired 95% interval {value['ci95']}. "
                     f"Dataset macro reduction {value['dataset_macro_mean_mse_reduction']:.6f}; interval {value['dataset_macro_ci95']}.")
    (Path(output) / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
