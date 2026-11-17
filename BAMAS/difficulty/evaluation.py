"""Prediction quality and explicit offline tier-utility diagnostics."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import GROUPS
from .model import metrics, targets


def safe_spearman(prediction, truth):
    # Float32 std() can be nonzero even for identical values due to roundoff.
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if len(p) < 2 or np.ptp(p) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(p, y).statistic)
    return value if np.isfinite(value) else None


def summarize(prediction, truth, rows):
    result = metrics(prediction, truth)
    by_dataset = {}
    for dataset in sorted({r["eval_name"] for r in rows}):
        mask = np.array([r["eval_name"] == dataset for r in rows])
        by_dataset[dataset] = metrics(prediction[mask], truth[mask])
    result["by_dataset"] = by_dataset
    available = [value["macro_mse"] for value in by_dataset.values() if value["macro_mse"] is not None]
    result["dataset_macro_mse"] = float(np.mean(available)) if available else None
    for i, group in enumerate(GROUPS):
        mask = np.isfinite(truth[:, i])
        p, y = prediction[mask, i], truth[mask, i]
        if not len(y):
            result[group].update({"prediction_std": None, "target_std": None, "mean_bias": None, "spearman": None})
            continue
        result[group].update({"prediction_std": float(p.std()), "target_std": float(y.std()),
                              "mean_bias": float(np.mean(p - y)),
                              "spearman": safe_spearman(p, y)})
    return result


def gain_diagnostics(prediction, truth, rows, threshold=0.1):
    result = {"threshold": threshold, "interpretation": "Recorded strong-minus-weak pool mean quality, not agent or token-cap utility."}
    true_gain, predicted_gain = truth[:, 0] - truth[:, 2], prediction[:, 0] - prediction[:, 2]
    for dataset in ["all"] + sorted({r["eval_name"] for r in rows}):
        mask = np.ones(len(rows), dtype=bool) if dataset == "all" else np.array([r["eval_name"] == dataset for r in rows])
        mask &= np.isfinite(true_gain)
        y, p = true_gain[mask], predicted_gain[mask]
        if not len(y):
            result[dataset] = {"n": 0, "positive_rate": None, "gain_mae": None, "auc": None, "spearman": None}
            continue
        event = y > threshold
        result[dataset] = {"n": len(y), "positive_rate": float(event.mean()), "gain_mae": float(np.mean(np.abs(p - y))),
                           "auc": float(roc_auc_score(event, p)) if len(np.unique(event)) == 2 else None,
                           "spearman": safe_spearman(p, y)}
    return result


def utility_diagnostics(prediction, truth):
    valid = np.isfinite(truth).all(axis=1)
    prediction, truth = prediction[valid], truth[valid]
    tier_costs = np.array([1.0, 2.0, 3.0])
    result = {"interpretation": "Illustrative offline pool-mean utility with arbitrary tier costs; NOT measured dollars, token use or MAS outcomes.",
              "tier_costs": tier_costs.tolist(), "points": []}
    if not len(truth):
        return result
    for coefficient in (0.0, 0.025, 0.05, 0.1, 0.2):
        choices = np.argmax(1 - prediction - coefficient * tier_costs, axis=1)
        measured_utility = 1 - truth - coefficient * tier_costs
        selected = measured_utility[np.arange(len(truth)), choices]
        result["points"].append({"cost_weight": coefficient, "mean_recorded_quality": float(np.mean(1 - truth[np.arange(len(truth)), choices])),
                                 "mean_proxy_cost": float(tier_costs[choices].mean()),
                                 "mean_utility": float(selected.mean()),
                                 "oracle_regret": float((measured_utility.max(axis=1) - selected).mean()),
                                 "selection_counts": {g: int(np.sum(choices == i)) for i, g in enumerate(GROUPS)}})
    return result


def baseline_predictions(training_rows, evaluation_rows):
    train_y = targets(training_rows)
    means = np.nanmean(train_y, axis=0)
    constant = np.tile(means, (len(evaluation_rows), 1))
    by_dataset = {}
    for dataset in {r["eval_name"] for r in training_rows}:
        subset = train_y[[r["eval_name"] == dataset for r in training_rows]]
        values = np.nanmean(subset, axis=0)
        by_dataset[dataset] = np.where(np.isfinite(values), values, means)
    dataset_mean = np.array([by_dataset.get(r["eval_name"], means) for r in evaluation_rows])
    return {"global_mean": constant, "dataset_mean": dataset_mean}


def read_saved_predictions(run, rows):
    path = Path(run) / "test_predictions.jsonl"
    with path.open(encoding="utf-8") as stream:
        by_key = {(r["eval_name"], r["sample_id"]): r for r in map(json.loads, stream) if r}
    keys = [(r["eval_name"], r["sample_id"]) for r in rows]
    if set(keys) != set(by_key):
        raise ValueError("Saved predictions are for a different test set")
    truth = np.array([[by_key[k]["target"][g] for g in GROUPS] for k in keys], dtype=float)
    if not np.allclose(truth, targets(rows), equal_nan=True):
        raise ValueError("Saved predictions use different target labels")
    return np.array([[by_key[k]["prediction"][g] for g in GROUPS] for k in keys], dtype=np.float32)


def paired_interval(reference, candidate, truth, rows, repetitions=3000, seed=42):
    """Paired family-stratified bootstrap, respecting missing pools and source clusters.

    The estimand matches metrics(): average the pool-specific MSEs. Averaging
    question-wise errors would give a different answer when pools have missing
    labels. RAG rows may set bootstrap_cluster to their shared source article.
    """
    valid = np.isfinite(truth)
    differences = np.where(valid, (reference - np.nan_to_num(truth)) ** 2 - (candidate - np.nan_to_num(truth)) ** 2, 0)

    def average(sums, counts):
        observed = counts > 0
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=observed)
        n = observed.sum(axis=-1)
        return np.divide(means.sum(axis=-1), n, out=np.full_like(n, np.nan, dtype=float), where=n > 0)

    point = float(average(differences.sum(axis=0), valid.sum(axis=0)))
    if not np.isfinite(point):
        raise ValueError("Paired comparison needs observed target pools")
    rng = np.random.default_rng(seed)
    total_sums, total_counts = np.zeros((repetitions, len(GROUPS))), np.zeros((repetitions, len(GROUPS)))
    dataset_samples, dataset_means = [], []
    clustered = False
    for dataset in sorted({r["eval_name"] for r in rows}):
        indices = np.flatnonzero([r["eval_name"] == dataset for r in rows])
        clusters = {}
        for index in indices:
            key = rows[index].get("bootstrap_cluster", rows[index].get("sample_id", str(index)))
            clusters.setdefault(key, []).append(index)
        members = list(clusters.values())
        clustered |= len(members) < len(indices)
        sums = np.array([differences[group].sum(axis=0) for group in members])
        counts = np.array([valid[group].sum(axis=0) for group in members])
        family_point = float(average(sums.sum(axis=0), counts.sum(axis=0)))
        if not np.isfinite(family_point):
            continue
        draws = rng.integers(0, len(members), size=(repetitions, len(members)))
        sampled_sums, sampled_counts = sums[draws].sum(axis=1), counts[draws].sum(axis=1)
        total_sums += sampled_sums
        total_counts += sampled_counts
        dataset_samples.append(average(sampled_sums, sampled_counts))
        dataset_means.append(family_point)
    samples = average(total_sums, total_counts)
    return {"mean_mse_reduction": point, "ci95": np.nanquantile(samples, [0.025, 0.975]).tolist(),
            "dataset_macro_mean_mse_reduction": float(np.mean(dataset_means)),
            "dataset_macro_ci95": np.nanquantile(np.nanmean(dataset_samples, axis=0), [0.025, 0.975]).tolist(),
            "repetitions": repetitions, "method": "Paired source-cluster bootstrap stratified by dataset; positive means candidate is better." if clustered else
                "Paired question bootstrap stratified by dataset; positive means candidate is better."}
