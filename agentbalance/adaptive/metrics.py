"""Appendix D upper concave envelopes, interpolation, and bounded AUC."""

import csv
import json
from pathlib import Path
import statistics

PAPER_BUDGETS = {
    "mmlu": {"cost": [.07, .13, .32, .75], "latency": [19., 42., 63., 135.]},
    "math": {"cost": [.08, .17, .50, 1.05], "latency": [28., 51., 100., 210.]},
    "humaneval": {"cost": [.03, .04, .12, .60], "latency": [9., 11., 40., 100.]},
}


def frontier(points, anchor):
    """Upper concave envelope, anchored at zero and the shared base endpoint."""
    end, base_score = anchor
    if end <= 0:
        return None  # Zero-price experiments cannot establish dollar efficiency.
    values = {0.: 0., end: base_score}
    for budget, score in points:
        if budget < 0 or not 0 <= score <= 1:
            raise ValueError("Invalid frontier point")
        if 0 < budget <= end:
            values[budget] = max(values.get(budget, 0.), score)
    hull = []
    for point in sorted(values.items()):
        while len(hull) >= 2:
            a, b = hull[-2:]
            # Slopes must decrease along an upper concave envelope.
            cross = (b[0]-a[0])*(point[1]-b[1])-(b[1]-a[1])*(point[0]-b[0])
            if cross < 0:
                break
            hull.pop()
        hull.append(point)
    return hull


def performance_at_budget(hull, budget):
    if not hull or budget < hull[0][0] or budget > hull[-1][0]:
        return None  # Never extrapolate outside the observed shared range.
    for a, b in zip(hull, hull[1:]):
        if a[0] <= budget <= b[0]:
            return a[1]+(b[1]-a[1])*(budget-a[0])/(b[0]-a[0])
    return hull[-1][1]


def auc(hull):
    if not hull:
        return None
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(hull, hull[1:]))


def measurement(report, allow_fixtures=False):
    if report.get("status") != "completed" or report.get("evaluated") != report.get("requested"):
        raise ValueError("Frontiers require complete evaluations; partial runs are excluded")
    runs = report["runs"]
    real = [r["report"] for r in runs if "report" in r]
    if not real:
        raise ValueError("Evaluation contains no executed graphs")
    if any(not r["totals"]["cost_complete"] for r in real):
        raise ValueError("Unknown usage cannot be used for a cost frontier")
    if not allow_fixtures and any(r["mode"] != "live" for r in real):
        raise ValueError("Simulated inference cannot establish a measured frontier")
    keys = sorted(r["key"] for r in runs)
    return {"accuracy": report["accuracy"], "cost": sum(r["totals"]["cost_usd"] for r in real),
            "latency": statistics.mean(r["totals"]["latency_seconds"] for r in real),
            "tokens": sum(r["totals"]["input_tokens"]+r["totals"]["output_tokens"] for r in real),
            "calls": sum(r["totals"]["calls"] for r in real), "keys": keys,
            "failed": sum(r["status"] != "completed" for r in real),
            "dataset_fingerprint": report.get("dataset_fingerprint")}


def summarize(index_path, output, allow_fixtures=False):
    index_path, output = Path(index_path), Path(output)
    entries = json.loads(index_path.read_text())["evaluations"]
    rows, excluded = [], []
    for entry in entries:
        path = Path(entry["report"])
        if not path.is_absolute():
            path = index_path.parent/path
        try:
            m = measurement(json.loads(path.read_text(encoding="utf-8")), allow_fixtures)
            rows.append({**entry, **m})
        except (OSError, ValueError) as exc:
            excluded.append({"report": str(path), "reason": str(exc)})
    results = {}
    for benchmark in sorted({r["benchmark"] for r in rows}):
        selected = [r for r in rows if r["benchmark"] == benchmark]
        if len({tuple(r["keys"]) for r in selected}) != 1:
            raise ValueError("Compared evaluations used different tasks")
        if len({r["dataset_fingerprint"] for r in selected}) != 1:
            raise ValueError("Compared evaluations used different labels or tests")
        base = [r for r in selected if r.get("base")]
        if not base:
            raise ValueError(f"No completed base evaluation for {benchmark}")
        methods = {}
        for method in sorted({r["method"] for r in selected}):
            items = [r for r in selected if r["method"] == method]
            # Calculate each seed's frontier before averaging metrics.
            seed_results = []
            for seed in sorted({r["seed"] for r in items}):
                points = [r for r in items if r["seed"] == seed]
                anchors = [r for r in base if r["seed"] == seed]
                if not anchors:
                    continue
                anchor = anchors[0]
                entry = {"seed": seed}
                for axis in ("cost", "latency"):
                    hull = frontier([(r[axis], r["accuracy"]) for r in points], (anchor[axis], anchor["accuracy"]))
                    entry[axis] = {"frontier": hull, "auc": auc(hull),
                                   "p_at": [performance_at_budget(hull, b) for b in PAPER_BUDGETS[benchmark][axis]],
                                   "budgets": PAPER_BUDGETS[benchmark][axis]}
                seed_results.append(entry)
            methods[method] = {"seeds": seed_results, "accuracy_mean": statistics.mean(r["accuracy"] for r in items),
                               "accuracy_stdev": statistics.stdev(r["accuracy"] for r in items) if len(items) > 1 else None}
            for axis in ("cost", "latency"):
                values = [s[axis]["auc"] for s in seed_results if s[axis]["auc"] is not None]
                methods[method][axis+"_auc_mean"] = statistics.mean(values) if values else None
                methods[method][axis+"_auc_stdev"] = statistics.stdev(values) if len(values)>1 else None
                budget_metrics = []
                for i, budget in enumerate(PAPER_BUDGETS[benchmark][axis]):
                    values = [s[axis]["p_at"][i] for s in seed_results if s[axis]["p_at"][i] is not None]
                    budget_metrics.append({"budget": budget, "seed_count": len(values),
                        "mean": statistics.mean(values) if values else None,
                        "stdev": statistics.stdev(values) if len(values)>1 else None})
                methods[method][axis+"_p_at"] = budget_metrics
        results[benchmark] = methods
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps({"benchmarks": results, "excluded": excluded,
        "note": "Accuracy is a fraction; cost is total evaluation USD; latency is mean graph seconds. "
                "AUC uses a shared base endpoint, without extrapolation. Zero-cost AUC is undefined. "
                "Fixed baselines here are not AgentPrune/AgentDropout/G-Designer/MasRouter."}, indent=2)+"\n")
    fields = ["benchmark", "method", "seed", "point", "accuracy", "cost", "latency", "tokens", "calls", "failed", "report"]
    with (output/"runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        raise ValueError("No complete evaluations to summarize; inspect excluded reports")
    return results
