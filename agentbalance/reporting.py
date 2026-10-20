"""Readable, portable summaries of existing runs; never makes model requests.

Reports describe observed decisions, not a recovered global policy rule. Older
rollouts on different queries cannot establish how one query's routing changed.
New training reports include deterministic probes at each optimizer update.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from .files import write_json, replace_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def policy_decision(plan):
    """Keep interpretable choices/probabilities; omit prompts and embeddings."""
    selection = plan.get("selection", {})
    assignments = plan.get("assignments", {})
    return {
        "topology": plan.get("topology"), "assignments": assignments,
        "active_agents": len(assignments),
        "active_workers": sum(role != "synthesizer" for role in assignments),
        "edges": plan.get("graph_edges", plan.get("edges", [])),
        "difficulty": plan.get("difficulty"),
        **{key: selection[key] for key in (
            "ease", "effective_difficulty", "selected_pool", "pool_probabilities",
            "pools", "filtered_models", "model_probabilities", "keep_probabilities",
            "edge_probabilities", "hop_limit", "hop_probabilities", "pruned_agents",
            "hop_pruned_edges", "sampled", "feasible", "joint_log_probability") if key in selection},
    }


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    replace_file(temporary, path)


def _markdown(path, lines):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def _accuracy(value):
    return f"{value:.1%}" if value is not None else "unavailable"


def _usage(graph):
    totals = graph.get("totals", {})
    return {**totals, "tokens": totals.get("input_tokens", 0) + totals.get("output_tokens", 0)}


def question_preview(task):
    lines = task.splitlines() if isinstance(task, str) else []
    return lines[0][:160] if lines else ""


def _policy_overview(decisions):
    assignments = Counter(model for decision in decisions for model in decision.get("assignments", {}).values())
    signatures = {json.dumps({"assignments": d.get("assignments"), "edges": d.get("edges")}, sort_keys=True)
                  for d in decisions}
    return {"model_assignments": dict(assignments), "distinct_observed_configurations": len(signatures),
            "mean_active_agents": statistics.mean(d["active_agents"] for d in decisions) if decisions else None,
            "mean_connections": statistics.mean(len(d["edges"]) for d in decisions) if decisions else None}


def export_evaluation(source, output, report=None):
    source, output = Path(source), Path(output)
    report = read_json(source) if report is None else report
    output.mkdir(parents=True, exist_ok=True)
    questions, decisions = [], []
    models = {}
    for number, run in enumerate(report.get("runs", []), 1):
        graph = run.get("report", {})
        plan = graph.get("plan", run.get("plan", {}))
        decision = policy_decision(plan)
        decisions.append({"question": number, "key": run.get("key"), **decision})
        usage = _usage(graph)
        questions.append({"question": number, "key": run.get("key"), "source_id": run.get("source_id"),
            "question_preview": question_preview(graph.get("task")),
            "score": run.get("score"), "status": graph.get("status", run.get("status")),
            "answer": graph.get("answer"), "calls": usage.get("calls", 0), "tokens": usage["tokens"],
            "cost_usd": usage.get("cost_usd"), "cost_complete": usage.get("cost_complete"),
            "graph_seconds": usage.get("latency_seconds"), "query_seconds": run.get("query_seconds"),
            "agents": decision["active_agents"], "connections": len(decision["edges"]),
            "assignments": decision["assignments"]})
        for model in graph.get("model_profiles", []):
            models[model["id"]] = model
    complete = report.get("status") == "completed" and report.get("evaluated") == report.get("requested")
    modes = sorted({r.get("report", {}).get("mode", "unknown") for r in report.get("runs", [])})
    usage_complete = all(q["cost_complete"] is True for q in questions if q["status"] != "infeasible")
    priced = any(m.get("input_usd_per_million", 0) or m.get("output_usd_per_million", 0) for m in models.values())
    summary = {"source": source.name, "status": report.get("status"), "complete": complete,
        "valid_live_evaluation": complete and modes == ["live"], "modes": modes,
        "accuracy": report.get("accuracy"), "evaluated": report.get("evaluated"),
        "requested": report.get("requested"), "calls": report.get("calls"),
        "tokens": sum(q["tokens"] for q in questions), "accounted_usd": report.get("accounted_usd"),
        "cost_usage_complete": usage_complete, "nonzero_configured_prices": bool(priced),
        "total_query_seconds": sum(q["query_seconds"] or 0 for q in questions),
        "policy_steps": report.get("training", {}).get("policy_steps"),
        "sample_configuration": report.get("sample_configuration"),
        "request_spacing_seconds": sorted({r["report"].get("settings", {}).get("min_request_interval_seconds", 0)
                                            for r in report.get("runs", []) if "report" in r}),
        "observed_policy": _policy_overview(decisions), "model_profiles": list(models.values()),
        "notes": ["Accuracy is a fraction; a partial or simulated evaluation is not a completed live benchmark.",
                  "USD uses configured prices, not a provider billing statement. Zero prices cannot show monetary savings.",
                  "Latency includes quota pacing. No training improvement is established without a matched baseline.",
                  "Policy decisions are observations on these questions, not a universal routing rule."]}
    write_json(output / "summary.json", summary)
    write_json(output / "policy.json", {"observed_policy": summary["observed_policy"], "questions": decisions})
    write_csv(output / "questions.csv", questions, ["question", "question_preview", "source_id", "score", "status", "answer",
        "calls", "tokens", "cost_usd", "cost_complete", "graph_seconds", "query_seconds", "agents", "connections", "assignments"])
    _markdown(output / "README.md", [f"# {source.stem}", "",
        f"Status: **{summary['status']}**. Accuracy: **{_accuracy(summary['accuracy'])}** ({summary['evaluated']}/{summary['requested']} evaluated).",
        f"Requests: **{summary['calls']}**. Tokens: **{summary['tokens']:,}**. Recorded query time: **{summary['total_query_seconds']/60:.1f} minutes**.", "",
        "- [Per-question results](questions.csv): answers, correctness and usage.",
        "- [Observed policy](policy.json): selected models, workers, connections and their probabilities.",
        "- [Summary](summary.json): totals, model prices and validity flags.", "",
        *summary["notes"]])
    return summary


def _attempt_chain(source):
    """Follow explicit optimizer-resume provenance; do not guess continuation."""
    attempts, seen = [], set()
    current = Path(source)
    if not current.exists():
        raise ValueError(f"Missing training report: {current.name}")
    while current.exists():
        resolved = current.resolve()
        if resolved in seen:
            raise ValueError("Training resume provenance contains a cycle")
        seen.add(resolved)
        report = read_json(current)
        attempts.append((current, report))
        previous = report.get("resumed_from")
        if not previous:
            break
        previous = Path(previous)
        if not previous.exists():
            previous = current.parent / previous.name
        current = previous.with_name(previous.name.replace(".optimizer.pt", ".training.json"))
        if not current.exists():
            raise ValueError(f"Missing resumed training report: {current.name}; preserve the attempt chain")
    return list(reversed(attempts))


def export_training(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    attempts = _attempt_chain(source)
    progress, snapshots, failed = [], [], []
    for attempt_number, (path, report) in enumerate(attempts, 1):
        log = path.with_name(path.name.replace(".training.json", ".rollouts.jsonl"))
        traces = []
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    traces.append(json.loads(line))
        # A task may be repeated across epochs. Match successful occurrences in
        # order, never attach the metrics of a failed attempt to a successful row.
        for trace in traces:
            graph = trace.get("report", {})
            if graph.get("status") != "completed":
                failed.append({"attempt": attempt_number, "key": trace.get("key"),
                               "status": graph.get("status"), "calls": _usage(graph).get("calls", 0)})
        unused = list(traces)
        for run in report.get("runs", []):
            trace_index = next((i for i, t in enumerate(unused) if t.get("key") == run.get("key")
                                and t.get("report", {}).get("status") == "completed"), None)
            graph = unused.pop(trace_index).get("report", {}) if trace_index is not None else {}
            decision = run.get("policy") or policy_decision(graph.get("plan", {}))
            usage = _usage(graph)
            progress.append({"rollout": len(progress)+1, "attempt": attempt_number, "key": run.get("key"),
                "question_preview": question_preview(graph.get("task")),
                "epoch": run.get("epoch"), "score": run.get("accuracy"), "reward": run.get("reward"),
                "loss": run.get("loss"), "policy_steps_before": run.get("policy_steps_before"),
                "policy_steps_after": run.get("policy_steps_after"), "agents": run.get("agents"),
                "calls": usage.get("calls"), "tokens": usage.get("tokens") if graph else None,
                "cost_usd": usage.get("cost_usd"), "graph_seconds": usage.get("latency_seconds"),
                "assignments": decision.get("assignments"), "connections": len(decision.get("edges", [])),
                "policy": decision})
        snapshots.extend({"attempt": attempt_number, **snapshot} for snapshot in report.get("policy_snapshots", []))
    final = attempts[-1][1]
    training = final.get("training", {})
    summary = {"source": source.name, "status": final.get("status"), "settings": final.get("settings", {}),
        "attempts": [{"source": path.name, "status": report.get("status"), "calls": report.get("calls"),
                      "accounted_usd": report.get("accounted_usd")} for path, report in attempts],
        "policy_steps": training.get("policy_steps"), "difficulty_steps": training.get("difficulty_steps"),
        "fixture": training.get("fixture"), "unique_training_questions": len(training.get("policy_train_keys", [])),
        "successful_rollouts": len(progress), "failed_graphs": len(failed),
        "training_accuracy": statistics.mean(p["score"] for p in progress) if progress else None,
        "calls_all_attempts": sum(report.get("calls", 0) for _, report in attempts),
        "accounted_usd_all_attempts": sum(report.get("accounted_usd", 0) for _, report in attempts),
        "fixed_probe_snapshots": len(snapshots), "observed_policy": _policy_overview([p["policy"] for p in progress]),
        "notes": ["Training scores are not held-out evaluation scores.",
                  "Policy-gradient loss is not prediction error; a smaller loss alone does not establish better accuracy or efficiency.",
                  "Rollout decisions combine learning, different questions and random sampling.",
                  "Only fixed-probe snapshots compare the same questions after parameter updates; older runs have none.",
                  "Old update boundaries are left unknown rather than reconstructed from batch size."]}
    write_json(output / "summary.json", summary)
    write_csv(output / "progress.csv", progress, ["rollout", "question_preview", "attempt", "epoch", "score", "reward", "loss",
        "policy_steps_before", "policy_steps_after", "agents", "calls", "tokens", "cost_usd", "graph_seconds",
        "connections", "assignments"])
    write_json(output / "policy-history.json", {"notes": summary["notes"], "rollouts": progress,
                                               "fixed_probes": snapshots, "failed_graphs": failed})
    probe_rows = [{"attempt": snapshot["attempt"], "policy_steps": snapshot["policy_steps"],
                   "probe": number, "key": probe["key"], "question_preview": probe.get("question_preview"),
                   "agents": probe["active_agents"],
                   "connections": len(probe["edges"]), "assignments": probe["assignments"],
                   "model_probabilities": probe.get("model_probabilities", {}),
                   "keep_probabilities": probe.get("keep_probabilities", {}),
                   "edge_probabilities": probe.get("edge_probabilities", {})}
                  for snapshot in snapshots for number, probe in enumerate(snapshot["probes"], 1)]
    write_csv(output / "probe-history.csv", probe_rows, ["attempt", "policy_steps", "probe", "question_preview", "key",
        "agents", "connections", "assignments", "model_probabilities", "keep_probabilities", "edge_probabilities"])
    latest = snapshots[-1] if snapshots else None
    write_json(output / "current-policy.json", {
        "basis": "deterministic fixed training probes" if latest else "last sampled training rollout",
        "policy_steps": summary["policy_steps"], "snapshot": latest,
        "last_observed_training_decision": progress[-1]["policy"] if progress else None,
        "note": "A query-conditioned neural policy cannot be reduced to a single global routing rule."})
    _markdown(output / "README.md", [f"# {source.name.replace('.training.json', '')} training", "",
        f"Status: **{summary['status']}**. Unique training questions: **{summary['unique_training_questions']}**. Updates: **{summary['policy_steps']}**.",
        f"Successful rollouts: **{len(progress)}**. Requests across the available resume chain: **{summary['calls_all_attempts']}**.", "",
        "- [Training progress](progress.csv): scores, rewards, losses, agent counts and model choices.",
        "- [Policy history](policy-history.json): probabilities and graphs over time, including fixed probes when recorded.",
        "- [Probe history](probe-history.csv): decisions on the same training probes at each update (empty for older runs).",
        "- [Current policy](current-policy.json): latest recorded probe decisions or last sampled training decision.",
        "- [Summary](summary.json): settings, attempts and limits of this evidence.", "", *summary["notes"]])
    return summary


def export_run(source, output):
    source, output = Path(source), Path(output)
    if source.name.endswith(".training.json"):
        kind, name = "training", source.name.removesuffix(".training.json")
        summary = export_training(source, output / "training" / name)
    else:
        kind, name = "evaluation", source.stem
        summary = export_evaluation(source, output / "evaluation" / name)
    return {"kind": kind, "name": name, "status": summary["status"],
            "accuracy": summary.get("accuracy", summary.get("training_accuracy")),
            "questions": summary.get("evaluated", summary.get("unique_training_questions")),
            "policy_steps": summary.get("policy_steps"),
            "calls": summary.get("calls", summary.get("calls_all_attempts")),
            "report": f"{kind}/{name}/README.md"}


def export_directory(source, output):
    source, output = Path(source), Path(output)
    rows, skipped = [], []
    for path in sorted(source.glob("*.json")):
        # Archived retries belong in their explicit training resume chain.
        if ".failed-" in path.name or path.name.endswith(".summary.json"):
            continue
        try:
            report = read_json(path)
            if path.name.endswith(".training.json") or ("evaluated" in report and "requested" in report and "runs" in report):
                rows.append(export_run(path, output))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            skipped.append({"source": path.name, "reason": str(exc)})
    if not rows:
        raise ValueError("No training/evaluation reports found")
    write_json(output / "index.json", {"runs": rows, "skipped": skipped})
    write_csv(output / "runs.csv", rows, ["kind", "name", "status", "accuracy", "questions", "policy_steps", "calls", "report"])
    _markdown(output / "README.md", [f"# {source.name} results", "",
        "Training and evaluation are separate. This table shows percentages; JSON/CSV use fractions. Incomplete evaluations remain labeled; they are not benchmark results.", "",
        "| Type | Run | Status | Accuracy | Questions | Updates | Requests |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        *[f"| {r['kind']} | [{r['name']}]({r['report']}) | {r['status']} | {_accuracy(r['accuracy'])} | {r['questions']} | {r['policy_steps'] if r['policy_steps'] is not None else '—'} | {r['calls']} |" for r in rows], "",
        "[CSV index](runs.csv) · [JSON index](index.json)"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export compact training/evaluation summaries without API calls.")
    parser.add_argument("--source", type=Path, required=True, help="Report JSON or directory containing reports")
    parser.add_argument("--out", type=Path, required=True, help="Readable report directory, e.g. reports/my-run")
    args = parser.parse_args(argv)
    try:
        if args.source.is_dir():
            rows = export_directory(args.source, args.out)
            print(f"Exported {len(rows)} runs: {args.out / 'README.md'}")
        else:
            row = export_run(args.source, args.out)
            print(f"Report: {args.out / row['report']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
