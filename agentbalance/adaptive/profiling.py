"""Collect measured single-model accuracy and latency before constructing pools."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

from .. import providers
from ..files import write_json
from ..graph import token_reserve
from .scoring import Scorer


def completed_measurements(trace_path, config, records):
    """Validate an existing trace and select reusable successful outcomes."""
    expected = {(model.id, row["key"]) for model in config.models for row in records}
    completed = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        pair = (item.get("model_id"), item.get("key"))
        if pair not in expected:
            raise ValueError("Profile measurements do not match the current models/tasks")
        completion = item.get("completion", {})
        if (item.get("status") in {None, "completed"}
                and completion.get("text", "").strip()
                and completion.get("finish_reason") in {"stop", "completed"}):
            for field in ("score", "cost_usd", "latency_seconds"):
                value = item.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid saved profile {field}")
            if pair in completed:
                raise ValueError("Duplicate successful profile measurement")
            completed[pair] = item
    return completed


def profile_models(config, records, args):
    scorer = Scorer(getattr(args, "code_grader_image", None), protocol=getattr(args, "scoring_protocol", "local"))
    scorer.preflight(records)
    trace_path = args.out.with_suffix(".measurements.jsonl")
    completed = {}
    if trace_path.exists():
        if not getattr(args, "resume_profile", False):
            raise ValueError("Measurements file already exists; use --resume-profile with unchanged inputs")
        completed = completed_measurements(trace_path, config, records)
    if args.max_calls < len(config.models) * len(records) - len(completed):
        raise ValueError("max_calls is too small to profile every model on the same tasks")
    if any(not math.isfinite(v) for v in (args.total_budget, args.budget, args.latency_budget)) or args.total_budget < 0 or args.budget < 0 or args.latency_budget <= 0:
        raise ValueError("Invalid profiling budgets")
    if any(m.provider != "mock" for m in config.models) and not args.allow_live:
        raise ValueError("Profiling live models requires --allow-live")
    providers.preflight(config.models)
    models, spent = [], 0.0
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for model in config.models:
        outcomes = []
        for row in records:
            saved = completed.get((model.id, row["key"]))
            if saved is not None:
                outcomes.append(saved)
                continue
            prompt = [{"role": "system", "content": "Answer the question exactly as requested."},
                      {"role": "user", "content": row["task"]}]
            reserve = model.cost(token_reserve(prompt), config.max_output_tokens)
            if reserve > args.budget or spent + reserve > args.total_budget + 1e-12:
                raise ValueError("Profiling budget exhausted; completed measurements were preserved")
            start = time.perf_counter()
            from ..runner import pace_requests
            pace_requests(config.min_request_interval_seconds)
            try:
                completion = providers.complete(model, prompt, config.max_output_tokens,
                                                 min(config.request_timeout_seconds, args.latency_budget))
            except providers.ProviderError:
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key":row["key"], "model_id":model.id, "status":"error",
                                             "cost_usd":None, "reserved_cost_usd":reserve})+"\n")
                raise
            duration = time.perf_counter() - start
            cost = model.cost(completion.input_tokens, completion.output_tokens)
            spent += cost
            item = {"key": row["key"], "model_id": model.id, "score": None,
                    "cost_usd": cost, "latency_seconds": duration, "completion": asdict(completion)}
            try:
                item["score"] = scorer.score(completion.text, row)
            except (ValueError, OSError) as exc:
                item.update(status="grader_error", error=str(exc))
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False)+"\n")
                raise
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if not completion.text.strip() or completion.finish_reason not in {"stop", "completed"}:
                raise ValueError("Profiling stopped on incomplete output; inspect measurements")
            if spent > args.total_budget + 1e-12 or cost > args.budget + 1e-12 or duration > args.latency_budget:
                raise ValueError("Profiling overran a budget; further calls stopped")
            outcomes.append(item)
            print(f"Profile {model.id}: score={item['score']:g}, time={duration:.2f}s", flush=True)
        quality = statistics.mean(r["score"] for r in outcomes)
        models.append({**asdict(model), "quality": quality,
                       "calibration_cost_usd": statistics.mean(r["cost_usd"] for r in outcomes),
                       "latency_seconds": statistics.mean(r["latency_seconds"] for r in outcomes),
                       "performance_profile": f"Accuracy {quality:.4f} using {sorted({r.get('scorer', 'exact') for r in records})} on {len(records)} tasks from {args.data}"})
    output = {**asdict(config), "models": models}
    write_json(args.out, output)
