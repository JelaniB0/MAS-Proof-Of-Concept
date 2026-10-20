"""Supervised difficulty fitting and bounded on-policy controller training."""

import json
import hashlib
import math
from pathlib import Path
import random
import time

import torch

from ..providers import ProviderError
from ..files import write_json, replace_file
from ..runner import execute
from ..reporting import policy_decision, question_preview
from .scoring import Scorer


def dataset_fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def policy_snapshot(controller, records, budget, latency, topology):
    """Observe the same training probes without calls, gradients or RNG changes."""
    was_training = controller.training
    try:
        controller.eval()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            probes = []
            for row in records[:3]:
                plan, _, _, _ = controller.select(row["task"], budget, latency,
                                                   sample=False, topology=topology)
                probes.append({"key": row["key"], "question_preview": question_preview(row["task"]),
                               **policy_decision(plan.to_dict())})
        return {"policy_steps": controller.training_info["policy_steps"],
                "decoding": "deterministic", "probes": probes}
    finally:
        controller.train(was_training)


def fit_difficulty(controller, train, validation, epochs=50, learning_rate=.001):
    x = controller.encoder.encode([r["task"] for r in train])
    paper = controller.config.controller_method == "paper"
    y = torch.tensor([1-r["difficulty"] if paper else r["difficulty"] for r in train], dtype=torch.float)
    vx = controller.encoder.encode([r["task"] for r in validation])
    vy = torch.tensor([1-r["difficulty"] if paper else r["difficulty"] for r in validation], dtype=torch.float)
    optimizer = torch.optim.Adam(controller.difficulty.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(controller.difficulty(x), y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = torch.nn.functional.mse_loss(controller.difficulty(vx), vy)
        history.append({"train_mse": float(loss.detach()), "validation_mse": float(validation_loss)})
    controller.training_info["difficulty_steps"] += epochs
    controller.training_info["difficulty_train_keys"] = sorted(set(controller.training_info.get("difficulty_train_keys", [])) | {r["key"] for r in train})
    controller.training_info["difficulty_validation_keys"] = [r["key"] for r in validation]
    return {"train_size": len(train), "validation_size": len(validation), "epochs": history,
            "note": "Held-out difficulty error is not multi-agent answer quality."}


class RunAllowance:
    """Reserve each graph against both an overall USD limit and a hard call-count cap."""
    def __init__(self, config, max_calls, total_budget, allow_live=False):
        if max_calls < 1 or not math.isfinite(total_budget) or total_budget < 0:
            raise ValueError("max_calls must be positive; total_budget finite and nonnegative")
        self.config, self.max_calls, self.total_budget = config, max_calls, total_budget
        self.allow_live, self.calls, self.spent = allow_live, 0, 0.0

    def run(self, plan, task):
        live = any(self.config.get(mid).provider != "mock" for mid in plan.assignments.values())
        if live and not self.allow_live:
            raise ValueError("Live model calls require --allow-live and explicit run limits")
        if self.calls + len(plan.assignments) > self.max_calls:
            return None
        if self.spent + plan.reserved_cost_usd > self.total_budget + 1e-12:
            return None
        report = execute(self.config, plan, task)
        self.calls += report["totals"]["calls"]
        self.spent += report["totals"]["cost_usd"] + report["totals"]["unknown_usage_reserved_usd"]
        return report


def train_policy(controller, records, output, allowance, epochs=1, budget=.02, latency=120,
                 learning_rate=.001, cost_weight=.1, latency_weight=.1, entropy_weight=.01,
                 length_weight=.01, topology="adaptive", seed=42, scorer=None, batch_size=1,
                 reward_units="normalized", resume_state=None):
    scorer = scorer or Scorer()
    scorer.preflight(records)
    if batch_size < 1 or reward_units not in {"normalized", "raw"}:
        raise ValueError("Invalid batch size or reward units")
    paper = controller.config.controller_method == "paper"
    if paper:
        head_ids = {id(p) for p in controller.difficulty.parameters()}
        optimizer = torch.optim.Adam([
            {"params": [p for p in controller.parameters() if id(p) not in head_ids], "lr": learning_rate},
            {"params": controller.difficulty.parameters(), "lr": 1e-5}])
    else:
        optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    controller.train()
    summary = {"runs": [], "status": "running", "scorers": sorted({r.get("scorer", "exact") for r in records}),
               "settings": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                            "cost_weight": cost_weight, "latency_weight": latency_weight,
                            "entropy_weight": entropy_weight, "length_weight": length_weight,
                            "reward_units": reward_units, "seed": seed, "topology": topology},
               "note": "Training scores are not held-out evaluation; no claim of paper reproduction."}
    summary["settings"].update(budget=budget, latency=latency)
    trace_path = Path(output).with_suffix(".training.json")
    log = Path(output).with_suffix(".rollouts.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists() or Path(output).exists():
        raise ValueError("Training output already exists; choose a new checkpoint path")
    keys = set(controller.training_info.get("policy_train_keys", []))
    keys.update(row["key"] for row in records)
    controller.training_info["policy_train_keys"] = sorted(keys)
    controller.training_info["fixture"] = (controller.training_info.get("fixture", False)
                                            or any(m.provider == "mock" for m in controller.config.models))
    baseline = 0.0
    pending = []
    pending_rows = []
    next_epoch, next_index = 0, 0
    if resume_state:
        state = torch.load(resume_state, weights_only=True, map_location="cpu")
        if state["dataset_fingerprint"] != dataset_fingerprint(records):
            raise ValueError("Resume dataset changed")
        if state["policy_steps"] != controller.training_info["policy_steps"]:
            raise ValueError("Checkpoint and optimizer sidecar are from different updates")
        for key in ("batch_size", "learning_rate", "cost_weight", "latency_weight", "entropy_weight",
                    "length_weight", "reward_units", "seed", "topology", "budget", "latency"):
            if state["settings"][key] != summary["settings"][key]:
                raise ValueError(f"Resume setting changed: {key}")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        baseline = state["baseline"]
        next_epoch, next_index = state["next_epoch"], state["next_index"]
        if next_epoch >= epochs:
            raise ValueError("Resume has already completed the requested epochs; increase --epochs")
        summary["resumed_from"] = str(resume_state)
    saved_epoch, saved_index, saved_rng, saved_baseline = next_epoch, next_index, torch.get_rng_state(), baseline
    summary["policy_snapshots"] = [policy_snapshot(controller, records, budget, latency, topology)]

    def save_optimizer():
        path = Path(output).with_suffix(".optimizer.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"optimizer": optimizer.state_dict(), "baseline": saved_baseline,
                    "torch_rng": saved_rng, "next_epoch": saved_epoch, "next_index": saved_index,
                    "dataset_fingerprint": dataset_fingerprint(records), "settings": summary["settings"],
                    "policy_steps": controller.training_info["policy_steps"]}, temporary)
        replace_file(temporary, path)

    def update():
        nonlocal saved_epoch, saved_index, saved_rng, saved_baseline
        if not pending:
            return
        optimizer.zero_grad()
        torch.stack(pending).mean().backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        saved_epoch, saved_index, saved_rng, saved_baseline = next_epoch, next_index, torch.get_rng_state(), baseline
        controller.training_info["policy_steps"] += 1
        if "target_policy_steps" in controller.training_info:
            controller.training_info["target_policy_steps"] += 1
        for row in pending_rows:
            row["policy_steps_after"] = controller.training_info["policy_steps"]
        pending_rows.clear()
        pending.clear()
        summary["policy_snapshots"].append(policy_snapshot(controller, records, budget, latency, topology))
        controller.save(output)
        save_optimizer()
        write_json(trace_path, summary)
    try:
        start_epoch, start_index = next_epoch, next_index
        for epoch in range(start_epoch, epochs):
            order = list(records)
            random.Random(seed + epoch).shuffle(order)
            for position, row in enumerate(order):
                if epoch == start_epoch and position < start_index:
                    continue
                start = time.perf_counter()
                plan, log_probability, entropy, length_penalty = controller.select(
                    row["task"], budget, latency, sample=True, topology=topology)
                configuration_seconds = time.perf_counter() - start
                report = None
                if plan.selection["feasible"]:
                    report = allowance.run(plan, row["task"])
                    if report is None:
                        update()
                        summary["status"] = "limit_reached"
                        return summary
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": row["key"], "report": report}, ensure_ascii=False) + "\n")
                    if report["status"] != "completed" or not report["totals"]["cost_complete"]:
                        update()
                        summary["status"] = "provider_or_budget_failure"
                        summary["last_error"] = report["error"]
                        return summary  # Do not train on transient provider outages or retry a rate limit.
                    accuracy = scorer.score(report["answer"], row)
                    cost, duration = report["totals"]["cost_usd"], report["totals"]["latency_seconds"]
                    if reward_units == "normalized":
                        reward = accuracy - cost_weight * cost / max(budget, 1e-9) - latency_weight * duration / latency
                    else:
                        reward = accuracy - cost_weight * cost - latency_weight * duration
                else:
                    accuracy, reward = 0.0, -1.0
                loss = (-reward*log_probability + length_weight*length_penalty if paper else
                        -(reward-baseline)*log_probability-entropy_weight*entropy+length_weight*length_penalty)
                pending.append(loss)
                baseline = .9 * baseline + .1 * reward
                next_epoch, next_index = ((epoch+1, 0) if position+1 == len(order) else (epoch, position+1))
                training_row = {"key": row["key"], "epoch": epoch, "accuracy": accuracy,
                                        "reward": reward, "loss": float(loss.detach()),
                                        "configuration_seconds": configuration_seconds,
                                        "joint_log_probability": float(log_probability.detach()),
                                        "feasible": plan.selection["feasible"], "agents": len(plan.assignments),
                                        "policy_steps_before": controller.training_info["policy_steps"],
                                        "policy_steps_after": None, "policy": policy_decision(plan.to_dict())}
                summary["runs"].append(training_row)
                pending_rows.append(training_row)
                print(f"Training rollout {len(summary['runs'])}: score={accuracy:g}, reward={reward:.3f}, calls={allowance.calls}", flush=True)
                if len(pending) >= batch_size:
                    update()
            update()
        summary["status"] = "completed"
        return summary
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        raise
    except (ValueError, OSError, ProviderError, RuntimeError) as exc:
        summary.update(status="error", last_error=str(exc))
        raise
    finally:
        summary.update(calls=allowance.calls, accounted_usd=allowance.spent,
                       training=dict(controller.training_info))
        controller.save(output)
        save_optimizer()
        write_json(trace_path, summary)


def evaluate(controller, records, output, allowance, budget=.02, latency=120, topology="adaptive", scorer=None,
             sample=False):
    scorer = scorer or Scorer()
    scorer.preflight(records)
    trained = set(controller.training_info.get("policy_train_keys", [])) | set(controller.training_info.get("difficulty_train_keys", []))
    if trained & {r["key"] for r in records}:
        raise ValueError("Evaluation prompts overlap training data; use an unseen test set")
    result = {"runs": [], "status": "running", "dataset_fingerprint": dataset_fingerprint(records),
              "sample_configuration": sample,
              "scorers": sorted({r.get("scorer", "exact") for r in records})}
    controller.eval()
    for row in records:
        start = time.perf_counter()
        with torch.no_grad():
            plan, _, _, _ = controller.select(row["task"], budget, latency, topology=topology, sample=sample)
        configuration_seconds = time.perf_counter() - start
        if not plan.selection["feasible"]:
            result["runs"].append({"key": row["key"], "score": 0, "status": "infeasible", "plan": plan.to_dict()})
            continue
        report = allowance.run(plan, row["task"])
        if report is None:
            result["status"] = "limit_reached"
            break
        result["runs"].append({"key": row["key"], "source_id": row.get("source_id"),
                              "score": 0,
                              "configuration_seconds": configuration_seconds,
                              "query_seconds": configuration_seconds + report["totals"]["latency_seconds"], "report": report})
        result.update(calls=allowance.calls, accounted_usd=allowance.spent)
        write_json(output, result)  # Preserve billed usage even if the grader raises.
        if report["status"] == "completed":
            result["runs"][-1]["score"] = scorer.score(report["answer"], row)
        write_json(output, result)
        if report["status"] != "completed":
            result["status"] = "provider_or_budget_failure"
            break
    else:
        result["status"] = "completed"
    scores = [r["score"] for r in result["runs"]]
    result.update(evaluated=len(scores), requested=len(records), accuracy=sum(scores) / len(scores) if scores else None,
                  calls=allowance.calls, accounted_usd=allowance.spent,
                  training=dict(controller.training_info))
    write_json(output, result)
    return result


def evaluate_baseline(config, records, output, allowance, policy="balanced", topology="star", budget=.02, latency=120, scorer=None):
    scorer = scorer or Scorer()
    scorer.preflight(records)
    from ..planner import InfeasiblePlan, make_plan
    result = {"runs": [], "status": "running", "policy": policy, "dataset_fingerprint": dataset_fingerprint(records),
              "scorers": sorted({r.get("scorer", "exact") for r in records})}
    for row in records:
        try:
            plan = make_plan(config, row["task"], topology, policy, budget_usd=budget, latency_budget_seconds=latency)
        except InfeasiblePlan as exc:
            result["runs"].append({"key": row["key"], "score": 0, "status": "infeasible", "error": str(exc)})
            continue
        report = allowance.run(plan, row["task"])
        if report is None:
            result["status"] = "limit_reached"
            break
        result["runs"].append({"key": row["key"], "score": 0, "report": report})
        result.update(calls=allowance.calls, accounted_usd=allowance.spent)
        write_json(output, result)
        if report["status"] == "completed":
            result["runs"][-1]["score"] = scorer.score(report["answer"], row)
        write_json(output, result)
        if report["status"] != "completed":
            result["status"] = "provider_or_budget_failure"
            break
    else:
        result["status"] = "completed"
    scores = [r["score"] for r in result["runs"]]
    result.update(evaluated=len(scores), requested=len(records), accuracy=sum(scores) / len(scores) if scores else None,
                  calls=allowance.calls, accounted_usd=allowance.spent)
    write_json(output, result)
    return result
