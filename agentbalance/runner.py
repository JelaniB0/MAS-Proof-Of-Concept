"""Execute dependency waves with budget admission and auditable per-agent traces."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import time
from threading import Lock

from . import providers
from .graph import dag_waves, config_messages, token_reserve

_request_lock = Lock()
_last_request = 0.


def pace_requests(interval):
    """Process-wide start spacing across graphs; configurable quota aid."""
    global _last_request
    if interval <= 0:
        return
    with _request_lock:
        delay = interval - (time.perf_counter() - _last_request)
        if delay > 0:
            time.sleep(delay)
        _last_request = time.perf_counter()


def execute(config, plan, task):
    schedule = dag_waves(plan.assignments, plan.edges)
    incoming = {role: [a for a, b in plan.edges if b == role] for role in plan.assignments}
    models = {role: config.get(model_id) for role, model_id in plan.assignments.items()}
    providers.preflight(models.values())  # Fail before any charge if any configuration is missing.
    if config.agent_tools != "none":
        from .adaptive.scoring import Scorer
        Scorer(config.agent_tool_image).preflight([{"scorer": "humaneval"}])
    started = time.perf_counter()
    outputs, traces, warnings = {}, [], []
    status, error = "completed", None
    spent, uncertain_cost = 0.0, 0.0

    def call(role, prompt, reserved, timeout):
        begin = time.perf_counter()
        model = models[role]
        trace = {"role": role, "model_id": model.id, "provider": model.provider,
                 "parents": incoming[role], "messages": prompt,
                 "reserved_cost_usd": reserved, "started_seconds": begin - started}
        attempted, result = False, None
        try:
            if config.agent_tools == "humaneval" and role != "synthesizer":
                from .agent_tools import check_contributions
                contributions, solved, checks = check_contributions(config.agent_tool_image, task,
                    {p: outputs[p] for p in incoming[role]})
                trace["tool_checks"] = checks
                if solved:
                    trace.update(text=solved, cost_usd=0., input_tokens=0, output_tokens=0, llm_calls=0,
                                 usage_source="internal_tool", finish_reason="stop", status="completed",
                                 latency_seconds=time.perf_counter()-begin)
                    return trace
                if checks:
                    prompt = config_messages(config, task, role, contributions, plan.instructions)
                    trace["messages"] = prompt
            pace_requests(config.min_request_interval_seconds)
            attempted = True
            result = providers.complete(model, prompt, config.max_output_tokens, timeout)
            succeeded = bool(result.text.strip()) and result.finish_reason not in {
                "failed", "cancelled", "in_progress", "queued"}
            trace.update(asdict(result), cost_usd=model.cost(result.input_tokens, result.output_tokens),
                         status="completed" if succeeded else "error")
            if succeeded and config.agent_tools == "math" and "programming_expert" in role:
                from .agent_tools import run_code, extract_code
                feedback = run_code(config.agent_tool_image, extract_code(result.text))
                trace["tool_checks"] = [feedback]
                trace["text"] += "\nThe answer is "+str(feedback["answer"])
            if not succeeded:
                trace["error"] = "Provider returned no answer text (possibly exhausted reasoning tokens or filtered)"
                if not result.text.strip() and result.finish_reason in {"length", "max_output_tokens"}:
                    trace["error"] = (f"Provider reached its completion-token limit before returning any answer text "
                                      f"(configured cap: {config.max_output_tokens}). Increase max_output_tokens "
                                      "or reduce reasoning_effort if the model supports it.")
                if result.text.strip():
                    trace["error"] = f"Provider did not finish the response: {result.finish_reason}"
        except (providers.ProviderError, ValueError, OSError) as exc:
            trace.update(status="error", error=str(exc), text="")
            if result is None:
                trace.update(cost_usd=None if attempted else 0., llm_calls=int(attempted),
                             usage_source="unknown" if attempted else "internal_tool",
                             input_tokens=None if attempted else 0, output_tokens=None if attempted else 0)
        trace["latency_seconds"] = time.perf_counter() - begin
        return trace

    def accept(batch):
        nonlocal spent, uncertain_cost, status, error
        traces.extend(batch)
        for trace in batch:
            spent += trace["cost_usd"] or 0
            if trace["cost_usd"] is None:
                uncertain_cost += trace["reserved_cost_usd"]
                warnings.append("A failed request may have incurred charges; its usage is unknown.")
            if trace["status"] == "completed":
                outputs[trace["role"]] = trace["text"]
            if trace.get("finish_reason") not in {None, "stop", "completed"}:
                warnings.append(f"{trace['role']}: finish reason {trace['finish_reason']}; answer may be incomplete.")
            if trace["usage_source"] == "estimated_reservation":
                warnings.append("Provider omitted usage; cost uses conservative token reservations.")
        if any(t["status"] == "error" for t in batch):
            failures = "; ".join(f"{t['role']} ({t['model_id']}): {t.get('error', 'agent failed')}"
                                 for t in batch if t["status"] == "error")
            status, error = "error", failures+" Downstream calls were stopped."
        elif spent > plan.budget_usd + 1e-12:
            status, error = "budget_exceeded", "Reported usage exceeded the cost reservation; further calls stopped"

    if config.scheduler == "ready":
        pending = list(plan.assignments)
        running = {}
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            while pending or running:
                ready = [r for r in pending if all(p in outputs for p in incoming[r])] if status == "completed" else []
                remaining = plan.latency_budget_seconds-(time.perf_counter()-started)
                if ready and remaining <= 0:
                    status, error = "budget_stopped", "Wall-clock latency budget exhausted before next call"
                    ready = []
                prompts = {r: config_messages(config, task, r, {p: outputs[p] for p in incoming[r]},
                                             plan.instructions) for r in ready}
                reserves = {r: models[r].cost(token_reserve(prompts[r]), config.max_output_tokens) for r in ready}
                reserved_running = sum(value for value in running.values())
                if ready and spent+uncertain_cost+reserved_running+sum(reserves.values()) > plan.budget_usd+1e-12:
                    status, error = "budget_stopped", "Concurrent call reservations exceed remaining budget"
                    ready = []
                for r in ready:
                    future = executor.submit(call, r, prompts[r], reserves[r], min(config.request_timeout_seconds, remaining))
                    running[future] = reserves[r]
                    pending.remove(r)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    del running[future]
                    accept([future.result()])
    for wave in schedule if config.scheduler == "waves" else []:
        remaining = plan.latency_budget_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            status, error = "budget_stopped", "Wall-clock latency budget exhausted before next wave"
            break
        prompts = {r: config_messages(config, task, r, {p: outputs[p] for p in incoming[r]},
                                      plan.instructions) for r in wave}
        reservations = {r: models[r].cost(token_reserve(prompts[r]), config.max_output_tokens) for r in wave}
        if spent + sum(reservations.values()) > plan.budget_usd + 1e-12:
            status, error = "budget_stopped", "Cost reservation for next wave exceeds remaining budget"
            break
        # Admit the entire wave before starting parallel calls; no budget race.
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [executor.submit(call, r, prompts[r], reservations[r],
                                       min(config.request_timeout_seconds, remaining)) for r in wave]
            batch = [f.result() for f in futures]
        accept(batch)
        if status != "completed":
            break
    elapsed = time.perf_counter() - started
    if elapsed > plan.latency_budget_seconds and status == "completed":
        status, error = "budget_exceeded", "Execution exceeded the wall-clock latency budget"
    mock_count = sum(m.provider == "mock" for m in models.values())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "task": task,
        "mode": "offline_simulation" if mock_count == len(models) else "mixed_simulation" if mock_count else "live",
        "status": status, "error": error, "plan": plan.to_dict(),
        "settings": {"max_output_tokens": config.max_output_tokens, "context_chars": config.context_chars,
                     "request_timeout_seconds": config.request_timeout_seconds, "scheduler": config.scheduler,
                     "min_request_interval_seconds": config.min_request_interval_seconds,
                     "agent_tools": config.agent_tools},
        "model_profiles": [{"id": m.id, "provider": m.provider,
                            "model": m.model if m.provider == "mock" else providers.expand(m.model),
                            "quality": m.quality, "input_usd_per_million": m.input_usd_per_million,
                            "output_usd_per_million": m.output_usd_per_million,
                            "latency_seconds": m.latency_seconds} for m in {m.id: m for m in models.values()}.values()],
        "totals": {"cost_usd": spent, "unknown_usage_reserved_usd": uncertain_cost,
                   "cost_complete": all(t["cost_usd"] is not None for t in traces),
                   "input_tokens": sum(t["input_tokens"] or 0 for t in traces),
                   "output_tokens": sum(t["output_tokens"] or 0 for t in traces),
                   "latency_seconds": elapsed, "calls": sum(t.get("llm_calls", 1) for t in traces)},
        "answer": outputs.get("synthesizer"), "warnings": list(dict.fromkeys(warnings)), "agents": traces,
    }
    if config.trace_level == "compact":
        # Execution still uses complete prompts/answers internally. Persist only
        # decisions, final answer, usage and errors unless debugging is requested.
        report["plan"].pop("instructions", None)
        report["plan"].get("selection", {}).pop("training", None)
        for trace in report["agents"]:
            for field in ("messages", "text"):
                trace.pop(field, None)
    return report
