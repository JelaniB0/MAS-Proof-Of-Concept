import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from .config import load_config
from .env import load_env
from .planner import InfeasiblePlan, make_plan
from .providers import ProviderError, preflight
from .runner import execute


def parser():
    root = argparse.ArgumentParser(description="AgentBalance research prototype: fixed baselines and learned adaptive graphs.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("research", help="Profile models, train, and evaluate the learned controller")
    commands.add_parser("report", help="Export readable training/evaluation reports without API calls")
    for command in ("chat", "plan", "run", "compare"):
        p = commands.add_parser(command, help={"plan": "Preview without API calls", "run": "Execute one graph",
                                               "chat": "Type queries or load task files interactively",
                                               "compare": "Run both graphs with all three policies"}[command])
        p.add_argument("--trace-level", choices=("compact", "full"), help="Persist metrics/decisions (default) or full debugging prompts")
        source = p.add_mutually_exclusive_group()
        source.add_argument("--config", type=Path, help="Use this model configuration (skips provider selection)")
        source.add_argument("--provider", choices=("groq", "azure"), help="Select a provider without prompting")
        env = p.add_mutually_exclusive_group()
        env.add_argument("--env-file", type=Path, help="Load this file instead of .env in the current directory")
        env.add_argument("--no-env", action="store_true", help="Use only existing environment variables")
        task = p.add_mutually_exclusive_group(required=command in {"plan", "compare"})
        task.add_argument("--task", help="Original task supplied to every agent")
        task.add_argument("--task-file", type=Path, help="UTF-8 task file")
        p.add_argument("--budget", type=float, default=0.02, help="USD cost budget PER RUN (default: .02)")
        p.add_argument("--latency-budget", type=float, default=120, help="Seconds PER RUN (default: 120)")
        p.add_argument("--difficulty", type=float, default=0.5, help="Manual task demand, 0..1 (default: .5)")
        p.add_argument("--max-strong-agents", type=int, default=1, choices=range(4),
                       help="Strong-agent cap for balanced policy only (default: 1)")
        if command != "chat":
            p.add_argument("--out", type=Path, help="JSON output; default is a unique file under results/")
        if command != "compare":
            p.add_argument("--topology", choices=("star", "linear", "adaptive"),
                           help="Communication graph; interactive sessions prompt, other commands default to star")
            p.add_argument("--policy", choices=("cheap", "strong", "balanced", "learned", "medium"),
                           help="Model assignment; interactive sessions prompt, other commands default to balanced")
            p.add_argument("--checkpoint", type=Path, help="Trained controller weights for --policy learned")
            p.add_argument("--allow-untrained", action="store_true", help="Explicitly permit untrained/fixture checkpoints for inspection")
    return root


def save(path, result):
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def describe(report):
    p, t = report["plan"], report["totals"]
    print(f"{p['topology']:6} {p['policy']:8} {report['status']:16} "
          f"cost=${t['cost_usd']:.6f} time={t['latency_seconds']:.3f}s "
          f"tokens={t['input_tokens'] + t['output_tokens']} [{report['mode']}]")
    print("  " + ", ".join(f"{r}={m}" for r, m in p["assignments"].items()))
    if p["policy"] == "learned":
        print("  Edges: " + ", ".join(f"{a}->{b}" for a, b in p["edges"]))
        print("  Pruned: " + (", ".join(p["selection"].get("pruned_agents", [])) or "none"))
    if report["error"]:
        print("  " + report["error"])
    for agent in report["agents"]:
        if agent.get("error"):
            print(f"  {agent['role']}: {agent['error']}")
    for warning in report["warnings"]:
        print("  " + warning)


def read_task_file(path):
    task = path.read_text(encoding="utf-8-sig")
    if not task.strip():
        raise ValueError(f"Task file is empty: {path}")
    return task


def prompt_choice(label, choices, default=None):
    print(f"\n{label}:")
    for index, choice in enumerate(choices, 1):
        print(f"  {index}. {choice}")
    while True:
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"{label}{suffix}> ").strip().lower()
        except EOFError:
            return None
        if answer in {"/exit", "/quit"}:
            return None
        if not answer and default:
            return default
        if answer in choices:
            return answer
        if answer in {str(i) for i in range(1, len(choices) + 1)}:
            return choices[int(answer) - 1]
        print("Choose " + ", ".join(choices) + " (name or number), or /exit.")


def provider_config(provider):
    local = Path.cwd() / "configs" / f"{provider}.local.json"
    if local.is_file():
        return local
    shared = Path.cwd() / "config.local.json"
    if shared.is_file():
        config = load_config(shared)
        if all(config.get(mid).provider == provider for mid in (config.cheap_model, config.strong_model)):
            return shared
    return Path(__file__).resolve().parents[1] / "configs" / f"{provider}.example.json"


def prompt_task():
    while True:
        try:
            line = input("You> ").strip()
        except EOFError:
            return None
        if not line:
            continue
        command, _, argument = line.partition(" ")
        if line in {"/exit", "/quit"}:
            return None
        if line == "/help":
            print("Type a query, /file PATH to read a UTF-8 text document, or /exit to quit.")
            continue
        if command == "/file":
            path = argument.strip()
            if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
                path = path[1:-1]
            if not path:
                print("Usage: /file examples/task.txt")
                continue
            try:
                return read_task_file(Path(path).expanduser())
            except (ValueError, OSError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                continue
        return line


def report_path(command):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{command}-{stamp}-{uuid4().hex[:8]}.json"


def selected_plan(config, args, task, options):
    if args.policy == "learned":
        return args.controller.plan(task, topology=args.topology, budget_usd=args.budget,
                                    latency_budget_seconds=args.latency_budget)
    return make_plan(config, task, args.topology, args.policy, **options)


def run_task(config, args, task, options, output):
    plan = selected_plan(config, args, task, options)
    print("Working...", flush=True)
    report = execute(config, plan, task)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    describe(report)
    if report["answer"]:
        print("\n" + report["answer"])
    print(f"\nReport: {output.resolve()}")
    return 0 if report["status"] == "completed" else 1


def chat(config, args, options, task=None):
    print("Type a query, /file examples/task.txt, /help, or /exit.")
    print("Each query is independent; budgets apply to each query.")
    if any(model.provider == "mock" for model in config.models):
        print("This configuration includes mock models: fixture responses do not solve your queries.")
    while True:
        if task is None:
            task = prompt_task()
        if task is None:
            return 0
        try:
            run_task(config, args, task, options, report_path("chat"))
        except (ValueError, TypeError, KeyError, OSError, ProviderError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
        task = None


def main(argv=None):
    # Preserve the user's encoding, but don't crash on unsupported LLM output
    # characters in Windows terminals or redirected ASCII streams.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "research":
        from .research import main as research_main
        return research_main(argv[1:])
    if argv and argv[0] == "report":
        from .reporting import main as report_main
        return report_main(argv[1:])
    if not argv or (argv[0].startswith("--") and argv[0] != "--help"):
        argv.insert(0, "chat")
    args = parser().parse_args(argv)
    try:
        if not args.no_env:
            load_env(args.env_file if args.env_file is not None else Path.cwd() / ".env",
                     required=args.env_file is not None)
        interactive = args.command == "chat" or (
            args.command == "run" and args.task is None and args.task_file is None)
        if interactive and args.config is None and args.provider is None:
            args.provider = prompt_choice("Provider", ("groq", "azure"))
            if args.provider is None:
                return 0
        if args.command != "compare" and args.policy is None:
            args.policy = prompt_choice("Policy", ("cheap", "strong", "balanced", "learned", "medium"), "balanced") if interactive else "balanced"
            if args.policy is None:
                return 0
        if args.command != "compare" and args.topology is None:
            topologies = ("star", "linear", "adaptive") if args.policy == "learned" else ("star", "linear")
            default_topology = "adaptive" if args.policy == "learned" else "star"
            args.topology = prompt_choice("Topology", topologies, default_topology) if interactive else default_topology
            if args.topology is None:
                return 0
        if args.command != "compare" and args.topology == "adaptive" and args.policy != "learned":
            raise ValueError("Adaptive topology requires --policy learned and a trained --checkpoint")
        config_path = args.config or (provider_config(args.provider) if args.provider
                                      else Path(__file__).with_name("demo.json"))
        try:
            config = load_config(config_path)
            if args.trace_level:
                config = replace(config, trace_level=args.trace_level)
        except ValueError as exc:
            if args.provider:
                raise ValueError(f"Cannot load {args.provider} configuration {config_path}: {exc}. "
                                 f"Set numeric input/output prices in configs/{args.provider}.local.json "
                                 f"using configs/{args.provider}.example.json as a template.") from exc
            raise
        provider_models = (config.models if getattr(args, "policy", None) == "learned" else
                           [config.get(mid) for mid in (config.cheap_model, config.strong_model, config.medium_model) if mid is not None])
        if args.provider and any(model.provider != args.provider for model in provider_models):
            raise ValueError(f"{config_path} must use {args.provider} for every candidate model profile")
        if args.command != "compare" and args.policy == "learned":
            if args.checkpoint is None and interactive:
                try:
                    checkpoint = input("Controller checkpoint path (/exit to quit)> ").strip().strip('\"')
                except EOFError:
                    return 0
                if checkpoint in {"/exit", "/quit"}:
                    return 0
                args.checkpoint = Path(checkpoint) if checkpoint else None
            if args.checkpoint is None:
                raise ValueError("Learned policy requires --checkpoint; see python -m agentbalance research --help")
            try:
                from .adaptive.controller import load_controller
            except ImportError as exc:
                raise ValueError("Learned policy requires research dependencies; run in .venv-research or install '.[research]'") from exc
            args.controller = load_controller(args.checkpoint, config, args.allow_untrained)
            if args.allow_untrained:
                print("Inspection mode: untrained or fixture weights may produce arbitrary configurations.")
        if interactive:
            print(f"Configuration: {config_path.resolve()}")
            print("Models: " + ", ".join(f"{model.id}={model.provider}" for model in config.models))
            print(f"Policy: {args.policy}")
            print(f"Topology: {args.topology}")
        task = args.task if args.task is not None else read_task_file(args.task_file) if args.task_file else None
        options = dict(budget_usd=args.budget, latency_budget_seconds=args.latency_budget,
                       difficulty=args.difficulty, max_strong_agents=args.max_strong_agents)
        if args.command == "chat":
            return chat(config, args, options, task)
        if task is None:
            print("Type a query or /file examples/task.txt.")
            task = prompt_task()
            if task is None:
                return 0
        if args.command == "plan":
            plan = selected_plan(config, args, task, options)
            print(json.dumps(plan.to_dict(), indent=2))
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                save(args.out, plan.to_dict())
            return 0
        output = args.out or report_path(args.command)
        if output.suffix.lower() != ".json":
            raise ValueError("--out must end in .json")
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "run":
            return run_task(config, args, task, options, output)
        plans, skipped = [], []
        for topology in ("star", "linear"):
            for policy in ("cheap", "strong", "balanced"):
                try:
                    plans.append(make_plan(config, task, topology, policy, **options))
                except InfeasiblePlan as exc:
                    skipped.append({"topology": topology, "policy": policy, "reason": str(exc)})
        preflight(config.get(mid) for p in plans for mid in p.assignments.values())
        print(f"Comparing {len(plans)} feasible runs; budget is ${args.budget:g} per run "
              f"(${args.budget * len(plans):g} combined budget).", flush=True)
        report = {"task": task, "runs": [], "skipped": skipped,
                  "note": "Single-task cost/latency comparison. Quality is not automatically scored. "
                          "Offline fixtures do not measure LLM quality or real inference latency."}
        save(output, report)
        for plan in plans:
            result = execute(config, plan, task)
            report["runs"].append(result)
            save(output, report)  # Preserve completed runs if a later run is interrupted.
            describe(result)
            if result["status"] == "error":
                report["aborted_reason"] = "Provider failure; remaining experiments were not called"
                save(output, report)
                break
        fields = ("topology", "policy", "status", "mode", "cost_usd", "cost_complete",
                  "input_tokens", "output_tokens", "latency_seconds", "calls", "answer")
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for run in report["runs"]:
                writer.writerow({**run["plan"], **run, **run["totals"]})
        for item in skipped:
            print(f"Skipped {item['topology']}/{item['policy']}: {item['reason']}")
        print(f"\nReports: {output.resolve()} and {output.with_suffix('.csv').resolve()}")
        return 0 if report["runs"] and all(r["status"] == "completed" for r in report["runs"]) else 1
    except (ValueError, TypeError, KeyError, OSError, ProviderError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. An in-flight provider request may still incur charges.", file=sys.stderr)
        return 130
