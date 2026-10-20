"""Research CLI, isolated from the standard-library baseline until explicitly used."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import load_config, number
from .env import load_env
from .providers import ProviderError


def print_failure(result):
    if result.get("status") == "completed":
        return
    detail = result.get("last_error") or next((row.get("report", {}).get("error")
        for row in reversed(result.get("runs", [])) if row.get("report", {}).get("error")), None)
    if detail:
        print(f"Failure: {detail}")


def readable_report(source):
    from .reporting import export_run
    try:
        row = export_run(source, source.parent / "summaries")
        print(f"Readable results: {source.parent / 'summaries' / row['report']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Summaries must not turn a successful, billed run into a retry.
        print(f"Could not export readable results: {exc}. Run 'agentbalance report' on the saved artifact.")


def parser():
    root = argparse.ArgumentParser(prog="agentbalance research", description="Train and evaluate an AgentBalance-style controller")
    commands = root.add_subparsers(dest="operation", required=True)
    importer = commands.add_parser("import-routerbench", help="Convert an explicit CSV score export to difficulty JSONL")
    importer.add_argument("--csv", type=Path, required=True)
    importer.add_argument("--out", type=Path, required=True)
    importer.add_argument("--task-column", required=True)
    importer.add_argument("--score-columns", nargs="+", required=True)
    importer.add_argument("--model-column", help="For long-format CSV only")
    native = commands.add_parser("convert-routerbench", help="Convert the official processed wide RouterBench export")
    native.add_argument("--input", type=Path, required=True)
    native.add_argument("--out", type=Path, required=True)
    native.add_argument("--trust-pickle", action="store_true", help="Allow deserialization of a trusted pickle (can execute code)")
    native.add_argument("--exclude-eval", nargs="*", default=[], help="Exclude eval_name substrings, e.g. mmlu humaneval")
    fetch = commands.add_parser("fetch-dataset", help="Download an explicit public Hugging Face dataset slice as raw JSONL")
    fetch.add_argument("--dataset", required=True)
    fetch.add_argument("--subset", required=True)
    fetch.add_argument("--split", required=True)
    fetch.add_argument("--limit", type=int, required=True)
    fetch.add_argument("--offset", type=int, default=0)
    fetch.add_argument("--out", type=Path, required=True)
    benchmark = commands.add_parser("prepare-benchmark", help="Normalize native MMLU/MATH/HumanEval exports to scored JSONL")
    benchmark.add_argument("--benchmark", choices=("mmlu", "math", "humaneval"), required=True)
    benchmark.add_argument("--source", type=Path, required=True, help="Native CSV directory, JSON/JSONL export, or HumanEval JSONL.gz")
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--offset", type=int, default=0)
    benchmark.add_argument("--seed", type=int, help="Optional shuffle before applying offset/limit; omit to preserve source order")
    from .roles import AGENT_SETS
    for operation in ("configure", "transfer", "pools", "init", "train-difficulty", "train", "evaluate", "profile"):
        p = commands.add_parser(operation)
        p.add_argument("--config", type=Path, default=Path(__file__).with_name("demo.json"))
        env = p.add_mutually_exclusive_group()
        env.add_argument("--env-file", type=Path)
        env.add_argument("--no-env", action="store_true")
        p.add_argument("--out", type=Path, required=True)
        if operation == "configure":
            p.add_argument("--agent-set", choices=AGENT_SETS, required=True)
            p.add_argument("--method", choices=("legacy", "paper"), default="legacy")
            p.add_argument("--difficulty-offset", type=float, default=0.)
            p.add_argument("--pool-upper-bound", type=int)
            p.add_argument("--decision-model")
            p.add_argument("--min-request-interval", type=float, default=0.)
            p.add_argument("--reference-dir", type=Path, help="Pinned author checkout; reads literal role/decision prompts only")
        if operation == "transfer":
            p.add_argument("--source-config", type=Path, required=True)
        if operation in {"transfer", "pools", "init", "train-difficulty", "train"}:
            p.add_argument("--pool-count", type=int)
        if operation in {"transfer", "init", "train-difficulty", "train", "evaluate"}:
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2",
                           help="Frozen sentence-transformer name or local path; hash-test is fixture-only")
            p.add_argument("--seed", type=int, default=42)
        if operation == "init":
            p.add_argument("--difficulty-checkpoint", type=Path,
                           help="Copy only the pretrained difficulty module, with fresh policy weights")
        if operation in {"train-difficulty", "train", "evaluate", "profile"}:
            p.add_argument("--data", type=Path, required=True)
        if operation == "train-difficulty":
            p.add_argument("--validation-data", type=Path)
            p.add_argument("--recipe", choices=("frozen", "paper"), default="frozen")
            p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
            p.add_argument("--encoder-learning-rate", type=float, default=1e-5)
            p.add_argument("--head-learning-rate", type=float, default=5e-4)
            p.add_argument("--weight-decay", type=float, default=1e-3)
            p.add_argument("--batch-size", type=int, default=2)
            p.add_argument("--max-length", type=int, default=384)
            p.add_argument("--no-domain-balance", action="store_true")
        if operation in {"train-difficulty", "train"}:
            p.add_argument("--epochs", type=int, default=20 if operation == "train-difficulty" else 1)
            p.add_argument("--learning-rate", type=float, default=.001)
        if operation in {"train", "evaluate", "profile"}:
            p.add_argument("--scoring-protocol", choices=("local", "paper"), default="local")
            p.add_argument("--code-grader-image", help="Opt in to HumanEval tests in a locally available Docker Python image")
            p.add_argument("--allow-live", action="store_true")
            p.add_argument("--max-calls", type=int, required=True)
            p.add_argument("--total-budget", type=float, required=True, help="Overall USD budget using configured prices")
            p.add_argument("--budget", type=float, default=.02)
            p.add_argument("--latency-budget", type=float, default=120)
        if operation in {"train", "evaluate"}:
            p.add_argument("--topology", choices=("adaptive", "star", "linear"))
            p.add_argument("--trace-level", choices=("compact", "full"),
                           help="Persist metrics/decisions (default) or full debugging prompts")
        if operation == "profile":
            p.add_argument("--resume-profile", action="store_true",
                           help="Reuse successful measurements with unchanged config/data/scoring; limits cover new calls")
        if operation == "train":
            p.add_argument("--resume-training", action="store_true", help="Restore optimizer/RNG/cursor from checkpoint sidecar")
            p.add_argument("--cost-weight", type=float, default=.1)
            p.add_argument("--latency-weight", type=float, default=.1)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--entropy-weight", type=float, default=.01)
            p.add_argument("--length-weight", type=float, default=.01)
            p.add_argument("--reward-units", choices=("normalized", "raw"), default="normalized",
                           help="raw uses accuracy - cost_weight*USD - latency_weight*seconds")
        if operation == "evaluate":
            p.add_argument("--sample-configuration", action="store_true", help="Sample learned discrete choices during evaluation")
            p.add_argument("--allow-untrained", action="store_true")
            p.add_argument("--policy", choices=("learned", "cheap", "medium", "strong", "balanced"), default="learned")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .adaptive.data import import_scores, load_records, split_records
        from .adaptive.pools import construct_pools
        if args.operation == "convert-routerbench":
            from .adaptive.routerbench import convert_routerbench
            result = convert_routerbench(args.input, args.out, args.trust_pickle, args.exclude_eval)
            print(f"Prepared {result['output_prompts']} prompts from {len(result['score_columns'])} model outcome columns; excluded {result['excluded_rows']} rows.")
            return 0
        if args.operation == "fetch-dataset":
            from .adaptive.benchmarks import fetch_dataset
            count = fetch_dataset(args.dataset, args.subset, args.split, args.out, args.limit, args.offset)
            print(f"Downloaded {count} records: {args.out}. Run prepare-benchmark next.")
            return 0
        if args.operation == "prepare-benchmark":
            from .adaptive.benchmarks import prepare
            count = prepare(args.source, args.benchmark, args.out, args.limit, args.offset, args.seed)
            print(f"Prepared {count} {args.benchmark} tasks: {args.out}")
            return 0
        if args.operation == "import-routerbench":
            if args.out.exists():
                raise ValueError("Output already exists; choose a new path")
            count = import_scores(args.csv, args.out, args.task_column, args.score_columns, args.model_column)
            print(f"Wrote {count} prompts; difficulty = 1 - mean(model scores).")
            return 0
        if not args.no_env:
            load_env(args.env_file or Path.cwd() / ".env", required=args.env_file is not None)
        config = load_config(args.config)
        if getattr(args, "trace_level", None):
            config = replace(config, trace_level=args.trace_level)
        if args.out.exists():
            raise ValueError("Output already exists; choose a new path")
        if args.operation == "configure":
            from .roles import with_agent_set
            config = with_agent_set(config, args.agent_set)
            config = replace(config, controller_method=args.method, difficulty_offset=args.difficulty_offset,
                             pool_upper_bound=args.pool_upper_bound, decision_model=args.decision_model,
                             scheduler="ready" if args.method == "paper" else config.scheduler,
                             min_request_interval_seconds=args.min_request_interval)
            if args.reference_dir:
                if args.agent_set not in {"mmlu", "math", "humaneval"}:
                    raise ValueError("Reference prompts support mmlu, math, humaneval")
                from .reference import reference_templates
                config = replace(config, agent_templates=reference_templates(args.reference_dir, args.agent_set))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
            print(f"Configured {len(config.agent_templates) - 1} candidate workers plus synthesizer: {args.out}")
            print("Adaptive mode learns which workers and forward edges to retain; this roster requires new controller training.")
            return 0
        if args.operation == "pools":
            pools, filtered = construct_pools(config.models,
                                              config.pool_count if args.pool_count is None else args.pool_count,
                                              config.pool_min_size, config.pool_max_size)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"pools": pools, "filtered_models": filtered}, indent=2) + "\n", encoding="utf-8")
            print(f"Pools: {pools}; filtered: {filtered}")
            return 0
        if args.operation == "profile":
            from .adaptive.profiling import profile_models
            profile_models(config, load_records(args.data), args)
            print(f"Measured profiles: {args.out}")
            return 0
        try:
            import torch
            from .adaptive.controller import build_controller, load_controller, transfer_controller
            from .adaptive.encoder import TextEncoder
            from .adaptive.training import RunAllowance, evaluate, evaluate_baseline, fit_difficulty, train_policy, write_json
            from .adaptive.scoring import Scorer
        except ImportError as exc:
            raise ValueError("Research components require optional dependencies: python -m pip install -e '.[research]'") from exc
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        if args.operation == "transfer":
            if args.checkpoint is None:
                raise ValueError("transfer requires --checkpoint")
            controller = transfer_controller(args.checkpoint, load_config(args.source_config), config, args.pool_count)
            controller.save(args.out)
            print(f"Transferred shared controller weights: {args.out}; target pools={controller.pools}")
            print("Zero-shot transfer, not target-provider training. Evaluate on unseen tasks before drawing conclusions.")
            return 0
        if args.operation == "evaluate" and args.policy == "learned" and args.checkpoint is None:
            raise ValueError("evaluate requires --checkpoint")
        if hasattr(args, "epochs"):
            if args.epochs < 1:
                raise ValueError("epochs must be positive")
            number(args.learning_rate, "learning_rate", minimum=1e-9)
        for key in ("cost_weight", "latency_weight", "entropy_weight", "length_weight", "total_budget", "budget"):
            if hasattr(args, key):
                number(getattr(args, key), key)
        if hasattr(args, "latency_budget"):
            number(args.latency_budget, "latency_budget", minimum=.001)
        if hasattr(args, "batch_size") and args.batch_size < 1:
            raise ValueError("batch_size must be positive")
        scorer = Scorer(getattr(args, "code_grader_image", None), protocol=getattr(args, "scoring_protocol", "local"))
        if args.operation == "evaluate" and args.policy != "learned":
            if args.topology == "adaptive":
                raise ValueError("Adaptive topology requires the learned policy")
            allowance = RunAllowance(config, args.max_calls, args.total_budget, args.allow_live)
            result = evaluate_baseline(config, load_records(args.data), args.out, allowance, args.policy,
                                       args.topology or "star", args.budget, args.latency_budget, scorer)
            print(f"Baseline evaluation: {result['status']}; accuracy={result['accuracy']}; calls={result['calls']}")
            readable_report(args.out)
            print_failure(result)
            return 0 if result["status"] == "completed" else 1
        if args.checkpoint:
            controller = load_controller(args.checkpoint, config,
                                         allow_untrained=args.operation != "evaluate" or args.allow_untrained)
            if getattr(args, "pool_count", None) is not None and args.pool_count != len(controller.pools):
                raise ValueError("Checkpoint fixes the pool count; use research transfer to explicitly rebuild pools")
        else:
            controller = build_controller(config, TextEncoder(args.encoder, normalize=config.controller_method != "paper"),
                                          pool_count=args.pool_count)
        if args.operation == "init":
            if args.difficulty_checkpoint:
                if config.controller_method != "paper":
                    raise ValueError("Difficulty-only initialization requires paper mode")
                data = torch.load(args.difficulty_checkpoint, weights_only=True, map_location="cpu")
                if data.get("method") != "paper" or not data["training"].get("difficulty_steps"):
                    raise ValueError("Need a pretrained paper difficulty checkpoint")
                weights = {k.removeprefix("difficulty."): v for k,v in data["state_dict"].items() if k.startswith("difficulty.")}
                controller.difficulty.load_state_dict(weights, strict=True)
                controller.difficulty_encoder = TextEncoder(data["difficulty_encoder"], normalize=False,
                    max_length=data["training"].get("difficulty_max_length", 384))
                controller.training_info.update({k:v for k,v in data["training"].items() if k.startswith("difficulty_")})
                controller.training_info["difficulty_checkpoint"] = str(args.difficulty_checkpoint)
            controller.save(args.out)
            print(f"Untrained checkpoint: {args.out}. It is not a usable learned policy yet.")
        elif args.operation == "train-difficulty":
            records = load_records(args.data, "difficulty")
            validation = load_records(args.validation_data, "difficulty") if args.validation_data else None
            train, validation = split_records(records, validation, args.seed)
            if args.recipe == "paper":
                if config.controller_method != "paper":
                    raise ValueError("The paper recipe requires a paper controller configuration")
                from .difficulty_training import fit_paper_difficulty
                result = fit_paper_difficulty(controller, train, validation, args)
            else:
                result = fit_difficulty(controller, train, validation, args.epochs, args.learning_rate)
            controller.training_info["difficulty_source"] = str(args.data)
            controller.save(args.out)
            write_json(args.out.with_suffix(".difficulty.json"), result)
            print(f"Difficulty checkpoint: {args.out}; validation MSE={result['epochs'][-1]['validation_mse']:.4f}")
        else:
            records = load_records(args.data)
            allowance = RunAllowance(config, args.max_calls, args.total_budget, args.allow_live)
            if args.operation == "train":
                resume_state = None
                if args.resume_training:
                    if not args.checkpoint:
                        raise ValueError("--resume-training requires --checkpoint")
                    resume_state = args.checkpoint.with_suffix(".optimizer.pt")
                    if not resume_state.exists():
                        raise ValueError("Optimizer sidecar is missing; omit --resume-training for a warm start")
                result = train_policy(controller, records, args.out, allowance, args.epochs, args.budget,
                                      args.latency_budget, args.learning_rate, args.cost_weight, args.latency_weight,
                                      entropy_weight=args.entropy_weight, length_weight=args.length_weight,
                                      topology=args.topology or "adaptive", seed=args.seed, scorer=scorer,
                                      batch_size=args.batch_size, reward_units=args.reward_units, resume_state=resume_state)
            else:
                result = evaluate(controller, records, args.out, allowance, args.budget, args.latency_budget,
                                  args.topology or "adaptive", scorer, sample=args.sample_configuration)
            print(f"{args.operation}: {result['status']}; calls={result['calls']}; output={args.out}")
            if args.operation == "evaluate":
                print(f"Accuracy={result['accuracy']}; evaluated={result['evaluated']}/{result['requested']}")
            print_failure(result)
            readable_report(args.out.with_suffix(".training.json") if args.operation == "train" else args.out)
            return 0 if result["status"] == "completed" else 1
        return 0
    except (ValueError, OSError, TypeError, KeyError, ProviderError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed training steps are saved. An in-flight request may incur charges.")
        return 130
