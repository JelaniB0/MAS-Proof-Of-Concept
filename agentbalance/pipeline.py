"""Resumable experiment stages: prepare, configure, pretrain, run, summarize.

Run ``python -m agentbalance.pipeline --help``. Inference is never implicit in
data preparation/pretraining. The run command requires explicit live limits.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .env import load_env
from .files import write_json, replace_file
from .paper_data import acquire, download, digest, REFERENCE_COMMIT

POINTS = {
    "mmlu": [(40., .005, .3), (10., .001, .3), (2., .0005, .4), (0., 0., .6)],
    "math": [(20., .003, .3), (5., .001, .3), (1., .0002, .4), (0., 0., .6)],
    "humaneval": [(100., .01, .3), (10., .005, .3), (1., .001, .4), (0., 0., .6)],
}


def invoke(arguments):
    command = [sys.executable, "-m", "agentbalance", "research", *map(str, arguments)]
    print("Running: "+subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise ValueError(f"Stage stopped with exit {result.returncode}; inspect its checkpoint/report. "
                         "Completed stages are reused. Failed evaluations can be rerun; "
                         "profiling can continue with --resume-profiles; "
                         "training can continue with --resume-training after the provider limit resets.")


def validate_split(path):
    manifest = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest.exists():
        raise ValueError(f"Missing prepared data or manifest: {path}; run prepare first")
    if digest(path) != json.loads(manifest.read_text())["sha256"]:
        raise ValueError(f"Prepared data changed: {path}")


def archive_failed_evaluation(output):
    """Preserve an incomplete attempt before the CLI's no-overwrite guard."""
    output = Path(output)
    temporary = output.with_suffix(output.suffix+".tmp")
    if not output.exists() and not temporary.exists():
        return []
    for path in (output, temporary):
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ValueError(f"Cannot verify evaluation snapshot {path}; inspect it before retrying") from exc
            if report.get("status") == "completed":
                raise ValueError(f"Completed evaluation is missing from the stage ledger: {path}; "
                                 "inspect/reconcile it instead of overwriting it")
    attempt = 1
    while True:
        archive = output.with_name(f"{output.stem}.failed-{attempt:03d}{output.suffix}")
        archive_tmp = archive.with_suffix(archive.suffix+".tmp")
        if not archive.exists() and not archive_tmp.exists():
            break
        attempt += 1
    preserved = []
    for source, target in ((output, archive), (temporary, archive_tmp)):
        if source.exists():
            replace_file(source, target)
            preserved.append(str(target))
    print("Preserved incomplete evaluation: "+", ".join(preserved), flush=True)
    return preserved


def stage_usage(kind, output):
    """Read only this attempt's artifacts; prefer the latest valid usage snapshot."""
    output = Path(output)
    if kind == "profile":
        measurements = output.with_suffix(".measurements.jsonl")
        if not measurements.exists():
            return 0, 0.
        rows = [json.loads(s) for s in measurements.read_text(encoding="utf-8").splitlines()]
        return len(rows), sum(r["cost_usd"] if r["cost_usd"] is not None
                              else r.get("reserved_cost_usd", 0.) for r in rows)
    report = output.with_suffix(".training.json") if kind == "train" else output
    snapshots = []
    for path in (report, report.with_suffix(report.suffix+".tmp")):
        if path.exists():
            try:
                snapshots.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                continue  # A failed temporary write must not hide a valid saved report.
    if not snapshots:
        return 0, 0.
    return (max(s.get("calls", 0) for s in snapshots),
            max(s.get("accounted_usd", 0.) for s in snapshots))


def training_artifacts(output):
    paths = [output, output.with_suffix(".optimizer.pt"), output.with_suffix(".training.json"),
             output.with_suffix(".rollouts.jsonl")]
    return [*paths, *[path.with_suffix(path.suffix+".tmp") for path in paths]]


def recover_training_attempt(index, index_path, output):
    """Archive a failed attempt, preserving a checkpoint for exact CLI recovery."""
    key = str(output.resolve())
    retries = index.setdefault("training_retries", [])
    previous = next((entry for entry in reversed(retries) if entry["stage"] == key), None)

    def finish_archive(entry):
        for source, target in entry["artifacts"]:
            source, target = Path(source), Path(target)
            if source.exists():
                if target.exists():
                    raise ValueError(f"Training archive already exists: {target}; inspect before recovery")
                replace_file(source, target)
        entry["status"] = "ready"
        write_json(index_path, index)

    if previous and previous["status"] == "archiving":
        finish_archive(previous)
    artifacts = training_artifacts(output)
    if any(path.exists() for path in artifacts):
        if not output.exists() or not output.with_suffix(".optimizer.pt").exists():
            raise ValueError(f"Training recovery needs both checkpoint and optimizer sidecar: {output}")
        for report in (output.with_suffix(".training.json"), output.with_suffix(".training.json.tmp")):
            if report.exists() and json.loads(report.read_text(encoding="utf-8")).get("status") == "completed":
                raise ValueError(f"Completed training is missing from the stage ledger: {output}; inspect/reconcile it")
        attempt = 1
        while True:
            checkpoint = output.with_name(f"{output.stem}.failed-{attempt:03d}{output.suffix}")
            targets = training_artifacts(checkpoint)
            if not any(path.exists() for path in targets):
                break
            attempt += 1
        previous = {"stage": key, "checkpoint": str(checkpoint.resolve()), "status": "archiving",
                    "artifacts": [[str(source.resolve()), str(target.resolve())]
                                  for source, target in zip(artifacts, targets) if source.exists()]}
        retries.append(previous)
        write_json(index_path, index)
        finish_archive(previous)
        print(f"Preserved incomplete training: {checkpoint}", flush=True)
    if not previous:
        return None
    checkpoint = Path(previous["checkpoint"])
    if not checkpoint.exists() or not checkpoint.with_suffix(".optimizer.pt").exists():
        raise ValueError("Recorded training recovery checkpoint or optimizer sidecar is missing")
    return checkpoint


def execute_stage(index, index_path, kind, cfg, data, output, flags, allowance_factory, *,
                  resume_profiles=False, resume_training=False):
    output = Path(output)
    key = str(output.resolve())
    if key in index["stages"]:
        if not output.exists():
            raise ValueError("Recorded stage artifact is missing")
        return
    allowance = allowance_factory()
    previous_calls, previous_dollars = 0, 0.
    if kind == "evaluate":
        archives = archive_failed_evaluation(output)
        if archives:
            index.setdefault("evaluation_retries", []).append({"stage": key, "archives": archives})
            write_json(index_path, index)
    elif kind == "train" and resume_training:
        checkpoint = recover_training_attempt(index, index_path, output)
        if checkpoint:
            flags = list(flags)
            if "--checkpoint" in flags:
                position = flags.index("--checkpoint")
                flags[position+1] = checkpoint
            else:
                flags += ["--checkpoint", checkpoint]
            flags += ["--resume-training"]
    elif kind in {"train", "profile"}:
        artifacts = ([output, output.with_suffix(".measurements.jsonl")] if kind == "profile" else
                     training_artifacts(output))
        archived_training = kind == "train" and any(
            entry["stage"] == key for entry in index.get("training_retries", []))
        if any(p.exists() for p in artifacts) or archived_training:
            if kind == "profile" and resume_profiles and not output.exists():
                previous_calls, previous_dollars = stage_usage(kind, output)
                flags = [*flags, "--resume-profile"]
            else:
                recovery = ("--resume-profiles" if kind == "profile" else "--resume-training") + " for explicit recovery"
                raise ValueError(f"Incomplete {kind} artifacts already exist at {output}; use {recovery} or a new output. "
                                 "No requests were made or added to the ledger.")
    else:
        raise ValueError("Unknown experiment stage")
    command = [kind, "--config", cfg, "--data", data, "--out", output, *flags, *allowance]
    try:
        invoke(command)
    finally:
        calls, dollars = stage_usage(kind, output)
        index["calls"] += calls - previous_calls
        index["accounted_usd"] += dollars - previous_dollars
        write_json(index_path, index)
    index["stages"].append(key)
    write_json(index_path, index)


def configure(args):
    from .roles import templates
    from .adaptive.pools import construct_pools, pareto_models
    load_env()
    config = load_config(args.config)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    provenance = output/"configuration.json"
    record = {"source_sha256": digest(args.config), "reference_commit": REFERENCE_COMMIT,
              "benchmarks": args.benchmarks, "min_request_interval": args.min_request_interval,
              "reference_dir": str(args.reference_dir) if args.reference_dir else None,
              "code_grader_image": args.code_grader_image}
    if provenance.exists() and json.loads(provenance.read_text()) != record:
        raise ValueError("Run configuration changed; choose a fresh output directory")
    for benchmark in args.benchmarks:
        path = output/f"{benchmark}.json"
        if path.exists():
            continue
        roles = templates(benchmark)
        if args.reference_dir:
            from .reference import reference_templates
            roles = reference_templates(args.reference_dir, benchmark)
        prefixes = {}
        if args.reference_dir:
            from .reference import reference_prefixes
            prefixes = reference_prefixes(args.reference_dir, benchmark, roles)
        models = tuple(replace(m, extra_body={**m.extra_body, "temperature": 0}) for m in config.models)
        retained = len(pareto_models(models))
        count = len(config.explicit_pools) if config.explicit_pools else min(config.pool_count or 4, retained)
        min_size = min(config.pool_min_size, retained)
        # Small model inventories use fewer pools; the manifest records this adaptation.
        target = replace(config, models=models, agent_templates=roles, min_workers=2,
                         controller_method="paper", decision_model=config.decision_model or config.strong_model,
                         pool_count=count, pool_min_size=min_size, scheduler="ready",
                         pool_max_size=config.pool_max_size or max(3, min_size),
                         min_request_interval_seconds=args.min_request_interval,
                         prompt_prefixes=prefixes,
                         agent_tools=benchmark if args.code_grader_image and benchmark in {"math", "humaneval"} else "none",
                         agent_tool_image=args.code_grader_image)
        path.write_text(json.dumps(asdict(target), indent=2)+"\n")
    provenance.write_text(json.dumps(record, indent=2)+"\n")


def pretrain(args):
    load_env()
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    config = args.config
    native = output/"routerbench_0shot.pkl"
    prepared = args.difficulty_data or output/"routerbench-difficulty.jsonl"
    checkpoint = output/"difficulty.pt"
    record = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != "operation"}
    record["config_sha256"] = digest(config)
    if args.difficulty_data:
        record["difficulty_sha256"] = digest(args.difficulty_data)
    elif args.routerbench_source:
        record["routerbench_sha256"] = digest(args.routerbench_source)
    provenance = output/"pretraining.json"
    if provenance.exists() and json.loads(provenance.read_text()) != record:
        raise ValueError("Pretraining arguments or input changed; choose a fresh output directory")
    if checkpoint.exists():
        metrics = checkpoint.with_suffix(".difficulty.json")
        if not metrics.exists():
            raise ValueError("Difficulty pretraining is incomplete; choose a new directory")
        print(f"Completed difficulty checkpoint: {checkpoint}")
        return
    provenance.write_text(json.dumps(record, indent=2)+"\n")
    if not prepared.exists():
        native = args.routerbench_source or native
        if native.suffix.lower() in {".pkl", ".pickle"} and not args.trust_pickle:
            raise ValueError("RouterBench uses pickle. Explicitly pass --trust-pickle for the trusted official download")
        if not args.routerbench_source:
            download("https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl?download=true", native)
        flags = ["--trust-pickle"] if args.trust_pickle else []
        invoke(["convert-routerbench", "--input", native, "--out", prepared, *flags,
                "--exclude-eval", *args.exclude_eval])
    if args.limit:
        from .adaptive.data import load_records, split_records
        rows = load_records(prepared, "difficulty")
        # Stratified bounded fixture selection retains all three balancing domains.
        from .difficulty_training import domain
        groups = {g: [r for r in rows if domain(r) == g] for g in ("coding", "math_reasoning", "general")}
        import random
        selected = []
        for index, group in enumerate(groups.values()):
            random.Random(args.seed+index).shuffle(group)
            selected.extend(group[:max(4, args.limit//3)])
        prepared = output/"routerbench-subset.jsonl"
        if not prepared.exists():
            prepared.write_text("".join(json.dumps(r)+"\n" for r in selected), encoding="utf-8")
    command = ["train-difficulty", "--config", config, "--data", prepared, "--out", checkpoint,
               "--recipe", "paper", "--device", args.device, "--batch-size", args.batch_size,
               "--epochs", args.epochs, "--max-length", args.max_length, "--seed", args.seed]
    if args.no_domain_balance:
        command.append("--no-domain-balance")
    invoke(command)


def run(args):
    from .adaptive.data import load_records
    from .adaptive.pools import construct_pools, pareto_models
    from .adaptive.scoring import Scorer
    load_env()
    if not args.allow_live:
        raise ValueError("run requires --allow-live, --max-calls and --total-budget")
    if args.max_calls < 1 or not math.isfinite(args.total_budget) or args.total_budget < 0:
        raise ValueError("Invalid experiment allowances")
    if "strong" not in args.baselines:
        raise ValueError("Include the strong baseline to establish the shared frontier endpoint")
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                if k not in {"operation", "resume_profiles", "resume_training"}}
    settings["input_sha256"] = {str(path): digest(path) for path in [args.difficulty_checkpoint,
        *[args.configs/f"{b}.json" for b in args.benchmarks],
        *[args.data/f"{b}-{s}.jsonl" for b in args.benchmarks for s in ("train", "eval")]]}
    index_path = output/"index.json"
    index = {"settings": settings, "reference_commit": REFERENCE_COMMIT,
             "calls": 0, "accounted_usd": 0., "stages": [], "evaluations": []}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["settings"] != settings:
            raise ValueError("Run arguments changed; use a fresh experiment directory")
    for benchmark in args.benchmarks:
        for split in ("train", "eval"):
            validate_split(args.data/f"{benchmark}-{split}.jsonl")
        Scorer(args.code_grader_image).preflight(load_records(args.data/f"{benchmark}-eval.jsonl"))
        load_config(args.configs/f"{benchmark}.json")
    if not args.difficulty_checkpoint.exists():
        raise ValueError("Missing difficulty checkpoint; run pretrain first")

    def remaining():
        calls = args.max_calls-index["calls"]
        dollars = args.total_budget-index["accounted_usd"]
        if calls <= 0 or dollars < -1e-12:
            raise ValueError("Overall experiment allowance exhausted")
        result = ["--allow-live", "--max-calls", str(calls), "--total-budget", str(max(0., dollars)),
                  "--budget", str(args.budget), "--latency-budget", str(args.latency_budget),
                  "--scoring-protocol", "paper"]
        if args.code_grader_image:
            result += ["--code-grader-image", args.code_grader_image]
        return result

    def stage(kind, cfg, data, out, flags):
        execute_stage(index, index_path, kind, cfg, data, out, flags, remaining,
                      resume_profiles=args.resume_profiles, resume_training=args.resume_training)

    for benchmark in args.benchmarks:
        source_cfg = args.configs/f"{benchmark}.json"
        train = args.data/f"{benchmark}-train.jsonl"
        test = args.data/f"{benchmark}-eval.jsonl"
        cfg = output/f"{benchmark}-measured.json"
        if args.profile:
            stage("profile", source_cfg, train, cfg, [])
        else:
            cfg = source_cfg
        config = load_config(cfg)
        retained = len(pareto_models(config.models))
        count = len(config.explicit_pools) if config.explicit_pools else min(config.pool_count or 4, retained)
        min_size = min(config.pool_min_size, retained)
        for seed in args.seeds:
            for point in args.points:
                cost, latency, delta = POINTS[benchmark][point-1]
                target_cfg = output/f"{benchmark}-s{seed}-p{point}.json"
                if not target_cfg.exists():
                    target = replace(config, pool_count=count, pool_min_size=min_size,
                                     pool_upper_bound=min(point-1, count-1), difficulty_offset=delta)
                    write_json(target_cfg, asdict(target))
                initial = output/f"{benchmark}-s{seed}-p{point}-initial.pt"
                if not initial.exists():
                    invoke(["init", "--config", target_cfg, "--difficulty-checkpoint", args.difficulty_checkpoint,
                            "--out", initial, "--seed", seed])
                trained = output/f"{benchmark}-s{seed}-p{point}.pt"
                stage("train", target_cfg, train, trained, ["--checkpoint", initial, "--epochs", args.epochs,
                    "--batch-size", args.batch_size, "--learning-rate", .1, "--cost-weight", cost,
                    "--latency-weight", latency, "--length-weight", .2, "--entropy-weight", 0,
                    "--reward-units", "raw", "--seed", seed, "--topology", "adaptive"])
                report = output/f"{benchmark}-learned-s{seed}-p{point}.json"
                flags = ["--checkpoint", trained, "--seed", seed, "--topology", "adaptive"]
                if args.sample_configuration:
                    flags.append("--sample-configuration")
                stage("evaluate", target_cfg, test, report, flags)
                entry = {"benchmark": benchmark, "method": "AgentBalance-paper-mode", "seed": seed,
                         "point": point, "report": report.name, "base": False}
                if entry not in index["evaluations"]:
                    index["evaluations"].append(entry)
                write_json(index_path, index)
            for policy in args.baselines:
                if policy == "medium" and config.medium_model is None:
                    continue
                report = output/f"{benchmark}-{policy}-s{seed}.json"
                stage("evaluate", cfg, test, report, ["--policy", policy, "--topology", "star", "--seed", seed])
                entry = {"benchmark": benchmark, "method": f"fixed-{policy}-star", "seed": seed,
                         "point": 0, "report": report.name, "base": policy == "strong"}
                if entry not in index["evaluations"]:
                    index["evaluations"].append(entry)
                write_json(index_path, index)
    print(f"Complete: {index_path}; calls={index['calls']}, modeled USD={index['accounted_usd']}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare", help="Acquire and prepare all three benchmark datasets")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--benchmarks", nargs="+", choices=POINTS, default=list(POINTS))
    prepare.add_argument("--eval-limit", type=int, help="Optional laptop integration subset")
    prepare.add_argument("--math-source", type=Path, help="Authors' native sampled_test directory when available")
    subset = commands.add_parser("subset", help="Copy bounded integration splits from already prepared data")
    subset.add_argument("--data", type=Path, required=True)
    subset.add_argument("--out", type=Path, required=True)
    subset.add_argument("--benchmarks", nargs="+", choices=POINTS, default=list(POINTS))
    subset.add_argument("--train-limit", type=int, default=4)
    subset.add_argument("--eval-limit", type=int, default=3)
    configure_p = commands.add_parser("configure")
    configure_p.add_argument("--config", type=Path, required=True)
    configure_p.add_argument("--out", type=Path, required=True)
    configure_p.add_argument("--benchmarks", nargs="+", choices=POINTS, default=list(POINTS))
    configure_p.add_argument("--reference-dir", type=Path)
    configure_p.add_argument("--code-grader-image", help="Enable reference-style math/public-example agent tools")
    configure_p.add_argument("--min-request-interval", type=float, default=2.)
    pre = commands.add_parser("pretrain")
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--trust-pickle", action="store_true")
    pre.add_argument("--routerbench-source", type=Path, help="Processed CSV/Parquet export, or explicitly trusted pickle")
    pre.add_argument("--difficulty-data", type=Path, help="Existing normalized difficulty JSONL; avoids pickle/download")
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    pre.add_argument("--epochs", type=int, default=5)
    pre.add_argument("--batch-size", type=int, default=2)
    pre.add_argument("--max-length", type=int, default=384)
    pre.add_argument("--limit", type=int, help="Optional stratified integration subset; omitted means full conversion")
    pre.add_argument("--seed", type=int, default=42)
    pre.add_argument("--no-domain-balance", action="store_true")
    pre.add_argument("--exclude-eval", nargs="*", default=["mmlu", "math", "humaneval"])
    runner = commands.add_parser("run")
    runner.add_argument("--configs", type=Path, required=True)
    runner.add_argument("--data", type=Path, required=True)
    runner.add_argument("--difficulty-checkpoint", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--benchmarks", nargs="+", choices=POINTS, default=list(POINTS))
    runner.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    runner.add_argument("--points", nargs="+", type=int, choices=range(1,5), default=[1,2,3,4])
    runner.add_argument("--epochs", type=int, default=5)
    runner.add_argument("--batch-size", type=int, default=4)
    runner.add_argument("--baselines", nargs="+", choices=("cheap", "medium", "strong", "balanced"),
                        default=["cheap", "medium", "strong", "balanced"])
    runner.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    runner.add_argument("--resume-profiles", action="store_true",
                        help="Resume incomplete profiles using saved successful measurements; count only new calls")
    runner.add_argument("--resume-training", action="store_true",
                        help="Archive failed training attempts and restore optimizer/RNG/cursor; count only new calls")
    runner.add_argument("--sample-configuration", action=argparse.BooleanOptionalAction, default=True,
                        help="Sample learned graph choices at evaluation, as in the reference; disable for deterministic decoding")
    runner.add_argument("--code-grader-image")
    runner.add_argument("--allow-live", action="store_true")
    runner.add_argument("--max-calls", type=int, required=True)
    runner.add_argument("--total-budget", type=float, required=True)
    runner.add_argument("--budget", type=float, default=.05)
    runner.add_argument("--latency-budget", type=float, default=600.)
    summary = commands.add_parser("summarize")
    summary.add_argument("--index", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--allow-fixtures", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.operation == "prepare":
            print(json.dumps(acquire(args.out, args.benchmarks, args.eval_limit, args.math_source), indent=2))
        elif args.operation == "subset":
            from .paper_data import write_split
            if min(args.train_limit, args.eval_limit) < 1:
                raise ValueError("Subset counts must be positive")
            for benchmark in args.benchmarks:
                for split, limit in (("train", args.train_limit), ("eval", args.eval_limit)):
                    path = args.data/f"{benchmark}-{split}.jsonl"
                    validate_split(path)
                    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()][:limit]
                    write_split(rows, args.out/path.name, "integration subset; not the paper evaluation", path)
        elif args.operation == "configure":
            configure(args)
        elif args.operation == "pretrain":
            pretrain(args)
        elif args.operation == "run":
            run(args)
        else:
            from .adaptive.metrics import summarize
            summarize(args.index, args.out, args.allow_fixtures)
            print(f"Results: {args.out/'summary.json'} and {args.out/'runs.csv'}")
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed artifacts are retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
