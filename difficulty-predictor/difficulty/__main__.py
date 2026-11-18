"""Run from the difficulty-predictor folder: python -m difficulty --help."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Three-pool RouterBench difficulty experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    eval_download = commands.add_parser("download-evaluation-data", help="Download checksum-pinned MATH test and HumanEval data")
    eval_download.add_argument("--output", default="difficulty/data/generalization_eval")
    benchmark = commands.add_parser("benchmark", help="Test fixed predictor on labeled GSM8K/MBPP and external MATH/HumanEval/new prompts")
    benchmark.add_argument("--study", default="difficulty/runs/all_routerbench_v1")
    benchmark.add_argument("--assets", default="difficulty/data/generalization_eval")
    benchmark.add_argument("--gsm-test", default="data/gsm8k/test.jsonl")
    benchmark.add_argument("--output", default="difficulty/runs/generalization_test")
    benchmark.add_argument("--device", default="auto")
    download = commands.add_parser("download", help="Download and verify the pinned official 0shot-named export")
    download.add_argument("--output", default="difficulty/data/routerbench_0shot.pkl")
    assets = commands.add_parser("download-routerbench", help="Download verified five-shot, raw or archived RAG exports as data only")
    assets.add_argument("--assets", nargs="+", choices=["five","raw","rag"], required=True)
    assets.add_argument("--output", default="difficulty/data")
    all_data = commands.add_parser("all-routerbench", help="Train all released task families, including a separate archived RAG cohort")
    all_data.add_argument("--assets", default="difficulty/data")
    all_data.add_argument("--original-data", default="difficulty/data/prepared_overlap")
    all_data.add_argument("--gsm-audit", default="difficulty/runs/gsm_correctness_audit_v2")
    all_data.add_argument("--encoder", default="difficulty/data/mpnet")
    all_data.add_argument("--output", required=True)
    all_data.add_argument("--device", default="auto")
    all_data.add_argument("--prepare-only", action="store_true")
    finalize_all = commands.add_parser("finalize-all-routerbench", help="Verify all-family/RAG checkpoints and fit input-support checks")
    finalize_all.add_argument("--output", required=True)
    finalize_all.add_argument("--device", default="auto")
    encoder_download = commands.add_parser("download-encoder", help="Download an MPNet safetensors snapshot for offline training")
    encoder_download.add_argument("--output", default="difficulty/data/mpnet")
    encoder_download.add_argument("--revision")
    audit = commands.add_parser("audit", help="Audit label semantics, prompts and question-level split overlap")
    audit.add_argument("--data", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--zero-jsonl")
    audit.add_argument("--five-pickle")
    regrade = commands.add_parser("regrade-gsm", help="Audit saved GSM8K answers against numeric references; quarantine ambiguous answers")
    regrade.add_argument("--source", default="difficulty/data/routerbench_0shot.pkl")
    regrade.add_argument("--references", default="data/gsm8k/train.jsonl")
    regrade.add_argument("--data", default="difficulty/data/prepared_overlap")
    regrade.add_argument("--output", required=True)
    research = commands.add_parser("research", help="Train frozen MPNet head ablations, select by validation and compare")
    research.add_argument("--data", required=True)
    research.add_argument("--encoder", required=True, help="Local pretrained MPNet directory")
    research.add_argument("--output", required=True)
    research.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    research.add_argument("--epochs", type=int, default=100)
    research.add_argument("--patience", type=int, default=12)
    research.add_argument("--device", default="auto")
    convert = commands.add_parser("convert", help="Extract a RouterBench pickle as inert data; never unpickle")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="Split questions, freeze model groups, compute targets")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--groups", help="JSON mapping weak/intermediate/strong to disjoint model lists")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--min-coverage", type=float, default=1.0)
    prepare.add_argument("--datasets", nargs="+", help="Exact eval names; BAMAS overlap: grade-school-math mbpp")
    train = commands.add_parser("train", help="Fit a baseline predictor; use all-routerbench for the current main model")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, help="Default: 50 for text features; 10 for agentbalance")
    train.add_argument("--patience", type=int, help="Default: 8 for text features; 3 for agentbalance")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--backend", choices=["tfidf", "sentence-transformer", "agentbalance"], default="tfidf")
    train.add_argument("--encoder", help="Encoder ID or local directory; agentbalance defaults to all-mpnet-base-v2")
    train.add_argument("--batch-size", type=int, help="Default: 256 for text features; 8 for agentbalance")
    train.add_argument("--hidden-dim", type=int, help="Default: 128 for text features; 512 for agentbalance")
    train.add_argument("--unfreeze-last", type=int, default=4, help="MPNet final layers to adapt; 0 freezes all")
    train.add_argument("--max-length", type=int, default=384)
    train.add_argument("--truncation-side", choices=["left", "right"], default="left")
    train.add_argument("--encoder-lr", type=float, default=1e-5)
    train.add_argument("--head-lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0 (agentbalance)")
    train.add_argument("--balance-datasets", action="store_true", help="Sample training datasets with equal expected weight")
    train.add_argument("--revision", help="Hugging Face encoder revision")
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--max-train-samples", type=int, help="Deterministic subset for smoke testing")
    train.add_argument("--max-eval-samples", type=int, help="Deterministic validation/test subsets for smoke testing")
    train.add_argument("--accumulation-steps", type=int, default=1)
    train.add_argument("--threads", type=int, default=4)
    predict = commands.add_parser("predict", help="Predict three difficulty scores from task text")
    predict.add_argument("--model", required=True)
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input", help="JSONL with prompt or task_description")
    predict.add_argument("--output")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--details", action="store_true", help="Include a versioned feature vector, quality estimates and quality gains for later controllers")
    args = parser.parse_args()
    if args.command == "download-evaluation-data":
        from .evaluation_assets import download_assets
        print(json.dumps(download_assets(args.output),indent=2))
    elif args.command == "benchmark":
        from .benchmark import run_benchmark
        result=run_benchmark(args.study,args.assets,args.gsm_test,args.output,args.device)
        print(json.dumps({"labeled_mse":result["labeled"]["models"]["predictor"]["macro_mse"],
                          "external_counts":{k:v["questions"] for k,v in result["external"].items()},
                          "report":str(Path(args.output)/"RESULTS.md")},indent=2))
    elif args.command == "all-routerbench":
        from .all_routerbench import prepare_all, train_all
        if args.prepare_only:
            result=prepare_all(args.assets,args.original_data,args.gsm_audit,Path(args.output)/"prepared")
            print(json.dumps(result,indent=2))
        else:
            result=train_all(args.assets,args.original_data,args.gsm_audit,args.encoder,args.output,args.device)
            print(json.dumps({cohort:r["selected_configuration"] for cohort,r in result.items()},indent=2))
    elif args.command == "finalize-all-routerbench":
        from .all_routerbench import finalize_all
        result=finalize_all(args.output,args.device)
        print(json.dumps({cohort:r["recommendation"]["model_relative_path"] for cohort,r in result.items()},indent=2))
    elif args.command == "download-routerbench":
        from .routerbench_assets import fetch_asset
        print(json.dumps([fetch_asset(name,args.output) for name in args.assets],indent=2))
    elif args.command == "download":
        import hashlib
        import urllib.request
        from .safe_pickle import ZERO_SHOT_SHA256
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/withmartian/routerbench/resolve/7840214/routerbench_0shot.pkl", partial)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != ZERO_SHOT_SHA256:
                raise ValueError("Downloaded dataset checksum mismatch")
            partial.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != ZERO_SHOT_SHA256:
            raise ValueError("Existing dataset checksum mismatch")
        print(f"Verified {path}: {ZERO_SHOT_SHA256}")
    elif args.command == "download-encoder":
        from .assets import download_encoder
        print(json.dumps(download_encoder(args.output, args.revision), indent=2))
    elif args.command == "audit":
        from .audit import run_audit
        print(json.dumps(run_audit(args.data, args.output, args.zero_jsonl, args.five_pickle), indent=2))
    elif args.command == "regrade-gsm":
        from .gsm_correctness import prepare_correctness
        print(json.dumps(prepare_correctness(args.source, args.references, args.data, args.output), indent=2))
    elif args.command == "research":
        from .research import run_suite
        report = run_suite(args.data, args.encoder, args.output, args.seeds, args.epochs, args.patience, args.device)
        print(json.dumps({"selected_configuration": report["selected_configuration"],
                          "selected_validation": report["selected_validation"]}, indent=2))
    elif args.command == "convert":
        from .safe_pickle import convert
        print(json.dumps(convert(args.input, args.output), indent=2))
    elif args.command == "prepare":
        from .data import prepare
        print(json.dumps(prepare(args.input, args.output, args.groups, args.seed, args.min_coverage, args.datasets), indent=2))
    elif args.command == "train":
        if args.backend == "agentbalance":
            from .transformer import train_transformer, DEFAULT_ENCODER
            report = train_transformer(
                args.data, args.output, epochs=args.epochs if args.epochs is not None else 10,
                patience=args.patience if args.patience is not None else 3, seed=args.seed,
                encoder=args.encoder or DEFAULT_ENCODER,
                hidden_dim=args.hidden_dim if args.hidden_dim is not None else 512,
                batch_size=args.batch_size if args.batch_size is not None else 8,
                unfreeze_last=args.unfreeze_last, max_length=args.max_length,
                encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
                device=args.device, balance_datasets=args.balance_datasets, revision=args.revision,
                local_files_only=args.local_files_only, max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples, accumulation_steps=args.accumulation_steps,
                truncation_side=args.truncation_side, threads=args.threads)
        else:
            from .model import train
            report = train(args.data, args.output, epochs=args.epochs if args.epochs is not None else 50,
                           patience=args.patience if args.patience is not None else 8, seed=args.seed,
                           backend=args.backend, encoder=args.encoder or "sentence-transformers/all-MiniLM-L6-v2",
                           hidden_dim=args.hidden_dim if args.hidden_dim is not None else 128,
                           batch_size=args.batch_size if args.batch_size is not None else 256)
        print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
    else:
        from .model import DifficultyPredictor
        predictor = DifficultyPredictor.load(args.model, device=args.device)
        if args.text is not None:
            rows = [{"prompt": args.text}]
        else:
            rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [row.get("prompt", row.get("task_description")) for row in rows]
        if args.details:
            from .inputs import controller_features
            features = controller_features(predictor, texts)
            output_rows = [{**row, "predicted_difficulty": feature["difficulty"], "controller_features": feature}
                           for row, feature in zip(rows, features)]
        else:
            output_rows = [{**row, "predicted_difficulty": prediction} for row, prediction in zip(rows, predictor.predict(texts))]
        result = "\n".join(json.dumps(row, allow_nan=False) for row in output_rows) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(result, encoding="utf-8")
        else:
            print(result, end="")


if __name__ == "__main__":
    main()
