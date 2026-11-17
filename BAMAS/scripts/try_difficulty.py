"""Load a difficulty predictor once and query it interactively from the terminal."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_STUDY = PROJECT_ROOT / "BAMAS/difficulty/runs/all_routerbench_v1"


def default_model(cohort="main"):
    filename = "recommendation.json" if cohort == "main" else "rag_recommendation.json"
    study = ALL_STUDY
    recommendation = study / filename
    if recommendation.is_file():
        selected = (study / json.loads(recommendation.read_text(encoding="utf-8"))["model_relative_path"]).resolve()
        if not selected.is_relative_to(study.resolve()):
            raise ValueError("Recommended model path escapes its study directory")
        return selected
    raise FileNotFoundError(
        f"No verified {cohort} recommendation at {recommendation}. "
        "Train and finalize the all-routerbench study, or select a checkpoint with --model."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="Checkpoint directory; defaults to the verified all-family study recommendation")
    parser.add_argument("--cohort", choices=["main","rag"], default="main",
                        help="Main eleven-model pools or separate five-model RAG pools")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    parser.add_argument("--show-unsupported", action="store_true",
                        help="Also show raw experimental scores for inputs outside the support threshold")
    args = parser.parse_args()
    if args.model is None:
        try:
            args.model = default_model(args.cohort)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    if not (args.model / "model.json").is_file():
        parser.error(f"No predictor checkpoint found at {args.model}")

    # Support running this file directly without installing BAMAS as a package.
    sys.path.insert(0, str(PROJECT_ROOT))
    from BAMAS.difficulty.model import DifficultyPredictor
    from BAMAS.difficulty.inputs import controller_features

    print(f"Loading predictor: {args.model}", flush=True)
    try:
        predictor = DifficultyPredictor.load(args.model, device=args.device)
        print("Ready. Type a question and press Enter. Type quit to stop.")
        print("Datasets: " + ", ".join(predictor.manifest.get("datasets", [])))
        print("Model cohort: " + predictor.manifest.get("cohort", "historical main model pools"))
        print("Target: " + predictor.manifest.get("target_definition", "historical RouterBench quality deficit"))
        print("Experimental scores: 0 to 1, higher means greater difficulty. Input similarity does not guarantee accuracy.")
        if not getattr(predictor, "metadata", {}).get("input_support"):
            print("This checkpoint has no input-support check.")
        while True:
            question = input("\nQuestion> ").strip()
            if question.casefold() in {"quit", "exit", "/quit", "/exit"}:
                break
            if not question:
                continue
            features = controller_features(predictor, [question])[0]
            support = features.get("input_support")
            if support and not support["within_support"]:
                print("  Outside the training-support threshold; withholding a difficulty estimate.")
                if not args.show_unsupported:
                    print("  Use --show-unsupported to inspect the model's raw scores.")
                    continue
                print("  Raw scores below are unsupported experimental outputs:")
            scores = features["difficulty"]
            print(f"  Weak:         {scores['weak']:.4f}")
            print(f"  Intermediate: {scores['intermediate']:.4f}")
            print(f"  Strong:       {scores['strong']:.4f}")
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nBye.")


if __name__ == "__main__":
    main()
