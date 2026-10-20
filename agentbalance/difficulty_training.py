"""Appendix F pretraining with bounded minibatches and a separate tuned MPNet."""

from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from .adaptive.encoder import TextEncoder


def domain(row):
    text = " ".join(row.get("eval_names", [])).casefold()
    if any(k in text for k in ("humaneval", "mbpp", "code", "apps")):
        return "coding"
    if any(k in text for k in ("gsm8k", "math", "reasoning", "arc", "hellaswag")):
        return "math_reasoning"
    if not text:
        raise ValueError("Domain balancing needs eval_names on each RouterBench row")
    return "general"


def domain_weights(rows):
    groups = [domain(r) for r in rows]
    counts = Counter(groups)
    if len(counts) != 3:
        raise ValueError("The paper's 1:1:4 balance needs coding, math/reasoning, and general rows; "
                         "adjust exclusions or explicitly use --no-domain-balance for an ablation")
    ratios = {"coding": 1, "math_reasoning": 1, "general": 4}
    return [ratios[g]/counts[g] for g in groups], dict(counts)


def fit_paper_difficulty(controller, train, validation, args):
    if args.batch_size < 1 or not 8 <= args.max_length <= 384:
        raise ValueError("batch-size must be positive and max-length between 8 and 384")
    for value in (args.encoder_learning_rate, args.head_learning_rate):
        if value <= 0:
            raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be nonnegative")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("This PyTorch installation cannot use CUDA; install a compatible build or use --device cpu")
    encoder = TextEncoder(args.encoder, normalize=False, device=device, max_length=args.max_length)
    parameters = encoder.train_last_layers(4)
    head = controller.difficulty.to(device)
    optimizer = torch.optim.AdamW([{"params": parameters, "lr": args.encoder_learning_rate},
                                  {"params": head.parameters(), "lr": args.head_learning_rate}],
                                 weight_decay=args.weight_decay)
    counts = None
    weights = None
    if not args.no_domain_balance:
        weights, counts = domain_weights(train)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    output_encoder = args.out.with_suffix(".encoder").resolve()
    if output_encoder.exists():
        raise ValueError("Encoder output directory exists; choose a new checkpoint path")

    def loss(prediction, target):
        return .5*(F.binary_cross_entropy(prediction.clamp(1e-6, 1-1e-6), target)
                   + F.mse_loss(prediction, target))

    try:
        for epoch in range(args.epochs):
            head.train()
            encoder.model.train()
            indices = (list(WeightedRandomSampler(weights, len(train), replacement=True, generator=generator))
                       if weights else torch.randperm(len(train), generator=generator).tolist())
            total, seen = 0., 0
            for start in range(0, len(indices), args.batch_size):
                batch = [train[i] for i in indices[start:start+args.batch_size]]
                target = torch.tensor([1-r["difficulty"] for r in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = head(encoder.forward_train([r["task"] for r in batch]))
                value = loss(prediction, target)
                value.backward()
                torch.nn.utils.clip_grad_norm_([*parameters, *head.parameters()], 1.)
                optimizer.step()
                encoder.cache.clear()
                total += float(value.detach())*len(batch)
                seen += len(batch)
                if (start//args.batch_size) % 50 == 0:
                    print(f"Difficulty epoch {epoch+1}/{args.epochs}: {seen}/{len(train)}, loss={total/seen:.4f}", flush=True)
            head.eval()
            encoder.model.eval()
            mse, val_loss = 0., 0.
            with torch.no_grad():
                for start in range(0, len(validation), args.batch_size):
                    batch = validation[start:start+args.batch_size]
                    target = torch.tensor([1-r["difficulty"] for r in batch], device=device)
                    prediction = head(encoder.forward_train([r["task"] for r in batch]))
                    mse += float(F.mse_loss(prediction, target))*len(batch)
                    val_loss += float(loss(prediction, target))*len(batch)
            history.append({"train_loss": total/seen, "validation_mse": mse/len(validation),
                            "validation_loss": val_loss/len(validation)})
    except KeyboardInterrupt:
        # Persist a usable encoder/head together; controller save is done below.
        controller.training_info["difficulty_interrupted"] = True
        raise
    finally:
        encoder.freeze()
        encoder.model.save(str(output_encoder), safe_serialization=True)
        encoder.name = str(output_encoder)
        encoder.model.to("cpu")
        encoder.device = "cpu"
        head.to("cpu").eval()
        controller.difficulty_encoder = encoder
        controller.training_info.update(difficulty_steps=controller.training_info["difficulty_steps"]+len(history),
                                        difficulty_train_keys=sorted({r["key"] for r in train}),
                                        difficulty_validation_keys=[r["key"] for r in validation],
                                        difficulty_recipe="appendix-f", difficulty_max_length=args.max_length)
        controller.save(args.out)
    return {"recipe": "appendix-f", "epochs": history, "train_size": len(train),
            "validation_size": len(validation), "device": device, "batch_size": args.batch_size,
            "max_length": args.max_length, "encoder_layers": 4, "domain_counts": counts,
            "domain_ratio": "1:1:4" if weights else "disabled",
            "domain_mapping": "coding: humaneval/mbpp/code/apps; math_reasoning: gsm8k/math/reasoning/arc/hellaswag; other: general",
            "encoder_learning_rate": args.encoder_learning_rate, "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay}
