"""AgentBalance-style MPNet fine-tuning with three pool-relative outputs.

Architecture/training source: https://arxiv.org/html/2512.11426v1#A6
This adapts the supervised estimator, not AgentBalance's downstream RL system.
"""

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import GROUPS, prompt_text, write_json
from .model import load_splits, metrics, targets
from .inputs import input_text

DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch installation has no available CUDA device")
    return selected


class MPNetDifficultyNetwork(nn.Module):
    def __init__(self, encoder, hidden_dim=512, unfreeze_last=4, dropout=0.1, normalize_embeddings=False):
        super().__init__()
        if encoder.config.model_type != "mpnet":
            raise ValueError("The agentbalance backend requires an MPNet encoder")
        if not 0 <= unfreeze_last <= len(encoder.encoder.layer):
            raise ValueError("unfreeze_last must be between zero and the number of encoder layers")
        self.encoder = encoder
        self.unfreeze_last = unfreeze_last
        self.normalize_embeddings = normalize_embeddings
        self.head = nn.Sequential(nn.Linear(encoder.config.hidden_size, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, len(GROUPS)))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if unfreeze_last:
            for layer in self.encoder.encoder.layer[-unfreeze_last:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen embeddings/prefix stay deterministic; only adapted layers use dropout.
        self.encoder.eval()
        if mode and self.unfreeze_last:
            for layer in self.encoder.encoder.layer[-self.unfreeze_last:]:
                layer.train()
        return self

    def encode(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        if self.normalize_embeddings:
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))


def combined_loss(logits, truth):
    """0.5 * (soft-target BCE + MSE); missing pools receive zero gradient."""
    mask = torch.isfinite(truth)
    safe = torch.nan_to_num(truth)
    if ((safe < 0) | (safe > 1)).any():
        raise ValueError("Difficulty targets must be in [0, 1] or NaN")
    elementwise = 0.5 * (F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
                         + (logits.sigmoid() - safe).square())
    return (elementwise * mask).sum() / mask.sum().clamp_min(1)


def optimizer_for(network, encoder_lr=1e-5, head_lr=5e-4, weight_decay=1e-3):
    groups = [{"params": list(network.head.parameters()), "lr": head_lr}]
    encoder_parameters = [p for p in network.encoder.parameters() if p.requires_grad]
    if encoder_parameters:
        groups.insert(0, {"params": encoder_parameters, "lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def tokenize_rows(tokenizer, rows, max_length):
    """Cache token IDs, not embeddings: adapted encoder layers must receive gradients."""
    encoded = tokenizer([prompt_text(row["prompt"]) for row in rows], padding=False,
                        truncation=False, return_attention_mask=True, verbose=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    items = []
    for ids in encoded["input_ids"]:
        if len(ids) > max_length:
            # Preserve special tokens on both sides; MPNet uses a leading <s>.
            ids = ([ids[0]] + ids[-(max_length - 1):]) if tokenizer.truncation_side == "left" else (
                ids[:max_length - 1] + [ids[-1]])
        items.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return items, {"questions": len(rows), "truncated": sum(n > max_length for n in lengths),
                   "max_original_tokens": max(lengths, default=0),
                   "fraction_truncated": sum(n > max_length for n in lengths) / max(len(lengths), 1)}


def make_batch(tokenizer, items, indices, device):
    batch = tokenizer.pad([items[int(i)] for i in indices], padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}


def predict_items(network, tokenizer, items, device, batch_size):
    network.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = make_batch(tokenizer, items, range(start, min(start + batch_size, len(items))), device)
            outputs.append(network(**batch).sigmoid().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, len(GROUPS)), dtype=np.float32)


def save_checkpoint(network, tokenizer, directory, metadata, manifest):
    """Self-contained safetensors artifacts; loading does not require the original encoder."""
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=True)
    network.encoder.config.save_pretrained(directory / "encoder_config")
    tokenizer.save_pretrained(directory / "tokenizer")
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in network.state_dict().items()},
              str(directory / "model.safetensors"))
    write_json(directory / "model.json", metadata)
    write_json(directory / "manifest.json", manifest)


def transformer_fingerprint(directory, metadata, manifest):
    import hashlib
    from .inputs import contract_id
    from .support import file_sha256
    directory = Path(directory)
    tokenizer = {p.name:file_sha256(p) for p in (directory / "tokenizer").iterdir() if p.is_file()}
    payload = {"version":"transformer-inference-v1", "weights":file_sha256(directory / "model.safetensors"),
               "encoder_configuration":file_sha256(directory / "encoder_config/config.json"),"tokenizer":tokenizer,
               "contract":contract_id(manifest),"input_support":metadata.get("input_support"),
               "inference":{k:metadata.get(k) for k in ("input_mode","max_length","truncation_side","normalize_embeddings","hidden_dim","dropout","group_order")}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


class TransformerDifficultyPredictor:
    def __init__(self, network, tokenizer, metadata, manifest, device="cpu"):
        self.device = resolve_device(device)
        self.network = network.to(self.device).eval()
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.manifest = manifest
        self.input_support = None

    def predict(self, texts, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return []
        items, _ = tokenize_rows(self.tokenizer, [{"prompt": input_text(text, self.metadata.get("input_mode", "full"))}
                                                for text in texts], self.metadata["max_length"])
        values = predict_items(self.network, self.tokenizer, items, self.device, batch_size)
        return [{group: float(value) for group, value in zip(GROUPS, row)} for row in values]

    def embed(self, texts, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0,self.network.encoder.config.hidden_size),dtype=np.float32)
        items,_ = tokenize_rows(self.tokenizer,[{"prompt":input_text(text,self.metadata.get("input_mode","full"))}
                                                for text in texts],self.metadata["max_length"])
        values=[]
        with torch.inference_mode():
            for start in range(0,len(items),batch_size):
                batch=make_batch(self.tokenizer,items,range(start,min(start+batch_size,len(items))),self.device)
                values.append(self.network.encode(**batch).cpu().numpy())
        return np.concatenate(values)

    def predict_with_support(self,texts):
        if not texts:
            return []
        values=self.embed(texts)
        with torch.inference_mode():
            predicted=self.network.head(torch.from_numpy(values).to(self.device)).sigmoid().cpu().numpy()
        support=self.input_support.assess(values) if self.input_support else [None]*len(values)
        return [{"prediction":dict(zip(GROUPS,map(float,row))),"input_support":assessment}
                for row,assessment in zip(predicted,support)]

    @classmethod
    def load(cls, directory, device="cpu"):
        from safetensors.torch import load_file
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if metadata["group_order"] != list(GROUPS):
            raise ValueError("Unexpected checkpoint group order")
        config = AutoConfig.from_pretrained(directory / "encoder_config", local_files_only=True, trust_remote_code=False)
        network = MPNetDifficultyNetwork(AutoModel.from_config(config, trust_remote_code=False),
                                         metadata["hidden_dim"], metadata["unfreeze_last"], metadata["dropout"],
                                         metadata.get("normalize_embeddings", False))
        network.load_state_dict(load_file(str(directory / "model.safetensors")))
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
        tokenizer.truncation_side = metadata["truncation_side"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("predictor_id") and metadata["predictor_id"] != transformer_fingerprint(directory,metadata,manifest):
            raise ValueError("Transformer predictor fingerprint does not match checkpoint")
        predictor = cls(network, tokenizer, metadata, manifest, device)
        if metadata.get("input_support"):
            from .support import InputSupport
            predictor.input_support=InputSupport.load(directory,metadata["input_support"])
        return predictor


def sample_indices(rows, generator, balance_datasets=False):
    if not balance_datasets:
        return torch.randperm(len(rows), generator=generator).tolist()
    counts = Counter(row["eval_name"] for row in rows)
    weights = torch.tensor([1 / counts[row["eval_name"]] for row in rows], dtype=torch.double)
    return torch.multinomial(weights, len(rows), replacement=True, generator=generator).tolist()


def subset_rows(rows, maximum, seed):
    if maximum is None:
        return rows
    if maximum < 1:
        raise ValueError("Sample limits must be positive")
    indices = np.random.default_rng(seed).permutation(len(rows))[:maximum]
    return [rows[int(index)] for index in sorted(indices)]


def train_transformer(prepared, output, epochs=10, patience=3, seed=42, encoder=DEFAULT_ENCODER,
                      hidden_dim=512, batch_size=8, unfreeze_last=4, max_length=384,
                      encoder_lr=1e-5, head_lr=5e-4, weight_decay=1e-3, device="auto",
                      balance_datasets=False, revision=None, local_files_only=False,
                      max_train_samples=None, max_eval_samples=None, accumulation_steps=1,
                      truncation_side="left", threads=4, input_mode="full", loss_kind="combined",
                      dropout=0.1, normalize_embeddings=False):
    if min(epochs, patience, hidden_dim, batch_size, accumulation_steps, threads) < 1 or max_length < 3:
        raise ValueError("Positive training settings and max_length >= 3 are required")
    if min(encoder_lr, head_lr) <= 0 or weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight_decay nonnegative")
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be left or right")
    if input_mode not in {"full", "question", "task"} or loss_kind not in {"combined", "mse"} or not 0 <= dropout < 1:
        raise ValueError("Invalid input mode, loss or dropout")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    selected_device = resolve_device(device)
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_splits(prepared)
    rows["train"] = subset_rows(rows["train"], max_train_samples, seed)
    for split in ("validation", "test"):
        rows[split] = subset_rows(rows[split], max_eval_samples, seed)
    rows = {split: [{**row, "prompt": input_text(row["prompt"], input_mode)} for row in records]
            for split, records in rows.items()}
    if not np.isfinite(targets(rows["train"])).any(axis=0).all():
        raise ValueError("Training subset must contain observed targets for every pool")
    source_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    manifest = {**source_manifest, "training_subset_counts": {s: len(r) for s, r in rows.items()}}
    tokenizer = AutoTokenizer.from_pretrained(encoder, revision=revision, local_files_only=local_files_only,
                                             trust_remote_code=False)
    tokenizer.truncation_side = truncation_side
    backbone = AutoModel.from_pretrained(encoder, revision=revision, local_files_only=local_files_only,
                                         trust_remote_code=False, use_safetensors=True)
    if max_length > backbone.config.max_position_embeddings - backbone.config.pad_token_id - 1:
        raise ValueError("max_length exceeds MPNet's positional capacity")
    # Match the frozen-head initialization in normalization-controlled experiments.
    if normalize_embeddings:
        torch.manual_seed(seed)
    network = MPNetDifficultyNetwork(backbone, hidden_dim, unfreeze_last, dropout, normalize_embeddings).to(selected_device)
    metadata = {"backend": "agentbalance", "encoder": str(encoder), "encoder_revision": revision,
                "resolved_encoder_revision": getattr(backbone.config, "_commit_hash", None),
                "hidden_dim": hidden_dim, "dropout": dropout, "unfreeze_last": unfreeze_last,
                "input_mode": input_mode, "normalize_embeddings": normalize_embeddings, "loss_kind": loss_kind,
                "max_length": max_length, "truncation_side": truncation_side,
                "group_order": list(GROUPS), "target": manifest.get("target_definition", "difficulty = 1 - pool mean quality")}
    local_provenance = Path(encoder) / "encoder_source.json"
    if local_provenance.is_file():
        metadata["pretrained_source"] = json.loads(local_provenance.read_text(encoding="utf-8"))
    items, truncation = {}, {}
    for split in ("train", "validation"):
        items[split], truncation[split] = tokenize_rows(tokenizer, rows[split], max_length)
    truth = {split: targets(value) for split, value in rows.items()}
    optimizer = optimizer_for(network, encoder_lr, head_lr, weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_mse, best_epoch, stale = float("inf"), 0, 0
    history = []
    print(json.dumps({"device": str(selected_device), "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                      "total_parameters": sum(p.numel() for p in network.parameters()), "truncation": truncation}), flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        network.train()
        indices = sample_indices(rows["train"], generator, balance_datasets)
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, indices_batch in enumerate(batches):
            # The last accumulation window may contain fewer microbatches.
            window_start = (step // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, len(batches) - window_start)
            window_indices = [i for block in batches[window_start:window_start + window_size] for i in block]
            window_observations = max(int(np.isfinite(truth["train"][window_indices]).sum()), 1)
            logits = network(**make_batch(tokenizer, items["train"], indices_batch, selected_device))
            target = torch.from_numpy(truth["train"][indices_batch]).to(selected_device)
            if loss_kind == "combined":
                loss = combined_loss(logits, target)
            else:
                mask = torch.isfinite(target)
                loss = ((logits.sigmoid() - torch.nan_to_num(target)).square() * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            # Weight by observed targets, including a short final microbatch.
            observations = int(np.isfinite(truth["train"][indices_batch]).sum())
            (loss * observations / window_observations).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(batches):
                nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 25 == 0:
                print(f"epoch={epoch} batch={step + 1}/{len(batches)} loss={float(loss.detach()):.6f}", flush=True)
        predictions = predict_items(network, tokenizer, items["validation"], selected_device, batch_size)
        validation = metrics(predictions, truth["validation"])
        value = validation["macro_mse"]
        if value is None or not math.isfinite(value):
            raise ValueError("Validation subset needs finite targets")
        history.append({"epoch": epoch, "training_combined_loss": float(np.mean(losses)),
                        "validation_macro_mse": value, "seconds": time.perf_counter() - started})
        print(json.dumps(history[-1]), flush=True)
        if value < best_mse - 1e-6:
            best_mse, best_epoch, stale = value, epoch, 0
            save_checkpoint(network, tokenizer, output, metadata, manifest)
        else:
            stale += 1
        if stale >= patience:
            break
    from safetensors.torch import load_file
    network.load_state_dict(load_file(str(output / "model.safetensors"), device=str(selected_device)))
    # Held-out test encoding occurs after validation checkpoint selection.
    items["test"], truncation["test"] = tokenize_rows(tokenizer, rows["test"], max_length)
    predictions = predict_items(network, tokenizer, items["test"], selected_device, batch_size)
    report = evaluation_report(predictions, truth, rows)
    report.update({"backend": "agentbalance", "best_epoch": best_epoch, "seed": seed, "device": str(selected_device),
                   "loss": "0.5 * (soft-target BCE + MSE)" if loss_kind == "combined" else "masked MSE", "history": history, "truncation": truncation,
                   "training": {"epochs": epochs, "patience": patience, "encoder_lr": encoder_lr,
                                "head_lr": head_lr, "weight_decay": weight_decay, "batch_size": batch_size,
                                "accumulation_steps": accumulation_steps, "balance_datasets": balance_datasets,
                                "counts": {s: len(r) for s, r in rows.items()}},
                   "smoke_test": max_train_samples is not None or max_eval_samples is not None,
                   "interpretation": "Adaptation of AgentBalance estimator; three difficulty heads; no downstream BAMAS RL integration."})
    write_json(output / "metrics.json", report)
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, values in zip(rows["test"], predictions):
            stream.write(json.dumps({"sample_id": row["sample_id"], "eval_name": row["eval_name"],
                                     "target": row["difficulty"], "prediction": dict(zip(GROUPS, map(float, values)))}) + "\n")
    return report


def evaluation_report(predictions, truth, rows):
    mean = np.nanmean(truth["train"], axis=0)
    report = {"test": metrics(predictions, truth["test"]),
              "constant_baseline": metrics(np.tile(mean, (len(predictions), 1)), truth["test"]), "by_dataset": {}}
    for dataset in sorted({row["eval_name"] for row in rows["test"]}):
        test_mask = np.array([r["eval_name"] == dataset for r in rows["test"]])
        train_mask = np.array([r["eval_name"] == dataset for r in rows["train"]])
        dataset_mean = np.nanmean(truth["train"][train_mask], axis=0) if train_mask.any() else mean
        dataset_mean = np.where(np.isfinite(dataset_mean), dataset_mean, mean)
        report["by_dataset"][dataset] = {
            "network": metrics(predictions[test_mask], truth["test"][test_mask]),
            "dataset_mean_baseline": metrics(np.tile(dataset_mean, (test_mask.sum(), 1)), truth["test"][test_mask])}
    return report
