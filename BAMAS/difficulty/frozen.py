"""Frozen MPNet representations, cached features and small ensemble prediction heads."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .data import GROUPS, write_json
from .inputs import INPUT_VERSION, input_version, input_text, controller_features, contract_id
from .transformer import resolve_device, tokenize_rows, make_batch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictor_fingerprint(directory, metadata, manifest):
    directory = Path(directory).resolve()
    files = []
    for filename in metadata["heads"]:
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Head path escapes checkpoint directory")
        files.append(sha256_file(path))
    payload = {"encoder": metadata["encoder_weights_sha256"], "heads": files,
               "encoder_configuration": metadata.get("encoder_configuration_sha256"),
               "tokenizer": metadata.get("tokenizer_sha256"),
               "input_mode": metadata["input_mode"], "input_version": metadata["input_version"],
               "pool_contract_id": contract_id(manifest), "max_length": metadata["max_length"]}
    if metadata.get("input_support"):
        payload["input_support"] = metadata["input_support"]
    if metadata.get("prediction_level") == "model":
        payload["prediction_level"] = "model"
        payload["model_order"] = metadata["model_order"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class FrozenMPNet:
    def __init__(self, source, device="auto", max_length=384, batch_size=16, threads=4):
        self.source = Path(source).resolve()
        self.device = resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        if min(batch_size, threads) < 1 or max_length < 3:
            raise ValueError("Invalid encoder batch size, thread count or maximum length")
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, local_files_only=True, trust_remote_code=False)
        self.tokenizer.truncation_side = "left"
        self.encoder = AutoModel.from_pretrained(self.source, local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        if self.encoder.config.model_type != "mpnet":
            raise ValueError("This feature extractor requires MPNet")
        if max_length > self.encoder.config.max_position_embeddings - self.encoder.config.pad_token_id - 1:
            raise ValueError("max_length exceeds MPNet positional capacity")
        self.encoder.requires_grad_(False)
        self.weights_sha256 = sha256_file(self.source / "model.safetensors")
        self.configuration_sha256 = sha256_file(self.source / "config.json")
        tokenizer_files = {name: sha256_file(self.source / name) for name in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
            if (self.source / name).is_file()}
        self.tokenizer_sha256 = hashlib.sha256(json.dumps(tokenizer_files, sort_keys=True).encode()).hexdigest()

    def encode(self, texts, mode="question"):
        texts = [input_text(text, mode) for text in texts]
        if not texts:
            return np.empty((0, self.encoder.config.hidden_size), dtype=np.float32), {}
        items, truncation = tokenize_rows(self.tokenizer, [{"prompt": text} for text in texts], self.max_length)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                batch = make_batch(self.tokenizer, items, range(start, min(start + self.batch_size, len(items))), self.device)
                hidden = self.encoder(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                outputs.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
                if (start // self.batch_size + 1) % 128 == 0:
                    print(f"Encoded {min(start + self.batch_size, len(items))}/{len(items)} prompts", file=sys.stderr, flush=True)
        return np.concatenate(outputs).astype(np.float32), truncation

    def cached_encode(self, rows, cache, mode):
        texts = [input_text(row["prompt"], mode) for row in rows]
        descriptor = {"input_version": input_version(mode), "mode": mode, "encoder_weights_sha256": self.weights_sha256,
                      "encoder_configuration_sha256": self.configuration_sha256, "tokenizer_sha256": self.tokenizer_sha256,
                      "max_length": self.max_length, "pooling": "masked_mean_l2",
                      "text_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        arrays_path, metadata_path = cache / f"{key}.npz", cache / f"{key}.json"
        if arrays_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["descriptor"] == descriptor:
                with np.load(arrays_path, allow_pickle=False) as arrays:
                    values = arrays["embeddings"]
                if values.shape == (len(rows), self.encoder.config.hidden_size):
                    return values, metadata["truncation"]
        values, truncation = self.encode(texts, mode="full")
        np.savez_compressed(arrays_path, embeddings=values)
        write_json(metadata_path, {"descriptor": descriptor, "truncation": truncation})
        return values, truncation


class PoolHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.2, output_dim=None):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, output_dim or len(GROUPS)))

    def forward(self, x):
        return self.layers(x)


def aggregate_model_difficulty(values, model_order, groups):
    if len(model_order) != len(set(model_order)) or set(model_order) != {m for g in GROUPS for m in groups[g]}:
        raise ValueError("Model output order must partition the checkpoint pools")
    return np.stack([values[:,[model_order.index(model) for model in groups[group]]].mean(axis=1)
                     for group in GROUPS],axis=1)


class FrozenEnsemblePredictor:
    def __init__(self, encoder, heads, metadata, manifest):
        self.encoder = encoder
        self.heads = [head.eval().cpu() for head in heads]
        self.metadata = metadata
        self.manifest = manifest
        self.input_support = None

    def predict(self, texts):
        if not texts:
            return []
        values, _ = self.encoder.encode(texts, mode=self.metadata["input_mode"])
        with torch.inference_mode():
            predictions = torch.stack([head(torch.from_numpy(values)).sigmoid() for head in self.heads]).mean(0).numpy()
        if self.metadata.get("prediction_level") == "model":
            predictions=aggregate_model_difficulty(predictions,self.metadata["model_order"],self.manifest["groups"])
        return [{group: float(value) for group, value in zip(GROUPS, row)} for row in predictions]

    def features(self, texts):
        return controller_features(self, texts)

    def predict_with_support(self, texts):
        if not texts:
            return []
        values, _ = self.encoder.encode(texts, mode=self.metadata["input_mode"])
        with torch.inference_mode():
            predictions = torch.stack([head(torch.from_numpy(values)).sigmoid() for head in self.heads]).mean(0).numpy()
        model_values=predictions if self.metadata.get("prediction_level") == "model" else None
        if model_values is not None:
            predictions=aggregate_model_difficulty(predictions,self.metadata["model_order"],self.manifest["groups"])
        support = self.input_support.assess(values) if self.input_support else [None] * len(values)
        outputs=[{"prediction":dict(zip(GROUPS,map(float,row))),"input_support":assessment}
                 for row,assessment in zip(predictions,support)]
        if model_values is not None:
            for output,raw in zip(outputs,model_values):
                output["model_difficulty"]=dict(zip(self.metadata["model_order"],map(float,raw)))
        return outputs

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if metadata["group_order"] != list(GROUPS) or metadata["input_version"] != input_version(metadata["input_mode"]):
            raise ValueError("Unsupported input or group contract")
        encoder = FrozenMPNet(directory / "encoder", device, metadata["max_length"])
        if encoder.weights_sha256 != metadata["encoder_weights_sha256"]:
            raise ValueError("Encoder checksum does not match checkpoint")
        if metadata.get("tokenizer_sha256", encoder.tokenizer_sha256) != encoder.tokenizer_sha256:
            raise ValueError("Tokenizer checksum does not match checkpoint")
        if metadata.get("encoder_configuration_sha256", encoder.configuration_sha256) != encoder.configuration_sha256:
            raise ValueError("Encoder configuration checksum does not match checkpoint")
        heads = []
        for filename in metadata["heads"]:
            # Artifact paths must remain inside the checkpoint directory.
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Head path escapes checkpoint directory")
            head = PoolHead(metadata["input_dim"], metadata["hidden_dim"], metadata["dropout"],
                            len(metadata["model_order"]) if metadata.get("prediction_level") == "model" else None)
            head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            heads.append(head)
        if not heads:
            raise ValueError("Checkpoint must contain at least one prediction head")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        identity = predictor_fingerprint(directory, metadata, manifest)
        if metadata.get("predictor_id", identity) != identity:
            raise ValueError("Prediction head fingerprint does not match checkpoint")
        metadata["predictor_id"] = identity
        model = cls(encoder, heads, metadata, manifest)
        if metadata.get("input_support"):
            from .support import InputSupport
            model.input_support = InputSupport.load(directory, metadata["input_support"])
        return model


def export_ensemble(source_encoder, head_files, destination, metadata, manifest):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    encoder_destination = destination / "encoder"
    encoder_destination.mkdir(exist_ok=True)
    for filename in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.txt", "encoder_source.json"):
        source = Path(source_encoder) / filename
        if source.is_file():
            shutil.copy2(source, encoder_destination / filename)
    filenames = []
    for index, source in enumerate(head_files):
        filename = f"head_{index}.pt"
        shutil.copy2(source, destination / filename)
        filenames.append(filename)
    metadata = {**metadata, "backend": "frozen-mpnet-ensemble", "group_order": list(GROUPS),
                "heads": filenames, "input_version": input_version(metadata["input_mode"])}
    metadata["predictor_id"] = predictor_fingerprint(destination, metadata, manifest)
    write_json(destination / "model.json", metadata)
    write_json(destination / "manifest.json", manifest)
