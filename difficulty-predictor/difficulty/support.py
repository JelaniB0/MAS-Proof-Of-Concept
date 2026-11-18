"""Validation-calibrated input-support checks; not confidence probabilities."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Support features must be nonempty finite matrices")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero support vectors are invalid")
    return values / norms


class InputSupport:
    def __init__(self, training, threshold):
        self.training = unit_vectors(training)
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 2:
            raise ValueError("Invalid support threshold")

    def distances(self, values):
        values = unit_vectors(values)
        return np.concatenate([np.clip(1 - (values[start:start+256] @ self.training.T).max(axis=1), 0, 2)
                               for start in range(0, len(values), 256)])

    def assess(self, values):
        return [{"within_support": bool(value <= self.threshold), "distance":float(value), "threshold":self.threshold,
                 "method":"nearest-training cosine distance; validation 95th percentile",
                 "interpretation":"Input-similarity heuristic, not confidence or error probability."}
                for value in self.distances(values)]

    @classmethod
    def load(cls, directory, metadata):
        directory = Path(directory).resolve()
        path = (directory / metadata["file"]).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != metadata["sha256"]:
            raise ValueError("Input-support file path or checksum is invalid")
        with np.load(path, allow_pickle=False) as data:
            values = data["embeddings"]
        return cls(values, metadata["threshold"])


def attach_support(directory, prepared, device="auto"):
    from .data import write_json
    from .model import DifficultyPredictor, load_splits
    from .frozen import predictor_fingerprint
    directory, prepared = Path(directory), Path(prepared)
    predictor = DifficultyPredictor.load(directory, device=device)
    metadata = predictor.metadata
    if metadata.get("input_support"):
        return metadata["input_support"]
    rows = load_splits(prepared)
    # Only training and validation text enters the support fit; no outcomes or test inputs.
    if metadata["backend"] == "frozen-mpnet-ensemble":
        cache = prepared.parent / "frozen_embedding_cache"
        training, _ = predictor.encoder.cached_encode(rows["train"], cache, metadata["input_mode"])
        validation, _ = predictor.encoder.cached_encode(rows["validation"], cache, metadata["input_mode"])
    else:
        training = predictor.embed([r["prompt"] for r in rows["train"]], batch_size=32)
        validation = predictor.embed([r["prompt"] for r in rows["validation"]], batch_size=32)
    provisional = InputSupport(training, 2)
    threshold = float(np.quantile(provisional.distances(validation), .95))
    path = directory / "input_support.npz"
    np.savez_compressed(path, embeddings=training)
    support = {"version":"nearest-cosine-v1", "file":path.name, "sha256":file_sha256(path),
               "threshold":threshold,"validation_quantile":.95,"training_questions":len(training),
               "validation_questions":len(validation),"fit_uses":"Training/validation text only; no test/probe thresholds or labels."}
    metadata["input_support"] = support
    if metadata["backend"] == "frozen-mpnet-ensemble":
        metadata["predictor_id"] = predictor_fingerprint(directory, metadata, predictor.manifest)
    else:
        from .transformer import transformer_fingerprint
        metadata["predictor_id"] = transformer_fingerprint(directory, metadata, predictor.manifest)
    write_json(directory / "model.json", metadata)
    return support
