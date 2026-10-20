"""Frozen sentence embeddings, with an explicitly labelled offline test encoder."""

import hashlib
from pathlib import Path
import re

import torch
from torch import nn


class TextEncoder:
    def __init__(self, name="sentence-transformers/all-mpnet-base-v2", dimension=128, normalize=True,
                 device="cpu", max_length=384):
        self.name = name
        self.normalize = normalize
        self.device = device
        self.cache = {}
        self.model = None
        if name == "hash-test":
            self.dimension = dimension
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ValueError("Install research dependencies: python -m pip install -e '.[research]'") from exc
            # Loading requires an explicitly requested model or local model directory.
            self.model = SentenceTransformer(name, device=device, trust_remote_code=False,
                                             cache_folder=str(Path.cwd() / "checkpoints/hf-cache/hub"))
            self.model.max_seq_length = max_length
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            dimension_method = getattr(self.model, "get_embedding_dimension", None)
            self.dimension = (dimension_method() if dimension_method is not None
                              else self.model.get_sentence_embedding_dimension())

    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing and self.model is not None:
            with torch.no_grad():
                vectors = self.model.encode(missing, batch_size=8, convert_to_tensor=True, normalize_embeddings=self.normalize,
                                            show_progress_bar=False).detach().cpu()
            self.cache.update(zip(missing, vectors))
        elif missing:
            for text in missing:
                vector = torch.zeros(self.dimension)
                for token in re.findall(r"\w+", text.lower()):
                    digest = hashlib.sha256(token.encode()).digest()
                    vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1 if digest[4] % 2 else -1
                self.cache[text] = nn.functional.normalize(vector, dim=0)
        return torch.stack([self.cache[t] for t in texts])

    def train_last_layers(self, count=4):
        if self.model is None:
            raise ValueError("Encoder fine-tuning requires MPNet, not hash-test")
        transformer = self.model[0].auto_model
        if transformer.config.model_type != "mpnet":
            raise ValueError("Paper difficulty training requires an MPNet encoder")
        layers = transformer.encoder.layer
        if len(layers) < count:
            raise ValueError("Encoder has fewer layers than requested")
        for p in self.model.parameters():
            p.requires_grad_(False)
        for layer in layers[-count:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        self.cache.clear()
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward_train(self, texts):
        features = self.model.tokenize(texts)
        features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        return self.model(features)["sentence_embedding"]

    def freeze(self):
        if self.model is not None:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.cache.clear()


class DifficultyEstimator(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, query):
        return self.layers(query).squeeze(-1).sigmoid()
