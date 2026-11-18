"""Frozen text features and a three-output neural regressor, independent of BAMAS RL."""

import copy
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import torch
from torch import nn

from .data import GROUPS, prompt_text, write_json


class TextFeatures:
    def __init__(self, backend="tfidf", encoder="sentence-transformers/all-MiniLM-L6-v2", dimensions=128):
        self.backend = backend
        self.encoder_name = encoder
        self.dimensions = dimensions
        self.encoder = None

    def fit_transform(self, texts):
        if self.backend == "sentence-transformer":
            return self.transform(texts)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(texts)
        dimensions = min(self.dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if dimensions < 1:
            raise ValueError("Need more training prompts and vocabulary")
        svd = TruncatedSVD(n_components=dimensions, random_state=42)
        svd.fit(matrix)
        self.components = svd.components_.astype(np.float32)
        return normalize(matrix @ self.components.T).astype(np.float32)

    def transform(self, texts):
        if self.backend == "sentence-transformer":
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.encoder_name)
            return self.encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                       show_progress_bar=False).astype(np.float32)
        return normalize(self.vectorizer.transform(texts) @ self.components.T).astype(np.float32)

    def save(self, directory):
        write_json(directory / "features.json", {"backend": self.backend, "encoder": self.encoder_name})
        if self.backend == "tfidf":
            np.savez_compressed(directory / "features.npz", vocabulary=self.vectorizer.get_feature_names_out().astype(str),
                                idf=self.vectorizer.idf_, components=self.components)

    @classmethod
    def load(cls, directory):
        meta = json.loads((directory / "features.json").read_text(encoding="utf-8"))
        features = cls(meta["backend"], meta["encoder"])
        if features.backend == "tfidf":
            with np.load(directory / "features.npz", allow_pickle=False) as arrays:
                features.vectorizer = TfidfVectorizer(
                    vocabulary={str(word): i for i, word in enumerate(arrays["vocabulary"])},
                    ngram_range=(1, 2), sublinear_tf=True)
                features.vectorizer.idf_ = arrays["idf"]
                features.components = arrays["components"]
        return features


class DifficultyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(hidden_dim, 3))

    def forward(self, features):
        return self.layers(features)  # logits; sigmoid applied by loss/inference


def targets(rows):
    return np.array([[r["difficulty"][g] if r["difficulty"][g] is not None else np.nan
                      for g in GROUPS] for r in rows], dtype=np.float32)


def masked_mse(predictions, truth):
    mask = torch.isfinite(truth)
    # Equal weight per pool, with missing targets excluded independently.
    squared = (predictions - torch.nan_to_num(truth)).square() * mask
    return (squared.sum(0) / mask.sum(0).clamp_min(1)).mean()


def metrics(predictions, truth):
    results = {}
    for i, group in enumerate(GROUPS):
        mask = np.isfinite(truth[:, i])
        y, p = truth[mask, i], predictions[mask, i]
        if len(y) == 0:
            results[group] = {"n": 0, "mae": None, "rmse": None, "mse": None,
                              "calibration_error": None}
            continue
        calibration = 0.0
        bins = np.minimum((p * 10).astype(int), 9)
        for b in range(10):
            selected = bins == b
            if selected.any():
                calibration += selected.mean() * abs(float(p[selected].mean() - y[selected].mean()))
        mse = float(np.mean((p - y) ** 2))
        results[group] = {"n": len(y), "mae": float(np.mean(np.abs(p - y))),
                          "rmse": float(np.sqrt(mse)), "mse": mse,
                          "calibration_error": float(calibration)}
    available = [entry["mse"] for entry in results.values() if entry["mse"] is not None]
    results["macro_mse"] = float(np.mean(available)) if available else None
    return results


def load_splits(directory):
    rows = {}
    seen_ids, seen_prompts = set(), set()
    for split in ("train", "validation", "test"):
        with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
            rows[split] = [json.loads(line) for line in stream if line.strip()]
        if not rows[split]:
            raise ValueError(f"Empty {split} split")
        ids = {r["sample_id"] for r in rows[split]}
        prompts = {r["prompt_hash"] for r in rows[split]}
        if ids & seen_ids or prompts & seen_prompts:
            raise ValueError("Question identities overlap between prepared splits")
        seen_ids.update(ids)
        seen_prompts.update(prompts)
        if not np.isfinite(targets(rows[split])).any(axis=0).all():
            raise ValueError(f"Each pool needs at least one observed target in {split}")
    return rows


class DifficultyPredictor:
    """Inference API: predict(texts) -> [{weak: D_w, intermediate: D_m, strong: D_s}]."""
    def __init__(self, network, features, manifest):
        self.network = network.eval()
        self.features = features
        self.manifest = manifest

    def predict(self, texts):
        if not texts:
            return []
        x = torch.from_numpy(self.features.transform([prompt_text(text) for text in texts]))
        with torch.no_grad():
            values = self.network(x).sigmoid().numpy()
        return [{g: float(value) for g, value in zip(GROUPS, row)} for row in values]

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if meta.get("backend") == "agentbalance":
            from .transformer import TransformerDifficultyPredictor
            return TransformerDifficultyPredictor.load(directory, device=device)
        if meta.get("backend") == "frozen-mpnet-ensemble":
            from .frozen import FrozenEnsemblePredictor
            return FrozenEnsemblePredictor.load(directory, device=device)
        network = DifficultyNetwork(meta["input_dim"], meta["hidden_dim"])
        network.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return cls(network, TextFeatures.load(directory), manifest)


def train(prepared, output, epochs=50, patience=8, seed=42, backend="tfidf",
          encoder="sentence-transformers/all-MiniLM-L6-v2", hidden_dim=128, batch_size=256):
    if min(epochs, patience, hidden_dim, batch_size) < 1:
        raise ValueError("epochs, patience, hidden_dim and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_splits(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    features = TextFeatures(backend, encoder)
    x_train = features.fit_transform([r["prompt"] for r in rows["train"]])
    x_val = features.transform([r["prompt"] for r in rows["validation"]])
    y = {split: targets(entries) for split, entries in rows.items()}
    network = DifficultyNetwork(x_train.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-3)
    training = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y["train"]))
    loader = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x_val), torch.from_numpy(y["validation"])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        for inputs, truth in loader:
            if not torch.isfinite(truth).any():
                continue
            optimizer.zero_grad()
            loss = masked_mse(network(inputs).sigmoid(), truth)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(network(val_x).sigmoid(), val_y))
        history.append({"epoch": epoch, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
        print(f"epoch={epoch} validation_mse={val_loss:.6f}", flush=True)
        if stale >= patience:
            break
    network.load_state_dict(best_state)
    network.eval()
    # Test features and predictions are computed only after model selection.
    x_test = features.transform([r["prompt"] for r in rows["test"]])
    with torch.no_grad():
        predicted = network(torch.from_numpy(x_test)).sigmoid().numpy()
    mean = np.nanmean(y["train"], axis=0)
    constant = np.tile(mean, (len(predicted), 1))
    by_dataset = {}
    for dataset in sorted({r["eval_name"] for r in rows["test"]}):
        test_mask = np.array([r["eval_name"] == dataset for r in rows["test"]])
        train_mask = np.array([r["eval_name"] == dataset for r in rows["train"]])
        dataset_mean = np.nanmean(y["train"][train_mask], axis=0) if train_mask.any() else mean
        dataset_mean = np.where(np.isfinite(dataset_mean), dataset_mean, mean)
        by_dataset[dataset] = {"network": metrics(predicted[test_mask], y["test"][test_mask]),
                               "dataset_mean_baseline": metrics(np.tile(dataset_mean, (test_mask.sum(), 1)), y["test"][test_mask])}
    report = {"best_epoch": best_epoch, "seed": seed, "backend": backend,
              "test": metrics(predicted, y["test"]), "constant_baseline": metrics(constant, y["test"]),
              "by_dataset": by_dataset, "history": history,
              "interpretation": "Held-out pool-relative quality deficit; not causal token-budget effects or agent success."}
    features.save(output)
    torch.save(network.state_dict(), output / "weights.pt")
    write_json(output / "model.json", {"input_dim": x_train.shape[1], "hidden_dim": hidden_dim})
    write_json(output / "manifest.json", manifest)
    write_json(output / "metrics.json", report)
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, values in zip(rows["test"], predicted):
            stream.write(json.dumps({"sample_id": row["sample_id"], "eval_name": row["eval_name"],
                                     "target": row["difficulty"], "prediction": dict(zip(GROUPS, map(float, values)))}) + "\n")
    return report
