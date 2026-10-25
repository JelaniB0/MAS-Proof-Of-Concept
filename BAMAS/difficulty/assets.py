"""Download only data/configuration files and safetensors for the MPNet encoder."""

from pathlib import Path
import json


def download_encoder(destination, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    model_id = "sentence-transformers/all-mpnet-base-v2"
    commit = HfApi().model_info(model_id, revision=revision or "main").sha
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, revision=commit, local_dir=destination,
                      cache_dir=destination / ".hub_cache",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json"])
    if not (destination / "model.safetensors").is_file():
        raise ValueError("The encoder snapshot did not contain safetensors weights")
    metadata = {"model_id": model_id, "revision": commit}
    (destination / "encoder_source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
