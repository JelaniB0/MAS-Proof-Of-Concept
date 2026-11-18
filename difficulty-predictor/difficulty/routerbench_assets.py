"""Download pinned public RouterBench assets; verify bytes before accepting them."""

import hashlib
from pathlib import Path
import urllib.request

ASSETS = {
    "five": {"revision":"784021482c3f320c6619ed4b3bb3b41a21424fcb", "file":"routerbench_5shot.pkl",
             "size":171091324,"sha256":"fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe"},
    "raw": {"revision":"784021482c3f320c6619ed4b3bb3b41a21424fcb", "file":"routerbench_raw.pkl",
            "size":1198134681,"sha256":"18e919267500a1d49302e0efe25cf7d9aa2f5b74aa209f5230442c2cfbee7971"},
    "rag": {"revision":"784e39ca67a6c6291685b4862a05dd95fb0b40da", "file":"routerbench_rag.pkl",
            "size":1581385,"sha256":"334450c9d4a70c8aaf845387856e3a3a1f0afea963cf61edcec17f5d44eb6f2c"},
}


def fetch_asset(name, directory):
    asset = ASSETS[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / asset["file"]
    if target.exists():
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size != asset["size"] or digest != asset["sha256"]:
            raise ValueError(f"Existing {target.name} does not match the pinned public asset")
        return {**asset, "path":str(target), "status":"verified existing"}
    url = f"https://huggingface.co/datasets/withmartian/routerbench/resolve/{asset['revision']}/{asset['file']}"
    partial = target.with_suffix(".part")
    digest, size, progress = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
            size += len(block)
            if size - progress >= 64 * 1024**2:
                print(f"{target.name}: {size / 1024**2:.0f}/{asset['size'] / 1024**2:.0f} MiB",flush=True)
                progress = size
    if size != asset["size"] or digest.hexdigest() != asset["sha256"]:
        raise ValueError("Downloaded asset fails the pinned length/checksum verification")
    partial.replace(target)
    return {**asset, "path":str(target), "status":"downloaded and verified"}
