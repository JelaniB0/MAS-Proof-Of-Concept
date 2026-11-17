"""Download only pinned public evaluation data; never execute dataset code."""

import hashlib
import json
from pathlib import Path
import urllib.request


def download_assets(output):
    manifest = json.loads(Path(__file__).with_name("evaluation_sources.json").read_text(encoding="utf-8"))
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    for entry in [*manifest["math"]["files"],manifest["humaneval"]]:
        destination = (output / entry["path"]).resolve()
        if not destination.is_relative_to(output):
            raise ValueError("Invalid evaluation asset path")
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest()==entry["sha256"]:
            continue
        if not entry["url"].startswith(("https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/"+manifest["math"]["revision"]+"/",
                                       "https://raw.githubusercontent.com/openai/human-eval/"+manifest["humaneval"]["revision"]+"/")):
            raise ValueError("Unexpected evaluation source URL")
        request = urllib.request.Request(entry["url"],headers={"User-Agent":"BAMAS-difficulty-evaluation"})
        with urllib.request.urlopen(request,timeout=60) as stream:
            content = stream.read(entry["bytes"]+1)
        if len(content)!=entry["bytes"] or hashlib.sha256(content).hexdigest()!=entry["sha256"]:
            raise ValueError("Evaluation data size/checksum mismatch")
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
        print(f"Verified {entry['path']}",flush=True)
    (output/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return {"directory":str(output),"math_test_questions":5000,"humaneval_questions":164,
            "math_revision":manifest["math"]["revision"],"humaneval_revision":manifest["humaneval"]["revision"]}
