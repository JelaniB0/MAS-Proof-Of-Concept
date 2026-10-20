"""Public benchmark acquisition and reference-compatible selection manifests."""

import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.parse import urlencode
from urllib.request import urlopen

from .adaptive.benchmarks import normalize, source_rows, fetch_dataset
from .adaptive.data import task_key
from .files import replace_file

REFERENCE_COMMIT = "e5aeac4366fe6296d4a1381ef537508e3af784e9"


def download(url, target):
    target = Path(target)
    manifest = target.with_suffix(target.suffix+".download.json")
    if target.exists():
        if not manifest.exists():
            raise ValueError(f"Unverified existing download: {target}; choose a fresh directory")
        expected = json.loads(manifest.read_text())["sha256"]
        if digest(target) != expected:
            raise ValueError(f"Download hash mismatch: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+".partial")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024*1024):
            handle.write(chunk)
    replace_file(temporary, target)
    manifest.write_text(json.dumps({"url": url, "sha256": digest(target)}, indent=2)+"\n")
    return target


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(rows, target, protocol, source):
    target = Path(target)
    keys = [task_key(r["task"]) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Selected split is empty or contains duplicate tasks")
    manifest = target.with_suffix(".manifest.json")
    if target.exists() or manifest.exists():
        raise ValueError(f"Split already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    manifest.write_text(json.dumps({"count": len(rows), "sha256": digest(target), "source": str(source),
                                    "protocol": protocol, "reference_commit": REFERENCE_COMMIT}, indent=2)+"\n")


def prepare_mmlu(source, output, train_count=40, eval_count=153):
    """Sorted native subject CSVs, dev interleaving, NumPy seed 888."""
    import numpy as np
    source, output = Path(source), Path(output)
    for split, count, name in (("dev", train_count, "train"), ("val", eval_count, "eval")):
        rows = list(source_rows(source/split, "mmlu"))
        if split == "dev":
            if len(rows) % 5:
                raise ValueError("Reference MMLU dev needs five examples per subject")
            rows = [rows[i] for i in np.arange(len(rows)).reshape(-1, 5).T.ravel()]
        order = np.random.default_rng(888).permutation(len(rows))
        selected = [normalize(rows[int(i)], "mmlu", str(source)) for i in order[:count]]
        if len(selected) != count:
            raise ValueError("Requested more MMLU examples than the source contains")
        write_split(selected, output/f"mmlu-{name}.jsonl", "native sorted CSV; dev interleave; numpy.default_rng(888)", source)


def fetch_math(output, limit=None):
    """Full public MATH test export; sampled_test from the authors is unavailable."""
    output = Path(output)
    rows = []
    configs = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
               "number_theory", "prealgebra", "precalculus")
    for subset in configs:
        raw = output/f"math-{subset}-raw.jsonl"
        if not raw.exists():
            query = urlencode({"dataset": "EleutherAI/hendrycks_math", "config": subset,
                               "split": "test", "offset": 0, "length": 1})
            with urlopen("https://datasets-server.huggingface.co/rows?"+query, timeout=60) as response:
                total = json.load(response)["num_rows_total"]
            fetch_dataset("EleutherAI/hendrycks_math", subset, "test", raw, min(total, limit) if limit else total)
        rows.extend(source_rows(raw, "math"))
    return rows


def prepare_math(rows, output, train_count=40, eval_count=160, source="public MATH test"):
    import numpy as np
    total = train_count+eval_count
    if len(rows) < total:
        raise ValueError("MATH source does not contain enough examples")
    order = np.random.default_rng(888).permutation(len(rows))[:total]
    normalized = [normalize(rows[int(i)], "math", source) for i in order]
    protocol = "numpy seed 888, first training then evaluation; substitute for missing authors sampled_test"
    write_split(normalized[:train_count], Path(output)/"math-train.jsonl", protocol, source)
    write_split(normalized[train_count:], Path(output)/"math-eval.jsonl", protocol, source)


def prepare_humaneval(source, output, train_count=20, eval_count=None):
    rows = [normalize(r, "humaneval", str(source)) for r in source_rows(source, "humaneval")]
    if train_count >= len(rows):
        raise ValueError("HumanEval needs held-out rows after training")
    write_split(rows[:train_count], Path(output)/"humaneval-train.jsonl", "original ordered file; first 20 training", source)
    heldout = rows[train_count:][:eval_count] if eval_count else rows[train_count:]
    write_split(heldout, Path(output)/"humaneval-eval.jsonl", "original ordered file; remainder evaluation", source)


def acquire(output, benchmarks=("mmlu", "math", "humaneval"), eval_limit=None, math_source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if eval_limit is not None and eval_limit < 1:
        raise ValueError("eval-limit must be positive")
    for benchmark in benchmarks:
        expected = {"mmlu": (40, 153), "math": (40, 160), "humaneval": (20, 144)}[benchmark]
        paths = [output/f"{benchmark}-{s}.jsonl" for s in ("train", "eval")]
        if any(p.exists() for p in paths):
            for path, count in zip(paths, (expected[0], eval_limit or expected[1])):
                manifest = path.with_suffix(".manifest.json")
                if not path.exists() or not manifest.exists():
                    raise ValueError(f"Incomplete prepared split: {path}; choose a new output directory")
                info = json.loads(manifest.read_text())
                if info["sha256"] != digest(path) or info["count"] != count:
                    raise ValueError(f"Existing split differs from requested selection: {path}; choose a new directory")
    if "mmlu" in benchmarks and not (output/"mmlu-train.jsonl").exists():
        archive = download("https://people.eecs.berkeley.edu/~hendrycks/data.tar", output/"mmlu.tar")
        root = (output/"native-mmlu").resolve()
        # Extract regular benchmark files only, with checked paths; no links.
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                target = (root/member.name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe path in MMLU archive")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as dest:
                    dest.write(src.read())
        prepare_mmlu(root/"data", output, eval_count=eval_limit or 153)
    if "math" in benchmarks and not (output/"math-train.jsonl").exists():
        rows = list(source_rows(math_source, "math")) if math_source else fetch_math(output)
        prepare_math(rows, output, eval_count=eval_limit or 160,
                     source=str(math_source) if math_source else "EleutherAI/hendrycks_math test; all seven categories")
    if "humaneval" in benchmarks and not (output/"humaneval-train.jsonl").exists():
        source = download("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
                          output/"HumanEval.jsonl.gz")
        prepare_humaneval(source, output, eval_count=eval_limit)
    return {p.name: json.loads(p.read_text()) for p in output.glob("*-*.manifest.json")}
