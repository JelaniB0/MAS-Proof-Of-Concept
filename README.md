# Difficulty predictor

This branch contains the standalone difficulty predictor, its prepared datasets,
and the selected trained checkpoints. It excludes the BAMAS executor, LinUCB
implementations, Azure routing application, API credentials, generated cloud
responses, and large intermediate training caches.

Start with the [readable difficulty/LinUCB guide](BAMAS/difficulty/DIFFICULTY_AND_LINUCB_README.md)
for score interpretation, training, and the proposed routing design. Its
retrieval and bandit sections describe integration work in the broader workspace;
this branch runs the trained neural predictor independently.

## What is included?

| Component | Location |
| --- | --- |
| Predictor, training/evaluation code, and offline tests | `BAMAS/difficulty/` |
| Main prepared train/validation/test splits | `BAMAS/difficulty/runs/all_routerbench_v1/prepared/main/` |
| Separate archived RAG splits | `BAMAS/difficulty/runs/all_routerbench_v1/prepared/rag/` |
| Selected main checkpoint | `BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model/` |
| Selected separate RAG checkpoint | `BAMAS/difficulty/runs/all_routerbench_v1/rag/selected_model/` |
| Interactive terminal entry point | `BAMAS/scripts/try_difficulty.py` |
| Training reports, source manifests, and label audits | `BAMAS/difficulty/reports/` |
| Dataset scope and provenance | [DATASETS.md](DATASETS.md) |
| Release file hashes | [RELEASE_FILES.json](RELEASE_FILES.json) |

The main predictor uses frozen MPNet with an ensemble of three trained MLP heads.
Each head predicts eleven historical-model quality deficits; fixed groups yield
weak, intermediate, and strong difficulty scores. The RAG predictor uses a
different five-model roster and must remain separate.

The main corpus has **25,710 training, 5,406 validation, and 5,381 test rows**.
The separate RAG corpus has **445 training, 81 validation, and 131 test rows**.
These are prepared question records, with outcomes masked where unavailable.

## Download the branch and its large files

Install [Git LFS](https://git-lfs.com/) first. Dataset JSONL files, encoder weights,
prediction heads, and support arrays are stored in LFS. Both checkpoints reference
the same encoder content, so LFS stores that shared content once.

```powershell
git lfs install
git clone --branch difficulty-predictor --single-branch https://github.com/JelaniB0/MAS-Proof-Of-Concept.git
cd MAS-Proof-Of-Concept
git lfs pull
```

A downloaded source ZIP may contain LFS pointers rather than the actual assets,
depending on repository settings. Use the clone/pull commands for a working copy.
GitHub's file viewer may also show pointers for the large files.

## Install dependencies

Use Python 3.11 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r BAMAS/difficulty/requirements.txt
```

The included weights allow local inference without Azure/OpenAI API keys or paid
model calls. First startup loads the ML libraries, encoder, and support arrays.

## Ask questions interactively

```powershell
.\.venv\Scripts\python.exe BAMAS/scripts/try_difficulty.py
```

Wait for `Question>`, enter a question, and press Enter. The predictor loads once
per session. Type `/quit` to stop. Out-of-support inputs are flagged and their
scores withheld by default; inspect raw experimental scores with
`--show-unsupported` when needed. Similarity support is not prediction confidence.

Use the separate RAG cohort with `--cohort rag`.

For one prediction with detailed fields:

```powershell
.\.venv\Scripts\python.exe -m BAMAS.difficulty predict --model BAMAS/difficulty/runs/all_routerbench_v1/main/selected_model --text "Why does increasing temperature shift a chemical equilibrium?" --details
```

These commands return neural difficulty estimates. They do not run an answering
LLM or add the Azure-neighbor blend from the separate routing experiment.

## Verify and reproduce

```powershell
git lfs fsck
.\.venv\Scripts\python.exe -m unittest discover -s BAMAS/difficulty -t . -p "test*.py" -v
```

The tests check software behavior; they do not establish accuracy on current
cloud models. Keep validation and test labels out of training/reference fitting.
For source preparation and training procedures, see the
[technical README](BAMAS/difficulty/README.md#reproduce-the-current-study).
Raw export archives and additional external evaluation sets can be obtained using
the pinned download code; they are not additional training questions in this release.

Scores describe historical model pools and are not calibrated success
probabilities for current deployments. The documented 0.40/0.60 LinUCB preference
bands are a proposed policy, not runtime behavior in this branch.

Project code retains the accompanying [license](LICENSE). Dataset and pretrained
model assets retain their upstream terms; see [DATASETS.md](DATASETS.md).
