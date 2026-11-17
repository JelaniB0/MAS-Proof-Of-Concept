# Included datasets and pretrained assets

This release includes the exact prepared data used by the main and separate RAG
difficulty-prediction studies. The original files are preserved byte for byte
and stored in Git LFS. `RELEASE_FILES.json` lists their SHA256 hashes and sizes.

## Prepared main corpus

Location: `BAMAS/difficulty/runs/all_routerbench_v1/prepared/main/`.

| Split | Rows |
| --- | ---: |
| Training | 25,710 |
| Validation | 5,406 |
| Test | 5,381 |
| Total | 36,497 |

The reporting families are GSM8K, MBPP, MMLU, ARC-Challenge, HellaSwag,
Winogrande, MT-Bench, and additional released evaluations. The files contain
normalized task text, observed historical model scores, derived pool difficulty
labels where available, identifiers, and preparation provenance.

Missing outcomes are not failures. A recorded pool target requires outcomes for
all fixed members of that pool; the per-model training head masks individual
missing outcomes. Test data is included for evaluation, not training.

## Separate archived RAG corpus

Location: `BAMAS/difficulty/runs/all_routerbench_v1/prepared/rag/`.

| Split | Rows |
| --- | ---: |
| Training | 445 |
| Validation | 81 |
| Test | 131 |
| Total | 657 |

This corpus covers news and Wikipedia retrieval tasks with its own five-model
roster. Its checkpoint/pool contract is not interchangeable with the main model.

## GSM8K reference answers

The source reference files are included as `BAMAS/data/gsm8k/train.jsonl` and
`BAMAS/data/gsm8k/test.jsonl`. They support reviewed numeric-answer regrading and
external GSM8K evaluation. They are reference assets, not additional independent
questions added to the prepared training split above. Their hashes are also in
the release inventory; the archived regrading protocol records the training
reference checksum used by the study.

## Sources, transformations, and licenses

The historical records come from the
[official RouterBench exports](https://huggingface.co/datasets/withmartian/routerbench/tree/main)
and the archived RAG export at
[revision 784e39ca67a6c6291685b4862a05dd95fb0b40da](https://huggingface.co/datasets/withmartian/routerbench/tree/784e39ca67a6c6291685b4862a05dd95fb0b40da).
The [RouterBench paper](https://arxiv.org/abs/2403.12031) describes the source benchmark.
Exact export checksums, model rosters, conditions, and preparation decisions are
recorded in each prepared `manifest.json`, the saved protocols, and
`BAMAS/difficulty/reports/all_routerbench_2026-10-24/REPORT.md`.

GSM8K outcomes were regraded with the reviewed final-numeric-answer procedure;
other tasks retain published binary/graded scores. The label-review evidence is
under `BAMAS/difficulty/reports/label_audit/`. The raw long-form export is not a
second independent question corpus and is not included here.

The pretrained encoder is
[`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2),
pinned to revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. The selected
checkpoints include local encoder/tokenizer files, trained prediction heads,
and input-support arrays; their fingerprints are recorded in `model.json`.

Dataset content and pretrained weights remain subject to their respective
upstream licenses and attribution requirements. The project's code license does
not relicense the underlying benchmarks or pretrained assets. Consult the
upstream dataset/model records for those terms.

## Scope

This branch excludes raw pickle archives, redundant encoder caches, candidate
training checkpoints, unrelated BAMAS datasets, API keys, and live routing logs.
External GSM8K/MATH/HumanEval evaluation downloads are not part of the training
corpus and can be fetched with the included evaluation download code. The
prepared main/RAG splits and selected checkpoints are included directly so local
prediction does not depend on downloading those raw archives.
