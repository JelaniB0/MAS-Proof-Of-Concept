# Integrating the predictor into another branch

The component is the `difficulty-predictor/` folder. It includes its source,
datasets, checkpoints, documentation, license, ignore rules, and scoped Git LFS
attributes. Its Python package is `difficulty`; a folder name with a hyphen is
not the Python import name.

## Recommended handoff: import the folder

This release branch started with an independent root commit. It has no common
ancestor with the original `bamas-code` branch. Importing the folder avoids
joining those histories or merging their root README/license/configuration files.

On the receiving branch, start with a clean working tree and confirm that
`difficulty-predictor/` does not already exist. If it exists, review its contents
first: `git restore` replaces the selected paths.

From the receiving repository root, assuming `origin` points to the shared repo:

```powershell
git fetch origin difficulty-predictor
git lfs install
git lfs fetch origin difficulty-predictor
git restore --source=origin/difficulty-predictor --staged --worktree -- difficulty-predictor
git lfs checkout difficulty-predictor
git diff --cached --stat
git commit -m "Add standalone difficulty predictor"
```

This adds a normal commit to the receiving branch. It brings in the predictor
folder and leaves paths outside that folder unchanged. It does not create a
Git merge relationship with the standalone release branch. Review later updates
before importing again, especially if the receiving branch edits this folder.

Git LFS must be installed: the component `.gitattributes` records which files
are datasets or weights. Without the LFS objects, pointer files cannot be used
for prediction. If the branch is in a different repository, configure a remote
for the source repository and replace `origin` in the fetch/source commands.

Install and run from the component folder:

```powershell
cd difficulty-predictor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r difficulty/requirements.txt
.\.venv\Scripts\python.exe scripts/try_difficulty.py
```

For Python imports, run from this component folder or add its absolute path to
the Python module search path in your application, then use:

```python
from difficulty.model import DifficultyPredictor
```

## If you need an actual history merge

Git refuses an ordinary merge of unrelated histories. An explicit
`git merge origin/difficulty-predictor --allow-unrelated-histories` permits it,
but same-path additions such as `README.md`, `.gitignore`, `.gitattributes`, and
`LICENSE` may need conflict resolution. A new component directory usually adds
cleanly when the receiving branch has no files at those paths; no conflict-free
guarantee is possible without inspecting that branch.

A file existing only in the receiving branch is not automatically deleted just
because it is absent from this independently rooted release. Deletions recorded
relative to a shared ancestor can matter in branches that do share history.

See the official [Git merge documentation](https://git-scm.com/docs/git-merge)
and [Git restore documentation](https://git-scm.com/docs/git-restore).
