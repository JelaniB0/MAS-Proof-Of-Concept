"""Atomic artifact writes with bounded retries for Windows file locks.

Readers, scanners and sync clients can briefly deny replacement of a file.
Retry only the filesystem operation, never model requests. On persistent failure,
keep the old target and the new temporary snapshot for inspection/recovery.
"""

import json
from pathlib import Path
import time


def retry_windows_file_operation(operation):
    for attempt in range(9):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 8:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def replace_file(temporary, target):
    temporary, target = Path(temporary), Path(target)
    return retry_windows_file_operation(lambda: temporary.replace(target))


def write_json(path, data):
    path = Path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    retry_windows_file_operation(lambda: temporary.write_text(payload, encoding="utf-8"))
    replace_file(temporary, path)
