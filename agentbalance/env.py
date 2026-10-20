"""Load a small, literal subset of dotenv syntax without an extra dependency."""

import os
from pathlib import Path
import re


def load_env(path=".env", *, required=False):
    """Load missing process variables. Existing variables (even empty) win.

    Supports KEY=value, optional export, single/double quotes, and comments.
    Values are literal: no interpolation, escape processing, or shell execution.
    Parse the whole file before changing the environment; errors never echo values.
    """
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        if required:
            raise ValueError(f"Environment file not found: {path}") from None
        return
    except UnicodeError:
        raise ValueError(f"Environment file must be UTF-8: {path}") from None
    values = {}
    for line_number, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assignment = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        error = f"Invalid environment file syntax at {path}:{line_number}; use KEY=value on one line"
        if not assignment or "\x00" in raw:
            raise ValueError(error)
        key, value = assignment.groups()
        if value.startswith(("'", '"')):
            quoted = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", value)
            if not quoted:
                raise ValueError(error)
            value = quoted.group(2)
        else:
            value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)
