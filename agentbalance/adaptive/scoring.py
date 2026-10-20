"""Explicit benchmark scorers; generated Python runs only in an opted-in container."""

from fractions import Fraction
import ast
import json
import re
import subprocess
from uuid import uuid4


def boxed(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def math_normalize(text):
    text = (boxed(text) or text).strip().strip("$")
    for token in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\displaystyle"):
        text = text.replace(token, "")
    text = re.sub(r"\s+", "", text).replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    fraction = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    try:
        if fraction:
            return str(Fraction(int(fraction[1]), int(fraction[2])))
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
            return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    return text


# Runs inside a disposable Docker container, never the host interpreter.
CODE_HARNESS = """
import contextlib, io, json, sys
row = json.load(sys.stdin)
namespace = {}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(row['program'], namespace)
        exec(row['test'], namespace)
        namespace['check'](namespace[row['entry_point']])
except BaseException:
    sys.exit(1)
sys.stdout.write('AGENTBALANCE_TESTS_PASSED')
"""


class Scorer:
    def __init__(self, code_image=None, code_timeout=10, protocol="local"):
        self.code_image, self.code_timeout = code_image, code_timeout
        if protocol not in {"local", "paper"}:
            raise ValueError("Unknown scoring protocol")
        self.protocol = protocol

    def preflight(self, records):
        if not any(r.get("scorer") == "humaneval" for r in records):
            return
        if not self.code_image:
            raise ValueError("HumanEval needs functional tests, not exact match. Supply --code-grader-image for an isolated Docker grader before model calls.")
        if self.code_image.startswith("-") or not self.code_image.strip():
            raise ValueError("Invalid Docker image name")
        try:
            result = subprocess.run(["docker", "image", "inspect", self.code_image], capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("Docker grader unavailable; start Docker and pull the chosen image first") from exc
        if result.returncode:
            raise ValueError("Docker grader image is unavailable locally; pull the chosen Python image first")

    def score(self, answer, row):
        answer = answer or ""
        kind = row.get("scorer", "exact")
        if kind == "exact":
            from .data import exact_match
            return exact_match(answer, row["answers"])
        if kind == "mmlu":
            if self.protocol == "paper":
                text = answer.strip()
                if "answer is" in text:
                    text = text.split("answer is", 1)[1].strip(": ").removeprefix("Option").strip()
                return float(bool(text) and text[0] in row["answers"])
            lines = answer.strip().splitlines()
            first = lines[0].strip() if lines else ""
            match = re.fullmatch(r"(?:\*\*)?\(?([A-D])\)?[.)]?(?:\*\*)?", first)
            return float(bool(match) and match[1] in row["answers"])
        if kind == "math":
            if self.protocol == "paper":
                from ..paper_scoring import normalize_math, extract_math
                return float(normalize_math(extract_math(answer)) in {normalize_math(a) for a in row["answers"]})
            return float(math_normalize(answer) in {math_normalize(a) for a in row["answers"]})
        if kind != "humaneval":
            raise ValueError(f"Unknown scorer: {kind}")
        if not self.code_image:
            raise ValueError("HumanEval requires --code-grader-image")
        match = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
        completion = match[1] if match else answer
        full_function = False
        try:
            full_function = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == row["entry_point"]
                                for n in ast.parse(completion).body)
        except SyntaxError:
            pass
        payload = {"program": completion if full_function else row["code_prompt"]+completion,
                   "test": row["test"], "entry_point": row["entry_point"]}
        name = "agentbalance-grade-" + uuid4().hex
        command = ["docker", "run", "--rm", "--pull", "never", "--name", name, "--network", "none",
                   "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                   "--pids-limit", "64", "--memory", "256m", "--cpus", "1", "--user", "65534:65534",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "-i", self.code_image, "python", "-I", "-c", CODE_HARNESS]
        try:
            result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=self.code_timeout)
            if result.returncode in {125, 126, 127}:
                raise ValueError("Docker grader failed to start; training stopped without treating this as model error")
            return float(result.returncode == 0 and result.stdout == "AGENTBALANCE_TESTS_PASSED")
        except subprocess.TimeoutExpired:
            return 0.0
        finally:
            # --rm handles normal completion; force removal also stops timed-out containers.
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError(f"Could not confirm Docker grader cleanup: {name}")
