"""Read literal prompt data from a pinned source checkout, without executing it."""

import ast
from collections import Counter
from pathlib import Path
import re


def literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left)+literal(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(n) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k,v in zip(node.keys, node.values)}
    raise ValueError("Reference prompt contains unsupported dynamic data")


def reference_templates(root, benchmark):
    file = {"mmlu": "mmlu_prompt_set.py", "math": "gsm8k_prompt_set.py", "humaneval": "humaneval_prompt_set.py"}[benchmark]
    tree = ast.parse((Path(root)/"AgentBalance/prompt"/file).read_text(encoding="utf-8"))
    descriptions, roles, methods = None, None, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            name = ast.unparse(node.targets[0])
            if name == "ROLE_DESCRIPTION":
                descriptions = literal(node.value)
            if name == "roles" and isinstance(node.value, ast.Call):
                roles = literal(node.value.args[0])
        elif isinstance(node, ast.ClassDef):
            for function in node.body:
                if isinstance(function, ast.FunctionDef) and function.name in {
                        "get_decision_constraint", "get_decision_role", "get_decision_few_shot"}:
                    returns = [n for n in function.body if isinstance(n, ast.Return)]
                    if returns:
                        methods[function.name] = literal(returns[-1].value)
    if not descriptions or not roles:
        raise ValueError("Reference does not contain expected prompt dictionaries")
    templates, count = {}, Counter()
    for i in range(5):
        role = roles[i % len(roles)]
        identifier = re.sub(r"[^a-z0-9]+", "_", role.casefold()).strip("_")
        count[identifier] += 1
        key = identifier if count[identifier] == 1 else f"{identifier}_{count[identifier]}"
        templates[key] = descriptions[role]
    templates["synthesizer"] = "\n".join(methods.get(n, "") for n in (
        "get_decision_role", "get_decision_constraint", "get_decision_few_shot"))
    return templates


def reference_prefixes(root, benchmark, templates):
    if benchmark != "math":
        return {}
    tree = ast.parse((Path(root)/"AgentBalance/prompt/gsm8k_prompt_set.py").read_text(encoding="utf-8"))
    data = next(literal(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "FEW_SHOT_DATA")
    result = {}
    for key in templates:
        label = re.sub(r"_\d+$", "", key).replace("_", " ").casefold()
        for role, examples in data.items():
            if role.casefold() == label:
                result[key] = examples
    return result
