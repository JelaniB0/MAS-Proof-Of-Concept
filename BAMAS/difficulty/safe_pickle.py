"""Data-only reader for RouterBench's pandas pickle export.

Never imports pickle, resolves globals, invokes reducers, or constructs serialized
classes. Object opcodes become inert syntax nodes. Only the known pandas table
layout is then interpreted as primitive columns. This is deliberately NOT a
general pickle loader. Unsupported layouts/opcodes fail closed.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickletools

import numpy as np
import pandas as pd

ZERO_SHOT_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"


@dataclass
class Node:
    name: str
    args: tuple = ()
    state: object = None


def parse_data(path, max_text_characters=None):
    """Parse a restricted opcode grammar; globals are just strings in Nodes."""
    stack, memo = [], {}
    marker = object()
    if max_text_characters is not None and max_text_characters < 256:
        raise ValueError("Inspection text limit must be at least 256 characters")

    def marked():
        index = len(stack) - 1
        while index >= 0 and stack[index] is not marker:
            index -= 1
        if index < 0:
            raise ValueError("Missing MARK")
        items = stack[index + 1:]
        del stack[index:]
        return items

    with Path(path).open("rb") as stream:
        for op, arg, _ in pickletools.genops(stream):
            name = op.name
            if name in {"PROTO", "FRAME"}:
                continue
            if name == "MARK":
                stack.append(marker)
            elif name == "NONE":
                stack.append(None)
            elif name in {"NEWTRUE", "NEWFALSE"}:
                stack.append(name == "NEWTRUE")
            elif name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "BINFLOAT",
                          "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINBYTES",
                          "BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
                if max_text_characters is not None and isinstance(arg, str) and len(arg) > max_text_characters:
                    arg = "__OMITTED_LONG_TEXT_SHA256__:" + hashlib.sha256(arg.encode("utf-8")).hexdigest()
                stack.append(arg)
            elif name in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE"}:
                stack.append({"EMPTY_LIST": list, "EMPTY_DICT": dict, "EMPTY_TUPLE": tuple}[name]())
            elif name == "MEMOIZE":
                memo[len(memo)] = stack[-1]
            elif name in {"BINPUT", "LONG_BINPUT"}:
                memo[arg] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET"}:
                stack.append(memo[arg])
            elif name == "TUPLE":
                stack.append(tuple(marked()))
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                count = int(name[-1])
                values = tuple(stack[-count:])
                del stack[-count:]
                stack.append(values)
            elif name == "APPEND":
                value = stack.pop()
                stack[-1].append(value)
            elif name == "APPENDS":
                values = marked()
                stack[-1].extend(values)
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                stack[-1][key] = value
            elif name == "SETITEMS":
                values = marked()
                stack[-1].update(zip(values[::2], values[1::2]))
            elif name == "STACK_GLOBAL":
                attribute, module = stack.pop(), stack.pop()
                if not isinstance(attribute, str) or not isinstance(module, str):
                    raise ValueError("Invalid global name")
                stack.append(Node(module + "." + attribute))
            elif name in {"REDUCE", "NEWOBJ"}:
                args, symbol = stack.pop(), stack.pop()
                if not isinstance(symbol, Node) or not isinstance(args, tuple):
                    raise ValueError("Invalid object syntax")
                stack.append(Node(symbol.name, args))
            elif name == "BUILD":
                state = stack.pop()
                if not isinstance(stack[-1], Node):
                    raise ValueError("Invalid BUILD target")
                stack[-1].state = state
            elif name == "STOP":
                if len(stack) != 1:
                    raise ValueError("Unexpected pickle stack")
                return stack[0]
            else:
                raise ValueError(f"Unsupported data opcode: {name}")
    raise ValueError("Missing STOP")


def array_data(node):
    if not isinstance(node, Node):
        raise ValueError("Expected an array syntax node")
    if node.name in {"numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"}:
        version, shape, dtype, fortran, values = node.state
        if version != 1:
            raise ValueError("Unsupported array version")
    elif node.name in {"numpy.core.numeric._frombuffer", "numpy._core.numeric._frombuffer"}:
        values, dtype, shape, order = node.args
        fortran = order == "F"
    else:
        raise ValueError(f"Unsupported array representation: {node.name}")
    if dtype.name != "numpy.dtype" or not isinstance(dtype.args[0], str):
        raise ValueError("Expected primitive NumPy dtype")
    code = dtype.args[0]
    if code in {"O8", "O4", "O"}:
        if not isinstance(values, list) or any(not isinstance(v, (str, int, float, bool, type(None))) for v in values):
            raise ValueError("Only primitive object-array values are accepted")
        result = np.asarray(values, dtype=object)
    else:
        # Explicit primitive whitelist; no structured or arbitrary object dtypes.
        if code not in {"f8", "f4", "i8", "i4", "i2", "i1", "u8", "u4", "u2", "u1", "b1", "?"}:
            raise ValueError(f"Unsupported numeric dtype: {code}")
        byteorder = dtype.state[1] if dtype.state else "="
        result = np.frombuffer(values, dtype=np.dtype(code).newbyteorder(byteorder))
    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid array shape")
    return result.reshape(shape, order="F" if fortran else "C")


def read_routerbench(path, max_text_characters=None):
    """Extract primitive DataFrame blocks without running pickle reconstruction."""
    table = parse_data(path, max_text_characters=max_text_characters)
    if table.name not in {"pandas.core.frame.DataFrame", "pandas.DataFrame"} or not isinstance(table.state, dict):
        raise ValueError("Expected RouterBench DataFrame layout")
    manager = table.state["_mgr"]
    if manager.name != "pandas.core.internals.managers.BlockManager":
        raise ValueError("Expected BlockManager layout")
    blocks, axes = manager.args
    columns = array_data(axes[0].args[1]["data"]).tolist()
    if any(not isinstance(c, str) for c in columns) or len(columns) != len(set(columns)):
        raise ValueError("Invalid column names")
    extracted = {}
    for block in blocks:
        if block.name != "pandas._libs.internals._unpickle_block":
            raise ValueError("Unsupported block representation")
        values, locations, ndim = block.args
        if ndim != 2 or not isinstance(locations, Node):
            raise ValueError("Unsupported block placement")
        if locations.name == "builtins.slice":
            indices = list(range(len(columns)))[slice(*locations.args)]
        else:
            placement = array_data(locations)
            if placement.ndim != 1 or placement.dtype.kind not in "iu":
                raise ValueError("Block positions must be a one-dimensional integer array")
            indices = placement.tolist()
            if len(indices) != len(set(indices)) or any(i < 0 or i >= len(columns) for i in indices):
                raise ValueError("Block positions are duplicated or outside the column range")
        values = array_data(values)
        if values.ndim != 2 or values.shape[0] != len(indices):
            raise ValueError("Invalid block dimensions")
        for index, column_values in zip(indices, values):
            if columns[index] in extracted:
                raise ValueError("Overlapping blocks")
            extracted[columns[index]] = column_values
    if set(extracted) != set(columns):
        raise ValueError("Missing blocks")
    result = pd.DataFrame(extracted, columns=columns)
    result.attrs["inspection_text_limit"] = max_text_characters
    return result


def convert(source, destination):
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    frame = read_routerbench(source)
    # Preserve only inference-time text, identities, and per-model quality scores.
    keep = [c for c in frame if "|" not in c and c not in {
        "embedding", "oracle", "oracle_model_to_route_to", "no_model_correct",
        "gold_response", "true_label", "model_response"}]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_json(destination, orient="records", lines=True, force_ascii=False)
    return {"source_sha256": digest, "rows": len(frame), "columns": keep}
