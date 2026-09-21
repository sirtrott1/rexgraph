"""Finite program declarations with no executable object decoding."""
from dataclasses import fields
from fractions import Fraction
import base64
import hashlib
import json
import math

from . import ast

_NODES = {cls.__name__: cls for cls in (
    ast.Literal, ast.Parameter, ast.Reference, ast.LetBinding, ast.Call, ast.ListExpr,
    ast.Comparison, ast.MatchBinding, ast.Member, ast.Alias, ast.Query)}


def encode(value, depth=0):
    if depth > 96:
        raise ValueError("program declaration nesting is too deep")
    if type(value).__name__ in _NODES and type(value) is _NODES[type(value).__name__]:
        return {"node": type(value).__name__, "fields":
                {f.name: encode(getattr(value, f.name), depth+1) for f in fields(value)}}
    if value is None or type(value) in (bool, str):
        return {"literal": type(value).__name__, "value": value}
    if type(value) is int:
        return {"literal": "int", "value": hex(value)}
    if isinstance(value, Fraction):
        return {"literal": "rational", "value": [hex(value.numerator), hex(value.denominator)]}
    if type(value) is float and math.isfinite(value):
        return {"literal": "float", "value": value.hex()}
    if type(value) is bytes:
        return {"literal": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if type(value) in (tuple, list):
        return {"literal": type(value).__name__, "value": [encode(v, depth+1) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {"literal": "dict", "value": [[k, encode(value[k], depth+1)] for k in sorted(value)]}
    raise TypeError("program literals must be finite data; bind native values as parameters")


def decode(record, depth=0):
    if depth > 96 or not isinstance(record, dict):
        raise ValueError("invalid program declaration")
    if "node" in record:
        if set(record) != {"node", "fields"} or record["node"] not in _NODES:
            raise ValueError("unknown program syntax node")
        cls = _NODES[record["node"]]
        payload = record["fields"]
        if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("program syntax fields do not match the schema")
        return cls(**{k: decode(v, depth+1) for k, v in payload.items()})
    if set(record) != {"literal", "value"}:
        raise ValueError("invalid program literal")
    kind, value = record["literal"], record["value"]
    if kind == "NoneType" and value is None:
        return None
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "str" and type(value) is str:
        return value
    if kind == "int" and type(value) is str:
        return int(value, 16)
    if kind == "rational" and isinstance(value, list) and len(value) == 2:
        return Fraction(int(value[0], 16), int(value[1], 16))
    if kind == "float" and type(value) is str:
        result = float.fromhex(value)
        if math.isfinite(result):
            return result
    if kind == "bytes" and type(value) is str:
        return base64.b64decode(value, validate=True)
    if kind in {"tuple", "list"} and type(value) is list:
        result = [decode(v, depth+1) for v in value]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict" and type(value) is list:
        out = {}
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or pair[0] in out:
                raise ValueError("invalid or repeated program record key")
            out[pair[0]] = decode(pair[1], depth+1)
        return out
    raise ValueError("unsupported program literal")


def dumps(value):
    return json.dumps(encode(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf8")


def loads(raw):
    if not isinstance(raw, (bytes, str)):
        raise TypeError("program data must be bytes or text")
    if len(raw) > 4*1024*1024:
        raise ValueError("program declaration exceeds the size limit")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate encoded field")
            result[k] = v
        return result
    try:
        result = decode(json.loads(raw, object_pairs_hook=pairs))
        if dumps(result) != (raw.encode("utf8") if isinstance(raw, str) else raw):
            raise ValueError("program declaration is not canonical")
        return result
    except (KeyError, TypeError, RecursionError, UnicodeError, json.JSONDecodeError, ZeroDivisionError) as exc:
        raise ValueError("invalid finite program data") from exc


def digest(value):
    return hashlib.sha256(dumps(value)).hexdigest()


def operators(query):
    found = set()
    def walk(value):
        if isinstance(value, ast.Call):
            found.add(value.name)
        if type(value).__name__ in _NODES and type(value) is _NODES[type(value).__name__]:
            for f in fields(value):
                walk(getattr(value, f.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
    for expression in (*query.bindings, *query.returns, *query.matches,
                       *((query.where,) if query.where is not None else ()), *query.order):
        walk(expression)
    return tuple(sorted(found))


def implementation_identity():
    """Identify installed Python definitions and compiled native modules."""
    import importlib.util
    import importlib.metadata
    from pathlib import Path
    h = hashlib.sha256()
    for package in ("rexgraph", "rcql", "rcdb"):
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError("program implementation package is not installed")
        root = Path(spec.origin).parent
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".so", ".pyd"} or "tests" in path.parts:
                continue
            name = (package+"/"+path.relative_to(root).as_posix()).encode()
            h.update(len(name).to_bytes(8, "big")); h.update(name)
            h.update(hashlib.sha256(path.read_bytes()).digest())
    for package in ("numpy", "scipy", "torch", "rdkit"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "absent"
        h.update((package+"="+version).encode())
    return h.hexdigest()
