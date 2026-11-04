"""The editor's kernel: read weights and save edits with numpy only (no model execution).

The agent imports this module from its own scripts:

    from kernel import list_tensors, load, save_edit, drop_edit, current_edit

- weights are read straight from the base safetensors file (bfloat16 -> float32);
- save_edit() writes a full replacement tensor to edit/<name>.npy after checking the rules;
- the forward-pass tool (./fwd) applies every tensor in edit/ on top of the base model.

Rules enforced by save_edit (from the source paper's harness):
- no edits to the token embedding, the output head or the final norm;
- no edit that only multiplies a whole tensor by a constant;
- shapes must match the base tensor; values must be finite.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

BASE = os.environ["WS_BASE_SAFETENSORS"]
EDIT_DIR = os.environ.get("WS_EDIT_DIR", "edit")
FORBIDDEN = ("embed_tokens", "lm_head", "model.language_model.norm.weight")
_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}

with open(BASE, "rb") as _f:
    _n = struct.unpack("<Q", _f.read(8))[0]
    _HEADER = json.loads(_f.read(_n))
    _OFFSET = 8 + _n
_HEADER.pop("__metadata__", None)


def list_tensors(prefix: str = "model.language_model.") -> dict[str, tuple[list[int], str]]:
    """Return {name: (shape, dtype)} for the text-model tensors."""
    return {k: (v["shape"], v["dtype"]) for k, v in _HEADER.items() if k.startswith(prefix)}


def load(name: str, edited: bool = False) -> np.ndarray:
    """Load one tensor as float32. edited=True returns the current edit if one is saved."""
    path = os.path.join(EDIT_DIR, f"{name}.npy")
    if edited and os.path.exists(path):
        return np.load(path)
    meta = _HEADER[name]
    start, end = meta["data_offsets"]
    with open(BASE, "rb") as f:
        f.seek(_OFFSET + start)
        raw = np.frombuffer(f.read(end - start), dtype=_DTYPES[meta["dtype"]])
    if meta["dtype"] == "BF16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32).reshape(meta["shape"])


def save_edit(name: str, tensor: np.ndarray) -> str:
    """Save a replacement for `name` after checking the rules. Returns the saved path."""
    if name not in _HEADER:
        raise KeyError(f"unknown tensor {name}")
    if any(f in name for f in FORBIDDEN):
        raise PermissionError(f"edits to {name} are not allowed")
    base = load(name)
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.shape != base.shape:
        raise ValueError(f"shape {tensor.shape} != {base.shape}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("non-finite values")
    nz = np.abs(base) > 1e-12
    if nz.any():
        ratio = tensor[nz] / base[nz]
        if np.allclose(ratio, ratio.flat[0], rtol=1e-4) and np.allclose(tensor[~nz], 0):
            raise PermissionError("whole-tensor scaling by a constant is not allowed")
    os.makedirs(EDIT_DIR, exist_ok=True)
    path = os.path.join(EDIT_DIR, f"{name}.npy")
    np.save(path, tensor)
    return path


def drop_edit(name: str) -> None:
    """Remove the saved replacement for `name` (the base tensor is used again)."""
    path = os.path.join(EDIT_DIR, f"{name}.npy")
    if os.path.exists(path):
        os.remove(path)


def current_edit() -> list[str]:
    """Names of the tensors currently replaced in edit/."""
    if not os.path.isdir(EDIT_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))
