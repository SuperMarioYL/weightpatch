"""Fixtures: tiny BF16 safetensors checkpoints and a tiny qwen3 GGUF.

Everything is written to ``tmp_path`` byte by byte, so the suite needs no
model download and no network. Tensors use Qwen3-style names so the GGUF name
map and the reversed ``ne`` shape are exercised by the same fixtures as the
safetensors paths.
"""

from __future__ import annotations

import itertools
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import gguf
import numpy as np
import pytest

from weightpatch import Checkpoint, Patch, diff_checkpoints, write_patch

#: name -> (shape, seed). Two of these are edited by :data:`EDITS`.
DEFAULT_TENSORS: dict[str, tuple[tuple[int, ...], int]] = {
    "model.embed_tokens.weight": ((16, 8), 3),
    "model.layers.0.self_attn.q_proj.weight": ((8, 8), 5),
    "model.layers.0.mlp.up_proj.weight": ((8, 8), 7),
    "model.norm.weight": ((8,), 9),
}

#: The edited elements: q_proj rows 1-2, cols 2-5; up_proj row 0, cols 1-6.
EDITS: dict[str, dict[tuple[int, ...], int]] = {
    "model.layers.0.self_attn.q_proj.weight": {(1, 2): 0x1000, (2, 5): 0x2000},
    "model.layers.0.mlp.up_proj.weight": {(0, 1): 0x3000, (0, 6): 0x4000},
}


def bf16_words(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Raw BF16 bit patterns shaped ``shape`` (never float-decoded)."""
    return np.random.default_rng(seed).integers(0, 65536, size=shape, dtype=np.uint16)


def write_safetensors(
    path: Path, tensors: Mapping[str, tuple[str, tuple[int, ...], np.ndarray]]
) -> None:
    """Write a single-shard safetensors file from raw word arrays."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    cursor = 0
    for name, (dtype, shape, words) in tensors.items():
        raw = words.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def build_checkpoint(
    directory: Path,
    *,
    tensors: Mapping[str, tuple[tuple[int, ...], int]] | None = None,
    edits: Mapping[str, Mapping[tuple[int, ...], int]] | None = None,
    drop: Sequence[str] = (),
    extra: Mapping[str, tuple[tuple[int, ...], int]] | None = None,
) -> Path:
    """Write one BF16 checkpoint directory and return it."""
    layout = dict(DEFAULT_TENSORS if tensors is None else tensors)
    written: dict[str, tuple[str, tuple[int, ...], np.ndarray]] = {}
    for name, (shape, seed) in layout.items():
        if name in drop:
            continue
        words = bf16_words(shape, seed)
        changes = (edits or {}).get(name, {})
        if changes:
            words = words.copy()
            for index, value in changes.items():
                words[index] = value
        written[name] = ("BF16", shape, words)
    for name, (shape, seed) in (extra or {}).items():
        written[name] = ("BF16", shape, bf16_words(shape, seed))

    directory.mkdir(parents=True, exist_ok=True)
    write_safetensors(directory / "model.safetensors", written)
    return directory


def write_gguf(
    path: Path,
    *,
    arch: str,
    block_count: int,
    tensors: Mapping[str, tuple[int, ...]],
) -> Path:
    """Write a GGUF whose tensors use GGML names and logical (safetensors) shapes."""
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_block_count(block_count)
    for name, shape in tensors.items():
        seed = zlib.crc32(name.encode("utf-8"))
        writer.add_tensor(
            name,
            bf16_words(shape, seed),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def tree(directory: Path) -> dict[str, bytes]:
    """Every file of a checkpoint directory, by name."""
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


@dataclass(frozen=True)
class Stack:
    """A two-patch chain: the second patch was diffed on top of the first."""

    base: Path
    after_first: Path
    after_second: Path
    first: Patch
    second: Patch


@pytest.fixture
def make_checkpoint(tmp_path: Path):
    """Build a checkpoint directory under ``tmp_path``."""

    def build(name: str, **kwargs) -> Path:
        return build_checkpoint(tmp_path / name, **kwargs)

    return build


@pytest.fixture
def checkpoint_pair(make_checkpoint) -> tuple[Path, Path]:
    """A base directory and a copy with :data:`EDITS` applied."""
    return make_checkpoint("base"), make_checkpoint("edited", edits=EDITS)


@pytest.fixture
def patch_file(checkpoint_pair, tmp_path: Path) -> Path:
    """The ``.wpatch`` recorded from :func:`checkpoint_pair`."""
    base, edited = checkpoint_pair
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)
    return write_patch(result, tmp_path / "edit.wpatch").path


@pytest.fixture
def stack_setup(make_checkpoint, tmp_path: Path) -> Stack:
    """Two patches that edit disjoint slices of the same tensor, in order."""
    first_edit = {"model.layers.0.self_attn.q_proj.weight": {(0, 0): 0x1111, (1, 1): 0x2222}}
    second_edit = {"model.layers.0.self_attn.q_proj.weight": {(4, 0): 0x3333, (5, 1): 0x4444}}
    both_edits = {
        "model.layers.0.self_attn.q_proj.weight": {
            **first_edit["model.layers.0.self_attn.q_proj.weight"],
            **second_edit["model.layers.0.self_attn.q_proj.weight"],
        }
    }

    base = make_checkpoint("base")
    after_first = make_checkpoint("after-first", edits=first_edit)
    after_second = make_checkpoint("after-second", edits=both_edits)

    def record(left: Path, right: Path, name: str) -> Patch:
        with Checkpoint(left) as a, Checkpoint(right) as b:
            return write_patch(diff_checkpoints(a, b), tmp_path / name)

    return Stack(
        base=base,
        after_first=after_first,
        after_second=after_second,
        first=record(base, after_first, "first.wpatch"),
        second=record(after_first, after_second, "second.wpatch"),
    )


@pytest.fixture
def packed_export(tmp_path: Path) -> Path:
    """A requantized-style export: packed U32 weights next to scales and biases."""
    directory = tmp_path / "packed"
    directory.mkdir()
    packed = np.arange(16, dtype=np.uint32)
    write_safetensors(
        directory / "model.safetensors",
        {
            "model.layers.0.self_attn.q_proj.weight": ("U32", (4, 4), packed),
            "model.layers.0.self_attn.q_proj.scales": ("BF16", (4, 1), bf16_words((4, 1), 31)),
            "model.layers.0.self_attn.q_proj.biases": ("BF16", (4, 1), bf16_words((4, 1), 32)),
            "model.layers.0.mlp.up_proj.weight": ("U32", (4, 4), packed),
        },
    )
    return directory


@pytest.fixture
def make_gguf(tmp_path: Path):
    """Write GGUF files whose tensors use GGML names."""
    counter = itertools.count(1)

    def build(
        tensors: Mapping[str, tuple[int, ...]],
        *,
        arch: str = "qwen3",
        block_count: int = 1,
        name: str = "fixture",
    ) -> Path:
        path = tmp_path / f"{name}-{next(counter)}.gguf"
        return write_gguf(path, arch=arch, block_count=block_count, tensors=tensors)

    return build


@pytest.fixture
def linear_attention_patch(make_checkpoint, tmp_path: Path) -> Patch:
    """A patch whose tensor namespace a qwen35 name map cannot resolve."""
    layout = {"model.language_model.layers.0.linear_attn.in_proj.weight": ((8, 8), 41)}
    edits = {
        "model.language_model.layers.0.linear_attn.in_proj.weight": {(0, 0): 0x5151}
    }
    base = make_checkpoint("linear-base", tensors=layout)
    edited = make_checkpoint("linear-edited", tensors=layout, edits=edits)
    with Checkpoint(base) as a, Checkpoint(edited) as b:
        return write_patch(diff_checkpoints(a, b), tmp_path / "linear.wpatch")
