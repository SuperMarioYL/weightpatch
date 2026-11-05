#!/usr/bin/env python3
"""Build the demo's edited checkpoint from a base safetensors directory.

Copies ``BASE`` to ``OUT`` and rewrites a 64-row slice of two BF16 tensors by
flipping one bit in every raw ``uint16`` word. No float decoding, no torch, no
randomness: the same base bytes always produce the same edit, so the patch id
recorded in ``docs/demo-results.json`` can be reproduced byte for byte.

Usage::

    python examples/make_demo_edit.py ./Qwen3-0.6B ./Qwen3-0.6B-neurons

Any checkpoint with these two Qwen3-0.6B tensors works; the shapes are read
from the file, not assumed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from weightpatch import Checkpoint

#: Rows to rewrite, counted from row 0 of each tensor.
SLICE_ROWS = 64

#: One bit in the raw BF16 word, applied to every element of the slice.
FLIP_MASK = np.uint16(0x0080)

TARGETS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.12.mlp.down_proj.weight",
)


def edit_tensor(checkpoint: Checkpoint, name: str) -> None:
    """Flip one bit across the first ``SLICE_ROWS`` rows of ``name``."""
    info = checkpoint.tensor(name)
    if info.dtype != "BF16" or len(info.shape) != 2:
        raise SystemExit(
            f"{name}: expected a 2-D BF16 tensor, got {info.dtype}{info.shape}"
        )
    if info.shape[0] < SLICE_ROWS:
        raise SystemExit(
            f"{name}: has {info.shape[0]} rows, fewer than the {SLICE_ROWS} to edit"
        )

    # Rows 0..SLICE_ROWS-1 are contiguous in the shard, so one seek and one
    # write cover the slab; read_slab returns them already in row-major order.
    slab = checkpoint.read_slab(name, [[0, 0, SLICE_ROWS], [1, 0, info.shape[1]]])
    slab = slab.copy()
    slab ^= FLIP_MASK
    with info.file.open("r+b") as handle:
        handle.seek(info.offset)
        handle.write(slab.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="Base safetensors checkpoint directory")
    parser.add_argument("output", type=Path, help="Edited copy to create (must not exist)")
    args = parser.parse_args(argv)

    if args.output.exists():
        raise SystemExit(f"{args.output}: already exists; remove it or pick another name")

    shutil.copytree(args.base, args.output)
    with Checkpoint(args.output) as checkpoint:
        for name in TARGETS:
            edit_tensor(checkpoint, name)

    with Checkpoint(args.output) as checkpoint:
        print(f"{args.output}: {len(checkpoint.tensors)} tensors")
        for name in TARGETS:
            info = checkpoint.tensor(name)
            shape = "×".join(str(dim) for dim in info.shape)
            print(
                f"  {name}  {info.dtype} {shape}  "
                f"rows 0:{SLICE_ROWS}  sha256 {checkpoint.tensor_sha256(name)[:16]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
