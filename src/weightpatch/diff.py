"""Tensor-by-tensor sparse diff between two safetensors checkpoints.

Each common tensor is compared as raw words (``uint16`` for BF16, opaque
``uint8`` bytes otherwise), so no float decoding is needed and BF16 -- which
numpy cannot represent -- compares exactly. Unchanged tensors are dropped; a
changed tensor records both full-tensor digests, the ``[dim, start, end]``
ranges of the smallest slab covering every changed element, and the raw
before/after values of that slab.

The emitted manifest follows the plan. Its ``id`` is derived by the writer
(``patch.py``) from ``(base, parents, edit)``; nothing here mints an id.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .checkpoint import Checkpoint

VALUES_FILENAME = "values.npz"

#: Called as ``progress(name, index, total)`` before each common tensor.
ProgressCallback = Callable[[str, int, int], None]


class DiffError(Exception):
    """Two checkpoints cannot be expressed as a slice patch."""


def values_key(name: str, side: str) -> str:
    """The ``values.npz`` array key holding one side of one changed tensor."""
    if side not in ("before", "after"):
        raise ValueError(f"side must be 'before' or 'after', got {side!r}")
    return f"{name}.{side}"


@dataclass
class TensorDiff:
    """One changed tensor: digests, changed slab, and the slab's raw values."""

    name: str
    dtype: str
    element_size: int
    shape: tuple[int, ...]
    ranges: list[list[int]]
    before_sha256: str
    after_sha256: str
    before: np.ndarray
    after: np.ndarray

    def manifest_entry(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "element_size": self.element_size,
            "shape": list(self.shape),
            "ranges": [list(span) for span in self.ranges],
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "ref": {
                "values": VALUES_FILENAME,
                "before": values_key(self.name, "before"),
                "after": values_key(self.name, "after"),
            },
        }


@dataclass
class CheckpointDiff:
    """The sparse difference between a base checkpoint and an edited one."""

    base: dict
    tensors: list[TensorDiff] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def manifest(
        self,
        *,
        parents: Sequence[str] = (),
        probe: dict | None = None,
    ) -> dict:
        """Render the plan's manifest document (without its derived ``id``)."""
        return {
            "base": self.base,
            "parents": list(parents),
            "edit": {
                "tensors": [tensor.manifest_entry() for tensor in self.tensors],
                "removed": list(self.removed),
                "added": list(self.added),
            },
            "probe": probe
            if probe is not None
            else {"kind": "declared", "claim": "", "metrics": []},
        }


def diff_checkpoints(
    base: Checkpoint,
    target: Checkpoint,
    *,
    progress: ProgressCallback | None = None,
) -> CheckpointDiff:
    """Diff ``target`` against ``base`` and return only the changed tensors."""
    base_names = set(base.tensors)
    target_names = set(target.tensors)

    common = sorted(base_names & target_names)
    changed: list[TensorDiff] = []
    for index, name in enumerate(common):
        if progress is not None:
            progress(name, index, len(common))
        tensor_diff = _diff_tensor(base, target, name)
        if tensor_diff is not None:
            changed.append(tensor_diff)

    return CheckpointDiff(
        base=base.describe(),
        tensors=changed,
        added=sorted(target_names - base_names),
        removed=sorted(base_names - target_names),
    )


def _diff_tensor(base: Checkpoint, target: Checkpoint, name: str) -> TensorDiff | None:
    before_info = base.tensor(name)
    after_info = target.tensor(name)
    if before_info.dtype != after_info.dtype or before_info.shape != after_info.shape:
        raise DiffError(
            f"tensor {name!r} is not slice-comparable: "
            f"{before_info.dtype}{before_info.shape} -> "
            f"{after_info.dtype}{after_info.shape}; a patch carries slices, not reshaped tensors"
        )

    before_digest = hashlib.sha256()
    after_digest = hashlib.sha256()
    first_changed = -1
    last_changed = -1
    position = 0
    for before_block, after_block in zip(
        base.iter_blocks(name), target.iter_blocks(name)
    ):
        before_digest.update(before_block.tobytes())
        after_digest.update(after_block.tobytes())
        differs = before_block != after_block
        if differs.ndim > 1:
            differs = differs.any(axis=-1)
        if differs.any():
            indices = np.flatnonzero(differs)
            if first_changed < 0:
                first_changed = position + int(indices[0])
            last_changed = position + int(indices[-1]) + 1
        position += differs.shape[0]

    if first_changed < 0:
        return None

    ranges = _bounding_ranges(before_info.shape, first_changed, last_changed)
    return TensorDiff(
        name=name,
        dtype=before_info.dtype,
        element_size=before_info.element_size,
        shape=before_info.shape,
        ranges=ranges,
        before_sha256=before_digest.hexdigest(),
        after_sha256=after_digest.hexdigest(),
        before=base.read_slab(name, ranges),
        after=target.read_slab(name, ranges),
    )


def _bounding_ranges(
    shape: Sequence[int], start: int, stop: int
) -> list[list[int]]:
    """Smallest ``[dim, start, end]`` slab covering elements ``[start, stop)``.

    The slab is a superset: elements of the flat run can be scattered once a
    dimension wraps, and a wrapped dimension is covered in full. Storing both
    sides' raw values for the slab keeps apply and revert exact either way.
    """
    if not shape:
        return []
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    last = stop - 1
    ranges: list[list[int]] = []
    for dim, (size, stride) in enumerate(zip(shape, strides)):
        first_row = start // stride
        last_row = last // stride
        if last_row - first_row >= size - 1:
            ranges.append([dim, 0, size])
            continue
        first_coordinate = first_row % size
        last_coordinate = last_row % size
        if last_coordinate < first_coordinate:
            ranges.append([dim, 0, size])
        else:
            ranges.append([dim, first_coordinate, last_coordinate + 1])
    return ranges
