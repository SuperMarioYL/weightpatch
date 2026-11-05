"""weightpatch — sparse, reviewable patch files for edited open-weight checkpoints.

The public surface is deliberately small: open a checkpoint, diff two of them,
read or write a ``.wpatch``, apply or revert it, and screen a derivative
artifact for the recorded edits.
"""

from .checkpoint import Checkpoint, CheckpointError, TensorInfo
from .diff import CheckpointDiff, DiffError, TensorDiff, diff_checkpoints
from .patch import (
    Patch,
    PatchError,
    TensorEdit,
    apply_patch,
    read_patch,
    revert_patch,
    write_patch,
)
from .screen import ScreenError, ScreenReport, ScreenRow, screen

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Checkpoint",
    "CheckpointError",
    "TensorInfo",
    "CheckpointDiff",
    "DiffError",
    "TensorDiff",
    "diff_checkpoints",
    "Patch",
    "PatchError",
    "TensorEdit",
    "read_patch",
    "write_patch",
    "apply_patch",
    "revert_patch",
    "ScreenError",
    "ScreenReport",
    "ScreenRow",
    "screen",
]
