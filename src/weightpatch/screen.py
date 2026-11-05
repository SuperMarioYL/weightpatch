"""Check whether the edits recorded in a patch survived a derivative artifact.

Two derivative kinds, two questions.

A quantized **safetensors** export is compared on stored bytes: ``kept`` needs
the name, dtype, shape and byte digest of the recorded post-edit tensor to
match, i.e. an unquantized copy. A requantized export packs weights, so e.g.
``model.layers.0.mlp.down_proj.weight`` is ``U32 [2560, 1216]`` next to
``.scales [2560, 152]`` / ``.biases`` where the base is one BF16
``[2560, 9728]``; such a tensor is ``reshaped``, with the packed dtype, shape
and companion names beside it. A tensor whose bytes match neither digest is
reported ``missing``: the recorded edit is not there -- either the pre-edit
tensor is still stored, or something else was written.

A **GGUF** is checked for presence and shape only, never by dequantizing. Three
steps, in this order, because skipping any of them makes every declared edit
look absent:

1. resolve ``general.architecture`` to a ``gguf.MODEL_ARCH``. An architecture
   this ``gguf`` build does not declare is ``unsupported-arch``, with no name
   match attempted.
2. build ``gguf.get_tensor_name_map(arch, block_count)`` and map each
   safetensors name through ``get_name(..., try_suffixes=[".weight", ".bias"])``.
   GGUF names are GGML style (``token_embd.weight``, ``blk.12.attn_q.weight``),
   so an unmapped name is ``missing``, never matched by string similarity. When
   the map resolves none of the declared tensors, that is one ``unmapped-arch``
   row for the whole screen: a declared but unmappable namespace is a name-map
   gap, not per-tensor loss.
3. reverse ``GGUFReader.shape`` before comparing. That field is raw GGML ``ne``
   order, so every matrix arrives transposed; comparing it directly fails for
   every matrix tensor.

Only the sparse slice edits are screened. ``edit.added`` / ``edit.removed``
names carry no values in this format, so they are reported as notes rather
than turned into ``kept`` / ``missing`` rows.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gguf

from .checkpoint import Checkpoint, CheckpointError
from .patch import Patch

#: Every status a row can hold, in the order a summary lists them.
STATUS_ORDER = ("kept", "missing", "reshaped", "unsupported-arch", "unmapped-arch")

_ARCH_BY_NAME = {name.lower(): arch for arch, name in gguf.MODEL_ARCH_NAMES.items()}


class ScreenError(Exception):
    """A derivative artifact cannot be read well enough to screen it."""


@dataclass(frozen=True)
class ScreenRow:
    """The verdict for one declared edit, or for the screen as a whole."""

    name: str
    status: str
    detail: str
    mapped_name: str | None = None
    found_dtype: str | None = None
    found_shape: tuple[int, ...] | None = None
    companions: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "mapped_name": self.mapped_name,
            "found_dtype": self.found_dtype,
            "found_shape": list(self.found_shape)
            if self.found_shape is not None
            else None,
            "companions": list(self.companions),
        }


@dataclass(frozen=True)
class ScreenReport:
    """What a derivative artifact kept, lost or repacked of one patch."""

    source: Path
    kind: str
    rows: tuple[ScreenRow, ...]
    arch: str | None = None
    notes: tuple[str, ...] = ()

    def counts(self) -> dict[str, int]:
        counts = {"kept": 0, "missing": 0, "reshaped": 0}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts

    def summary(self) -> str:
        counts = self.counts()
        return " · ".join(
            f"{counts[status]} {status}" for status in STATUS_ORDER if status in counts
        )

    @property
    def ok(self) -> bool:
        """True only when every declared edit is present as recorded."""
        return all(row.status == "kept" for row in self.rows)

    def as_dict(self) -> dict:
        return {
            "source": str(self.source),
            "kind": self.kind,
            "arch": self.arch,
            "ok": self.ok,
            "summary": self.summary(),
            "counts": self.counts(),
            "rows": [row.as_dict() for row in self.rows],
            "notes": list(self.notes),
        }


def screen(source: str | Path, patch: Patch) -> ScreenReport:
    """Screen ``patch`` against a GGUF file or a safetensors checkpoint directory."""
    path = Path(source)
    if not path.exists():
        raise ScreenError(f"{path}: no such file or directory")
    if path.is_dir():
        return screen_safetensors(path, patch)
    if path.suffix.lower() == ".gguf":
        return screen_gguf(path, patch)
    raise ScreenError(
        f"{path}: expected a GGUF file or a safetensors checkpoint directory"
    )


# -- safetensors -----------------------------------------------------------


def screen_safetensors(directory: str | Path, patch: Patch) -> ScreenReport:
    """Compare the stored bytes of each declared tensor with the patch."""
    directory = Path(directory)
    try:
        with Checkpoint(directory) as checkpoint:
            rows = _screen_safetensors_rows(checkpoint, patch)
    except CheckpointError as exc:
        raise ScreenError(str(exc)) from exc
    return ScreenReport(
        source=directory,
        kind="safetensors",
        rows=tuple(rows),
        notes=_manifest_notes(patch),
    )


def _screen_safetensors_rows(checkpoint: Checkpoint, patch: Patch) -> list[ScreenRow]:
    available = checkpoint.tensors
    names = sorted(available)
    rows: list[ScreenRow] = []
    for edit in patch.edits:
        info = available.get(edit.name)
        if info is None:
            rows.append(
                ScreenRow(edit.name, "missing", "tensor is absent from the export")
            )
            continue
        if info.dtype != edit.dtype or info.shape != edit.shape:
            companions = _companions(edit.name, names)
            detail = (
                f"export stores {info.dtype}{_shape_text(info.shape)}; the patch "
                f"records {edit.dtype}{_shape_text(edit.shape)}"
            )
            if companions:
                detail += f"; packed companions: {', '.join(companions)}"
            rows.append(
                ScreenRow(
                    edit.name,
                    "reshaped",
                    detail,
                    found_dtype=info.dtype,
                    found_shape=info.shape,
                    companions=companions,
                )
            )
            continue
        digest = checkpoint.tensor_sha256(edit.name)
        if digest == edit.after_sha256:
            rows.append(
                ScreenRow(
                    edit.name,
                    "kept",
                    "stored bytes match the recorded post-edit tensor",
                )
            )
        elif digest == edit.before_sha256:
            rows.append(
                ScreenRow(
                    edit.name,
                    "missing",
                    "the export still stores the pre-edit tensor, byte for byte",
                )
            )
        else:
            rows.append(
                ScreenRow(
                    edit.name,
                    "missing",
                    "stored bytes match neither the recorded pre- nor post-edit tensor",
                )
            )
    return rows


def _companions(name: str, available: Sequence[str]) -> tuple[str, ...]:
    """Sibling ``.scales`` / ``.biases`` style names of one packed tensor."""
    stem = name.rpartition(".")[0]
    if not stem:
        return ()
    return tuple(
        sorted(
            other
            for other in available
            if other != name and other.rpartition(".")[0] == stem
        )
    )


# -- GGUF ------------------------------------------------------------------


def screen_gguf(path: str | Path, patch: Patch) -> ScreenReport:
    """Map declared names to GGML names and compare shapes, without dequantizing."""
    path = Path(path)
    reader = _open_gguf(path)
    notes = list(_manifest_notes(patch))
    arch_name = _architecture(reader, path)

    if not patch.edits:
        notes.append("the patch declares no slice edits")
        return ScreenReport(
            source=path, kind="gguf", rows=(), arch=arch_name, notes=tuple(notes)
        )

    arch = _ARCH_BY_NAME.get(arch_name.lower())
    if arch is None:
        rows = tuple(
            ScreenRow(
                edit.name,
                "unsupported-arch",
                f"gguf does not declare the architecture {arch_name!r}; "
                "no name map was attempted",
            )
            for edit in patch.edits
        )
        return ScreenReport(
            source=path, kind="gguf", rows=rows, arch=arch_name, notes=tuple(notes)
        )

    block_count = _block_count(reader, path, arch_name)
    name_map = gguf.get_tensor_name_map(arch, block_count)
    available = {tensor.name: tensor for tensor in reader.tensors}

    rows: list[ScreenRow] = []
    resolved = 0
    for edit in patch.edits:
        mapped = name_map.get_name(edit.name, try_suffixes=[".weight", ".bias"])
        if mapped is None:
            rows.append(
                ScreenRow(
                    edit.name,
                    "missing",
                    f"no GGML name for {edit.name!r} in the {arch_name} map",
                )
            )
            continue
        resolved += 1

        tensor = available.get(mapped)
        if tensor is None:
            rows.append(
                ScreenRow(
                    edit.name,
                    "missing",
                    f"{mapped!r} is not in the GGUF",
                    mapped_name=mapped,
                )
            )
            continue

        raw_shape = tuple(int(dim) for dim in tensor.shape)
        shape = tuple(reversed(raw_shape))
        dtype = _type_name(tensor.tensor_type)
        if shape == edit.shape:
            rows.append(
                ScreenRow(
                    edit.name,
                    "kept",
                    f"present as {mapped!r}; raw ne {_shape_text(raw_shape)} "
                    f"reverses to the recorded {_shape_text(edit.shape)} "
                    "(values not compared)",
                    mapped_name=mapped,
                    found_dtype=dtype,
                    found_shape=shape,
                )
            )
        else:
            rows.append(
                ScreenRow(
                    edit.name,
                    "reshaped",
                    f"{mapped!r} has raw ne {_shape_text(raw_shape)}, which "
                    f"reverses to {_shape_text(shape)}, not the recorded "
                    f"{_shape_text(edit.shape)}",
                    mapped_name=mapped,
                    found_dtype=dtype,
                    found_shape=shape,
                )
            )

    if resolved == 0:
        rows = [
            ScreenRow(
                arch_name,
                "unmapped-arch",
                f"the {arch_name} name map resolved none of the "
                f"{len(patch.edits)} declared tensor(s); "
                "no per-tensor rows are reported",
            )
        ]

    return ScreenReport(
        source=path, kind="gguf", rows=tuple(rows), arch=arch_name, notes=tuple(notes)
    )


def _open_gguf(path: Path) -> gguf.GGUFReader:
    try:
        return gguf.GGUFReader(str(path))
    except (OSError, ValueError, struct.error, IndexError) as exc:
        raise ScreenError(f"{path}: cannot read GGUF ({exc})") from exc


def _architecture(reader: gguf.GGUFReader, path: Path) -> str:
    field = reader.get_field(gguf.KEY_GENERAL_ARCHITECTURE)
    if field is None:
        raise ScreenError(f"{path}: no {gguf.KEY_GENERAL_ARCHITECTURE} metadata")
    value = field.contents()
    if not isinstance(value, str) or not value:
        raise ScreenError(
            f"{path}: {gguf.KEY_GENERAL_ARCHITECTURE} is not a string"
        )
    return value


def _block_count(reader: gguf.GGUFReader, path: Path, arch_name: str) -> int:
    key = f"{arch_name}.block_count"
    field = reader.get_field(key)
    if field is None:
        raise ScreenError(
            f"{path}: no {key} metadata; the GGML name map needs the block count"
        )
    value = field.contents()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScreenError(f"{path}: {key} is not a non-negative integer")
    return value


def _type_name(tensor_type: object) -> str:
    return getattr(tensor_type, "name", str(tensor_type))


def _manifest_notes(patch: Patch) -> tuple[str, ...]:
    notes = []
    if patch.added:
        notes.append(
            f"{len(patch.added)} added tensor(s) are not screened: "
            "a .wpatch carries no values for them"
        )
    if patch.removed:
        notes.append(
            f"{len(patch.removed)} removed tensor(s) are not screened: "
            "a .wpatch carries no values for them"
        )
    return tuple(notes)


def _shape_text(shape: Sequence[int]) -> str:
    return "×".join(str(dim) for dim in shape) if shape else "scalar"
