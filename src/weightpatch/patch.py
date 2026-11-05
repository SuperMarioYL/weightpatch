"""Read, verify, apply, stack and revert ``.wpatch`` files.

A ``.wpatch`` file is a zip with two members:

``manifest.json``
    ``id`` -- ``sha256`` of the canonical ``(base, parents, edit)`` document, so
    a patch is content addressed; ``base`` -- shard names, sizes and digests of
    the checkpoint the edit was recorded against; ``parents`` -- ids of the
    patches this one was recorded on top of; ``edit`` -- the sparse slice
    edits; ``probe`` -- the author's declared claim.
``values.npz``
    the raw before/after words of every changed slice. A BF16 slice is
    ``uint16``, every other dtype is opaque ``uint8`` bytes shaped
    ``(*slice, element_size)``. No floats, no decoding.

The id covers the manifest, not ``values.npz``, so the stored words are
checked where they are used: once the slices are written, every touched tensor
is hashed again and must equal the digest the manifest recorded. That check is
also what makes ``revert`` byte-identical rather than merely shape-preserving.
Pre-images are verified before anything is copied, so a patch that does not
fit its base writes nothing.

Identity is per touched tensor, not per checkpoint: a patch applies to any
checkpoint whose touched tensors still hash to ``before_sha256``. That is what
allows cross-base application.

Stacking means applying a later patch to the output of an earlier one, so the
later patch must have been diffed against that state (its ``before_sha256`` is
the digest after the earlier patches). Two patches applied in one call may
touch the same tensor only on disjoint slices; an overlap is refused before
anything is written.

v0.1 carries values for slice edits only. A patch that records ``added`` or
``removed`` tensors has no bytes to write or restore, so it can be shown and
screened but not applied; :func:`apply_patch` refuses it.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .checkpoint import INDEX_FILENAME, Checkpoint, CheckpointError, TensorInfo
from .diff import VALUES_FILENAME, CheckpointDiff, values_key

MANIFEST_FILENAME = "manifest.json"


class PatchError(Exception):
    """A ``.wpatch`` file is malformed, or does not fit the checkpoint."""


def canonical_document(document: dict) -> bytes:
    """The byte form a patch id is derived from."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def patch_id(base: dict, parents: Sequence[str], edit: dict) -> str:
    """``sha256`` over the canonical ``(base, parents, edit)`` document."""
    document = {"base": base, "parents": list(parents), "edit": edit}
    return hashlib.sha256(canonical_document(document)).hexdigest()


@dataclass(frozen=True)
class TensorEdit:
    """One recorded slice edit and the raw words of both sides."""

    name: str
    dtype: str
    element_size: int
    shape: tuple[int, ...]
    ranges: tuple[tuple[int, int, int], ...]
    before_sha256: str
    after_sha256: str
    before: np.ndarray
    after: np.ndarray

    @property
    def slab_shape(self) -> tuple[int, ...]:
        """Shape of the stored words of one side, before the byte width."""
        return tuple(end - start for _, start, end in self.ranges)


@dataclass(frozen=True)
class Patch:
    """An opened ``.wpatch``: its manifest, its verified slice edits, its path."""

    path: Path
    manifest: dict
    edits: tuple[TensorEdit, ...]

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def base(self) -> dict:
        return self.manifest["base"]

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(self.manifest["parents"])

    @property
    def added(self) -> tuple[str, ...]:
        return tuple(self.manifest["edit"].get("added") or ())

    @property
    def removed(self) -> tuple[str, ...]:
        return tuple(self.manifest["edit"].get("removed") or ())

    @property
    def probe(self) -> dict:
        return self.manifest["probe"]


# -- writing ---------------------------------------------------------------


def write_patch(
    diff: CheckpointDiff,
    path: str | Path,
    *,
    parents: Sequence[str] = (),
    probe: dict | None = None,
) -> Patch:
    """Write ``diff`` as a ``.wpatch`` and return it opened for reading."""
    document = diff.manifest(parents=parents, probe=probe)
    manifest = {
        "id": patch_id(document["base"], document["parents"], document["edit"]),
        **document,
    }

    arrays: dict[str, np.ndarray] = {}
    edits: list[TensorEdit] = []
    for tensor in diff.tensors:
        before_key = values_key(tensor.name, "before")
        after_key = values_key(tensor.name, "after")
        arrays[before_key] = tensor.before
        arrays[after_key] = tensor.after
        edits.append(
            TensorEdit(
                name=tensor.name,
                dtype=tensor.dtype,
                element_size=tensor.element_size,
                shape=tensor.shape,
                ranges=tuple(tuple(span) for span in tensor.ranges),
                before_sha256=tensor.before_sha256,
                after_sha256=tensor.after_sha256,
                before=tensor.before,
                after=tensor.after,
            )
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            MANIFEST_FILENAME, json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        if arrays:
            archive.writestr(VALUES_FILENAME, _npz_bytes(arrays))

    return Patch(path=path, manifest=manifest, edits=tuple(edits))


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


# -- reading ---------------------------------------------------------------


def read_patch(path: str | Path) -> Patch:
    """Open a ``.wpatch`` and verify its structure, content id and values."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            if MANIFEST_FILENAME not in members:
                raise PatchError(f"{path}: no {MANIFEST_FILENAME}; not a .wpatch file")
            try:
                manifest = json.loads(archive.read(MANIFEST_FILENAME))
            except json.JSONDecodeError as exc:
                raise PatchError(
                    f"{path}: {MANIFEST_FILENAME} is not valid JSON ({exc})"
                ) from exc
            edits = _parse_manifest(manifest, path, archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PatchError(f"{path}: unreadable .wpatch file ({exc})") from exc

    return Patch(path=path, manifest=manifest, edits=edits)


def _parse_manifest(
    manifest: object, path: Path, archive: zipfile.ZipFile
) -> tuple[TensorEdit, ...]:
    where = f"{path}: manifest"
    if not isinstance(manifest, dict):
        raise PatchError(f"{where} is not a JSON object")
    for key in ("id", "base", "parents", "edit", "probe"):
        if key not in manifest:
            raise PatchError(f"{where} has no {key!r}")

    base = manifest["base"]
    parents = manifest["parents"]
    edit = manifest["edit"]
    if not isinstance(base, dict):
        raise PatchError(f"{where}: base is not an object")
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) for parent in parents
    ):
        raise PatchError(f"{where}: parents is not a list of ids")
    if not isinstance(manifest["probe"], dict):
        raise PatchError(f"{where}: probe is not an object")
    if not isinstance(edit, dict):
        raise PatchError(f"{where}: edit is not an object")
    entries = edit.get("tensors")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise PatchError(f"{where}: edit.tensors is not a list of tensor entries")
    for key in ("added", "removed"):
        names = edit.get(key)
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise PatchError(f"{where}: edit.{key} is not a list of names")

    derived = patch_id(base, parents, edit)
    if manifest["id"] != derived:
        raise PatchError(
            f"{path}: content id {manifest['id']!r} does not match the manifest "
            f"body ({derived}); the patch was changed after it was written"
        )

    arrays = _load_values(archive, path) if entries else {}
    return tuple(
        _parse_entry(entry, index, path, arrays) for index, entry in enumerate(entries)
    )


def _load_values(archive: zipfile.ZipFile, path: Path) -> dict[str, np.ndarray]:
    if VALUES_FILENAME not in archive.namelist():
        raise PatchError(f"{path}: no {VALUES_FILENAME}; the patch carries no values")
    try:
        with np.load(
            io.BytesIO(archive.read(VALUES_FILENAME)), allow_pickle=False
        ) as npz:
            return {key: npz[key] for key in npz.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise PatchError(f"{path}: unreadable {VALUES_FILENAME} ({exc})") from exc


def _parse_entry(
    entry: object, index: int, path: Path, arrays: dict[str, np.ndarray]
) -> TensorEdit:
    where = f"{path}: edit.tensors[{index}]"
    if not isinstance(entry, dict):
        raise PatchError(f"{where} is not an object")

    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise PatchError(f"{where}: bad name {name!r}")
    dtype = entry.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise PatchError(f"{where}: bad dtype {dtype!r}")
    element_size = entry.get("element_size")
    if (
        not isinstance(element_size, int)
        or isinstance(element_size, bool)
        or element_size < 1
    ):
        raise PatchError(f"{where}: bad element_size {element_size!r}")
    shape = _parse_shape(entry.get("shape"), where)
    ranges = _parse_ranges(entry.get("ranges"), shape, where)
    before_sha256 = _parse_digest(entry.get("before_sha256"), where, "before_sha256")
    after_sha256 = _parse_digest(entry.get("after_sha256"), where, "after_sha256")

    ref = entry.get("ref")
    if not isinstance(ref, dict) or ref.get("values") != VALUES_FILENAME:
        raise PatchError(f"{where}: ref does not point into {VALUES_FILENAME}")

    slab = tuple(end - start for _, start, end in ranges)
    word_shape = slab if dtype == "BF16" else (*slab, element_size)
    word_dtype = np.uint16 if dtype == "BF16" else np.uint8
    sides: dict[str, np.ndarray] = {}
    for side in ("before", "after"):
        key = ref.get(side)
        if not isinstance(key, str) or not key:
            raise PatchError(f"{where}: ref.{side} is not a values key")
        array = arrays.get(key)
        if array is None:
            raise PatchError(f"{where}: {VALUES_FILENAME} has no array {key!r}")
        if array.dtype != word_dtype or tuple(array.shape) != word_shape:
            raise PatchError(
                f"{where}: {key!r} is {array.dtype}{tuple(array.shape)}, "
                f"expected {np.dtype(word_dtype)}{word_shape}"
            )
        sides[side] = array

    return TensorEdit(
        name=name,
        dtype=dtype,
        element_size=element_size,
        shape=shape,
        ranges=ranges,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        before=sides["before"],
        after=sides["after"],
    )


def _parse_shape(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in value
    ):
        raise PatchError(f"{where}: bad shape {value!r}")
    return tuple(value)


def _parse_ranges(
    value: object, shape: tuple[int, ...], where: str
) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list) or len(value) != len(shape):
        raise PatchError(
            f"{where}: ranges must hold one [dim, start, end] per dimension"
        )
    ranges: list[tuple[int, int, int]] = []
    for dim, span in enumerate(value):
        if (
            not isinstance(span, list)
            or len(span) != 3
            or any(not isinstance(part, int) or isinstance(part, bool) for part in span)
        ):
            raise PatchError(f"{where}: bad range {span!r}")
        if span[0] != dim:
            raise PatchError(f"{where}: range {span!r} is out of order")
        _, start, end = span
        if not 0 <= start < end <= shape[dim]:
            raise PatchError(
                f"{where}: range {span!r} is outside dimension {dim} of {shape}"
            )
        ranges.append((dim, start, end))
    return tuple(ranges)


def _parse_digest(value: object, where: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PatchError(f"{where}: bad {field} {value!r}")
    return value


# -- apply and revert ------------------------------------------------------


@dataclass(frozen=True)
class _Write:
    """One tensor to rewrite: the digest it must have now and once written."""

    edit: TensorEdit
    source_sha256: str
    values: np.ndarray
    result_sha256: str


def apply_patch(
    target: str | Path, patches: Sequence[Patch], out_dir: str | Path
) -> None:
    """Write a copy of ``target`` with every patch in ``patches`` applied.

    The output holds the target's tensor shards and shard index; weights are
    the whole contract, so configuration and tokenizer files are not copied.
    """
    stack = list(patches)
    if not stack:
        raise PatchError("no patches to apply")
    for patch in stack:
        _reject_valueless_edits(patch)
    _check_slice_conflicts(stack)
    writes = [
        _Write(edit, edit.before_sha256, edit.after, edit.after_sha256)
        for patch in stack
        for edit in patch.edits
    ]
    _write_edits(target, writes, out_dir, operation="apply")


def revert_patch(target: str | Path, patch: Patch, out_dir: str | Path) -> None:
    """Write a copy of ``target`` with ``patch`` undone."""
    _reject_valueless_edits(patch)
    writes = [
        _Write(edit, edit.after_sha256, edit.before, edit.before_sha256)
        for edit in patch.edits
    ]
    _write_edits(target, writes, out_dir, operation="revert")


def _reject_valueless_edits(patch: Patch) -> None:
    if not (patch.added or patch.removed):
        return
    names = [*patch.added, *patch.removed]
    raise PatchError(
        f"{patch.path}: the patch records {len(names)} added or removed "
        f"tensor(s) ({', '.join(names)}), but a .wpatch carries values for "
        "slice edits only; it can be shown and screened, not applied"
    )


def _write_edits(
    target: str | Path,
    writes: Sequence[_Write],
    out_dir: str | Path,
    *,
    operation: str,
) -> None:
    if not writes:
        raise PatchError(f"nothing to {operation}: the patch declares no slice edits")

    try:
        source = Checkpoint(target)
    except CheckpointError as exc:
        raise PatchError(str(exc)) from exc

    with source:
        infos = [
            _require_layout(source, write.edit, operation) for write in writes
        ]
        promises = _verify_pre_images(source, writes, operation)
        out = _prepare_output(out_dir)
        for shard in sorted({info.file for info in source.tensors.values()}):
            shutil.copyfile(shard, out / shard.name)
        index = source.directory / INDEX_FILENAME
        if index.is_file():
            shutil.copyfile(index, out / index.name)

    for info, write in zip(infos, writes):
        _write_slab(out / info.file.name, info, write.edit, write.values)

    try:
        staged = Checkpoint(out)
    except CheckpointError as exc:
        raise PatchError(
            f"{out}: cannot re-open the written checkpoint ({exc})"
        ) from exc
    with staged:
        for name, expected in promises.items():
            _verify_digest(
                staged, name, expected, operation=operation, side="post-image"
            )


def _verify_pre_images(
    source: Checkpoint, writes: Sequence[_Write], operation: str
) -> dict[str, str]:
    """Check every pre-image before copying, so a moved base writes nothing.

    A later write to an already edited tensor must expect the digest an earlier
    write promised; that is what lets a stacked chain be checked up front
    without staging the intermediate states. The returned map is each tensor's
    digest after its last write, which the post-image pass then verifies.
    """
    target_digests: dict[str, str] = {}
    promises: dict[str, str] = {}
    for write in writes:
        name = write.edit.name
        if name in promises:
            expected = promises[name]
        else:
            if name not in target_digests:
                target_digests[name] = source.tensor_sha256(name)
            expected = target_digests[name]
        if expected != write.source_sha256:
            raise PatchError(
                f"cannot {operation} {name!r}: its pre-image digest is "
                f"{expected[:12]}, but the patch records {write.source_sha256[:12]}"
            )
        promises[name] = write.result_sha256
    return promises


def _prepare_output(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    if out.exists():
        if not out.is_dir():
            raise PatchError(f"{out}: output path is not a directory")
        if any(out.iterdir()):
            raise PatchError(f"{out}: output directory is not empty")
    else:
        out.mkdir(parents=True)
    return out


def _require_layout(
    checkpoint: Checkpoint, edit: TensorEdit, operation: str
) -> TensorInfo:
    info = checkpoint.tensors.get(edit.name)
    if info is None:
        raise PatchError(
            f"cannot {operation} {edit.name!r}: "
            f"{checkpoint.directory} has no such tensor"
        )
    if info.dtype != edit.dtype or info.shape != edit.shape:
        raise PatchError(
            f"cannot {operation} {edit.name!r}: the patch records "
            f"{edit.dtype}{_shape_text(edit.shape)}, the checkpoint has "
            f"{info.dtype}{_shape_text(info.shape)}"
        )
    return info


def _verify_digest(
    checkpoint: Checkpoint, name: str, expected: str, *, operation: str, side: str
) -> None:
    actual = checkpoint.tensor_sha256(name)
    if actual != expected:
        raise PatchError(
            f"cannot {operation} {name!r}: its {side} digest is {actual[:12]}, "
            f"but the patch records {expected[:12]}"
        )


def _write_slab(
    shard: Path, info: TensorInfo, edit: TensorEdit, values: np.ndarray
) -> None:
    """Write one slab into an already laid out tensor, row by row."""
    with shard.open("r+b") as handle:
        if not edit.shape:
            handle.seek(info.offset)
            handle.write(values.tobytes())
            return
        spans = [(start, end) for _, start, end in edit.ranges]
        strides = _row_strides(edit.shape)
        leading = spans[:-1]
        last_start = spans[-1][0]
        for local in itertools.product(*(range(end - start) for start, end in leading)):
            absolute = [start + offset for offset, (start, _) in zip(local, leading)]
            flat = last_start + sum(
                coordinate * stride for coordinate, stride in zip(absolute, strides)
            )
            handle.seek(info.offset + flat * info.element_size)
            handle.write(values[local].tobytes())


def _row_strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return strides


def _check_slice_conflicts(patches: Sequence[Patch]) -> None:
    for index, left in enumerate(patches):
        for right in patches[index + 1 :]:
            for left_edit in left.edits:
                for right_edit in right.edits:
                    if (
                        left_edit.name != right_edit.name
                        or left_edit.shape != right_edit.shape
                    ):
                        continue
                    overlap = _slice_overlap(left_edit.ranges, right_edit.ranges)
                    if overlap is not None:
                        raise PatchError(
                            f"{left.path.name} and {right.path.name} both edit "
                            f"{left_edit.name!r} at overlapping slices "
                            f"{_range_text(overlap)}"
                        )


def _slice_overlap(
    left: Sequence[tuple[int, int, int]], right: Sequence[tuple[int, int, int]]
) -> list[list[int]] | None:
    overlap: list[list[int]] = []
    for (dim, left_start, left_end), (_, right_start, right_end) in zip(left, right):
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if start >= end:
            return None
        overlap.append([dim, start, end])
    return overlap


def _range_text(ranges: Sequence[Sequence[int]]) -> str:
    return (
        " ".join(f"[{dim}:{start}:{end}]" for dim, start, end in ranges)
        or "the whole tensor"
    )


def _shape_text(shape: Sequence[int]) -> str:
    return "×".join(str(dim) for dim in shape) if shape else "scalar"
