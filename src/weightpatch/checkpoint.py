"""Read safetensors checkpoints without a deep-learning framework.

The ``safetensors`` numpy backend cannot open BF16 tensors: numpy has no
``bfloat16`` dtype, so ``safe_open(..., framework="numpy")`` raises
``TypeError: data type 'bfloat16' not understood``. This module therefore
parses the container itself -- an 8-byte little-endian header length followed
by a JSON map of ``{dtype, shape, data_offsets}`` -- and seeks to byte ranges.

Tensor payloads are exposed as raw machine words:

* ``uint16`` words for BF16, one word per element;
* ``uint8`` bytes shaped ``(elements, element_size)`` for every other dtype.

Comparing, hashing or storing those words needs no float decoding, and a BF16
slice round-trips exactly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence

import numpy as np

INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_BLOCK_ELEMENTS = 1 << 16

_HEADER_LEN_BYTES = 8
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class CheckpointError(Exception):
    """A checkpoint directory or shard is missing or malformed."""


@dataclass(frozen=True)
class TensorInfo:
    """Where one tensor lives and how its bytes are laid out."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    offset: int  # absolute byte offset of element 0
    nbytes: int

    @property
    def numel(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count

    @property
    def element_size(self) -> int:
        return self.nbytes // self.numel if self.numel else 0


def _parse_shape(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in value
    ):
        raise CheckpointError(f"{where}: invalid shape {value!r}")
    return tuple(value)


def _parse_shard(path: Path) -> dict[str, TensorInfo]:
    """Parse one shard header and return its tensor map."""
    with path.open("rb") as handle:
        prefix = handle.read(_HEADER_LEN_BYTES)
        if len(prefix) != _HEADER_LEN_BYTES:
            raise CheckpointError(f"{path}: truncated header length")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len == 0 or header_len > _MAX_HEADER_BYTES:
            raise CheckpointError(f"{path}: implausible header length {header_len}")
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise CheckpointError(f"{path}: truncated header")

    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise CheckpointError(f"{path}: header is not a JSON object")

    data_start = _HEADER_LEN_BYTES + header_len
    data_size = path.stat().st_size - data_start
    if data_size < 0:
        raise CheckpointError(f"{path}: header is longer than the file")

    tensors: dict[str, TensorInfo] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise CheckpointError(f"{path}: entry {name!r} is not an object")
        dtype = entry.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise CheckpointError(f"{path}: tensor {name!r} has no dtype")
        shape = _parse_shape(entry.get("shape"), f"{path}: tensor {name!r}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
        ):
            raise CheckpointError(f"{path}: tensor {name!r} has invalid data_offsets {offsets!r}")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise CheckpointError(
                f"{path}: tensor {name!r} offsets [{start}, {end}) outside the data section"
            )
        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype,
            shape=shape,
            file=path,
            offset=data_start + start,
            nbytes=end - start,
        )
    return tensors


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class Checkpoint:
    """An open safetensors checkpoint directory.

    Shards are enumerated from ``model.safetensors.index.json`` when it exists
    (single-file checkpoints carry no index), otherwise from every
    ``*.safetensors`` file in the directory. Shard files are opened lazily and
    kept open until :meth:`close`; reads never load a whole shard.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise CheckpointError(f"not a checkpoint directory: {self.directory}")
        self._tensors: dict[str, TensorInfo] = {}
        self._shards: dict[str, Path] = {}
        self._handles: dict[Path, BinaryIO] = {}
        self._file_digests: dict[str, str] = {}

        index_path = self.directory / INDEX_FILENAME
        if index_path.is_file():
            self._load_from_index(index_path)
        else:
            self._load_from_directory()
        if not self._tensors:
            raise CheckpointError(f"no tensors found in {self.directory}")

    # -- enumeration -----------------------------------------------------

    def _load_from_index(self, index_path: Path) -> None:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"{index_path}: unreadable ({exc})") from exc

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"{index_path}: missing weight_map")

        shard_names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or not name for name in shard_names):
            raise CheckpointError(f"{index_path}: weight_map has an invalid shard name")

        shard_tensors: dict[str, dict[str, TensorInfo]] = {}
        for shard_name in shard_names:
            path = self.directory / shard_name
            if not path.is_file():
                raise CheckpointError(f"{index_path}: shard {shard_name!r} is missing")
            shard_tensors[shard_name] = _parse_shard(path)
            self._shards[shard_name] = path

        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise CheckpointError(f"{index_path}: invalid tensor name {tensor_name!r}")
            if not isinstance(shard_name, str):
                raise CheckpointError(f"{index_path}: tensor {tensor_name!r} has no shard")
            info = shard_tensors[shard_name].get(tensor_name)
            if info is None:
                raise CheckpointError(
                    f"{index_path}: lists {tensor_name!r} in {shard_name!r}, "
                    "but that shard does not contain it"
                )
            self._tensors[tensor_name] = info

    def _load_from_directory(self) -> None:
        shards = sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix == ".safetensors"
        )
        if not shards:
            raise CheckpointError(
                f"{self.directory}: no {INDEX_FILENAME} and no *.safetensors files"
            )
        for path in shards:
            for name, info in _parse_shard(path).items():
                previous = self._tensors.get(name)
                if previous is not None:
                    raise CheckpointError(
                        f"tensor {name!r} appears in both {previous.file.name!r} "
                        f"and {path.name!r}"
                    )
                self._tensors[name] = info
            self._shards[path.name] = path

    # -- introspection ---------------------------------------------------

    @property
    def tensors(self) -> Mapping[str, TensorInfo]:
        return dict(self._tensors)

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError:
            raise CheckpointError(f"{self.directory}: no tensor {name!r}") from None

    def describe(self) -> dict:
        """The plan's ``base`` object: shard bytes plus digests and a count."""
        files = []
        for shard_name in sorted(self._shards):
            path = self._shards[shard_name]
            digest = self._file_digests.get(shard_name)
            if digest is None:
                digest = _sha256_file(path)
                self._file_digests[shard_name] = digest
            files.append(
                {"name": shard_name, "bytes": path.stat().st_size, "sha256": digest}
            )
        return {"files": files, "tensor_count": len(self._tensors)}

    # -- byte access -----------------------------------------------------

    def _handle(self, path: Path) -> BinaryIO:
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        return handle

    def _read(self, info: TensorInfo, offset: int, nbytes: int) -> bytes:
        handle = self._handle(info.file)
        handle.seek(offset)
        raw = handle.read(nbytes)
        if len(raw) != nbytes:
            raise CheckpointError(f"{info.file.name}: tensor {info.name!r} is truncated")
        return raw

    def iter_blocks(
        self,
        name: str,
        *,
        start: int = 0,
        stop: int | None = None,
        block_elements: int = DEFAULT_BLOCK_ELEMENTS,
    ) -> Iterator[np.ndarray]:
        """Yield element-aligned blocks of ``[start, stop)`` for one tensor.

        A BF16 block has shape ``(n,)`` with dtype ``uint16``; any other dtype
        yields shape ``(n, element_size)`` with dtype ``uint8``.
        """
        if block_elements <= 0:
            raise ValueError("block_elements must be positive")
        info = self.tensor(name)
        stop = info.numel if stop is None else stop
        if not 0 <= start <= stop <= info.numel:
            raise ValueError(
                f"range [{start}, {stop}) is outside tensor {name!r} "
                f"with {info.numel} elements"
            )
        element_size = info.element_size
        if start == stop:
            return
        if element_size == 0:
            raise CheckpointError(f"tensor {name!r} has elements but no byte width")

        position = start
        while position < stop:
            count = min(block_elements, stop - position)
            raw = self._read(info, info.offset + position * element_size, count * element_size)
            if info.dtype == "BF16":
                yield np.frombuffer(raw, dtype=np.uint16)
            else:
                yield np.frombuffer(raw, dtype=np.uint8).reshape(count, element_size)
            position += count

    def read_slab(self, name: str, ranges: Sequence[Sequence[int]]) -> np.ndarray:
        """Read one rectangular slab given ``[dim, start, end]`` ranges.

        The result keeps the tensor's word form: ``uint16`` with the slab shape
        for BF16, ``uint8`` with shape ``(*slab_shape, element_size)``
        otherwise. An empty ``ranges`` list addresses a scalar tensor.
        """
        info = self.tensor(name)
        shape = info.shape
        if len(ranges) != len(shape):
            raise ValueError(
                f"tensor {name!r} has {len(shape)} dimensions, got {len(ranges)} ranges"
            )
        spans: list[tuple[int, int]] = []
        for expected_dim, entry in enumerate(ranges):
            if len(entry) != 3:
                raise ValueError(f"range {entry!r} must be [dim, start, end]")
            dim, begin, end = (int(value) for value in entry)
            if dim != expected_dim:
                raise ValueError(f"range {entry!r} is out of order")
            if not 0 <= begin < end <= shape[dim]:
                raise ValueError(f"range {entry!r} is outside dimension {dim}")
            spans.append((begin, end))

        element_size = info.element_size
        if element_size == 0:
            raise CheckpointError(f"tensor {name!r} has no byte width")

        if not shape:
            raw = self._read(info, info.offset, info.nbytes)
            if info.dtype == "BF16":
                return np.frombuffer(raw, dtype=np.uint16).reshape(())
            return np.frombuffer(raw, dtype=np.uint8).reshape(element_size)

        strides = [1] * len(shape)
        for dim in range(len(shape) - 2, -1, -1):
            strides[dim] = strides[dim + 1] * shape[dim + 1]

        leading = spans[:-1]
        last_begin, last_end = spans[-1]
        row_elements = last_end - last_begin
        row_bytes = row_elements * element_size

        chunks = []
        for coordinates in itertools.product(*(range(begin, end) for begin, end in leading)):
            flat = last_begin + sum(
                coordinate * stride for coordinate, stride in zip(coordinates, strides)
            )
            chunks.append(
                self._read(info, info.offset + flat * element_size, row_bytes)
            )
        raw = b"".join(chunks)
        lengths = tuple(end - begin for begin, end in spans)
        if info.dtype == "BF16":
            return np.frombuffer(raw, dtype=np.uint16).reshape(lengths)
        return np.frombuffer(raw, dtype=np.uint8).reshape(*lengths, element_size)

    def tensor_sha256(self, name: str) -> str:
        digest = hashlib.sha256()
        for block in self.iter_blocks(name):
            digest.update(block.tobytes())
        return digest.hexdigest()

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False
