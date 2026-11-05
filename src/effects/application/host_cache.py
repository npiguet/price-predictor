"""Prepared batches kept small on the host, and unpacked bit for bit
(research E2; the control score's cache, ``control_score``).

The control score prepares the host half of every record it reads once, at
startup, and keeps it for the run, so its memory is what a collated batch costs
at rest. A collated batch is mostly padding and zeros: a card slot holds some
twenty non-zero values of its 539 features, and an id list a few small
integers in an int64. Each tensor is kept in whichever of these forms is
smallest, and :func:`unpack` rebuilds it exactly:

=============  ================================================================
form           kept as
=============  ================================================================
``_Coo``       a float or integer tensor's non-zero elements: flat positions
               and values
``_Rows``      a float tensor's rows (over its last dimension) holding a
               non-zero value
``_Bits``      a boolean tensor, or an integer one of zeros and ones, a bit per
               element
``_Narrow``    an integer tensor whose values fit 16 or 32 bits, in them
``_RowDiff``   a tensor whose rows are one record's grid cells: its first row,
               and where each row differs from it
``_Blocks``    a ``(n, k)`` list grouped by a leading batch-row column (the
               relation triples): each row's block, kept only where it differs
               from the row before
=============  ================================================================

The last two are what make a grid cheap: every cell of a grid edits the same
record, so the cells of one record, collated together, differ only in the few
features an edit touches (a power, a toughness, a granted keyword), and share
their relation triples but where an edit adds a line. :func:`pack` tries them
on tensors whose first dimension is the batch's ``rows``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields

import numpy as np
import torch


def _bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


_INT_RANGES = tuple((dtype, torch.iinfo(dtype)) for dtype in (torch.int16, torch.int32))


def _narrowest(tensor: torch.Tensor) -> torch.Tensor:
    """An integer tensor in the narrowest of int16 and int32 holding its
    values, else itself."""
    if tensor.is_floating_point() or tensor.dtype == torch.bool or tensor.numel() == 0:
        return tensor
    low, high = int(tensor.min()), int(tensor.max())
    for dtype, limits in _INT_RANGES:
        if limits.min <= low and high <= limits.max:
            return tensor if tensor.dtype == dtype else tensor.to(dtype)
    return tensor


def _positions(count: int, positions: torch.Tensor) -> torch.Tensor:
    """Flat positions as int32 where every position fits it."""
    return positions.to(torch.int32) if count <= torch.iinfo(torch.int32).max else positions


@dataclass(frozen=True, slots=True)
class _Coo:
    """A tensor kept as the flat positions and values of its non-zero elements."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    index: torch.Tensor
    values: torch.Tensor

    @classmethod
    def of(cls, tensor: torch.Tensor) -> _Coo:
        flat = tensor.reshape(-1)
        index = flat.nonzero().squeeze(1)
        return cls(tuple(tensor.shape), tensor.dtype, _positions(flat.numel(), index),
                   _narrowest(flat[index]))

    def dense(self) -> torch.Tensor:
        flat = torch.zeros(math.prod(self.shape), dtype=self.dtype)
        flat[self.index.long()] = self.values.to(self.dtype)
        return flat.reshape(self.shape)

    @property
    def nbytes(self) -> int:
        return _bytes(self.index) + _bytes(self.values)


@dataclass(frozen=True, slots=True)
class _Rows:
    """A float tensor kept as its rows holding a non-zero value. A tensor of
    three or more dimensions is cut into rows over its last dimension, a
    two-dimensional one into single elements."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    rows: torch.Tensor
    values: torch.Tensor

    @classmethod
    def of(cls, tensor: torch.Tensor) -> _Rows:
        width = tensor.shape[-1] if tensor.dim() >= 3 else 1
        flat = tensor.reshape(-1, width)
        rows = (flat != 0).any(-1).nonzero().squeeze(1)
        return cls(tuple(tensor.shape), tensor.dtype, rows.to(torch.int32), flat[rows])

    def dense(self) -> torch.Tensor:
        width = self.shape[-1] if len(self.shape) >= 3 else 1
        flat = torch.zeros((math.prod(self.shape) // max(width, 1), width), dtype=self.dtype)
        flat[self.rows.long()] = self.values
        return flat.reshape(self.shape)

    @property
    def nbytes(self) -> int:
        return _bytes(self.rows) + _bytes(self.values)


@dataclass(frozen=True, slots=True)
class _Bits:
    """A boolean tensor, or an integer one holding only zeros and ones, kept a
    bit per element."""

    shape: tuple[int, ...]
    bits: torch.Tensor
    dtype: torch.dtype = torch.bool

    @classmethod
    def of(cls, tensor: torch.Tensor) -> _Bits:
        flags = tensor.reshape(-1).to(torch.bool).numpy()
        return cls(tuple(tensor.shape), torch.from_numpy(np.packbits(flags)), tensor.dtype)

    def dense(self) -> torch.Tensor:
        count = math.prod(self.shape)
        flat = torch.from_numpy(np.unpackbits(self.bits.numpy(), count=count).astype(bool))
        return flat.reshape(self.shape).to(self.dtype)

    @property
    def nbytes(self) -> int:
        return _bytes(self.bits)


@dataclass(frozen=True, slots=True)
class _Narrow:
    """An integer tensor kept in a narrower integer type its values fit."""

    dtype: torch.dtype
    values: torch.Tensor

    def dense(self) -> torch.Tensor:
        return self.values.to(self.dtype)

    @property
    def nbytes(self) -> int:
        return _bytes(self.values)


@dataclass(frozen=True, slots=True)
class _RowDiff:
    """A tensor whose rows (its first dimension) are near copies of its first:
    the first row, packed, and every element of a later row that differs from
    the first row's, by flat position."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    first: object
    index: torch.Tensor
    values: torch.Tensor

    @classmethod
    def of(cls, tensor: torch.Tensor) -> _RowDiff:
        first = tensor[:1]
        flat = tensor.reshape(-1)
        index = (tensor != first).reshape(-1).nonzero().squeeze(1)
        values = flat[index]
        if tensor.dtype != torch.bool:
            values = _narrowest(values)
        return cls(tuple(tensor.shape), tensor.dtype, _single(first.contiguous()),
                   _positions(flat.numel(), index), values)

    def dense(self) -> torch.Tensor:
        first = _dense(self.first)
        out = first.expand(self.shape).clone()
        out.reshape(-1)[self.index.long()] = self.values.to(self.dtype)
        return out

    @property
    def nbytes(self) -> int:
        return _size(self.first) + _bytes(self.index) + _bytes(self.values)


@dataclass(frozen=True, slots=True)
class _Blocks:
    """A ``(n, k)`` integer list whose first column is a non-decreasing batch
    row in ``[0, rows)``: each row's block of the other columns, kept where it
    differs from the block of the row before. ``same[b]`` marks a row whose
    block repeats row ``b − 1``'s; ``blocks`` holds the others in row order,
    ``sizes`` their lengths."""

    dtype: torch.dtype
    width: int
    same: torch.Tensor
    sizes: torch.Tensor
    blocks: torch.Tensor

    @classmethod
    def of(cls, tensor: torch.Tensor, rows: int) -> _Blocks | None:
        if (tensor.dim() != 2 or tensor.shape[1] < 2 or tensor.is_floating_point()
                or tensor.dtype == torch.bool):
            return None
        batch = tensor[:, 0]
        if tensor.shape[0] and (int(batch.min()) < 0 or int(batch.max()) >= rows
                                or bool((batch[1:] < batch[:-1]).any())):
            return None
        counts = torch.bincount(batch, minlength=rows).tolist() if tensor.shape[0] else (
            [0] * rows)
        rest = tensor[:, 1:]
        same, kept, sizes = [], [], []
        start, previous = 0, None
        for count in counts:
            block = rest[start:start + count]
            start += count
            repeat = previous is not None and torch.equal(block, previous)
            same.append(repeat)
            if not repeat:
                kept.append(block)
                sizes.append(count)
            previous = block
        blocks = torch.cat(kept) if kept else rest[:0]
        return cls(tensor.dtype, int(tensor.shape[1]), torch.tensor(same, dtype=torch.bool),
                   _narrowest(torch.tensor(sizes, dtype=torch.long)), _narrowest(blocks))

    def dense(self) -> torch.Tensor:
        sizes = self.sizes.long().tolist()
        kept = torch.split(self.blocks.to(self.dtype), sizes) if sizes else ()
        parts, cursor, previous = [], 0, None
        for row, repeat in enumerate(self.same.tolist()):
            if not repeat:
                previous = kept[cursor]
                cursor += 1
            parts.append(torch.cat([torch.full((previous.shape[0], 1), row, dtype=self.dtype),
                                    previous], dim=1))
        if not parts:
            return torch.zeros((0, self.width), dtype=self.dtype)
        return torch.cat(parts)

    @property
    def nbytes(self) -> int:
        return _bytes(self.same) + _bytes(self.sizes) + _bytes(self.blocks)


PACKED = (_Coo, _Rows, _Bits, _Narrow, _RowDiff, _Blocks)


def _size(value) -> int:
    return _bytes(value) if isinstance(value, torch.Tensor) else value.nbytes


def _dense(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else value.dense()


def _single(tensor: torch.Tensor):
    """One tensor's smallest form, without the row codecs."""
    if tensor.numel() == 0 or tensor.dim() == 0:
        return tensor
    options: list = [tensor]
    if tensor.dtype == torch.bool:
        options.append(_Bits.of(tensor))
    elif tensor.is_floating_point():
        nonzero = int(torch.count_nonzero(tensor))
        if nonzero * 8 < _bytes(tensor):
            options.append(_Coo.of(tensor))
        if tensor.dim() >= 2:
            options.append(_Rows.of(tensor))
    else:
        low, high = int(tensor.min()), int(tensor.max())
        if low >= 0 and high <= 1:
            options.append(_Bits.of(tensor))
        narrow = _narrowest(tensor)
        if narrow is not tensor:
            options.append(_Narrow(tensor.dtype, narrow))
        if int(torch.count_nonzero(tensor)) * 6 < _bytes(narrow):
            options.append(_Coo.of(tensor))
    return min(options, key=_size)


def _packed(tensor: torch.Tensor, rows: int | None):
    """``tensor`` in its smallest host form, or itself: a tensor of fewer than
    two dimensions stays as it is."""
    if tensor.dim() < 2 or tensor.numel() == 0:
        return tensor
    options = [_single(tensor)]
    if rows is not None and rows > 1:
        if tensor.shape[0] == rows:
            options.append(_RowDiff.of(tensor))
        blocks = _Blocks.of(tensor, rows)
        if blocks is not None:
            options.append(blocks)
    return min(options, key=_size)


def map_tensors(value, fn):
    """``value`` with ``fn`` applied to every tensor (and packed tensor) it
    holds, through dataclasses, mappings and sequences."""
    if isinstance(value, (torch.Tensor, *PACKED)):
        return fn(value)
    if is_dataclass(value) and not isinstance(value, type):
        changed = {}
        for spec in dataclass_fields(value):
            if not spec.init:
                continue
            held = getattr(value, spec.name)
            mapped = map_tensors(held, fn)
            if mapped is not held:
                changed[spec.name] = mapped
        return replace(value, **changed) if changed else value
    if isinstance(value, dict):
        mapped = {key: map_tensors(item, fn) for key, item in value.items()}
        if all(mapped[key] is item for key, item in value.items()):
            return value
        return type(value)(mapped)
    if isinstance(value, (list, tuple)):
        items = [map_tensors(item, fn) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    return value


def pack(value, *, rows: int | None = None):
    """Every tensor of two or more dimensions ``value`` holds in its smallest
    host form. ``rows`` is the batch's record count, which lets a tensor of
    near-identical rows (one record's grid cells) be kept as its differences."""
    return map_tensors(value, lambda t: _packed(t, rows) if isinstance(t, torch.Tensor) else t)


def unpack(value):
    """``value`` with every packed tensor rebuilt exactly."""
    return map_tensors(value, lambda t: t.dense() if isinstance(t, PACKED) else t)


def host_bytes(value) -> int:
    """The bytes of every tensor ``value`` holds, packed ones as packed."""
    total = 0

    def count(tensor):
        nonlocal total
        total += _size(tensor)
        return tensor

    map_tensors(value, count)
    return total


__all__ = ["PACKED", "host_bytes", "map_tensors", "pack", "unpack"]
