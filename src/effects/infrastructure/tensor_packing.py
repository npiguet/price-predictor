"""Moving a message full of tensors between processes as one buffer.

``torch.multiprocessing`` pays a shared-memory mapping per tensor, and on
Windows that cost is per tensor rather than per byte, so a message is pickled
with its tensors set aside and their bytes packed into one flat buffer.
Pinning that one buffer pins the whole message.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass

import torch

#: Byte alignment of each tensor in a packed buffer: a view of a byte buffer
#: as a wider dtype needs its offset to be a multiple of the element size.
ALIGNMENT = 64


@dataclass(frozen=True)
class PackedStep:
    """A message with every tensor moved into one flat buffer.

    ``torch.multiprocessing`` pays a shared-memory mapping per tensor, and on
    Windows that cost is per tensor rather than per byte.
    """

    #: The message pickled with each tensor replaced by its index in ``layout``.
    skeleton: bytes
    #: ``(dtype, shape, offset, nbytes)`` per tensor, in pickling order.
    layout: tuple[tuple[torch.dtype, tuple[int, ...], int, int], ...]
    flat: torch.Tensor


class _Packer(pickle.Pickler):
    """Pickles a message with its tensors set aside, by index."""

    def __init__(self, file) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.tensors: list[torch.Tensor] = []

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return len(self.tensors) - 1
        return None


class _Unpacker(pickle.Unpickler):
    def __init__(self, file, tensors: list[torch.Tensor]) -> None:
        super().__init__(file)
        self.tensors = tensors

    def persistent_load(self, pid):
        return self.tensors[pid]


def pack(message) -> PackedStep:
    """``message`` as a skeleton and one flat buffer of every tensor's bytes."""
    stream = io.BytesIO()
    packer = _Packer(stream)
    packer.dump(message)
    tensors = [tensor.detach().contiguous() for tensor in packer.tensors]
    layout = []
    offset = 0
    for tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        layout.append((tensor.dtype, tuple(tensor.shape), offset, nbytes))
        offset += -(-nbytes // ALIGNMENT) * ALIGNMENT
    flat = torch.empty(offset, dtype=torch.uint8)
    for tensor, (_dtype, _shape, start, nbytes) in zip(tensors, layout):
        if nbytes:
            flat[start:start + nbytes] = tensor.reshape(-1).view(torch.uint8)
    return PackedStep(stream.getvalue(), tuple(layout), flat)


def unpack(packed: PackedStep, *, pin: bool = False):
    """The message ``pack`` took apart, its tensors views of one buffer,
    which ``pin`` pins in one copy."""
    flat = packed.flat.pin_memory() if pin else packed.flat
    tensors = [
        flat[start:start + nbytes].view(dtype).reshape(shape) if nbytes
        else torch.empty(shape, dtype=dtype, pin_memory=pin)
        for dtype, shape, start, nbytes in packed.layout
    ]
    return _Unpacker(io.BytesIO(packed.skeleton), tensors).load()
