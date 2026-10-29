"""Worker processes that prepare training steps: the host half, off the GIL.

Shard ``k`` goes to worker ``k mod N``, each with a queue of its own, which
keeps the run order fixed with no sequencing protocol. Pinning happens in the
training process, because page-locked memory does not survive the trip
between processes.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import pickle
import queue
import threading
import traceback
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.multiprocessing as torch_mp

from effects.application.step_preparation import (
    PreparerSettings,
    ShardClosed,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.train_effect_model import load_shard

logger = logging.getLogger(__name__)

#: Byte alignment of each tensor in a packed buffer: a view of a byte buffer
#: as a wider dtype needs its offset to be a multiple of the element size.
ALIGNMENT = 64

#: How long a blocking queue call waits before checking the other end is alive.
POLL_SECONDS = 0.5

#: Prepared steps the receiver thread holds ready beyond the one training;
#: each is pinned memory, which this bounds.
READY_STEPS = 2


@dataclass(frozen=True)
class PackedStep:
    """A ``StepReady`` with every tensor moved into one flat buffer.

    ``torch.multiprocessing`` pays a shared-memory mapping per tensor, and on
    Windows that cost is per tensor rather than per byte.
    """

    #: The message pickled with each tensor replaced by its index in ``layout``.
    skeleton: bytes
    #: ``(dtype, shape, offset, nbytes)`` per tensor, in pickling order.
    layout: tuple[tuple[torch.dtype, tuple[int, ...], int, int], ...]
    flat: torch.Tensor


@dataclass(frozen=True)
class WorkerFailed:
    """What a worker sends in place of the rest of a shard when it raises."""

    worker: int
    traceback: str


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


def pack(message: StepReady) -> PackedStep:
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


def unpack(packed: PackedStep, *, pin: bool = False) -> StepReady:
    """The ``StepReady`` ``pack`` took apart, its tensors views of one buffer,
    which ``pin`` pins in one copy."""
    flat = packed.flat.pin_memory() if pin else packed.flat
    tensors = [
        flat[start:start + nbytes].view(dtype).reshape(shape) if nbytes
        else torch.empty(shape, dtype=dtype, pin_memory=pin)
        for dtype, shape, start, nbytes in packed.layout
    ]
    return _Unpacker(io.BytesIO(packed.skeleton), tensors).load()


def _exit_with_parent() -> None:
    """Exit this worker when the training process does, even when it is
    killed outright and runs none of its shutdown."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def work(index: int, settings: PreparerSettings, tasks, results) -> None:
    """A worker's whole life: prepare each task's shard until told to stop.

    An exception is sent back in place of the rest of its shard. The worker
    then never exits on its own: on Windows the shared memory of steps it
    queued ahead would die with it, before the traceback is read.
    """
    _exit_with_parent()
    # Single-threaded Python; torch's own pool would only oversubscribe cores.
    torch.set_num_threads(1)
    try:
        preparer = StepPreparer(settings)
        while (task := tasks.get()) is not None:
            records = load_shard(task.shard)
            for message in preparer.steps(task, records):
                results.put(
                    pack(message) if isinstance(message, StepReady) else message,
                )
            # Before the next read, so one shard is all a worker holds.
            del records
    except KeyboardInterrupt:
        return
    except BaseException:
        results.put(WorkerFailed(index, traceback.format_exc()))
        while tasks.get() is not None:
            continue


class StepWorkerPool:
    """``workers`` step-preparing processes and the thread that reads them.

    ``shard`` yields the next submitted shard's messages (``ShardOpened``, its
    steps, ``ShardClosed``); ``depth`` is how many prepared steps each worker
    may hold queued.
    """

    def __init__(
        self, settings: PreparerSettings, *, workers: int, depth: int,
        pin: bool,
    ) -> None:
        context = torch_mp.get_context("spawn")
        self.workers = workers
        self.pin = pin
        self._tasks = [context.Queue() for _ in range(workers)]
        # Two beyond the steps: the shard's opening and closing messages.
        self._results = [context.Queue(maxsize=depth + 2) for _ in range(workers)]
        self._processes = [
            context.Process(
                target=work, args=(index, settings, self._tasks[index],
                                   self._results[index]),
                name=f"step-worker-{index}", daemon=True,
            )
            for index in range(workers)
        ]
        for process in self._processes:
            process.start()
        self._submitted = 0
        #: The worker each submitted shard went to, in run order.
        self._order: queue.Queue[int] = queue.Queue()
        self._ready: queue.Queue = queue.Queue(maxsize=READY_STEPS)
        self._stop = threading.Event()
        self._receiver = threading.Thread(
            target=self._receive, name="step-receiver", daemon=True,
        )
        self._receiver.start()

    def submit(self, task: ShardTask) -> None:
        worker = self._submitted % self.workers
        self._submitted += 1
        self._tasks[worker].put(task)
        self._order.put(worker)

    def shard(self) -> Iterator:
        """The next submitted shard's messages, in order, ending at its close;
        raises a worker's exception at the message it took the place of."""
        while True:
            item = self._take()
            if isinstance(item, BaseException):
                raise item
            yield item
            if isinstance(item, ShardClosed):
                return

    def _take(self):
        while True:
            try:
                return self._ready.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if not self._receiver.is_alive():
                    return RuntimeError("the step receiver thread is gone")

    def _receive(self) -> None:
        """Read each shard back from its worker, unpack and pin each step."""
        try:
            while not self._stop.is_set():
                try:
                    worker = self._order.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                while True:
                    message = self._read(worker)
                    if message is None:
                        return
                    if isinstance(message, PackedStep):
                        message = unpack(message, pin=self.pin)
                    elif isinstance(message, WorkerFailed):
                        message = RuntimeError(
                            f"step-preparation worker {message.worker} "
                            f"failed:\n{message.traceback}"
                        )
                    if not self._hand_over(message):
                        return
                    if isinstance(message, (ShardClosed, BaseException)):
                        break
        except BaseException as exc:  # noqa: BLE001 — handed to the trainer
            self._hand_over(exc)

    def _read(self, worker: int):
        """The worker's next message, or None once the pool is stopping."""
        process = self._processes[worker]
        while not self._stop.is_set():
            try:
                return self._results[worker].get(timeout=POLL_SECONDS)
            except queue.Empty:
                if not process.is_alive():
                    return self._died(worker)
            except Exception:
                # A step whose shared memory went with a worker that died
                # holding it: the death is the error worth reporting.
                if process.is_alive():
                    raise
                return self._died(worker)
        return None

    def _died(self, worker: int) -> WorkerFailed:
        return WorkerFailed(
            worker,
            f"exited with code {self._processes[worker].exitcode} without "
            "reporting an error",
        )

    def _hand_over(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._ready.put(item, timeout=POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def close(self) -> None:
        """Terminate every worker and stop the receiver; safe to call more
        than once. A worker holds nothing worth flushing."""
        self._stop.set()
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=10)
        self._receiver.join(timeout=10)
        for channel in (*self._tasks, *self._results):
            # Nothing written to a queue whose reader is gone may hold up the
            # training process's own exit.
            channel.cancel_join_thread()
            channel.close()
