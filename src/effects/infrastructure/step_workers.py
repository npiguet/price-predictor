"""Worker processes that prepare training steps: the host half, off the GIL.

The trainer's host half — parsing a shard, constructing its records, and for
every step tokenizing, building surfaces, collating and deriving every target —
is pure Python, and on threads it competed for the one interpreter lock with
the training step's own Python side. Here it runs in ``--prefetch-workers``
processes instead, the ``DataLoader``-with-``num_workers`` pattern without the
``Dataset`` abstraction, which has no notion of a shard read once and planned
from.

Topology. Shard ``k`` of the run goes to worker ``k mod N``. Each worker reads
its shard, plans it and prepares every step of it in order onto a bounded queue
of its own, then moves on to its next shard. A receiver thread in the training
process reads the shards back in run order, one worker queue at a time, unpacks
each step and pins it, and hands it to the training loop through a short local
queue. Round-robin plus one queue per worker is what keeps the order fixed with
no sequencing protocol: worker ``k mod N`` produces shard ``k``'s steps in
order, and nothing else writes to that queue. A queue holds up to one shard's
steps, so all ``N`` workers prepare at once rather than one preparing while the
others wait for the training loop to reach them.

What crosses the boundary is never an ``EffectRecord``: a task names a shard
path, and what comes back is finished CPU tensors. Each step's tensors are
packed into **one** flat byte buffer plus a small pickled skeleton that names
each tensor's dtype, shape and offset. ``torch.multiprocessing`` gives every
tensor put on a queue a shared-memory mapping of its own, and on Windows that
cost is paid per tensor rather than per byte; a step holds several dozen small
tensors, so packing makes it one. The training process rebuilds them as views
of the buffer, and pinning that one buffer pins them all — in the training
process, because page-locked memory does not survive the trip between
processes.

Workers are spawned (the only start method on Windows), take everything they
need as one picklable ``PreparerSettings``, build their own tokenizer and
sidecar cache, and exit with the training process: on a normal end and on an
exception the pool terminates them, and a watchdog thread in each exits it if
the training process dies without saying so.
"""

from __future__ import annotations

import io
import logging
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

logger = logging.getLogger(__name__)

#: Byte alignment of each tensor in a packed buffer: a view of a byte buffer
#: as a wider dtype needs its offset to be a multiple of the element size.
ALIGNMENT = 64

#: How long a blocking queue call waits before checking whether the other end
#: is still alive. Short enough that a dead worker is noticed within a second,
#: long enough that the polling costs nothing.
POLL_SECONDS = 0.5

#: Prepared steps the receiver thread holds ready beyond the one training.
#: Each is pinned memory, so the bound is what keeps pinned memory to a
#: couple of steps whatever the workers have queued.
READY_STEPS = 2


@dataclass(frozen=True)
class PackedStep:
    """A ``StepReady`` with every tensor moved into one flat buffer."""

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
    """The ``StepReady`` ``pack`` took apart, its tensors views of one buffer.

    ``pin`` copies the buffer into page-locked memory first, once, so every
    tensor of the step is pinned by one copy rather than one per tensor.
    """
    flat = packed.flat.pin_memory() if pin else packed.flat
    tensors = [
        flat[start:start + nbytes].view(dtype).reshape(shape) if nbytes
        else torch.empty(shape, dtype=dtype, pin_memory=pin)
        for dtype, shape, start, nbytes in packed.layout
    ]
    return _Unpacker(io.BytesIO(packed.skeleton), tensors).load()


def _exit_with_parent() -> None:
    """Exit this worker when the training process does, however it ends.

    A training process killed outright runs none of its shutdown, and a
    worker blocked on a full queue would otherwise wait forever for a reader
    that is gone — an orphaned ``python.exe`` per worker.
    """
    import multiprocessing

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def work(index: int, settings: PreparerSettings, tasks, results) -> None:
    """A worker's whole life: prepare each task's shard until told to stop.

    Module-level so a spawned process can import it, and doing nothing at
    import time. An exception is sent back with its traceback in place of the
    rest of the shard it belongs to, so the training process raises it at that
    shard; an interrupt ends the worker quietly, since the training process
    received the same one and is shutting the pool down.

    A worker never exits on its own while the run lasts, not even after
    failing. On Windows a tensor's shared memory lives only while some
    process holds it open, so the steps this worker queued ahead of its
    failure would vanish with it, and the training process would read an
    unopenable mapping where the traceback should have been.
    """
    _exit_with_parent()
    # The host half is single-threaded Python; torch's own pool would only
    # oversubscribe the cores the training process and the other workers use.
    torch.set_num_threads(1)
    try:
        from effects.application.train_effect_model import load_shard

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
        # Until the pool terminates it; see above.
        while tasks.get() is not None:
            continue


class StepWorkerPool:
    """``workers`` step-preparing processes and the thread that reads them.

    ``submit`` queues shards in run order; ``shard`` yields the next one's
    messages — a ``ShardOpened``, its steps, a ``ShardClosed`` — in that same
    order. ``depth`` is how many prepared steps each worker may hold queued.
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
        """The next submitted shard's messages, in order, ending at its close.

        Raises a worker's exception, with the worker's traceback, at the
        message it took the place of.
        """
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

    # ── the receiver thread ─────────────────────────────────────────────

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

    # ── shutdown ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop every worker and the receiver; safe to call more than once.

        Terminated rather than asked to finish: a worker is usually blocked
        on a full queue holding steps of a shard nobody will train on, and
        it holds nothing worth flushing.
        """
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

    def __enter__(self) -> StepWorkerPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
