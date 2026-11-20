"""A pool's signal handlers last as long as its run (finding 16.5, task L05):
the previous SIGINT and SIGTERM handlers are back when ``run()`` ends, by
return or by exception."""

from __future__ import annotations

import signal
from unittest.mock import patch

import pytest

from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool


def _pool(tmp_path) -> ForgeWorkerPool:
    return ForgeWorkerPool(
        worker_count=1, spawn_worker=lambda i: None, output_path=tmp_path / "out.txt",
    )


def _handlers() -> tuple:
    return signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)


def _idle_run(pool: ForgeWorkerPool, loop) -> None:
    with patch.object(ForgeWorkerPool, "_monitor_worker"), \
            patch.object(ForgeWorkerPool, "_supervisor_loop", loop), \
            patch.object(ForgeWorkerPool, "_terminate_all"):
        pool.run()


def test_the_pool_handles_signals_while_it_runs_and_restores_them_after(tmp_path):
    before = _handlers()
    pool = _pool(tmp_path)
    seen: list[tuple] = []

    _idle_run(pool, lambda self: seen.append(_handlers()))

    assert seen == [(pool._handle_signal, pool._handle_signal)]
    assert _handlers() == before


def test_the_handlers_are_restored_when_the_run_raises(tmp_path):
    before = _handlers()

    def boom(self) -> None:
        raise RuntimeError("supervisor failed")

    with pytest.raises(RuntimeError):
        _idle_run(_pool(tmp_path), boom)

    assert _handlers() == before
