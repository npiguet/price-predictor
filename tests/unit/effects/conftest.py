"""Fixtures every effects unit test gets."""

from __future__ import annotations

import os
import sys

import pytest

from effects.application import parallel_map

#: The real ``run_parallel``, taken before anything replaces it.
_REAL_RUN_PARALLEL = parallel_map.run_parallel

#: The widest pool a test may start without opting in.
_WIDEST_POOL = 4


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wide_process_pool: let this test start a run_parallel pool sized by the "
        "core count or wider than four processes",
    )


@pytest.fixture(autouse=True)
def _no_accidental_process_pool(request, monkeypatch):
    """Fail a test whose ``run_parallel`` would start a pool it did not size.

    ``workers=None`` means one process per core, and the collectors' default
    ``--workers`` is 12: a counter test that leaves either in place starts
    that many processes. The guard cannot tell a defaulted 12 from an explicit
    one, so it fails ``workers=None`` and any pool wider than four processes;
    the deliberate parallel tests pass explicit counts of four or fewer. The
    check runs at the call, before the pool exists, and replaces
    ``run_parallel`` at every name an effects module imported it under.
    """
    if request.node.get_closest_marker("wide_process_pool"):
        return

    def guarded(items, task, *, workers, **kwargs):
        cores = os.process_cpu_count() or 1
        size = min(cores if workers is None else workers, len(items))
        if size > 1 and (workers is None or size > _WIDEST_POOL):
            pytest.fail(
                f"run_parallel would start {size} processes from workers={workers!r}; "
                f"pass workers=1 (or an explicit count up to {_WIDEST_POOL}), or mark "
                "the test wide_process_pool",
                pytrace=False,
            )
        return _REAL_RUN_PARALLEL(items, task, workers=workers, **kwargs)

    for name, module in list(sys.modules.items()):
        if module is None or name.split(".")[0] != "effects":
            continue
        for attr, value in list(vars(module).items()):
            if value is _REAL_RUN_PARALLEL:
                monkeypatch.setattr(module, attr, guarded)
