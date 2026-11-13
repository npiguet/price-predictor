"""No effects unit test starts a process pool it did not size.

``run_parallel`` gives ``workers=None`` one process per core, and the
collectors' ``--workers`` default is 12: a test that leaves either in place
starts a pool far wider than it needs. ``tests/unit/effects/conftest.py``
fails such a test; a test that needs a wide pool opts in with
``@pytest.mark.wide_process_pool``.

``run_parallel`` is a generator: calling it starts nothing until it is
iterated, so none of these checks starts a process.
"""

from __future__ import annotations

import sys

import pytest

from effects.application import collect_coverage, parallel_map

_REAL = parallel_map.__dict__["run_parallel"]


def _call(workers, items=("a", "b", "c", "d", "e", "f")):
    return collect_coverage.run_parallel(
        list(items), str, workers=workers, label="Counted", progress_every=0,
    )


def test_every_name_run_parallel_is_imported_under_is_guarded():
    holders = [
        f"{name}.{attr}"
        for name, module in list(sys.modules.items())
        if module is not None and name.split(".")[0] == "effects"
        for attr, value in list(vars(module).items())
        if value is _REAL
    ]
    assert holders == []


def test_a_pool_sized_by_the_core_count_fails_the_test():
    with pytest.raises(pytest.fail.Exception, match="workers=None"):
        _call(None)


def test_a_pool_wider_than_four_fails_the_test():
    with pytest.raises(pytest.fail.Exception, match="12"):
        _call(12)


def test_an_explicit_narrow_pool_is_allowed():
    _call(4)


def test_one_item_starts_no_pool_whatever_the_count():
    _call(None, items=("a",))


@pytest.mark.wide_process_pool
def test_the_marker_allows_a_wide_pool():
    _call(None)
