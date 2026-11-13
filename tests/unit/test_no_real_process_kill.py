"""No unit test reaches the real process-tree kill.

The real ``kill_process_tree`` runs ``taskkill /F /T /PID <pid>`` on Windows
(``os.killpg`` elsewhere). A fake process carries a made-up PID, so a unit test
that let it through would kill whatever process holds that PID on the machine.
``tests/unit/conftest.py`` swaps the kill for a stand-in at every name it is
looked up under; a test that means to kill a real child opts out with
``@pytest.mark.real_process_kill``.
"""

from __future__ import annotations

import os

import pytest

import sealed.infrastructure.evaluation_connector as evaluation_connector
from price_predictor.infrastructure import forge_jvm
from price_predictor.infrastructure.forge_jvm import kill_process_tree as real_kill
from tests.unit.sealed.conftest import FakeProcess


def _record_os_kills(monkeypatch) -> list:
    """Record, instead of run, the calls the real kill would make."""
    calls: list = []
    monkeypatch.setattr(forge_jvm.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(a), raising=False)
    return calls


def test_the_pool_kill_never_reaches_the_os(monkeypatch, killed_process_trees):
    calls = _record_os_kills(monkeypatch)
    proc = FakeProcess(pid=1000, hang=True)

    forge_jvm.kill_process_tree(proc)

    assert calls == []
    assert killed_process_trees == [proc]
    assert proc.poll() is not None, "the stand-in still ends the fake process"


@pytest.mark.parametrize("module", [forge_jvm, evaluation_connector])
def test_every_name_the_kill_is_looked_up_under_is_replaced(module):
    assert module.kill_process_tree is not real_kill


@pytest.mark.real_process_kill
@pytest.mark.parametrize("module", [forge_jvm, evaluation_connector])
def test_the_marker_leaves_the_real_kill_in_place(module):
    assert module.kill_process_tree is real_kill
