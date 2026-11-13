"""No unit test reaches the real process-tree kill.

The real ``kill_process_tree`` runs ``taskkill /F /T /PID <pid>`` on Windows
(``os.killpg`` elsewhere). A fake process carries a made-up PID, so a unit test
that let it through would kill whatever process holds that PID on the machine.
``tests/unit/conftest.py`` imports every module of ours once per session and
swaps the kill for a stand-in at every name that holds it, for the whole
session; a test that means to kill a real child opts out with
``@pytest.mark.real_process_kill``.

Every check here compares identities; none calls the real kill.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import types

import pytest

from price_predictor.infrastructure import forge_jvm
from price_predictor.infrastructure.forge_jvm import kill_process_tree as real_kill
from tests.unit.sealed.conftest import FakeProcess

_OWN_PACKAGES = ("price_predictor", "sealed", "draft", "effects")


def _own_modules() -> list[types.ModuleType]:
    return [
        module for name, module in list(sys.modules.items())
        if module is not None and name.split(".")[0] in _OWN_PACKAGES
    ]


def _holders_of_the_real_kill() -> list[str]:
    return [
        f"{module.__name__}.{attr}"
        for module in _own_modules()
        for attr, value in list(vars(module).items())
        if value is real_kill
    ]


@pytest.fixture(scope="module", autouse=True)
def _between_tests():
    """What the kill is outside any one test: before this file's first test
    sets up and after its last one tears down, when a pool's monitor thread
    that outlived its test would look it up."""
    state = {"real_before": forge_jvm.kill_process_tree is real_kill}
    yield state
    assert forge_jvm.kill_process_tree is not real_kill, "the real kill is back between tests"
    assert _holders_of_the_real_kill() == []


def test_the_kill_is_replaced_before_any_test_sets_up(_between_tests):
    assert _between_tests["real_before"] is False


def test_every_module_of_ours_is_imported():
    """A module holding its own name for the kill is only found if it is loaded."""
    missing = []
    for package_name in _OWN_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            if not info.name.endswith(".__main__") and info.name not in sys.modules:
                missing.append(info.name)
    assert missing == []


def test_no_module_of_ours_holds_the_real_kill():
    assert _holders_of_the_real_kill() == []


def test_a_module_that_binds_the_real_kill_at_load_is_caught(monkeypatch, _kill_guard):
    """A module loaded before the patch, or during a marked test, keeps the real
    function under whatever name it imported it as; the next test's setup finds it."""
    probe = types.ModuleType("effects._real_kill_probe")
    probe.kill_process_tree = real_kill
    probe.terminate_tree = real_kill
    monkeypatch.setitem(sys.modules, probe.__name__, probe)

    _kill_guard.patch_new_sites()

    assert probe.kill_process_tree is not real_kill
    assert probe.terminate_tree is not real_kill


def test_the_pool_kill_never_reaches_the_os(monkeypatch, killed_process_trees):
    calls: list = []
    monkeypatch.setattr(forge_jvm.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(a), raising=False)
    proc = FakeProcess(pid=1000, hang=True)

    forge_jvm.kill_process_tree(proc)

    assert calls == []
    assert killed_process_trees == [proc]
    assert proc.poll() is not None, "the stand-in still ends the fake process"


def test_each_test_gets_its_own_record(killed_process_trees):
    assert killed_process_trees == []


@pytest.mark.real_process_kill
def test_the_marker_puts_the_real_kill_back_for_its_test(_kill_guard):
    assert _kill_guard.sites, "the session patched at least the defining module"
    for module, attr in _kill_guard.sites:
        assert getattr(module, attr) is real_kill, f"{module.__name__}.{attr}"
