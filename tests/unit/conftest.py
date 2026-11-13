"""Fixtures every unit test gets."""

from __future__ import annotations

import pytest

import sealed.infrastructure.evaluation_connector as evaluation_connector
from price_predictor.infrastructure import forge_jvm

#: Every module that holds its own name for ``kill_process_tree``: the module
#: that defines it (the worker pool and the draft commands, which import it at
#: call time, look it up there) and each module that imported the name at load.
_KILL_SITES = (forge_jvm, evaluation_connector)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_process_kill: let this test reach the real process-tree kill "
        "(taskkill /F /T); only for a test that kills a real child of its own",
    )


@pytest.fixture(autouse=True)
def killed_process_trees(request, monkeypatch) -> list:
    """Replace the process-tree kill; returns the processes it was handed.

    The real ``kill_process_tree`` runs ``taskkill /F /T /PID <pid>`` on
    Windows, and a fake process's PID is made up: a unit test that reached it
    would kill whatever holds that PID on the machine. The stand-in records the
    process and calls its own ``kill()``, which ends a fake (and, for a real
    child, only that child). A test patching the kill itself still works: its
    patch sits on top of this one. ``@pytest.mark.real_process_kill`` opts out.
    """
    killed: list = []
    if request.node.get_closest_marker("real_process_kill"):
        return killed

    def _kill(proc) -> None:
        killed.append(proc)
        proc.kill()

    for module in _KILL_SITES:
        monkeypatch.setattr(module, "kill_process_tree", _kill)
    return killed


@pytest.fixture(autouse=True)
def _no_java_version_probe(monkeypatch):
    """Answer the JVM-version question without spawning ``java -version``.

    ``build_jvm_command`` asks it once per process to decide on JDK 24+'s
    Unsafe memory-access option. Unanswered here, the first unit test to build
    a command would spawn a JVM, and a test that patches ``subprocess.run``
    would see the probe as one of its own calls depending on test order. An
    unidentified JVM gets the command a pre-24 JDK gets, which is what every
    command-shape assertion was written against.
    """
    monkeypatch.setattr(forge_jvm, "java_feature_version", lambda: None)
