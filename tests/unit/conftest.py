"""Fixtures every unit test gets."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from price_predictor.infrastructure import forge_jvm

#: The packages whose modules can hold a name for the process-tree kill.
_OWN_PACKAGES = ("price_predictor", "sealed", "draft", "effects")

#: The real ``kill_process_tree``, taken before anything replaces it.
_REAL_KILL = forge_jvm.kill_process_tree


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_process_kill: let this test reach the real process-tree kill "
        "(taskkill /F /T); only for a test that kills a real child of its own",
    )


def _import_every_own_module() -> None:
    """Load every module of ours, so each one that binds the kill at load is
    found. ``__main__`` modules run their command when imported and are left out."""
    for package_name in _OWN_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            if not info.name.endswith(".__main__"):
                importlib.import_module(info.name)


class KillGuard:
    """Holds the stand-in for the process-tree kill at every name that held the
    real one, from the first test's setup to the session's end.

    The real ``kill_process_tree`` runs ``taskkill /F /T /PID <pid>`` on
    Windows, and a fake process's PID is made up: a unit test that reached it
    would kill whatever holds that PID on the machine. The patch lasts the
    session, not a test, because a worker pool's daemon monitor threads can
    outlive the test that started them (``ForgeWorkerPool.run`` joins them with
    a timeout) and still look the kill up afterwards.

    The stand-in records the process in ``killed`` and calls its own
    ``kill()``, which ends a fake (and, for a real child, only that child).
    """

    def __init__(self) -> None:
        self.killed: list = []
        #: Every (module, attribute) the guard replaced.
        self.sites: list[tuple[ModuleType, str]] = []
        self._patch = pytest.MonkeyPatch()

    def stand_in(self, proc) -> None:
        self.killed.append(proc)
        proc.kill()

    def patch_new_sites(self) -> None:
        """Replace the real kill wherever a loaded module of ours still holds it:
        a module loaded before the session patch, or during a marked test."""
        for name, module in list(sys.modules.items()):
            if module is None or name.split(".")[0] not in _OWN_PACKAGES:
                continue
            for attr, value in list(vars(module).items()):
                if value is _REAL_KILL:
                    self._patch.setattr(module, attr, self.stand_in)
                    self.sites.append((module, attr))

    def undo(self) -> None:
        self._patch.undo()


@pytest.fixture(scope="session", autouse=True)
def _kill_guard() -> Iterator[KillGuard]:
    _import_every_own_module()
    guard = KillGuard()
    guard.patch_new_sites()
    yield guard
    guard.undo()


@pytest.fixture(autouse=True)
def killed_process_trees(request, _kill_guard: KillGuard) -> Iterator[list]:
    """The processes this test handed the process-tree kill, in order.

    A test patching the kill itself still works: its patch sits on top of the
    stand-in. ``@pytest.mark.real_process_kill`` puts the real kill back at
    every name for that test alone.
    """
    _kill_guard.patch_new_sites()
    _kill_guard.killed = []
    if request.node.get_closest_marker("real_process_kill") is None:
        yield _kill_guard.killed
        return
    with pytest.MonkeyPatch.context() as real:
        for module, attr in _kill_guard.sites:
            real.setattr(module, attr, _REAL_KILL)
        yield _kill_guard.killed


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
