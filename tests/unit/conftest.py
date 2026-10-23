"""Fixtures every unit test gets."""

from __future__ import annotations

import pytest

from price_predictor.infrastructure import forge_jvm


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
