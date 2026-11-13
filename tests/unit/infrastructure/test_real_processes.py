"""The real-process test helpers leave nothing running when a start fails."""

from __future__ import annotations

import pytest

from tests.unit.infrastructure import real_processes


def test_a_grandchild_that_cannot_be_owned_is_ended_with_its_parent(tmp_path, monkeypatch):
    """OwnedProcess raising after the grandchild started left it sleeping."""
    real_owned = real_processes.OwnedProcess

    def refuse(pid):
        raise RuntimeError("could not open the grandchild")

    monkeypatch.setattr(real_processes, "OwnedProcess", refuse)
    pid_file = tmp_path / "grandchild.pid"

    with pytest.raises(RuntimeError, match="could not open"):
        real_processes.start_with_grandchild(pid_file)

    with pytest.raises(RuntimeError, match="not running"):
        real_owned(int(pid_file.read_text(encoding="utf-8")))
