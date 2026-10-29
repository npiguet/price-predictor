"""What both collectors share: the supervisor is stopped however a run ends."""

from __future__ import annotations

import pytest

from effects.application.collector_common import collector_supervisor
from effects.domain.collection_caps import CollectionCaps


class TestTheSupervisorIsAlwaysStopped:
    """``CollectorSupervisor.stop()`` is the only caller of
    ``WorkerLogFiles.close_all()``. Unless the run stops it, the worker log
    handles opened so the latched effect-record failure reporters are visible
    to an operator stay open until the interpreter exits on its own.
    """

    def _open(self, tmp_path):
        return collector_supervisor(
            workers=1, effect_records=tmp_path, caps=CollectionCaps(),
        )

    def test_stop_runs_when_the_rounds_end(self, tmp_path, fake_supervisor):
        with self._open(tmp_path) as supervisor:
            assert supervisor is fake_supervisor
            fake_supervisor.stop.assert_not_called()

        fake_supervisor.stop.assert_called_once()

    def test_stop_runs_even_if_a_round_raises(self, tmp_path, fake_supervisor):
        """The property a bare call after the loop cannot give."""
        with pytest.raises(RuntimeError, match="boom"), self._open(tmp_path):
            raise RuntimeError("boom")

        fake_supervisor.stop.assert_called_once()
