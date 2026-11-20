"""Unit tests for MatchWorkerConnector subprocess construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerDecks,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)

RUN_ID = "a3f4b8c2-1234-4abc-9def-0123456789ab"
BEST_OF = 3


def _plain(output_file, best_of=BEST_OF) -> WorkerLaunch:
    """A launch with an output file and nothing else."""
    return WorkerLaunch(output_file, run_id=RUN_ID, best_of=best_of)


@pytest.fixture
def stub_classpath():
    """Stub build_forge_classpath at the consumer site."""
    with patch(
        "sealed.infrastructure.match_worker_connector.build_forge_classpath",
        return_value="fake-classpath",
    ):
        yield


class TestMatchWorkerConnectorCommandConstruction:
    def test_command_includes_xmx_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert "-Xmx1200m" in cmd

    def test_command_includes_main_class(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert "com.pricepredictor.connector.MatchWorkerMain" in cmd

    def test_command_includes_classpath_flag(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert "-cp" in cmd
        cp_index = cmd.index("-cp")
        assert cmd[cp_index + 1] == "fake-classpath"

    def test_output_file_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        output_file = tmp_path / "sealed" / "match-outcomes.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=WorkerLaunch(output_file, run_id=RUN_ID, best_of=BEST_OF))
        cmd = mock_popen.call_args[0][0]
        assert f"-Doutput.file={output_file}" in cmd

    def test_start_returns_popen_handle(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc):
            result = connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        assert result is mock_proc

    def test_log_file_used_when_provided(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        log_file = MagicMock()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                ),
                log_file=log_file,
            )
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["stdout"] is log_file
        assert kwargs["stderr"] is log_file

class TestMatchWorkerConnectorSideDecks:
    """``WorkerDecks``' ``side_a`` / ``side_b`` / ``side_b_weight``
    propagate to the Java worker as ``-Dside.a.decks.file`` /
    ``-Dside.b.decks.file`` / ``-Dside.b.decks.weight`` system properties."""

    def test_side_a_path_added_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        side_a = tmp_path / "side-a.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    decks=WorkerDecks(side_a=side_a),
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dside.a.decks.file={side_a}" in cmd
        assert not any(arg.startswith("-Dside.b.decks.file=") for arg in cmd)
        # Without --side-b-decks the weight property is also omitted.
        assert not any(arg.startswith("-Dside.b.decks.weight=") for arg in cmd)

    def test_side_b_path_and_weight_added_as_system_properties(
        self, tmp_path, stub_classpath,
    ):
        connector = MatchWorkerConnector()
        side_b = tmp_path / "side-b.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    decks=WorkerDecks(side_b=side_b, side_b_weight=8),
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dside.b.decks.file={side_b}" in cmd
        assert "-Dside.b.decks.weight=8" in cmd

    def test_side_paths_omitted_when_none(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dside.a.decks.file=") for arg in cmd)
        assert not any(arg.startswith("-Dside.b.decks.file=") for arg in cmd)
        assert not any(arg.startswith("-Dside.b.decks.weight=") for arg in cmd)

class TestMatchWorkerConnectorDecksOnly:
    """``decks_only`` forces both sides onto the decks file, as
    ``-Dsealed.decks.only``. A coverage or variant deck belongs to no set, so
    the Forge-method branch would resolve its sentinel set code against
    Forge's booster/edition tables and throw; this flag exists to prevent the
    roll from ever reaching that branch."""

    def test_decks_only_true_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    decks=WorkerDecks(decks_only=True),
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert "-Dsealed.decks.only=true" in cmd

    def test_decks_only_omitted_by_default(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dsealed.decks.only=") for arg in cmd)


class TestMatchWorkerConnectorProgressFile:
    """``progress_file`` reaches the worker as ``-Dsealed.progress.file``,
    independent of ``output_file``.

    Round-1 review, CRITICAL finding: a records-only round's progress file
    never grew because nothing appended to it. The fix is this parameter --
    a channel of its own, so a caller can request a progress tally without
    also passing a real ``output_file`` (which would flip records-only mode
    off and construct a real sealed corpus (``SealedCorpus``), see
    ``MatchWorkerMain.java``'s ``recordsOnly()``)."""

    def test_progress_file_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        progress = tmp_path / "run-id.progress.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    progress_file=progress,
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dsealed.progress.file={progress}" in cmd

    def test_progress_file_omitted_by_default(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dsealed.progress.file=") for arg in cmd)

    def test_progress_file_works_with_output_file_absent(self, stub_classpath, tmp_path):
        # The exact records-only shape CollectorSupervisor.start_worker uses:
        # output_file=None (positional), effect_records_dir set, progress_file
        # set. Must not raise the "needs an output file, effect records, or
        # both" ValueError, and must still emit the progress property.
        connector = MatchWorkerConnector()
        progress = tmp_path / "run-id.progress.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    None,
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    records=WorkerRecords(directory=tmp_path / "records"),
                    progress_file=progress,
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dsealed.progress.file={progress}" in cmd
        assert not any(arg.startswith("-Doutput.file=") for arg in cmd)


class TestMatchWorkerConnectorRunId:
    """run_id is required and propagates to the worker as a system property."""

    def test_run_id_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt"))
        cmd = mock_popen.call_args[0][0]
        assert f"-Dmatch.run.id={RUN_ID}" in cmd


class TestMatchWorkerConnectorBestOf:
    """best_of plumbs through as -Dmatch.best.of and validates odd+positive."""

    def test_best_of_passed_as_system_property(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(launch=_plain(tmp_path / "outcomes.txt", best_of=7))
        cmd = mock_popen.call_args[0][0]
        assert "-Dmatch.best.of=7" in cmd

    def test_even_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(launch=_plain(tmp_path / "outcomes.txt", best_of=4))

    def test_zero_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(launch=_plain(tmp_path / "outcomes.txt", best_of=0))

    def test_negative_best_of_raises(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with pytest.raises(ValueError, match="positive odd integer"):
            connector.start(launch=_plain(tmp_path / "outcomes.txt", best_of=-3))


class TestMatchWorkerConnectorEffectRecords:
    """``effect_records_dir`` / ``worker_index`` reach the worker, and the
    per-JVM half of a record id deliberately does not.

    The pool restarts a worker slot hundreds of times per run, so anything this
    launcher passes is by definition the same for every lifetime of that slot.
    That is why the shard writer mints its own lifetime token: a token handed
    down from here would be constant across restarts and would reissue the
    previous JVM's record and game ids, which is exactly what the first
    collected corpus did.
    """

    def _cmd(self, tmp_path, mock_popen, records):
        connector = MatchWorkerConnector()
        mock_popen.return_value = MagicMock()
        connector.start(launch=WorkerLaunch(
            tmp_path / "outcomes.txt", run_id=RUN_ID, best_of=BEST_OF, records=records,
        ))
        return mock_popen.call_args[0][0]

    def test_records_dir_and_worker_index_passed_as_system_properties(
        self, tmp_path, stub_classpath
    ):
        records = tmp_path / "effects" / "records"
        with patch("subprocess.Popen") as mock_popen:
            cmd = self._cmd(
                tmp_path, mock_popen, WorkerRecords(directory=records, worker_index=3)
            )
        assert f"-Deffect.records.dir={records}" in cmd
        assert "-Deffect.worker.index=3" in cmd

    def test_the_lifetime_token_is_never_supplied_by_the_launcher(
        self, tmp_path, stub_classpath
    ):
        # The worker mints its own lifetime token. A launcher passing one
        # would pin one value for every restart of the slot, which is the
        # defect the token was added to fix.
        with patch("subprocess.Popen") as mock_popen:
            cmd = self._cmd(
                tmp_path,
                mock_popen,
                WorkerRecords(directory=tmp_path / "records", worker_index=0),
            )
        assert not any(arg.startswith("-Deffect.worker.lifetime") for arg in cmd)

    def test_a_respawned_slot_is_launched_identically(self, tmp_path, stub_classpath):
        # Two lifetimes of one slot, from the supervisor's point of view. Nothing
        # here tells them apart, so nothing the worker counts from zero may be
        # namespaced by what this command line carries.
        records = tmp_path / "records"
        with patch("subprocess.Popen") as mock_popen:
            first = self._cmd(
                tmp_path, mock_popen, WorkerRecords(directory=records, worker_index=2)
            )
        with patch("subprocess.Popen") as mock_popen:
            second = self._cmd(
                tmp_path, mock_popen, WorkerRecords(directory=records, worker_index=2)
            )
        assert first == second


class TestMatchWorkerConnectorRandomSeat:
    """The random-seat knobs ride the caps dict to the worker (FR-021)."""

    def test_both_properties_reach_the_command_line(self, tmp_path, stub_classpath):
        import argparse

        from sealed.infrastructure.cli import _effect_collection_caps

        caps = _effect_collection_caps(argparse.Namespace(
            random_seat_share=0.125, random_seat_probability=0.25,
        ))
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    records=WorkerRecords(directory=tmp_path / "records", collection_caps=caps),
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert "-Deffect.random.seat.share=0.125" in cmd
        assert "-Deffect.random.seat.probability=0.25" in cmd


class TestTheStopProtocolProperties:
    """The worker reads its stop file and recycle time from these two
    properties (``WorkerStop``); pinned to the Java names by
    ``tests/unit/infrastructure/test_worker_stop_protocol.py``."""

    def test_both_reach_the_command(self, tmp_path, stub_classpath):
        stop = tmp_path / "logs" / "run.0.stop"
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            MatchWorkerConnector().start(
                launch=WorkerLaunch(
                    None,
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    records=WorkerRecords(directory=tmp_path),
                    stop=WorkerStopProtocol(stop_file=stop, recycle_seconds=600),
                ),
            )
        cmd = mock_popen.call_args[0][0]
        assert f"-Dworker.stop.file={stop}" in cmd
        assert "-Dworker.recycle.seconds=600" in cmd

    def test_neither_by_default(self, tmp_path, stub_classpath):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            MatchWorkerConnector().start(launch=_plain(tmp_path / "o.txt"))
        cmd = mock_popen.call_args[0][0]
        assert not any(arg.startswith("-Dworker.") for arg in cmd)


class TestAWorkerUnderTheStopProtocolHasAProcessGroupOfItsOwn:
    """The console's Ctrl-C must reach the supervisor alone, which asks each
    worker to stop at its game's end; a worker in the console's group would
    close its shard at once, its game cut (FR-058i)."""

    def test_a_worker_given_a_stop_file_starts_in_a_new_process_group(
        self, tmp_path, stub_classpath,
    ):
        import os
        import subprocess

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            MatchWorkerConnector().start(
                launch=WorkerLaunch(
                    None,
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    records=WorkerRecords(directory=tmp_path),
                    stop=WorkerStopProtocol(stop_file=tmp_path / "run.0.stop", recycle_seconds=600),
                ),
            )
        options = mock_popen.call_args.kwargs
        if os.name == "nt":
            assert options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            assert options["start_new_session"] is True

    def test_one_without_keeps_the_console_s_group(self, tmp_path, stub_classpath):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            MatchWorkerConnector().start(launch=_plain(tmp_path / "o.txt"))
        options = mock_popen.call_args.kwargs
        assert "creationflags" not in options
        assert options["start_new_session"] is True


class TestEveryPropertyTheWorkerReadsIsOneTheLauncherWrites:
    """The worker's own property names (``MatchWorkerMain``, ``HoldoutCards``,
    ``WorkerStop``) beside the caps (pinned in ``test_collection_caps.py``):
    a launch with every option set writes each of them, and nothing else."""

    @staticmethod
    def _java_properties() -> set[str]:
        import re

        from tests.unit.forge_sources import connector_source

        found: set[str] = set()
        for source in ("MatchWorkerMain.java", "HoldoutCards.java", "WorkerStop.java"):
            found |= set(re.findall(r'static final String [A-Z_]+ = "([a-z]+(?:\.[a-z]+)+)";',
                                    connector_source(source)))
        return found

    def test_a_full_launch_writes_exactly_the_worker_s_properties(self, tmp_path, stub_classpath):
        connector = MatchWorkerConnector()
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            connector.start(
                launch=WorkerLaunch(
                    tmp_path / "outcomes.txt",
                    run_id=RUN_ID,
                    best_of=BEST_OF,
                    decks=WorkerDecks(
                        side_a=tmp_path / "a.txt",
                        side_b=tmp_path / "b.txt",
                        side_b_weight=6,
                        exclude_cards=tmp_path / "holdout.txt",
                        decks_only=True,
                    ),
                    records=WorkerRecords(
                        directory=tmp_path / "records",
                        worker_index=4,
                        variant_scripts=tmp_path / "variants",
                    ),
                    stop=WorkerStopProtocol(stop_file=tmp_path / "w.stop", recycle_seconds=600),
                    progress_file=tmp_path / "progress.txt",
                ),
            )
        cmd = mock_popen.call_args[0][0]
        written = {arg[2:].split("=", 1)[0] for arg in cmd if arg.startswith("-D")}

        java = self._java_properties()
        assert {"output.file", "sealed.exclude.cards", "effect.variant.scripts",
                "worker.stop.file", "worker.recycle.seconds"} <= java
        assert written == java
        assert f"-Dsealed.exclude.cards={tmp_path / 'holdout.txt'}" in cmd
        assert f"-Deffect.variant.scripts={tmp_path / 'variants'}" in cmd
