"""``--allow-degraded`` on the effects collectors (finding 11.1, task K01c):
off by default, it reaches the collector's config; a run refused for its
workers' engine exits 1 with the reason in ``run.json``'s outcome."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from effects.infrastructure.cli import build_parser, run_collect_coverage, run_collect_variants
from price_predictor.domain.worker_status import WorkerEngineRefused
from price_predictor.infrastructure.cli_run_manifest import REFUSAL_DEST
from tests.unit.conversion_fixtures import STAND_IN_CONVERSION, TREES_CONVERSION

COLLECTORS = pytest.mark.parametrize(("command", "module", "runner"), [
    ("collect-coverage", "effects.application.collect_coverage.run", run_collect_coverage),
    ("collect-variants", "effects.application.collect_variants.run", run_collect_variants),
])


def parse(command: str, *argv: str):
    extra = ["--variant-scripts", "variants"] if command == "collect-variants" else []
    return build_parser().parse_args([
        command, "--effect-records", "records", "--exclude-cards", "h.txt", *extra, *argv])


@pytest.fixture(autouse=True)
def _workers_could_start():
    with (
        patch("price_predictor.infrastructure.forge_jvm.check_worker_prerequisites"),
        patch(TREES_CONVERSION, return_value=STAND_IN_CONVERSION),
    ):
        yield


@COLLECTORS
def test_the_flag_is_off_by_default_and_reaches_the_config(command, module, runner):
    allowed = []
    for extra in ((), ("--allow-degraded",)):
        with patch(module, return_value=0) as collect:
            assert runner(parse(command, *extra)) == 0
        allowed.append(collect.call_args.args[0].allow_degraded)

    assert allowed == [False, True]


@COLLECTORS
def test_a_refused_engine_exits_1_and_its_reason_reaches_the_outcome(
    command, module, runner, caplog,
):
    args = parse(command)
    with patch(module, side_effect=WorkerEngineRefused("the engine is not whole: mode degraded")):
        assert runner(args) == 1

    assert getattr(args, REFUSAL_DEST) == "the engine is not whole: mode degraded"
    assert "the engine is not whole" in caplog.text
