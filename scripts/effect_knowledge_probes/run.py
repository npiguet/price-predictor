"""Probe one checkpoint's game knowledge, and how much of it lives in ``e``.

    # once per curated corpus, before any training
    python scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus DIR
    # then per checkpoint trained on that corpus
    python scripts/effect_knowledge_probes/run.py --checkpoint PATH

``--freeze-probe-set`` freezes the corpus's selection set
(``effects.application.freeze_selection_set``) from the corpus alone, with no
checkpoint: per validation stratum and half, the rung-3 probe items
(``labels.record_items``), the damage, combat and evasion grids' records
(``selection``), the rule tests and the deviance sample, written to
``selection-set-<corpus digest>.json`` under ``--probe-set-dir`` (default
``--reports-dir``). It exits 1, naming stratum, half and component, when the
corpus cannot meet the precision targets.

A probing run refuses without the selection set frozen for its checkpoint's
corpus, reads it, and writes one scorecard recording its digest. Every
board-dependent probe reads the records of the set's halves (``halves``), each
record once, and is reported per stratum and half (FR-070, FR-074):

* the read-out ladder over the frozen probe items (``ladder``), and each probe
  type's rung-3 score on the control score's scale;
* the frozen grids — damage, combat with one and with several blockers,
  evasion with its protection rows — scored as the control score scores them;
* the frozen rule tests, scored through the control score's own scorer
  (``halves.score_rule_tests``);
* the turn and step conditions, anthem and target-legality probes;
* the five board sweeps over the reporting half's records (``sweeps``).

Beside them, reported only and never on a stratum's scale (research E5): the
line and pair ladders, labelled from the scripts and from what the probe
games' and the first ``--profile-shards`` training shards' records were seen
to do, each record once though the training stratum repeats its copies; the
ablation of ``e`` (``ablation``); the board-abilities ablation (``board``);
and optionally every trunk layer (``--per-layer``) and a shallow trunk trained
on frozen ``e`` (``--method-c``).

A probing run reads gen-3 records only, and loads only a ``gen-3`` checkpoint.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The suite's entry points, which callers and tests reach through this script.
from effect_knowledge_probes import (  # noqa: E402
    common,
    halves,
    labels,
    ladder,  # noqa: E402, F401
)
from effect_knowledge_probes.freeze_run import freeze_selection  # noqa: E402
from effect_knowledge_probes.line_labelling import ObservedOutcomes  # noqa: E402, F401
from effect_knowledge_probes.line_ladders import (  # noqa: E402, F401
    MIN_LADDER_ITEMS,
    line_split,
)
from effect_knowledge_probes.method_c_probe import method_c  # noqa: E402, F401
from effect_knowledge_probes.pair_ladders import pair_ladder  # noqa: E402, F401
from effect_knowledge_probes.probe_stages import HALF_CHOICES  # noqa: E402
from effect_knowledge_probes.probing import probe  # noqa: E402
from effect_knowledge_probes.tables import write_tables  # noqa: E402
from price_predictor.infrastructure.run_manifest import run_script  # noqa: E402

#: Training shards read for the observed-outcome line labels.
PROFILE_SHARDS = 4
#: Validation-sample records per stratum the ablation reads.
ABLATION_RECORDS = 1024
METHOD_C_STEPS = 2000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        help="the checkpoint to probe; refused with --freeze-probe-set")
    parser.add_argument("--corpus", type=Path,
                        help="curated corpus; default the checkpoint's recorded one, "
                             "required with --freeze-probe-set")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="sidecar tree (repeatable); default the trees the "
                             "checkpoint recorded")
    parser.add_argument("--abilities-root", type=Path, default=common.DEFAULT_ABILITIES)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--keyword-definitions", type=Path)
    parser.add_argument("--freeze-probe-set", action="store_true")
    parser.add_argument("--families", type=lambda s: [f.strip() for f in s.split(",") if f],
                        help=f"comma list from: {', '.join(labels.FAMILIES)}")
    parser.add_argument("--half", choices=tuple(HALF_CHOICES), default="both",
                        help="the selection set's halves read and reported")
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--method-c", action="store_true")
    parser.add_argument("--method-c-layers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--method-c-steps", type=int, default=METHOD_C_STEPS)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-board", action="store_true",
                        help="skip the board-abilities ablation (board.py)")
    parser.add_argument("--no-gen3-probes", action="store_true",
                        help="skip the turn and step conditions, anthem and "
                             "target-legality probes")
    parser.add_argument("--reports-dir", type=Path, default=common.REPORTS,
                        help="where the dated report directory goes unless --output "
                             "is given, and the selection set unless --probe-set-dir is")
    parser.add_argument("--output", type=Path, default=None,
                        help="the report directory itself, its run.json included "
                             "(default: knowledge-probes-<checkpoint>-<date>/ under "
                             "--reports-dir); refused with --freeze-probe-set")
    parser.add_argument("--probe-set-dir", type=Path, default=None,
                        help="where the selection set is written and read "
                             "(default: --reports-dir); every arm of a sweep "
                             "shares one set while its reports go beside its run")
    parser.add_argument("--probe-records", type=int, default=halves.PROBE_RECORDS,
                        help="qualifying records each gen-3 probe reads per half")
    parser.add_argument("--probe-per-game", type=int, default=halves.PROBE_PER_GAME)
    parser.add_argument("--profile-shards", type=int, default=PROFILE_SHARDS)
    parser.add_argument("--ablation-records", type=int, default=ABLATION_RECORDS)
    parser.add_argument("--board-records", type=int, default=None,
                        help="records per board subset per stratum (default board.py's)")
    parser.add_argument("--board-shards", type=int, default=None,
                        help="validation shards read per stratum for the board subsets")
    return parser


def _check_freeze_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.checkpoint is not None:
        parser.error("--freeze-probe-set freezes from the corpus alone, before any "
                     "model exists: drop --checkpoint")
    if args.corpus is None:
        parser.error("--freeze-probe-set needs --corpus DIR")
    if args.output is not None:
        parser.error("--freeze-probe-set writes the selection set into "
                     "--probe-set-dir (or --reports-dir): drop --output")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    unknown = set(args.families or ()) - set(labels.FAMILIES)
    if unknown:
        parser.error(f"unknown families: {sorted(unknown)}")
    if args.freeze_probe_set:
        _check_freeze_args(parser, args)
    elif args.checkpoint is None:
        parser.error("--checkpoint is required (or --freeze-probe-set --corpus DIR)")
    return args


def report_dir(reports_dir: Path, checkpoint: Path, date: str) -> Path:
    label = common.checkpoint_label(checkpoint)
    return Path(reports_dir) / f"knowledge-probes-{label}-{date.replace('-', '')}"


def output_dir(args: argparse.Namespace, date: str) -> Path:
    """A probing run's report directory: ``--output``, else the one dated
    ``date`` under ``--reports-dir``."""
    if args.output is not None:
        return Path(args.output)
    return report_dir(args.reports_dir, args.checkpoint, date)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.freeze_probe_set:
        # Its output is the selection set, in the probe-set directory.
        return run_script(
            args.probe_set_dir or args.reports_dir, lambda: freeze_selection(args),
            script=__file__, argv=argv, inputs=[args.corpus, *(args.cards_folder or ())],
        )
    # The report directory is named for today, as the scorecard is dated.
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return run_script(
        output_dir(args, today), lambda: _probe_run(args),
        script=__file__, argv=argv,
        inputs=[
            args.checkpoint, args.corpus, *(args.cards_folder or ()), args.abilities_root,
            args.vocab_path, args.keyword_definitions, args.probe_set_dir or args.reports_dir,
        ],
    )


def _probe_run(args: argparse.Namespace) -> int:
    from effects.domain.shard_generation import UnacceptedGenerationError

    context = common.probe_context(
        args.checkpoint, corpus=args.corpus, cards_folders=args.cards_folder)
    selection = halves.load_selection(context, args.probe_set_dir or args.reports_dir)
    try:
        scorecard = probe(args, context, selection)
    except UnacceptedGenerationError as error:
        raise SystemExit(str(error)) from None
    out_dir = output_dir(args, scorecard.date)
    common.write_json(out_dir / "scorecard.json", scorecard.to_json())
    write_tables(out_dir, scorecard)
    peak = scorecard.runtime["peak_gpu_bytes"] / 2**30
    print(f"scorecard -> {out_dir / 'scorecard.json'} "
          f"({scorecard.runtime['wall_seconds']:.0f} s, peak GPU {peak:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
