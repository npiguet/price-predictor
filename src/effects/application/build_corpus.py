"""Build a fixed training corpus and two fixed validation strata (FR-135).

The survey (:mod:`effects.application.corpus_survey`), the decisions between
it and the write pass (:mod:`effects.application.corpus_decisions`), the write
pass (:mod:`effects.application.corpus_write`), the repack, the validation
samples and the manifest, in that order, and every refusal that stops a build
before ``--output`` is touched. ``rules/heads/`` and ``rules/scenarios/`` are
read once, here (FR-060a).
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.build_corpus_config import BuildCorpusConfig
from effects.application.corpus_decisions import Decisions, TextJoin, decide
from effects.application.corpus_report import (
    BuildOutcome,
    build_manifest,
    check_refusals,
    check_samples,
    log_report,
    report_write_pass,
)
from effects.application.corpus_rule_catalogue import RuleCatalogue
from effects.application.corpus_survey import (
    ReadConfig,
    Survey,
    SurveyConfig,
    WorkerContext,
    run_survey,
)
from effects.application.corpus_write import WriteConfig, WriteResult, run_write_pass
from effects.domain.ability_tokenizer import Surface
from effects.domain.corpus_manifest import CuratedShard
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import SourceTree, split_script_file
from effects.domain.records import acting_keys
from effects.domain.shard_generation import UnacceptedGenerationError
from effects.domain.split import HeldOutCards
from effects.domain.strata import OUTPUT_STRATA
from effects.domain.stratum_placement import rule_test_misses

if TYPE_CHECKING:
    from effects.infrastructure.build_inputs import KeywordTable
    from effects.infrastructure.converted_trees import ConvertedTrees
    from effects.infrastructure.corpus_store import CorpusStore

__all__ = ["build"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Sources:
    """What a build reads before its survey: the rules catalogue, the
    converted trees, the holdout over them, the encoding surface and the
    keyword-definition table (None without one)."""

    rules: RuleCatalogue
    trees: ConvertedTrees
    held_out: HeldOutCards
    surface: Surface
    keywords: KeywordTable | None

    @property
    def join(self) -> TextJoin:
        """The key-to-text join over the trees' sidecars."""
        return TextJoin(self.trees.sidecars, self.surface, self.keywords)


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code:
    1, after the corpus and its manifest are written, when a rule-test
    component already misses its precision target (research E5)."""
    from effects.infrastructure.build_inputs import read_keyword_table
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.vocabulary_file import vocabulary_surface

    store = CorpusStore(config.output)
    if config.verify:
        return _verify(config, store)

    _require_generation(config.records_dir)
    rules = _read_rules(config.rules_dir)
    roots = config.source_trees.sidecar_roots()
    trees = _converted_trees(config, roots)
    held_out = _select_holdout(config, trees)
    surface = vocabulary_surface(config.vocab_path)
    _log_holdout(config, held_out, surface)
    read = ReadConfig(
        records_dir=config.records_dir, sidecar_roots=roots,
        max_events=config.max_events_per_record, held_out=held_out, seed=config.seed,
        catalogue=rules.scenarios,
        rule_outputs={rule: tuple(entry.outputs) for rule, entry in rules.heads.items()},
    )
    keywords = read_keyword_table(config.keyword_definitions)
    with _unreadable_shards("nothing under --output was touched"):
        survey = _survey(config, read, rules, keywords)
    sources = _Sources(rules, trees, held_out, surface, keywords)
    decisions = _decide_or_refuse(config, survey, sources)
    _warn_on_resized_shards(config.records_dir, survey)

    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    with _unreadable_shards(
            "the shard changed after the survey read it whole; the dataset under "
            "--output was already cleared (its manifest is manifest.previous.json)"):
        written = run_write_pass(
            decisions.write_tasks(survey),
            config=WriteConfig(read=read, plan=decisions.write_plan(store)),
            workers=config.workers,
        )
    report_write_pass(survey, written)
    shards = _repack_outputs(store, written, shard_records=config.shard_records, seed=config.seed)
    _write_validation_samples(config, store, WorkerContext(read), shards)

    manifest = replace(build_manifest(config, BuildOutcome(
        surface=surface, held_out=held_out, keywords=keywords, survey=survey,
        decisions=decisions, written=written,
    )), shards=shards)
    store.save(manifest)
    log_report(manifest, survey, decisions, written)
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.training.kept.values()), store.directory, manifest.digest(),
    )
    return _precision_exit(config, decisions)


def _precision_exit(config: BuildCorpusConfig, decisions: Decisions) -> int:
    """1, naming each stratum, half and component, when a rule-test component
    already misses its precision target (FR-068b); 0 otherwise.

    Only the rule tests can be told here: the probes, grids and explained
    deviance are measured when the selection set is frozen.
    """
    misses = rule_test_misses(decisions.strata, decisions.inventories, config.placement)
    for miss in misses:
        logger.error("precision target missed — %s", miss.describe(config.max_component_error))
    if misses:
        logger.error(
            "%d half(s) miss the rule-test precision target; the corpus and its "
            "manifest are written and readable (research E5).", len(misses),
        )
        return 1
    return 0


def _log_holdout(config: BuildCorpusConfig, held_out: HeldOutCards, surface: Surface) -> None:
    logger.info(
        "Holdout: %d ability text(s) on %d card(s) and %d token "
        "script(s); encoding surface %r.",
        len(held_out.texts), len(held_out.names),
        sum(
            1 for script in held_out.script_files
            if split_script_file(script)[0] == SourceTree.TOKENSCRIPTS
        ),
        surface,
    )


def _verify(config: BuildCorpusConfig, store: CorpusStore) -> int:
    """``--verify``: the manifest's drift against the corpus on disk; writes nothing."""
    from effects.infrastructure.shard_io import list_shards

    records_dir = config.records_dir
    manifest = store.load()
    added, removed, resized = manifest.drift(list_shards(records_dir))
    drift = (("added", added), ("removed", removed), ("resized", resized))
    for label, names in drift:
        if names:
            logger.warning(
                "%d shard(s) %s since this dataset was built: %s%s",
                len(names), label, ", ".join(names[:5]),
                ", …" if len(names) > 5 else "",
            )
    if any(names for _, names in drift):
        logger.warning("Rebuild with `python -m effects build-corpus`.")
        return 1
    logger.info("Dataset is current against %s.", records_dir)
    return 0


def _converted_trees(config: BuildCorpusConfig, roots: dict[str, Path]) -> ConvertedTrees:
    """The converted trees, the variant tree among them when one is named.

    Read through this process's own cache over ``roots``, the one every
    :class:`WorkerContext` here reads too: the holdout, the fold of keys to
    texts, the validation samples and an in-process pass share one cache
    rather than each reading every sidecar again.
    """
    from effects.infrastructure.converted_trees import process_sidecars

    return config.source_trees.converted(sidecars=process_sidecars(roots))


def _select_holdout(config: BuildCorpusConfig, trees: ConvertedTrees) -> HeldOutCards:
    """The trainer's own holdout rule over the converted trees; refuses an empty one."""
    held_out = trees.holdout(config.holdout)
    if not held_out.names:
        tried = ", ".join(
            f"{folder} ({'found' if Path(folder).is_dir() else 'missing'})"
            for folder in config.cards_folders
        )
        raise BuildCorpusError(
            "Nothing is held out, so the card-disjoint stratum would be empty "
            "and gate 1 would have nothing to measure at all. Cards folders "
            f"tried: {tried}; {len(trees.card_files)} converted card(s) read, "
            f"{len(held_out.texts)} held-out ability text(s), no held-out card. "
            "Run from the repository root so the relative --cards-folder paths "
            "resolve, raise --holdout-permille, or check that the converted "
            "tree has sidecars with script text."
        )
    return held_out


def _require_generation(records_dir: Path) -> None:
    """Gen-3 only, before any record is read (FR-018, FR-020)."""
    from effects.infrastructure.shard_io import iter_shards, require_generation

    with _unreadable_shards("nothing under --output was touched"):
        require_generation(iter_shards(records_dir))


@contextlib.contextmanager
def _unreadable_shards(consequence: str) -> Iterator[None]:
    """A shard that cannot be read whole (a corrupt gzip member before the
    last, a line that does not decode, a record of an older generation or a
    ``generation`` naming none) makes the dataset unusable: the build stops
    with exit 1, naming the shard and line the error names, and saying what
    became of ``--output``."""
    from effects.infrastructure.shard_io import SHARD_DAMAGE

    try:
        yield
    except (*SHARD_DAMAGE, UnacceptedGenerationError) as exc:
        raise BuildCorpusError(f"a raw shard cannot be read: {exc}; {consequence}.") from exc


def _read_rules(rules_dir: Path) -> RuleCatalogue:
    """The rules catalogue, refused when it holds no scenario or no heads line:
    a build over it would tag and budget nothing and fail only at the
    precision exit, after replacing the dataset (FR-060a)."""
    rules = RuleCatalogue.read(rules_dir)
    if not rules.scenarios.scenarios or not rules.heads:
        raise BuildCorpusError(
            f"the rules catalogue under {Path(rules_dir).resolve()} holds "
            f"{len(rules.scenarios.scenarios)} scenario(s) and {len(rules.heads)} heads "
            "line(s): a build reads rules/scenarios/*.json and rules/heads/*.jsonl, "
            "relative to the working directory, and tags and budgets nothing without "
            "them. Run from the repository root."
        )
    return rules


def _warn_on_resized_shards(records_dir: Path, survey: Survey) -> None:
    """Shards whose size moved since the survey read them, every one named:
    the write pass writes none of the records past those the survey read
    (``dropped-unsurveyed``)."""
    def size(shard) -> int | None:
        path = Path(records_dir) / shard.name
        return path.stat().st_size if path.exists() else None

    moved = [shard.name for shard in survey.shards if size(shard) != shard.size]
    if moved:
        logger.warning(
            "%d raw shard(s) changed size since the survey read them (a collection "
            "still writing under --records-dir?): %s. The records past those the "
            "survey read are written to no stratum.", len(moved), ", ".join(moved))


def _known_keywords(keywords: KeywordTable | None) -> frozenset[str]:
    """Forge's keywords for the closed keyword list; none without the table,
    which leaves the overlay keywords alone."""
    if keywords is None:
        return frozenset()
    return frozenset(normalize_keyword(name) for name in keywords.names)


def _survey(
    config: BuildCorpusConfig, read: ReadConfig, rules: RuleCatalogue,
    keywords: KeywordTable | None,
) -> Survey:
    """The parallel survey, logged."""
    survey = run_survey(
        config.records_dir,
        config=SurveyConfig(
            read=read, heap_cap=config.heap_cap, placement=config.placement,
            game_disjoint_keywords=tuple(
                normalize_keyword(name) for name in config.game_disjoint_keywords
            ),
            budgeted=rules.budgeted,
            known_keywords=_known_keywords(keywords),
        ),
        workers=config.workers,
    )
    totals = survey.totals
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card, "
        "%d have a random seat; %d game(s) are placement candidates; %d "
        "turn-action record(s) are written nowhere; %d record(s) act through a "
        "held-out text in a game naming no held-out card.",
        totals.records, len(totals.games.games), len(totals.games.held_out),
        len(totals.games.random_seat), len({r.game for r in totals.candidates}),
        sum(totals.turn_actions.values()), sum(totals.held_out_text_dropped.values()),
    )
    return survey


def _refuse_every_game_cut(config: BuildCorpusConfig, survey: Survey) -> None:
    """No game ended by its marker while some were read: the collection wrote
    no marker at all (a connector older than the marker, or every worker
    killed), and every game would be dropped as cut."""
    cut = survey.totals.cut.games
    if cut and not survey.totals.games.games:
        raise BuildCorpusError(
            f"none of the {len(cut)} gen-3 game(s) under {config.records_dir} has a "
            "game-end marker (turn_action / game_end) in its shard, so every game is "
            "dropped as cut: the collection wrote no marker (a connector built before "
            "the marker existed?) or no game ended before its worker was killed.")


def _refuse_unconfigured_trees(config: BuildCorpusConfig, survey: Survey) -> None:
    """A records set naming a source tree this run has no root for (variant
    records without ``--variant-scripts``): its keys have no text to join, so
    the dataset would train on records without an acting text and place their
    games blind to a variant printing a held-out text. Nothing under
    ``--output`` has been touched."""
    unconfigured = survey.totals.reading.unconfigured
    if not unconfigured:
        return
    named = ", ".join(f"{tree!r} ({n} key read(s))" for tree, n in sorted(unconfigured.items()))
    raise BuildCorpusError(
        f"{config.records_dir} holds records naming the source tree(s) {named}, "
        "and this run has no converted root for them: pass --variant-scripts for "
        "the variant tree, or --cards-folder for a card or token tree. Nothing "
        "under --output was touched.")


def _decide_or_refuse(config: BuildCorpusConfig, survey: Survey, sources: _Sources) -> Decisions:
    """:func:`decide`, refusing a corpus the quality rules reject, one whose
    every game was cut, or a dataset gate 1 could not be scored on."""
    _refuse_every_game_cut(config, survey)
    _refuse_unconfigured_trees(config, survey)
    check_refusals(survey.totals.refusals, survey.totals.kept)
    decisions = decide(survey, join=sources.join, config=config, rules=sources.rules)
    if not decisions.strata.card_disjoint:
        raise BuildCorpusError(
            "No game is admitted to the card-disjoint stratum, so the dataset "
            "cannot score gate 1 and the trainer would refuse it in its first "
            f"minute. {config.records_dir} holds {len(survey.totals.games.games)} game(s), "
            f"{len(survey.totals.games.held_out)} of which name one of the "
            f"{len(sources.held_out.names)} held-out card(s). Depleted and "
            "full-strength shards live in sibling subdirectories of the raw "
            "corpus root: point --records-dir at the parent of both, and check "
            "that a full-strength run — pools generated without "
            "--exclude-cards — has actually been collected. A held-out game is "
            "admitted only while every held-out ability text it resolves is "
            "under --card-disjoint-text-cap, and one that resolves none is "
            "never admitted."
        )
    _refuse_an_unsampled_game_disjoint(config, decisions)
    return decisions


def _refuse_an_unsampled_game_disjoint(config: BuildCorpusConfig, decisions: Decisions) -> None:
    """An empty game-disjoint stratum leaves its validation sample empty, which
    the trainer validates on: refused here, before the old dataset is cleared,
    rather than when the sample comes out empty."""
    if config.validation_sample <= 0 or decisions.strata.game_disjoint:
        return
    raise BuildCorpusError(
        "No game is placed in the game-disjoint stratum, so its validation "
        f"sample would be empty although --validation-sample {config.validation_sample} "
        "asks for one. Raise --game-disjoint-share or --game-disjoint-keyword-share, "
        "or pass --validation-sample 0. Nothing under --output was touched."
    )


def _write_validation_samples(
    config: BuildCorpusConfig, store: CorpusStore, context: WorkerContext,
    shards: dict[str, tuple[CuratedShard, ...]],
) -> None:
    if config.validation_sample <= 0:
        logger.info("no validation sample written (--validation-sample 0)")
        return
    from effects.application.validation_samples import draw_samples, write_samples

    samples = draw_samples(
        {stratum: tuple(store.output_dir(stratum) / shard.name for shard in written)
         for stratum, written in shards.items()},
        mix=config.mix(), size=config.validation_sample, seed=config.seed,
        held_out_text_of=lambda record: context.held_out_text(acting_keys(record)),
    )
    sampled = write_samples(store, samples)
    for stratum, count in sampled.items():
        logger.info("%-22s %9d record(s) sampled for validation", stratum, count)
    check_samples(sampled, size=config.validation_sample)


def _repack_outputs(
    store: CorpusStore, written: WriteResult, *, shard_records: int, seed: int,
) -> dict[str, tuple[CuratedShard, ...]]:
    """Deal each stratum's parts into uniform, mixed shards, then drop the
    parts; each stratum's shards by name, with the records written into each.

    A part is one source shard, so unrepacked the trainer's per-shard step
    budget would be uneven and a shard's batches correlated by run.
    """
    from effects.infrastructure.shard_io import iter_shards, repack_shards

    listing: dict[str, tuple[CuratedShard, ...]] = {}
    for stratum in OUTPUT_STRATA:
        parts = iter_shards(store.parts_dir_for(stratum))
        paths = repack_shards(
            parts, store.output_dir(stratum), shard_records=shard_records,
            seed=seed, records=written.strata.records.get(stratum, 0),
        )
        logger.info(
            "%-22s %d part(s) repacked into %d shard(s)",
            stratum, len(parts), len(paths),
        )
        listing[str(stratum)] = tuple(sorted(
            (CuratedShard(name=path.name, records=records) for path, records in paths.items()),
            key=lambda shard: shard.name,
        ))
    shutil.rmtree(store.parts_dir, ignore_errors=True)
    return listing
