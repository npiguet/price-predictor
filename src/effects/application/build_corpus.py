"""Build a fixed training corpus and two fixed validation strata (FR-135).

Between the survey (:mod:`effects.application.corpus_survey`) and the write
pass (:mod:`effects.application.corpus_write`), this module folds provenance
keys to ability texts and decides the split and the plan.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.corpus_report import (
    build_manifest,
    check_refusals,
    check_samples,
    log_report,
    report_write_pass,
)
from effects.application.corpus_survey import (
    ReadConfig,
    Survey,
    SurveyConfig,
    WorkerContext,
    run_survey,
)
from effects.application.corpus_write import WriteConfig, WritePlan, WriteResult, run_write_pass
from effects.domain import default_paths
from effects.domain.ability_tokenizer import Surface
from effects.domain.corpus_plan import (
    BuildCorpusError,
    GameRouting,
    GameStrata,
    Keys,
    SelectionPlan,
    admit_card_disjoint,
    members_of,
    plan_selection,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey, SourceTree, first_text, split_script_file
from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
from effects.domain.records import acting_keys
from effects.domain.sampling import DEFAULT_KIND_MIX
from effects.domain.split import HeldOutCards, HoldoutRule
from effects.domain.strata import OUTPUT_STRATA

if TYPE_CHECKING:
    from effects.infrastructure.converted_trees import ConvertedTrees
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

__all__ = [
    "DEFAULT_GAME_DISJOINT_KEYWORDS", "BuildCorpusConfig", "BuildCorpusError",
    "Decisions", "build", "decide",
]

logger = logging.getLogger(__name__)


#: FR-045's default keyword list, in the spelling an operator types.
DEFAULT_GAME_DISJOINT_KEYWORDS: tuple[str, ...] = (
    "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
)


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus).

    ``help`` metadata is the flag's help, so a default lives here alone.
    """

    records_dir: Path
    output: Path = Path(default_paths.DEFAULT_CORPUS_OUTPUT)
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    vocab_path: str = default_paths.DEFAULT_VOCAB_PATH
    variant_scripts: str | None = None
    holdout: HoldoutRule = HoldoutRule()
    text_cap: int = field(default=200, metadata={"help": (
        "Max written training records per ability text per selection cell, "
        "repeats included; 0 means no cap (default: %(default)s)"
    )})
    reuse_cap: int = field(default=4, metadata={"help": (
        "Max copies of one record toward a short rule family's share (FR-049); "
        "1 means no repeats (default: %(default)s)"
    )})
    class_mix: dict[str, float] | None = None
    training_records: int = field(default=0, metadata={"help": (
        "Total training records after capping, subsampled per class; 0 means "
        "no ceiling (default: %(default)s)"
    )})
    game_disjoint_share: float = field(default=0.01, metadata={"help": (
        "A game naming no held-out card enters the game-disjoint stratum when "
        "crc32(game_id) mod 1e6 / 1e6 is below this, so a rebuild over a grown "
        "corpus keeps every game it placed (default: %(default)s)"
    )})
    game_disjoint_keyword_share: float = field(default=0.15, metadata={"help": (
        "The threshold instead for a game holding a qualifying combat for one "
        "of --game-disjoint-keywords (default: %(default)s)"
    )})
    game_disjoint_keywords: tuple[str, ...] = DEFAULT_GAME_DISJOINT_KEYWORDS
    #: Read by FR-015's check; None or a missing file skips it, since nothing
    #: can expand without a table.
    keyword_definitions: Path | None = field(
        default=Path(default_paths.DEFAULT_KEYWORD_DEFINITIONS), metadata={"help": (
            "The keyword table the tokenizer expands with; the build fails when "
            "a display-name match would expand a non-keyword line "
            "(default: %(default)s)"
        )},
    )
    card_disjoint_text_cap: int = field(default=50, metadata={"help": (
        "Max card-disjoint games admitted per held-out ability text, so each "
        "weighs comparably in gate 1; 0 means no cap (default: %(default)s)"
    )})
    shard_records: int = field(default=2000, metadata={"help": (
        "Records per output shard: whole games are dealt to shards by a seeded "
        "hash of their game id; 0 keeps one output shard per source shard "
        "(default: %(default)s)"
    )})
    max_events: int = MAX_EVENTS_PER_RECORD
    validation_sample: int = field(default=2048, metadata={"help": (
        "Records per stratum in validation/samples/, mixture-matched; 0 writes "
        "no sample (default: %(default)s)"
    )})
    seed: int = field(default=42, metadata={"help": (
        "Seeds the selection and the repacking (default: %(default)s)"
    )})
    workers: int | None = field(default=None, metadata={"help": (
        "Parallel shard readers; unset uses the CPU count"
    )})
    verify: bool = field(default=False, metadata={"help": (
        "Report drift against an existing manifest and write nothing"
    )})

    def __post_init__(self) -> None:
        self._refuse_negative_counts()
        if self.reuse_cap < 1:
            raise BuildCorpusError(
                f"--reuse-cap is {self.reuse_cap}; it is the most copies of one "
                "record a short family may write, and 1 already means no repeats."
            )
        for flag, share in (
            ("--game-disjoint-share", self.game_disjoint_share),
            ("--game-disjoint-keyword-share", self.game_disjoint_keyword_share),
        ):
            if not 0.0 <= share <= 1.0:
                raise BuildCorpusError(f"{flag} is {share}; a share is between 0 and 1.")
        self._refuse_unscored_keywords()

    def _refuse_negative_counts(self) -> None:
        """Refuse a negative count; zero is a real setting for each.

        - ``--text-cap 0`` — no per-text cap; ``--reuse-cap`` alone bounds a
          text's quota in a cell.
        - ``--card-disjoint-text-cap 0`` — every held-out game carrying a
          held-out text is admitted.
        - ``--game-disjoint-share 0`` — no game-disjoint stratum, unless the
          keyword share places some games.
        - ``--training-records 0`` — no ceiling beyond ``--text-cap``.
        - ``--shard-records 0`` — no repacking; one output shard per source
          shard.
        - ``--max-events-per-record 0`` — no event-flood rule; the other two
          quality rules stay on.
        - ``--validation-sample 0`` — nothing written under
          ``validation/samples/``.
        """
        for flag, value in (
            ("--text-cap", self.text_cap),
            ("--card-disjoint-text-cap", self.card_disjoint_text_cap),
            ("--training-records", self.training_records),
            ("--shard-records", self.shard_records),
            ("--max-events-per-record", self.max_events),
            ("--validation-sample", self.validation_sample),
        ):
            if value < 0:
                raise BuildCorpusError(
                    f"{flag} is {value}; it counts things and cannot be "
                    "negative. Zero is allowed and means no cap, no ceiling, "
                    "or an empty stratum depending on the flag — see --help."
                )

    def _refuse_unscored_keywords(self) -> None:
        unknown = [
            name for name in self.game_disjoint_keywords
            if normalize_keyword(name) not in KEYWORDS_BY_NAME
        ]
        if unknown:
            raise BuildCorpusError(
                f"--game-disjoint-keywords names {', '.join(unknown)}, which gate 2 "
                f"does not score; it scores {', '.join(sorted(KEYWORDS_BY_NAME))}."
            )

    def mix(self) -> dict[str, float]:
        return dict(self.class_mix or DEFAULT_KIND_MIX)

    @property
    def heap_cap(self) -> int:
        """How many smallest hashes the survey keeps per cell and key; every
        one under ``--text-cap 0``."""
        return self.text_cap if self.text_cap > 0 else sys.maxsize

    @property
    def routing(self) -> GameRouting:
        return GameRouting(
            game_disjoint_share=self.game_disjoint_share,
            keyword_share=self.game_disjoint_keyword_share,
        )


@dataclass(frozen=True, slots=True)
class Decisions:
    """Everything the write pass needs and the manifest records about the plan.

    Plans are keyed by cell and key tuple, so a worker applies them without
    folding keys to texts.
    """

    selection: SelectionPlan
    strata: GameStrata
    #: Game-disjoint games placed only because they hold a qualifying combat
    #: for a listed keyword (FR-053).
    keyword_threshold_games: frozenset[str]
    rarity: dict[str, int]
    key_text: dict[Keys, str]
    #: Kept records per legality mode (FR-053's ``legality_mode_counts``).
    mode_counts: dict[str, int]

    def write_plan(self, store: CorpusStore) -> WritePlan:
        """What the write pass applies: these decisions, its parts in ``store``."""
        return WritePlan(
            parts_dirs={
                stratum: str(store.parts_dir_for(stratum)) for stratum in OUTPUT_STRATA
            },
            strata=self.strata,
            plans=self.selection.plans,
            keyless_plans=self.selection.keyless,
            mode_counts=self.mode_counts,
        )


def _keyword_misfires(
    key_text: dict[Keys, str], sidecars: SidecarCache, definitions: Path | None,
) -> list[str]:
    """Texts a display-name keyword match would expand on a non-keyword line
    (FR-015), replacing a script with a reminder template it does not carry."""
    if definitions is None or not Path(definitions).exists():
        return []
    from effects.domain.ability_tokenizer import display_name_of
    from effects.domain.rule_families import KEYWORD_API_TYPE
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

    names = {name.lower() for name in load_keyword_definitions(Path(definitions))}
    found: list[str] = []
    for keys in key_text:
        for key in keys:
            line = sidecars.line_or_none(key)
            if line is None or not line.script_text:
                continue
            if line.script_api_type == KEYWORD_API_TYPE:
                continue
            if display_name_of(line.script_text).lower() in names:
                found.append(f"{key.script_file}: {line.script_text!r}")
    return sorted(set(found))


def decide(
    survey: Survey,
    *,
    sidecars: SidecarCache,
    surface: Surface,
    config: BuildCorpusConfig,
) -> Decisions:
    """Fold keys to texts, split the games, and plan every record's copies.

    ``survey.totals.supply`` must already be legality-resolved, as
    :func:`_survey` leaves it.
    """
    totals = survey.totals
    if totals.supply.heap_cap != config.heap_cap:
        raise ValueError(
            f"the survey kept {totals.supply.heap_cap} hashes per cell and key, "
            f"but config.text_cap {config.text_cap} keeps {config.heap_cap}. "
            "decide() must be run against the Survey that this same "
            "--text-cap produced, or the copy plans it writes would disagree "
            "with what the manifest is about to record."
        )

    from effects.application.ability_text import text_for_key

    def text_of(key: ProvenanceKey) -> str | None:
        # The trainer's own key-to-text join, so the rarity table keys match.
        return text_for_key(key, sidecars, surface)

    key_text: dict[Keys, str] = {}
    text_games: dict[str, set[int]] = {}
    for keys in totals.rarity.key_records:
        text = first_text(keys, text_of)
        if text is None:
            # A runtime-only trait or an unconverted script: no text to cap
            # against, so its records are sampled with the cell's keyless ones.
            continue
        key_text[keys] = text
        text_games.setdefault(text, set()).update(totals.rarity.key_games.get(keys, ()))

    misfires = _keyword_misfires(key_text, sidecars, config.keyword_definitions)
    if misfires:
        raise BuildCorpusError(
            f"{len(misfires)} non-keyword line(s) would expand as a keyword by "
            "display name (FR-015): the text before the first colon matches a "
            "keyword definition while the line's script_api_type is not "
            "Keyword. First: " + "; ".join(misfires[:5])
        )

    held_out = frozenset(totals.games.held_out)
    game_disjoint = frozenset(totals.games.game_disjoint) - held_out
    card_disjoint = admit_card_disjoint(
        held_out, totals.games.held_out_text_games,
        cap=config.card_disjoint_text_cap, seed=config.seed,
    )
    members, keys_of = members_of(totals.supply, key_text)
    return Decisions(
        selection=plan_selection(
            members, keys_of, mix=config.mix(), ceiling=config.training_records,
            text_cap=config.text_cap, reuse_cap=config.reuse_cap,
        ),
        strata=GameStrata(
            card_disjoint=card_disjoint, game_disjoint=game_disjoint, held_out=held_out,
        ),
        keyword_threshold_games=frozenset(
            game for game in game_disjoint if config.routing.under_keyword_threshold(game)
        ),
        rarity={text: len(games) for text, games in text_games.items()},
        key_text=key_text,
        mode_counts=dict(sorted(totals.mode_counts.items())),
    )


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code."""
    from effects.domain.ability_encoder import surface_of
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(config.output)
    if config.verify:
        return _verify(config, store)

    trees = _converted_trees(config)
    held_out = _select_holdout(config, trees)
    surface = surface_of(config.vocab_path)
    _log_holdout(config, held_out, surface)
    _refuse_mixed_generations(Path(config.records_dir))
    read = ReadConfig(
        records_dir=str(config.records_dir), sidecar_roots=trees.root_paths,
        max_events=config.max_events, held_out=held_out, seed=config.seed,
    )
    survey = _survey(config, read)
    decisions = _decide_or_refuse(config, survey, trees, surface, held_out)

    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    written = run_write_pass(
        [shard.name for shard in survey.shards],
        config=WriteConfig(read=read, plan=decisions.write_plan(store)),
        workers=config.workers,
    )
    report_write_pass(survey, written)
    _repack_outputs(store, written, shard_records=config.shard_records, seed=config.seed)
    _write_validation_samples(config, store, WorkerContext(read))

    manifest = build_manifest(
        config, surface=surface, held_out=held_out, survey=survey,
        decisions=decisions, written=written,
    )
    store.save(manifest)
    log_report(manifest, survey, decisions, written)
    logger.info(
        "Wrote %d training record(s) to %s (digest %s).",
        sum(written.training.kept.values()), store.directory, manifest.digest(),
    )
    return 0


def _log_holdout(config: BuildCorpusConfig, held_out: HeldOutCards, surface: Surface) -> None:
    logger.info(
        "Holdout (%s unit): %d ability text(s) on %d card(s) and %d token "
        "script(s); encoding surface %r.",
        config.holdout.unit, len(held_out.texts), len(held_out.names),
        sum(
            1 for script in held_out.script_files
            if split_script_file(script)[0] == SourceTree.TOKENSCRIPTS
        ),
        surface,
    )


def _verify(config: BuildCorpusConfig, store: CorpusStore) -> int:
    """``--verify``: the manifest's drift against the corpus on disk; writes nothing."""
    from effects.infrastructure.corpus_store import current_sources

    records_dir = Path(config.records_dir)
    manifest = store.load()
    added, removed, resized = manifest.drift(current_sources(records_dir))
    for label, names in (("added", added), ("removed", removed), ("resized", resized)):
        if names:
            logger.warning(
                "%d shard(s) %s since this dataset was built: %s%s",
                len(names), label, ", ".join(names[:5]),
                ", …" if len(names) > 5 else "",
            )
    if added or removed or resized:
        logger.warning("Rebuild with `python -m effects build-corpus`.")
        return 1
    logger.info("Dataset is current against %s.", records_dir)
    return 0


def _converted_trees(config: BuildCorpusConfig) -> ConvertedTrees:
    """The converted trees, the variant tree among them when one is named."""
    from effects.infrastructure.converted_trees import ConvertedTrees

    return ConvertedTrees.from_folders(config.cards_folders, config.variant_scripts)


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


def _refuse_mixed_generations(records_dir: Path) -> None:
    """Before any record is read (FR-033)."""
    from effects.infrastructure.record_io import (
        MixedGenerationsError,
        iter_shards,
        refuse_mixed_generations,
    )

    try:
        refuse_mixed_generations(iter_shards(records_dir))
    except MixedGenerationsError as exc:
        raise BuildCorpusError(str(exc)) from exc


def _survey(config: BuildCorpusConfig, read: ReadConfig) -> Survey:
    """The parallel survey, its supply's legality cells folded into families."""
    survey = run_survey(
        Path(config.records_dir),
        config=SurveyConfig(
            read=read, heap_cap=config.heap_cap, routing=config.routing,
            game_disjoint_keywords=tuple(
                normalize_keyword(name) for name in config.game_disjoint_keywords
            ),
        ),
        workers=config.workers,
    )
    totals = survey.totals
    supply = totals.supply
    logger.info(
        "Surveyed %d record(s) in %d game(s); %d game(s) name a held-out card, "
        "%d are game-disjoint; %d record(s) are training-bound, and %d more act "
        "through a held-out text.",
        totals.records, len(totals.games.games), len(totals.games.held_out),
        len(totals.games.game_disjoint),
        sum(supply.keyed.values()) + sum(supply.keyless.values()),
        sum(totals.held_out_text_dropped.values()),
    )
    return dataclasses.replace(survey, totals=dataclasses.replace(
        totals, supply=supply.resolved(totals.mode_counts),
    ))


def _decide_or_refuse(
    config: BuildCorpusConfig, survey: Survey, trees: ConvertedTrees, surface: Surface,
    held_out: HeldOutCards,
) -> Decisions:
    """:func:`decide`, refusing a corpus the quality rules reject or a dataset
    gate 1 could not be scored on."""
    check_refusals(survey.totals.refusals, survey.totals.kept)
    decisions = decide(survey, sidecars=trees.sidecars, surface=surface, config=config)
    if not decisions.strata.card_disjoint:
        raise BuildCorpusError(
            "No game is admitted to the card-disjoint stratum, so the dataset "
            "cannot score gate 1 and the trainer would refuse it in its first "
            f"minute. {config.records_dir} holds {len(survey.totals.games.games)} game(s), "
            f"{len(survey.totals.games.held_out)} of which name one of the "
            f"{len(held_out.names)} held-out card(s). Depleted and "
            "full-strength shards live in sibling subdirectories of the raw "
            "corpus root: point --records-dir at the parent of both, and check "
            "that a full-strength run — pools generated without "
            "--exclude-cards — has actually been collected. A held-out game is "
            "admitted only while every held-out ability text it resolves is "
            "under --card-disjoint-text-cap, and one that resolves none is "
            "never admitted."
        )
    return decisions


def _write_validation_samples(
    config: BuildCorpusConfig, store: CorpusStore, context: WorkerContext,
) -> None:
    if config.validation_sample <= 0:
        logger.info("no validation sample written (--validation-sample 0)")
        return
    from effects.application.validation_samples import draw_samples, write_samples

    samples = draw_samples(
        store, mix=config.mix(), size=config.validation_sample, seed=config.seed,
        held_out_text_of=lambda record: context.held_out_text(acting_keys(record)),
    )
    sampled = write_samples(store, samples)
    for stratum, count in sampled.items():
        logger.info("%-22s %9d record(s) sampled for validation", stratum, count)
    check_samples(sampled, size=config.validation_sample)


def _repack_outputs(
    store: CorpusStore, written: WriteResult, *, shard_records: int, seed: int,
) -> None:
    """Deal each stratum's parts into uniform, mixed shards, then drop the parts.

    A part is one source shard, so unrepacked the trainer's per-shard step
    budget would be uneven and a shard's batches correlated by run.
    """
    import shutil

    from effects.infrastructure.record_io import iter_shards, repack_shards

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
    shutil.rmtree(store.parts_dir, ignore_errors=True)
