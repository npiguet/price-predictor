"""Build a fixed training corpus and two fixed validation strata (FR-135).

Between the survey (:mod:`effects.application.corpus_survey`) and the write
pass (:mod:`effects.application.corpus_write`), this module folds provenance
keys to ability texts and decides the placement, the plan and the rule
budgets: :func:`~effects.domain.stratum_placement.place_games` over the whole
corpus, gen-2's natural share (``plan_selection``) over the training games it
leaves, then :func:`~effects.domain.rule_budgets.plan_rule_budgets` over the
training stratum and each validation stratum (FR-054, FR-068a). ``rules/heads/``
and ``rules/scenarios/`` are read once, here (FR-060a).
"""

from __future__ import annotations

import functools
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from effects.application.corpus_report import (
    build_manifest,
    check_refusals,
    check_samples,
    log_report,
    report_write_pass,
)
from effects.application.corpus_survey import (
    RarityTally,
    ReadConfig,
    Survey,
    SurveyConfig,
    SurveyedRecord,
    WorkerContext,
    run_survey,
)
from effects.application.corpus_write import (
    WriteConfig,
    WritePlan,
    WriteResult,
    WriteTask,
    run_write_pass,
)
from effects.application.rule_counts import none_rules
from effects.domain import default_paths
from effects.domain.ability_tokenizer import Surface
from effects.domain.corpus_curation import (
    natural_copies,
    one_sided_conditions,
    validation_multipliers,
)
from effects.domain.corpus_manifest import rule_curation_fields
from effects.domain.corpus_plan import (
    BuildCorpusError,
    GameStrata,
    Keys,
    SelectionPlan,
    TrainingSupply,
    members_of,
    plan_selection,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from effects.domain.keyword_names import display_name_of, normalize_keyword
from effects.domain.provenance import ProvenanceKey, SourceTree, first_text, split_script_file
from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
from effects.domain.records import acting_keys
from effects.domain.rule_budgets import BudgetedRecord, RuleBudgets, plan_rule_budgets
from effects.domain.rule_catalogue import (
    MapVerdict,
    Polarity,
    RuleMapEntry,
    ScenarioCatalogue,
)
from effects.domain.sampling import DEFAULT_KIND_MIX
from effects.domain.shard_generation import (
    MixedGenerationsError,
    ShardGeneration,
    UnacceptedGenerationError,
)
from effects.domain.split import HeldOutCards, HoldoutRule
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.domain.stratum_placement import (
    MAX_COMPONENT_ERROR,
    MAX_ITEM_ERROR,
    MIN_GAMES_PER_HALF,
    GameCensus,
    PlacementRule,
    RuleCell,
    place_games,
    rule_test_misses,
)

if TYPE_CHECKING:
    from effects.infrastructure.converted_trees import ConvertedTrees, SourceTrees
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

__all__ = [
    "DEFAULT_GAME_DISJOINT_KEYWORDS", "BuildCorpusConfig", "BuildCorpusError",
    "Decisions", "build", "decide", "parse_keyword_list",
]

logger = logging.getLogger(__name__)


#: FR-045's default keyword list, in the spelling an operator types.
DEFAULT_GAME_DISJOINT_KEYWORDS: tuple[str, ...] = (
    "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
)


def parse_keyword_list(text: str) -> tuple[str, ...]:
    """``--game-disjoint-keywords``: a comma list, blanks dropped."""
    return tuple(part.strip() for part in text.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class BuildCorpusConfig:
    """The `build-corpus` flags (root spec § Curated corpus).

    ``help`` metadata is the flag's help, so a default lives here alone. A
    directory's help names its default as an operator types it, trailing
    slash and all, which the ``Path`` the flag parses to cannot spell.
    """

    records_dir: Path = field(
        default=Path(default_paths.DEFAULT_RECORDS_DIR), metadata={"help": (
            "Raw corpus root; discovery recurses "
            f"(default: {default_paths.DEFAULT_RECORDS_DIR})"
        )},
    )
    output: Path = field(
        default=Path(default_paths.DEFAULT_CORPUS_OUTPUT), metadata={"help": (
            f"Curated dataset directory (default: {default_paths.DEFAULT_CORPUS_OUTPUT})"
        )},
    )
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    #: A string rather than a path: the manifest records it, and a Path would
    #: rewrite its separators on Windows.
    vocab_path: str = field(default=default_paths.DEFAULT_VOCAB_PATH, metadata={"help": (
        "Decides the encoding surface the rarity table's text keys are built "
        "on (default: %(default)s)"
    )})
    variant_scripts: str | None = None
    holdout: HoldoutRule = HoldoutRule()
    text_cap: int = field(default=200, metadata={"help": (
        "Max written training records per ability text per selection cell, "
        "repeats included; 0 means no cap (default: %(default)s)"
    )})
    reuse_cap: int = field(default=4, metadata={"help": (
        "Max copies of one record toward a short effect family's share (FR-049); "
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
    game_disjoint_keywords: tuple[str, ...] = field(
        default=DEFAULT_GAME_DISJOINT_KEYWORDS, metadata={
            "type": parse_keyword_list,
            "help": (
                "Comma list of damage-step keywords raising a game's threshold "
                f"(default: {','.join(DEFAULT_GAME_DISJOINT_KEYWORDS)})"
            ),
        },
    )
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
    max_events_per_record: int = field(default=MAX_EVENTS_PER_RECORD, metadata={"help": (
        "A record carrying more events than this is refused as a game dump "
        "rather than an ability; 0 disables the event-flood rule "
        "(default: %(default)s)"
    )})
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
    #: The rules catalogue (``scenarios/`` and ``heads/``), read once. No
    #: flag (contracts/cli.md): a field so a test can point at its own.
    rules_dir: Path = Path("rules")
    #: The precision thresholds of research E5. No operator flag overrides
    #: them; fields so a test can lower them.
    min_games_per_half: int = MIN_GAMES_PER_HALF
    max_item_error: float = MAX_ITEM_ERROR
    max_component_error: float = MAX_COMPONENT_ERROR

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
            ("--max-events-per-record", self.max_events_per_record),
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
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        from effects.infrastructure.converted_trees import SourceTrees

        return SourceTrees.of(self.cards_folders, self.variant_scripts)

    @property
    def heap_cap(self) -> int:
        """How many smallest hashes the survey keeps per cell and key; every
        one under ``--text-cap 0``."""
        return self.text_cap if self.text_cap > 0 else sys.maxsize

    @property
    def placement(self) -> PlacementRule:
        """Placement's flags and precision thresholds (research R3, E5)."""
        return PlacementRule(
            game_disjoint_share=self.game_disjoint_share,
            keyword_share=self.game_disjoint_keyword_share,
            card_disjoint_text_cap=self.card_disjoint_text_cap,
            seed=self.seed,
            min_games_per_half=self.min_games_per_half,
            max_item_error=self.max_item_error,
            max_component_error=self.max_component_error,
        )


@dataclass(frozen=True, slots=True)
class RuleCatalogue:
    """``rules/scenarios/`` and ``rules/heads/`` as one build read them (FR-060a).

    Budgets, the manifest's map snapshot and every reader after it take a
    rule's verdict and outputs from here, never from the files again.
    """

    scenarios: ScenarioCatalogue
    #: ``rule -> RuleMapEntry`` for every heads line.
    heads: dict[str, RuleMapEntry]
    #: ``rules_catalogue_io.heads_digest``: the map's content hash.
    digest: str

    @classmethod
    def read(cls, rules_dir: Path) -> RuleCatalogue:
        from effects.infrastructure.rules_catalogue_io import (
            heads_digest,
            load_heads,
            load_scenarios,
        )

        return cls(load_scenarios(rules_dir), load_heads(rules_dir), heads_digest(rules_dir))

    @classmethod
    def empty(cls) -> RuleCatalogue:
        """No scenario and no map: nothing is tagged, nothing budgeted."""
        return cls(ScenarioCatalogue(()), {}, "")

    @property
    def budgeted(self) -> frozenset[str]:
        """The rules the map marks ``trained`` (FR-054)."""
        return frozenset(
            rule for rule, entry in self.heads.items() if entry.verdict is MapVerdict.TRAINED)

    def snapshot(self, rules: Iterable[str]) -> dict[str, RuleMapEntry]:
        """The manifest's ``rule_map``: every tagged rule's verdict and outputs,
        ``missing`` for one no heads line names."""
        return {
            rule: self.heads.get(rule) or RuleMapEntry(rule=rule, verdict=MapVerdict.MISSING)
            for rule in sorted(set(rules))
        }


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
    #: The training stratum's rule budgets (FR-054), which the manifest reports.
    budgets: RuleBudgets = field(default_factory=lambda: plan_rule_budgets(
        (), class_targets={}, budgeted=(), reuse_cap=1, text_cap=0))
    #: Each validation stratum's, planned the same way over its own records.
    validation_budgets: dict[Stratum, RuleBudgets] = field(default_factory=dict)
    #: ``validation stratum -> class -> natural copies per record``, which
    #: write each validation stratum in the class mix (FR-059a).
    validation_copies: dict[Stratum, dict[str, float]] = field(default_factory=dict)
    #: ``game -> record hash -> drawn rule per added copy`` over every stratum.
    copies: dict[str, dict[int, tuple[str, ...]]] = field(default_factory=dict)
    #: Placement's per-game rule inventories, which the precision check reads.
    inventories: dict[str, dict[RuleCell, int]] = field(default_factory=dict)
    #: The manifest's gen-3 fields (``corpus_manifest.rule_curation_fields``).
    rule_fields: dict = field(default_factory=dict)

    def write_plan(self, store: CorpusStore) -> WritePlan:
        """What the write pass applies: these decisions, its parts in ``store``."""
        return WritePlan(
            parts_dirs={
                stratum: store.parts_dir_for(stratum) for stratum in OUTPUT_STRATA
            },
            strata=self.strata,
            plans=self.selection.plans,
            keyless_plans=self.selection.keyless,
            mode_counts=self.mode_counts,
            validation_copies=self.validation_copies,
        )

    def write_tasks(self, survey: Survey) -> list[WriteTask]:
        """One task per raw shard, each carrying its own games' added copies."""
        tasks = []
        for shard in survey.shards:
            games = survey.shard_games.get(shard.name, frozenset())
            tasks.append(WriteTask(shard=shard, copies={
                game: self.copies[game] for game in games if game in self.copies
            }))
        return tasks


def _keyword_misfires(
    key_text: dict[Keys, str], sidecars: SidecarCache, definitions: Path | None,
) -> list[str]:
    """Texts a display-name keyword match would expand on a non-keyword line
    (FR-015), replacing a script with a reminder template it does not carry."""
    if definitions is None or not Path(definitions).exists():
        return []
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

    names = {name.lower() for name in load_keyword_definitions(Path(definitions))}
    found: list[str] = []
    for keys in key_text:
        for key in keys:
            line = sidecars.line_or_none(key)
            if line is None or not line.script_text:
                continue
            if line.keyword_display_name is not None:
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
    rules: RuleCatalogue | None = None,
) -> Decisions:
    """Fold keys to texts, place the games, and plan every record's copies:
    the natural share, then the rule budgets of each stratum."""
    rules = rules or RuleCatalogue.empty()
    totals = survey.totals
    _refuse_another_heap_cap(totals.supply, config)

    from effects.application.ability_text import text_for_key

    def text_of(key: ProvenanceKey) -> str | None:
        # The trainer's own key-to-text join, so the rarity table keys match.
        return text_for_key(key, sidecars, surface)

    key_text, text_games = _fold_keys_to_texts(totals.rarity, text_of)
    _refuse_keyword_misfires(key_text, sidecars, config.keyword_definitions)
    census = _census(survey)
    strata = place_games(census, config.placement)
    selection = plan_selection(
        members_of(_training_supply(survey, strata), key_text), mix=config.mix(),
        ceiling=config.training_records, text_cap=config.text_cap,
        reuse_cap=config.reuse_cap,
    )
    validation_copies = _validation_copies(survey, strata, config)
    budgets, validation = _rule_budgets(
        survey, strata, selection, key_text, config=config, budgeted=rules.budgeted,
        validation_copies=validation_copies,
    )
    copies: dict[str, dict[int, tuple[str, ...]]] = dict(budgets.copies)
    for planned in validation.values():
        copies.update(planned.copies)
    rule_counts = totals.rule_counts.to_json(none_rules(rules.scenarios))
    return Decisions(
        selection=selection,
        strata=strata,
        keyword_threshold_games=frozenset(
            game for game in strata.game_disjoint
            if game in census.keyword_combat
            and config.placement.under_keyword_threshold(game)
            and config.placement.base_game_disjoint(game, keyword_combat=True)
        ),
        rarity={text: len(games) for text, games in text_games.items()},
        key_text=key_text,
        mode_counts=dict(sorted(totals.mode_counts.items())),
        budgets=budgets,
        validation_budgets=validation,
        validation_copies=validation_copies,
        copies=copies,
        inventories=dict(census.inventories),
        rule_fields=rule_curation_fields(
            budgets=budgets, strata=strata,
            rule_map=rules.snapshot(_rules_named(rule_counts, budgets, validation)),
            rule_map_digest=rules.digest, rule_counts=rule_counts,
            carried=totals.carried,
            one_sided_conditions=one_sided_conditions(totals.condition_counts),
            turn_action_left_out=dict(totals.turn_actions),
            subtypes=totals.subtypes, keywords=totals.keywords,
        ),
    )


def _rules_named(
    rule_counts: Mapping[str, Any], budgets: RuleBudgets,
    validation: Mapping[Stratum, RuleBudgets],
) -> set[str]:
    """Every rule the map snapshot must hold: each tagged rule, and each rule
    a copy can be drawn for, a protection rule no detector tags included,
    since the trainer reads a drawn rule's outputs off the manifest."""
    named = set(rule_counts["rules"])
    for planned in (budgets, *validation.values()):
        named |= {rule for rows in planned.rows.values() for rule in rows}
    return named


def _census(survey: Survey) -> GameCensus:
    """What placement reads of the survey (research R3)."""
    games = survey.totals.games
    return GameCensus(
        games=frozenset(games.games),
        held_out=frozenset(games.held_out),
        random_seat=frozenset(games.random_seat),
        keyword_combat=frozenset(games.keyword_combat),
        held_out_text_games=games.held_out_text_games,
        inventories=dict(survey.totals.inventories),
    )


def _training_supply(survey: Survey, strata: GameStrata) -> TrainingSupply:
    """The survey's supply with the candidate games placement trained folded
    in, their legality cells folded as the survey's were."""
    totals = survey.totals
    supply = TrainingSupply(totals.supply.heap_cap)
    supply.merge(totals.supply)
    for record in totals.candidates:
        if not record.held_out_text and strata.stratum_of(record.game) is Stratum.TRAINING:
            supply.add(record.cell.resolved(totals.mode_counts), record.keys, record.record_hash)
    return supply


def _validation_copies(
    survey: Survey, strata: GameStrata, config: BuildCorpusConfig,
) -> dict[Stratum, dict[str, float]]:
    """Each validation stratum's natural copies per record and class, which
    write it in the training stratum's ``--class-mix`` (FR-059a), logged
    beside the class counts they reshape."""
    copies: dict[Stratum, dict[str, float]] = {}
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT):
        counts = Counter(
            str(record.klass) for record in survey.totals.candidates
            if strata.stratum_of(record.game) is stratum
        )
        copies[stratum] = validation_multipliers(
            counts, {str(k): v for k, v in config.mix().items()}, reuse_cap=config.reuse_cap,
        )
        for name in sorted(counts):
            logger.info(
                "%-14s %-22s %7d record(s), %.2f natural copies each",
                stratum, name, counts[name], copies[stratum][name],
            )
    return copies


def _rule_budgets(
    survey: Survey, strata: GameStrata, selection: SelectionPlan, key_text: dict[Keys, str],
    *, config: BuildCorpusConfig, budgeted: frozenset[str],
    validation_copies: Mapping[Stratum, Mapping[str, float]] | None = None,
) -> tuple[RuleBudgets, dict[Stratum, RuleBudgets]]:
    """The training stratum's rule budgets over gen-2's natural share, and
    each validation stratum's over its own natural copies, the class mix's
    (FR-059a, FR-068a): every stratum is built with the same rule budgets, a
    class's target being the records it writes before any copy is added."""
    totals = survey.totals
    mode_counts = totals.mode_counts

    def natural(record: SurveyedRecord) -> int:
        return selection.copies(record.cell.resolved(mode_counts), record.keys,
                                record.record_hash)

    training = [*totals.tagged, *(
        record for record in totals.candidates
        if not record.held_out_text and strata.stratum_of(record.game) is Stratum.TRAINING
        and (record.tags or record.protection)
    )]
    plan = functools.partial(
        plan_rule_budgets, budgeted=budgeted, reuse_cap=config.reuse_cap,
        text_cap=config.text_cap,
    )
    budgets = plan(
        _budgeted_records(training, key_text, natural), class_targets=selection.targets,
    )
    validation: dict[Stratum, RuleBudgets] = {}
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT):
        records = [r for r in totals.candidates if strata.stratum_of(r.game) is stratum]
        multipliers = (validation_copies or {}).get(stratum, {})

        def held(record: SurveyedRecord, multipliers=multipliers) -> int:
            return natural_copies(multipliers.get(str(record.klass), 1.0), record.record_hash)

        targets: Counter[str] = Counter()
        for record in records:
            targets[str(record.klass)] += held(record)
        validation[stratum] = plan(
            _budgeted_records(records, key_text, held), class_targets=dict(targets),
        )
    return budgets, validation


def _budgeted_records(
    records: Iterable[SurveyedRecord], key_text: dict[Keys, str],
    natural: Callable[[SurveyedRecord], int],
) -> Iterator[BudgetedRecord]:
    """Each tagged record as the budgets read it, and beside it each record
    offering protection copies as a record of its own, natural share zero,
    tagged ``fires`` with every protection rule it offers (FR-058)."""
    for record in records:
        text = key_text.get(record.keys)
        if record.tags:
            yield BudgetedRecord(
                game=record.game, record_hash=record.record_hash, klass=str(record.klass),
                text=text, natural=natural(record), source=record.source, tags=record.tags,
            )
        if record.protection:
            yield BudgetedRecord(
                game=record.game, record_hash=record.protection_hash,
                klass=str(record.klass), text=text, natural=0, source=record.source,
                tags=tuple((rule, Polarity.FIRES) for rule in record.protection),
            )


def _refuse_another_heap_cap(supply: TrainingSupply, config: BuildCorpusConfig) -> None:
    if supply.heap_cap != config.heap_cap:
        raise ValueError(
            f"the survey kept {supply.heap_cap} hashes per cell and key, "
            f"but config.text_cap {config.text_cap} keeps {config.heap_cap}. "
            "decide() must be run against the Survey that this same "
            "--text-cap produced, or the copy plans it writes would disagree "
            "with what the manifest is about to record."
        )


def _fold_keys_to_texts(
    rarity: RarityTally, text_of: Callable[[ProvenanceKey], str | None],
) -> tuple[dict[Keys, str], dict[str, set[int]]]:
    """``acting keys -> ability text``, and each text's games, over every key
    tuple the survey saw; keys with no text are left out."""
    key_text: dict[Keys, str] = {}
    text_games: dict[str, set[int]] = {}
    for keys in rarity.key_records:
        text = first_text(keys, text_of)
        if text is None:
            # A runtime-only trait or an unconverted script: no text to cap
            # against, so its records are sampled with the cell's keyless ones.
            continue
        key_text[keys] = text
        text_games.setdefault(text, set()).update(rarity.key_games.get(keys, ()))
    return key_text, text_games


def _refuse_keyword_misfires(
    key_text: dict[Keys, str], sidecars: SidecarCache, definitions: Path | None,
) -> None:
    misfires = _keyword_misfires(key_text, sidecars, definitions)
    if misfires:
        raise BuildCorpusError(
            f"{len(misfires)} non-keyword line(s) would expand as a keyword by "
            "display name (FR-015): the text before the first colon matches a "
            "keyword definition while the line's script_api_type is not "
            "Keyword. First: " + "; ".join(misfires[:5])
        )


def build(config: BuildCorpusConfig) -> int:
    """Build a curated dataset, or verify an existing one. Returns an exit code:
    1, after the corpus and its manifest are written, when a rule-test
    component already misses its precision target (research E5)."""
    from effects.domain.ability_encoder import surface_of
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(config.output)
    if config.verify:
        return _verify(config, store)

    _require_generation(config.records_dir)
    rules = RuleCatalogue.read(config.rules_dir)
    roots = config.source_trees.sidecar_roots()
    trees = _converted_trees(config, roots)
    held_out = _select_holdout(config, trees)
    surface = surface_of(config.vocab_path)
    _log_holdout(config, held_out, surface)
    read = ReadConfig(
        records_dir=config.records_dir, sidecar_roots=roots,
        max_events=config.max_events_per_record, held_out=held_out, seed=config.seed,
        catalogue=rules.scenarios,
        rule_outputs={rule: tuple(entry.outputs) for rule, entry in rules.heads.items()},
    )
    survey = _survey(config, read, rules)
    decisions = _decide_or_refuse(config, survey, trees, surface, held_out, rules)

    # Before the first output shard, and only on a real build: a rebuild
    # replaces the dataset rather than layering over it (FR-144).
    store.clear_outputs()
    written = run_write_pass(
        decisions.write_tasks(survey),
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
    from effects.infrastructure.shard_io import list_shards

    records_dir = config.records_dir
    manifest = store.load()
    added, removed, resized = manifest.drift(list_shards(records_dir))
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

    try:
        require_generation(iter_shards(records_dir), ShardGeneration.GEN_3)
    except (MixedGenerationsError, UnacceptedGenerationError) as exc:
        raise BuildCorpusError(str(exc)) from exc


def _known_keywords(definitions: Path | None) -> frozenset[str]:
    """Forge's keywords for the closed keyword list; none without the table,
    which leaves the overlay keywords alone."""
    if definitions is None or not Path(definitions).exists():
        return frozenset()
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

    return frozenset(normalize_keyword(name)
                     for name in load_keyword_definitions(Path(definitions)))


def _survey(config: BuildCorpusConfig, read: ReadConfig, rules: RuleCatalogue) -> Survey:
    """The parallel survey, logged."""
    survey = run_survey(
        config.records_dir,
        config=SurveyConfig(
            read=read, heap_cap=config.heap_cap, placement=config.placement,
            game_disjoint_keywords=tuple(
                normalize_keyword(name) for name in config.game_disjoint_keywords
            ),
            budgeted=rules.budgeted,
            known_keywords=_known_keywords(config.keyword_definitions),
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


def _decide_or_refuse(
    config: BuildCorpusConfig, survey: Survey, trees: ConvertedTrees, surface: Surface,
    held_out: HeldOutCards, rules: RuleCatalogue,
) -> Decisions:
    """:func:`decide`, refusing a corpus the quality rules reject or a dataset
    gate 1 could not be scored on."""
    check_refusals(survey.totals.refusals, survey.totals.kept)
    decisions = decide(
        survey, sidecars=trees.sidecars, surface=surface, config=config, rules=rules,
    )
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
    from effects.infrastructure.shard_io import iter_shards, repack_shards

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
