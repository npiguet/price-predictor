"""What ``build-corpus`` decides between the survey and the write pass.

:func:`decide` folds provenance keys to ability texts and decides the
placement, the plan and the rule budgets:
:func:`~effects.domain.stratum_placement.place_games` over the whole corpus,
gen-2's natural share (``plan_selection``) over the training games it leaves,
then the rule budgets of the training stratum and of each validation stratum
(:mod:`effects.application.stratum_budgets`; FR-054, FR-068a).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from effects.application.build_corpus_config import BuildCorpusConfig
from effects.application.corpus_rule_catalogue import RuleCatalogue
from effects.application.corpus_survey import RarityTally, Survey
from effects.application.corpus_write import WritePlan, WriteTask
from effects.application.rule_counts import CatalogueStanding, rule_order
from effects.application.stratum_budgets import (
    BudgetPlanner,
    PlacedSurvey,
    training_supply,
    validation_weights,
)
from effects.domain.ability_tokenizer import Surface
from effects.domain.corpus_curation import one_sided_conditions
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
from effects.domain.corpus_types import CopyMap, RuleCell
from effects.domain.keyword_names import display_name_of
from effects.domain.provenance import ProvenanceKey, SidecarLine, first_text
from effects.domain.rule_budgets import BudgetInputs, RuleBudgets, plan_rule_budgets
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.domain.stratum_placement import GameCensus, PlacementRule, place_games

if TYPE_CHECKING:
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.sidecar_io import SidecarCache

__all__ = ["Decisions", "TextJoin", "decide", "tagged_nowhere_written"]


@dataclass(frozen=True, slots=True)
class TextJoin:
    """How a provenance key joins its ability text: the sidecars its line is
    read from, and the encoding surface the text is read on."""

    sidecars: SidecarCache
    surface: Surface

    def reader(self) -> Callable[[ProvenanceKey], str | None]:
        """The trainer's own key-to-text join, so the rarity table keys match."""
        from effects.application.ability_text import text_for_key

        sidecars, surface = self.sidecars, self.surface

        def text_of(key: ProvenanceKey) -> str | None:
            return text_for_key(key, sidecars, surface)

        return text_of


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
        (), BudgetInputs(class_targets={}, budgeted=(), reuse_cap=1, text_cap=0)))
    #: Each validation stratum's, planned the same way over its own records.
    validation_budgets: dict[Stratum, RuleBudgets] = field(default_factory=dict)
    #: ``validation stratum -> class -> class_weight``, which weighs each
    #: validation stratum into the class mix (FR-059a).
    validation_weights: dict[Stratum, dict[str, float]] = field(default_factory=dict)
    #: ``game -> record hash -> drawn rule per added copy`` over every stratum.
    copies: CopyMap = field(default_factory=dict)
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
            validation_weights=self.validation_weights,
        )

    def write_tasks(self, survey: Survey) -> list[WriteTask]:
        """One task per raw shard, each carrying its own games' added copies."""
        tasks = []
        for shard in survey.shards:
            games = survey.shard_games.get(shard.name, frozenset())
            tasks.append(WriteTask(shard=shard, copies={
                game: self.copies[game] for game in games if game in self.copies
            }, surveyed=survey.shard_records.get(shard.name),
                cut=survey.shard_cut.get(shard.name, frozenset())))
        return tasks


def decide(
    survey: Survey,
    *,
    join: TextJoin,
    config: BuildCorpusConfig,
    rules: RuleCatalogue | None = None,
) -> Decisions:
    """Fold keys to texts, place the games, and plan every record's copies:
    the natural share, then the rule budgets of each stratum."""
    rules = rules or RuleCatalogue.empty()
    totals = survey.totals
    _refuse_another_heap_cap(totals.supply, config)
    key_text, text_games = _fold_keys_to_texts(totals.rarity, join.reader())
    _refuse_keyword_misfires(key_text, join.sidecars, config.keyword_definitions)
    census = _census(survey)
    strata = place_games(census, config.placement)
    placed = PlacedSurvey(survey, strata)
    selection = plan_selection(
        members_of(training_supply(placed), key_text), mix=config.mix(),
        ceiling=config.training_records, text_cap=config.text_cap,
        reuse_cap=config.reuse_cap,
    )
    budgets, validation = BudgetPlanner(
        placed, key_text, config, rules.budgeted).plan(selection)
    copies = _all_copies(budgets, validation)
    rule_counts = totals.rule_counts.to_json(CatalogueStanding.of(rules.scenarios))
    nowhere = tagged_nowhere_written(rule_counts, budgets, validation)
    return Decisions(
        selection=selection,
        strata=strata,
        keyword_threshold_games=_keyword_threshold_games(strata, census, config.placement),
        rarity={text: len(games) for text, games in text_games.items()},
        key_text=key_text,
        mode_counts=dict(sorted(totals.mode_counts.items())),
        budgets=budgets,
        validation_budgets=validation,
        validation_weights=validation_weights(placed, config),
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
        ) | {"tagged_nowhere_written": nowhere},
    )


def _all_copies(budgets: RuleBudgets, validation: Mapping[Stratum, RuleBudgets]) -> CopyMap:
    """Every stratum's added copies, training's first, by game."""
    copies = dict(budgets.copies)
    for planned in validation.values():
        copies.update(planned.copies)
    return copies


def _keyword_threshold_games(
    strata: GameStrata, census: GameCensus, placement: PlacementRule,
) -> frozenset[str]:
    """Game-disjoint games placed only because they hold a qualifying combat
    for a listed keyword (FR-053)."""
    return frozenset(
        game for game in strata.game_disjoint
        if game in census.keyword_combat
        and placement.under_keyword_threshold(game)
        and placement.base_game_disjoint(game, keyword_combat=True)
    )


def tagged_nowhere_written(
    rule_counts: Mapping[str, Any], budgets: RuleBudgets,
    validation: Mapping[Stratum, RuleBudgets],
) -> tuple[str, ...]:
    """Rules the counts name that no stratum's rows write a copy of: tagged
    only on records no stratum holds (refused, turn actions, a training
    record acting through a held-out text), or on none the plan writes."""
    written = {
        rule for planned in (budgets, *validation.values())
        for rows in planned.rows.values() for rule, by_polarity in rows.items()
        if any(row.written_total for row in by_polarity.values())
    }
    return tuple(sorted(set(rule_counts["rules"]) - written, key=rule_order))


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


def _keyword_misfires(
    key_text: dict[Keys, str], sidecars: SidecarCache, definitions: Path | None,
) -> list[str]:
    """Texts a display-name keyword match would expand on a non-keyword line
    (FR-015), replacing a script with a reminder template it does not carry."""
    if definitions is None or not Path(definitions).exists():
        return []
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

    names = {name.lower() for name in load_keyword_definitions(Path(definitions))}
    found = [
        misfire for keys in key_text for key in keys
        if (misfire := _misfire(key, sidecars.line_or_none(key), names)) is not None
    ]
    return sorted(set(found))


def _misfire(key: ProvenanceKey, line: SidecarLine | None, names: set[str]) -> str | None:
    """``key``'s line, named, when it is no keyword line and its display name
    matches a keyword definition; None otherwise."""
    if line is None or not line.script_text:
        return None
    if line.keyword_display_name is not None:
        return None
    if display_name_of(line.script_text).lower() not in names:
        return None
    return f"{key.script_file}: {line.script_text!r}"


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
