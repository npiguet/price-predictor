"""What ``build-corpus`` records about a build, checks after it, and tells the operator.

The operator's report is read back off the manifest, so the log and the file
cannot disagree.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from effects.application.corpus_survey import RefusalTally, Survey
from effects.application.corpus_write import TrainingTally, WriteResult
from effects.domain.ability_tokenizer import Surface
from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, FamilyRow
from effects.domain.corpus_plan import BuildCorpusError, Keys
from effects.domain.coverage import HELD_OUT_GAME_FLOOR
from effects.domain.provenance import JOIN_GAP_KINDS
from effects.domain.record_quality import QualityDefect
from effects.domain.records import LegalityDecision
from effects.domain.sampling import SamplingClass
from effects.domain.strata import DROPPED_STRATA, OUTPUT_STRATA, Stratum
from effects.domain.text_holdout import normalize_script_text
from effects.infrastructure.build_inputs import code_version
from price_predictor.infrastructure.forge_classpath import project_root

if TYPE_CHECKING:
    from effects.application.build_corpus_config import BuildCorpusConfig
    from effects.application.corpus_decisions import Decisions
    from effects.domain.split import HeldOutCards
    from effects.infrastructure.build_inputs import KeywordTable

logger = logging.getLogger(__name__)

#: Join gaps that are expected answers, reported without a warning.
EXPECTED_JOIN_GAPS = frozenset({"runtime_only", "dropped"})

#: How much of one sampling class the quality rules may refuse before the
#: build stops, rather than silently losing a whole class.
MAX_REFUSED_SHARE = 0.5

#: Unjoinable-key records the build tolerates; it stops only past both, so a
#: small pilot does not stop on its one Flashback-cast charm.
MAX_UNJOINABLE_SHARE = 0.001
MAX_UNJOINABLE_RECORDS = 20

#: A directory whose share of held-out games is above zero and below this
#: holds a few held-out games where it should hold none: a leak in collection,
#: not a full-strength source.
_LEAK_SHARE = 0.05

#: A class delivered more than this far under its target is reported as short;
#: keyless cells are sampled by hash rate, exact in expectation only.
_DELIVERY_TOLERANCE = 0.01


def check_refusals(refusals: RefusalTally, kept: Mapping[SamplingClass, int]) -> None:
    """Stop when the quality rules refused most of any one sampling class, or
    past both unjoinable-key limits over every record read: those are the join
    failing, not a record being poor.

    Read from the survey, which judges every record under the write pass's own
    rule, so a refused build stops before it clears the previous dataset.
    """
    _check_unjoinable(refusals, kept)
    _check_class_refusals(refusals, kept)


def _check_unjoinable(refusals: RefusalTally, kept: Mapping[SamplingClass, int]) -> None:
    """Stop past both unjoinable-key limits over every record read."""
    unjoinable = refusals.by_reason.get(QualityDefect.UNJOINABLE_KEY, 0)
    total_read = sum(kept.values()) + sum(refusals.by_class.values())
    if (
        unjoinable > MAX_UNJOINABLE_RECORDS
        and unjoinable > MAX_UNJOINABLE_SHARE * total_read
    ):
        worst = ", ".join(
            f"{script} ({count})"
            for script, count in refusals.unjoinable_scripts.most_common(5)
        )
        raise BuildCorpusError(
            f"{unjoinable} of the {total_read} record(s) read act through a key "
            "inside a range their sidecar declares and in neither its lines nor "
            f"its dropped keys ({100.0 * unjoinable / total_read:.2f}%, past the "
            f"{100.0 * MAX_UNJOINABLE_SHARE:.1f}% and the {MAX_UNJOINABLE_RECORDS} "
            "records the build drops without stopping). "
            "That many means the sidecars do not describe the cards the "
            "records were collected against: the cards were reconverted "
            f"after collection. Worst scripts: {worst}."
        )


def _check_class_refusals(refusals: RefusalTally, kept: Mapping[SamplingClass, int]) -> None:
    """Stop when the quality rules refused most of any one sampling class."""
    for name in sorted(refusals.by_class):
        refused = refusals.by_class[name]
        total = refused + kept.get(name, 0)
        if not total or refused <= MAX_REFUSED_SHARE * total:
            continue
        raise BuildCorpusError(
            f"{name}: {refused} of the {total} record(s) read were refused by "
            f"the record-quality rules ({100.0 * refused / total:.1f}%), past "
            f"the {100.0 * MAX_REFUSED_SHARE:.0f}% a class may lose. A whole "
            "class refused is usually one of three things rather than a "
            "dirty corpus: an unpatched (degraded) checkout, which resolves "
            "no attribution and stamps every event `unresolved`; a quality "
            "rule that is wrong for this kind of record; or a corpus rebuilt "
            "before the converter's claims landed -- `no-acting-text` "
            "dominating `quality_dropped` means the sidecars still drop the "
            "keys the records act through, so reconvert and try again. Read "
            "the quality_dropped breakdown and the mode the shards were "
            "collected in before rebuilding."
        )


def check_samples(counts: Mapping[str, int], *, size: int) -> None:
    """Stop when a stratum's validation sample came out empty, unless
    ``size <= 0`` asked for none."""
    if size <= 0:
        return
    for stratum in sorted(counts):
        if counts[stratum]:
            continue
        raise BuildCorpusError(
            f"the {stratum} validation sample is empty: --validation-sample "
            f"{size} was asked for and no record of any sampled class was "
            f"found in the {stratum} stratum. The trainer validates on this "
            "sample, so a build that shipped it would validate nan every "
            "epoch and select no checkpoint. Check that the stratum holds "
            "records of the classes --class-mix names."
        )


def report_write_pass(survey: Survey, written: WriteResult) -> None:
    """The write pass's refusals and attribution, logged."""
    kept_records = sum(written.training.read.values())
    logger.info(
        "%-22s %9d record(s) kept (%.1f%%) name no producing clause for at "
        "least one event — watched, not refused; a rising share is the "
        "collector's attribution, not the corpus",
        "unattributed", written.unattributed,
        100.0 * written.unattributed / kept_records if kept_records else 0.0,
    )
    if survey.totals.refusals.by_reason != written.refusals.by_reason:
        logger.warning(
            "The survey and the write pass disagree about which records are "
            "refused: survey %s, write pass %s. Both read the same shards "
            "under the same rule, so this means the corpus changed between "
            "the passes.",
            dict(sorted(survey.totals.refusals.by_reason.items())),
            dict(sorted(written.refusals.by_reason.items())),
        )
    cut = survey.totals.cut.games
    if cut:
        read = len(cut | survey.totals.games.games)
        logger.warning(
            "%d of %d game(s) (%.1f%%) have no game-end marker in their shard: cut by a "
            "killed worker or a game that threw, their records incomplete; %d record(s) "
            "are written to no stratum (manifest cut_games, per_stratum \"dropped-cut\").",
            len(cut), read, 100.0 * len(cut) / read, survey.totals.cut.records)
    unsurveyed = written.strata.records.get(Stratum.DROPPED_UNSURVEYED, 0)
    if unsurveyed:
        logger.warning(
            "%d record(s) belong to games the survey never saw: a raw shard grew "
            "between the survey and the write pass (a collection still writing "
            "under --records-dir?). Those games were never placed, so their "
            "records are written to no stratum (manifest per_stratum "
            "\"dropped-unsurveyed\"); `build-corpus --verify` names the shards "
            "that changed.", unsurveyed,
        )
    # The decode tally alone: the two passes read the same bytes, but ask the
    # sidecar cache different numbers of times, so their join gaps differ.
    surveyed, rewritten = survey.totals.reading, written.reading
    if (surveyed.off_list, sorted(surveyed.damaged_tails)) != (
            rewritten.off_list, sorted(rewritten.damaged_tails)):
        logger.warning(
            "The survey and the write pass met different decode tallies reading the "
            "same shards: the corpus changed between the passes. The manifest "
            "records the survey's.")
    if survey.split_games:
        logger.warning(
            "%d game(s) appear in more than one raw shard. A raw shard is one "
            "JVM lifetime, so this should not happen; each such game was "
            "routed at the end of every shard it appears in, and its records "
            "may have been planned for training slots the write pass then "
            "routed elsewhere.", survey.split_games,
        )


def _nested(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (outer, inner), count in sorted(counts.items()):
        out.setdefault(outer, {})[inner] = count
    return out


def family_tables(
    decisions: Decisions, written: WriteResult,
) -> tuple[dict[str, dict[str, FamilyRow]], dict[str, dict[str, dict[str, int]]]]:
    """The manifest's per-family and per-signature tables (FR-053)."""
    return _family_rows(decisions, written), _signature_rows(written)


def _family_rows(decisions: Decisions, written: WriteResult) -> dict[str, dict[str, FamilyRow]]:
    """``class -> family -> row``: each planned family against what was written."""
    families: dict[str, dict[str, FamilyRow]] = {}
    for klass, per_family in decisions.selection.families.items():
        for label, row in per_family.items():
            wrote = written.families.written.get((klass, label), 0)
            distinct = written.families.distinct.get((klass, label), 0)
            families.setdefault(klass, {})[label] = FamilyRow(
                available=row.available,
                share=row.share,
                planned=row.planned,
                written=wrote,
                repeats=wrote - distinct,
                shortfall=max(0, row.share - wrote),
            )
    return families


def _signature_rows(written: WriteResult) -> dict[str, dict[str, dict[str, int]]]:
    """``class -> family -> signature -> records written``, the empty signature left out."""
    signatures: dict[str, dict[str, dict[str, int]]] = {}
    for (klass, label, signature), count in sorted(written.families.signatures.items()):
        if signature:
            signatures.setdefault(klass, {}).setdefault(label, {})[signature] = count
    return signatures


def held_out_text_report(
    survey: Survey, written: WriteResult, held_out_texts: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Held-out texts with no gate-one record, and those resolved in under five games."""
    held_games = survey.totals.games.held_out
    texts = {normalize_script_text(text) for text in held_out_texts}
    without = tuple(sorted(texts - written.strata.gate_one_texts))
    games = {
        text: len(survey.totals.games.held_out_text_games.get(text, set()) & held_games)
        for text in sorted(texts)
    }
    under = {text: count for text, count in games.items() if count < HELD_OUT_GAME_FLOOR}
    return without, under


def recorded_path(path: str | None) -> str:
    """``path`` as the manifest records it, inside the digest: POSIX, relative
    to the project root when under it, else absolute; "" for none. A path typed
    relative to the working directory or absolute, with either slash, is one
    value, so it pins one digest. A path outside the checkout (another drive, a
    scratch directory) is recorded absolute rather than refused: it is a valid
    input, and only its spelling must not move the digest."""
    if not path:
        return ""
    resolved = Path(path).resolve()
    root = project_root()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def _unique_texts(keys: Iterable[Keys], key_text: Mapping[Keys, str]) -> int:
    """Distinct ability texts behind ``keys``: what a record count cannot say."""
    return len({key_text.get(key, key) for key in keys})


@dataclass(frozen=True, slots=True)
class BuildOutcome:
    """What a build read, decided and wrote: everything its manifest records
    besides the config."""

    surface: Surface
    held_out: HeldOutCards
    keywords: KeywordTable | None
    survey: Survey
    decisions: Decisions
    written: WriteResult


def build_manifest(config: BuildCorpusConfig, outcome: BuildOutcome) -> CorpusManifest:
    """Everything the build decided and wrote, as the manifest records it."""
    held_out, survey = outcome.held_out, outcome.survey
    decisions, written = outcome.decisions, outcome.written
    families, signatures = family_tables(decisions, written)
    without_resolution, under_floor = held_out_text_report(survey, written, held_out.texts)
    strata = written.strata
    return CorpusManifest(
        surface=outcome.surface,
        delivered_mix=_delivered_mix(written.training),
        held_out_cards=tuple(sorted(held_out.names)),
        held_out_texts=tuple(sorted(held_out.texts)),
        card_disjoint_games=tuple(sorted(decisions.strata.card_disjoint)),
        game_disjoint_games=tuple(sorted(decisions.strata.game_disjoint)),
        rarity=decisions.rarity,
        sources=survey.shards,
        per_class=_per_class(written.training, decisions.key_text),
        # The dropped strata write nothing, so have no unique texts.
        per_stratum={name: strata.records.get(name, 0) for name in Stratum},
        unique_texts={
            name: _unique_texts(strata.keys.get(name, ()), decisions.key_text)
            for name in OUTPUT_STRATA
        },
        shortfall=decisions.selection.shortfall,
        class_targets=dict(decisions.selection.targets),
        quality_dropped=dict(written.refusals.by_reason),
        quality_dropped_by_class=_nested(written.refusals.by_reason_and_class),
        # The hundred worst: every permanent carries the implicit spell.
        no_acting_text_scripts=dict(written.refusals.textless_scripts.most_common(100)),
        games_by_source=survey.games_by_source,
        held_out_games_by_source=survey.held_out_games_by_source,
        unattributed_records=written.unattributed,
        families=families,
        signatures=signatures,
        policy_counts=_nested(written.families.policy),
        legality_counts=_nested(written.families.legality),
        legality_mode_counts=dict(sorted(decisions.mode_counts.items())),
        keyword_threshold_games=tuple(sorted(decisions.keyword_threshold_games)),
        held_out_texts_without_resolution=without_resolution,
        held_out_texts_under_five_games=under_floor,
        **_flag_fields(config),
        **decisions.rule_fields,
        **_collection_fields(config, outcome),
    )


def _flag_fields(config: BuildCorpusConfig) -> dict:
    """The flags the manifest records as the build ran under them."""
    return {
        "seed": config.seed,
        "vocab_path": recorded_path(config.vocab_path),
        "variant_scripts": recorded_path(config.variant_scripts),
        "holdout_permille": config.holdout.permille,
        "holdout_max_carriers": config.holdout.max_carriers,
        "text_cap": config.text_cap,
        "card_disjoint_text_cap": config.card_disjoint_text_cap,
        "training_records": config.training_records,
        "class_mix": config.mix(),
        "shard_records": config.shard_records,
        "validation_sample": config.validation_sample,
        "max_events_per_record": config.max_events_per_record,
        "game_disjoint_share": config.game_disjoint_share,
        "game_disjoint_keyword_share": config.game_disjoint_keyword_share,
        "game_disjoint_keywords": tuple(config.game_disjoint_keywords),
        "reuse_cap": config.reuse_cap,
    }


def _per_class(training: TrainingTally, key_text: Mapping[Keys, str]) -> dict[str, ClassCounts]:
    """Each class's training records read, kept, dropped by the cap, and unique texts."""
    return {
        name: ClassCounts(
            read=training.read.get(name, 0),
            kept=training.kept.get(name, 0),
            dropped_by_cap=training.dropped_by_cap.get(name, 0),
            unique_texts=_unique_texts(training.kept_keys.get(name, ()), key_text),
        )
        for name in sorted(training.read)
    }


def _delivered_mix(training: TrainingTally) -> dict[str, float]:
    """Each class's share of the training records kept; none when none was."""
    trained = sum(training.kept.values())
    return {
        name: count / trained for name, count in sorted(training.kept.items())
    } if trained else {}


def _under(where: str, records_dir: Path) -> str:
    """A tally's ``<shard path>, byte <n>`` with the shard's path relative to
    ``--records-dir``, so the manifest does not depend on where the raw
    corpus sits."""
    path, comma, rest = where.partition(", ")
    try:
        relative = Path(path).resolve().relative_to(Path(records_dir).resolve()).as_posix()
    except ValueError:
        return where
    return f"{relative}{comma}{rest}"


def _collection_fields(config: BuildCorpusConfig, outcome: BuildOutcome) -> dict:
    """How the records were collected and read, and what the build ran under: the manifest's
    ``validation_rules``, ``record_flags``, ``placement_thresholds``,
    ``keyword_definitions_digest``, ``decode_off_list``, ``damaged_tails``, ``join_gaps`` and
    ``cut_games``, the survey's: the write pass reads the same bytes, and a difference is warned of
    (:func:`report_write_pass`)."""
    import math

    totals, decisions = outcome.survey.totals, outcome.decisions
    reading = totals.reading
    flags: dict[str, dict[str, int]] = {}
    for (klass, flag), count in sorted(totals.flags.items()):
        flags.setdefault(klass, {})[flag] = count
    digest = outcome.keywords.digest if outcome.keywords is not None else ""
    return {
        "validation_rules": {str(stratum): planned.rows
                             for stratum, planned in sorted(decisions.validation_budgets.items())},
        "record_flags": flags,
        "placement_thresholds": {
            name: None if math.isinf(value) else value
            for name, value in (("min_games_per_half", config.min_games_per_half),
                                ("max_item_error", config.max_item_error),
                                ("max_component_error", config.max_component_error))},
        "keyword_definitions_digest": digest,
        "decode_off_list": {f"{member} {value}": count
                            for (member, value), count in sorted(reading.off_list.items())},
        "damaged_tails": tuple(sorted(
            _under(where, config.records_dir) for where in reading.damaged_tails)),
        "unsupervised_events": dict(sorted(totals.unsupervised.items())),
        "cut_games": len(totals.cut.games),
        "code_version": code_version(),
        "sidecar_digest": config.source_trees.converted_sidecar_digest(),
        "join_gaps": _join_gaps(reading),
    }


def _join_gaps(reading) -> dict[str, dict[str, int]]:
    """The survey's join gaps by kind, each sorted; an empty kind left out."""
    return {kind: dict(sorted(getattr(reading, kind).items()))
            for kind in JOIN_GAP_KINDS if getattr(reading, kind)}


def log_report(
    manifest: CorpusManifest, survey: Survey, decisions: Decisions, written: WriteResult,
) -> None:
    """The operator's report, read off the manifest the build just wrote."""
    _log_counts(manifest)
    _log_sources(survey)
    _log_refusals(manifest, written)
    _log_mixture(manifest)
    logger.info(
        "game-disjoint %d game(s), %d of them under the keyword threshold; "
        "%d held-out text(s) with no gate-one resolution record, %d under %d games",
        len(decisions.strata.game_disjoint), len(decisions.keyword_threshold_games),
        len(manifest.held_out_texts_without_resolution),
        len(manifest.held_out_texts_under_five_games), HELD_OUT_GAME_FLOOR,
    )
    _log_rules(manifest, decisions, written)
    _log_reading(manifest)


def _log_rules(manifest: CorpusManifest, decisions: Decisions, written: WriteResult) -> None:
    """The rule budgets, the halves and what was left out (FR-054-FR-060)."""
    logger.info(
        "%d tagged rule(s), %d budgeted; copies added: %s; protection copies: %s",
        len(manifest.rule_map), len(manifest.budgeted_rules),
        dict(sorted(written.added.items())) or "none",
        dict(sorted(written.protection.items())) or "none",
    )
    _log_one_sided(manifest)
    logger.info("%d (class, rule, polarity) half(s) short of their share", _short_halves(manifest))
    _log_halves(manifest, decisions)


def _log_one_sided(manifest: CorpusManifest) -> None:
    """The rules and the conditions seen on one side only, by class."""
    for klass, rules in sorted(manifest.one_sided_rules.items()):
        logger.info("%-22s %d one-sided rule(s): %s", klass, len(rules), ", ".join(rules[:10]))
    for klass, carriers in sorted(manifest.one_sided_conditions.items()):
        logger.info("%-22s condition(s) on one side only: %s", klass, ", ".join(carriers))


def _short_halves(manifest: CorpusManifest) -> int:
    """The training ``(class, rule, polarity)`` rows short of their share."""
    return sum(
        1 for by_rule in manifest.rules.values() for rows in by_rule.values()
        for row in rows.values() if row.shortfall
    )


def _log_halves(manifest: CorpusManifest, decisions: Decisions) -> None:
    """Each stratum's halves, the rules each half leaves out, and the
    turn-action records written nowhere."""
    for stratum, by_half in sorted(decisions.strata.halves.items()):
        logger.info(
            "%-22s halves: %s", stratum,
            ", ".join(f"{half} {len(games)} game(s)" for half, games in by_half.items()),
        )
    for stratum, by_half in sorted(manifest.left_out_of_half.items()):
        for half, rules in sorted(by_half.items()):
            logger.info("%-22s %s leaves out %d rule(s)", stratum, half, len(rules))
    logger.info(
        "turn_action record(s) written nowhere: %s",
        dict(manifest.turn_action_left_out) or "none",
    )


def _log_reading(manifest: CorpusManifest) -> None:
    """What reading the raw shards met without stopping: reported, never fatal."""
    if manifest.decode_off_list:
        logger.warning(
            "%d value(s) outside the wire's closed lists were read as written "
            "(manifest decode_off_list): %s", sum(manifest.decode_off_list.values()),
            ", ".join(f"{k}: {v}" for k, v in list(manifest.decode_off_list.items())[:5]))
    if manifest.damaged_tails:
        logger.warning(
            "%d raw shard(s) ended at a damaged last gzip member (manifest "
            "damaged_tails), e.g. %s", len(manifest.damaged_tails), manifest.damaged_tails[0])
    if manifest.unsupervised_events:
        logger.info(
            "events no output column reads, by kind (manifest unsupervised_events): %s",
            manifest.unsupervised_events)
    for kind, counts in manifest.join_gaps.items():
        log = logger.info if kind in EXPECTED_JOIN_GAPS else logger.warning
        log("%d key(s) read as no line, %s (manifest join_gaps), e.g. %s",
            sum(counts.values()), kind, ", ".join(list(counts)[:3]))


def _log_counts(manifest: CorpusManifest) -> None:
    """Records and unique texts per class and per stratum, and the legality halves."""
    for name, counts in manifest.per_class.items():
        logger.info(
            "%-22s read %9d  kept %9d  cap dropped %8d  unique texts %7d",
            name, counts.read, counts.kept, counts.dropped_by_cap, counts.unique_texts,
        )
    for name in OUTPUT_STRATA:
        logger.info(
            "%-22s %9d record(s)  %7d unique text(s)",
            name, manifest.per_stratum[name], manifest.unique_texts[name],
        )
    for stratum in DROPPED_STRATA:
        logger.info("%-22s %9d record(s)", stratum, manifest.per_stratum[stratum])
    for subkind, counts in sorted(manifest.legality_counts.items()):
        logger.info(
            "%-22s real %9d  what-if %9d",
            f"legality {subkind}", counts.get(LegalityDecision.REAL, 0),
            counts.get(LegalityDecision.WHAT_IF, 0),
        )


def _log_sources(survey: Survey) -> None:
    """Held-out games per source directory, warning on the share that means a leak."""
    for source in sorted(survey.games_by_source):
        games = survey.games_by_source[source]
        held = survey.held_out_games_by_source.get(source, 0)
        share = held / games if games else 0.0
        leak = 0 < share < _LEAK_SHARE
        (logger.warning if leak else logger.info)(
            "%-22s %7d game(s), %6d name a held-out card (%.1f%%)%s",
            source, games, held, 100.0 * share,
            " — a few held-out games in a directory that should hold none is "
            "a leak in collection, not a full-strength source" if leak else "",
        )


def _log_refusals(manifest: CorpusManifest, written: WriteResult) -> None:
    """Each refusal with its classes, then the join refusals' worst scripts."""
    for reason, count in sorted(manifest.quality_dropped.items()):
        by_class = manifest.quality_dropped_by_class.get(reason, {})
        logger.info(
            "%-22s %9d record(s) refused (%s)", reason, count,
            ", ".join(f"{klass} {n}" for klass, n in sorted(by_class.items())),
        )
    for label, scripts in (
        (QualityDefect.NO_ACTING_TEXT, written.refusals.textless_scripts),
        (QualityDefect.UNJOINABLE_KEY, written.refusals.unjoinable_scripts),
    ):
        if scripts:
            logger.info(
                "%s refusals by script, most first: %s", label,
                ", ".join(f"{script} ({count})" for script, count in scripts.most_common(5)),
            )


def _log_mixture(manifest: CorpusManifest) -> None:
    """The class mixture and targets requested against delivered, and every shortfall."""
    _log_class_mix(manifest)
    _log_targets(manifest)
    _log_shortfalls(manifest)


def _log_class_mix(manifest: CorpusManifest) -> None:
    """Each class's requested share against the delivered one."""
    requested, delivered = manifest.class_mix, manifest.delivered_mix
    for name in sorted(set(requested) | set(delivered)):
        asked, got = requested.get(name, 0.0), delivered.get(name, 0.0)
        logger.info(
            "%-22s requested %6.2f%%  delivered %6.2f%%  (%.2fx)",
            name, 100.0 * asked, 100.0 * got, (got / asked) if asked else 0.0,
        )


def _log_targets(manifest: CorpusManifest) -> None:
    """Each class's target against what was kept, warning on a short one."""
    for name, target in sorted(manifest.class_targets.items()):
        kept = manifest.per_class[name].kept if name in manifest.per_class else 0
        short = kept < (1.0 - _DELIVERY_TOLERANCE) * target
        (logger.warning if short else logger.info)(
            "%-22s targeted %9d  delivered %9d  (%.1f%%)%s",
            name, target, kept, 100.0 * kept / target if target else 0.0,
            " — the plan counted records the write pass did not deliver" if short else "",
        )


def _log_shortfalls(manifest: CorpusManifest) -> None:
    """Every class short of its share, then every family short of its share."""
    for name, missing in sorted(manifest.shortfall.items()):
        logger.warning(
            "%s is short %d record(s) of its share: --training-records asks for "
            "more than the corpus can supply at this mixture.", name, missing,
        )
    for klass, per_family in sorted(manifest.families.items()):
        for family, row in sorted(per_family.items()):
            if row.shortfall:
                logger.warning(
                    "%-22s family %-28s short %7d of its share %d (%d available, "
                    "%d written, %d repeats) — collect more",
                    klass, family, row.shortfall, row.share, row.available,
                    row.written, row.repeats,
                )
