"""What ``build-corpus`` records about a build, checks after it, and tells the operator.

The operator's report is read back off the manifest, so the log and the file
cannot disagree.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from effects.application.corpus_survey import Survey
from effects.application.corpus_write import WriteResult
from effects.domain.corpus_plan import BuildCorpusError, Keys
from effects.domain.record_quality import QualityDefect
from effects.domain.records import LegalityDecision
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.domain.text_holdout import HoldoutUnit

if TYPE_CHECKING:
    from effects.application.build_corpus import BuildCorpusConfig, Decisions
    from effects.domain.corpus_manifest import CorpusManifest
    from effects.domain.split import HeldOutCards

logger = logging.getLogger(__name__)

#: How much of one sampling class the quality rules may refuse before the
#: build stops, rather than silently losing a whole class.
MAX_REFUSED_SHARE = 0.5

#: Unjoinable-key records the build tolerates; it stops only past both, so a
#: small pilot does not stop on its one Flashback-cast charm.
MAX_UNJOINABLE_SHARE = 0.001
MAX_UNJOINABLE_RECORDS = 20

#: A held-out text recorded in fewer games than this is listed in the manifest
#: (FR-053), the floor the held-out coverage round collects to (FR-036).
HELD_OUT_GAME_FLOOR = 5

#: A directory whose share of held-out games is above zero and below this
#: holds a few held-out games where it should hold none: a leak in collection,
#: not a full-strength source.
_LEAK_SHARE = 0.05

#: A class delivered more than this far under its target is reported as short;
#: keyless cells are sampled by hash rate, exact in expectation only.
_DELIVERY_TOLERANCE = 0.01


def check_refusals(written: WriteResult) -> None:
    """Stop when the quality rules refused most of any one sampling class, or
    past both unjoinable-key limits over every record read: those are the join
    failing, not a record being poor."""
    refusals = written.refusals
    unjoinable = refusals.by_reason.get(QualityDefect.UNJOINABLE_KEY, 0)
    total_read = sum(written.training.read.values()) + sum(refusals.by_class.values())
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
    for name in sorted(refusals.by_class):
        refused = refusals.by_class[name]
        total = refused + written.training.read.get(name, 0)
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
    """The write pass's refusals and attribution, checked and logged."""
    check_refusals(written)
    kept_records = sum(written.training.read.values())
    logger.info(
        "%-22s %9d record(s) kept (%.1f%%) name no producing clause for at "
        "least one event — watched, not refused; a rising share is the "
        "collector's attribution, not the corpus",
        "unattributed", written.unattributed,
        100.0 * written.unattributed / kept_records if kept_records else 0.0,
    )
    if survey.refusals.by_reason != written.refusals.by_reason:
        logger.warning(
            "The survey and the write pass disagree about which records are "
            "refused: survey %s, write pass %s. Both read the same shards "
            "under the same rule, so this means the corpus changed between "
            "the passes.",
            dict(sorted(survey.refusals.by_reason.items())),
            dict(sorted(written.refusals.by_reason.items())),
        )
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
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]]]:
    """The manifest's per-family and per-signature tables (FR-053)."""
    families: dict[str, dict[str, dict[str, int]]] = {}
    for klass, per_family in decisions.selection.families.items():
        for label, row in per_family.items():
            wrote = written.families.written.get((klass, label), 0)
            distinct = written.families.distinct.get((klass, label), 0)
            families.setdefault(klass, {})[label] = {
                "available": row.available,
                "share": row.share,
                "planned": row.planned,
                "written": wrote,
                "repeats": wrote - distinct,
                "shortfall": max(0, row.share - wrote),
            }
    signatures: dict[str, dict[str, dict[str, int]]] = {}
    for (klass, label, signature), count in sorted(written.families.signatures.items()):
        if signature:
            signatures.setdefault(klass, {}).setdefault(label, {})[signature] = count
    return families, signatures


def held_out_text_report(
    survey: Survey, written: WriteResult, held_out_texts: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Held-out texts with no gate-one record, and those resolved in under five games."""
    from effects.domain.text_holdout import normalize_script_text

    held_games = survey.games.held_out
    texts = {normalize_script_text(text) for text in held_out_texts}
    without = tuple(sorted(texts - written.strata.gate_one_texts))
    games = {
        text: len(survey.games.held_out_text_games.get(text, set()) & held_games)
        for text in sorted(texts)
    }
    under = {text: count for text, count in games.items() if count < HELD_OUT_GAME_FLOOR}
    return without, under


def _unique_texts(keys: Iterable[Keys], key_text: Mapping[Keys, str]) -> int:
    """Distinct ability texts behind ``keys``: what a record count cannot say."""
    return len({key_text.get(key, key) for key in keys})


def build_manifest(
    config: BuildCorpusConfig, *, surface: str, held_out: HeldOutCards, survey: Survey,
    decisions: Decisions, written: WriteResult,
) -> CorpusManifest:
    """Everything the build decided and wrote, as the manifest records it."""
    from effects.domain.corpus_manifest import ClassCounts, CorpusManifest

    training = written.training
    per_class = {
        name: ClassCounts(
            read=training.read.get(name, 0),
            kept=training.kept.get(name, 0),
            dropped_by_cap=training.dropped_by_cap.get(name, 0),
            unique_texts=_unique_texts(training.kept_keys.get(name, ()), decisions.key_text),
        )
        for name in sorted(training.read)
    }
    trained = sum(training.kept.values())
    delivered_mix = {
        name: count / trained for name, count in sorted(training.kept.items())
    } if trained else {}
    families, signatures = family_tables(decisions, written)
    without_resolution, under_floor = held_out_text_report(survey, written, held_out.texts)
    strata = written.strata
    return CorpusManifest(
        seed=config.seed,
        surface=surface,
        vocab_path=config.vocab_path,
        variant_scripts=config.variant_scripts or "",
        holdout_permille=config.holdout_permille,
        holdout_max_carriers=config.holdout_max_carriers,
        text_cap=config.text_cap,
        card_disjoint_text_cap=config.card_disjoint_text_cap,
        training_records=config.training_records,
        class_mix=config.mix(),
        delivered_mix=delivered_mix,
        held_out_cards=tuple(sorted(held_out.names)),
        held_out_texts=tuple(sorted(held_out.texts)),
        card_disjoint_games=tuple(sorted(decisions.strata.card_disjoint)),
        game_disjoint_games=tuple(sorted(decisions.strata.game_disjoint)),
        rarity=decisions.rarity,
        sources=survey.shards,
        per_class=per_class,
        # The two dropped strata write nothing, so have no unique texts.
        per_stratum={name: strata.records.get(name, 0) for name in Stratum},
        unique_texts={
            name: _unique_texts(strata.keys.get(name, ()), decisions.key_text)
            for name in OUTPUT_STRATA
        },
        shortfall=decisions.selection.shortfall,
        class_targets=dict(decisions.selection.targets),
        quality_dropped=dict(written.refusals.by_reason),
        # The hundred worst: every permanent carries the implicit spell.
        no_acting_text_scripts=dict(written.refusals.textless_scripts.most_common(100)),
        games_by_source=survey.games_by_source,
        held_out_games_by_source=survey.held_out_games_by_source,
        shard_records=config.shard_records,
        validation_sample=config.validation_sample,
        max_events_per_record=config.max_events,
        unattributed_records=written.unattributed,
        holdout_unit=HoldoutUnit(config.holdout_unit),
        families=families,
        signatures=signatures,
        policy_counts=_nested(written.families.policy),
        legality_counts=_nested(written.families.legality),
        legality_mode_counts=dict(sorted(decisions.mode_counts.items())),
        keyword_threshold_games=tuple(sorted(decisions.keyword_threshold_games)),
        held_out_texts_without_resolution=without_resolution,
        held_out_texts_under_five_games=under_floor,
        game_disjoint_share=config.game_disjoint_share,
        game_disjoint_keyword_share=config.game_disjoint_keyword_share,
        game_disjoint_keywords=tuple(config.game_disjoint_keywords),
        reuse_cap=config.reuse_cap,
    )


def log_report(
    manifest: CorpusManifest, survey: Survey, decisions: Decisions, written: WriteResult,
) -> None:
    """The operator's report, read off the manifest the build just wrote."""
    _log_counts(manifest)
    _log_sources(survey)
    _log_refusals(written)
    _log_mixture(manifest)
    logger.info(
        "game-disjoint %d game(s), %d of them under the keyword threshold; "
        "%d held-out text(s) with no gate-one resolution record, %d under %d games",
        len(decisions.strata.game_disjoint), len(decisions.keyword_threshold_games),
        len(manifest.held_out_texts_without_resolution),
        len(manifest.held_out_texts_under_five_games), HELD_OUT_GAME_FLOOR,
    )


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
    for stratum in (Stratum.DROPPED_HELD_OUT, Stratum.DROPPED_HELD_OUT_TEXT):
        logger.info("%-22s %9d record(s)", stratum, manifest.per_stratum[stratum])
    for subkind, counts in sorted(manifest.legality_counts.items()):
        logger.info(
            "%-22s real %9d  what-if %9d  unknown %9d",
            f"legality {subkind}", counts.get(LegalityDecision.REAL, 0),
            counts.get(LegalityDecision.WHAT_IF, 0),
            counts.get(LegalityDecision.UNKNOWN, 0),
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


def _log_refusals(written: WriteResult) -> None:
    for reason, count in sorted(written.refusals.by_reason.items()):
        logger.info("%-22s %9d record(s) refused", reason, count)
    for label, scripts in (
        ("no-acting-text", written.refusals.textless_scripts),
        ("unjoinable-key", written.refusals.unjoinable_scripts),
    ):
        if scripts:
            logger.info(
                "%s refusals by script, most first: %s", label,
                ", ".join(f"{script} ({count})" for script, count in scripts.most_common(5)),
            )


def _log_mixture(manifest: CorpusManifest) -> None:
    """The class mixture and targets requested against delivered, and every shortfall."""
    requested, delivered = manifest.class_mix, manifest.delivered_mix
    for name in sorted(set(requested) | set(delivered)):
        asked, got = requested.get(name, 0.0), delivered.get(name, 0.0)
        logger.info(
            "%-22s requested %6.2f%%  delivered %6.2f%%  (%.2fx)",
            name, 100.0 * asked, 100.0 * got, (got / asked) if asked else 0.0,
        )
    for name, target in sorted(manifest.class_targets.items()):
        kept = manifest.per_class[name].kept if name in manifest.per_class else 0
        short = kept < (1.0 - _DELIVERY_TOLERANCE) * target
        (logger.warning if short else logger.info)(
            "%-22s targeted %9d  delivered %9d  (%.1f%%)%s",
            name, target, kept, 100.0 * kept / target if target else 0.0,
            " — the plan counted records the write pass did not deliver" if short else "",
        )
    for name, missing in sorted(manifest.shortfall.items()):
        logger.warning(
            "%s is short %d record(s) of its share: --training-records asks for "
            "more than the corpus can supply at this mixture.", name, missing,
        )
    for klass, per_family in sorted(manifest.families.items()):
        for family, row in sorted(per_family.items()):
            if row["shortfall"]:
                logger.warning(
                    "%-22s family %-28s short %7d of its share %d (%d available, "
                    "%d written, %d repeats) — collect more",
                    klass, family, row["shortfall"], row["share"], row["available"],
                    row["written"], row["repeats"],
                )
