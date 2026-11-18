"""Board abilities: the model's error where a permanent's own text decides (FR-085b, FR-074).

A snapshot lists every line a card carries now, statics, triggers and
replacements among them, each an ability slot of its own. This probe measures
what those lines are worth: on each subset below, every output field's loss
and the ablation of ``e`` (``ablation.run_ablation``), beside a control subset
of the same kind. Noise in the card slots replaces those lines' ``e`` too, so a
subset whose loss rises under it more than its control's is one where the model
reads the static, trigger or replacement it was not given by anything else.

======================  ====================================================
subset                  records
======================  ====================================================
legality-static         ``attackers``/``blockers``: a creature is kept out by
                        a static that is not a keyword line
legality-keyword        kept out by keyword lines only (protection, say),
                        which the keyword flags carry
legality-none           nobody is kept out
resolution-replacement  a resolution half, with a replacement line on a
                        permanent other than the ability's source
resolution-static       a non-keyword static line on another permanent, and
                        no replacement
resolution-none         no other permanent carries a static, trigger or
                        replacement line
======================  ====================================================

Records come from the corpus's validation strata, read in shard order, the
first ``per_subset`` of each subset per stratum, so every arm on one corpus
reads the same records without a frozen list.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

SUBSETS: tuple[str, ...] = (
    "legality-static", "legality-keyword", "legality-none",
    "resolution-replacement", "resolution-static", "resolution-none",
)
#: Records per subset per validation stratum.
RECORDS_PER_SUBSET = 300
#: Validation shards read per stratum before giving up on a short subset.
SHARDS_PER_STRATUM = 200
#: The line kinds a permanent's keyword flags do not carry.
UNLISTED_KINDS = frozenset({"static", "trigger", "replacement"})


def _unlisted(line) -> bool:
    return line.kind in UNLISTED_KINDS and not line.granted and line.in_force


def unlisted_kinds(entity) -> frozenset[str]:
    """The non-keyword static, trigger and replacement lines on ``entity``'s face.

    Its own lines in force (``entity.lines``), granted ones left out. A keyword
    line is left out whatever trait it folds in, because the keyword flags
    already carry it.
    """
    return frozenset(line.kind for line in entity.lines if _unlisted(line))


def _keyword_static(keys, sidecars) -> bool | None:
    """Whether a responsible static is a keyword line; None where it resolves to none."""
    for key in keys:
        line = sidecars.line_or_none(key)
        if line is not None:
            return line.keyword_display_name is not None
    return None


def _legality_subset(record, sidecars) -> str | None:
    """The subset of an attackers or blockers record, or None."""
    forbidden = record.payload.forbidden
    if not forbidden:
        return "legality-none"
    verdicts = [_keyword_static(entry.responsible_static, sidecars)
                for entry in forbidden if entry.responsible_static]
    if any(verdict is False for verdict in verdicts):
        return "legality-static"
    if verdicts and all(verdict is True for verdict in verdicts):
        return "legality-keyword"
    return None


def _resolution_subset(record) -> str | None:
    """The subset of a resolution half by the lines on the battlefield's other permanents."""
    source = record.state.refs.source
    kinds = frozenset().union(*(
        unlisted_kinds(entity) for entity in record.state.entities
        if entity.zone == "battlefield" and entity.id != source))
    if "replacement" in kinds:
        return "resolution-replacement"
    if "static" in kinds:
        return "resolution-static"
    return None if kinds else "resolution-none"


def subset_of(record, sidecars) -> str | None:
    """The subset ``record`` belongs to, or None."""
    from effects.domain.records import Moment, PlayabilitySubkind, RecordKind

    if record.kind is RecordKind.PLAYABILITY and record.subkind in (
            PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS):
        return _legality_subset(record, sidecars)
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        return _resolution_subset(record)
    return None


def collect(records: Iterable, sidecars, *, per_subset: int = RECORDS_PER_SUBSET
            ) -> dict[str, list]:
    """The first ``per_subset`` records of each subset, in the order read."""
    chosen: dict[str, list] = {name: [] for name in SUBSETS}
    for record in records:
        if _add(chosen, record, sidecars, per_subset) and _full(chosen, per_subset):
            break
    return chosen


def _add(chosen: dict[str, list], record, sidecars, per_subset: int) -> bool:
    """File ``record`` under its subset unless that is full; whether it was filed."""
    name = subset_of(record, sidecars)
    if name is None or len(chosen[name]) >= per_subset:
        return False
    chosen[name].append(record)
    return True


def _full(chosen: dict[str, list], per_subset: int) -> bool:
    return all(len(rows) >= per_subset for rows in chosen.values())


def run(probe_model, chosen: Mapping[str, Mapping[str, list]], stats) -> dict:
    """``{subset: {"n": {stratum: n}, "ablation": ...}}``, both strata pooled."""
    from effect_knowledge_probes import ablation

    out: dict[str, dict] = {}
    for name in SUBSETS:
        records = [r for by_subset in chosen.values() for r in by_subset.get(name, [])]
        out[name] = {"n": {stratum: len(by_subset.get(name, []))
                           for stratum, by_subset in chosen.items()}}
        if records:
            out[name]["ablation"] = ablation.run_ablation(probe_model, records, stats)
    return out


def table(result: Mapping[str, dict]) -> str:
    """Per field, each subset's loss and the cost of noise in the card slots."""
    from effect_knowledge_probes.board_table import render

    return render(result)
