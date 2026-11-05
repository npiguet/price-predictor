"""Board abilities: the model's error where a permanent's own text decides (FR-085b, FR-074).

A gen-2 snapshot gave each permanent its spell and activated abilities as
card-slot lines, and none of its statics, triggers or replacements, so the
model read those only through what they had already done to the board: a
pump already in the toughness, a granted keyword already in the flags. What a
static decides at the moment of the record, such as who may block or what a
destroy turns into, reached the model as no text at all. A gen-3 snapshot lists
every line a card carries now, statics, triggers and replacements among them,
each an ability slot of its own. This probe measures what those lines are
worth: on each subset below, every output field's loss and the ablation of
``e`` (``ablation.run_ablation``), beside a control subset of the same kind.
On a gen-3 checkpoint, noise in the card slots replaces those lines' ``e`` too,
so a subset whose loss rises under it more than its control's is one where the
model reads the static, trigger or replacement it was not given in gen-2.

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
#: The trait kinds a gen-2 snapshot left out of a permanent's card slots,
#: spelled as both a provenance key's trait kind and a gen-3 line's kind.
UNLISTED_KINDS = frozenset({"static", "trigger", "replacement"})


def _sidecar_errors() -> tuple[type[Exception], ...]:
    from effects.infrastructure.sidecar_io import UnconfiguredTree, UnconvertedScript

    return UnconvertedScript, UnconfiguredTree


def unlisted_kinds(entity, sidecars) -> frozenset[str]:
    """The non-keyword static, trigger and replacement lines on ``entity``'s face.

    On a gen-3 entity, its own lines in force (``entity.lines``, granted ones
    left out, as gen-2's printed face had none); on a gen-1 or gen-2 one, the
    sidecar of the script its printed keys name. A keyword line is left out
    whatever trait it folds in, because the keyword flags already carry it.
    """
    if entity.lines is not None:
        return frozenset(line.kind for line in entity.lines
                         if line.kind in UNLISTED_KINDS and not line.granted
                         and line.in_force)
    if not entity.printed:
        return frozenset()
    try:
        sidecar = sidecars.get(entity.printed[0].script_file)
    except _sidecar_errors():
        return frozenset()
    kinds: set[str] = set()
    for line in sidecar.lines:
        if line.keyword_display_name is not None:
            continue
        for key in line.provenance:
            if key.face == entity.face and key.trait_kind in UNLISTED_KINDS:
                kinds.add(key.trait_kind)
    return frozenset(kinds)


def _keyword_static(keys, sidecars) -> bool | None:
    """Whether a responsible static is a keyword line; None where it resolves to none."""
    for key in keys:
        line = sidecars.line_or_none(key)
        if line is not None:
            return line.keyword_display_name is not None
    return None


def subset_of(record, sidecars) -> str | None:
    """The subset ``record`` belongs to, or None."""
    from effects.domain.records import Moment, PlayabilitySubkind, RecordKind

    if record.kind is RecordKind.PLAYABILITY and record.subkind in (
            PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS):
        verdicts = [_keyword_static(entry.responsible_static, sidecars)
                    for entry in record.payload.forbidden if entry.responsible_static]
        if not record.payload.forbidden:
            return "legality-none"
        if any(verdict is False for verdict in verdicts):
            return "legality-static"
        if verdicts and all(verdict is True for verdict in verdicts):
            return "legality-keyword"
        return None
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        source = record.state.refs.source
        kinds: set[str] = set()
        for entity in record.state.entities:
            if entity.zone == "battlefield" and entity.id != source:
                kinds |= unlisted_kinds(entity, sidecars)
        if "replacement" in kinds:
            return "resolution-replacement"
        if "static" in kinds:
            return "resolution-static"
        if not kinds:
            return "resolution-none"
    return None


def collect(records: Iterable, sidecars, *, per_subset: int = RECORDS_PER_SUBSET
            ) -> dict[str, list]:
    """The first ``per_subset`` records of each subset, in the order read."""
    chosen: dict[str, list] = {name: [] for name in SUBSETS}
    for record in records:
        name = subset_of(record, sidecars)
        if name is not None and len(chosen[name]) < per_subset:
            chosen[name].append(record)
            if all(len(rows) >= per_subset for rows in chosen.values()):
                break
    return chosen


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
    from math import isnan

    from _lib.markdown import md_table

    from effect_knowledge_probes.common import fmt

    def shown(values) -> bool:
        """A row is worth a line when some subset measured more than rounding."""
        return any(v is not None and not isnan(v) and abs(v) >= 0.0005 for v in values)

    present = [name for name in SUBSETS if result.get(name, {}).get("ablation")]
    fields = sorted({field for name in present for field in result[name]["ablation"]})
    lines = ["# Board abilities", "",
             "Records per subset: " + ", ".join(
                 f"{name} {sum(result[name]['n'].values())}" for name in SUBSETS
                 if name in result) + ".", "",
             "## Loss per field", ""]

    def per_field(value_of) -> str:
        rows = [[field, *(value_of(result[name]["ablation"].get(field, {}))
                          for name in present)] for field in fields]
        return md_table(["field", *present],
                        [[row[0], *map(fmt, row[1:])] for row in rows if shown(row[1:])],
                        "l" + "r" * len(present))

    lines += [per_field(lambda entry: entry.get("base_loss")),
              "", "## Loss increase with noise in the card slots", "",
              per_field(lambda entry: entry.get("noise", {}).get("cards"))]
    return "\n".join(lines) + "\n"
