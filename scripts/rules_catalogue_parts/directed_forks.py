"""``directed-rules``: the rules enabled for directed forks, from gen-3's per-rule counts."""

import json

from effects.domain.variant_rounds import DEFAULT_STEER_FLOOR

from .directed_exclusions import ALWAYS_DIRECTED, NOT_DIRECTED, RARE_FIRES
from .ordering import rule_sort_key, rule_unit, unit_sort_key
from .reading import load_scenarios_by_rule

FIRES, DOES_NOT_FIRE = "fires", "does_not_fire"


def has_detector(scenario):
    """Whether the scenario's detector is a registered family: on gen-3
    records every such detector tags what its record kind carries."""
    from effects.domain.rule_detectors import families

    declaration = scenario.get("detector") or {}
    return declaration.get("family") in families()


#: The source of ``rule-counts`` no count here sums: a probe's, an
#: intervention's or a protection fork's record re-records a moment of its
#: live game, a combat once per probe (F01).
FORK_SOURCE = "fork"
#: The sources the list is decided from: the natural records alone. A
#: directed fork's records (``aimed``) are the forks' own, so a rule whose
#: side only they show would read as covered, leave the list and stop the
#: forks that make them (F15).
DECIDING_SOURCES = frozenset({"forge_ai", "random_seat"})


def side_records(by_source, sources=None):
    """One side's records over ``sources``; every source but ``fork`` where None."""
    return sum(n for source, n in by_source.items() if _summed(source, sources))


def _summed(source, sources):
    return source != FORK_SOURCE if sources is None else source in sources


def rule_counts(counts, rule, scenarios, sources=None):
    """``(fires, does_not_fire)`` over ``sources`` (every source but ``fork``
    where None), or None where the counts cannot speak for the rule. The
    counts list only rules some record tagged, so a rule they omit has none
    of either when a detector of its tags records; one not countable,
    declared ``none``, or with no detector is not shown by them at all."""
    found = counts["rules"].get(rule)
    if found is not None:
        return (side_records(found.get(FIRES, {}), sources),
                side_records(found.get(DOES_NOT_FIRE, {}), sources))
    if rule in counts["not_countable"] or rule in counts["none"]:
        return None
    return (0, 0) if any(has_detector(s) for s in scenarios) else None


def lacking_polarities(fires, does_not_fire):
    """The sides the counts lack: `fires` on a rare rule, `does_not_fire` on a one-sided one."""
    return (({FIRES} if fires < RARE_FIRES else set())
            | ({DOES_NOT_FIRE} if does_not_fire == 0 else set()))


def forkable_variants(scenarios, lacking, registered):
    """``(scenario id, label, polarity, operation)`` of every non-base variant
    classified into an operation and in ``registered`` (the labels the Java
    registry registers for the rule), those of a polarity the counts lack first."""
    found = [(s["id"], v["label"], v["polarity"], v["operation"])
             for s in scenarios for v in s.get("variants", [])[1:]
             if isinstance(v.get("operation"), str) and v["label"] in registered]
    return sorted(found, key=lambda f: f[2] not in lacking)


def registered_sides(rules_dir):
    """``rule -> {variant label -> Polarity}`` of every registered rule: the
    collector's own definition of a rule's sides (``forkable_sides``)."""
    from effects.domain.directed_sides import forkable_sides
    from effects.infrastructure.rules_catalogue_io import load_registered_variants, load_scenarios

    registered = load_registered_variants(rules_dir)
    return forkable_sides(load_scenarios(rules_dir), list(registered), registered)


def counted_sides(variants):
    """The polarities of a rule's registered variants that name a side: the
    collector counts a rule's forks under these alone, ``other`` is no side."""
    from effects.domain.directed_sides import COUNTED_SIDES

    return frozenset(str(p) for p in variants.values() if p in COUNTED_SIDES)


def directed_rules(counts, by_rule, sides):
    """The rules FR-014b enables, in unit order: ``rule -> {counts, natural,
    lacking, variants, by_counts, sides}``; ``counts`` are over every source
    but ``fork``, ``natural`` over ``DECIDING_SOURCES``, which decide
    ``lacking``; ``by_counts`` is False for a rule listed only as one of
    ``ALWAYS_DIRECTED``; ``sides`` are the counted polarities of the rule's
    registered variants, ``sides`` being ``registered_sides``. A rule of
    ``NOT_DIRECTED`` or one no variant is registered for is never listed."""
    out = {}
    for rule, scenarios in by_rule.items():
        if rule in NOT_DIRECTED or not sides.get(rule):
            continue
        natural = rule_counts(counts, rule, scenarios, DECIDING_SOURCES)
        lacking = lacking_polarities(*natural) if natural is not None else set()
        by_counts = bool(lacking)
        if not by_counts and rule not in ALWAYS_DIRECTED:
            continue
        variants = forkable_variants(scenarios, lacking, sides[rule])
        if variants:
            out[rule] = {"counts": rule_counts(counts, rule, scenarios), "natural": natural,
                         "lacking": lacking, "variants": variants,
                         "by_counts": by_counts, "sides": counted_sides(sides[rule])}
    return dict(sorted(out.items(), key=lambda item: (unit_sort_key(rule_unit(item[0])),
                                                      rule_sort_key(item[0]))))


def melds_into(text, face):
    """Whether the card's own face (the text before ``ALTERNATE``) is the one
    whose script melds the pair into ``face``: since the meld halves both print
    the melded face, only the holder's front face names it."""
    return f"meld them into {face}" in text.partition("\nALTERNATE")[0]


def converted_card_of_face(cards_folder):
    """``face name -> converted card name`` over a converted tree: a scenario
    may name a back face (Ribbons, a meld result), which the coverage round
    knows only by the card's own ``name:``, its first. A face several files
    print (both meld halves print the melded face) maps to the file whose first
    face melds into it, else to the sorted-first path, never by listing order."""
    from price_predictor.domain.tokenizer import extract_card_name

    out, holder = {}, set()
    for path in sorted(cards_folder.rglob("*.txt"), key=lambda p: p.as_posix()):
        text = path.read_text(encoding="utf-8")
        card = extract_card_name(text)
        for line in text.splitlines():
            if not line.startswith("name:"):
                continue
            face = line[len("name:"):].strip()
            if face in holder:
                continue
            if melds_into(text, face):
                holder.add(face)
                out[face] = card
            else:
                out.setdefault(face, card)
        out[card] = card
    return out


def wording_cards(chosen, by_rule, card_of_face):
    """The cards of the scenarios of the rules enabled for their counts, as
    the converted lower-case names (FR-038a), and the folded names no
    converted card carries."""
    from effects.domain.card_names import fold_card_name

    named = {fold_card_name(card) for rule, row in chosen.items() if row["by_counts"]
             for scenario in by_rule[rule] for card in scenario.get("cards", [])}
    unknown = sorted(name for name in named if name not in card_of_face)
    return sorted({card_of_face.get(name, name) for name in named}), unknown


#: The part of the counts ``directed_rules`` reads, checked in beside the list.
DIRECTED_COUNTS = "directed-fork-counts.json"


def decided_from(counts):
    """The members of a ``rule-counts`` file ``directed_rules`` reads: each
    rule's records by polarity and source, and the ids of the rules the
    counts cannot speak for (``not_countable``, ``none``)."""
    return {"rules": counts["rules"], "not_countable": sorted(counts["not_countable"]),
            "none": sorted(counts["none"])}


def write_decided_from(path, counts):
    """``decided_from(counts)`` as JSON, one rule a line."""
    kept = decided_from(counts)
    rules = ",\n".join(f"    {json.dumps(rule)}: {json.dumps(kept['rules'][rule], sort_keys=True)}"
                       for rule in sorted(kept["rules"], key=rule_sort_key))
    path.write_text("{\n"
                    f'  "not_countable": {json.dumps(kept["not_countable"])},\n'
                    f'  "none": {json.dumps(kept["none"])},\n'
                    f'  "rules": {{\n{rules}\n  }}\n'
                    "}\n", encoding="utf-8", newline="\n")


def directed_side_records(chosen):
    """``rule -> {side -> records}`` of each listed rule over the sides its
    registered variants make (the collector's sides), each side's records 0
    for a rule the counts cannot speak for. A rule whose registered variants
    are all polarity ``other`` has no side to fill and no entry:
    ``match-outcomes`` gives it the run's ``--directed-fork-probability``, as
    ``collect-variants`` does."""
    out = {}
    for rule, row in chosen.items():
        by_side = dict(zip((FIRES, DOES_NOT_FIRE), row["counts"] or (0, 0), strict=True))
        if row["sides"]:
            out[rule] = {side: by_side[side] for side in sorted(row["sides"])}
    return out


def probabilities_document(chosen, count):
    """The file ``match-outcomes --directed-fork-probabilities`` reads: the
    records a side needs and each listed rule's records per side, from which
    it computes each rule's probability at start and as its workers keep
    forks (``sealed.infrastructure.directed_fork_probabilities_file``, its one
    reader)."""
    return {"count_per_side": count, "rules": directed_side_records(chosen)}


def _write_lists(rules_dir, chosen, counts):
    """The enabled rules and the counts they were decided from."""
    (rules_dir / "directed-fork-rules.txt").write_text(
        "".join(f"{rule}\n" for rule in chosen), encoding="utf-8", newline="\n")
    write_decided_from(rules_dir / DIRECTED_COUNTS, counts)


def _write_probabilities(args, chosen):
    """With ``--probabilities-output``, each listed rule's records per registered side."""
    output = getattr(args, "probabilities_output", None)
    if output is None:
        return
    document = probabilities_document(
        chosen, getattr(args, "count_per_side", DEFAULT_STEER_FLOOR))
    output.write_text(json.dumps(document, indent=1) + "\n",
                      encoding="utf-8", newline="\n")


def _why_listed(row):
    """``rare`` and ``one-sided``, sorted, for the sides the counts lack."""
    return ", ".join(sorted(({"rare"} if FIRES in row["lacking"] else set())
                            | ({"one-sided"} if DOES_NOT_FIRE in row["lacking"] else set())))


def _print_rule(rule, row):
    fires, does_not_fire = row["natural"] if row["natural"] is not None else ("?", "?")
    print(f"{rule:12s} fires {fires:>9} does_not_fire {does_not_fire:>9}  "
          f"{_why_listed(row) or 'always'}")
    for scenario_id, label, polarity, operation in row["variants"]:
        mark = "*" if polarity in row["lacking"] else " "
        print(f"    {mark} {scenario_id} {label} ({polarity}): {operation}")


def _print_summary(chosen, cards):
    rare = sum(FIRES in row["lacking"] for row in chosen.values())
    one_sided = sum(DOES_NOT_FIRE in row["lacking"] for row in chosen.values())
    for rule, reason in ALWAYS_DIRECTED.items():
        print(f"always listed {rule}: {reason}")
    for rule, reason in NOT_DIRECTED.items():
        print(f"left out {rule}: {reason}")
    print(f"{len(chosen)} rules ({rare} rare, {one_sided} one-sided, "
          f"{sum(not row['by_counts'] for row in chosen.values())} always), {len(cards)} wording "
          f"cards; * marks a variant of a polarity the counts lack")


def cmd_directed_rules(args):
    rules_dir = args.rules_dir
    counts = json.loads(args.counts.read_text(encoding="utf-8"))
    by_rule = load_scenarios_by_rule(rules_dir)
    chosen = directed_rules(counts, by_rule, registered_sides(rules_dir))
    cards, unknown = wording_cards(chosen, by_rule, converted_card_of_face(args.cards_folder))
    _write_lists(rules_dir, chosen, counts)
    _write_probabilities(args, chosen)
    (rules_dir / "coverage-wording-cards.txt").write_text(
        "".join(f"{card}\n" for card in cards), encoding="utf-8", newline="\n")
    for rule, row in chosen.items():
        _print_rule(rule, row)
    _print_summary(chosen, cards)
    if unknown:
        print(f"no converted card under {args.cards_folder}: {', '.join(unknown)}")
