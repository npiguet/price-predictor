"""The detector categories: the records the rule detectors are tested on and
the shapes the detector families' parameters read."""

from __future__ import annotations

from collections.abc import Callable

from effects.domain.rule_detectors.continuous_contribution import condition_counter_types
from effects.domain.rule_detectors.rewrite_event import PREVENTS
from effects.domain.script_text import params as params_of
from effects.domain.script_text import root as root_of
from fixture_records.shapes import (
    acting_key,
    candidate_key,
    entities,
    entities_by_id,
    events,
    is_effect_resolution,
    params,
    subjects,
)
from fixture_records.sidecars import Sidecars
from fixture_records.state_based import (
    left_at_toughness_one,
    lethal_damage_death,
    sba_fires,
    toughness_zero_death,
)

#: Rule 111.10's predefined token kinds the fixture looks for.
_PREDEFINED_TOKENS = frozenset({"treasure", "food", "clue", "blood", "map", "gold",
                                "powerstone", "shard", "incubator", "junk"})


def _keyword_trigger(fired: bool) -> Callable[[dict], bool]:
    """A trigger a keyword generates: its acting key is the keyword's trait."""
    def test(record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "trigger" and bool(record["payload"].get("fired")) == fired
                and key is not None and key["trait_kind"] == "keyword")
    return test


def _subtypes(type_line: str) -> set[str]:
    _, dash, rest = type_line.partition(" - ")
    return {word.lower() for word in rest.split()} if dash else set()


def _makes_predefined_token(event: dict) -> bool:
    characteristics = params(event).get("characteristics") or {}
    return bool(_subtypes(str(characteristics.get("types") or "")) & _PREDEFINED_TOKENS)


def _predefined_token_made(record: dict) -> bool:
    return is_effect_resolution(record) and any(
        event["type"] == "token_created" and _makes_predefined_token(event)
        for event in events(record))


def _is_predefined_token(entity: dict) -> bool:
    return bool(entity.get("token_script_id")) and bool(
        {subtype.lower() for subtype in entity.get("subtypes") or ()} & _PREDEFINED_TOKENS)


def _predefined_token_source(record: dict) -> bool:
    if not is_effect_resolution(record):
        return False
    source = (record["state"].get("refs") or {}).get("source")
    return any(entity["id"] == source and _is_predefined_token(entity)
               for entity in entities(record))


def _leaves_battlefield(event: dict) -> bool:
    return event["type"] == "zone_change" and params(event).get("from_zone") == "battlefield"


def _in_combat(entity: dict | None) -> bool:
    combat = (entity or {}).get("combat") or {}
    return bool(combat.get("attacking") or combat.get("blocking"))


def combatant_leaves(record: dict) -> bool:
    """An effect half moving a creature that is attacking or blocking off the battlefield."""
    if not is_effect_resolution(record):
        return False
    found = entities_by_id(record)
    return any(_in_combat(found.get(subject)) for event in events(record)
               if _leaves_battlefield(event) for subject in subjects(event))


def _keyword_decision(record: dict) -> bool:
    key = candidate_key(record)
    return key is not None and key["trait_kind"] == "keyword"


def _targeting_decision(record: dict) -> bool:
    if candidate_key(record) is None:
        return False
    targets = record["payload"]["candidates"][0].get("legal_targets") or ()
    return 0 < len(targets) < 32


def _keyword_resolution(record: dict) -> bool:
    key = acting_key(record)
    return (is_effect_resolution(record) and key is not None
            and key["trait_kind"] == "keyword" and bool(events(record)))


def _token_static(record: dict) -> bool:
    key = acting_key(record)
    return (record["kind"] == "continuous" and key is not None
            and key["script_file"].startswith("tokenscripts/"))


class _DetectorShapes:
    """Records the detector families' parameters read where the record alone
    does not tell: a candidate's card printing a keyword, a cost part, a
    prevention shield, a counter-conditioned static."""

    def __init__(self, sidecars: Sidecars) -> None:
        self.sidecars = sidecars

    def keyword_card_decision(self, record: dict) -> bool:
        key = candidate_key(record)
        return (key is not None and key["trait_kind"] == "spell"
                and self.sidecars.prints_keyword(key["script_file"]))

    def cost_part_decision(self, record: dict) -> bool:
        key = candidate_key(record)
        return key is not None and "<" in (
            params_of(root_of(self.sidecars.script(key))).get("Cost") or "")

    def prevention_rewrite(self, record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "rewrite" and key is not None
                and record["payload"]["incoming"]["type"] == "damage_dealt"
                and record["payload"].get("result") in ("replaced", "updated")
                and bool(PREVENTS.search(self.sidecars.script(key))))

    def counter_conditioned_static(self, record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "continuous" and key is not None
                and bool(condition_counter_types(self.sidecars.script(key))))


def detector_shapes(sidecars: Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Records the detector families' parameters read: a keyword's own line
    acting, a candidate's card printing a keyword, a cost part, targets, a
    prevention shield, a combatant leaving, a token's or a counter-conditioned
    static."""
    shapes = _DetectorShapes(sidecars)
    return {
        "keyword-resolution": _keyword_resolution,
        "keyword-decision": _keyword_decision,
        "keyword-card-decision": shapes.keyword_card_decision,
        "cost-part-decision": shapes.cost_part_decision,
        "targeting-decision": _targeting_decision,
        "prevention-rewrite": shapes.prevention_rewrite,
        "combatant-leaves": combatant_leaves,
        "token-static": _token_static,
        "counter-conditioned-static": shapes.counter_conditioned_static,
    }


def _probe_deathtouch(record: dict) -> bool:
    return record["kind"] == "combat" and record["payload"].get("probed_keyword") == "deathtouch"


def _random_seat(record: dict) -> bool:
    return record.get("random_seat") is True


def detector_categories(sidecars: Sidecars) -> dict[str, Callable[[dict], bool]]:
    """The detector categories, matched before the shared ones."""
    return {
        **detector_shapes(sidecars),
        "704.5j-fires": sba_fires("legend_rule"),
        "704.5m-fires": sba_fires("aura_illegal"),
        "704.5i-fires": sba_fires("loyalty_zero"),
        "704.5f-fires": toughness_zero_death,
        "704.5g-fires": lethal_damage_death,
        "704.5f-does-not-fire": left_at_toughness_one,
        "keyword-trigger-fired": _keyword_trigger(True),
        "keyword-trigger-unfired": _keyword_trigger(False),
        "predefined-token-source": _predefined_token_source,
        "predefined-token-made": _predefined_token_made,
        "probe-deathtouch": _probe_deathtouch,
        "random-seat": _random_seat,
    }
