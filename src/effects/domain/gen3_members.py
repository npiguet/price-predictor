"""The members a gen-3 record writes, site by site: one table.

A site is one kind of JSON object on the wire: the envelope, a block of the
snapshot, one kind's payload, a contribution, a candidate, an event. Each row
names a member of its site and says whether the writer puts it on every
instance (``always``: null or empty where there is nothing to say) or only
``when`` a condition holds, read off the Java writer
(``forge-connector/.../effects/``): a member it leaves out when unset is a
``when`` row.

Three readers derive from the table and keep no list of their own:
``validate-corpus``'s presence check owes every ``always`` row on every
instance of its site (:func:`site_blocks` finds the instances on a raw
record), ``field-coverage`` rows every member by the paths its record-model
walk reaches (:data:`WALK_PATHS`) and tags the members constant by
construction (:data:`FIXED_PATHS`) apart from a dead field, and the gen-3
decode indexes each member :data:`ALWAYS` names for its site instead of
defaulting it.

The table stops at the sites below; a member's own nested objects (an
entity's ``history``, a cost half's ``costs``) are not sites yet.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

from effects.domain.records import Moment, PlayabilitySubkind, RecordKind, TurnActionSubkind


class Site(Enum):
    """A kind of JSON object on the wire, by the name the presence check
    reports it under."""

    RECORD = "record"
    STATE = "state"
    GLOBAL = "state.global"
    REFS = "state.refs"
    PLAYER = "state.players[]"
    ENTITY = "state.entities[]"
    LINE = "state.entities[].lines[]"
    STATIC_LINE = "state.entities[].lines[static]"
    KEYWORD_LINE = "state.entities[].lines[keyword]"
    LINK = "state.links[]"
    COST_HALF = "payload<resolution/activation>"
    EFFECT_HALF = "payload<resolution/resolution>"
    REWRITE = "payload<rewrite>"
    CONTINUOUS = "payload<continuous>"
    CONTRIBUTION = "payload<continuous>.contributions[]"
    COMBAT = "payload<combat>"
    TRIGGER = "payload<trigger>"
    DECISION = "payload<playability/decision>"
    CANDIDATE = "payload<playability/decision>.candidates[]"
    VERDICT = "payload<playability/decision>.candidates[].verdict"
    FAILED_CHECK = "payload<playability/decision>.candidates[].failed_checks[]"
    ATTACKERS = "payload<playability/attackers>"
    BLOCKERS = "payload<playability/blockers>"
    FORBIDDEN = "payload<playability>.forbidden[]"
    TURN_ACTION = "payload<turn_action>"
    GAME_END = "payload<turn_action/game_end>"
    #: Any event: a payload's, a rewrite's incoming and outgoing ones, a
    #: trigger's, the snapshot's pending one.
    EVENT = "event"

    @property
    def walks(self) -> tuple[str, ...]:
        """Where ``field-coverage``'s walk of the decoded record reaches this
        site's instances: the record model's path, one per place it sits."""
        return _WALKS[self]


_PAYLOAD = "record.payload"
_DECISION = f"{_PAYLOAD}<PlayabilityDecisionPayload>.candidates[]"
_LINES = "record.state.entities[].lines[]"
_WALKS: dict[Site, tuple[str, ...]] = {
    Site.RECORD: ("record",),
    Site.STATE: ("record.state",),
    Site.GLOBAL: ("record.state.global_",),
    Site.REFS: ("record.state.refs",),
    Site.PLAYER: ("record.state.players[]",),
    Site.ENTITY: ("record.state.entities[]",),
    Site.LINE: (_LINES,),
    Site.STATIC_LINE: (_LINES,),
    Site.KEYWORD_LINE: (_LINES,),
    Site.LINK: ("record.state.links[]",),
    Site.COST_HALF: (f"{_PAYLOAD}<ActivationPayload>",),
    Site.EFFECT_HALF: (f"{_PAYLOAD}<ResolutionPayload>",),
    Site.REWRITE: (f"{_PAYLOAD}<RewritePayload>",),
    Site.CONTINUOUS: (f"{_PAYLOAD}<ContinuousPayload>",),
    Site.CONTRIBUTION: (f"{_PAYLOAD}<ContinuousPayload>.contributions[]",),
    Site.COMBAT: (f"{_PAYLOAD}<CombatPayload>",),
    Site.TRIGGER: (f"{_PAYLOAD}<TriggerPayload>",),
    Site.DECISION: (f"{_PAYLOAD}<PlayabilityDecisionPayload>",),
    # The decode folds a candidate's verdict into the candidate.
    Site.CANDIDATE: (_DECISION,),
    Site.VERDICT: (_DECISION,),
    Site.FAILED_CHECK: (f"{_DECISION}.failed_checks[]",),
    Site.ATTACKERS: (f"{_PAYLOAD}<PlayabilityAttackersPayload>",),
    Site.BLOCKERS: (f"{_PAYLOAD}<PlayabilityBlockersPayload>",),
    Site.FORBIDDEN: (f"{_PAYLOAD}<PlayabilityAttackersPayload>.forbidden[]",
                     f"{_PAYLOAD}<PlayabilityBlockersPayload>.forbidden[]"),
    Site.TURN_ACTION: (f"{_PAYLOAD}<TurnActionPayload>",),
    Site.GAME_END: (f"{_PAYLOAD}<TurnActionPayload>",),
    Site.EVENT: (
        f"{_PAYLOAD}<ActivationPayload>.events[]", f"{_PAYLOAD}<ResolutionPayload>.events[]",
        f"{_PAYLOAD}<CombatPayload>.events[]", f"{_PAYLOAD}<RewritePayload>.incoming",
        f"{_PAYLOAD}<RewritePayload>.outgoing", f"{_PAYLOAD}<RewritePayload>.buffered_events[]",
        f"{_PAYLOAD}<TriggerPayload>.event", f"{_PAYLOAD}<TurnActionPayload>.events[]",
        "record.state.pending_event",
    ),
}


@dataclass(frozen=True, slots=True)
class Member:
    """One row: a member of a site, when the writer puts it there, and the
    record-model fields it decodes into."""

    site: Site
    name: str
    #: None when every instance of the site carries it; else the condition.
    when: str | None = None
    #: Why the member is constant by construction, where it is: a value the
    #: writer fixes, which no corpus can vary.
    fixed: str | None = None
    #: The decoded fields, relative to the site's walk path; None for the
    #: one of the member's own name, ``()`` for none the walk reaches.
    fields: tuple[str, ...] | None = None

    @property
    def always(self) -> bool:
        return self.when is None

    def walk_paths(self) -> tuple[str, ...]:
        names = (self.name,) if self.fields is None else self.fields
        return tuple(f"{walk}.{name}" for walk in self.site.walks for name in names)


def _always(site: Site, *names: str) -> tuple[Member, ...]:
    return tuple(Member(site, name) for name in names)


_CURATED = "a curated shard's record: build-corpus writes it, no collector does"

#: Every member a gen-3 record writes, in the writer's order within a site.
MEMBERS: tuple[Member, ...] = (
    Member(Site.RECORD, "record_id"),
    Member(Site.RECORD, "generation",
           fixed="3 on every record: the decode refuses any other generation"),
    *_always(Site.RECORD, "run_id", "timestamp", "game_id", "kind", "moment", "subkind",
             "link_id", "mirror_of"),
    Member(Site.RECORD, "variant_of", fixed="written null: no collector sets it"),
    *_always(Site.RECORD, "mode", "interventional", "fork"),
    Member(Site.RECORD, "fork_kind", when="a fork's record (fork true)"),
    Member(Site.RECORD, "synthetic",
           fixed="false from every collector; only build-corpus's protection copies are"),
    Member(Site.RECORD, "random_seat"),
    Member(Site.RECORD, "what_if", when="a blockers record (its own finding)"),
    Member(Site.RECORD, "aimed_at",
           when="a collect-coverage or collect-variants run's record, or a directed fork's"),
    Member(Site.RECORD, "directed", when="a directed fork's record"),
    Member(Site.RECORD, "copy_dropped",
           when="a fork's record whose copy left something out"),
    *_always(Site.RECORD, "actor_player", "ability", "ability_unresolved", "state"),
    Member(Site.RECORD, "payload", fields=()),
    Member(Site.RECORD, "curation", when=_CURATED, fixed=_CURATED),

    Member(Site.STATE, "global", fields=("global_",)),
    *_always(Site.STATE, "players", "entities", "links", "refs", "pending_event"),
    Member(Site.STATE, "tiers", fixed="the writer's frozen [1, 2, 3, 4]: every tier collected"),

    *_always(Site.GLOBAL, "turn", "phase", "active", "priority", "stack_size", "combat_substep",
             "emblems", "day_night", "floating_effects", "delayed_triggers", "dungeons"),
    *_always(Site.REFS, "targets", "targets_at_cast", "targets_on_stack", "source", "modes",
             "x", "choices", "random"),

    *_always(Site.PLAYER, "id", "life", "hand", "library", "graveyard", "poison", "energy",
             "this_turn"),
    # Null for an unlimited count (Fastbond), which the decode reads as its own flag.
    Member(Site.PLAYER, "max_land_plays", fields=("max_land_plays", "land_plays_unlimited")),
    *_always(Site.PLAYER, "floating_mana", "untapped_production", "speed", "city_blessing",
             "rad", "monarch", "initiative", "turn_counts"),

    *_always(Site.ENTITY, "id", "name", "zone", "owner", "controller", "face", "face_state",
             "copy_source", "token_script_id", "types", "subtypes", "supertypes", "colors",
             "mana_value", "mana_cost"),
    Member(Site.ENTITY, "base_pt", fields=("pt", "pt.base")),
    Member(Site.ENTITY, "pt_boosts", fields=("pt.boosts", "pt.counters")),
    *_always(Site.ENTITY, "printed_loyalty", "printed_defense", "tapped", "sick", "damage",
             "counters", "combat", "attached_to", "face_down", "real_face", "designations",
             "x_paid", "chosen", "costs_paid", "history", "order_rank", "granted_temporary",
             "lines", "stack_extras", "role"),

    *_always(Site.LINE, "keys", "kind", "granted"),
    Member(Site.LINE, "grant", when="a granted line: why it is granted"),
    Member(Site.LINE, "active",
           when="a static line (always); another on a face not up, a Room's door or a "
                "Class level"),
    Member(Site.LINE, "cast_with", when="a keyword line its spell was cast with (kicked)"),
    Member(Site.LINE, "chosen", when="the line on the stack, with the modes chosen"),
    Member(Site.LINE, "fragment", when="a keyword line (always)"),
    Member(Site.LINE, "affected",
           when="a static of the face up applied as a continuous effect, outside a "
                "continuous record's board"),
    Member(Site.STATIC_LINE, "active"),
    Member(Site.KEYWORD_LINE, "fragment"),

    Member(Site.LINK, "from", fields=("from_",)),
    *_always(Site.LINK, "to", "type"),
    Member(Site.LINK, "index", when="a targets link: the target's place in its clause"),

    *_always(Site.COST_HALF, "events", "costs"),
    *_always(Site.EFFECT_HALF, "events", "conditions_met", "outcome"),
    *_always(Site.REWRITE, "incoming", "outgoing", "result", "replaced_by"),
    Member(Site.REWRITE, "incoming_unknown", when="an incoming event Forge's bus does not name"),
    *(Member(Site.REWRITE, name, when="the replacement set its sub-ability aside (buffered)")
      for name in ("buffered", "buffered_ran", "buffered_events")),
    *_always(Site.CONTINUOUS, "contributions", "board_hash"),
    *_always(Site.CONTRIBUTION, "entity", "pt_boost"),
    Member(Site.CONTRIBUTION, "set_pt", when="the contribution sets power and toughness"),
    *_always(Site.CONTRIBUTION, "keywords", "types", "colors", "abilities", "hidden_keywords",
             "name"),
    Member(Site.CONTRIBUTION, "controller", when="the contribution changes control"),
    *_always(Site.COMBAT, "attackers", "blocks", "assignment_choices", "events"),
    *(Member(Site.COMBAT, name, when="a damage-step probe's record (fork_kind probe)")
      for name in ("probed_keyword", "probed_entity")),
    *_always(Site.TRIGGER, "event", "fired"),

    Member(Site.DECISION, "candidates"),
    Member(Site.CANDIDATE, "ability"),
    Member(Site.CANDIDATE, "verdict", fields=()),
    *_always(Site.CANDIDATE, "legal_targets", "cost_after_adjustment", "responsible_static",
             "failed_checks", "land_play"),
    *_always(Site.VERDICT, "can_play", "affordable", "has_legal_target", "timing"),
    *_always(Site.FAILED_CHECK, "check", "responsible_static"),

    *_always(Site.ATTACKERS, "anchor_attacker", "legal_defenders", "must_attack"),
    Member(Site.ATTACKERS, "tapped_by_attack", when="the anchor was declared an attacker"),
    *_always(Site.ATTACKERS, "legal_attackers", "forbidden"),
    *_always(Site.BLOCKERS, "anchor_attacker", "legal_blockers", "forbidden", "must_block",
             "min_blockers"),
    *_always(Site.FORBIDDEN, "entity", "responsible_static"),
    Member(Site.FORBIDDEN, "responsible_own", when="the responsible line is the creature's own"),
    Member(Site.FORBIDDEN, "responsible_unresolved", when="the responsible line has no key"),
    Member(Site.FORBIDDEN, "required_elsewhere",
           when="a requirement (a lure) holds the blocker elsewhere"),

    Member(Site.TURN_ACTION, "events"),
    *_always(Site.GAME_END, "events", "records_withheld"),
    *_always(Site.EVENT, "type", "subjects", "params", "duration", "attributed_to"),
)

#: Per site, the members every instance carries, in the table's order: what
#: the presence check owes and the gen-3 decode indexes rather than defaults.
ALWAYS: Mapping[Site, tuple[str, ...]] = {
    site: tuple(row.name for row in MEMBERS if row.site is site and row.always)
    for site in Site
}


#: Every path ``field-coverage``'s walk reaches for a member of the table,
#: once each, in the table's order.
WALK_PATHS: tuple[str, ...] = tuple(dict.fromkeys(
    path for row in MEMBERS for path in row.walk_paths()))

#: The walk paths of the members constant by construction, and why each is.
FIXED_PATHS: Mapping[str, str] = {
    path: row.fixed for row in MEMBERS if row.fixed for path in row.walk_paths()
}


# ── the strict gen-3 decode ─────────────────────────────────────────────


class MissingMemberError(ValueError):
    """A gen-3 block without a member the writer puts on every instance of
    its site."""


#: :data:`ALWAYS` as sets, the one shape a block's keys compare against.
_OWED: Mapping[Site, frozenset[str]] = {site: frozenset(names) for site, names in ALWAYS.items()}


def require_always(site: Site, block: Mapping) -> None:
    """Refuse a block of ``site`` without a member :data:`ALWAYS` owes it,
    naming each missing one as ``<site>.<member>``."""
    if not block.keys() >= _OWED[site]:
        missing = ", ".join(f"{site.value}.{name}" for name in ALWAYS[site] if name not in block)
        raise MissingMemberError(f"no {missing}: a gen-3 writer puts it on every {site.value}")


def owes[T](site: Site) -> Callable[[Callable[[Mapping], T]], Callable[[Mapping], T]]:
    """Make the gen-3 decode of one ``site``'s block name what it lacks.

    The decode reads every member :data:`ALWAYS` owes the site by index, so
    a missing one raises ``KeyError``; this turns it into the
    :class:`MissingMemberError` naming each owed member the block lacks. A
    ``KeyError`` with every owed member present (a member of a nested object
    the table does not cover) goes on as raised. Checked only on a failure:
    a check of every block before reading it cost the training loader a
    fifth of its decode time.
    """
    def strict(decode: Callable[[Mapping], T]) -> Callable[[Mapping], T]:
        def decode_owed(block: Mapping) -> T:
            try:
                return decode(block)
            except KeyError:
                require_always(site, block)
                raise
        return decode_owed
    return strict


#: The line kinds that owe members of their own beside a line's.
_LINE_KINDS: Mapping[str, Site] = {"static": Site.STATIC_LINE, "keyword": Site.KEYWORD_LINE}


def require_line_kind(block: Mapping) -> None:
    """Refuse a line without a member its kind owes beyond a line's
    (:data:`Site.STATIC_LINE`, :data:`Site.KEYWORD_LINE`), which the line's
    decode reads as optional on a line of another kind."""
    kind = _LINE_KINDS.get(block["kind"])
    if kind is not None:
        require_always(kind, block)


# ── where each site sits on a raw record ────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Child:
    """A site nested in another, under one member: one object or a list of
    them, a list's items narrowed to one line ``kind`` where named."""

    member: str
    site: Site
    many: bool = False
    kind: str | None = None

    def blocks(self, parent: Mapping) -> Iterator[Mapping]:
        value = parent.get(self.member)
        items = value if self.many and isinstance(value, list) else (value,)
        return (item for item in items if self._takes(item))

    def _takes(self, item: object) -> bool:
        return isinstance(item, Mapping) and self.kind in (None, item.get("kind"))


_CHILDREN: dict[Site, tuple[_Child, ...]] = {
    Site.RECORD: (_Child("state", Site.STATE),),
    Site.STATE: (
        _Child("global", Site.GLOBAL), _Child("refs", Site.REFS),
        _Child("players", Site.PLAYER, many=True), _Child("entities", Site.ENTITY, many=True),
        _Child("links", Site.LINK, many=True), _Child("pending_event", Site.EVENT),
    ),
    Site.ENTITY: (
        _Child("lines", Site.LINE, many=True),
        _Child("lines", Site.STATIC_LINE, many=True, kind="static"),
        _Child("lines", Site.KEYWORD_LINE, many=True, kind="keyword"),
    ),
    Site.COST_HALF: (_Child("events", Site.EVENT, many=True),),
    Site.EFFECT_HALF: (_Child("events", Site.EVENT, many=True),),
    Site.REWRITE: (
        _Child("incoming", Site.EVENT), _Child("outgoing", Site.EVENT),
        _Child("buffered_events", Site.EVENT, many=True),
    ),
    Site.CONTINUOUS: (_Child("contributions", Site.CONTRIBUTION, many=True),),
    Site.COMBAT: (_Child("events", Site.EVENT, many=True),),
    Site.TRIGGER: (_Child("event", Site.EVENT),),
    Site.DECISION: (_Child("candidates", Site.CANDIDATE, many=True),),
    Site.CANDIDATE: (
        _Child("verdict", Site.VERDICT), _Child("failed_checks", Site.FAILED_CHECK, many=True),
    ),
    Site.ATTACKERS: (_Child("forbidden", Site.FORBIDDEN, many=True),),
    Site.BLOCKERS: (_Child("forbidden", Site.FORBIDDEN, many=True),),
    Site.TURN_ACTION: (_Child("events", Site.EVENT, many=True),),
    Site.GAME_END: (_Child("events", Site.EVENT, many=True),),
}

#: A record's payload site by its kind and its moment or subkind.
_PAYLOADS: dict[tuple[str, str | None], Site] = {
    (RecordKind.RESOLUTION, Moment.ACTIVATION): Site.COST_HALF,
    (RecordKind.RESOLUTION, Moment.RESOLUTION): Site.EFFECT_HALF,
    (RecordKind.REWRITE, None): Site.REWRITE,
    (RecordKind.CONTINUOUS, None): Site.CONTINUOUS,
    (RecordKind.COMBAT, None): Site.COMBAT,
    (RecordKind.TRIGGER, None): Site.TRIGGER,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.DECISION): Site.DECISION,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.ATTACKERS): Site.ATTACKERS,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.BLOCKERS): Site.BLOCKERS,
    **{(RecordKind.TURN_ACTION, subkind): Site.TURN_ACTION for subkind in TurnActionSubkind},
    (RecordKind.TURN_ACTION, TurnActionSubkind.GAME_END): Site.GAME_END,
}


def payload_site(record: Mapping) -> Site | None:
    """The site of a raw record's payload; None for a kind the table does
    not know, which the closed-list check names."""
    return _PAYLOADS.get((record.get("kind"), record.get("moment") or record.get("subkind")))


def site_blocks(record: Mapping) -> Iterator[tuple[Site, Mapping]]:
    """Every instance of every site on one raw record, each with its site."""
    yield from _nested(Site.RECORD, record)
    site, payload = payload_site(record), record.get("payload")
    if site is not None and isinstance(payload, Mapping):
        yield from _nested(site, payload)


def _nested(site: Site, block: Mapping) -> Iterator[tuple[Site, Mapping]]:
    yield site, block
    for child in _CHILDREN.get(site, ()):
        for item in child.blocks(block):
            yield from _nested(child.site, item)
