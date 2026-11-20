"""Does Forge's combat AI steer around deathtouch and indestructible?

A combat record exists only when Forge attacked, so counting keyword
combats measures the attacks Forge chose and hides the ones it declined.
This script reads the decision points instead. A playability ``attackers``
record is written at every declare-attackers step, with the legal attackers
and the full board, whether or not anyone then attacks. A ``blockers``
record is written per attacker at declare-blockers, with its legal
blockers. Joining each to the combat record of the same game and turn says
what Forge actually chose among the options it had.

Attacking. Each legal attacker is classed by the most dangerous creature
that could block it: nothing, only creatures it survives, a creature that
kills it and dies too (a trade), one that kills it and survives (a free
kill), a deathtouch creature that kills it, or an indestructible creature
that kills it. The attack rate per class is the measurement. If Forge
treats deathtouch and indestructible as it treats the damage they deal,
their rows sit with the trade and free-kill rows, not with the safe row.

Blocking. Each legal (attacker, blocker) pair is classed by what the block
would do, and the class says whether a keyword on the attacker is what
turns it. The block rate per class is the measurement.

Only the two-player case is read. Creatures with first strike or double
strike are left out on either side, because they change who kills whom
before damage is compared. Evasion is approximated when building the
potential blockers of an attacker (flying needs flying or reach); the
blocking half reads Forge's own legal-blocker lists.

Usage
-----
    python scripts/analyze_keyword_combat_avoidance.py \\
        --cards-folder <cardsfolder> --tokenscripts <tokenscripts> \\
        --variant-scripts <variant-scripts> \\
        <records>/validation/game-disjoint <records>/validation/card-disjoint

Point it only at directories whose games are whole (the validation strata or
the raw corpus). The curated training directory samples records per record,
so its decision records and combat records no longer pair up.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from _lib.markdown import md_table

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.records import PlayabilitySubkind, RecordKind
from effects.infrastructure.shard_io import read_records
from effects.infrastructure.sidecar_io import SidecarCache

STRIKES_FIRST = {"first_strike", "double_strike"}

#: What a block does when neither side's keyword turns it, by
#: (the blocker kills the attacker, the attacker kills the blocker).
_ORDINARY_BLOCKS = {
    (True, True): "ordinary trade",
    (True, False): "ordinary free kill for the blocker",
    (False, True): "ordinary chump block",
    (False, False): "ordinary, nobody dies",
}


def _is_creature(entity) -> bool:
    return entity.is_creature and entity.on_battlefield


def _kills(hitter, hit, kw_hitter, kw_hit) -> bool:
    """Would ``hitter``'s combat damage destroy ``hit``?"""
    if "indestructible" in kw_hit or hitter.power < 1:
        return False
    return "deathtouch" in kw_hitter or hitter.power >= hit.remaining_toughness


def attack_class(attacker, blockers, kw) -> str:
    """The worst block ``attacker`` faces, as one label."""
    if not blockers:
        return "no blocker"
    ka = kw(attacker)
    killers = [b for b in blockers if _kills(b, attacker, kw(b), ka)]
    if not killers:
        return "only blockers it survives"
    ordinary = [b for b in killers if not ({"deathtouch", "indestructible"} & kw(b))]
    if ordinary:
        return _ordinary_attack_class(attacker, ordinary, ka, kw)
    return _keyword_attack_class(attacker, killers, kw)


def _ordinary_attack_class(attacker, ordinary, ka, kw) -> str:
    """The worst block when keyword-less creatures can kill ``attacker``."""
    if any(not _kills(attacker, b, ka, kw(b)) for b in ordinary):
        return "ordinary blocker kills it and survives"
    return "ordinary blocker trades with it"


def _keyword_attack_class(attacker, killers, kw) -> str:
    """The worst block when only deathtouch or indestructible creatures kill it."""
    if any("indestructible" in kw(b) for b in killers):
        return "indestructible blocker kills it"
    # Only deathtouch creatures can kill it. Split by whether it would have
    # survived them without the keyword, which is the case deathtouch decides.
    if all(b.power < attacker.remaining_toughness for b in killers):
        return "deathtouch blocker kills it (survives it otherwise)"
    return "deathtouch blocker kills it (dies anyway)"


def block_class(attacker, blocker, kw) -> str:
    ka, kb = kw(attacker), kw(blocker)
    blocker_kills = _kills(blocker, attacker, kb, ka)
    attacker_kills = _kills(attacker, blocker, ka, kb)
    if _deathtouch_decides(attacker, blocker, ka, attacker_kills):
        return (
            "trade, only because the attacker has deathtouch" if blocker_kills
            else "chump, only because the attacker has deathtouch"
        )
    if _indestructible_saves(attacker, blocker, ka, blocker_kills):
        return (
            "blocker dies, attacker saved by indestructible" if attacker_kills
            else "nobody dies, attacker saved by indestructible"
        )
    return _ORDINARY_BLOCKS[(blocker_kills, attacker_kills)]


def _deathtouch_decides(attacker, blocker, ka, attacker_kills) -> bool:
    """The attacker kills the blocker, and would not without deathtouch."""
    return "deathtouch" in ka and attacker_kills and attacker.power < blocker.remaining_toughness


def _indestructible_saves(attacker, blocker, ka, blocker_kills) -> bool:
    """The blocker's damage would kill the attacker but for indestructible."""
    return ("indestructible" in ka and not blocker_kills
            and blocker.power >= attacker.remaining_toughness)


def _evades(attacker, blocker, kw) -> bool:
    return "flying" in kw(attacker) and not ({"flying", "reach"} & kw(blocker))


def _unusable(entity, kw) -> bool:
    """Missing from the snapshot, or striking first (left out on either side)."""
    return entity is None or bool(STRIKES_FIRST & kw(entity))


def _potential_blocker(entity, active, kw) -> bool:
    """An untapped creature of the defending player's that does not strike first."""
    return (_is_creature(entity) and entity.controller != active and not entity.tapped
            and not (STRIKES_FIRST & kw(entity)))


def _not_the_attack_decision(record) -> bool:
    """The hook also fires while the AI simulates a combat it is only
    considering; the real decision is the active player's, at
    declare-attackers."""
    state = record.state
    return (
        len(state.players) != 2
        or state.global_.phase != "combat_declare_attackers"
        or record.actor_player != state.global_.active
    )


def _not_declared(attacker, kw) -> bool:
    """Only an attacker actually declared this turn is a real blocking
    decision; the rest are the attacking AI's hypotheticals."""
    return bool(
        attacker is None or attacker.combat is None
        or attacker.combat.attacking is None or STRIKES_FIRST & kw(attacker)
    )


@dataclass
class _Fresh:
    """The entities one game offers for the first time, the unusable left out."""

    kw: Callable
    seen: set = field(default_factory=set)

    def entities(self, state, ids, prefix: tuple) -> Iterator[tuple]:
        """``(id, entity)`` of each of ``ids`` whose key ``(*prefix, id)`` is new."""
        for entity_id in ids:
            key = (*prefix, entity_id)
            entity = state.entity(entity_id)
            if key in self.seen or _unusable(entity, self.kw):
                continue
            self.seen.add(key)
            yield entity_id, entity


class Tally:
    def __init__(self) -> None:
        self.attack = defaultdict(Counter)  # class -> {n, attacked}
        self.block = defaultdict(Counter)   # class -> {n, blocked}
        self.turns_with_decision = 0
        self.turns_with_combat = 0
        self.phases = Counter()

    def game(self, decisions, blocks, combats, kw) -> None:
        # combats: turn -> (attacking ids, set of (blocker, attacker) pairs)
        fresh_attack, fresh_block = _Fresh(kw), _Fresh(kw)
        for record in decisions:
            self._attack_decision(record, combats, fresh_attack)
        for record in blocks:
            self._block_decision(record, combats, fresh_block)

    def _attack_decision(self, record, combats, fresh: _Fresh) -> None:
        state = record.state
        turn = state.global_.turn
        g = state.global_
        self.phases[("attackers", g.phase, record.actor_player == g.active)] += 1
        if _not_the_attack_decision(record):
            return
        attacked, _pairs = combats.get(turn, (set(), set()))
        self.turns_with_decision += 1
        self.turns_with_combat += bool(attacked)
        kw = fresh.kw
        active = state.global_.active
        potential = [e for e in state.entities if _potential_blocker(e, active, kw)]
        for attacker_id, attacker in fresh.entities(
                state, record.payload.legal_attackers, (turn,)):
            can_block = [b for b in potential if not _evades(attacker, b, kw)]
            c = self.attack[attack_class(attacker, can_block, kw)]
            c["n"] += 1
            c["attacked"] += attacker_id in attacked

    def _block_decision(self, record, combats, fresh: _Fresh) -> None:
        state = record.state
        turn = state.global_.turn
        g = state.global_
        self.phases[("blockers", g.phase, record.actor_player == g.active)] += 1
        if len(state.players) != 2 or state.global_.phase != "combat_declare_blockers":
            return
        _attacked, pairs = combats.get(turn, (set(), set()))
        attacker = state.entity(record.payload.anchor_attacker)
        if _not_declared(attacker, fresh.kw):
            return
        for blocker_id, blocker in fresh.entities(
                state, record.payload.legal_blockers, (turn, attacker.id)):
            c = self.block[block_class(attacker, blocker, fresh.kw)]
            c["n"] += 1
            c["blocked"] += (blocker_id, attacker.id) in pairs


@dataclass
class _Game:
    """One game's decision records, and what its combat records chose."""

    decisions: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    #: turn -> (attacking ids, set of (blocker, attacker) pairs)
    combats: dict[int, tuple[set, set]] = field(default_factory=dict)

    def add(self, record) -> None:
        if record.kind is RecordKind.COMBAT:
            self._add_combat(record)
        elif record.kind is RecordKind.PLAYABILITY:
            self._add_playability(record)

    def _add_combat(self, record) -> None:
        attacked, pairs = self.combats.setdefault(record.state.global_.turn, (set(), set()))
        for e in record.state.entities:
            _note_combatant(e, attacked, pairs)

    def _add_playability(self, record) -> None:
        if record.subkind is PlayabilitySubkind.ATTACKERS:
            self.decisions.append(record)
        elif record.subkind is PlayabilitySubkind.BLOCKERS:
            self.blocks.append(record)


def _note_combatant(e, attacked: set, pairs: set) -> None:
    if e.combat is None:
        return
    if e.combat.attacking is not None:
        attacked.add(e.id)
    for target in e.combat.blocking or ():
        pairs.add((e.id, target))


def _games(records) -> Iterator[_Game]:
    """``records`` cut into games, each game's records consecutive."""
    current, game = None, _Game()
    for record in records:
        if record.game_id != current:
            if current is not None:
                yield game
            current, game = record.game_id, _Game()
        game.add(record)
    if current is not None:
        yield game


def run(directories: list[Path], sidecar_roots: dict[str, Path]) -> Tally:
    resolver = KeywordResolver(SidecarCache(sidecar_roots))
    kw = resolver.keywords_of
    tally = Tally()
    for directory in directories:
        for game in _games(read_records(directory)):
            tally.game(game.decisions, game.blocks, game.combats, kw)
    return tally


def _table(title, rows, outcome) -> str:
    ranked = sorted(rows.items(), key=lambda kv: -kv[1][outcome] / max(kv[1]["n"], 1))
    body = [(name, c["n"], f"{c[outcome] / c['n']:.1%}") for name, c in ranked]
    return f"\n{title}\n" + md_table(["class", "n", outcome], body, align="lrr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--cards-folder", type=Path, required=True,
                        help="the converted cardsfolder tree the records name")
    parser.add_argument("--tokenscripts", type=Path, required=True,
                        help="the converted tokenscripts tree the records name")
    parser.add_argument("--variant-scripts", type=Path, required=True,
                        help="the variant-script tree the records name")
    args = parser.parse_args()
    tally = run(args.directories, {
        "cardsfolder": args.cards_folder,
        "tokenscripts": args.tokenscripts,
        "variant-scripts": args.variant_scripts,
    })
    for key, n in tally.phases.most_common(12):
        print("record phase", key, n)
    print(
        f"declare-attackers decisions: {tally.turns_with_decision}, "
        f"of which followed by a combat record: {tally.turns_with_combat}"
    )
    print(_table("Attack rate by the worst block the attacker faces", tally.attack, "attacked"))
    print(_table("Block rate by what the block would do", tally.block, "blocked"))


if __name__ == "__main__":
    main()
