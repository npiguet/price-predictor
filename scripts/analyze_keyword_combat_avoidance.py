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
        output/effects/corpus/validation/game-disjoint \\
        output/effects/corpus/validation/card-disjoint

Point it only at directories whose games are whole (the validation strata or
the raw corpus). The curated training directory samples records per record,
so its decision records and combat records no longer pair up.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.damage_step_keywords import _power, _remaining_toughness
from effects.domain.records import PlayabilitySubkind, RecordKind
from effects.infrastructure.record_io import read_records
from effects.infrastructure.sidecar_io import SidecarCache

STRIKES_FIRST = {"first_strike", "double_strike"}


def _is_creature(entity) -> bool:
    return "creature" in entity.types and entity.zone == "battlefield"


def _kills(hitter, hit, kw_hitter, kw_hit) -> bool:
    """Would ``hitter``'s combat damage destroy ``hit``?"""
    if "indestructible" in kw_hit or _power(hitter) < 1:
        return False
    return "deathtouch" in kw_hitter or _power(hitter) >= _remaining_toughness(hit)


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
        if any(not _kills(attacker, b, ka, kw(b)) for b in ordinary):
            return "ordinary blocker kills it and survives"
        return "ordinary blocker trades with it"
    if any("indestructible" in kw(b) for b in killers):
        return "indestructible blocker kills it"
    # Only deathtouch creatures can kill it. Split by whether it would have
    # survived them without the keyword, which is the case deathtouch decides.
    if all(_power(b) < _remaining_toughness(attacker) for b in killers):
        return "deathtouch blocker kills it (survives it otherwise)"
    return "deathtouch blocker kills it (dies anyway)"


def block_class(attacker, blocker, kw) -> str:
    ka, kb = kw(attacker), kw(blocker)
    blocker_kills = _kills(blocker, attacker, kb, ka)
    attacker_kills = _kills(attacker, blocker, ka, kb)
    if "deathtouch" in ka and attacker_kills and _power(attacker) < _remaining_toughness(blocker):
        return (
            "trade, only because the attacker has deathtouch" if blocker_kills
            else "chump, only because the attacker has deathtouch"
        )
    if "indestructible" in ka and not blocker_kills and _power(blocker) >= _remaining_toughness(
        attacker
    ):
        return (
            "blocker dies, attacker saved by indestructible" if attacker_kills
            else "nobody dies, attacker saved by indestructible"
        )
    if blocker_kills and attacker_kills:
        return "ordinary trade"
    if blocker_kills:
        return "ordinary free kill for the blocker"
    if attacker_kills:
        return "ordinary chump block"
    return "ordinary, nobody dies"


def _evades(attacker, blocker, kw) -> bool:
    return "flying" in kw(attacker) and not ({"flying", "reach"} & kw(blocker))


class Tally:
    def __init__(self) -> None:
        self.attack = defaultdict(Counter)  # class -> {n, attacked}
        self.block = defaultdict(Counter)   # class -> {n, blocked}
        self.turns_with_decision = 0
        self.turns_with_combat = 0
        self.phases = Counter()

    def game(self, decisions, blocks, combats, kw) -> None:
        # combats: turn -> (attacking ids, set of (blocker, attacker) pairs)
        seen_attack, seen_block = set(), set()
        for record in decisions:
            state = record.state
            turn = state.global_.turn
            g = state.global_
            self.phases[("attackers", g.phase, record.actor_player == g.active)] += 1
            # The hook also fires while the AI simulates a combat it is only
            # considering; the real decision is the active player's, at
            # declare-attackers.
            if (
                len(state.players) != 2
                or state.global_.phase != "combat_declare_attackers"
                or record.actor_player != state.global_.active
            ):
                continue
            attacked, _pairs = combats.get(turn, (set(), set()))
            self.turns_with_decision += 1
            self.turns_with_combat += bool(attacked)
            active = state.global_.active
            potential = [
                e for e in state.entities
                if _is_creature(e) and e.controller != active and not e.tapped
                and not (STRIKES_FIRST & kw(e))
            ]
            for attacker_id in record.payload.legal_attackers:
                key = (turn, attacker_id)
                attacker = state.entity(attacker_id)
                if key in seen_attack or attacker is None or STRIKES_FIRST & kw(attacker):
                    continue
                seen_attack.add(key)
                can_block = [b for b in potential if not _evades(attacker, b, kw)]
                c = self.attack[attack_class(attacker, can_block, kw)]
                c["n"] += 1
                c["attacked"] += attacker_id in attacked
        for record in blocks:
            state = record.state
            turn = state.global_.turn
            g = state.global_
            self.phases[("blockers", g.phase, record.actor_player == g.active)] += 1
            if len(state.players) != 2 or state.global_.phase != "combat_declare_blockers":
                continue
            _attacked, pairs = combats.get(turn, (set(), set()))
            attacker = state.entity(record.payload.anchor_attacker)
            # Only an attacker actually declared this turn is a real blocking
            # decision; the rest are the attacking AI's hypotheticals.
            if (
                attacker is None or attacker.combat is None
                or attacker.combat.attacking is None or STRIKES_FIRST & kw(attacker)
            ):
                continue
            for blocker_id in record.payload.legal_blockers:
                key = (turn, attacker.id, blocker_id)
                blocker = state.entity(blocker_id)
                if key in seen_block or blocker is None or STRIKES_FIRST & kw(blocker):
                    continue
                seen_block.add(key)
                c = self.block[block_class(attacker, blocker, kw)]
                c["n"] += 1
                c["blocked"] += (blocker_id, attacker.id) in pairs


def run(directories: list[Path]) -> Tally:
    resolver = KeywordResolver(SidecarCache({
        "cardsfolder": Path("output/cardsfolder"),
        "tokenscripts": Path("output/tokenscripts"),
        "variant-scripts": Path("output/effects/variant-scripts"),
    }))
    kw = resolver.keywords_of
    tally = Tally()
    for directory in directories:
        current, decisions, blocks = None, [], []
        combats: dict[int, tuple[set, set]] = {}
        for record in read_records(directory):
            if record.game_id != current:
                if current is not None:
                    tally.game(decisions, blocks, combats, kw)
                current, decisions, blocks, combats = record.game_id, [], [], {}
            if record.kind is RecordKind.COMBAT:
                attacked, pairs = combats.setdefault(record.state.global_.turn, (set(), set()))
                for e in record.state.entities:
                    if e.combat is None:
                        continue
                    if e.combat.attacking is not None:
                        attacked.add(e.id)
                    for target in e.combat.blocking or ():
                        pairs.add((e.id, target))
            elif record.kind is RecordKind.PLAYABILITY:
                if record.subkind is PlayabilitySubkind.ATTACKERS:
                    decisions.append(record)
                elif record.subkind is PlayabilitySubkind.BLOCKERS:
                    blocks.append(record)
        if current is not None:
            tally.game(decisions, blocks, combats, kw)
    return tally


def _table(title, rows, outcome) -> str:
    lines = [f"\n{title}", f"| class | n | {outcome} |", "|---|---:|---:|"]
    for name, c in sorted(rows.items(), key=lambda kv: -kv[1][outcome] / max(kv[1]["n"], 1)):
        lines.append(f"| {name} | {c['n']} | {c[outcome] / c['n']:.1%} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    tally = run(args.directories)
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
