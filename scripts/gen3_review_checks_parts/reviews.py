"""The review's own counts, printed beside each check for comparison.

Each value is the review's figure as the named report states it, on the pilot
``output/effects/gen3/scratch/2026-11-15-pilot`` (its windows differ where noted).
A name missing here prints ``n/a``.
"""

from __future__ import annotations

REVIEW_COUNTS = {
    # report-slice-19.md, Q1 answer: keys declared by more than one line of one face, in cards.
    "keys-declared-by-two-lines": "451 cards",
    # report-verify-01.md, claims table and claim 1: live Equip halves holding `attached`.
    "equip-halves-without-stat-change": "356",
    "equip-halves-with-stat-change": "16",
    # report-verify-02.md, claim 1 data table: random seat, live, stack-targeting scripts, in
    # cast pairs. Measured: the table covers the pilot, the three stall runs, blind-kill-check
    # and pilot-checks together (the pilot alone has 20 random-seat live pairs).
    "counter-halves-random-seat-countered[spell-target]": "24 (six runs)",
    "counter-halves-random-seat-not-countered[stack-card-target]": "37 (six runs)",
    # report-verify-01.md, claims table, claim 2: 23 of 23 fizzled halves with stack extras.
    "fizzled-halves-with-zero-targets": "23 of 23",
    # report-verify-03.md, claim 8b: forced halves by phase (3,980 in all), 888 stamped early.
    "forced-halves[main2]": "1236",
    "forced-halves[main1]": "1035",
    "forced-halves[upkeep]": "352",
    "forced-halves[end_of_turn]": "264",
    "forced-halves[combat_damage]": "228",
    "forced-halves[combat_declare_attackers]": "224",
    "forced-halves[draw]": "222",
    "forced-halves[combat_declare_blockers]": "214",
    "forced-halves[combat_end]": "205",
    "forced-halves": "3980",
    "forced-halves-stamped-before-turn-action": "888",
    # report-slice-08.md, B1: forced halves with a state-based leave, 0 of 11 hold `unattached`.
    "forced-deaths-without-unattached": "11 of 11",
    # report-verify-03.md, claim 5 counts: 50 of the 131 forks whose creature held an attachment.
    "protection-forks-attachment-leaves-unnamed": "50 of 131",
    # report-verify-02.md, claim 5: 159 of 159 and 10 of 10 trigger records.
    "trigger-records-without-subject[DamageDealtOnce]": "159 of 159",
    "trigger-records-without-subject[DamageAll]": "10 of 10",
    # report-slice-03.md, B2: 101 of 101 events.
    "phase-skipped-with-unknown-phase": "101 of 101",
    # report-slice-04.md, B2: 2 pilot games.
    "marked-games-without-player-lost": "2",
    # report-slice-04.md, B4: 70 rescue records, 62 hold something else.
    "rescue-records-holding-more-than-the-outcome": "62 of 70",
    # report-verify-05.md, claim 1: 516 lethal creatures on game-ending combats, 0 deaths;
    # 15 of 1,440 on the others.
    "lethal-creatures-without-death[game-ending-combat]": "516 of 516",
    "lethal-creatures-without-death[other-combat]": "15 of 1440",
    # report-slice-09.md, pilot of 523 games (forks / records).
    "directed-forks[118.12b]": "517 forks / 3528 records",
    "directed-forks[510.2]": "153 forks / 925 records",
    "directed-forks[704.5f]": "1 fork",
    "directed-forks[704.5b]": "1 fork",
    "directed-forks[702.19c]": "3 forks",
    "directed-forks[702.84a]": "2 forks",
    "directed-forks[614.17d]": "2 forks",
    "directed-forks[702.124j]": "1 fork",
    "directed-forks[702.127a]": "1 fork",
    "directed-forks[702.133a]": "1 fork",
    # report-verify-01.md, claim 13: library cards below the top with neither link.
    "library-cards-without-owns-and-controls": "1701",
    # report-verify-01.md, claim 3: 326 non-token copy appearances, every line granted.
    "non-token-copy-entities-with-granted-line": "326 of 326",
    # MERGED-FINDINGS.md, 9.8: null attribution on mana records, in the review's sample.
    "mana-record-events-without-attribution[tapped]": "1375",
    "mana-record-events-without-attribution[mana_produced]": "1377",
    # report-verify-02.md, scan 2 data: 31 convoke candidates (own turn before attacks with an
    # untapped creature: cost never reduced, 2 of 2; own later or opponent's turn with an
    # untapped creature: reduced in 11 of 16).
    "convoke-candidates": "31",
}


def review_for(name: str) -> str:
    return REVIEW_COUNTS.get(name, "n/a")
