"""Gen-2 figures that exist only in the records' text, transcribed for ``build_data.py``.

Each block names where its figures were first measured. They were measured once, by a
script run whose output was not kept beside the run, or before a result file existed,
and cannot be regenerated from ``models/effects/runs/`` alone.
"""

# Gate 3's collapse canaries and the participation ratio for caches with no gen-2
# evaluation log: gen-1's shipped checkpoint and the stage-0 noise pilots (3 epochs
# each, on gen-1's corpus). Source: the gen-2 record's stage-0 outcome.
GATE3 = [
    ["gen-1", 0.152, 0.400, 3.93],
    ["noise pilot 0.05", 0.475, 0.285, 6.57],
    ["noise pilot 0.1", 0.482, 0.257, 7.07],
    ["noise pilot 0.2", 0.333, 0.315, 5.72],
]

# Linear embedding probes on gen-1's cache, all texts. Source: the gen-2 record's arm
# e64 outcome, from gen-1's embedding-probe report of 2026-10-08.
EMBEDDING_GEN1 = [
    ["gen-1", "API type", "accuracy", 0.641, None],
    ["gen-1", "line kind", "accuracy", 0.677, None],
    ["gen-1", "cost_mana", "R²", 0.057, None],
    ["gen-1", "amt_damage", "R²", 0.343, None],
    ["gen-1", "amt_counters", "R²", 0.175, None],
    ["gen-1", "amt_life", "R²", 0.086, None],
    ["gen-1", "amt_cards", "R²", 0.190, None],
    ["gen-1", "amt_power", "R²", 0.321, None],
]

# The lethal-boundary metric on 40 qualifying training records, for the arms' own
# checkpoints, measured before the continuation runs kept a boundary file per run.
BOUNDARY = [
    ["e64-lethality", 10, 0.224, 0.81, 0.71, 0.57, 0.24],
    ["e64-lethality", 17, 0.169, 0.82, 0.69, 0.47, 0.35],
    ["e64-lethality", 24, 0.187, 0.63, 0.43, 0.24, 0.39],
    ["e64-lethality", 28, 0.157, 0.78, 0.60, 0.36, 0.42],
    ["e128-lethality", 12, 0.206, 0.89, 0.78, 0.62, 0.26],
    ["e128-lethality", 17, 0.160, 0.89, 0.75, 0.54, 0.34],
    ["e128-lethality", 20, 0.173, 0.93, 0.84, 0.64, 0.29],
    ["e128-lethality", 22, 0.173, 0.71, 0.51, 0.30, 0.41],
]

# The damage and kind of each of the 20 toughness-sweep records, in scorecard order
# (scripts' sweep_diag over the frozen probe set ab6452679131).
TOUGHNESS_SWEEP_RECORDS = [
    [i, d, k] for i, (d, k) in enumerate(zip(
        [3, 1, 1, 1, 1, 3, 5, 2, 3, 1, 5, 1, 1, 3, 3, 1, 2, 2, 3, 3],
        ["spell", "spell", "triggered", "activated", "triggered", "activated", "triggered",
         "triggered", "spell", "activated", "spell", "activated", "activated", "spell",
         "triggered", "spell", "triggered", "spell", "spell", "spell"]))
]

TABLES = {
    "noise_pilot.csv": (
        ["measure", "ratio 0.05", "ratio 0.1", "ratio 0.2"],
        [["card-disjoint loss, epoch 3", 4.566, 4.509, 4.459],
         ["game-disjoint loss, epoch 3", 4.479, 4.510, 4.511],
         ["gate F1, epoch 3", 0.774, 0.775, 0.769],
         ["zone accuracy, epoch 3", 0.938, 0.947, 0.949],
         ["value-head training loss, epoch 3", 1.71, 1.82, 2.06]]),
    "pilot_collections.csv": (
        ["measure", "first pilot", "second pilot"],
        [["random seat: share of matches", 0.125, 0.25],
         ["random seat: share of its decisions taken at random", 0.25, 0.5],
         ["games", 833, 2698],
         ["games with a random seat", 80, 573],
         ["records", 321792, 812800],
         ["real attack declarations", 10950, None],
         ["real block declarations", 15296, None],
         ["what-if block queries collected", 70010, 0],
         ["real AI blocks against the random seat's attacks", 361, 3491],
         ["of those, against a random attack (expected)", 90, 1750],
         ["real resolutions that dealt damage", 1249, None],
         ["of those, the damaged creature dies in the record", 684, None]]),
    "first_strike_repair.csv": (
        ["corpus part", "records", "repaired", "dropped"],
        [["training", 892205, 20864, 2432],
         ["card-disjoint stratum", 3626459, 2490, 141],
         ["game-disjoint stratum", 1163084, 1855, 216],
         ["validation samples", 4096, 42, 0]]),
    # scratch scan of corpus-gen2/validation/game-disjoint, real (what_if false) records, 2026-10-21
    "legality_planned_blocks.csv": (
        ["game-disjoint stratum, real decisions", "count"],
        [["blockers records", 80306],
         ["  whose snapshot shows blocks the AI had already planned", 47053],
         ["forbidden blockers listed", 147584],
         ["  naming no restricting static", 98946],
         ["  naming a creature the snapshot shows blocking", 83141],
         ["  naming a tapped creature", 10952],
         ["attackers anchoring a record, per game and turn", 35093],
         ["  anchoring more than one real record", 18793],
         ["attackers records", 51233],
         ["  taken after attackers were declared", 0]]),
    # scratch scan of the 316 full-strength raw shards, observed (non-fork) records, 2026-10-21
    "hook_defects_in_corpus.csv": (
        ["full-strength raw shards", "count"],
        [["regular-step damage sources that also dealt first-strike damage", 1188],
         ["  with no row in the record's assignment_choices", 1167],
         ["regular-step damage sources with double strike in the snapshot", 1552],
         ["  with no row in the record's assignment_choices", 1181],
         ["observed combat records", 192895],
         ["  holding an event attributed to a resolving ability", 2601],
         ["card_drawn events in observed combat records", 2718],
         ["token_created events in observed combat records", 1904],
         ["card_discarded events in observed combat records", 1029],
         ["card_milled events in observed combat records", 1003],
         ["monarch_changed events in observed combat records", 562],
         ["TapsForMana trigger records", 4911],
         ["  fired", 2700],
         ["  repeating another's game, turn, step, land and verdict", 996],
         ["damage_prevented events", 4030],
         ["  of amount 0", 262]]),
    "lethal_hits_in_corpus.csv": (
        ["damage, 40 training shards", "lethal", "not lethal"],
        [["a spell or ability's damage to a targeted creature", 170, 36],
         ["its damage to a creature it did not target", 567, 254],
         ["-X/-X on a targeted creature", 14, 21],
         ["combat damage to a creature", 3252, 1314]]),
    "gen1_vs_e64_field_terms.csv": (
        ["run", "epoch 9", "epoch 12", "epoch 15", "epoch 18", "epoch 25"],
        [["gen-1, field terms (its whole loss)", 3.72, 3.49, 3.35, 3.29, 3.13],
         ["e64 (earlier inputs), field terms", 3.73, 3.59, 3.46, 3.48, None],
         ["e64, field terms", 3.89, 3.72, 3.47, 3.53, 3.27]]),
    # The rarity table's texts grouped into rule families: by Forge API type, trigger mode
    # or keyword. Source: the gen-2 record's rarity-weight section, on gen-1's corpus.
    "rule_families.csv": (
        ["rule family", "distinct texts", "games"],
        [["ChangesZone triggers", 5393, 312514],
         ["Pump", 1989, 73949],
         ["DealDamage", 1224, 66343],
         ["Flying", 608, 20153],
         ["Suspend", 52, 3095],
         ["Ninjutsu", 22, 738],
         ["Cascade", 11, 1052],
         ["Dredge", 7, 602],
         ["Cypher", 0, 0]]),
    "toughness_sweep_records.csv": (
        ["record", "damage", "kind"], TOUGHNESS_SWEEP_RECORDS),
    # The toughness sweep's five activated damage records, unedited toughness: the mean
    # P(affected) at the targeted creature after one edit to the record or its text.
    # Scratch scripts ping_counterfactual.py and ping_variants.py, 2026-10-21.
    "ping_gate_edits.csv": (
        ["edit", "e64-lethality", "B64 last", "e256-lethality", "B256 last"],
        [["as collected", 0.42, 0.56, 0.61, 0.96],
         ["moved to the first main phase", 0.43, 0.59, 0.63, 0.95],
         ["source untapped", 0.51, 0.63, 0.58, 0.96],
         ["read as a spell: AB$ to SP$, cost dropped", 0.60, 0.91, 0.46, 0.98],
         ["'any target' read as 'target creature'", 0.97, 0.93, 0.92, 0.98],
         ["a healer's 'prevent the next 1 damage' text in its place", 0.00, 0.00, 0.02, 0.01]]),
    # Training resolutions whose acting line's first segment is SP$/AB$ DealDamage, in the
    # first 150 of the 444 curated training shards (repeats included), by kind and
    # ValidTgts$. Scratch script ping_targets.py, 2026-10-21.
    "damage_targets_in_training.csv": (
        ["acting line", "resolutions", "targets that are players", "targets that are creatures",
         "creature targets", "creature targets marked affected"],
        [["spell, any target", 182, 0.21, 0.79, 115, 1.00],
         ["spell, target creature", 178, 0.01, 0.99, 168, 0.98],
         ["activated, any target, tap in its cost", 24, 0.35, 0.65, 15, 0.87],
         ["activated, any target, other cost", 16, 0.43, 0.50, 7, 0.86],
         ["activated, target creature, tap in its cost", 10, 0.00, 1.00, 10, 0.80],
         ["activated, target creature, other cost", 13, 0.00, 1.00, 13, 0.92]]),
    # P(affected) at the hit creature on the boundary chart's 40 training records, rows
    # below an even chance per checkpoint. Scratch script ping_gate.py, 2026-10-21.
    "boundary_records_gate.csv": (
        ["acting line", "rows", "e64-lethality", "e256-lethality", "B64 last", "B128 last",
         "B256 last"],
        [["spell", 23, 8, 4, 0, 1, 0],
         ["activated", 8, 1, 1, 0, 1, 0],
         ["mode of a modal spell", 4, 0, 0, 0, 0, 0],
         ["triggered", 5, 5, 5, 4, 1, 3],
         ["all rows", 40, 14, 10, 4, 3, 3]]),
}

# The damage grids of the arms probed before the grid files were kept: predicted death
# with damage 1 to 8 down the rows and the target's toughness 1 to 8 across.
DAMAGE_GRIDS = {
    ('e128-lethality', 'probe set'): [
        [0.75, 0.43, 0.14, 0.06, 0.03, 0.01, 0.01, 0.0],
        [0.77, 0.57, 0.26, 0.11, 0.05, 0.02, 0.01, 0.0],
        [0.78, 0.66, 0.36, 0.15, 0.05, 0.02, 0.01, 0.0],
        [0.79, 0.74, 0.52, 0.29, 0.13, 0.05, 0.02, 0.01],
        [0.81, 0.78, 0.68, 0.44, 0.24, 0.11, 0.05, 0.02],
        [0.84, 0.83, 0.79, 0.63, 0.43, 0.28, 0.15, 0.07],
        [0.84, 0.84, 0.85, 0.8, 0.64, 0.51, 0.37, 0.22],
        [0.86, 0.86, 0.87, 0.87, 0.82, 0.72, 0.59, 0.43],
    ],
    ('e64-lethality', 'probe set'): [
        [0.77, 0.57, 0.21, 0.09, 0.03, 0.01, 0.0, 0.0],
        [0.76, 0.7, 0.48, 0.25, 0.1, 0.03, 0.01, 0.01],
        [0.74, 0.72, 0.61, 0.38, 0.13, 0.04, 0.02, 0.01],
        [0.77, 0.76, 0.71, 0.51, 0.2, 0.06, 0.02, 0.01],
        [0.79, 0.8, 0.79, 0.72, 0.45, 0.19, 0.07, 0.03],
        [0.79, 0.8, 0.81, 0.8, 0.66, 0.4, 0.2, 0.08],
        [0.76, 0.78, 0.8, 0.82, 0.8, 0.67, 0.44, 0.24],
        [0.78, 0.8, 0.82, 0.84, 0.86, 0.82, 0.69, 0.49],
    ],
    ('e128', 'probe set'): [
        [0.81, 0.69, 0.46, 0.25, 0.15, 0.1, 0.07, 0.05],
        [0.82, 0.76, 0.6, 0.39, 0.24, 0.16, 0.11, 0.08],
        [0.83, 0.79, 0.69, 0.5, 0.34, 0.24, 0.17, 0.12],
        [0.83, 0.8, 0.73, 0.6, 0.45, 0.34, 0.25, 0.18],
        [0.82, 0.8, 0.75, 0.65, 0.52, 0.41, 0.31, 0.24],
        [0.83, 0.81, 0.78, 0.7, 0.59, 0.5, 0.4, 0.32],
        [0.83, 0.81, 0.78, 0.72, 0.63, 0.54, 0.45, 0.38],
        [0.84, 0.82, 0.79, 0.73, 0.65, 0.57, 0.48, 0.4],
    ],
    ('e64', 'probe set'): [
        [0.61, 0.6, 0.55, 0.45, 0.31, 0.21, 0.16, 0.09],
        [0.63, 0.63, 0.6, 0.53, 0.39, 0.28, 0.22, 0.14],
        [0.64, 0.64, 0.62, 0.57, 0.45, 0.33, 0.27, 0.18],
        [0.67, 0.67, 0.66, 0.63, 0.52, 0.4, 0.34, 0.24],
        [0.68, 0.69, 0.68, 0.65, 0.57, 0.45, 0.39, 0.29],
        [0.69, 0.7, 0.7, 0.69, 0.65, 0.57, 0.51, 0.42],
        [0.59, 0.59, 0.59, 0.58, 0.55, 0.48, 0.44, 0.36],
        [0.69, 0.69, 0.69, 0.68, 0.65, 0.58, 0.52, 0.43],
    ],
    ('e64 (earlier inputs)', 'probe set'): [
        [0.69, 0.64, 0.56, 0.45, 0.31, 0.19, 0.12, 0.07],
        [0.68, 0.65, 0.59, 0.5, 0.38, 0.26, 0.17, 0.11],
        [0.66, 0.64, 0.59, 0.52, 0.42, 0.32, 0.22, 0.15],
        [0.7, 0.68, 0.64, 0.58, 0.49, 0.38, 0.28, 0.2],
        [0.68, 0.66, 0.63, 0.57, 0.49, 0.4, 0.31, 0.22],
        [0.68, 0.66, 0.63, 0.59, 0.52, 0.44, 0.35, 0.26],
        [0.64, 0.62, 0.59, 0.54, 0.46, 0.37, 0.28, 0.21],
        [0.65, 0.63, 0.6, 0.55, 0.48, 0.39, 0.3, 0.22],
    ],
    ('e128-lethality', 'training records'): [
        [0.58, 0.43, 0.27, 0.14, 0.06, 0.03, 0.01, 0.01],
        [0.61, 0.51, 0.32, 0.18, 0.08, 0.04, 0.02, 0.01],
        [0.64, 0.58, 0.39, 0.2, 0.09, 0.04, 0.02, 0.01],
        [0.66, 0.63, 0.51, 0.29, 0.12, 0.05, 0.02, 0.01],
        [0.66, 0.65, 0.61, 0.46, 0.22, 0.09, 0.04, 0.02],
        [0.66, 0.67, 0.66, 0.64, 0.46, 0.25, 0.11, 0.05],
        [0.69, 0.71, 0.73, 0.76, 0.74, 0.61, 0.4, 0.21],
        [0.68, 0.7, 0.72, 0.75, 0.74, 0.65, 0.47, 0.29],
    ],
}
