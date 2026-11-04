"""The keywords that keep a flag (M5, FR-044a): generated, do not edit.

Written by ``python scripts/scan_keyword_flags.py`` from Forge's card scripts and
engine code. A keyword keeps its flag where a card-script selector or an engine check
names it; protection and split second keep theirs whatever the scan finds. Every
other keyword is read through its line alone. Each name's comment gives the
card-script selectors and engine checks the scan found naming it.
"""

KEPT_KEYWORD_FLAGS: tuple[str, ...] = (
    "affinity",  # scripts 0, engine 2
    "ascend",  # scripts 0, engine 1
    "assist",  # scripts 0, engine 1
    "awaken",  # scripts 1, engine 0
    "bands_with_other",  # scripts 0, engine 1
    "blitz",  # scripts 1, engine 0
    "cascade",  # scripts 2, engine 0
    "companion",  # scripts 0, engine 2
    "convoke",  # scripts 3, engine 1
    "cumulative_upkeep",  # scripts 1, engine 0
    "cycling",  # scripts 10, engine 0
    "deathtouch",  # scripts 18, engine 3
    "decayed",  # scripts 2, engine 0
    "defender",  # scripts 40, engine 0
    "delve",  # scripts 0, engine 1
    "devoid",  # scripts 1, engine 0
    "disguise",  # scripts 2, engine 0
    "disturb",  # scripts 5, engine 1
    "doctor's_companion",  # scripts 2, engine 0
    "double_agenda",  # scripts 0, engine 2
    "double_strike",  # scripts 9, engine 1
    "embalm",  # scripts 1, engine 0
    "emerge",  # scripts 1, engine 0
    "enchant",  # scripts 2, engine 6
    "epic",  # scripts 0, engine 2
    "escape",  # scripts 0, engine 1
    "eternalize",  # scripts 1, engine 0
    "fading",  # scripts 1, engine 0
    "first_strike",  # scripts 15, engine 1
    "flanking",  # scripts 3, engine 1
    "flash",  # scripts 14, engine 3
    "flashback",  # scripts 9, engine 1
    "flying",  # scripts 474, engine 1
    "foretell",  # scripts 5, engine 1
    "freerunning",  # scripts 2, engine 0
    "gift",  # scripts 0, engine 1
    "haste",  # scripts 20, engine 1
    "hexproof",  # scripts 4, engine 1
    "hidden_agenda",  # scripts 0, engine 1
    "horsemanship",  # scripts 6, engine 1
    "improvise",  # scripts 0, engine 1
    "indestructible",  # scripts 12, engine 2
    "infect",  # scripts 5, engine 2
    "jump-start",  # scripts 0, engine 1
    "kicker",  # scripts 5, engine 0
    "landwalk",  # scripts 4, engine 2
    "level_up",  # scripts 1, engine 0
    "lifelink",  # scripts 8, engine 1
    "madness",  # scripts 1, engine 0
    "megamorph",  # scripts 6, engine 0
    "menace",  # scripts 12, engine 1
    "modular",  # scripts 1, engine 1
    "morph",  # scripts 6, engine 0
    "multikicker",  # scripts 5, engine 0
    "mutate",  # scripts 4, engine 0
    "offering",  # scripts 0, engine 1
    "offspring",  # scripts 0, engine 1
    "partner",  # scripts 1, engine 0
    "persist",  # scripts 1, engine 0
    "phasing",  # scripts 2, engine 2
    "protection",  # scripts 1, engine 2
    "rampage",  # scripts 1, engine 0
    "reach",  # scripts 14, engine 2
    "read_ahead",  # scripts 0, engine 2
    "reconfigure",  # scripts 0, engine 2
    "retrace",  # scripts 0, engine 1
    "shadow",  # scripts 12, engine 5
    "skulk",  # scripts 1, engine 0
    "soulbond",  # scripts 1, engine 0
    "space_sculptor",  # scripts 0, engine 1
    "splice",  # scripts 0, engine 3
    "split_second",  # scripts 0, engine 1
    "spree",  # scripts 0, engine 1
    "start_your_engines",  # scripts 1, engine 1
    "starting_intensity",  # scripts 0, engine 2
    "sunburst",  # scripts 0, engine 1
    "suspend",  # scripts 19, engine 1
    "tiered",  # scripts 0, engine 1
    "toxic",  # scripts 10, engine 1
    "trample",  # scripts 16, engine 3
    "typecycling",  # scripts 10, engine 1
    "unearth",  # scripts 2, engine 0
    "vigilance",  # scripts 12, engine 0
    "warp",  # scripts 1, engine 0
    "wither",  # scripts 0, engine 1
)
