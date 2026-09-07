"""The ward canary's checked-in functional twins.

Ward is the cleanest test of whether the encoder read a keyword or memorized it.
The keyword is recent, but its effect is old: cards have printed the same
"whenever this becomes the target of an opponent's spell, counter it unless they
pay" clause in longhand for decades. If ward's ``e`` sits nearer those longhand
texts than it does to other bare keywords, the encoder learned what ward *does*
rather than that ward is a word.

The check passes when ward's ``e`` is closer in cosine distance to **each** twin
below than to the median of its distances to all bare single-keyword vectors
(FR-112). Comparing against a median rather than a fixed threshold keeps the
criterion meaningful whatever the embedding's overall scale turns out to be.

The twins are the converted-text form, because that is the surface the encoder
reads through stage three.
"""

from __future__ import annotations

#: Ward's own converted text, at the parameterization the twins spell out.
WARD_TEXT = (
    "ward {2} (whenever this permanent becomes the target of a spell or "
    "ability an opponent controls, counter it unless that player pays {2}.)"
)

#: Longhand texts with ward's effect and none of its vocabulary. Each is the
#: shape a real card printed before the keyword existed.
WARD_TWINS: tuple[str, ...] = (
    (
        "whenever this creature becomes the target of a spell or ability an "
        "opponent controls, counter that spell or ability unless its "
        "controller pays {2}."
    ),
    (
        "whenever this permanent becomes the target of a spell an opponent "
        "controls, counter that spell unless that player pays {2}."
    ),
    (
        "whenever this creature becomes the target of a spell or ability an "
        "opponent controls, that player sacrifices a permanent unless they "
        "pay {2}."
    ),
    (
        "spells and abilities your opponents control that target this "
        "creature cost {2} more to cast or activate."
    ),
)

#: Bare single-keyword lines the median is taken over. Deliberately mechanically
#: unrelated to ward: the comparison is meant to ask "is ward nearer its
#: meaning than it is to keyword-shaped text in general".
BARE_KEYWORDS: tuple[str, ...] = (
    "flying", "vigilance", "trample", "haste", "deathtouch", "lifelink",
    "first strike", "double strike", "reach", "menace", "defender",
    "hexproof", "indestructible", "flash", "shroud", "intimidate",
    "wither", "infect", "skulk", "fear",
)
