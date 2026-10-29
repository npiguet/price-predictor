"""The ward canary's checked-in functional twins.

Ward is recent but its effect has been printed in longhand for decades, so ward's
``e`` sitting nearer each longhand twin than the median bare keyword (FR-112)
says the encoder learned what ward *does*. Every entry is converted prose the
corpus actually prints; one that fails to resolve is named in the report rather
than dropped.
"""

from __future__ import annotations

from effects.domain.line_query import LineQuery

#: Ward at the parameterization the twins spell out.
WARD = LineQuery("ward {2}")

#: Longhand lines with ward's effect and none of its vocabulary. Each pins its
#: card, since the canary compares against one vector per twin.
WARD_TWINS: tuple[LineQuery, ...] = (
    # Ward {2} itself, spelled out on the card it protects.
    LineQuery(
        "whenever CARDNAME becomes the target of a spell or ability an opponent "
        "controls, counter that spell or ability unless its controller pays "
        "{2}.",
        card="frost titan",
    ),
    # The same clause, granted to a creature type rather than printed.
    LineQuery(
        "whenever a sliver creature you control becomes the target of a spell "
        "or ability an opponent controls, counter that spell or ability unless "
        "its controller pays {2}.",
        card="diffusion sliver",
    ),
    # The same clause at a lower tax, protecting the player too.
    LineQuery(
        "whenever you or a permanent you control becomes the target of a spell "
        "or ability an opponent controls, counter that spell or ability unless "
        "its controller pays {1}.",
        card="unsettled mariner",
    ),
    # The tax collected up front instead of on the trigger.
    LineQuery(
        "spells your opponents cast that target CARDNAME cost {2} more to cast.",
        card="boreal elemental",
    ),
)

#: Bare single-keyword lines the median is taken over, deliberately unrelated
#: to ward mechanically.
BARE_KEYWORDS: tuple[LineQuery, ...] = tuple(
    LineQuery(keyword) for keyword in (
        "flying", "vigilance", "trample", "haste", "deathtouch", "lifelink",
        "first strike", "double strike", "reach", "menace", "defender",
        "hexproof", "indestructible", "flash", "shroud", "intimidate",
        "wither", "infect", "skulk", "fear",
    )
)
