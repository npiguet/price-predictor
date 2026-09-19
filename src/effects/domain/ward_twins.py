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

Every entry is a line the converted corpus actually prints, named by its
converted prose (see :mod:`effects.domain.line_query`) and resolved to whatever
surface the checkpoint encodes. A twin written from memory rather than copied
from the corpus resolves to nothing, and a canary comparing ward against fewer
twins than it lists is a weaker canary than it claims to be — so an entry that
fails to resolve is named in the report rather than dropped.
"""

from __future__ import annotations

from effects.domain.line_query import LineQuery

#: Ward at the parameterization the twins spell out. The converter drops the
#: reminder text, so the line is the bare keyword.
WARD = LineQuery("ward {2}")

#: Longhand lines with ward's effect and none of its vocabulary: a spell or
#: ability an opponent aims at the protected permanent is countered unless its
#: controller pays, or costs more to begin with. Each pins its card, since the
#: canary compares against one vector per twin.
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

#: Bare single-keyword lines the median is taken over. Deliberately mechanically
#: unrelated to ward: the comparison is meant to ask "is ward nearer its
#: meaning than it is to keyword-shaped text in general".
BARE_KEYWORDS: tuple[LineQuery, ...] = tuple(
    LineQuery(keyword) for keyword in (
        "flying", "vigilance", "trample", "haste", "deathtouch", "lifelink",
        "first strike", "double strike", "reach", "menace", "defender",
        "hexproof", "indestructible", "flash", "shroud", "intimidate",
        "wither", "infect", "skulk", "fear",
    )
)
