"""The grammar of a sidecar line's ``script_text``: segments and parameters.

A segment is ``Key$ value`` parameters separated by ``|``, split as Forge's
``FileSection.parseToMap`` splits a source script; a printed keyword is its
bare display name.
"""

from __future__ import annotations

import re

#: Separator between the segments of a chained script text (FR-002).
SEG = "[SEG]"

#: The label that opens a chained segment, in its FR-002a form.
SEGMENT_OPENER = re.compile(r"^\s*(SV\d+):\s*")

#: The parameters whose value names a chain segment (FR-002a).
CHAIN_KEYS = frozenset({"Execute", "SubAbility", "RepeatSubAbility", "ReplaceWith"})

#: Every parameter whose value the converter renames to ``SV#`` labels, beyond
#: the reading ``CHAIN_KEYS`` has: a source script and its sidecar differ in
#: these values. Mirrors ``TraitScript.java`` (``SINGLE_LINK_KEYS``, the
#: ``AbilityFactory.additionalAbilityKeys`` it adds, a chooser's ``Choices$``,
#: ``ResultSubAbilities$``) and ``GrantedTraits.java`` (``GRANTS``) in
#: forge-connector. ``Choices$`` is a label list only on a chooser API, and
#: ``ResultSubAbilities$`` holds ``range:label`` items.
RELABELLED_KEYS = CHAIN_KEYS | frozenset({
    # AbilityFactory.additionalAbilityKeys (Execute and RepeatSubAbility are above)
    "WinSubAbility", "OtherwiseSubAbility", "BidSubAbility", "ChooseNumberSubAbility",
    "Lowest", "Highest", "NotLowest", "GuessCorrect", "GuessWrong", "MatchedAbility",
    "UnmatchedAbility", "HeadsSubAbility", "TailsSubAbility", "LoseSubAbility",
    "TrueSubAbility", "FalseSubAbility", "ChosenPile", "UnchosenPile", "FallbackAbility",
    "ChooseSubAbility", "CantChooseSubAbility", "RegenerationAbility", "ReturnAbility",
    "GiftAbility", "VoteSubAbility", "VoteTiedAbility",
    # GrantedTraits.GRANTS
    "AddTrigger", "AddStaticAbility", "AddReplacementEffect", "AddAbility", "Abilities",
    "Triggers", "Replacements", "staticAbilities", "StaticAbilities", "ReplacementEffects",
    "AddTriggers", "AddAbilities", "GainTextAbilities", "AddStaticAbilities", "Trigger",
    "TriggersWhenSpent", "PreventionSubAbility", "StaticEffect", "ExtraTurnDelayedTrigger",
    "ExtraTurnDelayedTriggerExecute", "ExtraPhaseDelayedTrigger",
    "ExtraPhaseDelayedTriggerExcute",
})

#: The chooser APIs on which ``Choices$`` lists labels (``TraitScript.CHOOSER_APIS``).
CHOOSER_APIS = frozenset(
    {"Charm", "GenericChoice", "AssignGroup", "VillainousChoice", "Vote"})

#: The keys whose value names the segment's API type.
_API_KEYS: tuple[str, ...] = ("SP", "AB", "DB")


def segments(text: str) -> list[str]:
    """A chained text's segments, each stripped and without its ``SVn:`` opener."""
    return [SEGMENT_OPENER.sub("", part).strip() for part in text.split(SEG)]


def root(text: str) -> str:
    """The text before the first :data:`SEG`, as written: the trait's own line."""
    return text.split(SEG, 1)[0]


def params(segment: str) -> dict[str, str]:
    """``key -> raw value`` of one segment's ``Key$ value`` parameters.

    Split at the first ``$`` only, so ``Count$Valid Creature`` stays whole.
    The first occurrence of a key wins.
    """
    found: dict[str, str] = {}
    for part in segment.split("|"):
        key, separator, value = part.partition("$")
        key = key.strip()
        if separator and key and key not in found:
            found[key] = value.strip()
    return found


def root_param(text: str | None, name: str) -> str | None:
    """The value of ``name$`` on the root segment; None when absent or empty."""
    if not text:
        return None
    return params(root(text)).get(name) or None


def api_of(found: dict[str, str]) -> str | None:
    """The segment's effect API, whichever of ``SP$``/``AB$``/``DB$`` names
    it; None where none does or it is empty."""
    for key in _API_KEYS:
        if found.get(key):
            return found[key]
    return None


def is_bare_keyword(text: str | None) -> bool:
    """Whether ``text`` is a printed keyword's display name (White Knight's
    ``First Strike``) rather than a script."""
    return bool(text) and "$" not in text and "|" not in text
