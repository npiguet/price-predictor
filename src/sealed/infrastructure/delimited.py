"""The semicolon-record / pipe-list grammar shared by the sealed text files.

Every sealed data file (``pools.txt``, ``generated-decks.txt``,
``match-outcomes.txt``, ``cards-played.txt``) encodes one record per line as
``;``-separated fields, several of which are ``|``-separated card lists. These
two operations were re-implemented in each reader and had begun to drift (one
reader dropped the empty-field guard, yielding ``[""]`` instead of ``[]``).
They live here so all readers share one definition.
"""

from __future__ import annotations


def parse_pipe_list(field: str) -> list[str]:
    """Split a ``|``-separated field into its items.

    An empty field is an empty list (not ``[""]``) — the invariant the card-list
    columns rely on.
    """
    return field.split("|") if field else []


def split_record(line: str, expected_fields: int) -> list[str]:
    """Split a ``;``-separated record into exactly ``expected_fields`` fields.

    Raises ``ValueError`` naming the expected and actual field counts (and the
    offending line) when the count is wrong — the malformed-line guard every
    fixed-width reader needs.
    """
    fields = line.split(";")
    if len(fields) != expected_fields:
        raise ValueError(
            f"Expected {expected_fields} semicolon-delimited fields, got "
            f"{len(fields)}: {line!r}"
        )
    return fields
