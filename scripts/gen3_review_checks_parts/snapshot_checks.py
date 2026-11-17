"""Checks over the board snapshot: library cards and copies."""

from __future__ import annotations

from .model import TallyCheck, entities_of, state_of


def _linked(record: dict, link_type: str) -> set[str]:
    links = state_of(record).get("links") or []
    return {x["to"] for x in links if x.get("type") == link_type and isinstance(x.get("to"), str)}


class LibraryCards(TallyCheck):
    """Library cards below the top, which only an event names, with no `owns` or `controls`."""

    NAMES = ("library-cards-without-owns-and-controls",)

    def feed(self, record: dict) -> None:
        library = [e for e in entities_of(record) if e.get("role") == "library_card"]
        if not library:
            return
        owned, controlled = _linked(record, "owns"), _linked(record, "controls")
        for entity in library:
            if entity.get("order_rank") != 0:
                bare = entity["id"] not in owned and entity["id"] not in controlled
                self.add("library-cards-without-owns-and-controls", bare)


class CopyLines(TallyCheck):
    """Non-token copy entities, flagged when any of their lines is granted."""

    NAMES = ("non-token-copy-entities-with-granted-line",)

    def feed(self, record: dict) -> None:
        for entity in entities_of(record):
            if entity.get("copy_source") and entity.get("token_script_id") is None:
                granted = any(line.get("granted") for line in entity.get("lines") or ())
                self.add("non-token-copy-entities-with-granted-line", granted)
