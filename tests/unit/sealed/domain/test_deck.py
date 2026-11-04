"""Deck: which of its cards count as basic lands."""

from sealed.domain.deck import Deck


def test_nonbasic_cards_drops_every_basic_supertype_land():
    deck = Deck.of([
        "Grizzly Bears", "Forest", "Wastes", "Snow-Covered Island",
        "Snow-Covered Wastes", "Grizzly Bears", "Adarkar Wastes",
    ])

    assert deck.nonbasic_cards() == ["Grizzly Bears", "Grizzly Bears", "Adarkar Wastes"]
