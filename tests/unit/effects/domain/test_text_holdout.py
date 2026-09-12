"""Choosing the held-out ability texts (T156, FR-088/FR-088a).

The holdout is keyed on the ability text rather than the card, because the model
never sees a card's name and functional reprints compile to the same script. A
name-keyed holdout would send Searing Spear to the holdout and leave Lightning
Strike in training, and gate 1 would then drop those records for having text a
training card carries.

Common text cannot be held out at all: ``Flying`` is printed on thousands of
cards, and depleting every one of them would empty the training corpus. Hence
the carrier cap, which biases the holdout toward rare text — the population a
new mechanic belongs to.
"""

from __future__ import annotations

import os

from effects.domain.text_holdout import select_holdout


def test_an_eligible_text_is_held_out_when_its_hash_qualifies() -> None:
    """The hash decides membership, over texts few cards carry."""
    texts_by_card = {"Alpha": ["SP$ DealDamage | NumDmg$ 3"]}

    all_in = select_holdout(texts_by_card, permille=1000, max_carriers=8)
    none_in = select_holdout(texts_by_card, permille=0, max_carriers=8)

    assert all_in.texts == frozenset({"SP$ DealDamage | NumDmg$ 3"})
    assert all_in.cards == frozenset({"Alpha"})
    assert none_in.texts == frozenset()
    assert none_in.cards == frozenset()


def test_a_text_too_many_cards_carry_is_never_held_out() -> None:
    """Depleting every card that says `Flying` would empty the corpus."""
    common = "Flying"
    texts_by_card = {f"Bear {n}": [common] for n in range(9)}
    texts_by_card["Rare Thing"] = ["SP$ Bespoke | Weird$ True"]

    chosen = select_holdout(texts_by_card, permille=1000, max_carriers=8)

    assert common not in chosen.texts
    assert chosen.texts == frozenset({"SP$ Bespoke | Weird$ True"})
    assert chosen.cards == frozenset({"Rare Thing"})


def test_adding_unrelated_cards_does_not_move_an_existing_text() -> None:
    """A depleted corpus is composed once; a selection that drifts invalidates it."""
    kept = "SP$ DealDamage | NumDmg$ 3"
    before = select_holdout({"Alpha": [kept]}, permille=1000, max_carriers=8)

    after = select_holdout(
        {"Alpha": [kept], "Beta": ["SP$ Draw | NumCards$ 2"]},
        permille=1000, max_carriers=8,
    )

    assert (kept in before.texts) == (kept in after.texts)
    assert "Alpha" in after.cards


def test_the_hash_is_stable_across_processes() -> None:
    """`hash()` over `str` is salted per process; a drifting holdout is useless."""
    import subprocess
    import sys

    script = (
        "from effects.domain.text_holdout import text_is_held_out;"
        "print(text_is_held_out('SP$ DealDamage | NumDmg$ 3', permille=500))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }

    assert len(runs) == 1, f"verdict varied by hash seed: {runs}"
