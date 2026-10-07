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
from pathlib import Path

import pytest

from effects.domain.text_holdout import (
    holdout_key,
    masked_template,
    normalize_script_text,
    select_holdout,
)


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


# ── the masked template (FR-039, FR-040; T039) ──────────────────────────

_FIXTURE_SIDECARS = Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-sidecars"


@pytest.mark.parametrize(("left", "right"), [
    ("SP$ DealDamage | NumDmg$ 2 | ValidTgts$ Any",
     "SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any"),
    ("DB$ Draw | ConditionCompare$ GE3", "DB$ Draw | ConditionCompare$ GE4"),
    ("DB$ Token | TokenScript$ w_1_1_soldier", "DB$ Token | TokenScript$ w_2_2_soldier"),
    ("Mode$ Phase | Phase$ Main1", "Mode$ Phase | Phase$ Main2"),
    ("AB$ Pump | Cost$ 2 G | SpellDescription$ CARDNAME gets +2/+2.",
     "AB$ Pump | Cost$ 3 G | SpellDescription$ Grizzly Bears gets +3/+3."),
])
def test_texts_differing_only_in_digits_or_descriptions_share_a_template(left, right):
    """Spec Story 2 scenario 1: number-only variants are one unit."""
    assert masked_template(left) == masked_template(right)


@pytest.mark.parametrize(("left", "right"), [
    # A reference naming another segment is another mechanism.
    ("Execute$ SV1 | Mode$ ChangesZone [SEG] SV1: DB$ Draw [SEG] SV2: DB$ Pump",
     "Execute$ SV2 | Mode$ ChangesZone [SEG] SV1: DB$ Draw [SEG] SV2: DB$ Pump"),
    ("DB$ PutCounter | CounterType$ P1P1", "DB$ PutCounter | CounterType$ M1M1"),
    ("AB$ Pump | Cost$ 2 G", "AB$ Pump | Cost$ 2 R"),
    ("CharmNum$ 1 | Choices$ SV1,SV2", "CharmNum$ 1 | Choices$ SV1,SV3"),
])
def test_labels_letters_and_colours_keep_templates_apart(left, right):
    assert masked_template(left) != masked_template(right)


def test_chain_labels_keep_their_digits_and_everything_else_is_masked():
    text = (
        "Execute$ SV1 | Mode$ ChangesZone | TriggerDescription$ When CARDNAME "
        "dies, draw 2. [SEG] SV1: DB$ Draw | NumCards$ 2 | SubAbility$ SV2 "
        "[SEG] SV2: DB$ PutCounter | CounterType$ P1P1 | CounterNum$ 3"
    )
    assert masked_template(text) == (
        "Execute$ SV1 | Mode$ ChangesZone | TriggerDescription$ # [SEG] SV1: "
        "DB$ Draw | NumCards$ # | SubAbility$ SV2 [SEG] SV2: DB$ PutCounter | "
        "CounterType$ P#P# | CounterNum$ #"
    )


def test_a_keyword_line_is_masked_like_any_other():
    assert masked_template("Ward:2") == masked_template("Ward:4") == "Ward:#"


def test_whitespace_differences_are_one_template():
    assert masked_template("SP$  Draw |NumCards$ 1") == masked_template("SP$ Draw | NumCards$ 2")


def _templates_select_together(texts_by_card, **flags):
    chosen = select_holdout(texts_by_card, unit="template", **flags)
    by_template: dict[str, set[bool]] = {}
    for texts in texts_by_card.values():
        for text in texts:
            by_template.setdefault(masked_template(text), set()).add(
                normalize_script_text(text) in chosen.texts
            )
    return by_template


def test_texts_of_one_template_fall_on_the_same_side():
    """Spec Story 2 scenario 1 and SC-003, over a hundred synthetic printings."""
    texts_by_card = {
        f"card {i}": [f"SP$ DealDamage | NumDmg$ {i % 7} | ValidTgts$ Kind{i % 13}"]
        for i in range(100)
    }
    for permille in (100, 500, 900):
        sides = _templates_select_together(texts_by_card, permille=permille, max_carriers=50)
        assert all(len(side) == 1 for side in sides.values())


def test_sc003_over_real_sidecar_texts():
    """SC-003 on texts cut from real converted sidecars, with digit variants added."""
    from effects.infrastructure.sidecar_io import read_sidecar

    texts_by_card: dict[str, list[str]] = {}
    for path in sorted(_FIXTURE_SIDECARS.rglob("*.provenance.json")):
        sidecar = read_sidecar(path)
        real = [line.script_text for line in sidecar.lines if line.script_text]
        texts_by_card[sidecar.card] = real
        # The same card reprinted with every number bumped: one template.
        texts_by_card[f"{sidecar.card} (variant)"] = [
            "".join(str((int(c) + 1) % 10) if c.isdigit() else c for c in text)
            for text in real
        ]
    assert len(texts_by_card) > 40
    for permille in (200, 500, 800):
        sides = _templates_select_together(texts_by_card, permille=permille, max_carriers=8)
        assert all(len(side) == 1 for side in sides.values())


def test_a_template_over_the_carrier_cap_is_never_held_out():
    """Spec Story 2 scenario 2: carriers are counted per template."""
    texts_by_card = {
        f"card {i}": [f"SP$ DealDamage | NumDmg$ {i}"] for i in range(10)
    }
    chosen = select_holdout(texts_by_card, permille=1000, max_carriers=8, unit="template")
    assert chosen.texts == frozenset()
    # Under the text unit each printing is its own carrier and is eligible.
    assert select_holdout(
        texts_by_card, permille=1000, max_carriers=8, unit="text",
    ).texts


def test_the_text_unit_is_feature_023s_rule():
    """Spec Story 2 scenario 4: ``text`` keys the hash on the text itself."""
    from effects.domain.text_holdout import text_is_held_out

    texts_by_card = {f"c{i}": [f"SP$ Bespoke | Weird$ {i}"] for i in range(200)}
    chosen = select_holdout(texts_by_card, permille=300, max_carriers=8, unit="text")
    expected = {
        normalize_script_text(t)
        for texts in texts_by_card.values() for t in texts
        if text_is_held_out(t, permille=300)
    }
    assert chosen.texts == expected
    assert holdout_key("SP$ Bespoke | Weird$ 3", "text") == "SP$ Bespoke | Weird$ 3"


def test_an_unknown_unit_is_refused():
    with pytest.raises(ValueError, match="holdout unit"):
        holdout_key("SP$ Draw", "card")


_GEN2_SIDECARS = Path(__file__).parents[3] / "fixtures" / "effects" / "gen2-sidecars"


def test_a_selector_choices_value_is_masked_but_a_label_list_is_not():
    """Only chooser APIs list ``SV<n>`` labels in ``Choices$``; elsewhere it is
    a selector, and its digits are amounts like any other."""
    assert masked_template("DB$ ChooseCard | Choices$ Card.cmcLE3") == (
        "DB$ ChooseCard | Choices$ Card.cmcLE#"
    )
    assert masked_template("Choices$ SV2,SV3 | DB$ Charm") == "Choices$ SV2,SV3 | DB$ Charm"


def test_sc003_over_real_chained_gen2_texts():
    """SC-003 on chained, label-renamed texts from a real gen-2 conversion."""
    from effects.infrastructure.sidecar_io import read_sidecar

    texts_by_card: dict[str, list[str]] = {}
    chained = 0
    for path in sorted(_GEN2_SIDECARS.rglob("*.provenance.json")):
        sidecar = read_sidecar(path)
        real = [line.script_text for line in sidecar.lines if line.script_text]
        chained += sum("[SEG]" in text for text in real)
        texts_by_card[sidecar.card] = real
        texts_by_card[f"{sidecar.card} (variant)"] = [
            _bump_digits_outside_labels(text) for text in real
        ]
    assert chained, "the fixture should hold chained texts"
    for permille in (200, 500, 800):
        sides = _templates_select_together(texts_by_card, permille=permille, max_carriers=8)
        assert all(len(side) == 1 for side in sides.values())


def test_a_real_triggered_charm_keeps_its_labels_in_the_template():
    from effects.infrastructure.sidecar_io import read_sidecar

    sidecar = read_sidecar(_GEN2_SIDECARS / "cardsfolder" / "s" / "shambling_ghast.provenance.json")
    template = masked_template(sidecar.lines[0].script_text)
    assert "Execute$ SV1" in template and "SV1: Choices$ SV2,SV3" in template


def _bump_digits_outside_labels(text: str) -> str:
    import re

    labels = re.compile(r"SV\d+")
    out, last = [], 0
    for match in labels.finditer(text):
        out.append(_bump(text[last:match.start()]))
        out.append(match.group())
        last = match.end()
    out.append(_bump(text[last:]))
    return "".join(out)


def _bump(text: str) -> str:
    return "".join(str((int(c) + 1) % 10) if c.isdigit() else c for c in text)
