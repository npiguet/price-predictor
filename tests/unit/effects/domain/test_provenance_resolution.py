"""``resolution_of``: the four answers a key can get at the join."""

from __future__ import annotations

import pytest

from effects.domain.provenance import (
    KeyResolution,
    ProvenanceKey,
    ProvenanceSidecar,
    SidecarLine,
)

_SCRIPT = "cardsfolder/m/mogg_war_marshal.txt"
_LINE = ProvenanceKey(_SCRIPT, 0, "trigger", 0)
_DROPPED = ProvenanceKey(_SCRIPT, 0, "spell", 0)
#: A second dropped spell, two along from the first, so the face declares a
#: ``spell`` range with a gap in it. Index 1 is the only shape left that is a
#: mismatch rather than a runtime-only trait.
_DROPPED_FAR = ProvenanceKey(_SCRIPT, 0, "spell", 2)
_GAP = ProvenanceKey(_SCRIPT, 0, "spell", 1)


def _sidecar() -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card="mogg war marshal", script_file=_SCRIPT,
        lines=(SidecarLine(line_index=0, line_kind="triggered", provenance=(_LINE,),
                           script_text="when this enters or dies"),),
        dropped_keys=(_DROPPED, _DROPPED_FAR),
    )


def test_a_rendered_key_is_a_line():
    assert _sidecar().resolution_of(_LINE) is KeyResolution.LINE


def test_a_dropped_key_is_dropped():
    assert _sidecar().resolution_of(_DROPPED) is KeyResolution.DROPPED


def test_a_key_past_the_declared_indices_is_runtime_only():
    assert _sidecar().resolution_of(
        ProvenanceKey(_SCRIPT, 0, "spell", 7)
    ) is KeyResolution.RUNTIME_ONLY


def test_an_undeclared_trait_kind_is_runtime_only():
    """A static that grants a trigger to other permanents keys the granted
    trigger to the granting card's script, which declares no trigger at all."""
    assert _sidecar().resolution_of(
        ProvenanceKey(_SCRIPT, 0, "static", 0)
    ) is KeyResolution.RUNTIME_ONLY


def test_an_undeclared_face_is_runtime_only():
    """A disguise creature's face-down state keys to a face the converted
    script never wrote down."""
    assert _sidecar().resolution_of(
        ProvenanceKey(_SCRIPT, 1, "keyword", 0)
    ) is KeyResolution.RUNTIME_ONLY


def test_a_gap_inside_a_declared_range_raises():
    """The one shape left that means the sidecar describes another card.

    ``dropped_keys`` is declared-minus-claimed, so every index the converter
    parsed sits in one of the two lists. A gap between them can only come from
    a sidecar rebuilt against a different trait list.
    """
    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        _sidecar().resolution_of(_GAP)


# ── charm modes (FR-005a) ───────────────────────────────────────────────

_CHARM_SCRIPT = "cardsfolder/c/cryptic_command.txt"
_CHARM_ROOT = ProvenanceKey(_CHARM_SCRIPT, 0, "spell", 0)


def _mode(option: int) -> ProvenanceKey:
    return ProvenanceKey(_CHARM_SCRIPT, 0, "spell", 0, option)


def _charm_sidecar() -> ProvenanceSidecar:
    """Cryptic Command's real line layout, with the gen-2 mode keys on its options.

    The die-roll ``option`` line after the modes stands for a non-charm option:
    it carries no key, so it ends the root's run of mode rows.
    """
    return ProvenanceSidecar(
        card="cryptic command", script_file=_CHARM_SCRIPT,
        lines=(
            SidecarLine(line_index=3, line_kind="spell", provenance=(_CHARM_ROOT,),
                        script_api_type="Charm",
                        script_text="CharmNum$ 2 | Choices$ SV1,SV2 | SP$ Charm"),
            SidecarLine(line_index=4, line_kind="option", provenance=(_mode(0),),
                        script_api_type="Counter",
                        script_text="SV1: DB$ Counter | TargetType$ Spell"),
            SidecarLine(line_index=5, line_kind="option", provenance=(_mode(1),),
                        script_api_type="Draw",
                        script_text="SV1: DB$ Draw | NumCards$ 1"),
            SidecarLine(line_index=6, line_kind="option", provenance=(),
                        script_text=""),
            SidecarLine(line_index=7, line_kind="spell", provenance=(
                ProvenanceKey(_CHARM_SCRIPT, 0, "spell", 1),),
                script_text="SP$ Draw"),
        ),
    )


def test_a_mode_key_never_equals_its_root():
    assert _mode(0) != _CHARM_ROOT
    assert _mode(0).root == _CHARM_ROOT
    assert _CHARM_ROOT.root is _CHARM_ROOT


def test_a_mode_key_and_its_root_resolve_to_different_rows():
    sidecar = _charm_sidecar()
    assert sidecar.row_for(_CHARM_ROOT) == 0
    assert sidecar.row_for(_mode(0)) == 1
    assert sidecar.row_for(_mode(1)) == 2


def test_option_rows_after_a_charm_root_are_its_keyed_modes():
    assert _charm_sidecar().option_rows_after(0) == [1, 2]


def test_an_unkeyed_option_line_is_not_a_mode_row():
    sidecar = _charm_sidecar()
    assert sidecar.option_rows_after(2) == []
    assert sidecar.option_rows_after(3) == []


def test_a_line_with_no_options_has_no_option_rows():
    assert _charm_sidecar().option_rows_after(4) == []


def test_a_key_round_trips_with_and_without_option():
    for key in (_CHARM_ROOT, _mode(2)):
        assert ProvenanceKey.from_dict(key.as_dict()) == key
    assert "option" not in _CHARM_ROOT.as_dict()
    assert _mode(2).as_dict()["option"] == 2
