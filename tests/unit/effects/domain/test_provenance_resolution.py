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
