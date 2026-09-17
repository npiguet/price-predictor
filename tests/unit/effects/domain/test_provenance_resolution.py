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


def _sidecar() -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card="mogg war marshal", script_file=_SCRIPT,
        lines=(SidecarLine(line_index=0, line_kind="triggered", provenance=(_LINE,),
                           script_text="when this enters or dies"),),
        dropped_keys=(_DROPPED,),
    )


def test_a_rendered_key_is_a_line():
    assert _sidecar().resolution_of(_LINE) is KeyResolution.LINE


def test_a_dropped_key_is_dropped():
    assert _sidecar().resolution_of(_DROPPED) is KeyResolution.DROPPED


def test_a_key_past_the_declared_indices_is_runtime_only():
    assert _sidecar().resolution_of(
        ProvenanceKey(_SCRIPT, 0, "spell", 7)
    ) is KeyResolution.RUNTIME_ONLY


def test_a_key_in_neither_list_raises():
    """A kind this face declared nothing for at all is the mismatch case.

    ``is_runtime_only`` is positional: it answers True only past an index the
    sidecar declared, so a kind with nothing declared has no tail to sit in.
    """
    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        _sidecar().resolution_of(ProvenanceKey(_SCRIPT, 0, "static", 0))
