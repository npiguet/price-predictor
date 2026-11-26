"""The text an ability line is encoded from, and the role spans it carries.

Torch-free, so the survey workers that read sidecars can use it; the encoder
that reads the text is ``ability_encoder``'s.
"""

from __future__ import annotations

from effects.domain.ability_tokenizer import Surface
from effects.domain.provenance import RoleSpan, SidecarLine
from effects.domain.script_text import is_script_reading, script_reading


def encoding_text(line: SidecarLine | None, prose: str | None, surface: Surface) -> str:
    """The text a line is encoded from on ``surface``: the one place that
    decides it. A line with no prose (a variant script's) reads its
    :func:`script_reading` on either surface, which the tokenizer reads by the
    script grammar; a line with prose reads its prose, or on the script surface
    its script where it has one (a keyword-derived line has none); a line with
    neither reads the empty string, training's and the cache's alike."""
    script = line.script_text if line is not None else None
    if not prose:
        return script_reading(script) if script else ""
    if surface == Surface.SCRIPT:
        return script or prose
    return prose


def _carries_no_spans(
    line: SidecarLine | None, text: str | None, surface: Surface,
) -> bool:
    """Off the prose surface, with no line, or with no text there is nothing
    for the spans to index."""
    return surface != Surface.PROSE or line is None or not text


def _reads_script(line: SidecarLine, text: str) -> bool:
    """``text`` is the line's script or a script reading, not its prose."""
    return text == line.script_text or is_script_reading(text)


def encoding_role_spans(
    line: SidecarLine | None, text: str | None, surface: Surface,
) -> tuple[RoleSpan, ...]:
    """The role spans a line's encoding text is tokenized with (FR-069): only
    on a prose text, since the spans index the converted prose."""
    if _carries_no_spans(line, text, surface) or _reads_script(line, text):
        return ()
    return line.role_spans
