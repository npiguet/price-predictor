"""``train-effect-model``'s check that the head's closed output lists cover the corpus.

The head's subtype and keyword columns are drawn over the lists checked in as
:mod:`effects.domain.output_vocabularies`; the corpus manifest records the lists
its own records name (every subtype and keyword a record's effects grant,
remove or give a token). A name the manifest lists and the module lacks is a
target the head drops: no column holds it, and a reader takes the silence for
"none gained". Training refuses such a corpus instead of reading it so.
"""

from __future__ import annotations

from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED

REGENERATE = (
    "python scripts/scan_output_vocabularies.py --manifest <corpus>/manifest.json "
    "--keyword-definitions <keyword-definitions.json>"
)


class ClosedListMismatchError(ValueError):
    """The corpus manifest names subtypes or keywords the closed output lists lack."""


def missing_names(manifest: CorpusManifest) -> dict[str, tuple[str, ...]]:
    """``list -> names`` the manifest holds that the module's list lacks, in the
    manifest's order; a list with none is absent."""
    found = {
        "SUBTYPES_CHANGED": tuple(n for n in manifest.subtypes if n not in SUBTYPES_CHANGED),
        "KEYWORDS_CHANGED": tuple(n for n in manifest.keywords if n not in KEYWORDS_CHANGED),
    }
    return {name: names for name, names in found.items() if names}


def require_closed_lists(manifest: CorpusManifest) -> None:
    """Raise :class:`ClosedListMismatchError` naming every name the manifest's
    subtype and keyword lists hold that ``output_vocabularies`` lacks."""
    missing = missing_names(manifest)
    if not missing:
        return
    detail = "; ".join(f"{name} lacks {', '.join(names)}" for name, names in missing.items())
    raise ClosedListMismatchError(
        f"the corpus manifest names subtypes or keywords the head's closed output "
        f"lists lack ({detail}): those changes would train as 'none gained'. "
        f"Regenerate the lists with `{REGENERATE}`, which changes the output "
        "widths and so needs a new training run, or rebuild the corpus."
    )
