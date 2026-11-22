"""``run.py --freeze-probe-set``: freeze a corpus's selection set with no checkpoint.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from effect_knowledge_probes import common
from effect_knowledge_probes.progress import progress


def _deviance_reader(corpus: Path, cards_folders: tuple):
    """The deviance port: the corpus's targets, as a training run over it
    scores them."""
    from effects.application.selection_deviance import TargetDevianceLosses, scoring_targets

    def deviance(manifest):
        prepare = scoring_targets(corpus, manifest, cards_folders=cards_folders)
        return TargetDevianceLosses(prepare, manifest.effective_rule_outputs)

    return deviance


def freeze_selection(args: argparse.Namespace) -> int:
    """Freeze the corpus's selection set with no checkpoint (research E1): 0
    once written, 1 when a half or a component misses its target."""
    from effect_knowledge_probes import selection
    from effects.application import freeze_selection_set
    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.infrastructure.corpus_store import CorpusStore

    corpus = Path(args.corpus)
    manifest = CorpusStore(corpus).load()
    cards_folders = tuple(args.cards_folder or common.DEFAULT_CARDS_FOLDERS)
    variant_scripts = common.rooted(manifest.variant_scripts) if manifest.variant_scripts else None
    sidecars = common.sidecar_cache(cards_folders, variant_scripts)
    surface = common.manifest_surface(manifest)
    keywords_of = KeywordResolver(sidecars).keywords_of
    deviance = _deviance_reader(corpus, cards_folders)

    progress(f"freezing the selection set of {corpus}")
    return freeze_selection_set.run(
        corpus, reports_dir=args.probe_set_dir or args.reports_dir, lines=sidecars,
        probes=selection.ProbeItems(sidecars, common.acting_text_function(sidecars, surface)),
        grids=selection.grid_sources(sidecars, keywords_of), deviance=deviance,
        echo=progress,
    )
