"""The ``rule-counts`` flags as one value.

Each field's ``help`` metadata is its flag's help, so a default lives here
alone; ``cli.add_config_flags`` turns the fields into flags, and
:meth:`RuleCountsFlags.config` is what the command runs on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from effects.domain import default_paths
from effects.infrastructure.converted_trees import SourceTrees
from price_predictor.infrastructure.forge_worker_pool import MAX_WORKERS, worker_count


@dataclass(frozen=True, slots=True)
class RuleCountsFlags:
    """The ``rule-counts`` flags."""

    records_dir: Path = field(kw_only=True, metadata={"help": (
        "Shards to count: gen-3 records only"
    )})
    output: Path = field(kw_only=True, metadata={"help": (
        "Counts file, inside an inputs run"
    )})
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    #: Required, as ``validate-corpus``'s: an empty value reads none.
    variant_scripts: Path | None = field(kw_only=True)
    rules_dir: Path = field(default=Path("rules"), metadata={"help": (
        "Scenarios and their detector declarations (default: %(default)s)"
    )})
    workers: int = field(default=MAX_WORKERS, metadata={"type": worker_count, "help": (
        f"Parallel shard readers, 1 to {MAX_WORKERS} (default: %(default)s)"
    )})

    @property
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        return SourceTrees.of(self.cards_folders, self.variant_scripts)

    def config(self):
        """What :func:`effects.application.rule_counts.run` counts with."""
        from effects.application.rule_counts import RuleCountsConfig

        return RuleCountsConfig(
            records_dir=self.records_dir, output=self.output, rules_dir=self.rules_dir,
            sidecar_roots=self.source_trees.sidecar_roots(), workers=self.workers,
        )
