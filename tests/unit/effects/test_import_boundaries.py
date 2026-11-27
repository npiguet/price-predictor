"""The one-way dependency rule between the three Python packages (FR-002).

``effects`` imports downward from ``sealed`` and ``price_predictor`` and never
the reverse, and it imports only the surface FR-002 declares.

Depleted collection is the case that tempts an inversion: `sealed
generate-pools --exclude-cards` needs the holdout, which only `effects` can
compute. It takes a plain file of card names written by `effects holdout-cards`
rather than importing the rule, and the first test below is what keeps it that
way. Both halves are
asserted by parsing the source tree rather than by importing it, so the test
costs nothing and does not depend on torch.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.unit.effects.builders import fresh_interpreter_env

_SRC = Path(__file__).resolve().parents[3] / "src"

# FR-002's declared surface. A value of ``None`` allows the whole module; a set
# restricts the import to those names, which is how FR-002 words the two rows
# that name a single symbol.
_ALLOWED: dict[str, set[str] | None] = {
    "price_predictor.domain.tokenizer": None,
    "price_predictor.domain.keyword_tokens": {"keyword_token_form", "multi_word_keywords"},
    "price_predictor.infrastructure.tokenizer_store": None,
    "price_predictor.application.build_vocabulary": None,
    "price_predictor.application.ridge_probes": None,
    # The Forge JVM helpers and the worker pool, one module per concern.
    "price_predictor.infrastructure.forge_classpath": None,
    "price_predictor.infrastructure.custom_card_staging": None,
    "price_predictor.infrastructure.job_objects": None,
    "price_predictor.infrastructure.jvm_processes": None,
    "price_predictor.infrastructure.worker_stops": None,
    "price_predictor.infrastructure.worker_log_files": None,
    "price_predictor.infrastructure.worker_log_mark": None,
    "price_predictor.infrastructure.forge_worker_pool": None,
    "price_predictor.infrastructure.torch_checkpoint": None,
    "price_predictor.infrastructure.atomic_write": {"write_atomically"},
    "price_predictor.infrastructure.torch_training": {"clip_per_group"},
    "price_predictor.infrastructure.append_only": None,
    "price_predictor.infrastructure.card_filenames": {"sanitize_card_name"},
    # Feature 024: the value head's cost targets parse a line's `Cost$` with
    # the existing mana-cost parser rather than a second one (research.md §
    # Adjacent prior art, FR-058a).
    "price_predictor.domain.value_objects": {"ManaCost"},
    # One reading of a dataclass field's default, shared with the price and
    # sealed CLIs rather than spelled again per package.
    "price_predictor.domain.dataclass_defaults": {"field_default"},
    # The run manifest every writing command keeps (artifact-layout skill),
    # wired by one table per CLI, the same for all four packages.
    "price_predictor.infrastructure.cli_run_manifest": {
        "READ_ONLY", "REFUSAL_DEST", "ManifestTable", "Writes", "dispatch", "tag_subcommands",
    },
    # A collector's supervisor writes its run id, resolved caps and worker-log
    # failures into the running command's run.json entry.
    "price_predictor.infrastructure.run_manifest": {"annotate_run"},
    # The worker-log scan, shared with sealed match-outcomes.
    "price_predictor.domain.worker_failures": {"WorkerFailures"},
    # The worker status line's run totals, rendered for the round report,
    # and the refusal of a run whose workers' engine is not whole.
    "price_predictor.domain.worker_status": {
        "render_fork_refusals", "render_hook_failures", "ALLOW_DEGRADED_HELP",
        "WorkerEngineRefused",
    },
    "price_predictor.infrastructure.worker_log_scan": {"WorkerLogReport"},
    # The sidecars and their digest, one definition for convert, build-corpus
    # and the collectors, and the conversion a collection reads from its tree
    # and checks against the connector JAR.
    "price_predictor.infrastructure.converted_provenance": {
        "SIDECAR_SUFFIX", "sidecar_paths", "sidecar_digest", "collection_conversion",
        "record_conversion", "trees_conversion",
    },
    # A records directory holds one conversion: a collection into it and
    # build-corpus over it refuse another.
    "price_predictor.infrastructure.records_conversion": {"record_collection", "require_joinable"},
    "price_predictor.domain.converted_provenance": {"Conversion", "ConversionRefused"},
    # A directed rule's fork probability, one formula for collect-variants and
    # sealed match-outcomes, which never imports effects.
    "price_predictor.domain.directed_fork_probabilities": {"directed_fork_probability"},
    "sealed.domain.manabase": {"compute_basic_lands"},
    "sealed.infrastructure.converted_card_locator": {"ConvertedCardLocator"},
    # The cap flags every collecting supervisor carries, one definition for
    # match-outcomes and the effects collectors alike.
    "sealed.infrastructure.effect_collection_flags": {
        "add_cap_flags", "announce_probe_state", "random_seat_error",
        "candidate_rate_error", "add_directed_fork_flags", "rule_list",
        "positive_int", "non_negative_int",
    },
    "sealed.infrastructure.directed_fork_check": {"directed_fork_error"},
    "sealed.infrastructure.dropped_events": {"dropped_events_report"},
    # The caps' one definition, which ``sealed`` reads too.
    "sealed.domain.collection_caps": {
        "DEFAULT_DIRECTED_FORK_PROBABILITY", "DEFAULT_DIRECTED_FORKS_PER_GAME",
        "DEFAULT_INTERVENTIONS_PER_GAME", "DEFAULT_MANA_CAP",
        "DEFAULT_MAX_RECORDS_PER_GAME", "DEFAULT_PLAIN_CANDIDATE_RATE",
        "DEFAULT_PLAYABILITY_RATE", "DEFAULT_PROBE_KEYWORDS", "DEFAULT_PROBES_PER_GAME",
        "DEFAULT_RANDOM_SEAT_PROBABILITY", "DEFAULT_RANDOM_SEAT_SHARE",
        "DEFAULT_ZONE_CANDIDATE_RATE", "DIRECTED_RULE_PROBABILITY_PREFIX",
        "CollectionCaps",
    },
    # The connector and the launch its ``start`` takes, with its three parts.
    "sealed.infrastructure.match_worker_connector": {
        "MatchWorkerConnector", "WorkerLaunch", "WorkerDecks", "WorkerRecords",
        "WorkerStopProtocol",
    },
    "sealed.infrastructure.pool_file_reader": {"format_generated_deck"},
}

_GOVERNED_PACKAGES = ("price_predictor", "sealed")


def _python_files(package: str) -> list[Path]:
    root = _SRC / package
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


_Import = tuple[str, tuple[str, ...], int]


def _imports_of_node(node: ast.AST) -> list[_Import]:
    """The absolute imports one syntax node makes, as ``(module, names, lineno)``."""
    if isinstance(node, ast.Import):
        return [(alias.name, (), node.lineno) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return _absolute_from_import(node)
    return []


def _absolute_from_import(node: ast.ImportFrom) -> list[_Import]:
    if node.level or node.module is None:
        return []
    return [(node.module, tuple(a.name for a in node.names), node.lineno)]


def _imported_modules(path: Path) -> list[_Import]:
    """Every absolute import in ``path`` as ``(module, names, lineno)``.

    ``names`` is empty for a plain ``import x.y`` and holds the imported
    identifiers for ``from x.y import a, b``. Relative imports are skipped —
    they can only reach within the importing package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [found for node in ast.walk(tree) for found in _imports_of_node(node)]


def _import_sites(
    packages: tuple[str, ...], in_scope: Callable[[Path], bool] = lambda path: True,
) -> Iterator[tuple[Path, str, tuple[str, ...], int]]:
    """Every import of the ``packages``' files that ``in_scope`` keeps, with
    the file's path relative to ``src``."""
    files = (path for package in packages for path in _python_files(package))
    for path in filter(in_scope, files):
        for module, names, lineno in _imported_modules(path):
            yield path.relative_to(_SRC), module, names, lineno


def _imports_matching(
    packages: tuple[str, ...],
    reaches: Callable[[str], bool],
    in_scope: Callable[[Path], bool] = lambda path: True,
) -> list[str]:
    """One ``file:line imports module`` line per import ``reaches`` accepts."""
    return [
        f"{where}:{lineno} imports {module}"
        for where, module, _names, lineno in _import_sites(packages, in_scope)
        if reaches(module)
    ]


def _is_under(root: str) -> Callable[[str], bool]:
    """Whether a module is ``root`` itself or one of its submodules."""
    return lambda module: module == root or module.startswith(root + ".")


@pytest.mark.parametrize("package", _GOVERNED_PACKAGES)
def test_lower_packages_never_import_effects(package: str) -> None:
    """``price_predictor`` and ``sealed`` must not reach up into ``effects``."""
    violations = _imports_matching((package,), _is_under("effects"))
    assert not violations, (
        f"{package} imports effects, inverting the dependency direction "
        f"FR-002 fixes:\n  " + "\n  ".join(violations)
    )


def _surface_violations(
    where: Path, module: str, names: tuple[str, ...], lineno: int,
) -> list[str]:
    """What one cross-package import of ``effects`` takes beyond FR-002's surface."""
    if module not in _ALLOWED:
        return [f"{where}:{lineno} imports {module}"]
    allowed_names = _ALLOWED[module]
    if allowed_names is None:
        return []
    declared = f"FR-002 declares only {sorted(allowed_names)}"
    if not names:
        # ``import pkg.module`` reaches every name the module has.
        return [f"{where}:{lineno} imports {module} whole; {declared}"]
    return [
        f"{where}:{lineno} imports {module}.{name}; {declared}"
        for name in names
        if name not in allowed_names
    ]


def test_effects_imports_only_the_declared_surface() -> None:
    """Every cross-package import out of ``effects`` is one FR-002 declares."""
    violations = [
        violation
        for where, module, names, lineno in _import_sites(("effects",))
        if module.startswith(_GOVERNED_PACKAGES)
        for violation in _surface_violations(where, module, names, lineno)
    ]
    assert not violations, (
        "effects imports outside the surface FR-002 declares:\n  "
        + "\n  ".join(violations)
    )


#: The composition root, ``cli.py`` and the ``cli_<command>.py`` modules it
#: composes: they wire each command's application service to the adapters it
#: runs on, so they are the infrastructure modules that name both.
_COMPOSITION_ROOT = "cli.py"
_COMPOSITION_ROOT_PARTS = "cli_"


def _is_infrastructure_adapter(path: Path) -> bool:
    composition_root = path.name == _COMPOSITION_ROOT or path.name.startswith(
        _COMPOSITION_ROOT_PARTS,
    )
    return path.parent.name == "infrastructure" and not composition_root


def test_infrastructure_never_imports_application() -> None:
    """Infrastructure is the adapters' layer; orchestration lives above it.

    Lazy imports count too, since ``ast.walk`` reaches inside functions: a
    deferred import is still a dependency, only a later one.
    """
    violations = _imports_matching(
        ("effects",), _is_under("effects.application"), _is_infrastructure_adapter,
    )
    assert not violations, (
        "effects.infrastructure imports effects.application:\n  "
        + "\n  ".join(violations)
    )


def _is_above_the_domain(module: str) -> bool:
    return module.startswith(("effects.application", "effects.infrastructure"))


def test_the_domain_imports_neither_layer_above_it() -> None:
    violations = _imports_matching(
        ("effects",), _is_above_the_domain, lambda path: path.parent.name == "domain",
    )
    assert not violations, (
        "effects.domain imports a layer above it:\n  " + "\n  ".join(violations)
    )


def _is_validator(path: Path) -> bool:
    return path.name == "validate_corpus.py" or path.parent.name == "corpus_validation"


def test_the_validator_imports_no_infrastructure() -> None:
    """``validate_corpus`` and ``corpus_validation`` receive their shard reader
    and record renderer from the command (``corpus_validation.ports``)."""
    violations = _imports_matching(
        ("effects",), _is_under("effects.infrastructure"), _is_validator,
    )
    assert not violations, (
        "the validator imports effects.infrastructure:\n  " + "\n  ".join(violations)
    )


def test_no_package_imports_the_scripts() -> None:
    """``scripts/`` imports the packages and never the reverse (FR-078).

    The probe cores a training run scores through live in ``effects`` itself;
    a package reaching into ``scripts/`` would make a run depend on a tree
    that is not installed.
    """
    violations = _imports_matching(("effects", *_GOVERNED_PACKAGES), _is_under("scripts"))
    assert not violations, "a package imports scripts/:\n  " + "\n  ".join(violations)


#: Modules the corpus workers and record readers import, which must not pull in
#: torch: every build-corpus survey and write worker, and every rule-counts
#: worker, is a fresh process, and loading torch in each costs seconds and
#: hundreds of megabytes for nothing.
_TORCH_FREE = (
    "effects.domain.keyword_names",
    "effects.domain.records",
    "effects.domain.record_quality",
    "effects.domain.corpus_curation",
    "effects.domain.outcome_signature",
    "effects.domain.condition_balance",
    "effects.domain.sampling",
    "effects.domain.split",
    "effects.domain.baselines",
    "effects.domain.coverage",
    "effects.domain.canonical_digest",
    "effects.domain.strata",
    "effects.domain.corpus_plan",
    "effects.domain.corpus_types",
    "effects.domain.seeded_draw",
    "effects.domain.rule_budgets",
    "effects.domain.stratum_placement",
    "effects.domain.score_scales",
    "effects.domain.probe_readouts",
    "effects.domain.board_edits",
    "effects.domain.grid_edits",
    "effects.domain.selection_set",
    "effects.domain.fork_operation",
    "effects.domain.rule_catalogue",
    "effects.domain.rule_outcomes",
    "effects.domain.protection_copies",
    "effects.domain.protection_aura",
    "effects.domain.protection_chain",
    "effects.domain.protection_damage",
    "effects.domain.protection_model",
    "effects.domain.protection_opportunities",
    "effects.domain.protection_playability",
    "effects.domain.relation_types",
    "effects.domain.output_list_survey",
    "effects.domain.output_vocabularies",
    "effects.domain.input_vocabularies",
    "effects.domain.input_layout",
    "effects.domain.card_features",
    "effects.domain.slot_lines",
    "effects.domain.slot_relations",
    "effects.domain.surface_builder",
    "effects.domain.aux_head_targets",
    "effects.domain.rule_detectors",
    "effects.domain.rule_detectors.sba_zone_move",
    "effects.domain.rule_detectors.trigger_mode",
    "effects.domain.rule_detectors.keyword_combat",
    "effects.domain.rule_detectors.verdict_bit",
    "effects.domain.rule_detectors.continuous_contribution",
    "effects.domain.rule_detectors.rewrite_event",
    "effects.domain.rule_detectors.legality_membership",
    "effects.domain.rule_detectors.outcome_event",
    "effects.domain.rule_detectors.resolution_outcome",
    "effects.domain.rule_detectors.sba_game_loss",
    "effects.domain.rule_detectors.acting_line",
    "effects.domain.rule_breadth",
    "effects.domain.rule_mask",
    "effects.domain.rule_output_carriage",
    "effects.domain.shard_generation",
    "effects.domain.kept_keyword_flags",
    "effects.domain.forge_vocabularies",
    "effects.infrastructure.rules_catalogue_io",
    "effects.application.rule_counts",
    "effects.application.rule_breadth",
    "effects.application.corpus_survey",
    "effects.application.corpus_write",
    "effects.application.corpus_report",
    "effects.infrastructure.build_inputs",
    "effects.application.build_corpus",
    "effects.application.build_corpus_config",
    "effects.application.corpus_rule_catalogue",
    "effects.application.corpus_decisions",
    "effects.application.stratum_budgets",
    "effects.application.freeze_selection_set",
)


@pytest.mark.parametrize("module", _TORCH_FREE)
def test_corpus_side_modules_never_load_torch(module: str) -> None:
    import subprocess
    import sys

    probe = f"import sys, {module}\nprint('torch' in sys.modules)\n"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
        env=fresh_interpreter_env(),
    )
    assert result.stdout.strip() == "False", f"{module} loads torch"
