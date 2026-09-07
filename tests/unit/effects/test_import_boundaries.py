"""The one-way dependency rule between the three Python packages (FR-002).

``effects`` imports downward from ``sealed`` and ``price_predictor`` and never
the reverse, and it imports only the surface FR-002 declares. Both halves are
asserted by parsing the source tree rather than by importing it, so the test
costs nothing and does not depend on torch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src"

# FR-002's declared surface. A value of ``None`` allows the whole module; a set
# restricts the import to those names, which is how FR-002 words the two rows
# that name a single symbol.
_ALLOWED: dict[str, set[str] | None] = {
    "price_predictor.domain.tokenizer": None,
    "price_predictor.application.build_vocabulary": None,
    "price_predictor.application.ridge_probes": None,
    "price_predictor.infrastructure.forge_jvm": None,
    "price_predictor.infrastructure.torch_checkpoint": None,
    "price_predictor.infrastructure.torch_training": {"clip_per_group"},
    "price_predictor.infrastructure.append_only": None,
    "sealed.domain.manabase": {"compute_basic_lands"},
    "sealed.domain.card_embedding_layout": None,
    "sealed.infrastructure.converted_card_locator": {"ConvertedCardLocator"},
    "sealed.infrastructure.embedding_store": None,
}

_GOVERNED_PACKAGES = ("price_predictor", "sealed")


def _python_files(package: str) -> list[Path]:
    root = _SRC / package
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> list[tuple[str, tuple[str, ...], int]]:
    """Every absolute import in ``path`` as ``(module, names, lineno)``.

    ``names`` is empty for a plain ``import x.y`` and holds the imported
    identifiers for ``from x.y import a, b``. Relative imports are skipped —
    they can only reach within the importing package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, tuple[str, ...], int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, (), node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                continue
            found.append(
                (node.module, tuple(a.name for a in node.names), node.lineno)
            )
    return found


@pytest.mark.parametrize("package", _GOVERNED_PACKAGES)
def test_lower_packages_never_import_effects(package: str) -> None:
    """``price_predictor`` and ``sealed`` must not reach up into ``effects``."""
    violations = [
        f"{path.relative_to(_SRC)}:{lineno} imports {module}"
        for path in _python_files(package)
        for module, _names, lineno in _imported_modules(path)
        if module == "effects" or module.startswith("effects.")
    ]
    assert not violations, (
        f"{package} imports effects, inverting the dependency direction "
        f"FR-002 fixes:\n  " + "\n  ".join(violations)
    )


def test_effects_imports_only_the_declared_surface() -> None:
    """Every cross-package import out of ``effects`` is one FR-002 declares."""
    violations: list[str] = []
    for path in _python_files("effects"):
        where = path.relative_to(_SRC)
        for module, names, lineno in _imported_modules(path):
            if not module.startswith(_GOVERNED_PACKAGES):
                continue
            if module not in _ALLOWED:
                violations.append(f"{where}:{lineno} imports {module}")
                continue
            allowed_names = _ALLOWED[module]
            if allowed_names is None:
                continue
            for name in names:
                if name not in allowed_names:
                    violations.append(
                        f"{where}:{lineno} imports {module}.{name}; FR-002 "
                        f"declares only {sorted(allowed_names)}"
                    )
    assert not violations, (
        "effects imports outside the surface FR-002 declares:\n  "
        + "\n  ".join(violations)
    )


def test_effects_never_imports_the_sealed_application_layer() -> None:
    """FR-002 declares nothing from ``sealed.application``; it stays disjoint."""
    violations = [
        f"{path.relative_to(_SRC)}:{lineno} imports {module}"
        for path in _python_files("effects")
        for module, _names, lineno in _imported_modules(path)
        if module == "sealed.application" or module.startswith("sealed.application.")
    ]
    assert not violations, (
        "effects imports sealed.application; train-scorer Phase A is re-run as "
        "a subprocess so the two application layers stay disjoint:\n  "
        + "\n  ".join(violations)
    )
