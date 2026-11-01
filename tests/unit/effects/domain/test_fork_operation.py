"""The Python half of the fork-operation contract pair (T017).

Reads the Java enum's source and compares its wire names with
:class:`ForkOperation`, in order; ``ForkOperationContractTest`` reads this
side's source the other way.
"""

from __future__ import annotations

import re
from pathlib import Path

from effects.domain.fork_operation import ForkOperation
from effects.domain.rule_catalogue import ForkOperation as CatalogueForkOperation

_JAVA = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main" / "java"
         / "com" / "pricepredictor" / "connector" / "effects" / "directed" / "ForkOperation.java")


def test_java_and_python_spell_the_operations_alike():
    constants = re.findall(r'^\s+[A-Z_]+\("([a-z_]+)"\)[,;]', _JAVA.read_text(encoding="utf-8"),
                           flags=re.M)
    assert constants == [str(op) for op in ForkOperation]


def test_the_catalogue_names_the_same_enum():
    assert CatalogueForkOperation is ForkOperation
