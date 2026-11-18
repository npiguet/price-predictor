"""Put ``scripts/`` on the import path, as a script running from it would have."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402
from effect_knowledge_probes import common, labels  # noqa: E402

from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS, curated_fixture_records  # noqa: E402
from tests.unit.scripts.knowledge_probe_support import join_triggers  # noqa: E402


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


@pytest.fixture(scope="module")
def sidecars():
    return common.sidecar_cache(GEN3_SIDECAR_FOLDERS)


@pytest.fixture(scope="module")
def acting(sidecars):
    return common.acting_text_function(sidecars, "script")


@pytest.fixture(scope="module")
def items_by_record(records, sidecars, acting) -> dict[str, list]:
    """Every fixture record's board items by record id, triggers joined as a freeze joins them."""
    join = join_triggers(records)
    return {r.record_id: labels.record_items(r, sidecars, join, acting) for r in records}
