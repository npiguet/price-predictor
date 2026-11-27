"""The UMAP picture and its panels."""

from __future__ import annotations

import csv

import numpy as np

from effects.application.geometry_checks import (
    CachedVectors,
)
from effects.application.umap_check import check_umap
from effects.domain.evaluation_report import CheckStatus
from tests.unit.effects.selection_builders import read_cache


def _typed_cache(texts, types) -> CachedVectors:
    rng = np.random.default_rng(0)
    return CachedVectors(
        by_text={text: rng.normal(size=8) for text in texts},
        # One text with no recorded type: it is plotted as "none".
        api_types={text: types[i % 3] for i, text in enumerate(texts[:-1])},
    )


def _read_table(table) -> list[dict]:
    with table.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestUmap:
    def test_too_few_vectors_skips_and_writes_nothing(self, cache, tmp_path):
        cards, abilities = cache
        out = tmp_path / "evaluation"
        assert check_umap(
            read_cache(abilities, cards, surface="script"), out,
        ).status is CheckStatus.SKIPPED
        assert not out.exists()

    def test_a_populated_cache_writes_the_picture_and_the_table(self, tmp_path):
        """FR-110: the projection is kept, coloured by what each text does."""
        types = ["DealDamage", "Pump", "Draw"]
        texts = [f"text {i}" for i in range(30)]
        cached = _typed_cache(texts, types)
        out = tmp_path / "evaluation"

        result = check_umap(cached, out)

        assert result.status is CheckStatus.REPORTED
        png, table = out / "umap.png", out / "umap.csv"
        assert png.stat().st_size > 0
        assert table.exists()
        assert str(png) in result.detail
        assert str(table) in result.detail
        rows = _read_table(table)
        assert len(rows) == len(cached)
        assert [row["text"] for row in rows] == texts
        assert {row["api_type"] for row in rows} == {*types, "none"}

    def test_the_picture_is_the_same_file_every_run(self, tmp_path):
        rng = np.random.default_rng(1)
        cached = CachedVectors(by_text={f"t{i}": rng.normal(size=4) for i in range(20)})
        check_umap(cached, tmp_path / "a")
        check_umap(cached, tmp_path / "b")
        for name in ("umap.png", "umap.csv"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


class TestUmapPanels:
    def test_the_most_frequent_types_get_panels_and_the_rest_share_other(self):
        from effects.application.umap_check import api_type_panels

        types = ["A"] * 3 + ["B"] * 2 + ["C"] * 2 + ["D"]
        panels, panel_of = api_type_panels(types, top=2)
        # B and C tie on two; the tie goes by name, never by the cache's order.
        assert panels == ["A", "B", "other"]
        assert panel_of == ["A"] * 3 + ["B"] * 2 + ["other"] * 3

    def test_no_other_panel_when_every_type_fits(self):
        from effects.application.umap_check import api_type_panels

        assert api_type_panels(["A", "B", "A"], top=12) == (["A", "B"], ["A", "B", "A"])
