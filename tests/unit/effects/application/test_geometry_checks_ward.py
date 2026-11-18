"""The ward canary and its verdict."""

from __future__ import annotations

import math

import numpy as np

from effects.application.geometry_checks import (
    NEIGHBOUR_QUERIES,
    check_ward,
    evaluate_ward_canary,
)
from effects.domain.evaluation_report import CheckStatus
from effects.domain.line_query import normalize_prose
from effects.domain.ward_twins import BARE_KEYWORDS, WARD, WARD_TWINS
from tests.unit.effects.application.geometry_checks_support import (
    _ward_cache,
)
from tests.unit.effects.builders import EFFECTS_FIXTURES, read_cache


class TestWardCanary:
    def test_it_skips_when_ward_is_not_in_the_cache(self, cache):
        cards, abilities = cache
        result = check_ward(read_cache(abilities, cards, surface="script"))
        assert result.status is CheckStatus.SKIPPED
        assert "ward {2}" in result.detail

    def test_it_resolves_every_line_through_its_prose(self, tmp_path):
        result = check_ward(_ward_cache(tmp_path))
        assert result.status is CheckStatus.REPORTED
        assert "every functional twin" in result.detail
        assert result.values["twins_compared"] == len(WARD_TWINS)
        assert result.values["bare_keywords_compared"] == len(BARE_KEYWORDS)

    def test_a_twin_that_resolves_to_nothing_is_named(self, tmp_path):
        """Otherwise the canary compares against fewer twins than it lists."""
        absent = WARD_TWINS[-1]
        result = check_ward(_ward_cache(tmp_path, missing=(absent,)))
        assert result.values["twins_compared"] == len(WARD_TWINS) - 1
        assert "unresolved" in result.detail
        assert absent.prose in result.detail

    def test_the_ward_text_and_its_twins_are_checked_in(self):
        assert WARD.prose == "ward {2}"
        assert len(WARD_TWINS) >= 3
        assert len(BARE_KEYWORDS) >= 10
        assert "ward" not in " ".join(t.prose for t in WARD_TWINS).lower()

    def test_every_line_is_spelled_the_way_the_converter_renders_it(self):
        """Lowercase but ``CARDNAME``, no reminder text, a period on a sentence."""
        for twin in WARD_TWINS:
            bare = twin.prose.replace("CARDNAME", "")
            assert bare == bare.lower()
            assert "this creature" not in twin.prose
            assert twin.prose.endswith(".")
            assert twin.card
        for line in (WARD, *BARE_KEYWORDS):
            assert "(" not in line.prose
            assert not line.prose.endswith(".")

    def test_every_pinned_line_is_one_its_card_prints(self):
        """Twins and neighbour queries alike: a pin to a card that does not
        print the line resolves to nothing on the real corpus. Read on
        ``pinned-lines/``, each pinned card's converted script and sidecar
        copied from the gen-3 converted tree (``output/effects/gen3/converted``):
        a new pin needs its card copied there too."""
        from effects.infrastructure.sidecar_io import (
            converted_text_path,
            prose_lines,
            read_sidecar,
        )
        from sealed.infrastructure.converted_card_locator import (
            ConvertedCardLocator,
        )

        locator = ConvertedCardLocator(EFFECTS_FIXTURES / "pinned-lines" / "cardsfolder")
        for query in (*WARD_TWINS, *NEIGHBOUR_QUERIES):
            path = locator.expected_path(query.card, ".provenance.json")
            sidecar = read_sidecar(path)
            rendered = prose_lines(converted_text_path(path))
            printed = {
                normalize_prose(line.prose_in(rendered) or "")
                for line in sidecar.lines
            }
            assert normalize_prose(query.prose) in printed, query.card


class TestWardCanaryVerdict:
    def _vectors(self, twin_distance: float):
        ward = np.array([1.0, 0.0, 0.0])
        # Twins sit at the requested angle; bare keywords sit further away.
        angle = twin_distance
        twins = [np.array([1.0 - angle, math.sqrt(max(0.0, 2 * angle - angle ** 2)), 0.0])]
        bare = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        return ward, twins, bare

    def test_ward_nearer_its_twins_than_the_median_keyword_reads_as_such(self):
        ward, twins, bare = self._vectors(0.05)
        result = evaluate_ward_canary(ward, twins, bare)
        assert "every functional twin is nearer" in result.detail

    def test_a_twin_farther_than_the_median_is_reported(self):
        ward = np.array([1.0, 0.0, 0.0])
        twins = [np.array([-1.0, 0.0, 0.0])]  # opposite: distance 2
        bare = [np.array([0.0, 1.0, 0.0])]    # orthogonal: distance 1
        result = evaluate_ward_canary(ward, twins, bare)
        assert "sit farther from ward" in result.detail

    def test_the_canary_reports_and_does_not_block(self):
        ward, twins, bare = self._vectors(0.05)
        assert evaluate_ward_canary(ward, twins, bare).status is (
            CheckStatus.REPORTED
        )

    def test_an_empty_cache_skips_rather_than_failing(self):
        result = evaluate_ward_canary(np.ones(3), [], [])
        assert result.status is CheckStatus.SKIPPED
