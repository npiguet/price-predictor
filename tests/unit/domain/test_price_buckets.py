"""Tests for price-bucket value objects."""

from __future__ import annotations

import math

import pytest

from price_predictor.domain.price_buckets import (
    REPORTING_BUCKETS,
    SAMPLING_BUCKETS,
    BucketScheme,
    PriceBucket,
)


class TestPriceBucket:
    def test_contains_low_inclusive(self) -> None:
        b = PriceBucket("two-five", 2.0, 5.0)
        assert b.contains(2.0) is True

    def test_contains_high_exclusive(self) -> None:
        b = PriceBucket("two-five", 2.0, 5.0)
        assert b.contains(5.0) is False

    def test_contains_outside(self) -> None:
        b = PriceBucket("two-five", 2.0, 5.0)
        assert b.contains(1.99) is False
        assert b.contains(4.99) is True


class TestSamplingBuckets:
    @pytest.mark.parametrize("price,expected_idx", [
        (0.0, 0), (0.5, 0), (1.99, 0),
        (2.0, 1), (9.99, 1),
        (10.0, 2), (49.99, 2),
        (50.0, 3), (1_000_000.0, 3),
    ])
    def test_index_for(self, price: float, expected_idx: int) -> None:
        assert SAMPLING_BUCKETS.index_for(price) == expected_idx

    def test_has_four_buckets(self) -> None:
        assert len(SAMPLING_BUCKETS) == 4

    def test_negative_price_falls_into_first_bucket(self) -> None:
        assert SAMPLING_BUCKETS.index_for(-1.0) == 0


class TestReportingBuckets:
    @pytest.mark.parametrize("price,expected_label", [
        (0.05, "<€0.10"),
        (0.10, "€0.10–0.50"),
        (0.49, "€0.10–0.50"),
        (0.50, "€0.50–2"),
        (1.99, "€0.50–2"),
        (2.00, "€2–10"),
        (9.99, "€2–10"),
        (10.0, "€10–50"),
        (49.99, "€10–50"),
        (50.0, ">€50"),
    ])
    def test_label_for_price(self, price: float, expected_label: str) -> None:
        idx = REPORTING_BUCKETS.index_for(price)
        assert REPORTING_BUCKETS.buckets[idx].label == expected_label

    def test_has_six_buckets(self) -> None:
        assert len(REPORTING_BUCKETS) == 6


class TestBucketSchemeIteration:
    def test_iter_yields_buckets_in_order(self) -> None:
        scheme = BucketScheme((
            PriceBucket("a", 0.0, 1.0),
            PriceBucket("b", 1.0, math.inf),
        ))
        labels = [b.label for b in scheme]
        assert labels == ["a", "b"]
