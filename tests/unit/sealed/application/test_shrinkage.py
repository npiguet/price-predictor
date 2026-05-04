"""Unit tests for the Bayesian shrinkage helper (FR-011, SC-005)."""

from __future__ import annotations

from sealed.application.train_encoder import _shrink


class TestShrinkageBoundaries:
    def test_k_zero_recovers_raw_ratio(self):
        assert _shrink(2, 5, k=0) == 2 / 5

    def test_k20_low_n_pulls_toward_half(self):
        # FR-011: k=20, wp=2, wd=2 → (2 + 10) / (2 + 20) = 12/22 ≈ 0.545
        shrunk = _shrink(2, 2, k=20)
        assert shrunk < 0.6, f"shrunk={shrunk} should be meaningfully below 1.0"
        assert shrunk > 0.5, f"shrunk={shrunk} should still be above 0.5 (raw was 1.0)"

    def test_k20_high_n_close_to_raw(self):
        # SC-005: high-n labels within a few thousandths of raw. With k=20
        # the relative shrink is ~k/(wd+k) of |raw - 0.5|. Push wd far
        # enough that the shift collapses to thousandths.
        wp, wd = 8000, 10000
        raw = wp / wd
        shrunk = _shrink(wp, wd, k=20)
        assert abs(shrunk - raw) < 0.005, f"shrunk={shrunk} raw={raw}"

    def test_k_zero_zero_in_deck_invalid(self):
        # Caller must filter wins_when_in_deck == 0 before computing the
        # raw ratio (FR-012). _shrink doesn't blow up at k > 0 because the
        # denominator stays positive.
        assert _shrink(0, 0, k=20) == 0.5
