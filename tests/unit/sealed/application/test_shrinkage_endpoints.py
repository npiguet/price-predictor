"""SC-005 endpoint assertions for the shrinkage helper.

Tightens the boundary cases beyond ``test_shrinkage.py``:
- low-n cards (e.g. wins_when_in_deck = 2) shift visibly between k=0 and k=20.
- high-n cards (e.g. wins_when_in_deck >= 1000) stay within 0.005 of raw.
"""

from __future__ import annotations

from sealed.application.train_encoder import _shrink


class TestShrinkageEndpoints:
    def test_low_n_shifts_visibly(self):
        # SC-005: a card with two in-deck observations shifts noticeably
        # between k=0 and k=20.
        wp, wd = 2, 2
        raw = _shrink(wp, wd, k=0)
        with_k20 = _shrink(wp, wd, k=20)
        assert raw == 1.0
        # Visible shift: at least 0.05 between the two snapshots.
        assert abs(raw - with_k20) >= 0.05

    def test_low_n_zero_wins_shifts_visibly(self):
        wp, wd = 0, 2
        raw = _shrink(wp, wd, k=0)
        with_k20 = _shrink(wp, wd, k=20)
        assert raw == 0.0
        assert abs(raw - with_k20) >= 0.05

    def test_high_n_stable_under_k20(self):
        # Typical high-observation case from SC-005: wins_when_in_deck >= 1000.
        # With wd=10000 the relative shift collapses below 0.005.
        wp, wd = 7000, 10000
        raw = _shrink(wp, wd, k=0)
        with_k20 = _shrink(wp, wd, k=20)
        assert abs(raw - with_k20) < 0.005

    def test_balanced_high_n_unchanged(self):
        # raw=0.5 should be a fixed point under any k.
        wp, wd = 500, 1000
        assert _shrink(wp, wd, k=20) == 0.5
