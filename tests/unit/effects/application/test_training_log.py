"""The trainer's log lines: what each section says and in what order."""

from __future__ import annotations

from effects.application.step_preparation import NO_TEXT_BUCKET
from effects.application.training_log import (
    format_floor,
    format_norms,
    format_parts,
    format_shares,
)
from effects.domain.rarity import RARITY_BUCKETS


class TestProgressFormatting:
    """The within-shard progress line's two optional sections."""

    def test_gradient_norms_render_per_group(self):
        line = format_norms({"encoder": 0.8412, "head": 1.9733})
        assert "encoder 0.84" in line
        assert "head 1.97" in line

    def test_no_gradient_section_before_the_first_clip(self):
        assert format_norms({}) == ""

    def test_field_terms_are_ordered_largest_first(self):
        """A blow-up is one field carrying the loss; name order buries it."""
        line = format_parts({"gate": 0.41, "zone": 0.63, "pt": 0.28})
        assert line.index("zone") < line.index("gate") < line.index("pt")

    def test_no_field_section_when_the_batch_reported_none(self):
        assert format_parts({}) == ""


class TestExplainedDeviance:
    """The card-disjoint breakdown says how far each head beat a constant.

    A field's loss is a number in nats with no scale of its own, so the epoch
    line carries `1 - loss/floor` beside it: zero is the base rate, negative is
    worse than predicting it (FR-127c).
    """

    def test_each_field_carries_its_explained_deviance(self):
        line = format_parts(
            {"gate": 2.878, "damage_taken": 0.761},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line == "\n  fields: gate 2.878 (7%), damage_taken 0.761 (60%)"

    def test_the_largest_term_still_comes_first(self):
        line = format_parts(
            {"damage_taken": 0.761, "gate": 2.878},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line.index("gate") < line.index("damage_taken")

    def test_a_head_worse_than_the_constant_reads_negative(self):
        line = format_parts({"gate": 4.0}, floor={"gate": 2.0})
        assert "gate 4.000 (-100%)" in line

    def test_a_field_with_no_deviance_to_explain_reads_not_applicable(self):
        """A field whose targets never vary has nothing for a head to beat."""
        line = format_parts({"types_gained": 0.0}, floor={"types_gained": 0.0})
        assert "types_gained 0.000 (n/a)" in line

    def test_the_per_shard_training_breakdown_is_unchanged(self):
        assert format_parts({"gate": 2.0}) == "\n  fields: gate 2.000"

    def test_the_floor_is_logged_largest_first(self):
        assert format_floor(
            {"damage_taken": 1.902, "gate": 3.105},
        ) == "gate 3.105, damage_taken 1.902"


class TestEpochShares:
    """FR-055: the share of trained records per rarity bucket and family."""

    def test_shares_are_reported_in_bucket_order(self):
        line = format_shares(
            {"20+": 1, "1": 3, NO_TEXT_BUCKET: 4}, (*RARITY_BUCKETS, NO_TEXT_BUCKET),
        )
        assert line == "1 37.5%, 20+ 12.5%, no-text 50.0%"
