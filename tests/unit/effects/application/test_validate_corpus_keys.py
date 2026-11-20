"""validate-corpus's provenance keys: an empty ability says why, keyword keys
join their sidecar, other trait kinds are watched, and a copy keys its lines.
"""

from __future__ import annotations

from dataclasses import replace

from effects.application.validate_corpus import validate_corpus
from effects.domain.provenance import ProvenanceKey
from tests.unit.effects.application.validate_corpus_support import (
    _BEARS,
    _healthy,
    _named,
    _resolution,
    _snapshot,
    _trigger,
    _write_bears_sidecar,
)
from tests.unit.effects.builders import make_entity


class TestAnEmptyAbilitySaysWhy:
    """`ability: []` alone conflates a missing line with a broken resolver."""

    def test_an_empty_ability_with_no_reason_fails(self):
        records = _healthy() + [
            _resolution(f"run.0-L1.{index}", ability=())
            for index in range(400, 405)
        ]
        finding = _named(validate_corpus(records), "empty ability says why")
        assert not finding.ok
        assert "5 of 5" in finding.measured
        assert any("run.0-L1.400" in line for line in finding.detail)

    def test_an_empty_ability_that_names_its_reason_passes(self):
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(),
                ability_unresolved="engine_effect",
            )
            for index in range(400, 405)
        ]
        finding = _named(validate_corpus(records), "empty ability says why")
        assert finding.ok, finding.measured
        assert "0 of 5" in finding.measured

    def test_a_null_ability_is_the_other_defect_and_not_this_one(self):
        """A kind that never calls .ability() is the acting-line check's job."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", ability=None)
            for index in range(300, 305)
        ]
        assert _named(validate_corpus(records), "empty ability says why").ok
        assert not _named(validate_corpus(records), "trigger and rewrite").ok


class TestKeywordKeysJoinTheirSidecar:
    """The launch blocker: two sides numbering one card's keywords differently.

    The converter and the collector disagreed on a keyword's ordinal, so every
    record still joined — to the wrong printed line. Nothing downstream could
    notice, because a wrong row and a right one are the same shape.
    """

    def _record(self, index: int, record_id: str = "run.0-L1.900"):
        return _trigger(
            record_id, ability=(ProvenanceKey(_BEARS, 0, "keyword", index),),
        )

    def test_a_planted_ordinal_mismatch_is_rejected(self, tmp_path):
        # The sidecar declares keyword ordinals 0 and 2; the collector's key
        # says 1, which is inside what the face declared and names no line.
        sidecars = _write_bears_sidecar(tmp_path, (0, 2))
        findings = validate_corpus([*_healthy(), self._record(1)], sidecars=sidecars)
        finding = _named(findings, "keyword provenance keys")
        assert not finding.ok, finding.measured
        assert "1/1 keyword keys fail to join" in finding.measured
        assert any("keyword[1]" in line for line in finding.detail)

    def test_a_key_the_sidecar_declares_joins(self, tmp_path):
        sidecars = _write_bears_sidecar(tmp_path, (0, 2))
        findings = validate_corpus([*_healthy(), self._record(2)], sidecars=sidecars)
        finding = _named(findings, "keyword provenance keys")
        assert finding.ok, finding.detail
        assert "0/1 keyword keys fail to join" in finding.measured

    def test_a_key_outside_every_declared_range_is_counted_not_joined(
        self, tmp_path,
    ):
        """It resolves to no line, so reporting it as a join hides it."""
        sidecars = _write_bears_sidecar(tmp_path, (0, 1))
        findings = validate_corpus([*_healthy(), self._record(7)], sidecars=sidecars)
        keyword = _named(findings, "keyword provenance keys")
        outside = _named(findings, "outside every declared range")
        assert keyword.ok, "a runtime-only trait past the declared tail is no mismatch"
        assert "1 outside every declared range" in keyword.measured
        assert outside.watched and outside.ok
        assert outside.measured.startswith("1/")
        assert any("keyword: 1/1" in line for line in outside.detail)

    def test_a_face_declaring_no_keyword_line_leaves_every_keyword_key_outside(
        self, tmp_path,
    ):
        """The launch blocker in the disguise a join check reads as healthy.

        With no keyword line declared for the face, every keyword ordinal is
        outside the declared range, so nothing is a mismatch and nothing joins
        either. The run stays green and the count says how much of it did.
        """
        sidecars = _write_bears_sidecar(tmp_path, (), spell_indices=(0,))
        findings = validate_corpus([*_healthy(), self._record(0)], sidecars=sidecars)
        keyword = _named(findings, "keyword provenance keys")
        assert keyword.ok, keyword.measured
        assert "0/1 keyword keys fail to join" in keyword.measured
        assert "1 outside every declared range" in keyword.measured
        assert _named(findings, "outside every declared range").measured.startswith("1/")

    def test_a_dropped_key_joins_and_is_not_counted_as_outside(self, tmp_path):
        """A contentless trait the converter listed is described, so it joins.

        The sidecar answers for it, which is the whole question this check
        asks; that the answer is "no line" is FR-148's business, not this
        finding's.
        """
        dropped = ProvenanceKey(_BEARS, 0, "keyword", 1)
        sidecars = _write_bears_sidecar(tmp_path, (0, 2), dropped=(dropped,))
        findings = validate_corpus([*_healthy(), self._record(1)], sidecars=sidecars)
        keyword = _named(findings, "keyword provenance keys")
        assert keyword.ok, keyword.measured
        assert "0/1 keyword keys fail to join" in keyword.measured
        assert "0 outside every declared range" in keyword.measured
        assert _named(findings, "outside every declared range").measured.startswith("0/")

    def test_an_unconverted_script_is_not_read_as_outside_the_range(
        self, tmp_path,
    ):
        """No sidecar at all is a different answer from a sidecar that has
        nothing for the key, and only the first is an operator's problem."""
        sidecars = _write_bears_sidecar(tmp_path, (0,))
        orphan = _trigger(
            "run.0-L1.901",
            ability=(ProvenanceKey("cardsfolder/x/nowhere.txt", 0, "keyword", 0),),
        )
        findings = validate_corpus([*_healthy(), orphan], sidecars=sidecars)
        keyword = _named(findings, "keyword provenance keys")
        assert not keyword.ok, "a card with no sidecar fails rather than being skipped"
        assert any("no sidecar" in line for line in keyword.detail)
        assert _named(findings, "outside every declared range").measured.startswith("0/")

    def test_a_key_in_a_tree_this_run_was_not_given_fails_as_unchecked(
        self, tmp_path,
    ):
        """V5 #20: a stage-four corpus names variant-scripts keys, which a run
        without that tree cannot join; they are no mismatch, but a run that
        checked none of them is no pass either: ``--variant-scripts`` names
        the tree."""
        sidecars = _write_bears_sidecar(tmp_path, (0,))
        variant = _trigger(
            "run.0-L1.901",
            ability=(
                ProvenanceKey("variant-scripts/bears_x1.txt", 0, "keyword", 0),
            ),
        )
        finding = _named(
            validate_corpus([*_healthy(), variant], sidecars=sidecars),
            "keyword provenance keys",
        )
        assert not finding.ok
        assert "0/" in finding.measured and "1 unchecked" in finding.measured
        assert any("variant-scripts" in line and "--variant-scripts" in line
                   for line in finding.detail)


class TestNonKeywordKeysAreWatchedNotJudged:
    """Every trait kind is joined; only the keyword kind carries the verdict.

    A spell or static mismatch has a cause a keyword mismatch does not — a
    reconversion between collection and training, or a line the converter never
    emits at all — and nobody has measured what its healthy value is. Reported
    as its own number rather than folded into the keyword line's tail, because
    a number in a passing check's detail is a number nobody reads.
    """

    def _spell_record(self, index: int):
        return _trigger(
            "run.0-L1.900", ability=(ProvenanceKey(_BEARS, 0, "spell", index),),
        )

    def test_a_spell_mismatch_is_reported_without_failing_the_run(self, tmp_path):
        # Spells 0 and 2 declared, so index 1 is a gap inside the range — the
        # one shape that is a mismatch rather than a runtime-attached trait.
        sidecars = _write_bears_sidecar(tmp_path, (0,), spell_indices=(0, 2))
        findings = validate_corpus(
            [*_healthy(), self._spell_record(1)], sidecars=sidecars,
        )
        finding = _named(findings, "non-keyword provenance keys")
        assert finding.watched and finding.ok
        assert "keys of every other trait kind fail to join" in finding.measured
        # Both spell keys in the window: the planted mismatch, and the healthy
        # window's own line, whose card has no sidecar in this tree.
        assert "spell: 2/2 unjoinable" in finding.detail
        assert _named(findings, "keyword provenance keys").ok

    def test_the_keyword_line_reports_only_keyword_keys(self):
        """It reported both tallies in one sentence, and the one that could
        fail the run was the one an operator stopped reading."""
        finding = _named(validate_corpus(_healthy()), "keyword provenance keys")
        assert "over all" not in finding.measured

    def test_both_findings_say_so_when_no_tree_was_readable(self):
        findings = validate_corpus(_healthy(), sidecars=None)
        for fragment in ("keyword provenance keys", "non-keyword provenance keys"):
            finding = _named(findings, fragment)
            assert finding.watched and finding.ok
            assert "none checked" in finding.measured


class TestACopyKeysItsLinesIntoTheCopiedCard:
    """A Clone copying a creature carries the creature's printed lines, so its
    keyword lines key into the copied card's file (copiable values, 707.2);
    one keyed into the copy's own file is the Clone-key gap (watched)."""

    @staticmethod
    def _board(copy_key_file: str):
        from effects.domain.state_snapshot import CardLine

        bears = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "keyword", 0)
        copy_key = ProvenanceKey(copy_key_file, 0, "keyword", 0)
        original = make_entity("E1", "serra angel", lines=(
            CardLine(keys=(bears,), kind="keyword", granted=False),))
        clone = make_entity("E2", "serra angel", copy_source="E1", lines=(
            CardLine(keys=(copy_key,), kind="keyword", granted=False),))
        state = replace(_snapshot(), entities=(original, clone))
        return _resolution("run.0-L1.960", state=state)

    def test_a_copy_keying_into_the_copied_card_holds(self):
        record = self._board("cardsfolder/s/serra_angel.txt")
        finding = _named(validate_corpus([*_healthy(), record]), "copied card")
        assert finding.watched and "0/1" in finding.measured

    def test_a_copy_keying_into_its_own_card_is_named(self):
        record = self._board("cardsfolder/c/clone.txt")
        finding = _named(validate_corpus([*_healthy(), record]), "copied card")
        assert "1/1" in finding.measured and any("E2" in line for line in finding.detail)
