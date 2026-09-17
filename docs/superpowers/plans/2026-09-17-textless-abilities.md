# Textless Abilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the effect model from reading phantom zero-vector ability tokens, stop the corpus builder from keeping resolution records whose acting ability has no text, and make the converter claim every key whose text it renders, so the join loses no real ability.

**Architecture:** Three Python changes (surface builder skips no-line keys; sidecar mismatches raise; `build-corpus` refuses no-acting-text records and counts them per script) and two Java changes in the converter's `RulesParser` (synthetic land mana claims its keys; deduplication, secondary triggers, keyword-derived traits and Class level lines keep their attribution). An audit script measures the corpus before and after. The rendered card text never changes.

**Tech Stack:** Python 3.14, PyTorch (CPU for tests), pytest, ruff; Java 17, Maven, JUnit 5, sibling Forge checkout at `../forge`.

**Spec:** `docs/superpowers/specs/2026-09-17-textless-abilities.md` (design record); root spec `specs/023-ability-effect-model/spec.md` (FR-073, FR-148, new FR-152); contract `specs/023-ability-effect-model/contracts/provenance-sidecar.md`.

## Global Constraints

- Work in the main checkout `C:\Users\nicol\IdeaProjects\price-predictor` on branch `effects-coverage-deck-play`. Do not prefix commands with `cd` to the project directory.
- Every pytest run is CPU-pinned: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects -q`. Never run the full suite, never touch `models/` or `output/`.
- Lint: `CUDA_VISIBLE_DEVICES="" python -m ruff check src/effects tests/unit/effects` must print `All checks passed!`.
- Java tests: `mvn -q -f forge-connector/pom.xml test -Dtest=ProvenanceSidecarTest -DfailIfNoTests=false` (about two minutes; needs `../forge` built). Exit code 0 is the pass signal; `[malformed_card] null` warnings are normal.
- Before editing anything under `specs/`, invoke the `feature-workflow` skill. Specs are timeless present tense, WHAT not WHY, no benchmark numbers. The design record under `docs/superpowers/specs/` is not under `specs/` and needs no skill.
- Nothing is pre-existing: a failing test after your change is yours to fix.
- Follow neighbouring code's docstring style (explain why, not what). TDD: failing test first.
- One commit per task, conventional-commit subject, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Do not push.
- Python tasks 1 → 2 → 3 are sequential (they share `surface_batching.py` and `sidecar_io.py`). Java tasks 4 → 5 are sequential (they share `RulesParser.java`). The two lanes are independent of each other. Task 6 depends on nothing.

---

### Task 1: The surface builder skips a key that resolves to no line

**Files:**
- Modify: `src/effects/domain/effect_head_input.py` (signature of `build_effect_head_input` around line 352, docstring, the entity ability loop around line 432–447)
- Modify: `src/effects/application/surface_batching.py` (`surface_for`, around line 220–238)
- Test: `tests/unit/effects/domain/test_effect_head_input.py`
- Create test: `tests/unit/effects/application/test_surface_batching.py`
- Modify: `specs/023-ability-effect-model/spec.md` (FR-073, line 510) — feature-workflow skill first

**Interfaces:**
- Produces: `build_effect_head_input(record, *, e_for, e_dim, has_line=None, candidate_index=None, subtype_tokens_for=None, context_dropout=0.0, rng=None, masked_keywords=None)`. `has_line: Callable[[ProvenanceKey], bool] | None`. `None` keeps every key (today's behaviour); otherwise an entity key for which `has_line(key)` is `False` gets no `ABILITY` slot. The acting `[ACT]` slot is unchanged.
- Produces: `SurfaceBatcher.surface_for` passes `has_line=lambda key: self.text_of(key) is not None`.

- [ ] **Step 1: Write the failing domain tests**

Append to `tests/unit/effects/domain/test_effect_head_input.py`, inside `class TestSurfaceShape` (or as a new class at module level):

```python
class TestKeysWithNoLine:
    """A key the sidecar maps to no line is not an ability (FR-073).

    Forge attaches an implicit cast-this-permanent object to every permanent,
    and the converter renders no line for it. Without the predicate the
    surface carried a zero-vector token for it on almost every entity.
    """

    _PHANTOM = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0)

    def test_a_key_with_no_line_gets_no_ability_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM, _ANTHEM)),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        abilities = surface.of_kind(SlotKind.ABILITY)
        assert [slot.e for slot in abilities] == [_VECTORS[_ANTHEM]]

    def test_positions_stay_contiguous_after_a_skip(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM, _ANTHEM, _WARD)),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        positions = [
            slot.position for slot in surface.slots
            if slot.kind in (SlotKind.CARD, SlotKind.ABILITY)
        ]
        assert positions == [0, 1, 2]

    def test_a_masked_line_keeps_its_zero_token(self):
        """The state-only control zeroes ``e`` but keeps the geometry."""
        record = _record(state=_snapshot(entities=(_entity("E1", printed=(_ANTHEM,)),)))
        surface = build_effect_head_input(
            record, e_for=lambda key: None, e_dim=E_DIM, has_line=lambda key: True,
        )
        assert [slot.e for slot in surface.of_kind(SlotKind.ABILITY)] == [_ZERO]

    def test_without_the_predicate_every_key_keeps_its_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM,)),
        )))
        assert [slot.e for slot in _build(record).of_kind(SlotKind.ABILITY)] == [_ZERO]
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/domain/test_effect_head_input.py -q -k KeysWithNoLine`
Expected: 3 FAIL with `TypeError: build_effect_head_input() got an unexpected keyword argument 'has_line'`, 1 PASS (the no-predicate case).

- [ ] **Step 3: Implement the predicate in the builder**

In `src/effects/domain/effect_head_input.py`, add the parameter after `e_dim` and document it:

```python
def build_effect_head_input(
    record: EffectRecord,
    *,
    e_for,
    e_dim: int,
    has_line=None,
    candidate_index: int | None = None,
    subtype_tokens_for=None,
    context_dropout: float = 0.0,
    rng: random.Random | None = None,
    masked_keywords: dict[str, frozenset[str]] | None = None,
) -> EffectHeadInput:
```

Add to the docstring's `Args:` block, after `e_for`:

```
        has_line: ``ProvenanceKey -> bool``, whether the key maps to a rendered
            line at all. An entity key with no line contributes no token:
            Forge attaches an implicit cast-this-permanent object to every
            permanent, the converter renders no line for it, and a token for
            it is board content no rules text produced. Distinct from ``e_for``
            returning None, which keeps a zero token so a variant that masks
            ``e`` keeps the geometry. None keeps every key.
```

In the entity loop, before the dropout check:

```python
        for key in _ability_keys(entity):
            if has_line is not None and not has_line(key):
                continue
            if (
                is_context
                and context_dropout > 0.0
                and rng.random() < context_dropout
            ):
                continue
```

- [ ] **Step 4: Run the domain tests**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/domain/test_effect_head_input.py -q`
Expected: all PASS.

- [ ] **Step 5: Write the failing wiring test for the batcher**

Create `tests/unit/effects/application/test_surface_batching.py`:

```python
"""``SurfaceBatcher`` hands the builder the sidecar's answer to "is this a line".

The rule itself lives in the domain builder; this pins the wiring, because a
batcher that forgot to pass the predicate would silently keep every phantom
token and every number it reports would still move.
"""

from __future__ import annotations

import torch

from effects.application.surface_batching import SurfaceBatcher
from effects.application.train_effect_model import VariantMasks
from effects.domain.effect_head_input import SlotKind
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionPayload
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot

_ACT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_PHANTOM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "spell", 0)

_LINES = {
    _ACT: SidecarLine(line_index=0, line_kind="spell", provenance=(_ACT,),
                      script_text="deals 3 damage to any target"),
    _ANTHEM: SidecarLine(line_index=0, line_kind="static", provenance=(_ANTHEM,),
                         script_text="creatures you control get +1/+1"),
}


class _Sidecars:
    """The two answers a real cache gives: a line, or None for a dropped key."""

    def line_for(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        return None


def _record() -> EffectRecord:
    return EffectRecord(
        record_id="run.0.1", run_id="run", timestamp="t", game_id="run.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        ability=(_ACT,), payload=ResolutionPayload(),
        state=StateSnapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=1),
            players=(),
            entities=(EntityState(id="E1", name="Anthem", zone="battlefield",
                                  controller="P0", printed=(_PHANTOM, _ANTHEM)),),
        ),
    )


def _batcher(**masks) -> SurfaceBatcher:
    return SurfaceBatcher(
        tokenizer=None, sidecars=_Sidecars(), masks=VariantMasks(**masks),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )


def test_a_dropped_key_gets_no_ability_token():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher().surface_for(_record(), rows)
    assert [slot.e for slot in surface.of_kind(SlotKind.ABILITY)] == [0]


def test_the_state_only_variant_keeps_the_zero_token_for_a_real_line():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher(zero_e=True).surface_for(_record(), rows)
    abilities = surface.of_kind(SlotKind.ABILITY)
    assert len(abilities) == 1
    assert abilities[0].e == (0.0,) * 4
```

If `VariantMasks` has a required field, look at how `tests/unit/effects/application/test_training_loop_execute.py` constructs one and mirror it; the intent is "all masks off" for the first test and `zero_e=True` for the second.

- [ ] **Step 6: Run it to see the first test fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/application/test_surface_batching.py -q`
Expected: first test FAILS with `[(0.0, 0.0, 0.0, 0.0), 0] == [0]`; second PASSES.

- [ ] **Step 7: Wire the predicate in the batcher**

In `src/effects/application/surface_batching.py`, `surface_for`:

```python
    def surface_for(self, record, rows: dict[str, int]):
        """One record's token surface, with the variant's masks applied."""

        def e_for(key):
            if self.masks.zero_e:
                return None
            text = self.text_of(key)
            return None if text is None else rows.get(text)

        def has_line(key) -> bool:
            # Asked before the mask, so a masked line keeps its zero token
            # and a dropped key gets none: the two are different facts.
            return self.text_of(key) is not None

        return build_effect_head_input(
            record,
            e_for=e_for,
            has_line=has_line,
            e_dim=self.e_dim,
            context_dropout=self.context_dropout,
            rng=self.rng,
            masked_keywords=continuous_masked_keywords(record),
        )
```

- [ ] **Step 8: Run the effects suite and ruff**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects -q` then `CUDA_VISIBLE_DEVICES="" python -m ruff check src/effects tests/unit/effects`
Expected: all pass, `All checks passed!`.

- [ ] **Step 9: Amend FR-073**

Invoke the `feature-workflow` skill. Then in `specs/023-ability-effect-model/spec.md` replace the FR-073 bullet with:

```
- **FR-073**: An entity's ability tokens MUST be its printed and attachment-granted lines only;
  temporary grants ride the overlay. A key that resolves to no rendered line MUST contribute no
  token: Forge's implicit permanent-spell object is such a key on every permanent, and a token for
  it is board content no rules text produced. A line a variant masks out MUST keep its token, so
  the variant keeps the geometry it is compared on.
```

- [ ] **Step 10: Commit**

```bash
git add src/effects/domain/effect_head_input.py src/effects/application/surface_batching.py tests/unit/effects/domain/test_effect_head_input.py tests/unit/effects/application/test_surface_batching.py specs/023-ability-effect-model/spec.md
git commit -m "feat(effects): an entity key with no rendered line contributes no ability token (FR-073)"
```

---

### Task 2: A sidecar mismatch fails loudly at every lookup

**Files:**
- Modify: `src/effects/infrastructure/sidecar_io.py` (`UnconvertedScript` around line 188, `SidecarCache.path_for` around line 222)
- Modify: `src/effects/application/surface_batching.py` (`text_of` line ~97–100, `batch_texts.note` line ~121–126)
- Modify: `src/effects/application/train_effect_model.py` (`text_for_key` line ~1012–1015)
- Test: `tests/unit/effects/application/test_ability_text.py`, `tests/unit/effects/application/test_surface_batching.py` (from Task 1)
- Modify: `CLAUDE.md` (root) — no wording change needed if the sidecar paragraph already says a key in neither list fails loudly; verify and leave it.

**Interfaces:**
- Produces: `class UnconfiguredTree(KeyError)` in `effects.infrastructure.sidecar_io`, raised by `SidecarCache.path_for` for a tree with no configured root. `SidecarCache.line_for` still returns `None` for an `UnconvertedScript` and lets both `UnconfiguredTree` and the mismatch `KeyError` propagate.
- Produces: `text_for_key(key, sidecars, surface)` returns `None` on `UnconfiguredTree`, raises on a mismatch. `SurfaceBatcher.text_of` and `batch_texts` the same.
- Consumed by Task 3 (`UnconfiguredTree`).

- [ ] **Step 1: Write the failing tests for `text_for_key`**

Append to `tests/unit/effects/application/test_ability_text.py` (reuse its `_record` helper and imports; add `import pytest` and `from effects.infrastructure.sidecar_io import SidecarCache, UnconfiguredTree, UnconvertedScript, write_sidecar` plus `from effects.domain.provenance import ProvenanceSidecar` if not already imported):

```python
def test_a_key_in_neither_list_raises_rather_than_reading_as_no_text(tmp_path):
    """The contract's fail-loudly case: the sidecar does not describe the card."""
    from effects.application.train_effect_model import text_for_key
    from effects.infrastructure.sidecar_io import sidecar_path_for

    script = "cardsfolder/p/plague_sliver.txt"
    txt = tmp_path / "cardsfolder" / "p" / "plague_sliver.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("name: plague sliver\n", encoding="utf-8")
    rendered = ProvenanceKey(script, 0, "static", 0)
    write_sidecar(
        ProvenanceSidecar(
            card="plague sliver", script_file=script,
            lines=(SidecarLine(line_index=0, line_kind="static",
                               provenance=(rendered,), script_text="all slivers have"),),
            dropped_keys=(ProvenanceKey(script, 0, "spell", 0),),
        ),
        sidecar_path_for(txt),
    )
    sidecars = SidecarCache({"cardsfolder": tmp_path / "cardsfolder"})

    assert text_for_key(rendered, sidecars, "script") == "all slivers have"
    assert text_for_key(ProvenanceKey(script, 0, "spell", 0), sidecars, "script") is None
    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        text_for_key(ProvenanceKey(script, 0, "trigger", 0), sidecars, "script")


def test_an_unconfigured_tree_still_reads_as_no_text(tmp_path):
    """A variant key without ``--variant-scripts`` resolves to nothing, as pinned
    by ``test_without_the_variant_tree_a_variant_text_resolves_to_nothing``."""
    from effects.application.train_effect_model import text_for_key

    sidecars = SidecarCache({"cardsfolder": tmp_path})
    key = ProvenanceKey("variant-scripts/x.txt", 0, "spell", 0)
    with pytest.raises(UnconfiguredTree):
        sidecars.path_for(key.script_file)
    assert text_for_key(key, sidecars, "script") is None


def test_an_unconverted_script_reads_as_no_text(tmp_path):
    from effects.application.train_effect_model import text_for_key

    sidecars = SidecarCache({"cardsfolder": tmp_path})
    key = ProvenanceKey("cardsfolder/f/food_token.txt", 0, "spell", 0)
    with pytest.raises(UnconvertedScript):
        sidecars.get(key.script_file)
    assert text_for_key(key, sidecars, "script") is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/application/test_ability_text.py -q`
Expected: `ImportError: cannot import name 'UnconfiguredTree'` (collection error).

- [ ] **Step 3: Add `UnconfiguredTree` and narrow the three catches**

In `src/effects/infrastructure/sidecar_io.py`, next to `UnconvertedScript`:

```python
class UnconfiguredTree(KeyError):
    """A key names a source tree this run configured no converted root for.

    Distinct from :class:`UnconvertedScript` (the tree exists, the card was
    never converted) and from the mismatch ``KeyError`` (the sidecar exists and
    disagrees with the record). A variant key on a run without
    ``--variant-scripts`` is the expected case, and it reads as no text.
    """
```

In `SidecarCache.path_for`, raise it in place of the bare `KeyError`:

```python
        if tree not in self._roots:
            raise UnconfiguredTree(
                f"no converted root configured for source tree {tree!r}; "
                f"known trees: {sorted(self._roots)}"
            )
```

In `src/effects/application/train_effect_model.py`, `text_for_key`:

```python
    from effects.domain.ability_encoder import encoding_text
    from effects.infrastructure.sidecar_io import UnconfiguredTree

    try:
        line = sidecars.line_for(key)
    except UnconfiguredTree:
        return None
    if line is None:
        return None
```

(`line_for` already turns `UnconvertedScript` into `None`; the mismatch `KeyError` now propagates.)

In `src/effects/application/surface_batching.py`, import `UnconfiguredTree` from `effects.infrastructure.sidecar_io` and change both `except KeyError:` (in `text_of` and in `batch_texts.note`) to `except UnconfiguredTree:`. Update `text_of`'s docstring: add one sentence, "A key the sidecar does not describe raises: that is the contract's fail-loudly case, and reading it as no text is what hid it."

- [ ] **Step 4: Add the batcher test**

Append to `tests/unit/effects/application/test_surface_batching.py`:

```python
def test_a_sidecar_mismatch_raises_out_of_the_batcher():
    class _Mismatching(_Sidecars):
        def line_for(self, key):
            raise KeyError(f"provenance key {key} appears in neither the lines nor the dropped_keys")

    batcher = _batcher()
    batcher.sidecars = _Mismatching()
    with pytest.raises(KeyError, match="neither the lines"):
        batcher.batch_texts([_record()])
```

(add `import pytest` at the top of that file.)

- [ ] **Step 5: Run the effects suite and ruff**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects -q` then ruff.
Expected: all pass. If a test elsewhere relied on the old swallow, it was asserting the bug; fix the test to expect the raise, not the code.

- [ ] **Step 6: Commit**

```bash
git add src/effects/infrastructure/sidecar_io.py src/effects/application/surface_batching.py src/effects/application/train_effect_model.py tests/unit/effects/application/test_ability_text.py tests/unit/effects/application/test_surface_batching.py
git commit -m "fix(effects): a sidecar mismatch raises at every key lookup instead of reading as no text"
```

---

### Task 3: `build-corpus` refuses a resolution record whose acting keys map to no line

**Files:**
- Modify: `src/effects/domain/provenance.py` (`ProvenanceSidecar`, add `KeyResolution` enum and `resolution_of`)
- Modify: `src/effects/infrastructure/sidecar_io.py` (`SidecarCache.resolution_of`)
- Modify: `src/effects/domain/record_quality.py` (`NO_ACTING_TEXT`, `acting_text_defect`)
- Modify: `src/effects/application/build_corpus.py` (`SurveyConfig` line ~132, `ShardSurvey` line ~152, `Survey` merge line ~301–342, `survey_shard` line ~251, `WriteConfig` line ~789, `WriteResult` line ~816, `init_write_worker` line ~859, `write_shard_pass` line ~920, total merge line ~1008, config construction lines ~1297 and ~1342, manifest line ~1459, logging line ~1498)
- Modify: `src/effects/domain/corpus_manifest.py` (new field + `from_dict`)
- Test: `tests/unit/effects/domain/test_record_quality.py`, `tests/unit/effects/domain/test_provenance.py` (create if absent), `tests/unit/effects/application/test_build_corpus.py`
- Modify: `specs/023-ability-effect-model/spec.md` (FR-148), `specs/023-ability-effect-model/contracts/provenance-sidecar.md` (rules table) — feature-workflow skill first

**Interfaces:**
- Consumes: `UnconfiguredTree`, `UnconvertedScript` from Task 2.
- Produces: `KeyResolution` (`StrEnum`: `LINE`, `DROPPED`, `RUNTIME_ONLY`, `UNCONVERTED`) in `effects.domain.provenance`; `ProvenanceSidecar.resolution_of(key) -> KeyResolution` (never `UNCONVERTED`; raises the mismatch `KeyError`); `SidecarCache.resolution_of(key) -> KeyResolution` (returns `UNCONVERTED` on `UnconvertedScript` or `UnconfiguredTree`).
- Produces: `NO_ACTING_TEXT = "no-acting-text"` and `acting_text_defect(record, resolve) -> str | None` in `effects.domain.record_quality`, where `resolve: Callable[[ProvenanceKey], KeyResolution]`.
- Produces: manifest field `no_acting_text_scripts: dict[str, int]`.

- [ ] **Step 1: Write the failing domain tests**

Create `tests/unit/effects/domain/test_provenance_resolution.py`:

```python
"""``resolution_of``: the four answers a key can get at the join."""

from __future__ import annotations

import pytest

from effects.domain.provenance import (
    KeyResolution,
    ProvenanceKey,
    ProvenanceSidecar,
    SidecarLine,
)

_SCRIPT = "cardsfolder/m/mogg_war_marshal.txt"
_LINE = ProvenanceKey(_SCRIPT, 0, "trigger", 0)
_DROPPED = ProvenanceKey(_SCRIPT, 0, "spell", 0)


def _sidecar() -> ProvenanceSidecar:
    return ProvenanceSidecar(
        card="mogg war marshal", script_file=_SCRIPT,
        lines=(SidecarLine(line_index=0, line_kind="triggered", provenance=(_LINE,),
                           script_text="when this enters or dies"),),
        dropped_keys=(_DROPPED,),
    )


def test_a_rendered_key_is_a_line():
    assert _sidecar().resolution_of(_LINE) is KeyResolution.LINE


def test_a_dropped_key_is_dropped():
    assert _sidecar().resolution_of(_DROPPED) is KeyResolution.DROPPED


def test_a_key_past_the_declared_indices_is_runtime_only():
    assert _sidecar().resolution_of(ProvenanceKey(_SCRIPT, 0, "spell", 7)) is KeyResolution.RUNTIME_ONLY


def test_a_key_in_neither_list_raises():
    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        _sidecar().resolution_of(ProvenanceKey(_SCRIPT, 0, "trigger", 0).__class__(_SCRIPT, 0, "static", 0))
```

(The last line builds `ProvenanceKey(_SCRIPT, 0, "static", 0)`; write it that way directly. Check with `is_runtime_only` how "past declared" is computed: `spell 7` is past `spell 0`, the only spell index declared, so it is runtime-only; a `static 0` has no declared static at all, and `is_runtime_only` returns `False` for a kind with nothing declared, so it is the mismatch. If the existing rule differs, adjust the two keys so one is past a declared index and one is an undeclared kind.)

Append to `tests/unit/effects/domain/test_record_quality.py`:

```python
from effects.domain.provenance import KeyResolution, ProvenanceKey
from effects.domain.record_quality import NO_ACTING_TEXT, acting_text_defect


def _resolver(**by_kind):
    """``trait_kind -> KeyResolution`` stands in for the sidecar cache."""
    def resolve(key: ProvenanceKey) -> KeyResolution:
        return by_kind[key.trait_kind]
    return resolve


def test_a_record_acting_through_a_rendered_line_is_kept(make_record):
    assert acting_text_defect(make_record(), _resolver(spell=KeyResolution.LINE)) is None


def test_a_record_acting_only_through_dropped_keys_is_refused(make_record):
    assert acting_text_defect(make_record(), _resolver(spell=KeyResolution.DROPPED)) == NO_ACTING_TEXT


def test_an_unconverted_script_counts_as_no_text(make_record):
    assert acting_text_defect(make_record(), _resolver(spell=KeyResolution.UNCONVERTED)) == NO_ACTING_TEXT


def test_a_runtime_only_key_keeps_the_record(make_record):
    """Level up, bestow and scavenge add a spell the script never declared; its
    text sits on a keyword line the join does not reach yet, so the record stays."""
    assert acting_text_defect(make_record(), _resolver(spell=KeyResolution.RUNTIME_ONLY)) is None


def test_one_rendered_key_among_dropped_ones_keeps_the_record(make_record, ability_key):
    keys = (ability_key, ProvenanceKey(ability_key.script_file, 0, "trigger", 0))
    resolve = _resolver(spell=KeyResolution.DROPPED, trigger=KeyResolution.LINE)
    assert acting_text_defect(make_record(ability=keys), resolve) is None


def test_only_resolution_records_are_subject_to_the_rule(make_record):
    record = make_record(kind=RecordKind.COMBAT, payload=CombatPayload())
    assert acting_text_defect(record, _resolver()) is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/domain/test_provenance_resolution.py tests/unit/effects/domain/test_record_quality.py -q`
Expected: ImportError on `KeyResolution` / `acting_text_defect`.

- [ ] **Step 3: Implement the domain side**

In `src/effects/domain/provenance.py` (module already imports `StrEnum`? if not, `from enum import StrEnum`):

```python
class KeyResolution(StrEnum):
    """What a provenance key resolves to at the join.

    ``LINE`` is text. ``DROPPED`` is a trait the converter rendered no line for
    — Forge's implicit permanent spell on every permanent, most often — and is
    no text by design. ``RUNTIME_ONLY`` is a trait Forge adds only to a live
    card (level up, bestow, scavenge), whose text sits on a keyword line the
    join does not reach; it is kept apart from ``DROPPED`` because refusing on
    it would refuse real abilities. ``UNCONVERTED`` is a script no sidecar
    describes. The fifth outcome, a key in neither list, is not a value: it
    raises, because the sidecar does not describe the record's card.
    """

    LINE = "line"
    DROPPED = "dropped"
    RUNTIME_ONLY = "runtime-only"
    UNCONVERTED = "unconverted"
```

Add to `ProvenanceSidecar`, next to `row_for`:

```python
    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """Which of the join's answers this key gets; raises on a mismatch."""
        if key in self._row_by_key:
            return KeyResolution.LINE
        if key in self.dropped_keys:
            return KeyResolution.DROPPED
        if self.is_runtime_only(key):
            return KeyResolution.RUNTIME_ONLY
        raise KeyError(self._mismatch_message(key))
```

In `src/effects/infrastructure/sidecar_io.py`, `SidecarCache`:

```python
    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """The join's answer for a key, with the two no-sidecar cases folded in.

        An unconverted script and an unconfigured tree are both "no text"; the
        mismatch inside an existing sidecar still raises.
        """
        try:
            return self.get(key.script_file).resolution_of(key)
        except (UnconvertedScript, UnconfiguredTree):
            return KeyResolution.UNCONVERTED
```

In `src/effects/domain/record_quality.py`:

```python
NO_ACTING_TEXT = "no-acting-text"


def acting_text_defect(record: EffectRecord, resolve) -> str | None:
    """``NO_ACTING_TEXT`` when every acting key maps to no rendered line.

    A resolution record whose acting key is Forge's implicit permanent spell
    passes ``quality_defect`` — it has a key — and reaches the head with an
    empty acting slot. Three quarters of the resolution class was that record
    before this rule, weighted like the rarest text in the corpus.

    A ``RUNTIME_ONLY`` key keeps the record: level up, bestow and scavenge are
    real abilities whose text the join does not reach yet, and refusing them
    would be a loss rather than a cleanup. A key in neither of the sidecar's
    lists raises out of ``resolve``, as the contract requires.

    Args:
        resolve: ``ProvenanceKey -> KeyResolution``, the sidecar cache's answer.
    """
    if record.kind is not RecordKind.RESOLUTION or not record.ability:
        return None
    outcomes = {resolve(key) for key in record.ability}
    if KeyResolution.LINE in outcomes or KeyResolution.RUNTIME_ONLY in outcomes:
        return None
    return NO_ACTING_TEXT
```

Import `KeyResolution` from `effects.domain.provenance` at the top of `record_quality.py`. Update the module docstring's first line to "Which records a curated corpus refuses on sight (FR-148): three rules."

- [ ] **Step 4: Run the domain tests**

Run the two test files. Expected: all PASS.

- [ ] **Step 5: Write the failing build test**

In `tests/unit/effects/application/test_build_corpus.py`, extend `_write_card` with a `dropped` parameter and write it into the sidecar:

```python
def _write_card(
    root: Path, script_file: str, card: str, text: str, *,
    index_within_kind: int = 0, dropped: tuple[ProvenanceKey, ...] = (),
) -> None:
    ...
    write_sidecar(
        ProvenanceSidecar(
            card=card,
            script_file=script_file,
            lines=(
                SidecarLine(
                    line_index=999, line_kind="spell", provenance=(key,),
                    script_text=text,
                ),
            ),
            dropped_keys=dropped,
        ),
        sidecar_path_for(txt_path),
    )
```

Add module constants next to `_BOLT_KEY`:

```python
_BEAR_FILE = "cardsfolder/g/grizzly_bears.txt"
_BEAR_CARD = "Grizzly Bears"
_BEAR_TEXT = "grizzly bears static text"
_BEAR_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 1)
_BEAR_CAST_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 0)
_BEAR_LEVEL_UP_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 5)
```

In `a_corpus`, after the Food card line add `_write_card(root, _BEAR_FILE, _BEAR_CARD, _BEAR_TEXT, index_within_kind=1, dropped=(_BEAR_CAST_KEY,))`, and to `shard_b` add:

```python
        # The creature spell itself resolving: its key is the implicit permanent
        # spell, which the sidecar lists as dropped. Refused (FR-148).
        _resolution("g-clean-3.cast", "g-clean-3", ability=(_BEAR_CAST_KEY,)),
        # A runtime-only key (past every spell index the face declared): kept.
        _resolution("g-clean-3.level", "g-clean-3", ability=(_BEAR_LEVEL_UP_KEY,)),
```

Then add the test:

```python
def test_a_resolution_record_acting_only_through_a_dropped_key_is_refused(tmp_path, a_corpus):
    out = tmp_path / "curated"
    assert build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards,
        forge_tokenscripts=a_corpus.forge_tokenscripts, output=out,
        workers=1, game_disjoint_target=1,
    )) == 0
    manifest = CorpusStore(out).load()
    assert manifest.quality_dropped.get("no-acting-text") == 1
    assert manifest.no_acting_text_scripts == {_BEAR_FILE: 1}
    written = {r.record_id for r in read_records(CorpusStore(out).training_dir)}
    written |= {r.record_id for r in read_records(CorpusStore(out).card_disjoint_dir)}
    written |= {r.record_id for r in read_records(CorpusStore(out).game_disjoint_dir)}
    assert "g-clean-3.cast" not in written
    assert "g-clean-3.level" in written
```

Existing tests that count records or strata may need their expected numbers adjusted by the two added records (one kept, one refused); adjust the expectations, not the fixture, and say so in the commit message.

- [ ] **Step 6: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects/application/test_build_corpus.py -q -k dropped_key`
Expected: FAIL (`quality_dropped.get("no-acting-text")` is `None`).

- [ ] **Step 7: Thread the sidecars into both worker pools**

In `src/effects/application/build_corpus.py`:

1. `SurveyConfig` and `WriteConfig` each gain `sidecar_roots: dict[str, str] = field(default_factory=dict)` with the comment `#: Tree name -> converted root, so a worker can build its own SidecarCache: the refusal rule needs the sidecar's answer per key, and a cache does not pickle.`
2. A module-level lazily built cache shared by both worker kinds:

```python
_SIDECARS: SidecarCache | None = None


def _worker_sidecars(roots: dict[str, str]) -> SidecarCache:
    """One cache per worker process, built on first use from the config's roots."""
    global _SIDECARS
    if _SIDECARS is None:
        _SIDECARS = SidecarCache({name: Path(root) for name, root in roots.items()})
    return _SIDECARS
```

3. `ShardSurvey` and `WriteResult` gain `textless_scripts: Counter[str] = field(default_factory=Counter)`; `Survey` gains the same and the merge at line ~315 adds `textless_scripts.update(part.textless_scripts)`; the write total at line ~1008 adds `total.textless_scripts.update(part.textless_scripts)`.
4. In `survey_shard` and `write_shard_pass`, replace the refusal with:

```python
        defect = quality_defect(record, max_events=config.max_events)
        if defect is None:
            defect = acting_text_defect(
                record, _worker_sidecars(config.sidecar_roots).resolution_of,
            )
        if defect is not None:
            out.quality_dropped[defect] += 1
            if defect == NO_ACTING_TEXT:
                # Per script, so a converter regression that drops a whole
                # card family's keys shows up in the build output.
                out.textless_scripts[record.ability[0].script_file] += 1
            continue
```

(in `write_shard_pass` keep the existing `out.refused_by_class[name] += 1` line.)

5. In `build()`, pass `sidecar_roots={name: str(path) for name, path in roots.items()}` to both `SurveyConfig(...)` and `WriteConfig(...)`.
6. Manifest: `no_acting_text_scripts=dict(written.textless_scripts)` beside `quality_dropped=dict(written.quality_dropped)`. After the refusal loop at line ~1498:

```python
    if written.textless_scripts:
        worst = sorted(written.textless_scripts.items(), key=lambda kv: -kv[1])[:5]
        logger.info(
            "no-acting-text refusals by script, most first: %s",
            ", ".join(f"{script} ({count})" for script, count in worst),
        )
```

In `src/effects/domain/corpus_manifest.py`, add to `CorpusManifest` after `token_keys_ambiguous`:

```python
    #: Script file -> resolution records refused because every acting key
    #: mapped to no rendered line (FR-148). Read by an operator, not by code.
    no_acting_text_scripts: dict[str, int] = field(default_factory=dict)
```

and in `from_dict`: `no_acting_text_scripts={k: int(v) for k, v in data.get("no_acting_text_scripts", {}).items()},`.

- [ ] **Step 8: Run the effects suite and ruff**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit/effects -q` then ruff.
Expected: all pass. The survey/write agreement warning must not fire in the build tests (both passes apply the same rule).

- [ ] **Step 9: Amend FR-148 and the sidecar contract**

Invoke the `feature-workflow` skill. In `specs/023-ability-effect-model/spec.md`, change the opening of FR-148 from "exactly two reasons — a resolution record with no acting ability, and any record carrying more than `--max-events-per-record` events (default 64) —" to:

```
- **FR-148**: `build-corpus` MUST refuse a record for exactly three reasons — a resolution record
  with no acting ability; a resolution record whose every acting key maps to no rendered line (a key
  the sidecar lists as dropped, or one naming an unconverted script), a key past what the face
  declared keeping the record because its text sits on a keyword line the join does not reach; and
  any record carrying more than `--max-events-per-record` events (default 64) —
```

and after the sentence ending `as \`quality_dropped\`.` add: `It MUST record, as \`no_acting_text_scripts\`, the refused count per acting script file, and log the largest. A key in neither of the sidecar's lists MUST fail the build.`

In `specs/023-ability-effect-model/contracts/provenance-sidecar.md`, replace the second rule row with:

```
| A trait whose text a rendered line carries — merged, deduplicated, a secondary of a pair, keyword-derived, or a Class level — is claimed by that line | the join is many-to-one; `dropped_keys` holds only traits with no text, such as the implicit permanent spell on every permanent |
| A record whose acting keys all sit in `dropped_keys` (or name an unconverted script) | `build-corpus` refuses it as `no-acting-text`; on the board, the surface builder emits no token for such a key. A key past what the face declared is neither: it keeps the record and gets no token |
```

- [ ] **Step 10: Commit**

```bash
git add src/effects/domain/provenance.py src/effects/infrastructure/sidecar_io.py src/effects/domain/record_quality.py src/effects/application/build_corpus.py src/effects/domain/corpus_manifest.py tests/unit/effects/domain/test_provenance_resolution.py tests/unit/effects/domain/test_record_quality.py tests/unit/effects/application/test_build_corpus.py specs/023-ability-effect-model/spec.md specs/023-ability-effect-model/contracts/provenance-sidecar.md
git commit -m "feat(effects): build-corpus refuses a resolution record acting only through dropped keys (FR-148)"
```

---

### Task 4: The synthetic land-mana line claims the runtime mana abilities it stands for

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/RulesParser.java` (`parseFace`: the spell loop around line 270–320, the synthetic land mana block around line 367–371)
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceSidecarTest.java`

**Interfaces:**
- Consumes: `ProvenanceRecorder.attribute(Ability, ProvenanceKey, CardTraitBase)`, `ProvenanceRecorder.declare(ProvenanceKey)`, `TextAbility(AbilityType, String)`.
- Produces: a converted basic land whose `activated` line lists its runtime mana-ability key(s) under `provenance`, and whose `dropped_keys` holds only the play-land key `spell 0`.

- [ ] **Step 1: Write the failing test**

Add to `ProvenanceSidecarTest` (it has `convert(String)` returning `Converted` with `sidecar()`):

```java
    // ── the synthetic land mana line claims its runtime abilities ────────

    @Test
    void aBasicLandsManaLineClaimsTheRuntimeManaAbility() {
        // Forge builds the mana ability from the land type at runtime; the
        // converter renders it as one canonical line. Before this claim the
        // line had no key and the runtime ability sat in dropped_keys, so
        // every mana activation record joined to nothing.
        Converted mountain = convert("m/mountain.txt");
        List<ProvenanceSidecar.Line> mana = mountain.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("activated")).toList();
        assertEquals(1, mana.size(), mountain.lines().toString());
        List<ProvenanceKey> keys = mana.get(0).provenance();
        assertEquals(1, keys.size(), keys.toString());
        assertEquals(ProvenanceKey.KIND_SPELL, keys.get(0).traitKind());
        assertTrue(keys.get(0).indexWithinKind() >= 1, keys.toString());
        assertEquals(1, mountain.sidecar().droppedKeys().size(),
                mountain.sidecar().droppedKeys().toString());
        assertEquals(0, mountain.sidecar().droppedKeys().get(0).indexWithinKind());
    }

    @Test
    void aDualLandsManaLineClaimsBothRuntimeManaAbilities() {
        Converted tundra = convert("t/tundra.txt");
        List<ProvenanceSidecar.Line> mana = tundra.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("activated")).toList();
        assertEquals(1, mana.size(), tundra.lines().toString());
        assertEquals(2, mana.get(0).provenance().size(), mana.get(0).provenance().toString());
    }
```

Check the accessor names on `ProvenanceSidecar.Line` (`lineKind()`, `provenance()`) and on `ProvenanceKey` (`traitKind()`, `indexWithinKind()`) against the source before running; the existing tests use `line.provenance()` and `key.traitKind()`.

- [ ] **Step 2: Run it to see it fail**

Run: `mvn -q -f forge-connector/pom.xml test -Dtest=ProvenanceSidecarTest -DfailIfNoTests=false`
Expected: the two new tests fail (`expected: <1> but was: <0>` on the provenance size); the byte-identity test passes.

- [ ] **Step 3: Collect the skipped mana abilities and attribute them to the synthetic line**

In `parseFace`, before the spell loop:

```java
        // Runtime mana abilities of a basic-typed face, skipped in the loop
        // and rendered once by the synthetic block below. Kept with their
        // keys so the synthetic line can claim them: without the claim the
        // land's mana ability is a dropped key, and every activation record
        // that names it joins to nothing.
        List<SpellAbility> landManaAbilities = new ArrayList<>();
        List<ProvenanceKey> landManaKeys = new ArrayList<>();
```

Replace the skip inside the `isActivatedAbility()` branch:

```java
                if (sa.isManaAbility() && buildLandManaDescription(face) != null) {
                    landManaAbilities.add(sa);
                    landManaKeys.add(spellKey);
                    continue;
                }
```

Replace the synthetic block:

```java
        // --- Synthetic land mana ---
        String landDesc = buildLandManaDescription(face);
        if (landDesc != null) {
            TextAbility landMana = new TextAbility(AbilityType.ACTIVATED, landDesc);
            abilities.add(landMana);
            if (recorder != null) {
                for (int i = 0; i < landManaAbilities.size(); i++) {
                    recorder.attribute(landMana, landManaKeys.get(i), landManaAbilities.get(i));
                }
            }
        }
```

(`spellKey` is null when `recorder` is null, and `attribute` ignores a null key, so the non-recording path is unchanged.)

- [ ] **Step 4: Run the Java test class**

Run the same command. Expected: exit 0, all tests in the class pass including `convertedTextIsByteIdenticalToTheRecordingFreeRendering`.

- [ ] **Step 5: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/RulesParser.java forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceSidecarTest.java
git commit -m "fix(convert): the synthetic land mana line claims the runtime mana abilities it renders"
```

---

### Task 5: Deduplication, secondary triggers, keyword-derived traits and Class levels keep their attribution

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceRecorder.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/RulesParser.java` (spell loop ~270–320, trigger loop ~328–345, static loop ~346–355, `deduplicateByDescription` ~656, `applyClassPostProcessing` ~502, their call sites ~376–380)
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceSidecarTest.java`
- Modify: `CLAUDE.md` (root, the provenance sidecar paragraph: "A trait the converter deduplicated away maps to **no** line" becomes "A trait with no text of its own, such as the implicit permanent spell, maps to **no** line"), `specs/023-ability-effect-model/spec.md` (new FR-152) — feature-workflow skill first

**Interfaces:**
- Produces on `ProvenanceRecorder`: `void merge(Ability survivor, Ability duplicate)` (moves the duplicate's keys onto the survivor, creating a source if the survivor had none), `void transfer(Ability from, Ability to)` (moves a source to a replacement object), `void attributeToKeyword(ProvenanceKey key, CardTraitBase trait, String keywordOriginal)` (attributes `key` to every ability previously recorded for that keyword via `attributeKeyword`).
- Produces: `deduplicateByDescription(List<Ability>, ProvenanceRecorder)` and `applyClassPostProcessing(List<Ability>, Set<String>, ProvenanceRecorder)`.

- [ ] **Step 1: Write the failing tests**

Add to `ProvenanceSidecarTest`:

```java
    // ── every trait whose text a line carries is claimed by that line ────

    private static List<ProvenanceKey> claimedBy(Converted converted, String kind, String needle) {
        return converted.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind))
                .filter(line -> converted.lines().get(line.lineIndex()).contains(needle))
                .flatMap(line -> line.provenance().stream())
                .toList();
    }

    @Test
    void theSecondaryOfAnEntersOrDiesPairIsClaimedByTheRenderedLine() {
        // Forge registers "enters or dies" as two triggers with one
        // description; the converter renders one line. The second object is
        // still live and a record fired by it names its key.
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> keys = claimedBy(marshal, "triggered", "enters or dies");
        assertTrue(keys.stream().anyMatch(k -> k.indexWithinKind() == 0), keys.toString());
        assertTrue(keys.stream().anyMatch(k -> k.indexWithinKind() == 1), keys.toString());
    }

    @Test
    void aKeywordDerivedTriggerIsClaimedByTheKeywordLine() {
        // Echo's upkeep trigger is a runtime Trigger object with a key of its
        // own; its text is the keyword line.
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> keys = claimedBy(marshal, "triggered", "echo");
        assertTrue(keys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_KEYWORD)), keys.toString());
        assertTrue(keys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)), keys.toString());
    }

    @Test
    void onlyTheImplicitPermanentSpellIsDroppedOnMoggWarMarshal() {
        Converted marshal = convert("m/mogg_war_marshal.txt");
        List<ProvenanceKey> dropped = marshal.sidecar().droppedKeys();
        assertEquals(1, dropped.size(), dropped.toString());
        assertEquals(ProvenanceKey.KIND_SPELL, dropped.get(0).traitKind());
        assertEquals(0, dropped.get(0).indexWithinKind());
    }

    @Test
    void anEntersOrAttacksPairIsClaimedByTheRenderedLine() {
        Converted tidalmage = convert("s/stadium_tidalmage.txt");
        List<ProvenanceKey> keys = claimedBy(tidalmage, "triggered", "enters or attacks");
        assertEquals(2, keys.stream().filter(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)).count(), keys.toString());
    }

    @Test
    void aClassCardsLevelLinesCarryTheirTraitsKeys() {
        // Class post-processing rebuilds level lines as fresh objects; the
        // attribution has to move with them or every ability on the card is
        // textless to the model.
        Converted talent = convert("h/hunters_talent.txt");
        List<ProvenanceKey> levelKeys = talent.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals("level"))
                .flatMap(line -> line.provenance().stream())
                .toList();
        assertTrue(levelKeys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_TRIGGER)), levelKeys.toString());
        assertTrue(levelKeys.stream().anyMatch(k -> k.traitKind().equals(ProvenanceKey.KIND_STATIC)), levelKeys.toString());
        for (ProvenanceKey dropped : talent.sidecar().droppedKeys()) {
            assertFalse(dropped.traitKind().equals(ProvenanceKey.KIND_TRIGGER)
                    || dropped.traitKind().equals(ProvenanceKey.KIND_STATIC),
                    dropped + " has text on a level line and must not be dropped");
        }
    }
```

`claimedBy` needs `Line.lineIndex()`; check the accessor name in `ProvenanceSidecar.Line` (the record's first component is the rendered line's index).

- [ ] **Step 2: Run to see them fail**

Run the Java test command. Expected: the five new tests fail; everything else passes.

- [ ] **Step 3: Extend the recorder**

In `ProvenanceRecorder`, add a field `private final Map<String, List<Ability>> byKeyword = new LinkedHashMap<>();` and record into it inside `attributeKeyword` (after `declare(key)`): `byKeyword.computeIfAbsent(original, k -> new ArrayList<>()).addAll(abilities);` (skip nulls). Then add:

```java
    /**
     * Attribute a runtime trait to the abilities its keyword rendered.
     *
     * <p>A keyword-derived trigger or static returns no entry of its own — its
     * text is the keyword line — but Forge gives it a trait index, and a record
     * fired by it names that key. Claiming the key on the keyword's abilities
     * is what keeps such a record joinable.
     */
    public void attributeToKeyword(ProvenanceKey key, CardTraitBase trait, String original) {
        if (key == null) return;
        declare(key);
        List<Ability> abilities = byKeyword.get(original);
        if (abilities == null || abilities.isEmpty()) return;
        attribute(abilities, key, trait);
    }

    /** Fold a discarded duplicate's attribution into the ability that stays. */
    public void merge(Ability survivor, Ability duplicate) {
        Source from = byAbility.remove(duplicate);
        if (from == null || survivor == null) return;
        Source into = byAbility.get(survivor);
        if (into == null) {
            byAbility.put(survivor, from);
            return;
        }
        for (ProvenanceKey key : from.keys()) {
            if (!into.keys().contains(key)) into.keys().add(key);
        }
    }

    /** Move an ability's attribution to the object that replaces it. */
    public void transfer(Ability from, Ability to) {
        Source source = byAbility.remove(from);
        if (source != null && to != null) byAbility.put(to, source);
    }
```

Note `attribute(List, key, trait)` calls `TraitScript.of(trait)`; passing the keyword-derived trigger is fine, the keyword line keeps the script it already has because `attribute` only appends keys to an existing source.

- [ ] **Step 4: Claim in the parser**

In `parseFace`:

1. Spell loop: where `if (sa.getKeyword() != null) continue;` is, change to:

```java
            if (sa.getKeyword() != null) {
                if (recorder != null) {
                    recorder.attributeToKeyword(spellKey, sa, sa.getKeyword().getOriginal());
                }
                continue;
            }
```

2. Trigger loop: track the primary ability per Execute SVar and claim the secondary's key on it; claim keyword-derived triggers on the keyword line:

```java
        Map<String, Ability> primaryByExecute = new HashMap<>();
        int triggerIndex = -1;
        for (Trigger t : card.getTriggers()) {
            triggerIndex++;
            ProvenanceKey key = recorder == null ? null : new ProvenanceKey(
                    scriptFile, faceIndex, ProvenanceKey.KIND_TRIGGER, triggerIndex);
            if (recorder != null) recorder.declare(key);
            String exec = t.getParam("Execute");
            if ("True".equalsIgnoreCase(t.getParam("Secondary"))
                    && exec != null && primaryExecuteSVars.contains(exec)) {
                // The "dies" or "blocks" half of a pair the primary already
                // renders: one line, two runtime objects, both keys on it.
                if (recorder != null) recorder.attribute(primaryByExecute.get(exec), key, t);
                continue;
            }
            if (exec != null && !"True".equalsIgnoreCase(t.getParam("Secondary"))) {
                primaryExecuteSVars.add(exec);
            }
            Ability triggered = TriggeredAbilityEntry.of(t);
            addIfNotNull(abilities, triggered);
            if (recorder != null) {
                if (triggered == null && t.getKeyword() != null) {
                    recorder.attributeToKeyword(key, t, t.getKeyword().getOriginal());
                } else {
                    recorder.attribute(triggered, key, t);
                }
            }
            if (triggered != null && exec != null) primaryByExecute.put(exec, triggered);
        }
```

3. Static loop: after `Ability entry = StaticAbilityEntry.of(s);`:

```java
            if (recorder != null) {
                if (entry == null && s.getKeyword() != null) {
                    recorder.attributeToKeyword(key, s, s.getKeyword().getOriginal());
                } else {
                    recorder.attribute(entry, key, s);
                }
            }
```

4. `deduplicateByDescription` takes the recorder and merges:

```java
    private static List<Ability> deduplicateByDescription(List<Ability> abilities,
                                                          ProvenanceRecorder recorder) {
        Map<String, Ability> survivors = new HashMap<>();
        List<Ability> result = new ArrayList<>();
        for (Ability a : abilities) {
            String desc = a.descriptionText();
            if (desc == null) {
                result.add(a);
                continue;
            }
            Ability survivor = survivors.putIfAbsent(desc, a);
            if (survivor == null) {
                result.add(a);
            } else if (recorder != null) {
                recorder.merge(survivor, a);
            }
        }
        return result;
    }
```

and the call site becomes `abilities = deduplicateByDescription(abilities, recorder);`.

5. `applyClassPostProcessing` takes the recorder; the `removeIf` becomes a loop that merges each removed ability into the LEVEL ability with the same description, and the `result.set(i, ...)` transfers:

```java
    private List<Ability> applyClassPostProcessing(List<Ability> abilities,
                                                   Set<String> classLevelDescriptions,
                                                   ProvenanceRecorder recorder) {
        List<Ability> result = new ArrayList<>(abilities);

        Map<String, Ability> levelByDescription = new HashMap<>();
        for (Ability a : result) {
            if (a.type() == AbilityType.LEVEL) levelByDescription.putIfAbsent(a.descriptionText(), a);
        }
        Iterator<Ability> it = result.iterator();
        while (it.hasNext()) {
            Ability a = it.next();
            if (a.type() != AbilityType.LEVEL && classLevelDescriptions.contains(a.descriptionText())) {
                if (recorder != null) recorder.merge(levelByDescription.get(a.descriptionText()), a);
                it.remove();
            }
        }

        for (int i = 0; i < result.size(); i++) {
            Ability a = result.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                TextAbility level = new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1);
                if (recorder != null) recorder.transfer(a, level);
                result.set(i, level);
            }
        }
        // (the existing sort follows unchanged)
```

and the call site becomes `abilities = applyClassPostProcessing(abilities, classLevelDescriptions, recorder);`. Add `import java.util.Iterator;` and `import java.util.HashMap;` / `java.util.Map` if missing. If the level lines on Hunter's Talent are produced elsewhere than these two rewrites (look for where `classLevelDescriptions` is filled and where `LEVEL` abilities are created for `level[N]:` lines), apply the same `transfer` there: the rule is that every `new TextAbility(...)` that replaces an attributed ability calls `recorder.transfer(old, replacement)`.

- [ ] **Step 5: Run the Java test class**

Run the Java test command. Expected: exit 0. The byte-identity test must still pass: nothing here changes rendered text. If `aClassCardsLevelLinesCarryTheirTraitsKeys` still fails, print `talent.sidecar()` in the test temporarily and trace which rewrite loses the object; do not weaken the assertion.

- [ ] **Step 6: Add FR-152 and update CLAUDE.md**

Invoke the `feature-workflow` skill. In `specs/023-ability-effect-model/spec.md`, after FR-151 add:

```
- **FR-152**: The converter MUST claim, on a rendered line, every runtime trait whose text that line
  carries: the synthetic land-mana line claims the runtime mana abilities it stands for, the
  survivor of a deduplicated description claims the duplicate's keys, the primary of a secondary
  trigger pair claims the secondary's key, a keyword-derived trigger, static or spell is claimed by
  the keyword's line, and a Class card's level line inherits the attribution of the ability it
  replaces. `dropped_keys` MUST hold only traits with no text of their own. Claiming MUST NOT change
  the rendered text.
```

In the root `CLAUDE.md`, in the provenance sidecar paragraph, change "A trait the converter deduplicated away maps to **no** line and appears in `dropped_keys` — still live at runtime, so a record naming it is kept with nothing to join to." to "A trait with no text of its own, such as the implicit permanent spell on every permanent, maps to **no** line and appears in `dropped_keys`; a deduplicated or keyword-derived trait is claimed by the line that carries its text. `build-corpus` refuses a resolution record acting only through dropped keys, and the surface builder emits no token for one."

- [ ] **Step 7: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceRecorder.java forge-connector/src/main/java/com/pricepredictor/connector/RulesParser.java forge-connector/src/test/java/com/pricepredictor/connector/effects/ProvenanceSidecarTest.java specs/023-ability-effect-model/spec.md CLAUDE.md
git commit -m "fix(convert): every trait whose text a line carries is claimed by that line (FR-152)"
```

---

### Task 6: An audit script that measures textless slots and records

**Files:**
- Create: `scripts/effects_textless_audit.py`

**Interfaces:**
- Consumes: `effects.infrastructure.record_io.read_shard`, `effects.infrastructure.sidecar_io.SidecarCache`, `effects.domain.provenance.KeyResolution` (Task 3). If Task 3 has not landed yet, resolve with `line_for` plus `dropped_keys` membership as the fallback and note it in the module docstring; the script is a diagnostic, not a library.
- Produces: a CLI `python scripts/effects_textless_audit.py --corpus output/effects/corpus --cards-folder output/cardsfolder --cards-folder output/tokenscripts [--variant-scripts DIR] [--shards 10] [--records-per-shard 1500] [--seed 7]` printing two tables.

- [ ] **Step 1: Write the script**

```python
"""Measure how much of the curated corpus the effect model reads as no text.

Two tables. The first is per ability slot on the board: how many of the
tokens the head reads are zero vectors, by trait kind and by why (dropped by
the converter, runtime-only, unconverted script). The second is per
resolution record: whether the acting ability has text, and for the ones
that do not, which scripts they act through.

Run before and after a reconversion or a corpus rebuild. The numbers this was
written against are in docs/superpowers/specs/2026-09-17-textless-abilities.md.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import random
from pathlib import Path

from effects.domain.provenance import KeyResolution
from effects.domain.records import RecordKind
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cards-folder", action="append", type=Path, required=True)
    parser.add_argument("--variant-scripts", type=Path)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--records-per-shard", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    roots = {folder.name: folder for folder in args.cards_folder}
    if args.variant_scripts:
        roots["variant-scripts"] = args.variant_scripts
    sidecars = SidecarCache(roots)

    files = sorted(glob.glob(str(args.corpus / "training" / "shard-*.jsonl.gz")))
    random.seed(args.seed)
    files = random.sample(files, min(args.shards, len(files)))

    slots = 0
    zero_by = collections.Counter()
    records = 0
    acting = collections.Counter()
    textless_scripts = collections.Counter()
    for path in files:
        for record in itertools.islice(read_shard(Path(path)), args.records_per_shard):
            records += 1
            for entity in record.state.entities:
                for key in (*entity.printed, *entity.granted_attached):
                    slots += 1
                    outcome = sidecars.resolution_of(key)
                    if outcome is not KeyResolution.LINE:
                        zero_by[(key.trait_kind, outcome.value)] += 1
            if record.kind is RecordKind.RESOLUTION and record.ability:
                outcomes = {sidecars.resolution_of(key) for key in record.ability}
                if KeyResolution.LINE in outcomes:
                    acting["has text"] += 1
                elif KeyResolution.RUNTIME_ONLY in outcomes:
                    acting["runtime-only key (kept)"] += 1
                else:
                    acting["no text"] += 1
                    textless_scripts[record.ability[0].script_file] += 1

    zero = sum(zero_by.values())
    print(f"{records} records, {slots} ability slots, {zero} zero-vector "
          f"({100.0 * zero / slots if slots else 0.0:.1f}%)")
    for (kind, why), count in zero_by.most_common(10):
        print(f"  {count:9d}  {kind:<12} {why}")
    print("resolution records by acting text:")
    for label, count in acting.most_common():
        print(f"  {count:9d}  {label}")
    print("no-text acting scripts, most first:")
    for script, count in textless_scripts.most_common(10):
        print(f"  {count:9d}  {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it against the current corpus**

Run: `CUDA_VISIBLE_DEVICES="" python scripts/effects_textless_audit.py --corpus output/effects/corpus --cards-folder output/cardsfolder --cards-folder output/tokenscripts --variant-scripts output/effects/variant-scripts --shards 4`
Expected: about 80% zero-vector slots and about three quarters of resolution records with no text (the pre-fix numbers from the design record). Paste the output in the commit message body.

- [ ] **Step 3: Lint and commit**

Run: `CUDA_VISIBLE_DEVICES="" python -m ruff check scripts/effects_textless_audit.py`

```bash
git add scripts/effects_textless_audit.py
git commit -m "chore(effects): audit script for textless ability slots and acting keys"
```

---

## Operator runbook (after all six tasks are merged)

Order matters: the refusal rule (Task 3) reads the sidecars, so the sidecars must already carry the converter's claims (Tasks 4 and 5) before the corpus is rebuilt.

1. Rebuild the connector JAR: `mvn -q -f forge-connector/pom.xml package -DskipTests`
2. Reconvert (minutes): `python -m price_predictor convert --output-path ./output/cardsfolder --tokens-output-path ./output/tokenscripts`
3. Audit the existing corpus against the new sidecars: `python scripts/effects_textless_audit.py --corpus output/effects/corpus --cards-folder output/cardsfolder --cards-folder output/tokenscripts --variant-scripts output/effects/variant-scripts`. Expect basic-land mana activations, deduplicated triggers and Class abilities to have moved from "no text" to "has text", and the remaining "no text" to be permanent spells. A raised `KeyError` here means a card reconverted to a different shape than the record's; investigate before rebuilding.
4. Rebuild the corpus (about an hour): `python -m effects build-corpus --records-dir output/effects/records/ --output output/effects/corpus/ --variant-scripts output/effects/variant-scripts/ --vocab-path models/effects/vocab-script.txt`. Check the log for the `no-acting-text` count and its top scripts.
5. Audit the rebuilt corpus with the same command as step 3. Expect zero-vector slots to remain (the audit counts keys, and the phantom keys are still on the records) but every resolution record to have text or a runtime-only key.
6. Train as before. The first epoch line's per-field percentages and the gate F1 are the comparison against the run on `09.17b.log`.

No recollection is needed; the raw records' keys are runtime ordinals and join to the regenerated sidecars.

## Self-review notes

- Spec coverage: FR-073 (Task 1), fail-loudly rule of the sidecar contract (Task 2), FR-148 and manifest field (Task 3), FR-152 and the contract's rule table (Tasks 3 and 5), the design record's "order of operations" (runbook). The out-of-scope runtime-only join is kept out by the `RUNTIME_ONLY` exception in Task 3.
- Type consistency: `has_line` (Task 1) is the name used in both files; `UnconfiguredTree` (Task 2) is what Task 3 imports; `KeyResolution` and `resolution_of` (Task 3) are what Task 6 calls; `merge` / `transfer` / `attributeToKeyword` (Task 5) match between recorder and parser.
