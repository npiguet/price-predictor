# Gen-3 model layout

Rules to respect when changing the input layout, the relation attention or an output head. This file does not describe the architecture:

- What the layout is: `specs/2026-10-30-ability-effect-model-gen3.md` § 9 (input), § 10 (relation-aware trunk), § 11 (outputs).
- Code: `domain/effect_head_input.py`, `domain/relation_attention.py`, `domain/relation_types.py`, `domain/effect_head_layout.py`.

## Orders are part of a checkpoint

- The order of the relation types, the subtypes, the named counters and the closed output lists fixes rows and columns of the model's tables. A checkpoint records them and a loader refuses a difference, so reordering or inserting into any of them invalidates every existing checkpoint (`checkpoints.md`).
- Those lists are generated, not hand-edited:
  - `domain/kept_keyword_flags.py` by `scripts/scan_keyword_flags.py`; its test fails when the file is stale.
  - `domain/forge_vocabularies.py` by `scripts/scan_forge_vocabularies.py`.
  - `domain/output_vocabularies.py` by `scripts/scan_output_vocabularies.py --manifest`, from a built manifest. The lists checked in are provisional, surveyed from pilot collections (the module's docstring names them): before the first gen-3 training, regenerate them from the curated corpus's manifest, and update the head widths `tests/unit/effects/domain/test_layout_vocabularies.py` pins, which follow the lists' lengths. A checkpoint trained on the provisional lists loads against those lists only.

## Input slots

- `slot_lines` is the one list of a card's ability slots, read by both the batcher and the surface builder. `board_entities` is the one card list.
- A keyword keeps a card-level flag only where a Forge card-script selector or engine check names it. Every keyword is also read through its line's reminder text, a line with no key through `keyword-line/<fragment>`.
- A face that is not up gives inactive slots.
- The input reads targets as cast (`Refs.as_cast`: `targets_at_cast`, else `targets`) for `[ACT]`'s target count, every "targeted" bit and `[ACT]`'s `targets` relations. Reading `targets` on an effect half would hand the model the fizzle it is to predict. On a record without `targets_at_cast` the fallback still leaks it.
- A keyword reads in one spelling, defined in `domain/keyword_names.py` (`Ward:2` → `ward`, `Protection:Goblin` → `protection_from_goblin`, `Trample:Planeswalker` → `trample_over_planeswalkers`, which is trample too).
- Floating effects and delayed triggers with no entity in the snapshot are card slots after the board's. They are no per-entity target.
- A dungeon is a card slot with one keyword line for all its rooms, so a room has no line of its own.

## Relations

- `RelationEncoderLayer` keeps `nn.TransformerEncoderLayer`'s parameter names.
- The relation key and value tables start at zero and sit in the `head` parameter group.
- A record's relation triples are built once the slots are fixed, after context dropout and gate 2's strip. A line end with no slot reads its card's; a link to an object with no slot is dropped.
- `scripts/measure_trunk_step.py` measures a step on a curated shard.

## Outputs

- A field takes a silent value only where the collector writes its event for every occurrence.
- `protector_changed` has no source, since Forge publishes nothing for it, so it never trains.
