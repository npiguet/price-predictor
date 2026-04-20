# Sunday, March 29, 2026 — Spec analysis and fixes for 012

**TL;DR:** Ran the spec analyzer twice on the 012 (sealed stage 1 training) spec
artifacts, found a cluster of real issues, and applied targeted remediations
before moving to implementation.

The first analyzer run turned up a projection layer inconsistency that had been
lurking across six files. The sealed-deck-picker doc mentioned "a small
projection layer (512 → 512) sits between the encoder and pool transformer," and
that one sentence had propagated into the spec, plan, data-model, research, and
tasks files — with two different places claiming to own it (PoolLoader and
PoolTransformerModel). After tracing it back to the original spec sentence, I
decided the right call was to remove it entirely: the pool transformer receives
raw 516-dim slot features directly, and having a trainable linear layer before
the transformer adds complexity without a clear justification. All six files were
updated to drop every reference to the projection.

The second analyzer pass — on the revised artifacts — found cleaner but still
meaningful issues. The most structurally significant one was that EpisodeRunner
in the domain layer was typed to accept `EmbeddingStore`, which is an
infrastructure type. That would require a cross-layer import and violate the
hexagonal architecture the project follows. The fix was to define a
`CardEmbeddingPort` Protocol in the domain layer so EpisodeRunner stays
infrastructure-free. A second phantom type, `SlotState`, appeared in the
PoolLoader signature but was never defined anywhere; removing it clarified that
PoolLoader's job is to build the initial base tensor with all flags at zero, and
EpisodeRunner owns per-step mutation. There was also a `pool_names: str` vs.
`list[str]` confusion — the type is actually correct (it's a semicolon-separated
string matching the match-outcomes file format) but the plan and tasks lacked a
note making that explicit, so the format was documented rather than the type
changed.

Both runs also flagged a missing README update task that plan.md had promised
under its constitution check but that tasks.md had never created. Task T028 was
added.
