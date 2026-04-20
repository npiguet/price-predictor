# Saturday, 28 March 2026 — Sealed dataset spec and build

**TL;DR:** Spec'd and fully implemented the `sealed` module's Stage 0
pipeline in a single day — card embedding and pool generation. Also
wrote and clarified the Stage 1 training spec, catching a design flaw
that would have made the RL curriculum step pointless.

The first thing to land was the `sealed` package itself. It was spec'd
on branch 011-sealed-dataset and implemented the same day: a new
`encode-cards` command that reads each converted Forge card file,
strips the `name:` line so the card's identity doesn't bleed into
its embedding, and writes a `.npz` file alongside the original using
the same base filename. The encoder's output is a concatenation of
max-pool and mean-pool over the transformer's token outputs, yielding
a 2×d_model vector (512 when d_model=256). Re-running is safe and
idempotent; a `--clean` flag was added afterwards to force a full
re-encode.

A bug surfaced immediately after the first real run: the transformer
checkpoint stores its config as a plain dict (via `asdict()`), so
loading it directly and constructing `CardPriceTransformerModel(config)`
failed with an attribute error. The fix was to route through
`transformer_store.load_model()`, which already handles the dict-to-
dataclass reconstruction. After that, 32,116 cards encoded with zero
errors.

The pool generation side added `PoolGenerator.java` and `PoolMain.java`
to the forge-connector. Forge's booster generator can always fill a
pool for any set because cards in boosters are allowed to repeat, which
also disposed of the edge case about sets with too few unique cards —
it simply cannot happen. Basic lands can and do appear in booster
output, so the output filter for them is an explicit requirement, not a
silent assumption.

Partway through the spec work, I noticed the spec had been describing
card embeddings as "512-dimensional." That's only true for one specific
hyperparameter choice. I corrected it to "2×d_model-dimensional (512
when d_model=256)" and propagated the same correction to sealed-deck-
picker.md and the 007 transformer-arch spec.

The Stage 1 training spec (012-sealed-stage1-training) was written and
clarified the same day. The most consequential moment was catching a
flaw in the initial plan: the research document had proposed logit
masking to prevent the model from picking already-picked cards. I
pointed out that if the mask makes illegal picks impossible, the model
has nothing to learn — Stage 1 would be trivially solved from episode
one. Claude agreed and removed the mask. The `available_flag = 0` on
picked slots remains as an informational input feature, but selection
is unconstrained; the reward signal is what teaches avoidance.

A second correction to the Stage 1 design: actions must be recorded
in pool-index space, not shuffled-input-position space. If the same
card occupies pool slot 5 and appears at shuffled positions 1 and 35
in two different episodes, picking it twice is a duplicate regardless
of input position. The data-model, plan, and EpisodeRunner description
were all updated to make the permutation translation step explicit.

The five clarification questions resolved for Stage 1 were: batch size
(32), pool sampling strategy (sequential with shuffle at epoch boundary),
KL divergence response when exceeded (flush the replay buffer), missing
embedding behavior (fail fast with a clear error), and whether resume
preserves pool position (restart from beginning of shuffled list).
