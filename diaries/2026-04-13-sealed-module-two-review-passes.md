# Sunday 13 April 2026 — Sealed module two review passes

**TL;DR:** Two back-to-back top-to-bottom code reviews of the `sealed` module,
each producing a multi-phase refactor plan that Claude executed in full. The day
produced seven commits and found a silent mana-cost parsing bug along the way.

The first review identified 16 findings. The main problems were magic-number
literals (544 / 512 / 32) scattered across four files with no shared source of
truth, `evaluate_scorer.py` at 554 lines mixing five distinct concerns, and the
Forge JVM classpath being assembled four separate times across the codebase. The
plan grouped these into four phases: centralize the embedding-dimension constants
into a `card_embedding_layout` module, consolidate the mana-cost line parsers and
CLI argument blocks, tighten the use-case/CLI boundary, and extract
`GreedyDeckBuilder` and `RoundRobinOutcome` as named domain types. All four
phases landed before 2 AM.

The second review went deeper. It found 17 more findings, including a genuine
bug: `evaluate_scorer._extract_mana_cost` was doing a case-sensitive
`line.startswith("mana cost:")` with no `.strip()`, which would silently
produce wrong results on any card whose converted text had inconsistent
casing or leading whitespace. The canonical `extract_mana_cost_line` helper
from `price_predictor` already handled this correctly; the fix was to delete
the local reimplementation and call the shared one.

The second session also caught a double-count in hybrid mana value computation:
`deterministic_features.py` was walking mana symbols by hand rather than
delegating to `ManaCost.total_mana_value`, which had already accumulated the
correct logic. Switching to the domain object fixed it.

The largest structural change across the day was the train-scorer decomposition.
`TrainScorerUseCase.execute` had grown to 175 lines braiding data loading, model
construction, optimizer setup, the training loop, validation, and checkpointing
all in sequence. By the end of the second session it was around 50 lines of
orchestration calling named helpers, with a typed `TrainScorerResult` return type
and a `LoadedScorerCheckpoint` wrapper that auto-converts legacy dict-shaped
configs on load so existing on-disk checkpoints stay readable.

The regression check for the `deterministic_features` rewrite ran across all
32,116 cards in `output/cardsfolder/` and returned zero mismatches against the
old implementation, confirming the refactor was behavior-preserving. The hash-
based per-card embedding lookup was also replaced with a proper `nn.Embedding`
table keyed by index, which is the standard way to do it in PyTorch and avoids
the silent collision risk that hashing introduces.

By the third session (which generated the final `aef31ba` commit), the
deterministic-feature schema had named slot constants, the training split was
shuffled rather than deterministic-by-index, round-robin helpers had moved into
their own domain module, and the CLI was driving defaults from dataclass fields
rather than argparse defaults. All 648 non-integration tests passed.
