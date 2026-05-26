# May 10, 2026 — Encoder training: data cleanup and speedup

**TL;DR:** I shipped the full multi-head MLM encoder (spec 016), then
spent the rest of the day cleaning up a noisy training dataset and
cutting the per-epoch training time from ~2 m 40 s down to ~15 s.

The day started with Claude running the `speckit.implement` workflow
for the card-winnability pretraining spec. That involved a full rewrite
of `train_encoder.py` around nine regression heads (four signed, five
per-color), an MLM auxiliary loss, per-batch weighted MSE with FR-017
card weights, and a 5%-warmup-then-constant LR schedule. The `[MASK]`
token got seeded into the vocab between `cardname` and the first domain
term. The 23-column `cards-win-rates.txt` snapshot format had a
longstanding arithmetic typo (documented as 24 columns in several plan
and contract files) which Claude caught and corrected. The full sealed
suite — 877 unit tests plus 13 integration tests — came back green after
that.

Then I pointed the freshly-implemented encoder at the real
`cards-played.txt` dataset. What came back was 607 entries with
`unplayed_count = 0`, which I suspected were bogus. Claude analyzed
them: 259 turned out to be real cards (split-card halves, MKM rooms,
adventures — Forge logs the resolved face as "played"), 6 were cards in
Forge's corpus but not in the AllPrintings snapshot, and 342 were
genuine junk. The junk split into two groups: 256 Forge-internal
triggered-ability names like `"Goblin (4)'s Effect"` matched by a
single regex, and 86 Universes Beyond flavor names — `"Cloud Strife"`,
`"Squall Leonhart"` and so on. Claude then checked AllPrintings and
confirmed these are `flavorName` reskins of real Commander staples (`"Cloud
Strife"` → `"Najeela, the Blade-Blossom"`, `"Squall Leonhart"` →
`"Danitha Capashen, Paragon"`). That meant the cleaning job wasn't just
a drop-list but a full canonicalization pass.

The plan Claude put together had two parts. First, build an alias map
from AllPrintings `faceName` entries (split cards, adventures, DFCs,
rooms) and `flavorName` entries (UB reprints), plus a junk-drop
predicate for the triggered-ability and token-style names. Then stream
`cards-played.txt` through a rewriter that substitutes canonical names
and de-duplicates across the played/not-played columns when both halves
of a split card appear. A dry run over the 974 k rows found a bug in the
identity-vs-alias conflict resolution — `"Lightning Bolt"` is both a
canonical name and a `faceName` of a saga, and the original
lexicographic-smallest heuristic picked the saga. Fixing that required
the identity key always winning. Then another iteration was needed when
"dungeon" and "Monarch" names came up as unmapped: they're listed as
tokens in MTGJSON, not cards, and the walker had skipped tokens
entirely. After both fixes the unmapped residue dropped from 1,474 to
35 names (all Forge-internal set-effect names that the Forge fix would
eliminate going forward anyway).

On the Forge side, I noticed `PlayedCardCollector.java` was calling
`getName()` instead of `getOracleName()`, which is what produced the
flavor names in the first place. Claude swapped the call, all 260 Java
tests passed, and the jar was rebuilt. Future `match-outcomes` runs will
emit canonical names directly without needing the cleaning script.

There were also corpus-consistency failures when `train-encoder` tried
to resolve the cleaned names against the converted card files.
`ConvertedCardLocator` didn't know to look in the `rebalanced/`
subfolder for `A-*` alchemy cards, and didn't handle meld cards stored
as front-face-only files or cards with `&` in their names. Claude fixed
the locator (retries with just the front face for `Front // Back` names
that don't resolve directly, collapses `__` to `_` in sanitized names,
and adds a correction for `bespoke_bo.txt`'s stray space). After all
that, the corpus-consistency check came back at 27,983 cards with 0
missing. I also decided that missing cards should produce a warning and
be dropped rather than aborting the run, since Forge simply doesn't have
scripts for a handful of Secret Lair exclusives.

The vocab builder had a separate problem: `_extract_set_code_tokens` was
fragmenting every key in the MTGJSON `data{}` dict, which includes ~97
Art Series sets like `ABRO` and `ACLB` that never appear in any real
card's `set:` line. The fix was to only fragment set codes that appear in
at least one card's `printings` list.

By afternoon I was actually running `train-encoder` for real, and I
noticed the warm-up took over four minutes before epoch 1 and each epoch
ran ~2 m 40 s. I asked Claude to do a code review focused on the
performance problems. The review identified seven findings. Three were
high priority: `_WinnabilityDataset.__getitem__` was re-reading each
card's `.txt` file and re-tokenizing it on every single access (~22 k
disk reads per epoch); the 700 MB `cards-played.txt` was being streamed
twice in separate passes; and the MLM head was projecting all `B×T`
token positions to `(B, T, vocab_size)` then masking, rather than
gathering the ~15% masked positions first and projecting only those.
There were also two medium/low items: dynamic per-batch padding instead
of a global 608-token ceiling, vectorizing `_per_batch_weighted_mse` to
avoid per-head `.item()` syncs, and an O(n²) safety branch in
`_split_cards`. I initially had doubts about F3 (whether gathering on
the CPU first would negate GPU savings) and about F4 (whether the
transformer required fixed sequence length). Claude clarified both: the
gather in the current code is GPU-side already, and transformers are
length-agnostic by construction. I decided to apply all seven.

I asked Claude to profile the warm-up before starting so we had real
numbers. The profiler showed: the 700 MB parse itself takes only 5 s;
the card file reads are another 2.7 s; and `_aggregate` is ~96 s of
pure CPU loop work — specifically ~38 million `CardCounters()` default
constructions triggered by `dict.setdefault` evaluating its default
argument every call, even when the key already exists. Switching to
`dict.get` then a conditional assignment dropped that overhead, and
replacing the per-color `dict[str,int]` with a `list[int]` indexed by
WUBRG position shaved a bit more. The aggregate drop was ~15% on the
hot loop.

After all the fixes, measured timings on the real dataset: warm-up
~2 m 55 s (down from ~4 m 32 s), per-epoch ~15 s (down from ~2 m 40 s,
roughly 10×). The encoder diagnostics — perplexity and per-head val
correlation — were added at my request before the run so I could
actually read the training logs. By epoch 16, `val corr` for
`score_play` was ~0.46, `played_rate` ~0.51, MLM ppl had dropped from
1487 to 5.9, and masked-token top-1 accuracy was 56.5%. The color-lift
heads were near zero, which Claude explained is expected given sparse
per-color slice data and a 1/5 loss prefactor.
