# April 27, 2026 — Dataset load 45× speedup

**TL;DR:** The sealed scorer's training dataset was taking 415 seconds
to load 20k examples. A code review pinpointed two I/O hotspots, and
fixing them brought load time down to 9.1 seconds.

The starting observation was simple: 415 seconds to load 20k training
examples seemed wildly out of proportion to the data size. I kicked off
a `/review` on the dataset loading code to understand why.

Claude's review identified two root causes. The first was in
`ConvertedCardLocator._find_file`: every time a card name didn't have an
exact-match file — which is common for double-faced cards, split cards,
and adventures — the code rescanned the entire letter directory on disk.
With ~3,600 entries in a typical letter directory and thousands of unique
cards per training run, this was a filesystem-scan-per-card problem, not
an algorithmic one. The fix was a lazy per-letter index: scan each
directory at most once, then do dict lookups. The second root cause was
that `np.load` on `.npz` files was leaking open file handles (a zip
file stays open until GC runs), which on Windows added further drag.

The structural findings in `match_data_loader.py` — collapsing duplicate
drop-guard code, making `_report_drops` iterate over the enum instead of
hard-coding individual reasons, switching to `torch.nn.utils.rnn.pad_sequence`
— were addressed in the same pass.

After the fix, load time dropped from 415 s to 9.1 s: a 45× speedup,
confirmed on the real 27k-outcome dataset. Claude flagged this in the
summary, and I approved the commit.

After the commit, I asked whether the new per-letter index could reach
the `rebalanced/` and `upcoming/` subdirectories in `output/cardsfolder/`.
Claude checked and confirmed it couldn't — but also showed this was not
a regression: neither the old code nor the new code had ever routed any
card name to those folders. No card in the current `match-outcomes.txt`
touches either directory. The `rebalanced/` folder holds 229 Alchemy
cards (prefixed `a-`), and `upcoming/` holds 107 unreleased-set cards;
neither category is sealed-legal. I confirmed that sealed-legal sets are
a strict subset of paper-draftable sets, so Alchemy rebalances and
upcoming sets are excluded by construction. The routing gap is latent,
not active, and the right call is to leave it alone until a real case
surfaces.
