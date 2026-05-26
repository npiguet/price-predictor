# May 20, 2026 — Picker spec finalized and built

**TL;DR:** Ran the full speckit workflow on spec 017 (one-shot sealed
deck picker), caught and fixed a scoring-input bug in the spec, then
implemented the entire feature — 38 tasks, 504 tests passing — in a
single session.

The day started on the `017-one-shot-deck-picker` branch with
`/speckit.clarify`, which surfaced five open decision points in the
picker spec. I answered them in order: the auditor scorer gets an
optional `--auditor-scorer-checkpoint` flag on `train-picker` (off by
default); the baseline cross-scorer correlation is a documented manual
procedure with no CLI surface, same treatment as the cold-start check;
the best checkpoint artifact uses `best_{timestamp}.pt` (timestamp fixed
at run start, overwritten on each new val-reward best) plus `latest.pt`,
dropping per-epoch snapshots entirely — a decision I arrived at during
`/speckit.plan` when Claude pointed out it matched the rest of the
project better; the distributional summaries logged per epoch are mean
summaries plus a 5-bin CMC histogram rather than full per-deck
histograms; and the reproducibility seed is hardcoded to 42, no CLI
flag, matching `train-encoder`.

The planning artifacts (`research.md`, `data-model.md`, `contracts/`,
`quickstart.md`) went in cleanly. When `/speckit.analyze` ran, it
flagged three things worth acting on: a missing README task (constitution
VI requires every subcommand to be documented), undertested behavioral
FRs around the val split, early stop, and best-checkpoint selection, and
an apparent divergence where the picker's reward path included basic
lands in the scorer input while `GreedyDeckBuilder` scored chosen-only.

That third one turned out to be a real bug in the spec, not a design
choice. I pointed out that the scorer doesn't know what to do with basic
lands and that including them would be unnecessary noise — the basics are
added deterministically after the picks anyway. Claude agreed and fixed
the error in all five places it had propagated: `spec.md` FR-012 and
FR-019, the prescriptive design doc's §3.2, and the corresponding
implementation task descriptions. The fix also erased the scorer-input
divergence between the picker and the greedy builder, which matters for
keeping prior benchmark numbers comparable.

With the spec clean, `/speckit.implement` ran through all six phases.
The picker consists of `picker_model.py` (SAB trunk borrowed from the
scorer, per-card policy head, aux pool-quality head, deterministic
pick-decomposition walk), `picker_store.py`, `train_picker.py` (full
REINFORCE pipeline: GPU-batched sampler, Plackett-Luce log-prob, frozen
scorer, per-pool baseline, entropy decay schedule, KL penalty option,
Spearman audit hook), and `pick_decks.py` (one deterministic forward per
pool, manabase fill, append-and-skip resume). The full sealed unit suite
came out at 504 tests with ruff clean.

After the commit I asked what command line I would need to run the
cold-start sanity check described in §3.6 of the spec. Claude explained
there was no CLI subcommand by design and provided a ready-to-run
standalone script. I had it saved into `scripts/coldstart_check.py`
instead of leaving it as a paste, which it did using the same
`sys.path.insert` shim convention as the other scripts there.
