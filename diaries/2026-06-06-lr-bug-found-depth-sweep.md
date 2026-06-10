# June 6, 2026 — LR bug found, depth sweep

**TL;DR:** A silent `--lr` no-op on resume turned two days of training
into noise; fixing it unlocked a clean decay ladder to the gen-1 best
(val_loss 0.6148 → 0.5813 with 6 layers). A live-play spec was also
drafted.

The session opened with results from the first `train-draft-agent` run
on the greedy-labelled corpus: imitation top-1 rising 0.35→0.78, top-3
touching 0.98, and the critic settling at val_crit ≈ 0.27 (~73%
variance explained) with the expected p1>p2>p3 per-pack ordering. I
killed the run early because the trajectory looked flat, but Claude
pointed out the *envelope* had genuinely dropped ~0.07–0.08 from epoch
0 to epoch 1 while the critic had converged long before — the flatness
was the noise amplitude matching the remaining slope.

The real discovery came when I tried resuming at 3e-5 and then 3e-6 and
both runs came back with nearly identical trajectories. Claude traced
the wiring: `--lr` was reaching `config.lr` correctly, but on resume
`optimizer.load_state_dict` restored the stale `initial_lr=3e-4` baked
in from the original run, and the freshly-built `LambdaLR` used that
key (via `setdefault`) as its base — so every resume, regardless of the
`--lr` flag, silently trained at the original 3e-4. Both of my
"lower-LR" runs had actually run at 3e-4. The fix was a small helper
that pops `initial_lr` from all param groups before building the
scheduler, so the new `config.lr` becomes the base. A regression test
was added and it's the only place in the codebase with a warmup
`LambdaLR`, so nothing else was affected.

With the fix in place, a real 3e-5 resume descended monotonically
rather than overshooting, reaching val_loss 0.635/top1 0.860. A
follow-on 3e-6 resume polished that to **val_loss 0.6148 / top1 0.870**
before early-stopping. That's the gen-1 4-layer floor.

That manual decay ladder prompted the LR-annealing feature:
`--lr-decay-patience N` triggers a ×0.1 LR reduction after N
mini-epochs without a new best, reusing the same strict-best counter
that early-stop already uses. The key design choices were: opt-in flag
(today's behavior unchanged), decay folds into the existing warmup
`LambdaLR` via a `_PlateauLR` multiplier (single writer, no scheduler
conflict), `lr_decay_count` checkpointed so resumes continue the ladder,
and early-stop can only fire once `min_lr` is reached — so the patience
knob genuinely means "stop only when even the floor LR can't improve."
All 97 draft tests passed.

The experiment file went through several edits. An earlier "warmup
overshoot" subsection had drawn wrong causal conclusions from the
bug-era runs; I rewrote it, and then on reflection decided to drop the
invalid runs entirely rather than document them with a correction note
— they're just noise. The file now reads as if I ran the clean decay
ladder directly.

For the depth sweep: a 6-layer run using the new annealing feature
reached **val_loss 0.5813 / top1 0.877**, a clear gain over the 4-layer
floor and mostly in the policy head (val_imit −0.023, val_crit −0.011).
The per-pack critic improvement was proportionally larger at packs 2–3
(≈5–8% relative) than pack 1 (≈2%), which makes sense: pack 1's MSE is
dominated by irreducible outcome variance since the deck is barely
determined, while later packs hold more learnable structure. An 8-layer
run was added next, and it turned out to be a regression (val_loss
0.6215, top1 0.858). The interesting finding there was that it was an
*optimization* failure, not overfitting — train_imit was 0.356 vs 6L's
0.309, with near-identical train–val gaps, meaning the deeper model
simply settled into a worse basin under the same LR schedule. At the
high-LR stage (3e-4) the 8-layer model was actually *ahead*, which
confirms the capacity is usable — it just wants its own schedule. **6
layers is the gen-1 model.**

Alongside all of this, I drafted the live-play integration spec
(`specs/2026-06-04-draft-agent-live-play.md`): the Java worker emits a
`<<DRAFT-PICK-REQUEST>>` sentinel for model-piloted seats and blocks on
stdin; the Python supervisor maintains a per-seat online state tracker
and writes `<<DRAFT-PICK-RESPONSE>>` back. The pack-in-hand is the only
thing the Java side sends; full typed-token state is reconstructed on
the Python side. I also clarified the generation naming convention:
gen-0 = Forge AI, gen-1 = first trained agent — "generation" tracks the
agent training lineage, not features.
