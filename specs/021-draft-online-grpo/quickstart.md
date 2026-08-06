# Quickstart: Online GRPO self-play (Generation 3)

**Feature**: 021-draft-online-grpo | **Date**: 2026-08-05

One long-running command replaces gen-2's manual generate → train → repeat cycle.
You start it, watch four diagnostic lines per round, and pause it when the anchor
margin plateaus to run the (unchanged) cross-generation yardstick. Promotion
stays your judgment.

## Prerequisites

- A base draft-agent checkpoint to fine-tune (gen-1 or gen-2), e.g.
  `models/draft/agent/gen1/latest.pt`.
- `output/cardsfolder/` populated with `.npz` embeddings from the **same** encoder
  the agent was trained on (`python -m sealed encode-cards`).
- A frozen scorer (`models/sealed/scorer/latest.pt`) and, only if you use
  `--build-method picker`, a picker.
- A built sibling Forge checkout at `../forge`, and a **freshly built connector
  JAR** — gen-3 adds the `-Ddraft.required.agent` property:

  ```bash
  cd forge-connector && mvn package -DskipTests && cd ..
  ```

## Step 1 — Start the loop

```bash
python -m draft train-draft-agent-online \
    --learner gen-3=models/draft/agent/gen1/latest.pt \
    --frozen  gen-1=models/draft/agent/gen1/latest.pt \
    --mix "gen-3:5,gen-1:3,forge-r30:1,forge-r100:1" \
    --scorer-checkpoint models/sealed/scorer/latest.pt \
    --build-method greedy \
    --agent-temp "gen-3=2.0" --lr 1e-4 --drafts-per-round 10 --set BLB \
    2>&1 | tee output/draft/gen3-run.log
```

- The gen-1 checkpoint appears **twice** on purpose: once as the learner's warm
  start, once as the frozen anchor. They start identical; the learner moves, the
  anchor does not, so the margin reads as improvement over a fixed point.
- `--anchor` is omitted because there is exactly one `--frozen` label.
- `--build-method greedy` matches the corpus the gen-1 agent was trained on.
- `--agent-temp` is **required** and must name the learner. Start the learner at
  `2.0` and check the exploration band in Step 2. Every other label is omitted
  here, so the frozen anchor and the Forge bots play argmax — the default, and
  what you want unless you are deliberately experimenting with a sampled field.
- Keep the log: it is the run's provenance record (round-by-round config +
  diagnostics), and `--seed` does not make Forge-side rollouts reproducible.

Forge's ~20 s JVM startup is paid once, at launch, not per round.

## Step 2 — Read the four axes (each round)

```
round 12 | drafts 10 (120) | picks 1789 (2 seats dropped) | gen 74s train 38s | R +0.118+-1.842 | |A|<0.1 18% | ppl 2.41 | KL 0.0042 | margin +0.372
  reward   : learner seats=41 R mean=+0.118 std=1.842 | A std=1.000 |A|<0.1=18.2% |A|>0.5=61.0% max|A|=2.31
  explore  : H=0.881 ppl=2.413 off-argmax=31.4%   (band: ppl 2-3 / off-argmax 25-40%)
  movement : mean logpi=-1.204 policy_loss=-0.0153 grad_norm=0.83 KL(prev||new)=0.00421
  progress : anchor margin=+0.372 | gen-3=6.412 gen-1=6.040 forge-r30=5.101 forge-r100=3.884 | window=100 drafts
```

| What you see | What it means | What to do |
|---|---|---|
| `margin` rising over rounds | Working. | Keep going. |
| `margin` negative early | Expected, not a fault. The learner samples at its `--agent-temp` while the anchor plays argmax, so you start in the hole by the learner's sampling handicap — round 0 tells you how deep. Crossing zero means the learner has genuinely overtaken a properly-playing anchor. | Keep going. |
| `margin` flat, `\|A\|<0.1` climbing toward 100% | The reward isn't discriminating picks — nothing to learn. | Not a temperature problem. Pause and run the yardstick; consider more drafts per round. |
| `ppl` sagging toward 1, `off-argmax` toward 0 | Exploration collapse — the policy only ever samples its argmax, so the top pick can never be displaced. | Raise the learner's `--agent-temp` and restart from the latest checkpoint. |
| `KL(prev\|\|new)` large / erratic, `grad_norm` spiking | The step is too big for the round size. | Lower `--lr` (or raise `--drafts-per-round`). Watch `lr` on the same line once annealing is armed. |
| `KL(init\|\|new)` flat across many rounds | The step is too small to move the policy at all — whatever `KL(prev\|\|new)` says. | Raise `--lr`. Check this before concluding a run has plateaued: a stalled margin at a frozen policy is not a plateau. |
| `skipped (no signal)` rounds | Fewer than two surviving learner rewards, or zero variance. | Rare; if frequent, raise `--drafts-per-round` or check for failed builds. |

Entropy decays across rounds even at a fixed learner temperature, so watch the `explore` curve for
the whole run, not just round 0.

The two KLs answer different questions and you need both when sizing `--lr`:
`KL(prev||new)` is this round's step, `KL(init||new)` is how far the policy has
travelled from its warm start. A near-zero per-round KL does *not* mean the policy
is standing still — small steps in a consistent direction accumulate — so read
the cumulative one before deciding a run is stuck.

## Step 3 — Pause on a plateau

A flat anchor margin over the last ~100 drafts is the trigger to *check*, not to
stop or promote. Ctrl-C: the loop finishes its bookkeeping, writes a final
snapshot, and prints

```
Done after 137 rounds | 1370 drafts | 24451 learner picks | 4h12m
  latest checkpoint : models/draft/agent/latest.pt
  final snapshot    : models/draft/agent/20260805_181940.pt
  best anchor margin: +0.514 at round 108 (current +0.489)
```

```bash
CAND=models/draft/agent/best_20260805_172204.pt
BASE=models/draft/agent/gen1/latest.pt
```

Note that `models/draft/agent/latest.pt` tracked the in-progress gen-3 during the
run — pin `best_*.pt` (or a timestamped snapshot), not `latest.pt`, when you want
a stable candidate.

`best_*.pt` is the run's own nomination: the checkpoint at the highest anchor
margin. Two things to remember about it. The margin **lags** the policy by about
half the anchor window (in rounds), so the genuinely best policy may sit a few
rounds *earlier* — that is what the periodic snapshots are for, and it is worth
yardsticking the snapshot either side of it if the run was moving fast. And the
recorded best margin is a maximum over a noisy series, so it reads high; the
yardstick in Step 4 is the measurement, not that number.

### Optional: let the run manage itself

```bash
    --patience 30 --lr-decay-patience 15 --lr-decay-factor 0.1
```

Both are off by default. Armed, a stalled margin first anneals the LR (which
resets the stall counter) and only stops the run once the LR reaches `--min-lr` —
the same pattern that extracted extra quality from gen-1 after its first plateau.
Both must exceed the anchor window measured in rounds
(`--anchor-window / --drafts-per-round`), and `--lr-decay-patience` must be below
`--patience`; the command rejects anything else at startup. The startup echo
prints the window's length in rounds so you can size them.

## Step 4 — Cross-generation yardstick (unchanged from gen-2; no new code)

One **greedy** fixed-mix run that randomly co-seats every generation being
compared, so all face the same opponent distribution:

```bash
python -m draft generate-draft-data \
    --n-drafts 3000 \
    --agent-checkpoint cand=$CAND \
    --agent-checkpoint gen-1=$BASE \
    --agent-mix "cand:1,gen-1:1,forge-full:1,forge-r100:1" \
    --pick-mode argmax --build-method greedy \
    --output-path output/draft/yardstick-gen3.jsonl
```

```bash
for A in cand gen-1 forge-full; do
  python -m draft analyze-generated-decks \
      --drafts-path output/draft/yardstick-gen3.jsonl --agent $A
done
```

The `deck_score` mean/median/n block prints above each composition report.

## Step 5 — Promote (manual)

Promote if `cand`'s mean `deck_score` beats `gen-1`'s by more than the
run-to-run noise band — and ideally beats `forge-full`:

```bash
cp $CAND models/draft/agent/champion.pt
```

Otherwise adjust and resume: point `--learner` at the last snapshot and restart
the loop with the corrected knob (`--agent-temp`, `--lr`, or `--drafts-per-round`). There
is no `--resume`; a restart re-runs the LR warmup and resets the optimizer
moments, which is intentional (research D13).

**Keep the anchor fixed across the whole gen-3 campaign.** If you restart from a
later checkpoint, `--learner` changes but `--frozen`/`--anchor` must not — the
moment the anchor moves, the margin stops meaning "improvement over a fixed
point."

---

## Minimal smoke run (no Forge, no GPU needed for the logic)

Two rounds against a fake worker, as the integration test drives it:

```bash
pytest tests/integration/test_train_draft_agent_online_smoke.py -q
```

Expect: two rounds logged with all four axes, a loadable
`models/draft/agent/latest.pt` carrying `rl_metadata["algorithm"] ==
"online-grpo"`, and round 2 trained on drafts generated after round 1's update.
