# CLI contract: `python -m draft train-draft-agent-online`

**Feature**: 021-draft-online-grpo | Authority:
[`specs/2026-08-04-draft-agent-gen3-online-grpo.md`](../../2026-08-04-draft-agent-gen3-online-grpo.md) § 8,
[spec.md](../spec.md) FR-001…FR-029.

Runs the online, critic-free GRPO loop in one process: generate
`--drafts-per-round` fresh drafts from the current policy → one pass of
`−A·logπ_T(a|s)` → discard → regenerate. Runs until `--max-rounds` or Ctrl-C.

## Flags

### Agent wiring (three categories — spec FR-003)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--learner NAME=PATH` | `LABEL=PATH` | *(required, exactly one)* | The agent under training. `NAME` is its mix label; `PATH` warm-starts the policy at round 0. Its weights update every round, in memory. A bare `PATH` is **not** accepted — the label is load-bearing. |
| `--frozen NAME=PATH` | repeatable `LABEL=PATH` | *(none)* | An untrained reference bound to a checkpoint (e.g. the gen-1 anchor). Never enters the gradient. |
| `--anchor NAME` | `LABEL` | sole `--frozen` label | Which frozen label is the anchor-margin baseline. Required only when more than one `--frozen` is given. |
| `--mix "label:weight,…"` | mix spec | `gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` | Per-seat categorical over learner + frozen + Forge built-ins (`forge-full`, `forge-r30`, `forge-r100`). Weights are relative. |

Every generated pod carries **≥1 learner seat**: the worker forces one seat to
the learner label when the independent draw produces none
([rollout-stream.md](rollout-stream.md)).

### Deck building & reward (same meaning as `generate-draft-data`)

| Flag | Default | Meaning |
|---|---|---|
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer → each seat's `deck_score` = the reward. Must exist. |
| `--build-method {greedy,picker}` | `greedy` | Pool → deck before scoring. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Used only with `--build-method picker`; must exist then. |
| `--cards-path` | `output/cardsfolder/` | `.npz` embedding cache; fixes the model/scorer input width. |

### Rollout & optimisation

| Flag | Default | Meaning |
|---|---|---|
| `--agent-temp "LABEL=T,…"` | *(required, no default)* | Comma-separated per-label sampling temperatures. An omitted label plays argmax (`T = 0`). MUST name the `--learner` with a value > 0 — that value is also what **every** policy distribution (logπ, entropy, KL) is evaluated at. May name `--frozen` labels to sample them. Naming a Forge built-in or an unknown label, or any negative value, exits 2. |
| `--lr` | `1e-4` | AdamW learning rate. |
| `--drafts-per-round` | `10` | Fresh drafts generated and trained on (one pass) per round. Larger rounds raise margin precision and cut its lag in rounds, at the same drafts/hour. |
| `--anchor-window` | `100` | Sliding window (drafts) backing the anchor margin. Divided by `--drafts-per-round`, it is the window length in rounds — which bounds every run-control knob below. |
| `--snapshot-every` | `25` | Rounds between timestamped checkpoint snapshots (`latest.pt` every round; `best_*.pt` on each new best). |
| `--max-rounds` | *(none)* | Optional round budget; omitted ⇒ runs until Ctrl-C. |
| `--set` | *(none)* | Restrict every draft to one set; else a random sealed-legal set per draft. |
| `--output-path` | `output/draft/drafts.jsonl` | Corpus appended with every generated draft (shared file, clarified 2026-08-05). |
| `--seed` | `42` | Torch/numpy init, batch shuffling, pick-sampling RNG. Forge-side randomness is **not** seeded. |
| `--batch-size` | `32` | States per gradient step; fixed-and-forget (8 GB VRAM budget). |
| `--max-grad-norm` | `1.0` | Per-parameter-group gradient-norm cap. |
| `--warmup-steps` | `200` | Linear LR ramp over the first N optimizer steps of the run, then constant. |
| `--max-consecutive-faults` | `5` | Abort after this many consecutive pick-fault-abandoned drafts. |

### Run control (all opt-in; disabled by default — spec FR-033…FR-035)

| Flag | Default | Meaning |
|---|---|---|
| `--patience` | *(none; disabled)* | Stop after N consecutive rounds with no new best anchor margin. Must exceed `--anchor-window / --drafts-per-round`. |
| `--lr-decay-patience` | *(none; disabled)* | Rounds without a new best before the LR is annealed. Must satisfy `window_rounds < N < --patience` (the upper bound only when `--patience` is armed). |
| `--lr-decay-factor` | `0.1` | LR multiplier per decay. |
| `--min-lr` | `lr × 1e-3` | Annealing floor. An armed `--patience` can only fire here, since each decay resets the shared stall counter. |

**Not offered** (spec FR-006 / Out of Scope): `--value-weight`, `--gae-lambda`,
`--kl-coef`, `--entropy-coef` and any **loss-coefficient** schedule,
`--val-fraction`, `--epochs`, `--resume`, `--pick-mode` (fixed per category —
the learner samples at its `--agent-temp`, every other label plays argmax unless
the map names it; research D5),
`--num-workers` (one resident worker suffices).

## Exit codes

| Code | Condition |
|---|---|
| `0` | `--max-rounds` reached, the armed `--patience` stall stop firing, or Ctrl-C after a clean shutdown (final summary printed) |
| `1` | `--max-consecutive-faults` consecutive abandoned drafts |
| `2` | Startup validation failure (data-model § 1.1), missing file, bad flag value |
| `6` | Checkpoint architecture error (`DraftAgentArchitectureError`) |
| `130` | Interrupted before the loop started |

All validation runs **before** the Forge worker launches and before any update
(spec FR-024, SC-006).

## stdout contract

Timestamped `[YYYY-MM-DD HH:MM:SS] ` prefix on every line (the project's
`_log`). Lines are flushed as they are written so `... | tee run.log` is live.

### 1. Startup echo (FR-013) — once, before the worker launches

```
Online GRPO run <run_id>: generation 1 -> 2
  learner   : gen-3 <- models/draft/agent/gen1/latest.pt
  frozen    : gen-1 <- models/draft/agent/gen1/latest.pt  (anchor)
  mix       : gen-3:5,gen-1:3,forge-r30:1,forge-r100:1  (>=1 learner seat forced)
  reward    : scorer models/sealed/scorer/latest.pt | build-method greedy
  rollout   : T=2.0 | drafts/round=10 | set=BLB | seed=42 (Forge-side rollouts unseeded)
  optimiser : lr=1e-04 batch=32 clip=1.0 warmup=200 steps
  run ctrl  : patience 30 rounds | lr decay x0.1 after 15 rounds, floor 1e-07
  runtime   : device cuda | embedding width 528 | anchor window 100 drafts (10 rounds, ~5-round lag)
  outputs   : corpus output/draft/drafts.jsonl (append) | checkpoints models/draft/agent/ (snapshot every 25 rounds)
```

The `generation N -> N+1` figure is the **lineage counter** written to
`rl_metadata["generation"]` — `base_checkpoint's generation (1 when the base has
no rl_metadata) + 1`, exactly as the gen-2 trainer computes it. It is independent
of the `--learner` mix label: the run above is labeled `gen-3` in the mix but
warm-starts from a gen-1 base, so its counter is `2`. Basing the same run off a
gen-2 checkpoint prints `generation 2 -> 3`.

### 2. Per-round block (FR-014…FR-018) — one summary line + four detail lines

```
round 12 | drafts 10 (120) | picks 1789 (2 seats dropped) | gen 74s train 38s | R +0.118+-1.842 | |A|<0.1 18% | ppl 2.41 | KL 0.0042 | margin +0.372
  reward   : learner seats=41 R mean=+0.118 std=1.842 | A std=1.000 |A|<0.1=18.2% |A|>0.5=61.0% max|A|=2.31
  explore  : H=0.881 ppl=2.413 off-argmax=31.4%   (band: ppl 2-3 / off-argmax 25-40%)
  movement : mean logpi=-1.204 policy_loss=-0.0153 grad_norm=0.83 KL(prev||new)=0.00421 KL(init||new)=0.0842 lr=1.0e-04
  progress : anchor margin=+0.372 | gen-3=6.412 gen-1=6.040 forge-r30=5.101 forge-r100=3.884 | window=100 drafts
```

Required properties (SC-003, SC-004):

- Exactly one summary line and four detail lines per round, every round.
- `reward` carries the near-zero-advantage fraction → "nothing to learn" is
  readable.
- `explore` carries entropy, perplexity and the off-argmax rate → collapse
  (`ppl → 1`, `off-argmax → 0`) is readable.
- `movement` carries the pre-clip gradient norm, **both** KLs — to the previous
  round and to the run's warm start — and the current learning rate. An
  over-large step is readable from `KL(prev||new)`; a step size too small to move
  the policy at all is readable only from `KL(init||new)` staying flat, since
  small consistent steps accumulate and a near-zero per-round KL does not imply a
  stationary policy. Both remain readable once annealing starts moving the LR.
- `progress` carries the margin, every label's raw windowed mean, and the window
  size → progress/no-progress is readable without pausing training.

A degenerate round replaces the four detail lines with one line and takes no
optimizer step (FR-023):

```
round 12 | drafts 10 (120) | skipped (no signal): 1 surviving learner reward | margin +0.372
```

### 3. Incidental lines

- Worker restart, abandoned draft, and pick-fault lines are inherited verbatim
  from the live-play supervisor.
- Checkpoint writes: `saved models/draft/agent/latest.pt (round 12)`; on the
  snapshot cadence, `snapshot models/draft/agent/20260805_141233.pt (round 25)`;
  and on each new best,
  `best models/draft/agent/best_20260805_141233.pt (round 12, margin +0.372)`.
- LR decay: `LR decay #1 -> 1.00e-05 after 15 rounds without a new best margin`.

### 4. Final summary (FR-019) — on `--max-rounds`, Ctrl-C, or fault abort

```
Done after 137 rounds | 1370 drafts | 24451 learner picks | 4h12m
  latest checkpoint : models/draft/agent/latest.pt
  final snapshot    : models/draft/agent/20260805_181940.pt
  best checkpoint   : models/draft/agent/best_20260805_172204.pt
  best anchor margin: +0.514 at round 108 (current +0.489)
```

When the anchor window never filled, the last two lines report that no best was
recorded rather than naming a checkpoint chosen on partial evidence (FR-033).

## Behavioural contract

1. **On-policy by construction** — the learner seats of round *k* are piloted by
   the weights that round *k*'s pass updates; the batch is used once and dropped
   (FR-011, FR-012, SC-008).
2. **One resident worker** — the Forge JVM starts once per run and is driven every
   round; a crash restarts it and the run continues (FR-005).
3. **Continuity** — optimizer state and the LR schedule live in memory for the
   whole run; a single warmup at the start, then constant unless plateau
   annealing moves it (FR-035). A decay changes the schedule only: warmup is not
   re-run and the optimiser moments are not reset. Checkpoints are snapshots, not
   training state (FR-025, FR-026).
4. **Frozen anchor** — the anchor label's service is never updated and never
   re-bound during a run (FR-021).
5. **No automatic stop by default** — there is no held-out loss, and nothing
   stops the run unless the operator arms `--patience`. Promotion is always an
   operator judgment on the yardstick (FR-026, FR-030, FR-034).
5b. **Best selection is advisory** — `best_*.pt` is written on each new anchor
   margin best once the window is full, and never before. It nominates a
   candidate; the recorded best margin is an optimistic maximum and is not a
   measurement (FR-033). Annealing never rolls back to it (FR-035).
6. **Ctrl-C** — finishes the in-flight round's bookkeeping, writes a final
   snapshot, prints the final summary, terminates the worker, exits `0`.
