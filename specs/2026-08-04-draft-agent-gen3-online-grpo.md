# Draft agent — online self-play GRPO fine-tuning (generation 3)

Normative spec: the new `train-draft-agent-online` command, its streaming
self-play loop, the GRPO update, the per-round diagnostics, and the CLI. Rationale
and the gen-2 post-mortem live in
[`../experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md`](../experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md)
and
[`../experiments/2026-06-10-draft-agent-gen2-design.md`](../experiments/2026-06-10-draft-agent-gen2-design.md).

**Generations** (agent lineage, not a phase number):

| gen | what |
|-----|------|
| gen-0 | Forge AI |
| gen-1 | offline imitation + critic ([`2026-05-28`](2026-05-28-draft-agent.md)) |
| gen-2 | offline RL ([`2026-06-10`](2026-06-10-draft-agent-gen2-rl.md)) |
| **gen-3** | this feature — a base checkpoint (gen-1 or gen-2) fine-tuned online |

"gen-3" below means "the next generation" generically.

# 1. What it does

A single command fine-tunes the draft agent with online, critic-free, GRPO
reinforcement learning on its own self-play. Each round:

- generate a small batch of fresh drafts from the current policy;
- take one gradient pass over that batch; discard it;
- regenerate from the updated policy; repeat.

Every update is on-policy by construction. The loss is a single term — no critic,
GAE, KL anchor, or entropy bonus. Progress is read live from a printed anchor
margin (§ 6); promotion uses the gen-2 yardstick (§ 7). The deliverable is a gen-3
checkpoint that beats its base (and Forge) on that yardstick.

# 2. Scope

**In scope**

- The `train-draft-agent-online` command: owns the loop, drives one resident Forge
  worker, applies the update, writes checkpoints (§ 5, § 9).
- The critic-free GRPO update contract (§ 4).
- Per-round stdout diagnostics + the live anchor margin (§ 6).

**Out of scope** (reused unchanged, or deferred)

- Model architecture, typed-token state, recency scheme, `.npz` cache,
  `drafts.jsonl` schema — reused from gen-1 / live-play.
- Forge draft-driving (worker protocol, pick service, state reconstruction), the
  deck builder, and the scorer — reused; the loop changes only worker *lifetime*
  (§ 5).
- The gen-2 offline trainer `train-draft-agent-rl` — untouched; gen-3 is a
  separate command.
- The gen-2 yardstick — reused as-is; no new evaluation code.
- Deferred / excluded: automated promotion (stays manual), any corpus-schema or
  provenance change, in-draft critic use, PPO / vine advantage estimation, encoder
  / picker / scorer training, the Bradley-Terry "beat the pod" objective, and
  shipping the agent into the Forge client.

# 3. How it works

```
   base checkpoint π₀  (gen-1 or gen-2, an operator input; round 0 only)
            │
            ▼
   START a long-lived Forge draft worker ONCE  (~20 s JVM startup, paid here)
     pick services:  learner label → CURRENT policy πₖ  (updated every round)
                     anchor  label → FROZEN gen-1 agent (never updated)
            │
            ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  ROUND k   (worker stays resident — no JVM restart)            │
   │                                                                │
   │   drive the resident worker → ~10 fresh drafts, sample T      │
   │      πₖ pilots the LEARNER seats; frozen + Forge bots the rest │
   │      each seat's pool → build (--build-method) → score         │
   │      (--scorer-checkpoint) → deck_score = reward              │
   │            │                                                   │
   │            ▼  fresh batch = πₖ's own picks (consumed once)     │
   │   one pass:  L = − A · logπ_T(a|s)                             │
   │      A = standardise_over_round(pod-relative LOO deck_score)   │
   │            │                                                   │
   │            ▼  πₖ → πₖ₊₁   (optimizer + LR stay in RAM;         │
   │                 push πₖ₊₁ into the learner pick service)       │
   │   print round diagnostics (§ 6)  +  save latest.pt            │
   │      (+ best_*.pt on a new best anchor margin — § 6.1)         │
   └───────────────────────────────────────────────────────────────┘
            │  repeat: next round's learner seats are piloted by πₖ₊₁
            ▼
   watch the live anchor margin (§ 6); on a plateau, run the yardstick (§ 7)
```

- **On-policy by construction** — each round trains on the current policy's own
  fresh batch; no stale corpus, no provenance bookkeeping.
- **One resident worker** — started once, driven every round; the served policy is
  swapped between rounds. One worker suffices (drafts play no games).
- **New code** = this command: the loop, the persistent-worker driver, the
  single-term update, and the diagnostics. Everything else is reused.

# 4. The GRPO update contract

Input each round: the current checkpoint and the fresh batch it just generated in
sample mode at temperature `T`. Per learner seat, per pick `t` (a learner seat is
one whose mix label is the `--learner` label; § 8):

| Quantity | Definition |
|----------|------------|
| state `sₜ` | the typed-token state at `(seat, pack, pick)`, reconstructed exactly as gen-1's loader builds it. |
| action `aₜ` | the card recorded as taken at this pick. |
| reward `R` | the seat's pod-relative leave-one-out `deck_score` (the `--scorer-checkpoint`'s score on the seat's built deck) — `deck_score_seat − mean({other non-failed seats' deck_score})`. Terminal (γ=1): the same scalar for all 45 of the seat's picks. |
| advantage `Aₜ` | the batch-standardised reward — the round's learner-seat rewards centred and scaled to unit variance. No critic, no GAE. Shared across the seat's picks. |
| policy weight | `logπ_T(aₜ given sₜ)` — the trained policy's log-prob of the taken action, at temperature `T`. |

**Loss** (per learner pick, averaged over the round):

```
L = − Aₜ · logπ_T(aₜ | sₜ)
```

- The whole loss. No critic/value term, GAE, KL anchor, or entropy bonus.
- Exploration comes only from sampling at `T`; the pod-relative reward is the
  RLOO/group baseline (the pod is the group), so variance reduction lives in the
  reward, not a critic.
- One pass per round; the batch is discarded, never reused.
- Optimiser: AdamW, per-group max-norm clip, warmup-then-constant LR — with
  optimiser + LR state continuous across rounds (one warmup at run start).

**Failure / integrity**

- Failed build (`deck = []`, `deck_score = null`) → seat excluded from reward, pod
  mean, and gradient.
- Pod with no other non-failed seat → that learner's LOO reward is undefined →
  seat excluded.
- Round with fewer than two surviving learner rewards, or zero variance among them
  → safe no-op, logged as skipped/no-signal, never a divide-by-zero.

# 5. The online loop

The loop is owned by one in-process command. At startup it launches a single Forge
worker and binds two pick services: the learner label to the live in-training
policy (`--learner`, updated every round) and the anchor label to a frozen agent
(`--frozen` / `--anchor`, never updated). Each round:

1. **Generate** — drive the resident worker for `--drafts-per-round` fresh drafts.
   The current policy pilots the learner seats, **sampled at `T`**; frozen agents
   pilot theirs at **argmax**; Forge/random seats fill the rest (`--mix`, § 8.1).
   Each seat's pool is built (`--build-method`) and scored
   (`--scorer-checkpoint`) → `deck_score`.
2. **Update** — one-pass GRPO over the learner picks (§ 4): πₖ → πₖ₊₁.
3. **Swap** — push πₖ₊₁ into the learner pick service (frozen services untouched).
4. **Log + save** — print diagnostics (§ 6), write `latest.pt` (+ periodic
   snapshot), repeat.

Rules:

- **On-policy** — learner seats always draft with the current weights, regenerated
  every round.
- **Continuity** — optimiser + LR schedule live in memory across rounds (one
  warmup at run start); checkpoints are for snapshot/recovery only, not to carry
  training state.
- **Stop** — runs until Ctrl-C or `--max-rounds`. There is no held-out-loss guard.
  `--patience` (off by default) adds an opt-in stop on a stalled anchor margin
  (§ 6.1); with it unset the loop never stops itself. When LR decay is also on
  (§ 6.2) a stall anneals the LR before it stops the run, so an opt-in stop fires
  only at the LR floor.
- **Frozen anchor** — the anchor stays fixed for the whole run; never swapped to a
  later generation or "previous round".
- **Worker crash** — a crashed worker JVM is restarted (startup paid once more)
  and the run continues.

# 6. Diagnostics & live progress

The single-term loss is not a progress signal, so the run is steered by printed
diagnostics. Every round emits four axes to stdout — enough to diagnose progress,
no-progress, and collapse from the log alone:

| Axis | Printed | Catches |
|------|---------|---------|
| **Reward** | pod-relative reward mean/std; advantage spread + fraction near-zero (`abs(A) < 0.1`) | reward not discriminating picks → advantages → 0 (nothing learned) |
| **Exploration** | entropy, perplexity `exp(H)`, off-argmax rate | perplexity → 1 / off-argmax → 0 / entropy → 0 = exploration collapse |
| **Movement** | mean `logπ`, mean policy-loss term, gradient norm (pre-clip), KL-to-previous-round, KL-to-run-start, current LR | step too large for the round size → large per-round KL; a step size that is doing nothing at all → flat KL-to-run-start |
| **Progress** | the anchor margin + raw component means (learner / anchor / Forge) + window size | whether the learner is pulling away from the fixed reference |

**Anchor margin** — the live progress signal:

```
anchor_margin = mean(learner deck_score) − mean(frozen-anchor deck_score)   over the last --anchor-window drafts
```

- Anchor frozen ⇒ a progress curve against a fixed point (climbs, then
  plateaus); `learner − Forge` comes free.
- **It does not start at zero.** The learner samples at `T`, the anchor plays
  argmax (§ 8.1), so the margin opens negative by the learner's sampling
  handicap. Round 0 measures that offset exactly — learner and anchor begin from
  identical weights — and crossing zero means the learner has genuinely overtaken
  a properly-playing anchor.
- **It amplifies.** Seats compete for one pool of cards, so a learner that
  improves also takes cards its podmates would have had: the learner rises *and*
  the anchor falls. The margin moves at roughly twice the underlying skill gap.
  The yardstick (§ 7) co-seats too and shares this, so the two agree — but
  neither is an absolute measure of how much better one agent is.
- **It is not comparable across runs** that change `-T`, `--mix`, or the § 8.1
  pick modes. Each such change starts a new scale.
- Computed from `deck_score`s already in hand — no extra command; also
  recomputable post-hoc via `analyze-generated-decks`.
- Guideline exploration band: perplexity ≈ 2–3 / off-argmax ≈ 25–40 %; watch it
  across rounds (entropy decays even at fixed `T`).

**The two KLs** — the movement axis reports both, because one round's step and a
run's cumulative travel are different questions and a step size can fail in
either direction:

| | Measures | Reads wrong when |
|---|---|---|
| `KL(π_{k} ‖ π_{k+1})` | this round's step | large / erratic ⇒ the step is too big for the round size |
| `KL(π_0 ‖ π_{k})` | total distance from the run's warm start | flat ⇒ the LR is too small to move the policy at all |

Neither alone is sufficient. A per-round KL near zero does **not** prove the
policy is standing still: small steps taken in a consistent direction accumulate,
and only the distance from `π_0` shows it. Conversely a healthy per-round KL with
a cumulative KL that has stopped growing means the steps are cancelling rather
than compounding. `π_0` here is the learner's own warm-start checkpoint, not the
anchor — the two coincide only when `--learner` and `--frozen` name the same file.

Also printed:

- a consolidated round-summary line (round index; draft + learner-pick counts;
  generation + training wall-clock; dropped-seat count; headline figures);
- a startup echo of the resolved config + validation results, including the
  derived window geometry (§ 6.1) — window length in rounds and the implied lag;
- a final summary at run end / interrupt (rounds, total drafts, latest checkpoint,
  best-checkpoint path, best anchor margin and when).

A plateau in the anchor margin triggers a yardstick check (§ 7) — it is not the
promotion decision, and it stops the run only if `--patience` is set (§ 6.1).

## 6.1 Window geometry, best checkpoint, and patience

The anchor margin is both the live progress read and the quantity that selects
the best checkpoint, so its sampling properties are part of the contract.

**Window geometry.** The margin is averaged over `--anchor-window` drafts, which
spans `--anchor-window / --drafts-per-round` rounds. Two consequences follow, and
both are reported at startup:

| Property | Consequence |
|---|---|
| Precision | Standard error falls with the window's draft count. Learner and anchor seats share pods, so set-power variance cancels — it is a paired comparison. |
| Lag | The margin trails the current policy by about **half the window, measured in rounds**. A margin peak at round `k` means the policy peaked near round `k − ½·window_rounds`. |

Lag is governed by the window's length *in rounds*, not in drafts, so a larger
`--drafts-per-round` buys precision and low lag together. Rounds are also better
conditioned when larger (more learner seats per standardisation).

**Best checkpoint.** The run tracks the best anchor margin and writes
`best_{timestamp}.pt` on each new best (§ 9). Two rules keep the selection honest:

- **Window-full guard** — no best is recorded until the window holds a full
  `--anchor-window` drafts. Before that the margin is computed over fewer drafts
  and an early lucky round would otherwise pin `best` for the whole run. A run
  that ends before the window fills reports no best rather than a spurious one.
- **Selection is optimistic** — `best` is a maximum over a correlated series, so
  it overstates the true peak. It selects a checkpoint; it is not an unbiased
  estimate of that checkpoint's strength. The yardstick (§ 7) remains the
  measurement.

Because of the lag, periodic snapshots (`--snapshot-every`) stay useful even with
`best_*.pt`: the genuinely best policy may sit a few rounds *before* the recorded
best, and a snapshot is how it is recoverable.

**Patience.** `--patience N` (unset ⇒ disabled) stops the run after `N`
consecutive rounds with no new best margin, writing the final snapshot and summary
like any other termination (§ 5). It is a convenience for unattended runs, not a
quality verdict — promotion still goes through the yardstick.

`N` MUST exceed the window length in rounds (`--anchor-window /
--drafts-per-round`), or the run fails fast: a genuine improvement needs a full
window to propagate into the margin, so a shorter patience can stop before the
evidence arrives.

**The stall counter.** Patience and LR decay (§ 6.2) share one counter: rounds
since the last new best anchor margin. It starts once the first best is recorded
(i.e. once the window fills), resets on every new best, and resets again on each
LR decay.

## 6.2 LR decay on plateau

A stalled margin is treated first as a step-size problem and only then as a
stopping condition — the gen-1 trainer's plateau-annealing convention, in rounds.

`--lr-decay-patience N` (unset ⇒ disabled) arms it. When the stall counter
(§ 6.1) reaches `N`, the learning rate is multiplied by `--lr-decay-factor` and
the counter resets, giving the smaller step a fresh window to find a new best.
Decays continue while the next one would stay at or above `--min-lr`; at the floor
no further decay happens.

| Flag | Default | Meaning |
|------|---------|---------|
| `--lr-decay-patience` | _(none; disabled)_ | Rounds without a new best margin before the LR is annealed. |
| `--lr-decay-factor` | `0.1` | Multiplier applied per decay. |
| `--min-lr` | `lr × 1e-3` | Floor; no decay is taken that would land below it. |

Rules:

- **Decay pre-empts stopping.** Because each decay resets the counter, an armed
  `--patience` cannot fire until the LR floor is reached — so an unattended run
  anneals its way down before it gives up, rather than stopping at the first
  plateau.
- **Ordering.** `anchor_window / drafts_per_round < --lr-decay-patience <
  --patience`, else fail fast. The lower bound is § 6.1's: annealing on evidence
  the margin has not had time to show is just noise-chasing. The upper bound is
  what makes the previous rule hold.
- **Independent of `--patience`.** LR decay may be armed with stopping disabled:
  the run then anneals to the floor and keeps training until Ctrl-C.
- **Schedule composition.** The decay multiplies the post-warmup constant LR.
  Warmup is not re-run and the optimiser's moments are not reset — a decay is a
  schedule change, not a restart.
- **No rollback.** Decay continues from the current weights; it never reloads
  `best_*.pt`. The margin's best is an optimistic maximum (§ 6.1), so returning to
  it repeatedly would chase selection noise rather than progress.
- **Not restored across runs.** The decay count is recorded in the checkpoint as
  provenance (§ 9), but gen-3 has no `--resume`: restarting from a snapshot
  deliberately re-runs warmup and resets both the optimiser moments and the decay
  position.

The current LR is printed on the movement axis (§ 6) beside the gradient norm and
KL, and each decay is logged as it happens — step size, its effect, and the
response to it stay on one screen.

# 7. Yardstick + promotion

Reused from gen-2 (§ 6 of [`2026-06-10`](2026-06-10-draft-agent-gen2-rl.md)) — no
new code:

- One greedy (deterministic) run, one fixed agent mix (candidate + base + Forge +
  random-bot minority), randomly co-seated over many pods.
- Metric: per-agent mean `deck_score` (frozen scorer's raw scale), from
  `analyze-generated-decks --agent <each>`.
- Promote iff the candidate beats its base beyond the run-to-run noise band — a
  manual judgment; no auto-promote.
- Composition (colours, curve, types) inspectable with the same command
  (descriptive only).

# 8. CLI

`python -m draft train-draft-agent-online`. Rollout generation reuses live-play's
Forge draft worker — a single worker kept resident across rounds (§ 5), not a
per-round `generate-draft-data` subprocess.

**Agent wiring** (three categories, one flag prefix each — § 8.1):

| Flag | Default | Meaning |
|------|---------|---------|
| `--learner NAME=PATH` | _(required; exactly one)_ | The agent under training. `NAME` = its mix label; `PATH` = warm-start checkpoint. Weights update every round, held in memory. |
| `--frozen NAME=PATH` | _(repeatable)_ | A frozen (untrained) reference bound to a checkpoint — e.g. the gen-1 anchor. |
| `--anchor NAME` | _(sole `--frozen` label)_ | Which frozen label is the anchor-margin baseline (§ 6). Required only with >1 `--frozen`. |
| `--mix "label:weight,…"` | `gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` | Per-seat pod composition over learner + frozen + Forge labels. |

**Deck building & reward** (same inputs as `generate-draft-data`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer → each deck's `deck_score` (the reward). Required. |
| `--build-method` | `greedy` | Pool → deck before scoring: `greedy` (SA) or `picker`. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Used only when `--build-method picker`. |
| `--cards-path` | `output/cardsfolder/` | The `.npz` embedding cache (fixes model/scorer input width). |

**Rollout & optimisation:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--rollout-temperature` / `-T` | _(required; no default)_ | Sample temperature; all policy distributions (logπ, entropy) use it. Positive, or fail fast. |
| `--lr` | _(tunable)_ | Step size. Base value for the warmup ramp and for any plateau annealing (§ 6.2). |
| `--drafts-per-round` | `10` | Fresh drafts generated + trained on (one pass) per round. Larger rounds raise margin precision, cut the margin's lag in rounds, and condition the advantage better, at the same drafts/hour (§ 6.1). |
| `--anchor-window` | `~100` | Sliding window (drafts) for the anchor margin. Sets both its precision and — divided by `--drafts-per-round` — its lag (§ 6.1). |
| `--patience` | _(none; disabled)_ | Stop after N consecutive rounds with no new best anchor margin. Must exceed `--anchor-window / --drafts-per-round`, else fail fast (§ 6.1). |
| `--lr-decay-patience` | _(none; disabled)_ | Rounds without a new best margin before the LR is annealed. Must sit between the window length in rounds and `--patience` (§ 6.2). |
| `--lr-decay-factor` | `0.1` | Multiplier per decay. |
| `--min-lr` | `lr × 1e-3` | Annealing floor; stopping (if armed) fires only here. |
| `--snapshot-every` | _(N rounds)_ | Cadence of timestamped snapshots (besides per-round `latest.pt` and `best_*.pt` on each new best). |
| `--max-rounds` | _(none; until Ctrl-C)_ | Optional round budget. |
| `--set` | _(none; random set per draft)_ | Restrict rollouts to one set. |
| `--output-path` | `output/draft/drafts.jsonl` | Corpus every generated draft is appended to (shared file, always opened in append mode). |
| `--seed` | `42` | Torch/numpy init, per-round batch shuffling, pick-sampling RNG. Forge-side rollout randomness is not seeded. |
| `--max-consecutive-faults` | `5` | Inherited pick-fault abort: this many consecutive abandoned drafts ends the run nonzero. |
| `--warmup-steps` | `200` | Linear LR ramp over the first N optimizer steps of the **run**, then constant. An online run has no total step count, so the ramp is expressed in steps, not a fraction. |
| `--batch-size`, `--max-grad-norm` | _(gen-1 defaults; fixed)_ | Batch stays within the 8 GB VRAM budget. |

No `--gae-lambda` / `--kl-coef` / `--entropy-coef` / `--value-weight` — those
pieces are dropped, and with them gen-2's val-keyed *coefficient* schedules: there
are no loss-term coefficients left to schedule. The LR schedule (warmup, and
plateau annealing under § 6.2) is a separate thing and is kept.

## 8.1 Agent categories — `--learner`, `--frozen`, `--anchor`, `--mix`

A mix label names a kind of seat. Every label is exactly one of three categories:

| Category | Flag | Plays at | Notes |
|----------|------|----------|-------|
| **Forge-piloted** | _(none)_ | Forge's own logic | Built-ins `forge-full` (pure Forge AI), `forge-r30` / `forge-r100` (30 % / 100 % of picks made uniformly random). Just name them in `--mix`. Their randomisation is internal to Forge and unaffected by `T`. |
| **Frozen** | `--frozen NAME=PATH` (repeatable) | **argmax** | Untrained reference bound to a checkpoint; the gen-1 anchor is one. `--anchor` picks which is the margin baseline (defaults to the sole frozen agent). |
| **Learner** | `--learner NAME=PATH` (exactly one) | **sampled at `T`** | The policy being trained; weights held in memory, updated each round. One learner only. |

**Why only the learner samples.** Sampling is the learner's sole exploration
mechanism, so it must. A frozen agent must not, because a draft allocates one
fixed pool of cards: a card an agent wrongly declines does not vanish, it flows
downstream to its podmates. A sampled frozen agent therefore **hands the learner
cards a properly-playing agent would have kept**, making the training field weak
in precisely the dimension drafting is about — what wheels, and what is still
there when the pack comes back — and unlike the evaluation field (§ 7), where
every agent plays argmax.

`--mix "label:weight,…"` — each of the pod's seats is drawn independently from this
categorical (weights are relative, not exact counts).
`gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` ≈ 50 % / 30 % / 10 % / 10 % per seat, so
an 8-seat pod averages ~4 learner, ~2–3 anchor, ~1 of each random bot.

At least one learner seat per pod is guaranteed: because seats are drawn
independently a pod could come up learner-free (no on-policy picks, useless), so a
learner-free draw is resampled (or one seat forced to the learner) before the draft
is played. A high learner weight makes this rare.

**Startup validation** (fail fast):

- every `--mix` label is a Forge built-in, a `--frozen` label, or the `--learner`
  label;
- the `--learner` label appears in `--mix`;
- `--anchor` names a `--frozen` label in the mix (omit with a single `--frozen`);
- the `--learner` `PATH` and `--scorer-checkpoint` exist (and `--picker-checkpoint`
  when `--build-method picker`); `--learner` architecture matches the `.npz` width;
  `--rollout-temperature` is supplied and positive.

### Example — fine-tune gen-3 from gen-1, anchored to gen-1

```
python -m draft train-draft-agent-online \
  --learner  gen-3=models/draft/agent/gen1/latest.pt \   # agent under training (warm-started from gen-1)
  --frozen   gen-1=models/draft/agent/gen1/latest.pt \   # frozen reference — the anchor
  --anchor   gen-1 \                                      # optional: sole --frozen, so it defaults here
  --mix      gen-3:5,gen-1:3,forge-r30:1,forge-r100:1 \   # ~50/30/10/10 per seat
  --scorer-checkpoint models/sealed/scorer/latest.pt \   # frozen scorer → deck_score = reward
  --build-method greedy \
  -T 2.0  --lr 1e-4  --drafts-per-round 10  --set BLB
```

The gen-1 checkpoint appears twice — as the learner's warm-start and as the frozen
anchor. They start identical; the learner evolves each round while the anchor stays
put, so the anchor margin reads as improvement over a fixed point.

### Example — base off gen-2, still anchored to gen-1

```
python -m draft train-draft-agent-online \
  --learner  gen-3=models/draft/agent/gen2/latest.pt \   # start the learner from gen-2's weights
  --frozen   gen-1=models/draft/agent/gen1/latest.pt \   # anchor stays the frozen gen-1
  --mix      gen-3:5,gen-1:3,forge-r30:1,forge-r100:1 \
  --scorer-checkpoint models/sealed/scorer/latest.pt \
  --build-method greedy \
  -T 2.0  --lr 1e-4  --drafts-per-round 10
```

Only the `--learner` warm-start changes; `--anchor` defaults to the sole `--frozen`
label. Any consistent label set works as long as the validation rules hold.

# 9. Records & artifacts

- **Corpora** — the unchanged `drafts.jsonl` (live-play § 7); no schema or
  provenance change. The loop appends each round's drafts (for post-hoc
  margin/composition); training uses each batch once, then discards it.
- **Checkpoints** — three kinds under `models/draft/agent/`:

  | File | Written | Purpose |
  |------|---------|---------|
  | `latest.pt` | every round | Current policy; overwritten. Tools defaulting to it pick up the in-progress agent mid-run. |
  | `best_{timestamp}.pt` | on each new best anchor margin (§ 6.1) | The selection candidate. Not written before the window fills. |
  | `{timestamp}.pt` | every `--snapshot-every` rounds, and once at run end / interrupt | Periodic recovery, and the way to reach a policy a few rounds *before* the recorded best (§ 6.1 lag). |

  All three use the gen-1 format (trunk + policy + critic + recency/context
  tables, `config`, round counters) plus gen-3 `rl_metadata`: generation index,
  base-checkpoint identity, `algorithm=online-grpo`, and `lr` /
  `rollout_temperature` / `drafts_per_round`, plus the decay count reached (§ 6.2)
  as provenance — it records how far a run annealed, and is not restored, since
  there is no `--resume`. The critic head is carried unchanged (untrained,
  unused); no critic / GAE / KL / entropy params; no encoder weights (Phase A).
  No checkpoint carries a held-out loss — there is none.
