# CLI contract: `python -m draft <subcommand>`

Two subcommands, wired in `draft/infrastructure/cli.py` with argparse
`add_parser` / `set_defaults(func=…)`, mirroring `sealed/infrastructure/cli.py`.
A third deliverable (the builder-validation diagnostic) is a **script**, not a
subcommand (FR-042).

---

## `generate-draft-data`  (FR-004 … FR-012)

Spawns a supervised Java `DraftWorkerMain`; for each completed draft builds and
scores a deck per seat and appends one JSONL record.

| Flag | Default | Meaning |
|---|---|---|
| `--n-drafts` | **required** | Number of complete drafts to generate (counts pre-existing under `--resume`). |
| `--set` | none → random per draft | Restrict all drafts to one set code; else each draft independently picks a random sealed-legal set (FR-009). |
| `--agent-mix` | `forge-full:6,forge-r30:1,forge-r100:1` | Categorical weights; each of `pod_size` seats sampled independently (FR-006). |
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer labeling decks (FR-007). |
| `--build-method` | `picker` | `picker` or `greedy` (FR-007, FR-008, §4.3). |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Picker used when `--build-method picker`; ignored otherwise (FR-008). |
| `--cards-path` | `output/cardsfolder/` | `.npz` embedding cache. |
| `--output-path` | `output/draft/drafts.jsonl` | Destination JSONL; appended; created if missing (FR-012). |
| `--resume` | off | Append + count pre-existing drafts toward `--n-drafts` (FR-012). |

Behavior contract:
- One fresh `run_id` (UUID) generated at startup, stamped on every record (FR-005).
- Worker crash ⇒ supervisor restarts and continues toward `--n-drafts` (FR-011, SC-003).
- Failed deck build for a seat ⇒ `deck=[]`, `deck_score=null`, run continues (FR-007/FR-014).
- Exit 0 on reaching the target; Ctrl-C stops cleanly (signal handling).

Success: output file gains exactly `N` additional parseable records (modulo
supervisor-crash loss), each self-contained (SC-001, SC-002).

---

## `train-draft-agent`  (FR-030 … FR-039)

Trains policy + critic jointly on a recorded corpus.

| Flag | Default | Meaning |
|---|---|---|
| `--drafts-path` | `output/draft/drafts.jsonl` | Recorded corpus (§4). |
| `--cards-path` | `output/cardsfolder/` | `.npz` cache. |
| `--d-model` | derived (`embedding_dim` + feature widths) | Non-default inserts a `Linear` (FR-025). |
| `--n-layers` | 4 | SAB layers. |
| `--n-heads` | 8 | `d_model % n_heads == 0`, fail fast (FR-026, SC-006). |
| `--ff-dim` | `4 × d_model` | Feed-forward width. |
| `--dropout` | 0.0 | Transformer dropout. |
| `--imitation-weight` | 1.0 | CE coefficient; `0` ⇒ critic-only (FR-033). |
| `--critic-weight` | 1.0 | MSE coefficient; `0` ⇒ imitation-only (FR-033). |
| `--imitation-agents` | `forge-full` | Whitelist of agents whose picks are imitation targets; critic unaffected (FR-033). |
| `--lr` | `3e-4` | AdamW LR (FR-034). Resumable; re-applied to a resumed optimiser. |
| `--warmup-frac` | 0.05 | Linear-warmup fraction of **one epoch** (FR-034). Resumable. |
| `--batch-size` | 32 | States per gradient step. Resumable. |
| `--max-grad-norm` | 1.0 | Per-group gradient-norm cap (FR-034). Resumable. |
| `--epochs` | 100 | Max epochs (FR-039). Resumable. |
| `--val-fraction` | 0.0025 | Draft-disjoint validation fraction — small held-out monitor (FR-035). Resumable. |
| `--evals-per-epoch` | 100 | Validate + checkpoint this many times per epoch ("mini-epochs") (FR-039). Resumable. |
| `--patience` | 30 | Early-stop after this many mini-epochs w/o val improvement (FR-039). Resumable. |
| `--resume <ckpt>` | none | Continue a run: restore weights+optimizer+epoch+best-val and inherit its training settings (CLI flags override; **architecture flags forbidden**); mutually exclusive with `--checkpoint` (FR-039, SC-006). |
| `--checkpoint <ckpt>` | none | Bootstrap fresh run from weights only; **architecture flags forbidden**; mutually exclusive with `--resume` (FR-039). |

Resumable flags (every flag above except the architecture flags) resolve with the
precedence **explicit CLI value > resumed checkpoint's stored setting > default**, so
`--resume <ckpt> --lr 3e-5` keeps every prior setting and changes only the LR.

Behavior contract:
- Each `(draft, seat, pack, pick)` is one example (FR-030); loader reconstructs
  PACK/POOL/PASSED/TAKEN + recency via FR-016 geometry (FR-031).
- Loss `= imitation_weight·CE(policy, taken)` over whitelisted seats only
  `+ critic_weight·MSE(critic, standardized reward)` over all non-failed seats
  (FR-032, FR-033).
- Length-bucketed batches (similar-length states together, reshuffled each epoch)
  minimise padding/mask work (FR-036).
- Every mini-epoch (`--evals-per-epoch`×/epoch): validate, log the training and
  validation loss decomposition + val imitation top-1/top-3 accuracy + critic MSE
  sliced by `pack_number` (FR-037, SC-005), write `latest.pt`, update the best
  checkpoint, and check early-stopping.
- Best checkpoint by validation `L`; writes `{timestamp}.pt` + `latest.pt`
  under `models/draft/agent/` (FR-036, FR-041).
- Missing `.npz` cards warned (≤20 + total) and dropped (FR-038).
- Startup fail-fast on `d_model % n_heads != 0` and on architecture flags with
  `--resume`/`--checkpoint` (SC-006).

Success: runs to completion, emits a reloadable best checkpoint that produces
picks + critic scalars on held-out states (SC-004).

---

## Builder-validation diagnostic  (FR-042, script — not a subcommand)

A ~40-line script run once per picker/scorer checkpoint pair before a large
corpus run. Over a few hundred drafted pools, builds each pool with both the
picker and the SA `GreedyDeckBuilder`, scores both with the frozen scorer, and
prints:
- picker-vs-SA **Spearman** rank correlation (the gating number),
- the SA−picker score-gap distribution (median + spread),
- the SA-vs-SA reference correlation across independent SA restarts.

Gating: picker tracks SA ≈ as well as SA tracks itself ⇒ `--build-method
picker`; materially below, or composition-dependent gap ⇒ `--build-method
greedy` (SC-007).
