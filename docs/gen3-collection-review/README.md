# Gen-3 collection review before re-collecting (feature 025)

A working document for the review-then-fix round that runs before gen-3's records are collected again
from scratch. Read this first in a fresh session.

## Why
On 2026-11-09 a fork fidelity audit (`ForkFidelity`, connector 5958a393) compared every fork's copy with
the live game. In the first run 28% of forks differed from the real board. Several copier gaps were fixed
the same day:
- Forge 3aaa312c0a0, 416aae61a45, 52daa687cfa, 04e72e0518e;
- connector e6f71474, e9090adf, dc46a3a5 and others.

The user then decided to:
1. re-collect every gen-3 corpus from scratch (depleted, held-out coverage, full-strength), moving the old
   `output/effects/records-gen3/` aside rather than deleting it;
2. first review every module feature 025 touched, hunting collection bugs, and ask the user after each
   module's review what to do.

## Process (as run for the connector)
- The module is split into slices. Each slice goes to a fresh Opus reviewer subagent with `review-brief.md`
  (bug hunting first, the `/review` rubric second) and `review-hints.md`: bug classes from past collections,
  compiled from the gen-2 diaries, the gen-3 session transcripts and the fix commits.
- Reviewers are read-only and run no builds. They don't read `~/.claude`, `experiments/`, `diaries/` or git
  history.
- Each slice's findings then go to a fresh adversarial verifier that tries to refute them in the code, the
  Forge source and the Python readers, and may count real shards.
- At most 3 Opus agents at a time, never Fable.

## Status
| Module | Review | Verified | Findings |
|---|---|---|---|
| forge-connector | done 2026-11-09 | done | `connector-findings.md` |
| Forge `effect-record-hooks` (../forge) | in progress | | `forge-hooks-findings.md` (to come) |
| `src/sealed` + `forge_jvm` (collection supervisor) | to do | | |
| `src/effects` (corpus build, training, evaluation) | to do; may run during the re-collection, since it only reads records | | |
| `rules/` catalogue and `scripts/` | to do | | |

## The fixing round (fresh session)
Fix every confirmed collection finding in one planned pass, test-first. Each finding in the findings files
names the test that pins it.

Several attribution findings (connector A1, A2, A3; the mana hook timing in C2) can be fixed either in the
connector or where Forge fires the hook. Decide where after the Forge hooks review.

After fixing, verify before any long run:
1. the fork fidelity audit (20-minute collection with the full-strength flags, then count
   `fork fidelity:` lines by field; the audit is on in every collection);
2. `validate-corpus` and `field-coverage` on the first minutes of shards;
3. the score-discard count (`disagrees with the live`).

## User decisions so far
- **Mana records (C2):** read the board before the mana ability activates (source untapped, pool without
  the new mana), and add a `tapped` event for the source beside `mana_produced`. "Tapping a source that
  says add {G} produces {G}" must be clear. The cost half's `costs.mana_by_color` already says what mana
  paid. No link from a spell to the lands that paid for it.
- **Questions-to-the-user style:** in plain text in the reply, never through the question pop-up tool.

- **The record principle (decided 2026-11-09, covers A7, B5, C2):** every record describes one thing that
  happened at one moment. Its board is read just before that thing happens, and its outcome is what
  changed (the events are the "after"; no after-board is stored). A spell is several moments:
  1. **Cost half (casting or activating).** Read the board BEFORE: card still in hand (601.2a), costs
     unpaid. Today the board is read after payment: in a real Evolving Wilds record, Wilds is already in
     the graveyard. The outcome is what casting did: the card from hand to stack (for a spell; for an
     activated ability nothing moves), and the costs paid (mana by colour, tapped, sacrificed, discarded,
     exiled, life). A sacrificed card later shown tapped in the graveyard is fine.
  2. **Remove `outcome` from the cost half.** Today it holds the stack item's eventual fate (resolved,
     fizzled, countered, declined), written once known; that's a later moment. Record the fate when it
     happens:
     - resolved: the resolution record;
     - fizzled or declined: a resolution record whose outcome is "nothing happened";
     - countered: the counterspell's own resolution record, which must show the countered item leaving the
       stack (verify it does).
  3. **Effect half (resolution).** Read the board when resolution STARTS (a Forge "resolution starting"
     hook; we own the branch). Responses and simultaneous triggers have their own records. Fizzles stay
     learnable, because the resolution-time board already shows the target gone.
  4. **Library cards named by an event (B5).** Render them as they were in the library (untapped, no
     counters, not entered this turn, face up, owner-controlled, nothing chosen), and put "entered tapped"
     on the outcome (a flag on `zone_change` or a `tapped` event for the entering card). Today the leaked
     entity field is the only place it appears.
  5. **Mana records (C2):** already decided above, and the same principle.
  The user rejected the alternative of dropping records where something happened in response (it would
  lose fizzles and needs a fragile definition of "response").
