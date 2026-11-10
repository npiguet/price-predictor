# WP2 result: the copier API the connector must use (Forge cfb2ae28564)

WP2 of `FIX-PLAN.md` fixed the copier findings on `effect-record-hooks` (commits 90fa9b40cb7 …
cfb2ae28564, installed). Tests: `forge-gui-desktop/.../ai/simulation/GameCopierFidelityTest.java`.

## For the connector (WP3)
- `GameCopier.counterpartAbility(SpellAbility sa)` (public, after `makeCopy`) is the one ability mapper;
  `Forks.findInFork` must call it. Behind it is `forge.ai.simulation.AbilityMapper.map(sa, cards,
  players)` (also `locate`, `find`, `mapStatic`). A trigger ability that already fired maps to a fresh
  copy bound to the copy's trigger; every other ability maps to the copy card's own object; a spell
  found by neither identity nor id falls back to the host's single ability with the same API and
  description.
- `GameCopier.getNoCounterpartTally()` → `Map<GameCopier.NoCounterpart,Integer>` (kinds `COMMAND`,
  `DELAYED_TRIGGER`, `REMEMBERED`, `LINK`, `ATTACK`, `ABILITY`, `ACTIVATION`; reset per `makeCopy`, a
  kind with no drops absent). `ForkFidelity` should print it.
- `makeCopy` throws `IllegalStateException` for floating mana that adds counters, grants keywords or
  sets a when-spent trigger and whose source has no counterpart (H2-3); `Forks.copyAndVerify` already
  treats a throw as a lost fork.

## Audit rows explained
- `linked_by_text` (Experimental Synthesizer): traits sharing an Execute SVar share one ability object
  per card state; fixed by the structural mapper.
- `player.untapped_production` counting a colour as C: Cryptic Spires' marked colours, fixed by F6.
- The Robot token's +1/+0 and haste: not reproduced on a TestBoard (`kavaronsRobotKeepsItsBoostAndHaste`
  passes on the old and new copier); the audit after the fixing round will show whether it remains.

## Also fixed
- A live-game bug: copying a trigger or replacement wrote its card state onto the original's shared
  ability, blanking a live damage shield's ability state.
- The per-card copy also carries the turn's exert, crew, tap and regenerate counts.
