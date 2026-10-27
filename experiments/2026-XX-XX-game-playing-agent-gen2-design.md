# Game-playing agent — gen-2 planning over chained plays (draft)

## TL;DR

The gen-1 agent can carry out a chain of plays, where one spell sets up the next, but its training rarely discovers one. Imitation copies Forge's reflex AI, which judges each spell mostly on its own, and sparse single deviations seldom try a setup play together with its follow-up. Gen-2 plans to teach chains through search: the gen-1 search teacher is extended from one play to short sequences of the agent's own plays, the hidden cards are re-dealt in every game copy before it is searched, and policy gradient deviates over a stretch of consecutive decisions within one turn instead of isolated decisions. Nothing is built or run. The gen-1 ceiling probe decides whether this is worth building.

## Background

Gen-2 is the second generation of the game-playing agent. Gen-1 is described in [`2026-09-12-game-playing-agent-design.md`](2026-09-12-game-playing-agent-design.md), called the gen-1 record below. The terms policy, value function, seeded group, search teacher and ceiling probe are used as that record defines them.

A chain is a sequence of plays in which a later play depends on the result of an earlier one. Casting a landfall creature before playing the turn's land, making a token before activating a sacrifice outlet, and killing the only blocker before attacking are all chains. The first play of a chain is a setup play: on its own it is often worth little or less than nothing, and its value comes from the play it enables.

### Gen-1 can carry out a chain, but its training rarely finds one

The gen-1 architecture can represent a chain. The priority decision is asked again after each play resolves, so the next decision sees the result of the last. The card tokens include the hand cards that are not yet castable, and the history block carries the recent decisions.

Each of gen-1's training signals is weak at discovering a chain.

- Imitation learns only the chains Forge's reflex AI already plays.
- Policy gradient deviates on a few isolated decisions per game. A chain needs the setup play to be sampled off-argmax and then the follow-up to be sampled too, in the same turn. A setup play without its follow-up usually loses, so the gradient learns to avoid the setup play.
- The one-play search teacher scores the state right after the setup play. That credits the setup play only if the value head already knows that the follow-up it enables is worth something.

### Forge's reflex AI judges each spell mostly on its own, so it misses plays that pay off only after a follow-up

Forge's reflex AI decides at each priority window whether to play each candidate now, by per-effect logic that looks at that candidate and the current board. A handful of hand-written ordering rules cover common cases, such as playing the turn's land early. A setup play whose value depends on a follow-up looks bad to that logic and is not made. Forge's full-simulation AI does plan: its simulation picker builds a plan of up to three plays, scored by Forge's hand-coded evaluator, and follows it. That planner is too slow for the training loop, as the gen-1 record's timing table shows. It is also the planner the gen-1 ceiling probe seats against reflex Forge.

### Game copies hold the real hidden cards, so a deeper search would score plays with knowledge the seat lacks

A game copy holds the real order of both libraries and the opponent's real hand. The gen-1 teacher scores the copy's visible projection, which hides those cards from the value head, but plays in the copy still resolve against them. A draw spell in a copy draws the real top cards, and a discard effect hits the real hand. The teacher picks the best of several plays, each scored on one real outcome. That choice favours the plays whose outcome varies most and happened to come out well. It also favours lines that work only because of the actual cards. A sequence of several plays contains more such plays than a single one, so the bias grows with search depth.

## Description of the experiment

Gen-2 teaches chains mainly through search, and adds one change to policy gradient. Three moves serve it, in order of how much of the plan rests on them:

1. **A search teacher over sequences.** The teacher searches short sequences of the agent's own plays and scores the state at the end of the turn, so a setup play is credited with what its follow-up achieves.
2. **Hidden cards re-dealt in every copy.** Each copy is searched over several re-dealt versions of the hidden cards, so the teacher scores a line by what the seat can expect rather than by what the real cards happen to be.
3. **A deviated stretch in policy gradient.** The deviated stretch covers consecutive decisions within one of the agent's turns, so a setup play and its follow-up can be tried together.

### The search teacher plans over short sequences of the agent's own plays

The gen-2 teacher runs a beam search over the agent's own plays within one phase, starting from the policy's top candidates at each step. Each node of the beam is a game copy with one more play cast and resolved. Chains within a main phase mostly resolve each spell before the next is cast, so the gen-1 rule that search runs only with an empty stack rarely blocks one. Each finished sequence is scored after advancing its copy to the end of the turn, with combat declared by the policy at argmax, as the gen-1 combat teacher does, so that a removal spell that clears the way for an attack is credited with the attack. The policy is trained to imitate the first play of the best sequence. At the next decision the search runs again from the new state.

The teacher runs only where gen-1's does: at decisions where the policy is unsure. The beam's width and depth are set from the copy cost in the gen-1 record's timing table and from the number of teacher games an overnight run must produce. Forge's simulation picker searches three plays deep, which is the depth the ceiling probe measures and the first depth to try.

The ceiling probe is gen-2's go criterion. It seats Forge's three-play planner, with a perfect transition model and Forge's evaluator, against reflex Forge. Its margin bounds what sequence search with a hand-coded evaluator is worth in these games. A margin that barely clears reflex Forge means gen-2 is not built in this form.

### Every copy re-deals the hidden cards before it is searched

Before a copy is searched, the hidden cards are re-dealt several times, and each sequence is scored by its average over the re-deals. Each re-deal shuffles both libraries and deals the opponent a hand of the right size consistent with what the seat has seen. A card the seat saw go into the opponent's hand stays in the re-dealt hand until it has left the real one. The rest of the hand is drawn from a distribution of plausible cards. This is determinization. Information-set search is the formal version of the same idea and costs more. Determinization has one known weakness: it plans as if it would learn the hidden cards later, because each re-deal is searched with its cards fixed. It matters for a sequence whose later plays depend on what an earlier play reveals, such as a scry or a draw before the next spell is chosen.

The distribution the opponent's unseen cards are drawn from is an open decision. The seat does not know the opponent's decklist in sealed. Drawing from the opponent's real remaining deck leaks its composition. Drawing from the set's card pool, weighted by the colours the opponent has shown, leaks nothing but deals hands the opponent could not hold.

### Policy gradient deviates over a stretch of consecutive decisions within one turn

Gen-2 replaces gen-1's isolated deviated decisions with one deviated stretch per game: a run of consecutive priority decisions within one of the agent's turns, sampled at high temperature, with every other decision at argmax. A setup play and its follow-up can then be sampled in the same game. The seeded group's shared prefix ends at the stretch's first decision, and the advantage and loss are gen-1's. Gen-1 keeps deviations sparse because each off-argmax play is usually a blunder. One stretch per game keeps the number of deviations per game close to gen-1's while making them consecutive.

Playing a sequence the search teacher found, instead of sampling the stretch, is a more directed way to explore. It is off-policy for the policy-gradient loss, which assumes the deviated plays were sampled from the policy, and it stays open.

### Planning in a learned latent model is rejected

The principle: plan several plays ahead in a model of the game's dynamics learned by the network, as MuZero does, with no game copy. Rejected for the reasons the gen-1 record gives for rejecting planning with the effect head. A learned model has no reliable opponent responses or triggered chains, and its prediction errors compound with each step of depth. It also needs far more training data than the corpus holds. The game copier is a perfect transition model whose only cost is speed, and the teacher runs offline, where speed matters least.

## Result

Not yet run. Gen-2 depends on three gen-1 results: the ceiling probe, the one-play teacher and the value head it scores with.

## Future plans

None yet.
