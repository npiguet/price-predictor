# Game-playing agent — design rationale

## Background

Nothing in the stack plays a game. The price encoder, the sealed encoder, the deck scorer, the picker and the draft agent all decide what a deck should contain; once the cards are chosen, Forge's built-in AI pilots every game the project has ever recorded. The ability effect model ([`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md)) is a model of what an ability does to a board, and its design deferred "a full world model" as the natural next step towards learning to play. This document decides how that step is taken: what the agent observes, what it decides, how it is trained, and in what order the pieces are built and measured.

The objective is the one the rest of the stack already optimises. Forge's built-in AI is generation zero, and the agent is measured by the games it wins against it, in the sealed 40-card-deck distribution the pool and deck generators already produce. Forge's piloting quirks are the target, not a confound; an agent that exploits them is doing its job.

Three words recur below and are used in one sense each. A policy is a function from a game state to a probability over the legal actions. A value function is a function from a game state to the probability that the acting seat eventually wins. A pointer head is an output that selects one of the tokens already in the model's input, instead of one of a fixed list of categories.

## A policy that answers four of Forge's decisions is the critical path

The agent is a neural policy that takes over four of Forge's controller decisions and delegates the rest to Forge's own code. Forge asks a `PlayerController` 109 distinct questions over a game. Four of them carry almost all of the leverage: which spell or ability to play at priority, or pass; what it targets; which creatures attack; which creatures block whom. The other 105, from mana payment through mulligans to the two dozen effect-specific choices, each have a serviceable Forge heuristic and a small option space, and any of them can later be promoted into the policy as a pointer over an option list without changing the architecture.

Delegation is what makes the agent buildable and gives it a floor. A controller that had to answer all 109 questions from a neural network would spend its first months on the long tail before it could play a game. A subclass of `PlayerControllerAi` overrides the four and inherits the rest, so on day one the agent is Forge with four decisions swapped out, and every experiment measures those four in isolation.

The plug-in point is confirmed in Forge's source and needs no engine patch. A subclass of `LobbyPlayerAi` seats a subclass of `PlayerControllerAi`; the decision methods are public and non-final. The override runs on the game thread, outside the five-second evaluation timeout Forge applies to its own candidate loop, so a network round trip inside it is safe. Forge's own simulation picker is the existence proof for the rest: it enumerates rules-legal candidates without the AI's policy filter, installs targets on a candidate from outside the per-effect logic, and casts through the inherited mana-payment path. The agent controller copies that pattern.

Two engine behaviours shape the safety code around combat. The engine re-validates an attack declaration in an unbounded retry loop, so an invalid declaration hangs the game rather than failing. The engine never validates a block declaration at all; the only caller of the block validator is the desktop GUI. The controller therefore validates both declarations itself before returning and falls back to Forge's own attack or block controller when validation fails, with the fallback rate logged as a first-class metric.

## The state is the effect model's entity tokens with a short memory

The observation is the game-state snapshot the effect-record collector already renders, taken from the deciding seat's perspective. The token layout is the effect model's: one global token for turn, phase, priority, stack size and combat sub-step; one token per player with life, zone sizes, poison and energy, this-turn counters, floating mana and untapped mana production by colour; one token per card with its computed characteristics, power and toughness decomposed into base, boosts and counters, tapped and summoning-sick state, damage, counters, combat status, attachments, temporary grants, controller tag and zone; and after each card token, one token per ability line carrying that line's ability vector. Reusing the layout is deliberate. The effect model's trunk is then a valid initialisation, its heads are valid auxiliary losses, and every representation improvement it earns transfers to the agent for free.

Version one shows the seat its own hand, both graveyards, the stack, the battlefield and the command zone, and never the opponent's hand or either library. The collector's widest snapshot tier includes both players' hands, which is right for a model of what abilities do and wrong for a policy: an imitation corpus rendered that way would train a policy that plays as if it could see the opponent's cards, look excellent offline, and play badly live. The export therefore drops hand entities the deciding seat does not control, and drops referenced library cards, and the corpus validator asserts the rule.

Two small blocks give the policy a memory. The snapshot is nearly a complete description of the game, since graveyards show what has been spent and the this-turn counters show what has happened this turn, but it forgets everything before the current turn. A history block carries the last sixteen or so decisions with the chosen action and the events that followed. A seen-cards block carries opponent cards this seat has legitimately seen, from reveals, bounces and scries. Reveal-type events are rare, about one per game in the effect corpus, so the block is cheap. Whether history earns its place is an experiment: the same corpus trains a memoryless policy, one with the history block, and one with a full transcript, and the accuracy gap between the second and the first sizes the block.

The token budget is set at 384 slots. Snapshots with ability tokens measure about a hundred tokens on average and just over two hundred at the 99th percentile; candidates, history and seen cards add a few tens. A late game with two full graveyards can exceed the budget, in which case the oldest graveyard entities are dropped first.

## Actions are pointers over candidate tokens, validated in Java

Each of the four decisions is a pointer head over tokens in the input, masked by Forge's rules checks and validated in Java before the answer reaches the engine.

| Decision | Candidate tokens | Head | Validation and fallback |
|---|---|---|---|
| Priority action | one `[CAND]` token per rules-legal spell or ability variant, plus `PASS` | probability over candidates and pass | enumeration follows the simulation picker, plus timing, stack-legality and restriction checks; windows with no candidate auto-pass in Java with no round trip |
| Targets | legal targets of the chosen candidate | one pointer per target slot, chosen one at a time with a stop once the minimum is met | targets installed on the ability before casting; a cast that fails inside Forge's play path is logged and the window re-asked with that candidate masked |
| Attackers | legal attackers | a yes/no probability per creature | attack legality per creature, then the engine's own validator before returning; fallback to Forge's attack controller |
| Blockers | legal blockers, attackers | per blocker, a pointer over the attackers plus "no block", chosen one blocker at a time so gang blocks coordinate | block legality and minimum-blocker rules, then the block validator before returning; fallback to Forge's block controller |

A candidate token is one variant of one ability. A kicked cast and an unkicked cast are two candidates, as are a flashback and a normal cast. The token carries the host card's vector, the ability vector, cost and alternative-cost flags, phase context and the rules verdict: can play, affordable, has a legal target. Forge's own policy verdict on the candidate is recorded in the corpus and deliberately not an input; the reason is given with the imitation design below.

Targets are the decision most entangled with Forge's internals, and the entanglement is measured before anything is trained. Forge chooses targets inside its per-effect "should I play this" logic, as a side effect of evaluating a candidate. The agent path strips those targets and installs its own. Whether an externally targeted ability survives Forge's play path is a per-effect-type question. A high failure rate on some types would silently turn the agent into a Forge clone on modal, X and multi-target spells, which are exactly the plays where beating Forge is possible. Every headline metric would still report a working system, because the fallback is invisible in a win rate. The build order below therefore starts with a per-effect-type census of that survival rate.

Mana payment, X values, modes, mulligans, discards, ordering and confirmations stay with Forge in version one. X takes Forge's maximum-affordable rule, modes take Forge's choice, and both are promoted to pointer heads only after the transcripts show Forge's default losing games.

## One network carries policy, value and the teacher's evaluator

The network is the effect model's two transformers with new heads. The ability encoder is four layers wide 256; the trunk is six layers wide 256 with four attention heads; the four action heads sit on the candidate, entity and attacker tokens; a value head sits on the global token; and the effect model's per-entity outcome heads and a next-event head stay attached as auxiliaries. About ten million parameters in all. Training at batch 64 over 160 tokens is estimated under two gigabytes of the eight available, with the encoder run once per unique ability text per batch; inference is estimated at a few milliseconds for a batch of 32 states. Both estimates are confirmed on the first run before anything depends on them.

One network serves every role. The value head is trained jointly with the policy from the first imitation run and is read by the search teacher and by the lookahead milestone described below. A copied game is rendered through the same snapshot builder and the same export filter as a live state, so a copy is an ordinary input and needs no second network.

The ability encoder trains end to end, warm-started from the sealed encoder, under the imitation loss and the auxiliary effect losses together. The auxiliaries are what make end-to-end training sound: one chosen-or-passed bit per line is too weak a signal to teach an encoder what "deals 3 damage to any target" means, but the effect heads give it a per-line target from millions of observed resolutions. The effect encoder replaces it, through the same one-vector-per-ability-line cache interface, once the effect model's gates pass.

The sealed card vector is off by default. It is a per-card fingerprint, and the cheapest fit for a policy given a fingerprint is "Forge casts this card on turn three", which generalises to no unseen set. It is added only if the imitation ablation shows a gain on held-out sets.

## Twelve workers feed an online learner, and the corpus, not the wire, sets the sampling

Game throughput is not the binding constraint. The match workers already play thousands of games an hour with effect collection switched on, which makes online training against Forge at fifty thousand games a day realistic. The binding constraints are the single eight-gigabyte GPU shared by training and inference, and the engineering of the decision surface.

| Quantity | Value | Measured on |
|---|---|---|
| engine time per game per worker | 5.9 s | `match-outcomes.txt`, 4,894 best-of-7 matches, 2026-09-11, effect collection on |
| aggregate games per hour, 12 workers | ~4,200 | same corpus |
| priority windows per game with a legal non-land action, both seats | 95 mean; 1.7 candidates mean, 3 at p90, 10 max | 41-game effect shard |
| distinct decision points per game, both seats | 134 mean, 100 median, 279 at p90 | 60-game effect shard |
| decision windows per game counting targets and single-candidate windows | 150–300 | 140-game effect shard |
| turns per game | 21–23 mean, 19 median | both shards |
| snapshot size, tiers 1–4 | 20–25 KB mean, 51 KB max, 30–40 entities | 140-game effect shard |
| tokens per snapshot with ability tokens | 98 mean, 208 at p99 | 41-game effect shard |

Two of the measured figures drive the design arithmetic. The policy has a real choice about 75 times per seat per game, counting the priority windows with a legal action, the attack and block declarations and the target choices, and that is the horizon the reinforcement-learning signal has to cover. The corpus logs 150 to 300 windows per game, which at twenty-odd kilobytes each is several megabytes per game uncompressed and about a gigabyte per hour gzipped across twelve workers. The first corpus logs every consulted window with at least two options at full rate; sampling is a question for larger runs, not the first one.

The wire is a persistent Python inference server with one local socket per worker JVM, carrying line-delimited JSON frames with a request id and a protocol version. Java writes the request with the existing snapshot builder and parses a flat response; the connector has a JSON writer and a flat parser and no parser library, which is why the format is lines rather than a binary schema. One request is outstanding per worker. One round trip serves a whole priority window: the server runs the priority head and then the target head for the chosen candidate inside the same request. Combat is one round trip per declaration. Requests are batched across the twelve workers, up to 32 or three milliseconds, and keyed by game id, so a dropped connection or the sixty-second worker recycle loses that game and nothing else. Live volume at full snapshots is around thirteen megabytes a second, which a loopback socket carries without effort. The one hard rule on the Java side is that nothing may block inside the candidate hook or any per-effect AI code, because those run on the evaluation thread under Forge's timeout.

## Imitation of Forge is nearly free, and three corrections keep it from cloning Forge's faults

The first training signal is Forge's own decisions, logged from the self-play matches already running. A recording subclass of the AI controller sits on both seats behind an opt-in flag on the match-outcomes command, and at each consulted window writes the filtered snapshot, every candidate with its rules verdict, its policy verdict and its legal targets, and Forge's choice after the play: the chosen ability with its targets, the attack set, the block map. At game end one outcome record names the winner. Shards follow the effect corpus's contract: append-only, one file per worker JVM lifetime, complete gzip members, ids namespaced by lifetime. Imitation is the same route the draft agent took to its first generation ([`2026-05-30-draft-agent-design.md`](2026-05-30-draft-agent-design.md)), and the outcome record is what the value head trains on.

The loss is the log-loss of Forge's demonstrated choice on each of the four heads where Forge made that decision, plus the value head's loss against the outcome, plus the effect auxiliaries at a small weight. Three corrections keep the clone from inheriting Forge's known faults.

Candidates Forge refuses by policy are unlabelled, not negative. Forge hard-refuses some effect types and never casts the cards that carry them, which is a documented source of its losses. In a plain imitation corpus those candidates are always labelled "pass" while sitting legal in the mask, and the log-loss drives their probability to zero, which clones the fault. Forge's decision enum separates rules verdicts from policy verdicts, so a candidate refused by policy is excluded from the loss and left for exploration to discover.

Forge's policy verdict is not a model input. Fed in as a feature it would let the network recover "Forge refuses this, so pass" through the back door the unlabelling rule just closed.

After the first live agent exists, a second corpus is collected on the agent's own games. At each agent decision the delegated Forge path is called for a label and not executed, the model is retrained on both corpora, and the cycle repeats until the agent's disagreement rate with Forge on its own states stops falling; two rounds are the budget. This is the DAgger procedure, and it exists because a clone compounds its errors: one held-back creature puts the agent in a board Forge never produced, where the clone has no training.

Four diagnostics are read before any win rate. Pass is three priority labels in four, so accuracy is reported by phase, candidate count and open mana, with recall on non-pass choices separate. Combat labels are dice-rolled inside Forge's controllers, so combat accuracy is graded against the label's own repeat-consistency rather than against 100 percent. The sets used for evaluation are excluded from collection, and non-pass recall on held-out sets must sit within five points of held-in, or the encoder is fingerprinting cards. And the board-erasure test from [`2026-08-29-draft-agent-behaviour.md`](2026-08-29-draft-agent-behaviour.md) is run from the start: erase or relabel the opponent's board and count the attack and block decisions that move. The draft agent's fourth generation turned out to be a sharper fixed pick order that ignored the table, and the play analogue, a cast order plus attack-above-a-rating, is exactly what one bit of reward against a weak opponent would produce.

## Online policy gradient against frozen Forge, with seeded groups and sparse deviations

The second training signal is games won against Forge, learned online with the loop that took the draft agent from its first generation to its fourth ([`2026-06-15-draft-agent-gen3-online-grpo-design.md`](2026-06-15-draft-agent-gen3-online-grpo-design.md), [`2026-08-09-draft-agent-gen4-online-grpo.md`](2026-08-09-draft-agent-gen4-online-grpo.md)). Policy gradient means the update pushes up the probability of the actions taken in games that went better than expected and down in games that went worse. The expectation comes from a group of games rather than from a learned critic, which is what the draft loop calls group-relative. The opponent is Forge, frozen, so the margin is absolute rather than field-relative: it does not move when the field moves, because the field never moves.

Games allow one variance reducer drafts did not. A round is a set of groups; a group is one deck pair and one shuffle seed. Seeding Forge's global random source before the game starts pins both opening hands and both draw orders, so every game in a group starts from the same cards and mana screw and flood, the largest source of noise in a sealed game, cancel inside the group. The pinning holds until the first action that differs between two games; after that the opponent's own random draws diverge, so the cancellation covers the shared prefix and the advantage lands on what happened after the divergence.

Exploration is sparse and structured, which is what makes the shared prefix long. A few randomly chosen decisions per game are sampled at a high temperature, meaning with the action probabilities flattened toward uniform, and every other decision plays at argmax, the single most probable action. One game per group plays entirely at argmax as a counterfactual. Sampling every decision, as the draft loop did, would diverge the games of a group within two or three decisions. It would also blunder about twenty times a game, and the only lesson available from twenty blunders is "stop deviating". An off-argmax draft pick is a near-equal card; an off-argmax block is usually a mistake. Combat therefore stays near argmax until priority play has moved.

The reward is the game result, the advantage is the game's result minus the mean of the other games in its group, standardised per round, and the loss is the standard policy-gradient term restricted to the deviated decisions. There is no shaping term from a value function. Shaping that leaves the best policy unchanged adds the change in a potential between successive states to the game result. A per-decision advantage built from the change in a frozen value head's prediction drops the game result instead of adding to it. What remains is learning against a fixed value estimate fitted to Forge-versus-Forge play, and that estimate stops correlating with the return as soon as the learner drifts. That is the failure that collapsed the draft agent's offline second generation, with the staleness moved from the corpus to the value estimate. If a baseline is ever needed, a state-only value is refitted each round on that round's outcomes and discarded; it never enters the target.

Four things are logged per round, and each guards against a named failure. The fraction of groups whose outcomes are not unanimous, since a unanimous group carries no advantage and about one mirror match in five is a clean sweep before any pinning. Censoring: the worker recycle kills the longest-running game, deviations that lengthen games would be selected against, so a group with a killed game is dropped whole. Per-decision-kind entropy and off-argmax rate, and the two divergences the draft loop tracks, from the previous round and from the warm start. And a sliding-window win rate over at least thirty rounds that only nominates checkpoints; the top three nominees are confirmed on the evaluation set and the winner's figure is quoted from a fresh draw, because a maximum over a correlated series reads high.

The round budget is not fixed in advance. With around ten informative groups a round the signal is a few tens of bits per round, so progress will be slow, and the budget is set from the informative-group fraction the probes below measure. Three thousand rounds, about three days of wall time at one to two minutes a round, is the placeholder.

## Search is a teacher and a ceiling probe, not the training loop

Forge's search machinery is too slow for the training loop and too valuable to ignore. Forge ships a game copier, a simulator that casts a candidate on the copy and resolves the stack, and a hand-coded state evaluator; its full-simulation AI uses them at depth three.

| Quantity | Value |
|---|---|
| one game copy | 8–13 ms |
| Forge's evaluator, near-empty board | 0.1 ms |
| Forge's evaluator, 23+ permanents | 111–135 ms, because it copies the game again to simulate the coming combat |
| depth-3 simulation of one decision, turn 2, one candidate | 12 ms |
| depth-3 simulation of one decision, turn 16, five candidates, eight copies | 2.27 s |
| game with one full-simulation seat versus reflex Forge | 25–46 s against 4–13 s |

Measured with a headless probe on 80-card sealed games; the figures grow with board size because a copy re-parses every card from its script.

Three uses of search survive the timing table, and search at play time does not. One-ply lookahead, meaning one copy per candidate scored after the candidate resolves, triples game time, which cuts the twelve-worker throughput to less than half for every evaluation and every training rollout. The three uses that survive are a ceiling probe, an offline teacher, and a drop-in evaluator for Forge's own AI.

The ceiling probe runs before any model exists. Seat one side with Forge's own full-simulation AI against reflex Forge on the evaluation deck set. The copier is a perfect transition model and Forge's evaluator is a reasonable value function, so the result bounds how much better-than-reflex decisions are worth in this environment at all. If it barely beats reflex Forge, games here are not won at the decision points the agent takes, and the plan shrinks. If it wins clearly, the result becomes the roadmap's target.

The teacher is expert iteration: a slow, strong procedure decides, the fast policy learns to reproduce its decisions, and the improved policy makes the next round of search cheaper. For each of the policy's top candidates, one copy is cast and resolved and scored by the value head, and the fast policy imitates the argmax. Three rules bound it. The value head scores the copy's visible projection, never the raw copy, because a copy contains both hands and both libraries and the deliverable policy never sees either. Search runs only when the stack is empty, because the copier drops the stack by default and searching a response from a copy without one is simply wrong. And search runs only at decisions where the policy is unsure, a small top-two gap or three or more candidates, which skips most decisions.

Two facts make the teacher cheaper and more valuable than the timing table suggests. A learned evaluator is faster than Forge's late in a game, because Forge's evaluator makes a second copy to simulate combat and a batched network call does not; a teacher game costs about half a minute to a minute, and an overnight run on the twelve workers yields several thousand teacher games. And Forge's picker never searches combat: its simulator advances straight to combat damage and lets the heuristic attack and block controllers decide. Sampling a handful of attack sets from the attack head, copying, advancing to end of turn and scoring the visible projection improves combat immediately, with the shortest credit path in the game, and it is the first teacher target.

The value function is therefore trained first, and selected on ranking rather than on calibration. It trains on the decision corpus, where every logged window is an outcome-labelled state, with no rollouts and no reinforcement learning. Search argmaxes it, so what matters is whether it orders sibling states one action apart correctly: same-game pair-ranking accuracy, and the median spread across one decision's candidates against the head's own held-out error. A value head that reads 0.52 for every candidate is coin-flipping however well calibrated it is. Two invariants ride along: zero opponent-hand tokens in any value input, and a logged drift metric between the value of a live state and the value of a fresh copy of it, bucketed by turn, which is Forge's own copy-consistency check applied to the learned head. The copier also loses several this-turn counters, so the value head trains with dropout on that block and serves live states and copies alike.

The drop-in evaluator is a bonus agent. Forge's full-simulation AI with the learned value head in place of its hand-coded evaluator needs no Python policy at play time. The evaluator is constructed inline at four sites in Forge, so this needs one small pluggability hook, one more inert commit in the `effect-record-hooks` series.

## The effect model is an initialisation and an auxiliary loss, never a dependency

No effect checkpoint exists and its three gates are unrun, so nothing on the critical path may wait for it. The agent uses three things from the effect design that exist today: the token layout, the trunk class, and the per-entity heads as auxiliary losses on effect records, joined to decision records by game id when both collectors run. The one-vector-per-ability-line cache interface is what lets the effect encoder be swapped in later without touching the agent.

The effect corpus supplies outcome labels only from the day the game-outcome event was wired into it. The ten-million-record corpus collected on 2026-09-09 and 2026-09-10 carries no per-game winner; shards from 2026-09-11 onward carry a `player_won` event naming the winner inside the game's last record, and the trainer ignores it. Effect-record states are therefore a supplement to the value head's data, for games collected since then, and the decision corpus is its primary source.

## Evaluation is win rate against frozen Forge on held-out sets, with exact side alternation

Every stage is measured on one fixed deck set. About two hundred pools from four to eight sets held out of every collection run, built into decks by Forge's own builder so the objective keeps Forge's quirks, played as mirror pairs and cross pairs, five thousand games per measurement.

| Quantity | Value | Measured on |
|---|---|---|
| standard error at 5,000 games | 0.7 percentage points (pp) | |
| decisive difference between two checkpoints | 3 pp, about three standard errors (3σ) | |
| win rate of the player on the play, Forge mirrors | 47.9 % | 13,504 mirror games |
| clean sweeps in best-of-7 Forge mirrors | 490 of 2,439, about one in five | match corpus |
| wall time per measurement | 45 min Forge vs Forge; ~1 h with an agent seat; ~2 h with lookahead | |

Mirror pairs isolate piloting from deck quality, since both seats hold the same deck. Side alternation must be exact per pair and per side, because being on the play loses slightly in Forge's games and that effect is the size of the decisive margin. Each measurement also reports the Forge-fallback rate, since an agent that falls back on a fifth of its decisions is a fifth Forge, and the censored-game count per side.

## Build order: probes before models, value before policy, teacher before self-play

Each stage ends in a measurement on the evaluation set and a go criterion; a stage that misses its criterion changes the plan rather than running longer.

| Stage | Builds | Go criterion |
|---|---|---|
| 0 | controller subclass, recorder, inference server, a pure-delegate agent, four probes | delegate at 50 % ± 1.4 % in mirrors; illegal or failed actions under 0.5 %; targeting census at or above 95 % of casts surviving, failing effect types listed; p99 round trip measured with a dummy ten-million-parameter model while a training job shares the GPU; ceiling and signal probes read |
| 1 | decision corpus of 15–30k games, imitation with the value head, the four diagnostics | held-out sets within 5 points of held-in; board-erasure test shows combat reads the board |
| 1b | the value head alone, driving one-ply lookahead over Forge's own candidate list | at or above +3 pp in 1,500–2,000 paired common-seed mirrors |
| 2 | live imitation agent on priority and targets, then combat; two DAgger rounds | beats the control, a random disagreer at the clone's own disagreement rate, by 3σ |
| 3 | online policy gradient against frozen Forge | 3σ improvement of the argmax policy within the round budget the signal probe sets; target halfway from 50 % to the ceiling probe's result |
| 4 | search teacher, promoted long-tail decisions, a league of frozen snapshots plus Forge | frozen-Forge win rate stays the sole promotion criterion; target the ceiling probe's result |

The signal probe replaces a tempting but wrong experiment. Replacing Forge's choice with a uniformly random legal action at some rate and watching the win rate fall measures how fast play degrades below Forge, which it will do quickly, and says nothing about whether the gradient can see the far smaller gap between Forge's choice and a plausible better one. The probe that measures that gap deviates once per game to Forge's second-ranked candidate, and counts how many seeded groups carry any signal under the intended exploration.

Compute is comfortable and the run count is not one. One imitation pass over a few million decisions is two to three GPU-hours at realistic utilisation; a policy-gradient round is one to two minutes of wall time; an evaluation is under an hour. The draft agent's third generation needed six runs and its fourth needed four, so the budget is five to ten full runs, not one. The one interaction to measure early is GPU contention: the inference server and a training job share one card, and a training step that stalls the batcher for a quarter of a second stalls all twelve workers.

## Rejected and absorbed alternatives

Each entry states the principle, why it was rejected, and what it contributed to the design above.

### Value-first search as the agent

The principle: keep Forge's decision machinery and search, and replace only its hand-coded judgment with a learned value function. Rejected as the agent because the copier's cost triples game time at one ply and reaches seconds per decision at depth three, copies are omniscient and drop the stack, and Forge's search covers priority plays only, so combat and the other 105 decisions would be untouched. Absorbed almost whole as the complement: the value head trained first and selected on ranking, the lookahead milestone in stage 1b, the combat teacher, and the drop-in evaluator.

### World-model planning with the effect head

The principle: score a candidate action by predicting its consequences with the effect head, writing the prediction back onto the state tokens and evaluating the result, without any engine copy. Rejected as the play-time decision rule because honest planning depth is one: the effect head predicts no turn advance, no opponent response, no triggered chain, and expectations rather than samples of stochastic effects, and its own reconstructions are off-distribution inputs that compound error. It also depends on gates that are unrun, and on an accuracy gate the effect model does not have, since its first gate measures margins over a text-blind baseline rather than absolute accuracy. Absorbed as the auxiliary losses on the trunk, which need no gate, and as a candidate pruner inside the teacher, where a wrong prediction costs one extra copy rather than a game.

### A sequence model over game transcripts

The principle: treat a whole game as one token stream of events and decisions, train a causal model to predict the next decision conditioned on the outcome, and play from the growing prefix. Rejected because a corpus of event logs makes event-vocabulary completeness load-bearing, which is the reason the effect corpus stores snapshots ([`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md) rejects a replayable log for the same reason); because outcome conditioning on a stochastic game clones lucky mistakes; because a killed worker truncates a whole game rather than one window; and because the engineering, per-seat caches, visibility masking and its leak test, is far larger than a per-decision policy's. Absorbed as the history block, the seen-cards block, the next-event auxiliary head, trajectory filtering to winning games as an option, and the three-way experiment that sizes the history block.

### A random-deviation probe of the reward signal

The principle: measure whether one win/loss bit carries enough signal by replacing Forge's choice with a random legal action at increasing rates. Rejected because it measures degradation below Forge, not the gap the gradient has to see; replaced by the single-deviation probe and the informative-group count.

### Advantage shaping from a frozen value head

The principle: use the change in a frozen value head's prediction as a per-decision advantage, to give a dense signal over the 75-decision horizon. Rejected because it is not potential-based shaping and reproduces the stale-critic collapse of the draft agent's offline second generation; replaced by the game result with a per-round refitted baseline at most.

### Forge's policy verdict as a policy input

The principle: give the network Forge's own opinion of each candidate as a feature, so it can learn where Forge is wrong. Rejected because the feature reopens the refusal-cloning leak the unlabelling rule closes; the verdict is recorded for analysis only.

### The sealed card vector as a default input

The principle: concatenate the sealed encoder's per-card vector to every card token as a strong prior. Rejected as a default because it is a per-card fingerprint that invites memorising which cards Forge casts when; admitted only if the held-out ablation shows a gain.

### In-JVM inference

The principle: bundle an ONNX runtime in the connector's fat JAR and run the network inside each worker, removing the round trip. Deferred rather than rejected: it is one dependency line, but twelve CPU-bound sessions cannot share the GPU, the Python server reuses the draft agent's side-channel pattern, and the round trip is not the throughput risk. It returns if the drop-in evaluator's serial calls ever flood the batcher.

## Risks that would change the plan

The agent's biggest unproven claim is that one bit per game can lift a 75-decision policy above its Forge-cloned start. If policy gradient stalls while exploration is healthy and enough groups are informative, the teacher becomes the primary learning mechanism rather than the finisher, because its signal does not depend on the game outcome.

If the ceiling probe shows Forge's own perfect-model planner barely beating reflex Forge, the decisions this agent takes are not where games are won, and the effort moves to the delegated long tail or to deck quality.

If held-out accuracy collapses or the board-erasure test shows a board-blind cast order, the card prior is dropped and the encoder is trained on ability text alone.

If the targeting census fails for effect types that matter, the target head is replaced for those types by a pointer over Forge's own target proposals until the play path is understood.

Exploiting Forge rather than playing well is acceptable by the stated objective. Self-play and the league are the hedge if the agent's play looks degenerate against anything but Forge.

## Outcome / Result

Not yet run. The stage-0 probes decide the plan and none has been executed: the pure-delegate parity check, the failed-action rate, the per-effect-type targeting census, the ceiling probe of Forge's full-simulation AI against reflex Forge, and the single-deviation and informative-group probes. Fill in each with its figure and the round budget it implies, then the imitation diagnostics, the stage 1b lookahead result, and the stage 2 and stage 3 win rates on the evaluation set.
