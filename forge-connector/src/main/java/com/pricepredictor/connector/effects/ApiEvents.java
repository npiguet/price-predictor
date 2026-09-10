package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The event an effect API promises, built from the clause that ran.
 *
 * <p>Most of Forge's effects broadcast nothing. Damage and zone changes reach a
 * collector because the engine fires a bus event for them; an extra turn, a
 * prevented damage, a chosen colour reach nobody, and the corpus recorded those
 * abilities as abilities that did nothing. What every one of them does carry is
 * its own {@code SpellAbility}: the parameters it was scripted with, and for
 * some, a result remembered on the host.
 *
 * <p>Keyed by API name rather than by {@code ApiType} so a Forge rename costs a
 * missing event rather than a compile error, and so this file needs no import
 * of an enum whose members move. The names are pinned by
 * {@code test_every_effect_api_maps_to_an_event} on the Python side.
 *
 * <p>An API absent from this table emits nothing. That is the right default:
 * damage, counters, taps and zone changes already arrive on the bus, and a
 * second event for them would double-count an outcome.
 */
final class ApiEvents {

    private ApiEvents() {
    }

    /** What an emitter needs from before the clause ran, or null when nothing. */
    @FunctionalInterface
    private interface Memo {
        Object take(SpellAbility sa, Card host);
    }

    /** The event, given the clause and whatever its memo captured. */
    @FunctionalInterface
    private interface Emitter {
        EffectEvent emit(SpellAbility sa, Card host, Object memo);
    }

    /** @param eventType named in the once-only failure line an emitter's throw prints */
    private record Rule(String eventType, Memo memo, Emitter emitter) {
    }

    /**
     * Damage on the clause's targets, before it heals any of it away.
     *
     * <p>Declared ahead of {@link #RULES} on purpose: a static field read by
     * simple name from another field's own initializer has to be declared
     * first or the reference is an illegal forward reference and the class
     * does not compile at all -- unlike the emitter lambdas below, which call
     * methods declared later in this file with no such restriction.
     */
    private static final Memo DAMAGE_BEFORE = (sa, host) -> {
        int total = 0;
        if (sa.getTargets() != null) {
            for (Card card : sa.getTargets().getTargetCards()) {
                total += card.getDamage();
            }
        }
        return total;
    };

    private static final Map<String, Rule> RULES = Map.ofEntries(
            Map.entry("AddTurn", new Rule(EffectEvent.TURN_ADDED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.TURN_ADDED)
                                    .param("count", amount(sa, host, "NumTurns", "1")),
                            memo))),
            Map.entry("SkipTurn", new Rule(EffectEvent.TURN_SKIPPED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.TURN_SKIPPED)
                                    .param("count", amount(sa, host, "NumTurns", "1")),
                            memo))),
            Map.entry("AddPhase", new Rule(EffectEvent.PHASE_ADDED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_ADDED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                            .param("count", amount(sa, host, "NumPhases", "1")))),
            Map.entry("SkipPhase", new Rule(EffectEvent.PHASE_SKIPPED,
                    (sa, host) -> affectedPlayers(sa),
                    (sa, host, memo) -> withSubjects(
                            new EffectEvent(EffectEvent.PHASE_SKIPPED)
                                    .param("phase", sa.getParamOrDefault("Phase", "?")),
                            memo))),
            Map.entry("ChangeX", new Rule(EffectEvent.X_CHANGED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.X_CHANGED)
                            .param("value", amount(sa, host, "Value", "0")))),
            Map.entry("GainOwnership", new Rule(EffectEvent.OWNERSHIP_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.OWNERSHIP_CHANGE)
                            .param("owner", ownerOf(sa)))),

            // ── Task 8: choices, healed damage, granted abilities, retargets ──
            //
            // Ruling R18 (task-8-brief.md) corrected two keys below that were NOT
            // spelled like their effect classes. Verified against ApiType.java
            // before use:
            //   NameCard      (ApiType.java:135) declares ChooseCardNameEffect.class
            //   GenericChoice (ApiType.java:109) declares ChooseGenericEffect.class
            // A rule keyed by the CLASS stem matches nothing and fails silently.
            //
            // R18 also added a ChooseSector entry here, on the reasoning that it fits
            // choice_made and that leaving it out would be a voluntary omission. That
            // check covered ApiType.java and the Card accessor but never the Python
            // schema, which already had the answer -- see the comment below
            // GenericChoice, where that entry would otherwise sit. Fix round 1 removed
            // it.
            //
            // The brief's sample code for every entry below constructed `Rule` with
            // only two arguments (a memo-or-null, then the emitter), omitting the
            // eventType this record actually declares first -- that does not compile
            // against the real three-argument Rule above, so each entry here supplies
            // the real constant the same way every pre-existing rule already does.
            //
            // GenericChoice, ChooseDirection, ChooseEvenOdd and NameCard are not
            // optional extras needing a judgment call: all four are present in
            // event_schema.py's EFFECT_API_EVENTS (mapped to CHOICE_MADE) and absent
            // from EXCLUDED_EFFECT_APIS, so the schema itself declares that these APIs
            // produce choice_made. A rule that matches here and reads nothing back
            // would be worse than one left out of this table entirely -- it looks
            // wired. Their choice() read branches, below, are required, not optional.
            Map.entry("ChooseColor", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseType", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseCard", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("NameCard", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseNumber", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChoosePlayer", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseDirection", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseEvenOdd", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseSource", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),
            Map.entry("GenericChoice", new Rule(EffectEvent.CHOICE_MADE, null,
                    (sa, host, memo) -> choice(sa, host))),

            // ChooseSector is deliberately NOT wired here. Ruling R18 added it after
            // checking ApiType.java (a real member, ApiType.java:54) and
            // Card.getChosenSector() (a real accessor -- ChooseSectorEffect.java:13
            // calls setChosenSector) -- but never checked the Python schema, which
            // already had the answer: event_schema.py's EXCLUDED_EFFECT_APIS carries
            //   "ChooseSector": "Unfinity attraction; not reachable in sealed or draft"
            // with ChooseSector correspondingly absent from EFFECT_API_EVENTS. This
            // corpus is collected from sealed and draft pools only, which never
            // include Unfinity's silver-bordered attraction cards, so the clause this
            // would read can never actually run here. If ChooseSector caught your eye
            // in ApiType.java and you are about to re-add it the way R18 did: don't,
            // without re-checking the schema first -- everyChoiceMadeApiHasAReadableChoice
            // in ApiEventsTest will fail the moment you do, which is the point.
            Map.entry("HealDamage", new Rule(EffectEvent.DAMAGE_HEALED, DAMAGE_BEFORE,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.DAMAGE_HEALED)
                            .param("amount", memo instanceof Integer n ? n : 0))),
            Map.entry("Animate", new Rule(EffectEvent.ABILITY_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("AnimateAll", new Rule(EffectEvent.ABILITY_CHANGE, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("Effect", new Rule(EffectEvent.CONTINUOUS_EFFECT_CREATED, null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.CONTINUOUS_EFFECT_CREATED)
                            .param("layers", sa.getParamOrDefault("StaticAbilities", "")))),
            Map.entry("ChangeTargets", new Rule(EffectEvent.TARGETS_CHANGED, null,
                    (sa, host, memo) -> retargeted(sa)))
    );

    /** What the emitter for this clause needs from before it runs, if anything. */
    static Object before(SpellAbility sa) {
        Rule rule = ruleFor(sa);
        if (rule == null || rule.memo() == null) {
            return null;
        }
        return rule.memo().take(sa, sa.getHostCard());
    }

    /** The event this clause promises, or null when its API promises none. */
    static EffectEvent after(SpellAbility sa, Object memo) {
        Rule rule = ruleFor(sa);
        if (rule == null) {
            return null;
        }
        try {
            return rule.emitter().emit(sa, sa.getHostCard(), memo);
        } catch (RuntimeException e) {
            // An emitter reads engine state that a fizzled or redirected clause
            // may have left in a shape it did not expect. A missing event is a
            // gap; a thrown one would end the resolution the observer is only
            // watching -- so the throw stays swallowed. But swallowed and
            // "this mechanic never happens" must not look identical, which is
            // the failure this whole plan exists to end, so the first one is
            // not silent.
            reportEmitterFailure(rule.eventType(), e);
            return null;
        }
    }

    /** Set the first time any rule's emitter throws, so the line below prints once. */
    private static final AtomicBoolean EMITTER_FAILURE_REPORTED = new AtomicBoolean(false);

    /**
     * Mirror the treatment {@code AbilityUtils.reportClauseListenerFailure}
     * gives this same hook's other end, in the forge repo: latched to the
     * first throw, naming what failed and why, rather than repeated once per
     * clause -- a real playthrough resolves thousands of them, and a broken
     * rule would flood the log for the whole run instead of announcing itself
     * once.
     *
     * <p>Package-private rather than private, the same reason {@code
     * clauseHandler} is: forcing a rule to throw reliably would mean
     * contriving a malformed script value or a null host card, and this is
     * what {@code AbilityUtils} does instead for the hook's other end --
     * {@code EffectRecordClauseHookTest} calls {@code
     * reportClauseListenerFailure} through a seam, not through a real throw.
     */
    static void reportEmitterFailure(String eventType, RuntimeException e) {
        if (EMITTER_FAILURE_REPORTED.compareAndSet(false, true)) {
            System.err.println("ApiEvents: the emitter for \"" + eventType
                    + "\" threw and was ignored; further emitter failures will "
                    + "not be logged: " + e);
        }
    }

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void resetEmitterFailureLatchForTest() {
        EMITTER_FAILURE_REPORTED.set(false);
    }

    /**
     * Package-private seam for {@code ApiEventsTest}'s {@code choice_made}
     * completeness guard: every live {@code RULES} key whose declared event
     * type is {@code choice_made}, read off the table itself rather than
     * hand-copied, so a new one wired in later is picked up automatically
     * instead of the guard quietly checking a stale list.
     */
    static java.util.Set<String> choiceMadeApiKeysForTest() {
        java.util.Set<String> keys = new java.util.TreeSet<>();
        for (Map.Entry<String, Rule> entry : RULES.entrySet()) {
            if (EffectEvent.CHOICE_MADE.equals(entry.getValue().eventType())) {
                keys.add(entry.getKey());
            }
        }
        return keys;
    }

    private static Rule ruleFor(SpellAbility sa) {
        if (sa == null || sa.getApi() == null) {
            return null;
        }
        return RULES.get(sa.getApi().name());
    }

    private static int amount(SpellAbility sa, Card host, String key, String fallback) {
        return AbilityUtils.calculateAmount(host, sa.getParamOrDefault(key, fallback), sa);
    }

    private static String playerOf(SpellAbility sa) {
        return sa.getActivatingPlayer() == null
                ? null : SnapshotBuilder.playerId(sa.getActivatingPlayer());
    }

    /**
     * Every player {@code AddTurn}, {@code SkipTurn} and {@code SkipPhase}
     * actually affect -- the ones their own {@code resolve()} iterates, not
     * whoever activated them.
     *
     * <p>{@code AddTurnEffect.resolve}, {@code SkipTurnEffect.resolve} and
     * {@code SkipPhaseEffect.resolve} all iterate {@code getTargetPlayers(sa)}
     * — {@code SpellAbilityEffect.getPlayers(false, "Defined", sa)}, not
     * callable from here since it is private to a different package's class —
     * where {@code definedFirst=false} means <b>targeting wins over
     * {@code Defined$}</b> whenever the ability targets at all. The activator
     * is not a safe default: {@code empty_city_ruse.txt} and
     * {@code stonehorn_dignitary.txt} target {@code Opponent},
     * {@code blinding_angel.txt} reads {@code Defined$ TriggeredTarget}, and
     * of roughly eighteen real {@code SkipPhase} scripts, sixteen name a
     * player other than whoever cast or activated the ability.
     *
     * <p>Captured as a memo, before the clause resolves, rather than
     * re-read in {@link #after}: none of these three effects' {@code
     * resolve()} mutates {@code sa}'s own targets today, so reading either
     * side of resolution gives the same answer, but capturing before is the
     * pattern the rest of this codebase already uses for exactly this shape
     * of risk, and it is what lets the clause hook's push/pop pairing —
     * otherwise exercised only on its always-empty path by every other rule
     * here — be pinned against a real, non-trivial value instead of null.
     */
    private static List<Player> affectedPlayers(SpellAbility sa) {
        if (sa.usesTargeting()) {
            List<Player> targeted = new ArrayList<>();
            sa.getTargets().getTargetPlayers().forEach(targeted::add);
            return targeted;
        }
        return definedPlayers(sa, sa.getParamOrDefault("Defined", "You"));
    }

    /**
     * One subject per affected player, in the order the memo carries them.
     *
     * <p>{@code getTargetPlayers} returns every target, and {@code AddTurn}
     * naming more than one is a real script shape (a duel-taker rewarding
     * both duelists, say) -- collapsing to the first would silently drop
     * every player after it, the same failure shape as reading the wrong
     * player entirely.
     */
    private static EffectEvent withSubjects(EffectEvent event, Object memo) {
        if (memo instanceof List<?> players) {
            for (Object player : players) {
                if (player instanceof Player p) {
                    event.subject(SnapshotBuilder.playerId(p));
                }
            }
        }
        return event;
    }

    /**
     * The new owner {@code GainOwnership} promises: a named player where the
     * clause names one, the activating player otherwise.
     *
     * <p>{@code OwnershipGainEffect.resolve} reads {@code DefinedPlayer} first
     * and falls back to {@code sa.getActivatingPlayer()} only when that
     * resolves to nobody. A purely parameter-based rule that always read the
     * activating player would name the wrong owner on a real card: Tempest
     * Efreet's second clause is {@code DB$ GainOwnership | Defined$
     * CorrectedSelf | DefinedPlayer$ Targeted}, which hands the card to the
     * opponent it targeted, not to whoever sacrificed the Efreet; Bronze
     * Tablet's is {@code DefinedPlayer$ Remembered}, naming a player chosen
     * three clauses earlier. Both are real printed cards, not hypotheticals.
     */
    private static String ownerOf(SpellAbility sa) {
        if (sa.hasParam("DefinedPlayer")) {
            List<Player> named = definedPlayers(sa, sa.getParam("DefinedPlayer"));
            if (!named.isEmpty()) {
                return SnapshotBuilder.playerId(named.get(0));
            }
        }
        return playerOf(sa);
    }

    /**
     * {@code AbilityUtils.getDefinedPlayers} for one {@code Defined$}/
     * {@code DefinedPlayer$} value, extended to the {@code " & "}-joined
     * multi-value form {@code SpellAbilityEffect.getPlayers} also accepts
     * ({@code SpellAbilityEffect.java:314}): {@code String[] def =
     * sa.getParamOrDefault(definedParam, "You").split(" & ")}, one lookup per
     * token, unioned. No card in the cardsfolder uses this today for either
     * {@code Defined$} on these three rules or {@code DefinedPlayer$} on
     * {@code GainOwnership}, but this plan's accuracy constraint does not
     * permit an approximation that is this cheap to close.
     */
    private static List<Player> definedPlayers(SpellAbility sa, String def) {
        List<Player> players = new ArrayList<>();
        for (String token : def.split(" & ")) {
            players.addAll(AbilityUtils.getDefinedPlayers(sa.getHostCard(), token, sa));
        }
        return players;
    }

    /**
     * The choice a Choose* clause left on its host, as a kind and a value.
     *
     * <p>Reads whichever dedicated accessor the clause populated, in a fixed
     * priority order -- the API that actually ran is not consulted, only what
     * state it left behind, so a choice left on the host by an earlier clause
     * in the same chain can in principle outrank a later clause that
     * legitimately produced no reading of its own. This mirrors the
     * approximation {@link #ownerOf} already accepts for {@code
     * GainOwnership}'s single-scalar shape rather than introducing a new one.
     *
     * <p><b>{@code getChosenColor()} and {@code getNamedCard()} never return
     * {@code null}</b> -- an unset colour reads back as {@code ""} ({@code
     * Card.getChosenColor}), the same way an unset named card reads back as
     * {@code ""} rather than {@code null}. A bare {@code != null} check
     * against either would always be true, which for the colour branch --
     * checked first -- would report <i>every</i> clause routed through this
     * method as a colour choice, direction and number and all. {@code
     * hasChosenColor()} is what {@code Card} itself defines for this
     * distinction, so this reads that instead; the named-card branch was
     * already guarded correctly by its paired {@code !isEmpty()} check.
     *
     * <p>The direction, even/odd and mode branches are not in the brief's
     * sample for this method, though the {@code ChooseDirection}, {@code
     * ChooseEvenOdd} and {@code GenericChoice} rules above all route through
     * it. These are not optional extras: {@code event_schema.py}'s {@code
     * EFFECT_API_EVENTS} maps all three (and {@code NameCard}) to {@code
     * CHOICE_MADE} and excludes none of them, so the schema itself declares
     * that these APIs produce {@code choice_made} -- a rule that matches and
     * then reads nothing back would be worse than one left out of {@code
     * RULES} entirely, because it looks wired. {@code
     * ChooseDirectionEffect.resolve()} always calls {@code
     * setChosenDirection}, and {@code ChooseEvenOddEffect.resolve()} always
     * calls {@code setChosenEvenOdd} -- the identical "dedicated accessor
     * that stays there" shape as the six branches already read. {@code
     * GenericChoice} only reaches its branch for the {@code SetChosenMode$
     * True} scripts (Tarkir's Sieges, Fallout's Hoover Dam, 15 real cards);
     * the rest resolve a chosen sub-ability directly and leave no scalar on
     * the host to describe which one -- the schema still declares the API,
     * so the branch is what makes the reachable subset of it reachable.
     *
     * <p>A {@code ChooseSector} branch reading {@code getChosenSector()}
     * lived here briefly too. Removed along with the {@code RULES} entry
     * (see the comment above {@code HealDamage}) once it turned out {@code
     * event_schema.py} already excludes {@code ChooseSector} as unreachable
     * in this corpus's sealed/draft pools -- so the accessor is still real,
     * but nothing in {@code RULES} routes to this method carrying that API
     * any more, and the branch would have been dead weight at best and a
     * stale-state misattribution risk at worst (a host that happened to
     * carry a leftover {@code chosenSector} from something else entirely
     * would have outranked a later, real choice on the same card).
     */
    private static EffectEvent choice(SpellAbility sa, Card host) {
        if (host == null) {
            return null;
        }
        String kind = null;
        String value = null;
        if (host.hasChosenColor()) {
            kind = "color";
            value = host.getChosenColor();
        } else if (host.getChosenType() != null && !host.getChosenType().isEmpty()) {
            kind = "type";
            value = host.getChosenType();
        } else if (host.getChosenNumber() != null) {
            kind = "number";
            value = String.valueOf(host.getChosenNumber());
        } else if (host.getNamedCard() != null && !host.getNamedCard().isEmpty()) {
            kind = "card_name";
            value = host.getNamedCard();
        } else if (host.getChosenPlayer() != null) {
            kind = "player";
            value = SnapshotBuilder.playerId(host.getChosenPlayer());
        } else if (host.getChosenCards() != null && !host.getChosenCards().isEmpty()) {
            kind = "cards";
            value = String.valueOf(host.getChosenCards().size());
        } else if (host.getChosenDirection() != null) {
            kind = "direction";
            value = host.getChosenDirection().name();
        } else if (host.getChosenEvenOdd() != null) {
            kind = "even_odd";
            value = host.getChosenEvenOdd().name();
        } else if (host.getChosenMode() != null && !host.getChosenMode().isEmpty()) {
            kind = "mode";
            value = host.getChosenMode();
        }
        if (kind == null) {
            // The clause chose something this has no reading for. An event
            // naming the choice kind "?" would train the head on a distinction
            // the corpus cannot make.
            return null;
        }
        return new EffectEvent(EffectEvent.CHOICE_MADE)
                .param("choice_kind", kind)
                .param("value", value);
    }

    /**
     * What the retargeted spells point at now.
     *
     * <p>Read after the clause rather than reported by it, because a spell it
     * retargeted is still on the stack carrying its new {@code TargetChoices}
     * -- this is the one outcome in this group that survives on an object
     * other than the host.
     */
    private static EffectEvent retargeted(SpellAbility sa) {
        List<String> targets = new ArrayList<>();
        if (sa.getTargets() != null) {
            for (SpellAbility changed : sa.getTargets().getTargetSpells()) {
                if (changed.getTargets() == null) {
                    continue;
                }
                for (Card card : changed.getTargets().getTargetCards()) {
                    targets.add(SnapshotBuilder.entityId(card));
                }
            }
        }
        return new EffectEvent(EffectEvent.TARGETS_CHANGED).param("targets", targets);
    }
}
