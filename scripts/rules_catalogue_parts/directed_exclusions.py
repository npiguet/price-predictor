"""The rules enabled for directed forks whatever their counts, and those left out."""

#: Fewer `fires` records than this over all sources makes a rule rare (FR-014b).
RARE_FIRES = 100
#: Enabled for directed forks whatever their counts, each for its reason (FR-014b).
ALWAYS_DIRECTED = {
    "704.5f": "the directed forks' reference rule, which the first pilot checks (the user's T038 "
              "decision of 2026-11-02); its probability follows its counts like any listed rule's",
    "702.98a": "its 8 does_not_fire records on gen-2 are one Dead Reveler in one game, far "
               "below the 68 per polarity a rule test needs (the user's decision of 2026-11-06)",
}
#: Left out whatever their counts, each for its reason: no detector can tell
#: their fork from its live game, no fork of their board can be taken or kept,
#: or no sealed pool makes their point happen, so a directed fork of theirs
#: would add no record of the side it is for (the user's decisions of
#: 2026-11-06, T096, and 2026-11-14).
NOT_DIRECTED = {
    "104.2a": "the AI never concedes, the only forkable variant (add Platinum Angel) stops "
              "any other loss, and the lacking polarity is no event",
    "500.11": "500.11-1 (Eon Hub) has no detector, a static skip filing no event, and "
              "500.11-2's contrast is a choice",
    "701.12d": "701.12d-1 has no detector and its only forkable variant is polarity other; "
               "701.12d-2's contrast is ownership, not a board edit",
    "702.62d": "the suspend cast happens inside a resolution, never at a priority stop, so "
               "no playability record exists for it",
    **{rule: f"restates {other}, which is listed in its place (the user's decision of "
             "2026-11-14): its scenario declares the same fact as " + other + "'s"
       for rule, other in {"121.4": "704.5b", "703.4m": "510.2"}.items()},
    "701.42c": "restates 701.42b (the user's decision of 2026-11-14): its scenario declares the "
               "same fact as 701.42b's, which is left out for its own reason",
    "122.1g": "restates 704.5v (the user's decision of 2026-11-14): its scenario declares the "
              "same fact as 704.5v's, which is left out for its own reason",
    **dict.fromkeys(("505.5", "614.10a", "701.24f", "702.105b"),
                    "the record cannot show the rule's fact, so its scenarios declare no "
                    "detector (the user's decision of 2026-11-14)"),
    "122.1h": "the rewrite record names no replacing effect for a replacement Forge builds at "
              "runtime from the finality counter, so its scenario declares no detector (the "
              "user's decision of 2026-11-14)",
    "104.4c": "no record shows the draw: Forge's GameDrawn ends the game naming no loser and "
              "no winner and no record kind lists game_drawn, so 104.4c-1's declaration is "
              "untaggable; the effect half's conditions_met says whether the GameDrawn "
              "clause's condition held, but no detector family reads it",
    "115.7a": "its point is a target-changing spell or ability resolving, cast by a player "
              "at a spell on the stack after the boundary (every boundary a fork is taken at "
              "has the stack empty), so no precondition knows the spell or the other legal "
              "target its variant replaces",
    **dict.fromkeys(("701.10c", "701.11c"),
                    "its scenario declares no detector: no detector parameter reads whether "
                    "the power doubled or tripled was negative"),
    "712.4c": "its scenario declares no detector: no detector parameter reads why a "
              "transform instruction did nothing",
    **dict.fromkeys(("701.40c", "701.40d"),
                    "its variant puts another card in the manifested permanent's place, and "
                    "replace_card refuses every face-down permanent, so every fork would be "
                    "refused"),
    "712.16": "no sealed pool makes it happen: no Forge edition with a booster (play-booster "
              "sets included) holds both a card that turns a permanent face down (SetState "
              "TurnFaceDown: Ixidron and Vesuvan Shapeshifter in Time Spiral, Illithid "
              "Harvester in Commander Legends: Battle for Baldur's Gate, Backslide and "
              "Mischievous Quanar in Onslaught block) and a double-faced card; disguise, "
              "cloak, manifest and morph "
              "put cards onto the battlefield face down, which turns no permanent face down",
    # No single board edit yields the side the counts lack (the user's decision of
    # 2026-11-14, DF-A): each reason says why.
    "104.6": "its lacking side is a game restart, which every fork refuses before it "
             "resolves (Forge's restart resets the JVM-wide trigger id counter); its variant, "
             "Karn Liberated at 13 loyalty, writes no restart resolution",
    "400.6": "its lacking side is a zone change that meets a replacement and is not replaced: "
             "Forge's hooks fire only for a replacement that applies, so no board edit writes "
             "that rewrite record; its variant, no Rest in Peace, writes no rewrite at all",
    "118.11": "its lacking side is a cost's move to a graveyard that meets a replacement and is "
              "not replaced: Forge's hooks fire only for a replacement that applies, so no board "
              "edit writes that rewrite record; its variant, Rest in Peace added, writes the fires "
              "side, which the counts already hold (the user's decision of 2026-11-15)",
    "122.1d": "its lacking side is an untap that meets a stun counter's replacement and is not "
              "replaced, which never happens: the replacement applies to every untap of a "
              "permanent with a stun counter; its variant, no stun counter, writes no rewrite at "
              "all, and a second stun counter is still replaced",
    "702.19e": "its lacking side, an attacker with trample over planeswalkers whose planeswalker "
               "left combat, is what 702.19f's directed fork writes (Thrasta, Tempest's Roar in "
               "the place of an attacker without the keyword, on the same boards), whose combat "
               "records its declaration tags fires; a fork of its own would fork those boards "
               "twice",
    "701.42b": "its lacking side is a meld that fails because a partner is a token, and no "
               "operation makes a token or a copy (replace_card and add_permanent put cards, so "
               "its one classified variant, the card in the token's place, makes the meld the "
               "counts hold)",
    "704.7": "its lacking side is a game loss that meets a replacement and is not replaced: "
             "Forge's hooks fire only for a replacement that applies, so no board edit writes "
             "that rewrite record; its variant, no Lich's Mirror, writes no rewrite at all",
    "701.24g": "its scenario's declaration counts no near miss (the shuffle and the placement "
               "come from replacements, so no api narrows the detector), so no fork writes "
               "its lacking side",
    "702.35a": "its lacking side is a discard that meets madness's replacement and is not "
               "replaced: Forge's hooks fire only for a replacement that applies; its variant, "
               "Shock in Fiery Temper's place, writes no rewrite at all",
    "702.167a": "its lacking side is a craft ability that resolves and returns nothing, which "
                "needs the card gone after the activation; its variant, no material, leaves "
                "craft nothing to activate",
    "111.10n": "its lacking side is a Sorcerer Role's static contributing no boost, which the "
               "Role always gives; its variant creates no Role, so its fork writes no record "
               "of a Role's static",
    "704.5v": "its lacking side is a battle at zero defense put into the graveyard by the "
              "state-based check, which needs its defeat trigger countered in response (Stifle "
              "after the damage); a defeated Siege's own trigger exiles it",
    "730.2g": "its point needs a merge, a turn face down and a turn-face-up effect in one turn "
              "(Soul Summons, Cloudpiercer, Ixidron, Break Open): no single board edit or play "
              "brings them together",
    "707.10d": "its point needs a spell copied for each target, cast in response to another "
               "spell on the stack, which no fork taken at a phase boundary, its stack empty, "
               "plays",
    "712.11d": "its only classified variant removes the Disturb candidate its declaration reads "
               "(Hanweir Watchkeep in Beloved Beggar's place), so its fork writes no record of "
               "the rule; a one-land-short variant would make the near miss",
    "701.12h": "its classified variant is polarity other (two vanilla creatures) and its near "
               "miss needs the exchange's target destroyed in response; its fires side needs "
               "Exchange of Words's trigger targets chosen, which Forge's AI has no logic for",
    "702.70b": "its declaration reads a trigger whose line is poisonous's own, and no record has "
               "one: no Forge card prints poisonous (Virulent Sliver and Snake Cult Initiation "
               "grant it through a static, so its trigger is keyed to the static's line), and a "
               "keyword a fork's keyword edit grants keys to no line; played, a forced attack by "
               "Grizzly Bears given poisonous 1 fires the trigger and poisons the player, and "
               "its record is tagged neither side",
    "119.8": "119.8-2's scenario declares no detector, and 119.8-1's declaration reads a "
             "can't-be-cast static the decision candidate names, which a can't-lose-life "
             "static (Platinum Emperion's) never is: the record names no static for a cost the "
             "static makes unpayable",
    "702.62c": "the record cannot show the prohibition: a suspend candidate a can't-be-cast "
               "static forbids fails only Forge's own restriction check (script_restriction), "
               "and the collector names a prohibiting static for spells and activated "
               "abilities alone, never a special action",
    "702.154a": "the record cannot show the near miss: Forge's enlist is an optional attack "
                "cost (a static) whose reflexive trigger exists only once the cost is paid, so "
                "no enlist trigger is evaluated where nothing is enlisted",
    "702.85a": "the record cannot show the hit: Forge casts the cascaded card as a cast of its "
               "own after the cascade trigger resolves, so the trigger's effect half holds the "
               "cards exiled and no move from exile to the stack, which is in the cast's own "
               "record, naming no cascade; no declaration tells a cascade that hit from one that "
               "missed",
    "702.102c": "the record cannot tell a fused cast from its left half's: both candidates act "
                "through the split card's first spell line, so no declaration reads the fused "
                "cast's affordability apart from Give's",
}
