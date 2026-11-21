package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.player.Player;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;

/** One entity's share of one static's effect, accumulated across layers. */
final class Contribution {
    private final String entity;
    private int power;
    private int toughness;
    // The base a setting effect gives, each half null where the static
    // sets none; null altogether for a static that sets no base.
    private List<Integer> setPt;
    // The controller a static gives, or null for one that changes none.
    private String controller;
    private final TokenLine keywords = TokenLine.keywords();
    private final TokenLine types = TokenLine.distinct();
    private final TokenLine colors = TokenLine.distinct();
    // Unlike the token lines, a name contribution is a single value:
    // a static either overwrites the name or it does not, so the last
    // write standing is the answer rather than an accumulation. Null for
    // the statics that are not one of the handful that rename anything --
    // that is the correct, meaningful answer, not a missing one.
    private String name;
    // Layer 6: each ability granted by its key, each taken away as "-key".
    private final TokenLine abilities = TokenLine.everyToken();
    // Forge's hidden keywords the static grants, as Forge spells them.
    private final TokenLine hiddenKeywords = TokenLine.everyToken();

    Contribution(String entity) {
        this.entity = entity;
    }

    /** Add one P/T boost cell's (power, toughness). */
    void boost(Pair<Integer, Integer> added) {
        power += added.getLeft();
        toughness += added.getRight();
    }

    /** Set the base from one set-P/T cell; a half it leaves null keeps the half set before. */
    void setPt(Pair<Integer, Integer> base) {
        Integer setPower = base.getLeft();
        Integer setToughness = base.getRight();
        List<Integer> next = new ArrayList<>(2);
        next.add(setPower != null ? setPower : setPt == null ? null : setPt.get(0));
        next.add(setToughness != null ? setToughness : setPt == null ? null : setPt.get(1));
        setPt = next;
    }

    void controller(Player player) {
        controller = Refs.of(player);
    }

    void keyword(LayerToken token) {
        keywords.add(token);
    }

    void type(LayerToken token) {
        types.add(token);
    }

    void color(LayerToken token) {
        colors.add(token);
    }

    void ability(LayerToken token) {
        abilities.add(token);
    }

    void hiddenKeyword(LayerToken token) {
        hiddenKeywords.add(token);
    }

    void name(String value) {
        if (value != null && !value.isEmpty()) {
            name = value;
        }
    }

    /**
     * {@code set_pt} and {@code controller} only where the static sets a
     * base or gives control: absent says it does neither, as on a record
     * written before the two members existed.
     */
    String toJson() {
        Json.ObjectWriter json = Json.object()
                .value("entity", entity)
                .value("pt_boost", List.of(power, toughness));
        if (setPt != null) {
            json.value("set_pt", setPt);
        }
        json.raw("keywords", keywords.json())
                .raw("types", types.json())
                .raw("colors", colors.json())
                .raw("abilities", abilities.json())
                .raw("hidden_keywords", hiddenKeywords.json())
                .value("name", name);
        if (controller != null) {
            json.value("controller", controller);
        }
        return json.render();
    }
}
