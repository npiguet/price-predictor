package com.pricepredictor.connector.ability;

import forge.game.keyword.KeywordInterface;

/** A Class keyword's level, read from {@code Class:<level>:<cost>:...}. */
record ClassLevel(int number) {

    static ClassLevel of(KeywordInterface ki) {
        return new ClassLevel(Integer.parseInt(new KeywordSpec(ki.getOriginal()).part(1)));
    }
}
