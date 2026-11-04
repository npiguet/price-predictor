"""``scan_keyword_flags.py`` and the module it writes (T110, M5, FR-044a)."""

from __future__ import annotations

import pytest
import scan_keyword_flags as skf

from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS

#: The evasion keywords Forge names in a selector or a check: gate 2 strips the
#: damage-step keywords and the grids edit these through the flag (T118). Fear and
#: intimidate are absent: Forge implements each as its own generated static, which
#: names no keyword, and no card script selects either, so their lines carry them.
EVASION_FLAGS = frozenset({
    "flying", "reach", "horsemanship", "shadow", "skulk", "menace", "landwalk", "defender",
})
REQUIRED = frozenset(row.keyword for row in DAMAGE_STEP_KEYWORDS) | EVASION_FLAGS


def test_the_eight_damage_step_keywords_and_the_evasion_keywords_keep_their_flags():
    assert len({row.keyword for row in DAMAGE_STEP_KEYWORDS}) == 8
    assert REQUIRED <= set(KEPT_KEYWORD_FLAGS), sorted(REQUIRED - set(KEPT_KEYWORD_FLAGS))


def test_protection_and_split_second_keep_their_flags():
    assert {"protection", "split_second"} <= set(KEPT_KEYWORD_FLAGS)


def test_the_list_is_sorted_and_in_the_package_s_spelling():
    assert list(KEPT_KEYWORD_FLAGS) == sorted(set(KEPT_KEYWORD_FLAGS))
    assert all(name == name.lower() and " " not in name for name in KEPT_KEYWORD_FLAGS)


# --- the scan, over a small Forge tree ---------------------------------------------------

_KEYWORD_JAVA = """\
public enum Keyword {
    UNDEFINED("", SimpleKeyword.class, false, ""),
    DEATHTOUCH("Deathtouch", SimpleKeyword.class, true, "r"),
    FEAR("Fear", SimpleKeyword.class, true, "r"),
    FIRST_STRIKE("First Strike", SimpleKeyword.class, true, "r"),
    FLASH("Flash", SimpleKeyword.class, true, "r"),
    FLASHBACK("Flashback", KeywordWithCost.class, false, "r"),
    FLYING("Flying", SimpleKeyword.class, true, "r"),
    REACH("Reach", SimpleKeyword.class, true, "r"),
    SPLIT_SECOND("Split second", SimpleKeyword.class, true, "r"),
    TRAMPLE("Trample", Trample.class, true, "r"),
    WARD("Ward", Ward.class, false, "r"),
    WITHER("Wither", SimpleKeyword.class, true, "r"),
    ;
}
"""

_CARD = """\
Name:Fearless Flyer
Oracle:Destroy target creature withFear. Creatures withWard can't block.
A:SP$ Destroy | ValidTgts$ Creature.withFlying+withoutFirst Strike | SpellDescription$ x
S:Mode$ Continuous | Affected$ Card.withFlashback | AddKeyword$ Ward:2
T:Mode$ Attacks | ValidCard$ Creature.hasKeywordFlash | Execute$ X
SVar:X:Count$Valid Creature.withMostLife
"""

_ENGINE = """\
class CombatUtil {
    boolean a(Card c) { return c.hasKeyword(Keyword.DEATHTOUCH); }
    boolean b() { return hasKeyword(Keyword.WITHER); }
    boolean c(Card c) { return c.hasKeyword("Trample:Planeswalker"); }
    boolean d(KeywordInterface inst) { return inst.getKeyword() == Keyword.SPLIT_SECOND; }
    String e = "Mode$ CantBlockBy | ValidBlocker$ Creature.withoutReach";
    String f = "CARDNAME can't attack or block alone.";
}
"""

_VIEW = 'class CardView { boolean w(Card c) { return c.hasKeyword(Keyword.WARD); } }\n'


@pytest.fixture
def forge(tmp_path):
    (tmp_path / skf.KEYWORD_ENUM).parent.mkdir(parents=True)
    (tmp_path / skf.KEYWORD_ENUM).write_text(_KEYWORD_JAVA, encoding="utf-8")
    (tmp_path / skf.CARD_SCRIPTS / "f").mkdir(parents=True)
    (tmp_path / skf.CARD_SCRIPTS / "f" / "fearless_flyer.txt").write_text(_CARD,
                                                                            encoding="utf-8")
    engine = tmp_path / skf.ENGINE / "forge" / "game"
    engine.mkdir(parents=True, exist_ok=True)
    (engine / "CombatUtil.java").write_text(_ENGINE, encoding="utf-8")
    (engine / "CardView.java").write_text(_VIEW, encoding="utf-8")
    return tmp_path


def test_the_scan_keeps_what_a_selector_or_an_engine_check_names(forge):
    assert skf.scan(forge) == {
        "deathtouch": (0, 1),       # hasKeyword(Keyword.DEATHTOUCH)
        "first_strike": (1, 0),     # withoutFirst Strike: a name holding a space
        "flash": (1, 0),            # hasKeywordFlash
        "flashback": (1, 0),        # withFlashback names flashback, not flash
        "flying": (1, 0),
        "protection": (0, 0),       # always kept
        "reach": (0, 1),            # a selector in an engine string
        "split_second": (0, 1),     # getKeyword() == Keyword.SPLIT_SECOND
        "trample": (0, 1),          # hasKeyword("Trample:Planeswalker")
        "wither": (0, 1),           # hasKeyword on the card itself
    }


def test_prose_a_granted_keyword_and_the_ui_s_views_keep_no_flag(forge):
    kept = skf.scan(forge)
    # Fear only in the Oracle line, ward only granted (AddKeyword$) and checked by a view.
    assert "fear" not in kept and "ward" not in kept


def test_the_written_module_lists_the_scan_s_names(forge, tmp_path):
    output = tmp_path / "kept.py"
    skf.main(["--forge", str(forge), "--output", str(output)])
    namespace: dict = {}
    exec(output.read_text(encoding="utf-8"), namespace)  # noqa: S102 - our own output
    assert namespace["KEPT_KEYWORD_FLAGS"] == tuple(skf.scan(forge))


# --- the real Forge checkout ------------------------------------------------------------

_REAL = pytest.mark.skipif(not (skf.FORGE / skf.KEYWORD_ENUM).is_file(),
                           reason="no sibling ../forge checkout")


@pytest.fixture(scope="module")
def real_scan():
    return skf.scan(skf.FORGE)


@_REAL
def test_a_scan_of_forge_keeps_every_required_flag(real_scan):
    assert REQUIRED <= set(real_scan), sorted(REQUIRED - set(real_scan))


@_REAL
def test_the_checked_in_module_is_the_scan_of_forge(real_scan):
    """Rerun ``python scripts/scan_keyword_flags.py`` after a Forge merge."""
    assert skf.render(real_scan) == skf.OUTPUT.read_text(encoding="utf-8")
