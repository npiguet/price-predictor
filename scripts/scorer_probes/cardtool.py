"""Helper: inspect Forge edition set lists + converted card corpus."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator  # noqa: E402

EDITIONS = Path(__file__).resolve().parents[2].parent / "forge" / "forge-gui" / "res" / "editions"
CARDS = Path(__file__).resolve().parents[2] / "output" / "cardsfolder-512"
LOC = ConvertedCardLocator(CARDS)

_SET_CACHE: dict[str, dict[str, str]] = {}
_CODE_TO_FILE: dict[str, Path] | None = None


def _build_code_index() -> dict[str, Path]:
    global _CODE_TO_FILE
    if _CODE_TO_FILE is None:
        idx: dict[str, Path] = {}
        for f in EDITIONS.glob("*.txt"):
            try:
                head = f.read_text(encoding="utf-8", errors="replace")[:600]
            except OSError:
                continue
            m = re.search(r"^Code=(\S+)", head, re.M)
            if m:
                idx.setdefault(m.group(1).strip(), f)
        _CODE_TO_FILE = idx
    return _CODE_TO_FILE


def set_cards(code: str) -> dict[str, str]:
    """name -> rarity letter for the given set code."""
    if code in _SET_CACHE:
        return _SET_CACHE[code]
    f = _build_code_index().get(code)
    if f is None:
        _SET_CACHE[code] = {}
        return {}
    out: dict[str, str] = {}
    in_cards = False
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("["):
            in_cards = s.lower() == "[cards]"
            continue
        if not in_cards or not s or s.startswith("#"):
            continue
        m = re.match(r"^(\S+)\s+([CURMSL])\s+(.+?)(?:\s+@.*)?$", s)
        if m:
            out[m.group(3).strip()] = m.group(2)
    _SET_CACHE[code] = out
    return out


def card_text(name: str) -> str | None:
    p = LOC.text_path(name)
    if p is None:
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def summary(name: str) -> str:
    t = card_text(name)
    if t is None:
        return "MISSING"
    mc = types = pt = ""
    rules = []
    for line in t.splitlines():
        if line.startswith("mana cost:"):
            mc = line.split(":", 1)[1].strip()
        elif line.startswith("types:"):
            types = line.split(":", 1)[1].strip()
        elif line.startswith("power toughness:"):
            pt = line.split(":", 1)[1].strip()
        elif line.startswith("name:"):
            continue
        else:
            rules.append(line.strip())
    return f"{mc:<14} {types:<38} {pt:<6} | {' / '.join(rules)[:170]}"


PIP = re.compile(r"\{(.)\}")


def mv(mc: str) -> int:
    if not mc:
        return 0
    total = 0
    for sym in re.findall(r"\{([^}]*)\}", mc):
        if sym.isdigit():
            total += int(sym)
        elif sym == "X":
            continue
        else:
            total += 1
    return total


def colors(mc: str) -> set[str]:
    out = set()
    for sym in re.findall(r"\{([^}]*)\}", mc):
        for ch in re.split(r"[/]", sym):
            if ch in "WUBRG":
                out.add(ch)
    return out


def info(name: str):
    t = card_text(name)
    if t is None:
        return None
    mc = ""
    types = ""
    for line in t.splitlines():
        if line.startswith("mana cost:"):
            mc = line.split(":", 1)[1].strip()
        elif line.startswith("types:"):
            types = line.split(":", 1)[1].strip()
    return {"mana_cost": mc, "mv": mv(mc), "colors": colors(mc), "types": types}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        code = sys.argv[2]
        filt = sys.argv[3] if len(sys.argv) > 3 else ""
        cards = set_cards(code)
        print(f"== {code}: {len(cards)} cards ==")
        for n, r in sorted(cards.items()):
            i = info(n)
            if i is None:
                continue
            if filt and filt not in "".join(sorted(i["colors"])) if filt else False:
                continue
            print(f"{r} {n:<32} {summary(n)}")
    elif cmd == "check":
        for n in sys.argv[2:]:
            print(f"{n:<34} {summary(n)}")
    elif cmd == "sets":
        for c, f in sorted(_build_code_index().items()):
            print(c, f.name)
