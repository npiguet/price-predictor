"""How each ``train-effect-model`` flag is kept: whether a resume may change
it, and whether the run's settings record it.

Each config field declares both in its own metadata (:func:`setting`), beside
its help, so a flag added later is classified where it is declared and read
through one helper (:func:`setting_classes`). The settings a run records from
those classes are one dict, which every checkpoint and ``resume.pt`` hold and
a resume compares.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import Field, field, fields
from enum import Enum, StrEnum
from pathlib import PurePath
from typing import Any

#: The metadata keys a field's two classes are declared under.
RESUME_KEY = "resume"
RECORDED_KEY = "recorded"


class OnResume(StrEnum):
    """What a resume does with a flag."""

    #: A resume under another value is refused.
    COMPARED = "compared"
    #: A resume may set it differently: it places no step, or what it names
    #: is compared by digest or hash instead.
    MAY_CHANGE = "may change"


class Recorded(StrEnum):
    """Whether the run's settings hold a flag."""

    ALWAYS = "always"
    #: Only when set (truthy), so its absence means the flag was off.
    WHEN_SET = "when set"
    NEVER = "never"


def setting(resume: OnResume, recorded: Recorded, **options: Any) -> Any:
    """A config field with its two classes. ``options`` are ``field``'s, and
    the flag's ``help`` and parser ``type`` where it has them, which go into
    the metadata ``cli.add_config_flags`` reads."""
    flag = {key: options.pop(key) for key in ("help", "type") if key in options}
    return field(metadata={RESUME_KEY: resume, RECORDED_KEY: recorded, **flag}, **options)


def setting_classes(spec: Field) -> tuple[OnResume, Recorded]:
    """The two classes ``spec`` declares; ``KeyError`` naming the field when
    it declares none."""
    try:
        return OnResume(spec.metadata[RESUME_KEY]), Recorded(spec.metadata[RECORDED_KEY])
    except KeyError as missing:
        raise KeyError(
            f"the config field {spec.name} declares no {missing} class; declare it "
            "with run_settings.setting"
        ) from None


def plain(value: Any) -> Any:
    """``value`` as the settings record it: an enum by its value, a path as text."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    return value


def _records(spec: Field, value: Any) -> bool:
    recorded = setting_classes(spec)[1]
    return recorded is Recorded.ALWAYS or (recorded is Recorded.WHEN_SET and bool(value))


def recorded_settings(config: Any) -> dict[str, Any]:
    """Every field of ``config`` its class records, in declaration order, by
    its plain value."""
    return {
        spec.name: plain(getattr(config, spec.name))
        for spec in fields(config) if _records(spec, getattr(config, spec.name))
    }


def free_on_resume(config_type: type) -> frozenset[str]:
    """The fields of ``config_type`` a resume may change."""
    return frozenset(
        spec.name for spec in fields(config_type)
        if setting_classes(spec)[0] is OnResume.MAY_CHANGE
    )


def compared_settings(settings: Mapping[str, Any], config_type: type) -> dict[str, Any]:
    """``settings`` without the flags a resume may change: what a resume
    compares."""
    free = free_on_resume(config_type)
    return {name: value for name, value in settings.items() if name not in free}
