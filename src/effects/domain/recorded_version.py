"""Reading a versioned setting back from a checkpoint's training settings."""

from __future__ import annotations

from collections.abc import Collection, Mapping


class UnknownRecordedVersion(ValueError):
    """A checkpoint records a version this code was not written to read, or none."""


def recorded_version(
    training_settings: Mapping | None, key: str, known: Collection[str],
) -> str:
    """The version recorded under ``key``, one of ``known``.

    A version outside ``known`` raises rather than reading as another: any
    guess feeds the checkpoint inputs it never saw. So does none at all, which
    is a checkpoint older than the setting.
    """
    version = (training_settings or {}).get(key)
    if version is None:
        raise UnknownRecordedVersion(
            f"the checkpoint records no {key}, so it is older than this code reads; "
            f"this code reads {', '.join(sorted(known))} only"
        )
    if version not in known:
        raise UnknownRecordedVersion(
            f"the checkpoint records {key} {version!r}; this code reads "
            f"{', '.join(sorted(known))} only"
        )
    return version
