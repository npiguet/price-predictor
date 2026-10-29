"""Reading a versioned setting back from a checkpoint's training settings."""

from __future__ import annotations

from collections.abc import Collection, Mapping


class UnknownRecordedVersion(ValueError):
    """A checkpoint records a version this code was not written to read."""


def recorded_version(
    training_settings: Mapping | None, key: str, known: Collection[str],
) -> str | None:
    """The version recorded under ``key``; None where nothing is, which is a
    checkpoint trained before the setting existed.

    A version outside ``known`` raises rather than reading as the earlier or
    the current one: either guess feeds the checkpoint inputs it never saw.
    """
    version = (training_settings or {}).get(key)
    if version is not None and version not in known:
        raise UnknownRecordedVersion(
            f"the checkpoint records {key} {version!r}; this code reads "
            f"{', '.join(sorted(known))} or none"
        )
    return version
